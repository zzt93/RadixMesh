import socket
import threading
import time
from abc import ABC, abstractmethod
import logging
from typing import Optional, Callable

from src.communication import serializer
from src.radix.cache_oplog import CacheOplog

logger = logging.getLogger(__name__)


class Communicator(ABC):
    @abstractmethod
    def send(self, value: CacheOplog) -> int:
        pass

    @abstractmethod
    def register_rcv_callback(self, fn: Callable[[CacheOplog], None]) -> None:
        pass

    @abstractmethod
    def is_ordered(self) -> bool:
        pass

    @abstractmethod
    def target_address(self) -> str:
        pass



class MooncakeCommunicator(Communicator):

    def __init__(self, hostname: str, length: int, target: str, ib_device: Optional[str] = None):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.ib_device = ib_device
        self.target_session_id = target
        self.buf_len = length
        self.rcv_callback = None
        self.serializer = serializer.serializer()
        self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"

        self._allocate_buf(length)
        self._initialize_engine(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        self._recv_thread = threading.Thread(target=self._recv_daemon, daemon=True)
        self._recv_thread.start()

    def _recv_daemon(self):
        while True:
            try:
                if self.conn.poll(1):  # 1秒超时
                    msg = self.conn.recv()
                    if self.rcv_callback:
                        self.rcv_callback(msg)
            except EOFError:
                break
            except Exception as e:
                logger.info(f"Receive exception: {e}")

    def _allocate_buf(self, length):
        self.read_buffer = self.engine.allocate_managed_buffer(length)
        self.write_buffer = self.engine.allocate_managed_buffer(length)

    def target_address(self) -> str:
        return self.target_session_id

    def _initialize_engine(
            self,
            hostname: str,
            device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        ret_value = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "rdma",
            device_name if device_name is not None else "",
        )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        # TODO exchange peer addr
        self.target_ptr = None

    def send(self, value) -> int:
        length = self.serializer.serialize(value, self.write_buffer, self.buf_len)
        # logger.debug(f"Sending {length}, {str(self.write_buffer[:length])}")
        """Synchronously transfer data to the specified address."""
        try:
            # the first time: based on session_id (which contains remote_ip) to construct a queue pair, and cache the queue pair
            # later: based on the cached queue pair to send data
            ret = self.engine.transfer_sync_write(
                self.target_session_id, self.write_buffer, self.target_ptr, length
            )
        except Exception:
            # Mark transfer request as failed
            ret = -1

        if ret < 0:
            # Do not raise an exception here, since some transfer requests fail should be accepted and the execution thread should not be stopped.
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                self.write_buffer,
                self.target_session_id,
                self.target_ptr,
            )

        return ret

    def register_rcv_callback(self, fn: Callable[[CacheOplog], None]) -> None:
        self.rcv_callback = fn

    def is_ordered(self) -> bool:
        return True

    def get_session_id(self):
        return self.session_id


def parse_addr(addr):
    host, port = addr.split(':')
    return host, int(port)


class TcpCommunicator(Communicator):
    def __init__(self, local_addr: str, target: str, length: int):
        self.callback = None
        self._running = True
        self.buf_len = length
        self.rcv_callback = None
        self.serializer = serializer.serializer()
        self._allocate_buf(length)

        assert local_addr != "" or target != "", "invalid addr config"
        if local_addr != "":
            self.local_host, self.local_port = parse_addr(local_addr)
            self.listener_thread = threading.Thread(target=self._listen, daemon=True)
            self.listener_thread.start()

        if target != "":
            self.target_host, self.target_port = parse_addr(target)
            self._send_lock = threading.Lock()
            self._send_sock = None
            self._send_sock_lock = threading.Lock()  # 避免多线程重复连接
            self._connect_send_sock()

    def target_address(self) -> str:
        return f"{self.target_host}:{self.target_port}"

    def _connect_send_sock(self):
        # 尝试建立连接
        while self._running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
                sock.connect((self.target_host, self.target_port))
                logger.info(f"[TcpCommunicator] Connected to {self.target_host}:{self.target_port}")
                with self._send_sock_lock:
                    if self._send_sock:
                        self._send_sock.close()
                    self._send_sock = sock
                return
            except Exception as e:
                logger.error(
                    f"[TcpCommunicator] Waiting for connection to {self.target_host}:{self.target_port} ... ({e})")
                time.sleep(1)

    def register_rcv_callback(self, fn: Callable[[CacheOplog], None]) -> None:
        self.callback = fn

    def send(self, value) -> int:
        length = self.serializer.serialize(value, self.write_buf, self.buf_len)
        msg = length.to_bytes(4, 'big') + self.write_buf[:length]
        with self._send_lock:
            for retry in range(2):  # 尝试重连1次
                try:
                    if not self._send_sock:
                        self._connect_send_sock()
                        if not self._send_sock:
                            continue
                    self._send_sock.sendall(msg)
                    return length
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
                    logger.exception(f"[TcpCommunicator] Send failed, reconnecting... {e}")
                    try:
                        if self._send_sock:
                            self._send_sock.close()
                    except Exception:
                        pass
                    self._send_sock = None
                    time.sleep(0.1)
            logger.error("[TcpCommunicator] Send failed permanently.")
            return 0

    def _listen(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.local_host, self.local_port))
        logger.info(f"[TcpCommunicator]: Bound to {self.local_host}:{self.local_port}")
        srv.listen(5)
        while self._running:
            try:
                conn, addr = srv.accept()
                threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()
            except Exception as e:
                logger.info('Accept error:', e)
                continue

    def _handle_conn(self, conn):
        with conn:
            try:
                while True:
                    length_data = self._recv_all(conn, 4, True)
                    if not length_data or len(length_data) < 4:
                        break
                    length = int.from_bytes(length_data, 'big')
                    if length <= 0:
                        break
                    data = self._recv_all(conn, length)
                    if not data:
                        break
                    oplog: CacheOplog = self.serializer.deserialize(bytes(data), length)
                    if self.callback:
                        self.callback(oplog)
            except Exception as e:
                logger.exception(f'[TcpCommunicator] Handle error:{e}')
            logger.info('[TcpCommunicator] Connection closed.')

    def _recv_all(self, sock, length, clear=False):
        assert length <= len(self.read_buf), f'length {length}, buf {len(self.read_buf)}'
        buf = bytearray(length)
        view = memoryview(buf)
        total_recv = 0
        while total_recv < length:
            nbytes = sock.recv_into(view[total_recv:], length - total_recv)
            if nbytes == 0:
                return None
            total_recv += nbytes
        # logger.debug(f'recv {total_recv} bytes, {str(buf[:total_recv])}')
        return bytes(buf)

    def is_ordered(self) -> bool:
        return True

    def close(self):
        self._running = False
        try:
            if self._send_sock:
                self._send_sock.close()
        except Exception:
            pass

    def _allocate_buf(self, length):
        self.read_buf = bytearray(length)
        self.write_buf = bytearray(length)


def create_communicator(hostname: str, target: str, protocol: str, **kwargs) -> Communicator:
    if protocol == 'test':
        return TcpCommunicator(hostname, target=target, **kwargs)
    return MooncakeCommunicator(hostname, target=target, **kwargs)

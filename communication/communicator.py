from abc import ABC, abstractmethod
import logging
from typing import Optional, Callable

from communication import serializer
from radix.cache_oplog import CacheOplog

logger = logging.getLogger(__name__)


class Communicator(ABC):
    @abstractmethod
    def send(self, value) -> int:
        pass

    @abstractmethod
    def register_rcv_callback(self, fn: Callable[[CacheOplog], None]) -> None:
        pass

    @abstractmethod
    def is_ordered(self) -> bool:
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
        self._initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )

    def _allocate_buf(self, length):
        self.buffer_addr = self.engine.allocate_managed_buffer(length)

    def _initialize(
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
        length = self.serializer.serialize(value, self.buffer_addr, self.buf_len)
        """Synchronously transfer data to the specified address."""
        try:
            # the first time: based on session_id (which contains remote_ip) to construct a queue pair, and cache the queue pair
            # later: based on the cached queue pair to send data
            ret = self.engine.transfer_sync_write(
                self.target_session_id, self.buffer_addr, self.target_ptr, length
            )
        except Exception:
            # Mark transfer request as failed
            ret = -1

        if ret < 0:
            # Do not raise an exception here, since some transfer requests fail should be accepted and the execution thread should not be stopped.
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                self.buffer_addr,
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


def create_communicator(hostname: str, **kwargs) -> Communicator:
    return MooncakeCommunicator(hostname, **kwargs)

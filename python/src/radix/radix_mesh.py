import dataclasses
import logging
import threading
import time
from typing import List, Union, Tuple

import torch

from src.communication.communicator import create_communicator, Communicator
from src.config.cache_config import ServerArgs
from src.policy.conflict_resolve import NodeRankConflictResolver
from src.policy.sync_algo import get_sync_algo
from src.radix.cache_oplog import CacheOplog, CacheOplogType
from src.radix.core_enum import RadixMode
from src.radix.sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode, MatchResult
from src.util.thread import ThreadSafeDict

logger = logging.getLogger(__name__)


class PrefillRadixMeshTreeValue:
    value: torch.Tensor = None
    # may multiple deprecate values
    deprecate_values: List[torch.Tensor]
    deprecate_refs: List[int]
    node_rank: int

    def __init__(self, value: torch.Tensor, deprecate_values: List[torch.Tensor], deprecate_refs: List[int],
                 node_rank: int):
        self.value = value
        self.deprecate_values = deprecate_values
        self.deprecate_refs = deprecate_refs
        self.node_rank = node_rank

    def __getitem__(self, key):
        if isinstance(key, slice):
            return PrefillRadixMeshTreeValue(self.value[key], self.deprecate_values[key], self.deprecate_refs,
                                             self.node_rank)
        else:
            raise TypeError("Invalid argument type.")

    def __eq__(self, other):
        if isinstance(other, PrefillRadixMeshTreeValue):
            return torch.equal(self.value, other.value)
        return False

    def __len__(self):
        return len(self.value)

    def resolve(self, other):
        if not isinstance(other, PrefillRadixMeshTreeValue):
            raise TypeError("Invalid argument type.")

        if not NodeRankConflictResolver.keep(self.node_rank, other.node_rank):
            self.deprecate_values.append(self.value)
            self.deprecate_refs.append(self.node_rank)
            self.node_rank = other.node_rank
            self.value = other.value


class RouterRadixMeshTreeValue:
    node_rank: int

    def __init__(self, node_rank: int):
        self.node_rank = node_rank

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RouterRadixMeshTreeValue(self.node_rank)
        else:
            raise TypeError("Invalid argument type.")

    def __eq__(self, other):
        if isinstance(other, RouterRadixMeshTreeValue):
            return self.node_rank == other.node_rank
        return False

    def __len__(self):
        return 1

    def resolve(self, other):
        if not isinstance(other, RouterRadixMeshTreeValue):
            raise TypeError("Invalid argument type.")

        if not NodeRankConflictResolver.keep(self.node_rank, other.node_rank):
            self.node_rank = other.node_rank


@dataclasses.dataclass
class RouterMatchResult(MatchResult):
    prefill_node_rank: int = -1
    decode_node_rank: int = -1

    def __init__(self, prefill_node_rank: int, decode_node_rank: int = -1):
        super().__init__(device_indices=None,
                         last_device_node=None,
                         last_host_node=None, )
        self.prefill_node_rank = prefill_node_rank
        self.decode_node_rank = decode_node_rank


class RadixMesh(RadixCache):

    def __init__(self, args: ServerArgs, communicator: Communicator = None, routers: List[Communicator] = None):
        if args.prefill_node_rank != -1:
            self._global_node_rank = args.prefill_node_rank
            self.mode = RadixMode.PREFILL
        elif args.decode_node_rank != -1:
            self._global_node_rank = args.decode_node_rank
            self.mode = RadixMode.DECODE
        else:
            self._global_node_rank = args.router_node_rank
            self.mode = RadixMode.ROUTER
        self.sync_algo = get_sync_algo()

        super().__init__(page_size=1, req_to_token_pool=None, token_to_kv_pool_allocator=None, disable=False,
                         enable_kv_cache_events=False)

        self.logic_op_counter = 0
        self.communicator = communicator
        self.router_communicators = routers
        self.args = args

        logger.info(f"[RadixMesh] Initializing: {self.mode}@{self._global_node_rank}")
        next_prefill_node, routers_node, local_cache_addr = self.sync_algo.topo(args)

        if self.router_communicators is None and routers_node is not None:
            self.router_communicators = []
            for router in routers_node:
                r = create_communicator("", target=router,
                                        protocol=args.protocol,
                                        length=args.max_radix_cache_size)
                self.router_communicators.append(r)

        # first init connection to target, then setup server to receive oplog
        if self.communicator is None and (next_prefill_node != "" or local_cache_addr != ""):
            self.communicator = create_communicator(local_cache_addr, target=next_prefill_node,
                                                    protocol=args.protocol,
                                                    length=args.max_radix_cache_size)
        self.communicator.register_rcv_callback(self.oplog_received)

        if self.mode == RadixMode.DECODE:
            self.ring_tick_oplog = CacheOplog(
                oplog_type=CacheOplogType.TICK,
                node_rank=self.global_node_rank(),
                local_logic_id=0,
                key=[],
                value=[],
                # double loop
                ttl=self.sync_algo.tick_ttl(self.mode, len(args.prefill_cache_nodes)),
            )
            self.next_tick_oplog = CacheOplog(
                oplog_type=CacheOplogType.TICK,
                node_rank=self.global_node_rank(),
                local_logic_id=0,
                key=[],
                value=[],
                ttl=1,
            )
            self.ring_tick_interval_second = 10
            self.next_tick_interval_second = 1
            self.ticker_thread = threading.Thread(target=self._ticker, daemon=True)
            self.ticker_thread.start()
        self.tick_received = ThreadSafeDict()

        self._wait_all_nodes_ready()
        logger.info(f"[RadixMesh] Constructed: {self.mode}@{self._global_node_rank}")

        # TODO
        # clean tick_received every ring_tick_interval_second/2 second
        # ring_tick_oplog with local_logic_id set
        # start thread to timely check tick_received for topo check

    def _ticker(self):
        oplog = self.ring_tick_oplog
        interval_second = self.ring_tick_interval_second
        if self.args.local_node_rank(self.global_node_rank()) != 0:
            oplog = self.next_tick_oplog
            interval_second = self.next_tick_interval_second
        while True:
            if self.mode == RadixMode.DECODE:
                try:
                    self._send(oplog, False)
                except Exception as e:
                    logger.error(f"[RadixMesh] _ticker Exception: {e}")
            time.sleep(interval_second)

    def insert(self, key: List, value: torch.Tensor = None):
        assert self.mode == RadixMode.PREFILL or self.mode == RadixMode.DECODE, "only support prefill/decode node to insert"
        return self._insert(key, PrefillRadixMeshTreeValue(value, [], [], self.global_node_rank()))

    def _insert(self, key: List, value: Union[PrefillRadixMeshTreeValue, RouterRadixMeshTreeValue] = None):
        total_prefix_length = self._insert_helper(self.root_node, key, value)
        self.send_insert_event(key, value)
        return total_prefix_length

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if self.mode == RadixMode.PREFILL or self.mode == RadixMode.DECODE:
            if value:
                tensors = [v.value for v in value]
                value = torch.cat(tensors)
            else:
                value = torch.empty((0,), dtype=torch.int64, device=self.device)
            return MatchResult(
                device_indices=value,
                last_device_node=last_node,
                last_host_node=last_node,
            )
        else:
            ranks = [v.node_rank for v in value]
            prefill_node_rank = -1
            decode_node_rank = -1
            if len(ranks) == 0:
                pass
            elif len(ranks) == 1:
                prefill_node_rank = ranks[0]
            else:
                for rank in reversed(ranks):
                    if self.args.is_prefill_node_rank(rank):
                        prefill_node_rank = rank
                        break
                    else:
                        assert self.args.is_decode_node_rank(
                            rank), f"[RadixMesh] Prefix match node_rank {rank} is invalid"
                        if decode_node_rank == -1:
                            decode_node_rank = rank
                    assert prefill_node_rank != -1, f"[RadixMesh] Prefix match node_rank {rank} is invalid"
            return RouterMatchResult(prefill_node_rank=prefill_node_rank, decode_node_rank=decode_node_rank)

    def reset(self):
        super().reset()
        if self.mode == RadixMode.PREFILL or self.mode == RadixMode.DECODE:
            self.root_node.value = PrefillRadixMeshTreeValue(None, [], [], self.sync_algo.master_node_rank())
        else:
            self.root_node.value = RouterRadixMeshTreeValue(self.sync_algo.master_node_rank())

    def _match_prefix_helper(self, node: TreeNode, key: List) -> Tuple[
        List[Union[PrefillRadixMeshTreeValue, RouterRadixMeshTreeValue]], TreeNode]:
        if self.mode == RadixMode.PREFILL:
            return super()._match_prefix_helper(node, key)
        else:
            node.last_access_time = time.monotonic()
            child_key = self.get_child_key_fn(key)
            value = []
            while len(key) > 0 and child_key in node.children.keys():
                child = node.children[child_key]
                child.last_access_time = time.monotonic()
                prefix_len = self.key_match_fn(child.key, key)
                if prefix_len < len(child.key):
                    value.append(child.value[:prefix_len])
                    node = child
                    break
                else:
                    value.append(child.value)
                    node = child
                    key = key[prefix_len:]

                    if len(key):
                        child_key = self.get_child_key_fn(key)

            return value, node

    def _insert_helper(self, node: TreeNode, key: List,
                       value: Union[PrefillRadixMeshTreeValue, RouterRadixMeshTreeValue]):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len

            if value[:prefix_len] != node.value[:prefix_len]:
                # value conflict detected
                node.value[:prefix_len].resolve(value[:prefix_len])

            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._record_store_event(new_node)
        return total_prefix_length

    def send_insert_event(self, key: List, value: Union[PrefillRadixMeshTreeValue, RouterRadixMeshTreeValue]):
        if self.mode == RadixMode.ROUTER:
            return
        assert value.value is not None, "treenode should have value"
        oplog = CacheOplog(
            oplog_type=CacheOplogType.INSERT,
            node_rank=value.node_rank,
            local_logic_id=self.logic_op_count(),
            key=key,
            value=value.value.tolist(),
            ttl=self.sync_algo.ttl(self.mode, len(self.args.prefill_cache_nodes)),
        )
        self._send(oplog)

    def _send(self, oplog, send_router: bool = True):
        if not self.sync_algo.can_send(self.mode):
            return
        self.communicator.send(oplog)
        logger.debug(f"send to {self.communicator.target_address()} oplog {oplog}")
        if self._is_sync_to_router() and send_router:
            for router in self.router_communicators:
                logger.debug(f"send to {router.target_address()} oplog {oplog}")
                router.send(oplog)

    def _record_remove_event(self, node: TreeNode):
        # TODO whether need to support evict in distributed KV?
        pass

    def _is_sync_to_router(self):
        return self._global_node_rank == self.sync_algo.master_node_rank()

    def _tick_handle(self, oplog: CacheOplog):
        self.tick_received.incOrDefault(oplog.node_rank, 1)
        self._send(oplog)

    def oplog_received(self: 'RadixMesh', oplog: CacheOplog):
        logger.debug(f"node@{self._global_node_rank} received {oplog}")
        assert self.sync_algo.can_rcv(self.mode), "only support prefill/router mode"
        oplog.ttl -= 1
        if oplog.node_rank == self.global_node_rank() or oplog.ttl <= 0:
            return None
        if oplog.oplog_type == CacheOplogType.TICK:
            return self._tick_handle(oplog)

        if self.communicator.is_ordered():
            pass
        else:
            assert False, "not implemented yet"
            # check local id
            pass

        if oplog.oplog_type == CacheOplogType.INSERT:
            if self.mode == RadixMode.PREFILL:
                value = PrefillRadixMeshTreeValue(torch.tensor(oplog.value), [], [], oplog.node_rank)
            else:
                value = RouterRadixMeshTreeValue(oplog.node_rank)
            self._insert(oplog.key, value)
        elif oplog.oplog_type == CacheOplogType.DELETE:
            self.delete(oplog.key)
        else:
            # CacheOplogType.RESET:
            super().reset()

    def global_node_rank(self):
        return self._global_node_rank

    def delete(self, key):
        pass

    def logic_op_count(self):
        self.logic_op_counter += 1
        return self.logic_op_counter

    def _wait_all_nodes_ready(self):
        if self.sync_algo.ring():
            if self.sync_algo.can_rcv(self.mode):
                while len(self.tick_received) == 0 or not all(
                        value >= 2 for value in self.tick_received.values()):
                    logger.info(f"[RadixMesh] Received peer {self.tick_received}")
                    time.sleep(1)
                logger.info(f"[RadixMesh] Received peer {self.tick_received}")
            return

        assert False, "not implemented yet"

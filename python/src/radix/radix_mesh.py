import dataclasses
import enum
import logging
import time
from typing import List, Union, Tuple

import torch

from src.communication.communicator import create_communicator, Communicator
from src.config.cache_config import ServerArgs
from src.policy.conflict_resolve import NodeRankConflictResolver
from src.policy.sync_algo import get_sync_algo, MASTER_RANK
from src.radix.cache_oplog import CacheOplog, CacheOplogType
from src.radix.sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode, MatchResult

logger = logging.getLogger(__name__)


class PrefillRadixMeshTreeValue:
    value: torch.Tensor = None
    # may multiple deprecate values
    deprecate_values: list[torch.Tensor]
    deprecate_refs: list[int]
    node_rank: int

    def __init__(self, value: torch.Tensor, deprecate_values: list[torch], deprecate_refs: list[int], node_rank: int):
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


class RadixMode(str, enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    ROUTER = "router"

@dataclasses.dataclass
class RouterMatchResult(MatchResult):
    node_rank: int = 0

    def __init__(self, node_rank: int):
        super().__init__(device_indices=None,
                         last_device_node=None,
                         last_host_node=None, )
        self.node_rank = node_rank


class RadixMesh(RadixCache):

    def __init__(self, args: ServerArgs, communicator: Communicator = None, routers: list[Communicator] = None):
        if args.prefill_node_rank != -1:
            self._global_node_rank = args.prefill_node_rank
            self.mode = RadixMode.PREFILL
        elif args.decode_node_rank != -1:
            self._global_node_rank = args.decode_node_rank
            self.mode = RadixMode.DECODE
        else:
            self._global_node_rank = args.router_node_rank
            self.mode = RadixMode.ROUTER

        super().__init__(page_size=1, req_to_token_pool=None, token_to_kv_pool_allocator=None, disable=False,
                         enable_kv_cache_events=False)

        self.logic_op_counter = 0
        self.communicator = communicator
        self.router_communicators = routers
        self.args = args

        logger.info(f"[RadixMesh] Initializing: {self.mode}@{self._global_node_rank}")
        next_prefill_node, routers_node = get_sync_algo().next(args)

        # DECODE node no need start a server
        local_cache_addr = args.local_cache_addr
        if self.mode == RadixMode.DECODE:
            local_cache_addr = ""

        if self.communicator is None and next_prefill_node is not None:
            self.communicator = create_communicator(local_cache_addr, target=next_prefill_node,
                                                    protocol=args.protocol,
                                                    length=args.max_radix_cache_size)
        self.communicator.register_rcv_callback(self.oplog_received)
        if self.router_communicators is None and routers_node is not None:
            self.router_communicators = []
            for router in routers_node:
                r = create_communicator("", target=router,
                                        protocol=args.protocol,
                                        length=args.max_radix_cache_size)
                self.router_communicators.append(r)
        if self.router_communicators is not None:
            for router in self.router_communicators:
                router.register_rcv_callback(self.oplog_received)

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
            return RouterMatchResult(node_rank=last_node.value.node_rank)

    def reset(self):
        super().reset()
        if self.mode == RadixMode.PREFILL or self.mode == RadixMode.DECODE:
            self.root_node.value = PrefillRadixMeshTreeValue(None, [], [], MASTER_RANK)
        else:
            self.root_node.value = RouterRadixMeshTreeValue(MASTER_RANK)

    def _match_prefix_helper(self, node: TreeNode, key: List) -> Tuple[
        list[Union[PrefillRadixMeshTreeValue, RouterRadixMeshTreeValue]], TreeNode]:
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
        oplog = CacheOplog(CacheOplogType.INSERT, value.node_rank, self.logic_op_count(), key,
                           value.value.tolist())

        self.communicator.send(oplog)
        logger.debug("send oplog %s" % oplog)
        if self._is_sync_to_router():
            for router in self.router_communicators:
                router.send(oplog)

    def _record_remove_event(self, node: TreeNode):
        # TODO whether need to support evict in distributed KV?
        pass

    def _is_sync_to_router(self):
        return self._global_node_rank == MASTER_RANK

    def oplog_received(self: 'RadixMesh', oplog: CacheOplog):
        logger.debug("oplog received %s" % oplog)
        assert self.mode == RadixMode.PREFILL or self.mode == RadixMode.ROUTER, "only support prefill/router mode"
        if oplog.node_rank == self.global_node_rank():
            return

        # decode node write to prefill node
        if self.args.is_decode_node_rank(oplog.node_rank):
            assert self.mode == RadixMode.PREFILL, "only support prefill mode"
            # override decode node rank with prefill node rank, otherwise may cause infinite loop
            oplog.node_rank = self.global_node_rank()

        if self.communicator.is_ordered():
            pass
        else:
            assert False, "not implemented yet"
            # check local id
            pass

        match oplog.oplog_type:
            case CacheOplogType.INSERT:
                if self.mode == RadixMode.PREFILL:
                    value = PrefillRadixMeshTreeValue(torch.tensor(oplog.value), [], [], oplog.node_rank)
                else:
                    value = RouterRadixMeshTreeValue(oplog.node_rank)
                self._insert(oplog.key, value)
            case CacheOplogType.DELETE:
                self.delete(oplog.key)
            case CacheOplogType.RESET:
                super().reset()

    def global_node_rank(self):
        return self._global_node_rank

    def delete(self, key):
        pass

    def logic_op_count(self):
        self.logic_op_counter += 1
        return self.logic_op_counter

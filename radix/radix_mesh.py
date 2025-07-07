import logging
from typing import List

import torch

from communication.communicator import create_communicator
from config.cache_config import ServerArgs
from radix.cache_oplog import CacheOplog, CacheOplogType
from radix.sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

logger = logging.getLogger(__name__)


class RadixMeshTreeNode:
    # may multiple deprecate values
    deprecate_values: list[list]
    deprecate_refs: list[int]
    # router used
    node_rank: int


class RadixMesh(RadixCache):
    def __init__(self, args: ServerArgs):
        super().__init__(page_size=1, req_to_token_pool=None, token_to_kv_pool_allocator=None, disable=False,
                         enable_kv_cache_events=False)
        self.node_rank = args.cache_node_rank

        # bind addr & establish connection
        self.communicator = create_communicator(args.local_cache_addr, length=args.max_radix_cache_size)
        logger.info("Initializing RadixMesh(prefill mode)")
        logger.info("Initializing RadixMesh(router mode)")

    def _record_store_event(self, node: TreeNode):
        self.communicator.send()

    def _record_remove_event(self, node: TreeNode):
        self.communicator.send()

    def oplog_received(self, oplog: CacheOplog):
        if oplog.node_rank == self.current_node_rank():
            return

        if self.communicator.is_ordered():
            pass
        else:
            # check local id
            pass

        match oplog.oplog_type:
            case CacheOplogType.INSERT:
                super().insert(oplog.key, oplog.value)
            case CacheOplogType.DELETE:
                self.delete(oplog.key)
            case CacheOplogType.RESET:
                super().reset()

    def current_node_rank(self):
        return self.node_rank

    def delete(self, key):
        pass

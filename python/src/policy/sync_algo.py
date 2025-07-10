from abc import ABC, abstractmethod
from typing import NamedTuple, List

from src.config.cache_config import ServerArgs
from src.radix.core_enum import RadixMode

MASTER_RANK = 0


class TopoResult(NamedTuple):
    next_prefill_node: str
    routers_node: List[str]
    local_bind_addr: str


class BaseSyncAlgo(ABC):
    @abstractmethod
    def topo(self, args: ServerArgs) -> TopoResult:
        pass

    @abstractmethod
    def master_node_rank(self):
        pass

    @abstractmethod
    def ring(self) -> bool:
        pass

    @abstractmethod
    def can_send(self, mode: RadixMode) -> bool:
        pass

    @abstractmethod
    def can_rcv(self, mode: RadixMode) -> bool:
        pass

    def ttl(self, mode: RadixMode, prefill_count):
        pass

    def tick_ttl(self, mode: RadixMode, prefill_count):
        pass


class RingSyncAlgo(BaseSyncAlgo):
    def __init__(self):
        pass

    def master_node_rank(self):
        return MASTER_RANK

    def topo(self, args: ServerArgs) -> TopoResult:
        # DECODE node no need start a server
        local_cache_addr = args.local_cache_addr
        if args.decode_node_rank >= 0:
            local_cache_addr = ""

        if args.prefill_node_rank >= 0:
            if args.prefill_node_rank == self.master_node_rank():
                return TopoResult(args.prefill_cache_nodes[
                                      (args.prefill_node_rank + 1) % len(args.prefill_cache_nodes)],
                                  args.router_cache_nodes, local_cache_addr)
            else:
                return TopoResult(args.prefill_cache_nodes[
                                      (args.prefill_node_rank + 1) % len(args.prefill_cache_nodes)], None,
                                  local_cache_addr)
        if args.decode_node_rank >= 0:
            return TopoResult(args.prefill_cache_nodes[self.master_node_rank()], None, local_cache_addr)
        if args.router_node_rank >= 0:
            return TopoResult("", None, local_cache_addr)
        return TopoResult("", None, "")

    def ring(self) -> bool:
        return True

    def can_send(self, mode: RadixMode) -> bool:
        if mode == RadixMode.PREFILL:
            return True
        if mode == RadixMode.ROUTER:
            return False
        if mode == RadixMode.DECODE:
            return True
        assert False

    def can_rcv(self, mode: RadixMode) -> bool:
        if mode == RadixMode.PREFILL:
            return True
        if mode == RadixMode.DECODE:
            return False
        if mode == RadixMode.ROUTER:
            return True
        assert False

    def ttl(self, mode: RadixMode, prefill_count):
        if mode == RadixMode.PREFILL:
            return prefill_count
        if mode == RadixMode.DECODE:
            return prefill_count + 1
        assert False

    def tick_ttl(self, mode: RadixMode, prefill_count):
        return self.ttl(mode, prefill_count) * 2 - 1


def get_sync_algo() -> BaseSyncAlgo:
    return RingSyncAlgo()

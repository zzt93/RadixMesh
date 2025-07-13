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

    def ttl(self, mode: RadixMode, args: ServerArgs):
        pass

    def tick_ttl(self, mode: RadixMode, args: ServerArgs):
        pass

    def gc_ttl(self, mode: RadixMode, args: ServerArgs):
        pass

    def can_tick(self, mode: RadixMode, args: ServerArgs) -> bool:
        pass


class RingSyncAlgo(BaseSyncAlgo):
    def __init__(self):
        pass

    def master_node_rank(self):
        return MASTER_RANK

    def topo(self, args: ServerArgs) -> TopoResult:
        # DECODE node no need start a server
        local_cache_addr = args.local_cache_addr

        prefill_decode_nodes = args.prefill_cache_nodes + args.decode_cache_nodes
        if args.prefill_node_rank >= 0:
            if args.prefill_node_rank == self.master_node_rank():
                return TopoResult(prefill_decode_nodes[
                                      (args.prefill_node_rank + 1) % len(prefill_decode_nodes)],
                                  args.router_cache_nodes, local_cache_addr)
            else:
                return TopoResult(prefill_decode_nodes[
                                      (args.prefill_node_rank + 1) % len(prefill_decode_nodes)], None,
                                  local_cache_addr)
        if args.decode_node_rank >= 0:
            return TopoResult(prefill_decode_nodes[(args.decode_node_rank + 1) % len(prefill_decode_nodes)], None, local_cache_addr)
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
            return True
        if mode == RadixMode.ROUTER:
            return True
        assert False

    def ttl(self, mode: RadixMode, args: ServerArgs):
        if mode == RadixMode.PREFILL or mode == RadixMode.DECODE:
            return len(args.prefill_cache_nodes) + len(args.decode_cache_nodes)
        assert False

    def tick_ttl(self, mode: RadixMode, args: ServerArgs):
        return self.ttl(mode, args) * 2

    def gc_ttl(self, mode: RadixMode, args: ServerArgs):
        return self.ttl(mode, args)

    def can_tick(self, mode: RadixMode, args: ServerArgs) -> bool:
        return mode == RadixMode.DECODE and args.local_node_rank(args.decode_node_rank) == 0


def get_sync_algo() -> BaseSyncAlgo:
    return RingSyncAlgo()

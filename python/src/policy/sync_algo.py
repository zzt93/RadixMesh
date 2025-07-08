from abc import ABC, abstractmethod

from src.config.cache_config import ServerArgs

MASTER_RANK = 0


class BaseSyncAlgo(ABC):
    @abstractmethod
    def next(self, args: ServerArgs):
        pass


class RingSyncAlgo(BaseSyncAlgo):
    def __init__(self):
        pass

    def next(self, args: ServerArgs):
        if args.prefill_node_rank >= 0:
            return (args.prefill_cache_nodes[(args.prefill_node_rank + 1) % len(args.prefill_cache_nodes)],
                    args.router_cache_nodes)
        if args.decode_node_rank >= 0:
            return args.prefill_cache_nodes[MASTER_RANK], None
        if args.router_node_rank >= 0:
            return args.prefill_cache_nodes[MASTER_RANK], None



def get_sync_algo():
    return RingSyncAlgo()

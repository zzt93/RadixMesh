import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class ServerArgs:
    prefill_cache_nodes: List[str] = field(default_factory=list)
    router_cache_nodes: List[str] = field(default_factory=list)
    local_cache_addr: str = field(default="")
    max_radix_cache_size: int = field(default=16 * 1024 * 1024)
    mooncake_metadata_server: str = ''

    cache_node_rank: int = -1
    router_node_rank: int = -1


def load_server_args(yaml_file: str) -> ServerArgs:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    prefill = config.get('prefill_cache_nodes', [])
    router = config.get('router_cache_nodes', [])
    if len(router) > 1:
        raise NotImplementedError("Multiple routers not supported")

    cache_addr = config.get('cache_addr')
    try:
        node_rank = prefill.index(cache_addr)
    except ValueError:
        node_rank = -1
    try:
        router_node_rank = router.index(cache_addr)
    except ValueError:
        router_node_rank = -1

    if node_rank == -1 and router_node_rank == -1:
        raise ValueError("invalid config cache_addr")

    return ServerArgs(
        prefill_cache_nodes=prefill,
        router_cache_nodes=router,
        cache_node_rank=node_rank,
        router_node_rank=router_node_rank
    )

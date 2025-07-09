import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class ServerArgs:
    prefill_cache_nodes: List[str] = field(default_factory=list)
    router_cache_nodes: List[str] = field(default_factory=list)
    decode_cache_nodes: List[str] = field(default_factory=list)
    local_cache_addr: str = field(default="")
    max_radix_cache_size: int = field(default=16 * 1024 * 1024)
    mooncake_metadata_server: str = ''
    protocol: str = 'tcp'

    prefill_node_rank: int = -1
    decode_node_rank: int = -1
    router_node_rank: int = -1

    def is_decode_node_rank(self, node_rank: int) -> bool:
        if len(self.prefill_cache_nodes) <= node_rank < len(self.prefill_cache_nodes) + len(self.decode_cache_nodes):
            return True
        return False


def load_server_args(yaml_file: str) -> ServerArgs:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    args = ServerArgs(**config)

    prefill = args.prefill_cache_nodes
    router = args.router_cache_nodes
    decode = args.decode_cache_nodes
    if len(router) > 1:
        raise NotImplementedError("Multiple routers not supported")

    seen = 0
    cache_addr = args.local_cache_addr
    try:
        prefill_node_rank = prefill.index(cache_addr)
        seen += 1
    except ValueError:
        prefill_node_rank = -1

    try:
        decode_node_rank = decode.index(cache_addr) + len(prefill)
        seen += 1
    except ValueError:
        decode_node_rank = -1

    try:
        router_node_rank = router.index(cache_addr) + len(prefill) + len(decode)
        seen += 1
    except ValueError:
        router_node_rank = -1

    if seen != 1:
        raise ValueError("invalid config local_cache_addr")

    args.prefill_node_rank = prefill_node_rank
    args.router_node_rank = router_node_rank
    args.decode_node_rank = decode_node_rank
    return args

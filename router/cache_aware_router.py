from radix.radix_mesh import RadixMesh


class PrefillNode:
    addr: str

    def __init__(self):
        pass


class CacheAwareRadixTreeNode:
    node_rank: int

    def __init__(self):
        pass


class CacheAwareRouter:
    def __init__(self, radix_mesh: RadixMesh, prefill_nodes: PrefillNode):
        self.radix_mesh = radix_mesh

    def cache_aware_route(self, key: list[int]) -> PrefillNode:
        match_res = self.radix_mesh.match_prefix(key)
        return PrefillNode()

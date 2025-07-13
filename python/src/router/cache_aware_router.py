import dataclasses
from typing import List
import hashlib
from bisect import bisect, bisect_left, insort

from src.radix.radix_mesh import RadixMesh, RouterMatchResult

@dataclasses.dataclass
class RouteResult:
    prefill_addr: str
    decode_addr: str



class CacheAwareRouter:
    def __init__(self, radix_mesh: RadixMesh, skip_warm_up: bool = False):
        self.radix_mesh = radix_mesh
        self.warm_up_done = skip_warm_up

    def finish_warm_up(self):
        self.warm_up_done = True

    def cache_aware_route(self, key: List[int]) -> RouteResult:
        if not self.warm_up_done:
            match_res = RouterMatchResult(-1, -1)
        else:
            match_res = self.radix_mesh.match_prefix(key)
        if match_res.prefill_node_rank != -1:
            prefill_addr = self.radix_mesh.prefill_cache_nodes(match_res.prefill_node_rank)
        else:
            h = ConsistentHash(nodes=self.radix_mesh.prefill_cache_nodes())
            prefill_addr = h.get_node(key)
        if match_res.decode_node_rank != -1:
            decode_addr = self.radix_mesh.decode_cache_nodes(match_res.decode_node_rank)
        else:
            h = ConsistentHash(nodes=self.radix_mesh.decode_cache_nodes())
            decode_addr = h.get_node(key)

        return RouteResult(prefill_addr, decode_addr)


class ConsistentHash:
    def __init__(self, nodes=None, replicas=3):
        """
        初始化一致性哈希环
        :param nodes: 初始节点列表
        :param replicas: 每个节点的虚拟节点数量
        """
        self.replicas = replicas
        self.ring = []  # 哈希环，存储所有虚拟节点的哈希值
        self.ring_nodes = {}  # 哈希值到真实节点的映射
        self.nodes = set()  # 真实节点集合

        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node):
        """
        添加一个节点到哈希环
        :param node: 节点标识，可以是字符串或任何可哈希对象
        """
        if node in self.nodes:
            return

        self.nodes.add(node)

        # 为每个节点创建 replicas 个虚拟节点
        for i in range(self.replicas):
            virtual_node = f"{node}#{i}"
            key = self._hash(virtual_node)
            self.ring_nodes[key] = node
            insort(self.ring, key)

    def remove_node(self, node):
        """
        从哈希环中移除一个节点
        :param node: 要移除的节点
        """
        if node not in self.nodes:
            return

        self.nodes.remove(node)

        # 移除该节点的所有虚拟节点
        for i in range(self.replicas):
            virtual_node = f"{node}#{i}"
            key = self._hash(virtual_node)
            if key in self.ring_nodes:
                del self.ring_nodes[key]

            # 从环中删除该键
            index = bisect_left(self.ring, key)
            if index < len(self.ring) and self.ring[index] == key:
                self.ring.pop(index)

    def get_node(self, key):
        """
        获取给定键对应的节点
        :param key: 键
        :return: 负责该键的节点
        """
        if not self.ring:
            return None

        hash_key = self._hash(key)
        index = bisect(self.ring, hash_key) % len(self.ring)
        return self.ring_nodes[self.ring[index]]

    def _hash(self, key):
        """
        计算键的哈希值（使用MD5）
        :param key: 键
        :return: 哈希值（整数）
        """
        # 使用MD5哈希，然后取前4个字节作为32位整数
        md5 = hashlib.md5(str(key).encode('utf-8')).digest()
        return int.from_bytes(md5[:4], byteorder='big', signed=False)

    def __str__(self):
        return f"ConsistentHash(nodes={self.nodes}, replicas={self.replicas})"

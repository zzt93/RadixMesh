import enum
from typing import List, Tuple

from pydantic import BaseModel


class CacheState(str, enum.Enum):
    VALID = "valid"
    # delete when no active ref; not used for new request
    DEPRECATED = "deprecated"


class CacheOplogType(enum.IntEnum):
    """
    Cache oplog: idempotent operation for radix cache tree
    """
    INSERT = 1
    DELETE = 2
    RESET = 3
    GC_QUERY = 4
    GC_EXEC = 5
    TICK = 10


class ImmutableNodeKey(BaseModel):
    key: Tuple[int, ...]
    node_rank: int
    key_hash: int

    def __eq__(self, other):
        if isinstance(other, ImmutableNodeKey):
            return self.node_rank == other.node_rank and self.key == other.key
        return False

    def __hash__(self):
        return self.key_hash

    @classmethod
    def create(cls, key: List[int], node_rank: int) -> 'ImmutableNodeKey':
        return cls(key=tuple(key), node_rank=node_rank, key_hash=hash((node_rank, tuple(key))))


class GCQuery(BaseModel):
    agree: int = 1
    key: ImmutableNodeKey


class CacheOplog(BaseModel):
    oplog_type: CacheOplogType
    node_rank: int
    local_logic_id: int = 0
    key: List = []
    value: List = []
    ttl: int
    gc_query: List[GCQuery] = []
    gc_exec: List[ImmutableNodeKey] = []

    def to_dict(self):
        return {
            'oplog_type': self.oplog_type.value,
            'node_rank': self.node_rank,
            'local_logic_id': self.local_logic_id,
            'key': self.key,
            'value': self.value,
            'ttl': self.ttl,
        }

    def __str__(self):
        return (f"CacheOplog(oplog_type={self.oplog_type}, node_rank={self.node_rank}, "
                f"local_logic_id={self.local_logic_id}, key={self.key}, value={self.value})"
                f"ttl={self.ttl})")


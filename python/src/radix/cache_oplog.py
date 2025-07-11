import enum
from typing import List

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
    GC_START = 4
    TICK = 10


class CacheOplog(BaseModel):
    oplog_type: CacheOplogType
    node_rank: int
    local_logic_id: int
    key: List
    value: List
    ttl: int

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

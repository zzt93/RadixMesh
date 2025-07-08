import enum
from typing import List


class CacheState(str, enum.Enum):
    VALID = "valid"
    # delete when no active ref; not used for new request
    DEPRECATED = "deprecated"


class CacheOplogType(str, enum.Enum):
    """
    Cache oplog: idempotent operation for radix cache tree
    """
    INSERT = "insert"
    DELETE = "delete"
    RESET = "reset"


class CacheOplog:
    oplog_type: CacheOplogType
    node_rank: int
    local_logic_id: int
    key: List
    value: List

    def __init__(self, oplog_type: CacheOplogType, node_rank: int, local_logic_id: int, key: List, value: List):
        self.oplog_type = oplog_type
        self.node_rank = node_rank
        self.local_logic_id = local_logic_id
        self.key = key
        self.value = value

import enum
from typing import List

import torch

import util


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

    def __init__(self):
        pass


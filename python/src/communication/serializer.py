import json
from typing import Any

from src.radix.cache_oplog import CacheOplog


class Serializer:
    def serialize(self, value: Any, buf: bytearray, max_len: int) -> int:
        raise NotImplementedError

    def deserialize(self, buf: bytearray, size: int) -> Any:
        raise NotImplementedError


def dict_to_oplog(d):
    return CacheOplog(**d)


class JsonSerializer(Serializer):
    def serialize(self, value: CacheOplog, buf: bytearray, max_len: int) -> int:
        """
        Serialize a Python object to JSON, write to buf.
        """
        data: str = json.dumps(value.to_dict())
        encoded: bytes = data.encode('utf-8')
        if len(encoded) > max_len:
            raise ValueError(
                f"too large value length:{len(encoded)}, increase max_radix_cache_size:{max_len} if needed")
        buf[:len(encoded)] = encoded
        return len(encoded)

    def deserialize(self, buf: bytearray, size: int) -> CacheOplog:
        """
        Deserialize Python object from JSON in buf.
        """
        data: str = buf[:size].decode('utf-8')
        obj: CacheOplog = json.loads(data, object_hook=dict_to_oplog)
        return obj


def serializer():
    return JsonSerializer()



import json
from typing import Any
from pydantic import BaseModel

from src.radix.cache_oplog import CacheOplog


class Serializer:
    def serialize(self, value: CacheOplog) -> bytes:
        raise NotImplementedError

    def deserialize(self, buf: bytearray, size: int) -> Any:
        raise NotImplementedError


def dict_to_oplog(d):
    return CacheOplog(**d)


class JsonSerializer(Serializer):
    def serialize(self, value: CacheOplog) -> bytes:
        """
        Serialize a Python object to JSON, write to buf.
        """
        data: str = json.dumps(value.to_dict())
        encoded: bytes = data.encode('utf-8')
        return encoded

    def deserialize(self, buf: bytearray, size: int) -> CacheOplog:
        """
        Deserialize Python object from JSON in buf.
        """
        data: str = buf[:size].decode('utf-8')
        restored = CacheOplog.model_validate_json(data)
        return restored


def serializer():
    return JsonSerializer()



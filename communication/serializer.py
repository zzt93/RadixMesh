import json
from typing import Any


class Serializer:
    def serialize(self, value: Any, buf: bytearray, max_len: int) -> int:
        raise NotImplementedError

    def deserialize(self, buf: bytearray, size: int) -> Any:
        raise NotImplementedError


class JsonSerializer(Serializer):
    def serialize(self, value: Any, buf: bytearray, max_len: int) -> int:
        """
        Serialize a Python object to JSON, write to buf.
        """
        data: str = json.dumps(value)
        encoded: bytes = data.encode('utf-8')
        if len(encoded) > max_len:
            raise ValueError(
                f"too large value length:{len(encoded)}, increase max_radix_cache_size:{max_len} if needed")
        buf[:len(encoded)] = encoded
        return len(encoded)

    def deserialize(self, buf: bytearray, size: int) -> Any:
        """
        Deserialize Python object from JSON in buf.
        """
        data: str = buf[:size].decode('utf-8')
        obj: Any = json.loads(data)
        return obj


def serializer():
    return JsonSerializer()


# Optional: Simple test
if __name__ == "__main__":
    serializer = JsonSerializer()
    buffer: bytearray = bytearray(4096)
    obj = {"hello": "world", "num": 42, "arr": [1, 2, 3]}
    n: int = serializer.serialize(obj, buffer, self.buf_lenb)
    print("序列化字节数:", n)
    print("buffer:", buffer[:n])

    new_obj: Any = serializer.deserialize(buffer, n)
    print("反序列化结果:", new_obj)

import enum


class RadixMode(str, enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    ROUTER = "router"

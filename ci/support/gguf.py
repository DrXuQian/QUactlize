"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import io
import struct
from typing import Any
def _encoded_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _synthetic_gguf(metadata: list[tuple[str, Any]],
                    tensors: list[tuple[str, tuple[int, ...], int]]) -> bytes:
    out = io.BytesIO()
    out.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata:
        out.write(_encoded_string(key))
        if isinstance(value, str):
            out.write(struct.pack("<I", 8) + _encoded_string(value))
        elif isinstance(value, bool):
            out.write(struct.pack("<I?", 7, value))
        elif isinstance(value, int) and value >= 0:
            out.write(struct.pack("<II", 4, value))
        else:
            raise AssertionError(f"unsupported synthetic metadata value {key}={value!r}")
    for ordinal, (name, dims, qtype) in enumerate(tensors):
        out.write(_encoded_string(name) + struct.pack("<I", len(dims)))
        for dim in dims:
            out.write(struct.pack("<Q", dim))
        out.write(struct.pack("<IQ", qtype, ordinal * 4096))
    return out.getvalue()

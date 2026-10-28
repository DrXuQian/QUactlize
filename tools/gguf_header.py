"""GGUF header/model metadata utilities; no benchmark execution."""
import hashlib
import struct
from typing import Any, BinaryIO
from quactlize.gguf_roles import InventoryError
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


_SCALAR_FORMAT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise InventoryError(f"truncated GGUF header: wanted {size} bytes, got {len(data)}")
    return data


def _u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _string(stream: BinaryIO) -> str:
    length = _u64(stream)
    try:
        return _read_exact(stream, length).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError("GGUF header contains non-UTF-8 text") from exc


def _capture_metadata_key(key: str) -> bool:
    return (key in {"general.name", "general.architecture", *SPLIT_KEYS}
            or key.endswith(".expert_count")
            or key.endswith(".expert_used_count"))


def _read_or_skip_metadata(stream: BinaryIO, value_type: int, capture: bool) -> Any:
    if value_type == 8:
        value = _string(stream)
        return value if capture else None
    if value_type == 9:
        element_type, count = _u32(stream), _u64(stream)
        if element_type == 9:
            raise InventoryError("nested GGUF metadata arrays are forbidden")
        if capture:
            if count > 4096:
                raise InventoryError("selected GGUF metadata array exceeds 4096 entries")
            return [_read_or_skip_metadata(stream, element_type, True) for _ in range(count)]
        for _ in range(count):
            _read_or_skip_metadata(stream, element_type, False)
        return None
    fmt = _SCALAR_FORMAT.get(value_type)
    if fmt is None:
        raise InventoryError(f"unknown GGUF metadata value type {value_type}")
    value = struct.unpack(fmt, _read_exact(stream, struct.calcsize(fmt)))[0]
    return value if capture else None


def _sha256_range(stream: BinaryIO, start: int, end: int) -> str:
    saved = stream.tell()
    stream.seek(start)
    digest = hashlib.sha256()
    remaining = end - start
    while remaining:
        chunk = _read_exact(stream, min(1024 * 1024, remaining))
        digest.update(chunk)
        remaining -= len(chunk)
    stream.seek(saved)
    return digest.hexdigest()


def read_gguf_header(stream: BinaryIO, source: str, *, capture_keys=()) -> dict[str, Any]:
    if not stream.seekable():
        raise InventoryError(f"{source}: GGUF header source must be seekable")
    if _read_exact(stream, 4) != b"GGUF":
        raise InventoryError(f"{source}: not a GGUF file")
    version = _u32(stream)
    if version not in (2, 3):
        raise InventoryError(f"{source}: unsupported GGUF version {version}")
    tensor_count, metadata_count = _u64(stream), _u64(stream)
    metadata: dict[str, Any] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    for _ in range(metadata_count):
        key = _string(stream)
        if key in fingerprints:
            raise InventoryError(f"{source}: duplicate GGUF metadata key {key!r}")
        value_type = _u32(stream)
        start = stream.tell()
        capture = _capture_metadata_key(key) or key in capture_keys
        value = _read_or_skip_metadata(stream, value_type, capture)
        end = stream.tell()
        fingerprints[key] = {
            "value_type": value_type,
            "encoded_bytes": end - start,
            "sha256": _sha256_range(stream, start, end),
        }
        if capture:
            metadata[key] = value

    tensors = []
    seen = set()
    for ordinal in range(tensor_count):
        name = _string(stream)
        if name in seen:
            raise InventoryError(f"{source}: duplicate tensor name {name!r}")
        seen.add(name)
        ndims = _u32(stream)
        if ndims == 0:
            raise InventoryError(f"{source}: tensor {name!r} has zero dimensions")
        dims = [_u64(stream) for _ in range(ndims)]
        if any(dim <= 0 for dim in dims):
            raise InventoryError(f"{source}: tensor {name!r} has invalid dimensions {dims}")
        qtype, offset = _u32(stream), _u64(stream)
        tensors.append({"ordinal": ordinal, "name": name, "dims_gguf": dims,
                        "qtype": qtype, "offset": offset})
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "metadata_fingerprints": fingerprints,
        "tensors": tensors,
    }

"""Canonical packing ABI and host descriptors; importing does not load a GPU."""

import ctypes as C
from pathlib import Path
import re


class Arrangement(C.Structure):
    _fields_ = [(name, C.c_int32) for name in (
        "version", "layout", "bits", "high_bits", "artifact_tile_k",
        "transport_tile_k", "group_size", "reserved")] + [
        ("mapping_id", C.c_uint64)]


class Sizes(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "raw_bytes", "low_bytes", "high_bytes", "units_bytes")]


def canonical_arrangement(qtype):
    if type(qtype) is not int:
        raise ValueError("qtype must be an integer")
    if qtype == 8:
        return Arrangement(2, 4, 8, 0, 0, 32, 32, 0, 0x51384B5032540001)
    registry = Path(__file__).resolve().parents[1] / "include/ppu_format_config.inc"
    for q, low, high, group in re.findall(
            r'X\(Q\d_K,\s*"[^"]+",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),',
            registry.read_text()):
        if int(q) == qtype:
            low, high, group = map(int, (low, high, group))
            transport = 256 // (high or low)
            layout = 1 if qtype == 12 else 2
            mapping = 0x51344B5034540001 if qtype == 12 else 0x514B504B54000001
            return Arrangement(2, layout, low, high, 0, transport, group, 0, mapping)
    raise ValueError("unsupported qtype")


def bind(library):
    canonical = library.quactlize_ppu_kpack_canonical_arrangement_v1
    canonical.argtypes = [C.c_int, C.POINTER(Arrangement)]
    canonical.restype = C.c_int
    sizes = library.quactlize_ppu_kpack_sizes_for_arrangement_v1
    sizes.argtypes = [C.c_int] * 4 + [C.POINTER(Arrangement), C.POINTER(Sizes)]
    sizes.restype = C.c_int
    prepare = library.quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2
    prepare.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [
        C.POINTER(Arrangement), C.c_void_p]
    prepare.restype = C.c_int
    return canonical, sizes, prepare

"""Production ABI/catalog support."""
import ctypes as C
class Arrangement(C.Structure):
    _fields_ = [(n, C.c_int32) for n in (
        "version", "layout", "bits", "high_bits", "artifact_tile_k",
        "transport_tile_k", "group_size", "reserved")] + [("mapping_id", C.c_uint64)]


class Sizes(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in (
        "raw_bytes", "low_bytes", "high_bytes", "units_bytes")]

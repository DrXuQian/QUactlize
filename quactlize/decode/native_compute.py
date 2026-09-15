"""Tool-only bindings for explicit-compute parent modules; no selection."""
import ctypes as C

from quactlize.runtime.native import Call, Identity, Recipe, Resources


class DenseIO(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32), ("call", Call),
                ("input_type", C.c_int32), ("output_type", C.c_int32)]


class DenseComputeCall(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32), ("dense", DenseIO),
                ("compute_type", C.c_int32)]


class GroupedDeviceCall(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32), ("call", Call),
                ("max_rows", C.c_int32), ("reserved", C.c_int32)]


class GroupedComputeCall(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32), ("device_call", GroupedDeviceCall),
                ("compute_type", C.c_int32)]


class ComputeIdentity(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32),
                ("parent", C.POINTER(Identity)), ("compute_type", C.c_int32)]


def bind_compute(lib, grouped):
    prefix = "quactlize_kpack_grouped" if grouped else "quactlize_kpack_decode_dense"
    version = 3 if grouped else 2
    call = GroupedComputeCall if grouped else DenseComputeCall
    identity = getattr(lib, "quactlize_kpack_compute_identity_v3" if grouped
                       else "quactlize_kpack_decode_dense_identity_v2")
    identity.argtypes, identity.restype = [], C.POINTER(ComputeIdentity)
    query = getattr(lib, f"{prefix}_query_v{version}")
    query.argtypes, query.restype = [C.POINTER(call), C.POINTER(Recipe), C.POINTER(Resources)], C.c_int
    prepare = getattr(lib, f"{prefix}_prepare_v{version}")
    prepare.argtypes, prepare.restype = [C.POINTER(call), C.POINTER(Recipe), C.POINTER(C.c_void_p)], C.c_int
    lifecycle = "quactlize_kpack" if grouped else "quactlize_kpack_decode_dense"
    run, destroy = getattr(lib, f"{lifecycle}_run_v1"), getattr(lib, f"{lifecycle}_destroy_v1")
    run.argtypes, run.restype = [C.c_void_p, C.c_void_p], C.c_int
    destroy.argtypes, destroy.restype = [C.c_void_p], None
    return identity, query, prepare, run, destroy

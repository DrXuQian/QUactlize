"""Unified public FFN ABI; correctness-test bindings, no selection logic."""
import ctypes as C
from ci.support.dense import Arrangement, Sizes


class Weight(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('qtype', 'n', 'k', 'experts')] + [
        ('arrangement', Arrangement), ('sizes', Sizes)] + [(n, C.c_void_p) for n in ('low', 'high', 'units')]


class Params(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [(n, C.c_int32) for n in
        ('kind', 'tokens', 'topk', 'channels', 'merged', 'reserved')] + [(n, C.c_int64) for n in
        ('input_row_stride', 'input_token_stride', 'ids_stride', 'weights_stride', 'output_stride')] + [
        (n, Weight) for n in ('gate', 'up', 'down')]


class ProjectionInfo(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('route', 'policy', 'split', 'compute_type')] + [('config', C.c_char * 192)]


class Info(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('workspace_bytes', C.c_uint64)] + [
        (n, C.c_int32) for n in ('prepare_stage', 'prepare_helper', 'finish_helper', 'reserved')] + [
        ('projections', ProjectionInfo * 3)]


def projection_output_compute(kind, m, info):
    """Intermediate C type of the independent public projection, not the ABI.

    Mirrors dense/runtime.cpp's native_f16 boundary and the MoE TC BF16
    scatter. Small-M typed TC and SIMT store F32. Full reports its actual
    storage in compute_type. No recipe/precision is selected by this helper.
    """
    if kind not in (0, 1) or m < 1 or info.route not in (0, 1, 2, 3) or info.compute_type not in (0, 1):
        raise ValueError("unknown FFN projection precision contract")
    if info.route == 3:
        if kind:
            raise ValueError("unexpected Full provider in independent MoE projection")
        return "bf16" if info.compute_type else "f16"
    if info.route == 2:
        return None
    if kind:
        if info.compute_type != 1:
            raise ValueError("independent MoE TC requires BF16 compute")
        return "bf16"
    # QKS_LEGACY_DENSE=17 is the native-F16 fallback even below M9.
    if info.compute_type == 0 and (m > 8 or info.policy == 17):
        return "f16"
    return None


class Public:
    def __init__(self, library):
        def bind(name, args, result=C.c_int):
            fn = getattr(library, 'quactlize_ppu_ffn_' + name)
            fn.argtypes, fn.restype = args, result
            return fn
        self.supported = bind('supported_v1', [C.POINTER(Params)])
        self.create = bind('create_v1', [C.c_int, C.POINTER(C.c_void_p)])
        self.prepare = bind('prepare_v1', [C.c_void_p, C.POINTER(Params), C.POINTER(C.c_void_p), C.POINTER(Info)])
        self.run = bind('run_v1', [C.c_void_p] * 6 + [C.c_uint64, C.c_void_p])
        self.destroy = bind('destroy_v1', [C.c_void_p], None)
        self.error = bind('error_v1', [], C.c_char_p)

    def check(self, rc):
        if rc:
            raise AssertionError(f'public FFN rc={rc}: {self.error().decode(errors="replace")}')

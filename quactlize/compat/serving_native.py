"""Pure ctypes serving ABI; no Torch import or device allocation."""
import ctypes as C
from .native import fields, Merge
from quactlize._library import load_library, bind_symbol

BAD_ID, BAD_LOGIT, BAD_COUNT = 1, 2, 4


class Router(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) +
        fields('capacity experts topk dtype score_func renormalize', C.c_int32) +
        [('multiplier', C.c_float), ('reserved', C.c_int32)] +
        fields('logits_stride selected_stride ids_stride weights_stride', C.c_int64) +
        fields('logits_bytes selected_bytes ids_bytes weights_bytes', C.c_uint64) +
        fields('logits selected_logits ids weights num_valid_tokens device_error stream', C.c_void_p))


class SwiGLU(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) +
        fields('rows hidden dtype merged round_silu reserved', C.c_int32) +
        fields('gate_stride up_stride output_stride', C.c_int64) +
        fields('gate_bytes up_bytes output_bytes', C.c_uint64) +
        fields('gate up output stream', C.c_void_p))


class Rows(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) + fields('qtype dtype capacity columns', C.c_int32) +
        [('table_rows', C.c_int64)] + fields('raw_bytes ids_bytes output_bytes', C.c_uint64) +
        fields('raw ids output num_rows device_error stream', C.c_void_p))


class Library:
    def __init__(self, path=None, *, target=None):
        self.lib = load_library(path, target)
        def bind(name, args, result=C.c_int):
            return bind_symbol(self.lib, 'quactlize_serving_' + name, args, result)
        self.prepare = bind('prepare_v1', [C.c_int32, C.c_void_p])
        self.error = bind('error_v1', [], C.c_char_p)
        self.router = bind('router_v2', [C.POINTER(Router)])
        self.swiglu = bind('swiglu_v2', [C.POINTER(SwiGLU)])
        self.rows = bind('gather_rows_v1', [C.POINTER(Rows)])
        self.merge = bind('merge_v1', [C.POINTER(Merge)])

    def check(self, rc):
        if rc:
            raise RuntimeError(f'Quactlize serving rc={rc}: {self.error().decode(errors="replace")}')

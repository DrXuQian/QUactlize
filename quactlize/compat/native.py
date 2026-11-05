"""ctypes declarations for the additive, caller-owned native-16-bit C ABI.

No Torch/Python runtime is required by the library. This module allocates no
device buffers, chooses no CUDA stream and does not implement a CPU fallback.
"""
import ctypes as C


def fields(names, dtype):
    return [(name, dtype) for name in names.split()]


class Problem(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) + fields(
        'qtype layout mode dtype m n k experts topk channels device reserved', C.c_int32) +
        fields('input_row_stride input_token_stride ids_stride output_stride', C.c_int64))


class Sizes(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) + fields(
        'raw_bytes low_bytes high_bytes units_bytes input_bytes output_bytes ids_bytes workspace_bytes '
        'expanded_weight_bytes grouped_input_bytes grouped_output_bytes routing_bytes', C.c_uint64) +
        fields('route expert_tile column_tile reserved', C.c_int32))


class Buffers(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) + fields(
        'input raw low high units ids output workspace', C.c_void_p) + fields(
        'input_bytes raw_bytes low_bytes high_bytes units_bytes ids_bytes output_bytes workspace_bytes', C.c_uint64) +
        fields('device_error stream', C.c_void_p))


class Router(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) + fields('m experts topk reserved', C.c_int32) +
        fields('logits_stride ids_stride weights_stride', C.c_int64) + [('multiplier', C.c_float)] +
        fields('logits ids weights device_error stream', C.c_void_p))


class SwiGLU(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) + fields('rows hidden dtype merged', C.c_int32) +
        fields('gate_stride up_stride output_stride', C.c_int64) + fields('gate up output stream', C.c_void_p))


class Merge(C.Structure):
    _fields_ = (fields('version size', C.c_uint32) + fields('m topk n dtype', C.c_int32) +
        fields('input_stride weights_stride output_stride', C.c_int64) + fields('input weights output stream', C.c_void_p))


def problem(q, m, n, k, *, experts=1, topk=1, channels=1, dtype=1, layout=0, device=0, padding=0):
    indexed = experts > 1 or topk > 1
    return Problem(1, C.sizeof(Problem), q, layout, int(indexed), dtype, m, n, k,
        experts, topk, channels, device, 0, k + padding, channels * (k + padding),
        topk + padding if indexed else 0, n + padding)


class Library:
    def __init__(self, path, *, packed_route=None):
        self.lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        def bind(name, args, result=C.c_int):
            fn = getattr(self.lib, 'quactlize_ppu_' + name)
            fn.argtypes, fn.restype = args, result
            return fn
        self.query = bind('matmul_query_v1', [C.POINTER(Problem), C.c_void_p, C.POINTER(Sizes)])
        self.prepare = bind('matmul_prepare_v1', [C.POINTER(Problem), C.c_void_p, C.POINTER(Buffers), C.POINTER(C.c_void_p)])
        if packed_route is not None:
            query = bind('matmul_packed_query_v1', [C.POINTER(Problem), C.c_void_p, C.c_int, C.POINTER(Sizes)])
            prepare = bind('matmul_packed_prepare_v1', [C.POINTER(Problem), C.c_void_p, C.c_int,
                C.POINTER(Buffers), C.POINTER(C.c_void_p)])
            self.query = lambda p, a, s: query(p, a, packed_route, s)
            self.prepare = lambda p, a, b, h: prepare(p, a, packed_route, b, h)
        self.run = bind('matmul_run_v1', [C.c_void_p, C.POINTER(Buffers)])
        self.dequant = bind('matmul_dequant_v1', [C.c_void_p, C.POINTER(Buffers), C.c_int32, C.c_int32,
                                              C.c_int32, C.c_void_p, C.c_uint64])
        self.destroy = bind('matmul_destroy_v1', [C.c_void_p], None)
        self.error = bind('matmul_error_v1', [], C.c_char_p)
        self.router = bind('router_run_v1', [C.POINTER(Router)])
        self.swiglu = bind('swiglu_run_v1', [C.POINTER(SwiGLU)])
        self.merge = bind('merge_run_v1', [C.POINTER(Merge)])

    def check(self, rc):
        if rc:
            raise RuntimeError(f'Quactlize native matmul rc={rc}: {self.error().decode(errors="replace")}')

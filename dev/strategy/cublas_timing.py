"""Prebound native submission around the same installed cuBLAS function/handle."""
import ctypes as C

from quactlize.runtime.native import checked

FRONTEND = 'CPP_PREBOUND_CONTINUOUS'


class NativeCalls:
    def __init__(self, b, core):
        if b.w.operation != 'dense' or b.w.experts != 1:
            raise ValueError('native cuBLAS timer is dense-only')
        self.b, self.core = b, core
        provider = core.provider
        # Torch's current-stream context is unnecessary: this handle is already
        # bound to the nonblocking benchmark stream. Verify rather than rebind it
        # (SetStream would reset a caller-owned workspace).
        get_stream = provider.lib.cublasGetStream_v2
        get_stream.argtypes, get_stream.restype = [C.c_void_p, C.POINTER(C.c_void_p)], C.c_int
        stream = C.c_void_p()
        checked(get_stream(provider.handle, C.byref(stream)), 'cuBLAS timer stream query')
        if stream.value != b.r.stream.value:
            raise ValueError('cuBLAS handle and event stream differ')
        self.entry = C.cast(provider.lib.cublasGemmEx, C.c_void_p)
        self.weights = (C.c_void_p*core.copies)(*[w.data_ptr() for w in core.weights])
        self.a, self.out = core.a.data_ptr(), core.out.data_ptr()
        self.run = b.helpers.lib.strategy_prefill_cublas_launch
        self.run.argtypes = [C.c_void_p]*5+[C.c_int]*3
        self.run.restype = C.c_int
        self.timer = b.helpers.lib.strategy_prefill_cublas_events
        self.timer.argtypes = [C.c_void_p]*9+[C.c_int]*7+[C.POINTER(C.c_float), C.POINTER(C.c_double)]
        self.timer.restype = C.c_int

    def matmul(self, copy=0):
        if not 0 <= copy < self.core.copies: raise ValueError('weight ring index out of bounds')
        w = self.b.w
        return self.run(self.entry, self.core.provider.handle, self.a, self.weights[copy], self.out,
                        w.m, w.n, w.k)

    def measure(self, warmups):
        b, core = self.b, self.core
        count, rounds = max(3, core.copies), 3
        total = rounds*count*3
        gpu, host = (C.c_float*total)(), (C.c_double*total)()
        checked(self.timer(self.entry, core.provider.handle, b.r.stream, b.a.ptr, self.a,
            self.weights, self.out, b.out.ptr, b.error.ptr, b.w.m, b.w.n, b.w.k,
            core.copies, warmups, rounds, count, gpu, host), 'native cuBLAS event submission')
        phases = ('input', 'provider', 'output')
        return [dict(round=i//(count*3), sample=(i//3) % count, phase=phases[i % 3],
                     copy=(i//3) % core.copies, event_span_us=float(gpu[i]), host_submit_us=host[i])
                for i in range(total)]

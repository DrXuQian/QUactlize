"""Numerical/profile adapters for the shipping execution C entry points."""
import ctypes as C
from dataclasses import replace

from dev.gemv_simt.native import checked
from quactlize.execution.native import (
    SimtCallV2, SimtConfig, Q4DecodeConfig, Sizes, Arrangement, arrangement,
    bind_simt_compute,
)
from quactlize.execution.simt_codegen import inventory


class Library:
    def __init__(self, path, compute=0, arm=None):
        self.lib = C.CDLL(str(path.resolve(strict=True)), mode=C.RTLD_LOCAL)
        self.compute, self.arm = compute, arm
        self.query, self.run = bind_simt_compute(self.lib)

    def configs(self):
        return inventory(8, legacy=True)

    def prepare(self, call, config):
        if self.arm == 1:
            config = replace(config, variant=config.variant + 4)
        f = SimtConfig(config.variant, config.columns, config.warps, config.values, config.split)
        d, a, s = SimtCallV2(call, self.compute), arrangement(call.qtype), Sizes()
        checked(self.query(C.byref(d), C.byref(f), C.byref(a), C.byref(s)), 'production SIMT query')
        if s.workspace_bytes > call.workspace_bytes:
            raise ValueError('production SIMT workspace is too small')
        return lambda: self.run(C.byref(d), C.byref(f), C.byref(a))


class Q4Library:
    def __init__(self, path, compute):
        self.lib = C.CDLL(str(path.resolve(strict=True)), mode=C.RTLD_LOCAL)
        self.compute = compute
        self.select = self.lib.quactlize_kpack_q4_decode_select_v2
        self.run = self.lib.quactlize_kpack_q4_decode_run_v2
        self.select.argtypes = [C.POINTER(SimtCallV2), C.POINTER(Arrangement),
                                C.POINTER(Q4DecodeConfig), C.POINTER(Sizes)]
        self.run.argtypes = [C.POINTER(SimtCallV2), C.POINTER(Q4DecodeConfig), C.POINTER(Arrangement)]
        self.select.restype = self.run.restype = C.c_int

    def prepare(self, call, config):
        d, a, f, s = SimtCallV2(call, self.compute), arrangement(12), Q4DecodeConfig(), Sizes()
        checked(self.select(C.byref(d), C.byref(a), C.byref(f), C.byref(s)), 'production Q4 select')
        if any(getattr(f, k) != getattr(config, k) for k in ('reader','variant','columns','warps','values')):
            raise ValueError('Q4 selected recipe differs from the observed model recipe')
        if s.workspace_bytes > call.workspace_bytes:
            raise ValueError('production Q4 workspace is too small')
        return lambda: self.run(C.byref(d), C.byref(f), C.byref(a))

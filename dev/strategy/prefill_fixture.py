"""Large-M F32 caller fixture, without an M*N*K host allocation.

The resident row maps are declared inputs, not a claimed timed CPU router.
Current production GPU casts/gathers/finishes are measured separately.
"""
import ctypes as C
import hashlib

import numpy as np

from dev.bf16_compute.fixture import bf16_bits, round_compute
from dev.smallm_closure.bench import Buffer
from dev.strategy.indexed import RuntimeView
from quactlize.execution.native import arrangement
from quactlize.runtime.native import checked
from tools.run_kpack_gemv_gate import Resources


def routing(w, repeat=0):
    if w.operation == 'dense':
        owners = np.zeros(w.m, dtype='i4')
        source = np.arange(w.m, dtype='i4')
    else:
        router = w.router or 'real'
        if router == 'real':
            rng = np.random.default_rng(4013 + repeat)
            weight = np.ones(w.experts)
            weight[:min(16, w.experts)] = 4
            ids = np.stack([rng.choice(w.experts, w.topk, replace=False,
                p=weight/weight.sum()) for _ in range(w.m)]).astype('i4')
        else:
            buckets = np.arange(w.m)
            if router == 'cluster': buckets[:] = 0
            elif router.startswith('repeat'): buckets //= int(router[6:])
            elif router != 'spread': raise ValueError('unknown prefill router')
            ids = ((np.arange(w.topk)[None, :] + 13*buckets[:, None] + repeat*7) % w.experts).astype('i4')
        owners = ids.ravel()
        index = np.arange(w.m*w.topk)
        source = (index//w.topk*w.channels + index%w.topk%w.channels).astype('i4')
    order = np.argsort(owners, kind='stable').astype('i4')
    counts = np.bincount(owners, minlength=w.experts).astype('i4')
    offsets = np.r_[0, counts.cumsum()].astype('i4')
    active = np.flatnonzero(counts).astype('i4')
    return dict(owners=owners, source=source, sorted_source=source[order], order=order,
                sorted_owners=owners[order], counts=counts, offsets=offsets, active=active,
                sha256=hashlib.sha256(owners.tobytes()).hexdigest())


class Helpers:
    def __init__(self, path):
        self.lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        self.io = self.lib.strategy_prefill_io
        self.io.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int,
            C.c_int, C.c_int64, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
        self.io.restype = C.c_int
        self.active = self.lib.strategy_prefill_active
        self.active.argtypes = [C.c_void_p]*5 + [C.c_int, C.c_void_p]
        self.active.restype = C.c_int
        self.clone = self.lib.strategy_prefill_clone_capture
        self.clone.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
        self.clone.restype = C.c_int


class Context:
    def __init__(self, sdk, ring, w, helpers):
        self.sdk, self.ring, self.w, self.helpers = sdk, ring, w, helpers
        self.p = dict(q=w.qtype, n=w.n, k=w.k, tokens=w.m, mode=int(w.operation == 'indexed')*2,
                      experts=w.experts, channels=w.channels, topk=w.topk, compute=w.compute)
        self.r = Resources(sdk)
        self.rows = w.m*w.topk
        self.arr = arrangement(w.qtype)
        self.buffers = []
        self.a = self.allocate(w.m*w.channels*w.k*4)
        self.out = self.allocate(self.rows*w.n*4)
        self.offsets = self.allocate((w.experts+1)*4)
        self.source = self.allocate(self.rows*4)
        self.destination = self.allocate(self.rows*4)
        self.error = self.allocate(4)
        self.error.upload(np.zeros(1, dtype='i4'))
        self.rt = RuntimeView(self)
        self.update(0)

    def allocate(self, size):
        b = Buffer(self.r, size)
        self.buffers.append(b)
        return b

    def update(self, repeat, *, zero=False, amplitude=1):
        w = self.w
        self.route = routing(w, repeat)
        if (self.ring.receipt['domain'] != 'ALL_EXPERT_METADATA' and
                len(self.route['active']) < self.ring.receipt['minimum_active']):
            raise ValueError('current router violates the frozen cold-ring active lower bound')
        values = np.random.default_rng(931+repeat).uniform(-.5, .5, (w.m*w.channels, 4)).astype('f4')
        # Exact in F16/BF16 for fair route comparison; all routes see the same
        # F32 caller bytes. The BF16-only large-activation control is separate.
        self.coeff = round_compute(values, 'bf16') * np.float32(amplitude)
        data = np.zeros_like(self.coeff) if zero else self.coeff
        self.a.upload(np.ascontiguousarray(data[:, self.ring.w.categories], dtype='f4'))
        self.offsets.upload(self.route['offsets'])
        self.source.upload(self.route['sorted_source'])
        self.destination.upload(self.route['order'])
        self.error.upload(np.zeros(1, dtype='i4'))
        self.out.poison()

    def input(self, ptr, bf16):
        w = self.w
        return self.helpers.io(int(bf16), 1, self.a.ptr, ptr, self.rows, w.k, w.k,
            self.source.ptr if w.operation == 'indexed' else None, w.m*w.channels,
            self.error.ptr, self.r.stream)

    def finish(self, ptr, bf16):
        w = self.w
        return self.helpers.io(int(bf16), 0, ptr, self.out.ptr, self.rows, w.n, w.n,
            self.destination.ptr if w.operation == 'indexed' else None, self.rows,
            self.error.ptr, self.r.stream)

    def check(self, *, zero=False):
        self.rt.sync()
        if int(np.frombuffer(self.sdk.download(self.error.ptr, 4), dtype='i4')[0]):
            raise ValueError('resident row map was rejected on device')
        maximum, cells = 0., 0
        for first in range(0, self.rows, 64):
            stop = min(first+64, self.rows)
            got = np.frombuffer(self.sdk.download(self.out.ptr+first*self.w.n*4,
                (stop-first)*self.w.n*4), dtype='f4').reshape(stop-first, self.w.n)
            if not np.isfinite(got).all(): raise ValueError('prefill oracle: nonfinite output')
            if zero:
                if np.any(got): raise ValueError('prefill oracle: zero A produced nonzero output')
                continue
            patterns = self.route['owners'][first:stop] % len(self.ring.w.sums)
            coeff = self.coeff[self.route['source'][first:stop]].astype('f8')
            want = np.einsum('rg,rgn->rn', coeff, self.ring.w.sums[patterns])
            denom = np.einsum('rg,rgn->rn', np.abs(coeff), self.ring.w.absolute[patterns])
            error = np.abs(got.astype('f8')-want)/np.maximum(denom, 1e-30)
            maximum = max(maximum, float(error.max(initial=0)))
            cells += got.size
        if maximum >= .005: raise ValueError(f'prefill oracle: independent GGUF error {maximum}')
        for b in self.buffers: b.guard()
        return dict(relative_l1_error=maximum, cells=cells, nonfinite=0, tolerance=.005,
                    fixture='ORIGINAL_GGUF_COMMON_DYADIC_F64_FACTORIZED_DOT', routes=self.route['sha256'])

    def close(self):
        self.r.close()

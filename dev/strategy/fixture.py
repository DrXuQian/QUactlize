"""Factorized independent GGUF oracle; one shared weight fixture per family."""
import hashlib
import math
import numpy as np

from dev.bf16_compute.fixture import round_compute
from dev.gemv_model.fixture import expert_payload
from dev.gemv_model.plan import Point
from dev.gate_up_perf.bench import Buffer, WeightRing
from tools.kpack_warmup_fixture import activation_values


class Dense:
    def __init__(self, rt, workload, max_m, l2_bytes):
        w = workload
        self.rt, self.w, self.max_m = rt, w, max_m
        self.category = np.random.default_rng(727 + w.k).integers(0, 4, w.k)
        p = Point('strategy', w.qtype, w.n, w.k, int(w.compute == 'bf16'))
        payload, self.sums, self.absolute, raw = expert_payload(p, 0, self.category)
        self.weight_bytes = sum(v.nbytes for v in payload.values())
        self.copies = max(2, math.ceil(2.25 * l2_bytes / self.weight_bytes))
        self.ring = WeightRing(rt, {0: payload}, 1, self.copies)
        self.a, self.output = Buffer(rt, max_m * w.k * 4), Buffer(rt, max_m * w.n * 4)
        self.record = dict(raw_sha256=raw,
            category_sha256=hashlib.sha256(self.category.tobytes()).hexdigest(),
            plane_sha256={n: hashlib.sha256(v.tobytes()).hexdigest() for n,v in payload.items()},
            oracle='OFFICIAL_GGUF_DYADIC_WEIGHT_F32_DOT', l2_bytes=l2_bytes,
            copies=self.copies, active_weight_ring_bytes=self.copies*self.weight_bytes)
        self.update(w)

    def update(self, w, repeat=0, zero=False):
        self.w = w
        coefficients = activation_values(np.arange(w.m) + repeat * 137).astype('f4')
        if zero:
            coefficients.fill(0)
        self.rt.copy(self.a.ptr, np.ascontiguousarray(coefficients[:, self.category]))
        rounded = round_compute(coefficients, w.compute).astype('f8')
        self.gold = rounded @ self.sums
        self.denom = np.abs(rounded) @ self.absolute
        self.output.poison()
        self.rt.sync()

    def check(self, zero=False):
        self.rt.sync()
        self.a.check_guard()
        raw = self.output.read()
        size = self.w.m * self.w.n * 4
        if np.any(raw[size:] != 0xa5):
            raise ValueError('unowned output rows changed')
        got = raw[:size].view('f4').reshape(self.w.m, self.w.n)
        error = float(np.max(np.abs(got.astype('f8')-self.gold)/np.maximum(self.denom,1e-20)))
        if not np.isfinite(got).all() or not math.isfinite(error) or error >= .005 or (zero and np.any(got)):
            raise ValueError(f'independent GGUF dot oracle failed error={error}')
        return error

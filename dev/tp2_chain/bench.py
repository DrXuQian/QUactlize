"""Production helper chain with identical GGUF weights in two named layouts."""
import ctypes as C
import hashlib
import math

import numpy as np

from dev.bf16_compute.fixture import planes, round_compute
from dev.bf16_compute.native import Projection, MoePlan, MixedPlanV2, function
from dev.gate_up_perf.bench import Buffer, WeightRing, physical
from dev.gate_up_perf.plan import ring_copies
from dev.gemv_model.fixture import varied_weight
from dev.gemv_simt.native import checked
from dev.gemv_simt.production import Library as ShippingSimt
from dev.tp2_chain.plan import ARMS, projection_config
from quactlize.dispatch.native import Router, MoeFinish
from quactlize.execution.native import Call
from quactlize.execution.simt_codegen import Config
from quactlize.fusion.native import Library as PairedLibrary, MappedCall, FusionCall, Config as PairedConfig, Layout, integration_entries
from tools.kpack_warmup_fixture import activation_values


def swiglu(gate, up):
    gate, up = (round_compute(x.astype('f4'), 'bf16') for x in (gate, up))
    with np.errstate(over='ignore'):
        return (gate / (np.float32(1) + np.exp(-gate)) * up).astype('f4')


def finish_gold(activation, down, weights):
    """FP64 independent dot, explicit BF16 boundaries, ordered F32 slot sum."""
    a = round_compute(activation, 'bf16').astype('f8')
    raw = np.stack([w.astype('f8') @ row for w, row in zip(down, a)])
    absolute = np.stack([np.abs(w.astype('f8')) @ np.abs(row) for w, row in zip(down, a)])
    rounded = round_compute(raw.astype('f4'), 'bf16')
    products = (rounded.reshape(len(weights), 8, -1) * weights[:, :, None]).astype('f4')
    result = products[:, 0].copy()
    for slot in range(1, 8):
        result = np.add(result, products[:, slot], dtype='f4')
    denom = (absolute.reshape(len(weights), 8, -1) * np.abs(weights[:, :, None])).sum(axis=1)
    return result, denom, raw, absolute


def normalized_error(got, gold, denominator):
    if got.shape != gold.shape or not np.isfinite(got).all():
        raise ValueError('nonfinite or wrong-shape chain output')
    error = float(np.max(np.abs(got.astype('f8') - gold) / np.maximum(denominator, 1e-20)))
    if not math.isfinite(error) or error >= .005:
        raise ValueError(f'independent BF16/GGUF chain oracle differs: {error}')
    return error


class Bench:
    def __init__(self, rt, bundle, manifest, point, l2, profile=False, candidates=True):
        self.rt, self.point = rt, point
        self.library = C.CDLL(str(bundle / 'libquactlize_ppu_execution.so'), mode=C.RTLD_LOCAL)
        self.shipping = ShippingSimt(bundle / 'libquactlize_ppu_execution.so', 1)
        self.paired = PairedLibrary(bundle / 'libquactlize_ppu_gate_up.so')
        self.paired_run = integration_entries(self.paired)['run']
        self.layout = self.paired.arrangement(12)
        if candidates:
            record = next(r for r in manifest['records'] if r['point']['name'] == point.name)
            self.candidate_lib = C.CDLL(str(bundle / record['library']), mode=C.RTLD_LOCAL)
            self.candidate_run = function(self.candidate_lib, 'model_gemv_run',
                                          [C.c_void_p, C.POINTER(MappedCall), C.POINTER(Layout), C.c_int])
        self.query = function(self.library, 'quactlize_kpack_moe_simt_query_v1',
                              [C.POINTER(Call), C.POINTER(C.c_uint64)])
        self.bind = function(self.library, 'quactlize_kpack_moe_simt_bind_v1',
                             [C.POINTER(Call), C.c_int, C.c_void_p, C.c_uint64, C.POINTER(Projection)])
        self.prepare_fn = function(self.library, 'quactlize_kpack_moe_prepare_finish_v1',
                                   [C.POINTER(MixedPlanV2), C.POINTER(MoeFinish), C.c_void_p])
        self.activate_fn = function(self.library, 'quactlize_kpack_moe_mixed_stage_v2',
                                    [C.POINTER(MixedPlanV2), C.c_int, C.c_void_p])
        self.finish_fn = function(self.library, 'quactlize_kpack_moe_weighted_finish_v2',
                                  [C.POINTER(MixedPlanV2), C.POINTER(MoeFinish), C.c_void_p])
        self.expert_sets = ((13 + np.arange(16) * 29) % 256).astype('i4')
        self.category = np.random.default_rng(727 + point.k).integers(0, 4, point.k)
        canonical, paired, down, hashes = {}, {}, {}, {}
        self.gate_sums, self.up_sums, self.down_gold = {}, {}, {}
        for ordinal, expert in enumerate(self.expert_sets):
            e = int(expert)
            gate, wg = varied_weight(12, point.n, point.k, 181 + e * 71)
            up, wu = varied_weight(12, point.n, point.k, 947 + e * 71)
            raw_down, wd = varied_weight(13, point.k, point.n, 337 + e * 71)
            canonical[e] = planes(physical(gate, up, False), 12)
            paired[e] = planes(physical(gate, up, True), 12)
            down[e] = planes(raw_down, 13)
            self.gate_sums[e] = np.stack([wg[:, self.category == i].sum(axis=1, dtype='f8') for i in range(4)])
            self.up_sums[e] = np.stack([wu[:, self.category == i].sum(axis=1, dtype='f8') for i in range(4)])
            self.down_gold[e] = wd
            hashes[e] = {name: hashlib.sha256(raw.tobytes()).hexdigest()
                         for name, raw in (('gate', gate), ('up', up), ('down', raw_down))}
            print(f'PAIRED_CHAIN_FIXTURE point={point.name} experts={ordinal+1}/16', flush=True)
        size = lambda data: sum(next(iter(data.values()))[name].nbytes for name in ('low', 'high', 'units'))
        if size(canonical) != size(paired):
            raise ValueError('canonical/paired weight traffic differs')
        self.weight_bytes = (size(canonical) + size(down)) * 8
        self.copies = 1 if profile else ring_copies(l2, self.weight_bytes)
        self.rings = {name: WeightRing(rt, data, 256, self.copies)
                      for name, data in (('canonical', canonical), ('paired', paired), ('down', down))}
        self.input = Buffer(rt, 8 * point.k * 4)
        self.ids = Buffer(rt, 8 * 11 * 4)
        self.logits = Buffer(rt, 8 * 256 * 4)
        self.weights = Buffer(rt, 8 * 8 * 4)
        self.output = Buffer(rt, 8 * point.k * 4)
        self.gate_output = Buffer(rt, 64 * 2 * point.n * 4)
        self.down_input = Buffer(rt, 64 * point.n * 4)
        self.down_output = Buffer(rt, 64 * point.k * 4)
        self.buffers = (self.input, self.ids, self.logits, self.weights, self.output,
                        self.gate_output, self.down_input, self.down_output)
        self.scratch = []
        self.record = dict(raw_sha256=hashes, category_sha256=hashlib.sha256(self.category.tobytes()).hexdigest(),
                           canonical_layout='Q4_KPACK', paired_layout='PAIRED_N4', down_layout='Q5_KPACK',
                           experts=256, active_experts=8, physical_weight_bytes_per_call=self.weight_bytes,
                           copies=self.copies, rotating_active_bytes=self.copies*self.weight_bytes, l2_bytes=l2,
                           scope='ACU_CACHE_CONTROL_ALL' if profile else 'COLD_COMPLETE_CHAIN_WEIGHT_RING',
                           oracle='OFFICIAL_GGUF_FP64_DOT_BF16_BOUNDARIES_ORDERED_F32_FINISH')

    def configure(self, tokens, router=True, alias=True):
        if not 1 <= tokens <= 8:
            raise ValueError('token control outside 1..8')
        self.tokens, self.router_enabled, self.alias = tokens, router, alias
        self.weight_ptr = self.output.ptr if alias else self.weights.ptr
        parts, calls = [], []
        for q, n, k, channels, a, out in (
                (12, 2*self.point.n, self.point.k, 1, self.input.ptr, self.gate_output.ptr),
                (13, self.point.k, self.point.n, 8, self.down_input.ptr, self.down_output.ptr)):
            call = Call(version=1, size=C.sizeof(Call), qtype=q, n=n, k=k, experts=256,
                        rows=tokens*8, mode=2, input_type=1, channels=channels, topk=8,
                        a_row_stride=k, a_token_stride=channels*k, ids_stride=11, out_row_stride=n,
                        a=a, ids=self.ids.ptr, output=out, stream=self.rt.stream.value,
                        **self.rings['canonical' if q == 12 else 'down'].at(0))
            size = C.c_uint64()
            checked(self.query(C.byref(call), C.byref(size)), 'chain scratch query')
            scratch = Buffer(self.rt, size.value)
            part = Projection()
            checked(self.bind(C.byref(call), -1, scratch.ptr, size.value, C.byref(part)), 'production SIMT bind')
            if q == 12:
                call.output = part.output
            else:
                call.a = part.a
            parts.append(part); calls.append(call); self.scratch.append(scratch)
        route = Router(1, C.sizeof(Router), 0, 1, 0, 0, 2**-14, 1.25,
                       self.logits.ptr, None, self.weight_ptr) if router else Router()
        plan = MoePlan(1, C.sizeof(MoePlan), 1, 0, parts[0], Projection(), parts[1], route)
        self.typed = MixedPlanV2(2, C.sizeof(MixedPlanV2), plan, 5, 1)
        self.calls, self.parts = calls, parts
        self.finish = MoeFinish(1, C.sizeof(MoeFinish), 8, self.point.k, self.weight_ptr, self.output.ptr)
        self.retained = MoeFinish(1, C.sizeof(MoeFinish), 8, self.point.k, plan.down.outputs, self.output.ptr)

    def update(self, repeat, large=False, invalid=None):
        t, p = self.tokens, self.point
        for b in (*self.buffers, *self.scratch):
            b.poison()
        coeff = activation_values(np.arange(t)+repeat*137).astype('f4')
        if large:
            coeff[0, 0] = np.float32(243383.484375)
        a = round_compute(coeff, 'bf16').astype('f8')
        base = self.expert_sets[8:16] if repeat % 2 else self.expert_sets[:8]
        ids = np.full((t, 11), -91, 'i4')
        scores = np.full((t, 256), -80, 'f4')
        for token in range(t):
            ids[token, :8] = np.roll(base, token+repeat)
            scores[token, ids[token, :8]] = -np.arange(8, dtype='f4')/8
        vals = np.take_along_axis(scores, ids[:, :8], axis=1).astype('f8')
        ex = np.exp(vals-vals.max(axis=1, keepdims=True))
        weights = (ex/ex.sum(axis=1, keepdims=True)*1.25).astype('f4')
        owners = ids[:, :8].reshape(-1)
        gate = np.stack([a[i//8] @ self.gate_sums[int(e)] for i, e in enumerate(owners)])
        up = np.stack([a[i//8] @ self.up_sums[int(e)] for i, e in enumerate(owners)])
        self.activation_gold = swiglu(gate, up)
        self.gold, self.denom, self.down_dot, self.down_abs = finish_gold(
            self.activation_gold, [self.down_gold[int(e)] for e in owners], weights)
        if not np.isfinite(self.gold).all() or not np.any(self.gold):
            raise ValueError('degenerate full-chain fixture')
        if invalid is not None:
            if self.router_enabled:
                raise ValueError('invalid ID control must disable the router producer')
            ids[0, 0] = invalid
        self.rt.copy(self.input.ptr, np.ascontiguousarray(coeff[:, self.category]))
        self.rt.copy(self.ids.ptr, ids)
        self.rt.copy(self.logits.ptr, scores)
        self.rt.copy(self.weight_ptr, weights)
        self.ids_host, self.weights_host = ids, weights
        self.rt.sync()

    def launch(self, arm, copy=0):
        if arm not in ARMS or not 0 <= copy < self.copies:
            raise ValueError('invalid chain arm/copy')
        gate, down = (Call.from_buffer_copy(c) for c in self.calls)
        for call, name in ((gate, 'canonical'), (down, 'down')):
            for field, value in self.rings[name].at(copy).items():
                setattr(call, field, value)
        produce_down = self.shipping.prepare(down, Config(**projection_config(13)))
        if arm == 'unfused':
            produce_gate = self.shipping.prepare(gate, Config(**projection_config(12)))
            middle = (produce_gate, lambda: self.activate_fn(C.byref(self.typed), 2, self.rt.stream))
        else:
            gate.n = self.point.n
            gate.out_row_stride = self.point.n
            gate.output = self.parts[1].a
            for field, value in self.rings['paired'].at(copy).items():
                setattr(gate, field, value)
            mapped = MappedCall(FusionCall(gate, 1, 1, 1), None, self.parts[0].directory_header+4)
            cfg = PairedConfig(0, 1, 0, 8)
            if arm == 'paired-shipping':
                middle = (lambda: self.paired_run(C.byref(mapped), C.byref(cfg), C.byref(self.layout)),)
            else:
                index = int(arm.rsplit('-', 1)[1])
                middle = (lambda: self.candidate_run(None, C.byref(mapped), C.byref(self.layout), index),)
        calls = (lambda: self.prepare_fn(C.byref(self.typed), C.byref(self.finish), self.rt.stream),
                 *middle, produce_down,
                 lambda: self.finish_fn(C.byref(self.typed), C.byref(self.retained), self.rt.stream))

        def invoke():
            for fn in calls:
                rc = fn()
                if rc:
                    return rc
            return 0
        return invoke

    def check(self, invalid=False):
        self.rt.sync()
        t, p = self.tokens, self.point
        raw = self.output.read()
        got = raw[:t*p.k*4].view('f4').reshape(t, p.k)
        if not np.all(raw[t*p.k*4:] == 0xA5):
            raise ValueError('finish wrote unowned token rows')
        if invalid:
            if not np.isnan(got).all():
                raise ValueError('invalid route did not poison the entire chain')
            return 0.
        error = normalized_error(got, self.gold, self.denom)
        activation = self.rt.download(self.parts[1].a, t*8*p.n*4).view('f4').reshape(t*8, p.n)
        norm = np.full_like(self.activation_gold, max(np.max(np.abs(self.activation_gold)), 1e-20))
        activation_error = normalized_error(activation, self.activation_gold, norm)
        down = self.down_output.read()[:t*8*p.k*4].view('f4').reshape(t*8, p.k)
        down_error = normalized_error(down, self.down_dot, self.down_abs)
        saved = self.rt.download(self.parts[1].outputs, t*8*4).view('f4').reshape(t, 8)
        if not np.allclose(saved, self.weights_host, rtol=2e-6, atol=1e-7):
            raise ValueError('production routing/retained weights differ')
        for b in (*self.buffers, *self.scratch):
            b.check_guard()
        return dict(finish_error=error, activation_error=activation_error, down_error=down_error)

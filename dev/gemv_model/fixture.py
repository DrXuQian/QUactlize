"""Chunked official-GGUF factor oracle and guarded resident active-expert ring."""

import ctypes as C
import hashlib
import math
import numpy as np

from dev.bf16_compute.fixture import raw_weight, planes, round_compute
from dev.gate_up_perf.bench import Buffer, WeightRing, physical
from dev.gate_up_perf.plan import ring_copies
from tools.kpack_warmup_fixture import activation_values
from quactlize.execution.native import Call
from reference import gguf_kpack as ref


def varied_weight(q, n, k, seed):
    """Use nonuniform signed group metadata, still exactly representable B16."""
    raw, gold = raw_weight(q, n, k, seed)
    if q != 8:
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        spec = ref.SPECS[q]
        if q == 14:
            # Vary all three axes; distinguish both halves of the 36-byte unit.
            v = ((np.arange(n)[:, None, None] + np.arange(k // 256)[None, :, None] * 3
                  + np.arange(16)[None, None, :] * 5 + seed) % 7 - 3).astype("i1")
            raw[..., spec.scale_offset:spec.scale_offset + 16] = v.view("u1")
        else:
            template = bytearray(spec.raw_bytes)
            for g in range(spec.groups):
                ref._metadata_put(template, 0, spec, g, 1 + (g * 3 + seed) % 3, 1 + (g + seed) % 3)
            raw[..., spec.scale_offset:spec.scale_offset + 12] = np.frombuffer(template, "u1")[spec.scale_offset:spec.scale_offset + 12]
        gold = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(n, k).astype("f4")
    if not np.isfinite(gold).all() or not np.any(gold):
        raise ValueError("degenerate independent weight oracle")
    for compute in ("f16", "bf16"):
        if not np.array_equal(gold, round_compute(gold, compute)):
            raise ValueError("fixture weight is not exact in " + compute)
    return raw, gold


def expert_payload(point, expert, category):
    p = point
    full, sums, absolute = {}, np.empty((4, p.physical_n), "f8"), np.empty((4, p.physical_n), "f8")
    raw_hash = hashlib.sha256()
    # The output head is not materialized as an N*K float64 CPU matrix.
    for start in range(0, p.n, 512):
        nc = min(512, p.n - start)
        raw, gold = varied_weight(p.q, nc, p.k, 181 + expert * 71 + start * 7)
        if p.paired:
            up, up_gold = varied_weight(p.q, nc, p.k, 947 + expert * 71 + start * 7)
            raw, gold = physical(raw, up, True), physical(gold, up_gold, True)
        raw_hash.update(raw.tobytes())
        packed = planes(raw, p.q)
        first = start * (2 if p.paired else 1)
        last = first + len(gold)
        for name in ("low", "high", "units"):
            value = packed[name]
            if name not in full:
                full[name] = (np.empty((value.shape[0], p.physical_n, *value.shape[2:]), value.dtype)
                              if value.size else np.empty(0, "u1"))
            if value.size:
                full[name][:, first:last] = value
        for g in range(4):
            selected = gold[:, category == g]
            sums[g, first:last] = selected.sum(axis=1, dtype="f8")
            absolute[g, first:last] = np.abs(selected).sum(axis=1, dtype="f8")
    return full, sums, absolute, raw_hash.hexdigest()


class Bench:
    def __init__(self, rt, point, l2, profile=False):
        self.rt, self.point = rt, point
        p = point
        self.topk = 8 if p.mode else 1
        self.max_rows = 8 * self.topk
        self.category = np.random.default_rng(727 + p.k).integers(0, 4, p.k)
        self.expert_sets = ((13 + np.arange(16) * 29) % 256).astype("i4") if p.mode else np.array([0], "i4")
        payloads, self.sums, self.absolute, hashes = {}, {}, {}, {}
        for i, e in enumerate(self.expert_sets):
            payloads[int(e)], self.sums[int(e)], self.absolute[int(e)], hashes[int(e)] = expert_payload(p, int(e), self.category)
            print(f"MODEL_GEMV_FIXTURE point={p.name} experts={i+1}/{len(self.expert_sets)}", flush=True)
        self.weight_bytes = sum(v.nbytes for v in next(iter(payloads.values())).values()) * self.topk
        self.copies = 1 if profile else ring_copies(l2, self.weight_bytes)
        self.ring = WeightRing(rt, payloads, p.experts, self.copies)
        self.a = Buffer(rt, 8 * p.channels * p.k * 4)
        self.ids = Buffer(rt, 8 * 11 * 4) if p.mode else None
        self.mapping = Buffer(rt, self.max_rows * 4) if p.paired and p.mode else None
        # M1 routed down is SIMT: no compact row-map in the real chain.
        # Shared gate/up has neither a row-map nor an upstream directory status.
        self.mapped_call = True  # Explicit non-null mapping numerical controls.
        self.status = Buffer(rt, 4) if p.paired and p.mode else None
        self.output = Buffer(rt, self.max_rows * p.n * 4)
        self.workspace = Buffer(rt, self.max_rows * p.physical_n * 8 * 4)
        self.record = dict(raw_sha256=hashes, category_sha256=hashlib.sha256(self.category.tobytes()).hexdigest(),
                           active_experts_per_call=self.topk, address_space_experts=p.experts,
                           physical_weight_bytes_per_call=self.weight_bytes, l2_bytes=l2, copies=self.copies,
                           rotating_active_bytes=self.weight_bytes * self.copies,
                           oracle="OFFICIAL_GGUF_NONUNIFORM_METADATA_FACTOR_DOT_FP32_ACCUM",
                           cache_scope="COLD_ACTIVE_WEIGHT_RING" if not profile else "ACU_CACHE_CONTROL_ALL")
        self.update(1, 0)

    def update(self, tokens, repeat, mapped=False, large=False):
        p = self.point
        self.tokens, self.rows = tokens, tokens * self.topk
        coeff = activation_values(np.arange(tokens * p.channels) + repeat * 137).astype("f4")
        if large:
            if not p.compute:
                raise ValueError("large-range control only for BF16")
            coeff[0, 0] = np.float32(243383.484375)
        rounded = round_compute(coeff, "bf16" if p.compute else "f16")
        self.rt.copy(self.a.ptr, np.ascontiguousarray(coeff[:, self.category]))
        if p.mode:
            base = self.expert_sets[8 if repeat % 2 else 0:16 if repeat % 2 else 8]
            ids = np.full((tokens, 11), -91, "i4")
            for t in range(tokens):
                ids[t, :8] = np.roll(base, t + repeat)
            self.ids_host = ids
            owners = ids[:, :8].reshape(-1)
            a_rows = np.arange(self.rows) // 8 * p.channels + np.arange(self.rows) % 8 % p.channels
            self.rt.copy(self.ids.ptr, ids)
        else:
            owners, a_rows = np.zeros(tokens, "i4"), np.arange(tokens)
        rowmap = np.arange(self.rows - 1, -1, -1) if mapped else np.arange(self.rows)
        if self.mapping:
            self.rt.copy(self.mapping.ptr, rowmap.astype("i4"))
        if self.status:
            self.rt.fill(self.status.ptr, 4, 0)
        dot = np.stack([rounded[a_rows[r]].astype("f8") @ self.sums[int(owners[r])] for r in rowmap])
        absolute = np.stack([np.abs(rounded[a_rows[r]].astype("f8")) @ self.absolute[int(owners[r])] for r in rowmap])
        if p.paired:
            gate_indices = (np.arange(p.n) // 4 * 8 + np.arange(p.n) % 4)
            gate, up = dot[:, gate_indices].astype("f4"), dot[:, gate_indices + 4].astype("f4")
            if p.compute:
                gate, up = round_compute(gate, "bf16"), round_compute(up, "bf16")
            with np.errstate(over="ignore"):
                self.gold = gate / (np.float32(1) + np.exp(-gate)) * up
            self.denom = None
        else:
            self.gold, self.denom = dot, absolute
        if not np.isfinite(self.gold).all() or not np.any(self.gold):
            raise ValueError("degenerate activation oracle")
        self.rt.sync()

    def call(self, copy=0):
        p = self.point
        return Call(version=1, size=C.sizeof(Call), qtype=p.q, n=p.n, k=p.k,
                    experts=p.experts, rows=self.rows, mode=p.mode, input_type=1,
                    channels=p.channels, topk=self.topk, a_row_stride=p.k,
                    a_token_stride=p.channels * p.k, ids_stride=11, out_row_stride=p.n,
                    a=self.a.ptr, ids=self.ids.ptr if self.ids else None,
                    output=self.output.ptr, workspace=self.workspace.ptr,
                    workspace_bytes=self.workspace.size, stream=self.rt.stream.value, **self.ring.at(copy))

    def poison(self):
        self.output.poison()
        self.workspace.poison()
        self.rt.sync()

    def result(self):
        raw = self.output.read()
        size = self.rows * self.point.n * 4
        if not np.all(raw[size:] == 0xA5):
            raise ValueError("unowned output rows written")
        for b in (self.a, self.workspace, self.ids, self.mapping, self.status):
            if b:
                b.check_guard()
        return raw[:size].view("f4").reshape(self.rows, self.point.n).copy()

    def check(self):
        got = self.result()
        delta = np.abs(got.astype("f8") - self.gold)
        error = float(np.max(delta) / max(1e-20, np.max(np.abs(self.gold)))) if self.point.paired else float(np.max(delta / np.maximum(self.denom, 1e-20)))
        if not np.isfinite(got).all() or not math.isfinite(error) or error >= 0.005:
            raise ValueError(f"independent GGUF{'/SwiGLU' if self.point.paired else ''} error={error}")
        return error

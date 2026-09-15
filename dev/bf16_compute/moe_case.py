"""Actual TC/SIMT producers composed through the production MoE helper ABI."""
import ctypes as C

import numpy as np

from dev.bf16_compute.cases import weights, source, selected_record
from dev.bf16_compute.plan import Parent
from dev.bf16_compute.fixture import bf16_bits, bf16_float, compare, round_compute, digest
from dev.bf16_compute.native import (
    Buffer, TensorCore, Resources, SimtCall, SimtCallV2, SimtConfig, Sizes, Projection,
    MoePlan, MoePlanV2, MixedPlanV2, IndexedIO, MoeFinish, arrangement,
    bind_simt_compute, load, function,
)
from quactlize.runtime.native import checked
from tools.run_kpack_grouped_decode_probe import Replay


def swiglu(gate, up):
    with np.errstate(over="ignore"):
        result = (np.asarray(gate, dtype="f4") / (np.float32(1) + np.exp(-np.asarray(gate, dtype="f4")))) * np.asarray(up, dtype="f4")
    return round_compute(result, "bf16")


class Endpoint:
    def __init__(self, root, package, sdk, r, w, point, role, source_ptr, ids, simt, output):
        self.simt, self.role, self.w, self.r = simt, role, w, r
        tokens, m = point["tokens"], point["tokens"] * 8
        self.tc = None
        if not simt:
            parent = Parent(w.q, "sf-grouped" if w.q == 8 else "fq-grouped")
            record = selected_record(package, dict(parent=parent))
            self.tc = TensorCore(root, record, sdk, r, w, m, tokens,
                                 split=1 if role == 1 else 2)
            row_ids = Buffer(r, m * 4)
            channels = 8 if role == 2 else 1
            io = IndexedIO(1, C.sizeof(IndexedIO), tokens, 8, channels, 0,
                8, w.k, channels * w.k, w.n, ids.ptr, source_ptr, output.ptr, row_ids.ptr)
            self.projection = self.tc.bind_indexed(io)
            return
        self.lib = load(root, package["simt"])
        query, self.simt_run = bind_simt_compute(self.lib)
        self.arr = arrangement(w.q)
        self.config = SimtConfig(1 if w.q == 8 else 3, 4, 4, 4, 1)
        ps = {name: r.upload(value) if value.size else None for name, value in w.planes.items()}
        sdk.synchronize(None)
        channels = 8 if role == 2 else 1
        self.call = SimtCall(version=1, size=C.sizeof(SimtCall), qtype=w.q, n=w.n, k=w.k,
            experts=w.experts, rows=m, mode=2, input_type=1, channels=channels, topk=8,
            a_row_stride=w.k, a_token_stride=channels * w.k, ids_stride=8, out_row_stride=w.n,
            a=source_ptr, low=ps["low"], high=ps["high"], units=ps["units"], ids=ids.ptr,
            output=output.ptr, stream=r.stream.value)
        request = SimtCallV2(self.call, 1)
        checked(query(C.byref(request), C.byref(self.config), C.byref(self.arr), C.byref(Sizes())), "MoE SIMT query")
        self.helper = load(root, package["moe"])
        scratch_query = function(self.helper, "quactlize_kpack_moe_simt_query_v1",
                                 [C.POINTER(SimtCall), C.POINTER(C.c_uint64)])
        scratch_bind = function(self.helper, "quactlize_kpack_moe_simt_bind_v1",
            [C.POINTER(SimtCall), C.c_int, C.c_void_p, C.c_uint64, C.POINTER(Projection)])
        size = C.c_uint64()
        checked(scratch_query(C.byref(self.call), C.byref(size)), "MoE SIMT scratch query")
        # The production bind requires 256-byte alignment. Keep red zones
        # around the aligned scratch, not an unaligned +16 test pointer.
        base = r.alloc(size.value + 512)
        r.fill(base, 0xa5, size.value + 512)
        self.scratch = (base, size.value)
        self.projection = Projection()
        checked(scratch_bind(C.byref(self.call), -1, base + 256, size.value, C.byref(self.projection)), "MoE SIMT descriptor bind")
        if role == 2:
            self.call.a = self.projection.a
        else:
            self.call.output = self.projection.output
        self.request = SimtCallV2(self.call, 1)

    def produce(self, plan):
        if self.simt:
            return self.simt_run(C.byref(self.request), C.byref(self.config), C.byref(self.arr))
        return self.tc.stage(self.tc.handle, C.byref(plan), 1, self.r.stream)

    def values(self):
        p = self.projection
        shape = (p.m, p.n)
        if self.simt:
            ptr = p.io.output if self.role == 2 else p.output
            raw = self.r.sdk.download(ptr, p.m * p.n * 4)
            return round_compute(np.frombuffer(raw, dtype="<f4").reshape(shape), "bf16")
        if p.splits == 1:
            raw = self.r.sdk.download(p.output, p.m * p.n * 2)
            sorted_values = bf16_float(np.frombuffer(raw, dtype="<u2").reshape(shape))
        else:
            raw = self.r.sdk.download(p.partials, p.splits * p.m * p.n * 4)
            partial = np.frombuffer(raw, dtype="<f4").reshape(p.splits, p.m, p.n)
            value = np.zeros(shape, dtype="f4")
            for part in partial:
                value += part
            sorted_values = round_compute(value, "bf16")
        row_ids = np.frombuffer(self.r.sdk.download(p.io.row_ids, p.m * 4), dtype="<i4")
        result = np.empty(shape, dtype="f4")
        result[row_ids] = sorted_values
        return result

    def close(self):
        if self.tc:
            self.tc.workspace.guard()
            self.tc.close()
        else:
            base, size = self.scratch
            if (self.r.sdk.download(base, 256) != b"\xa5" * 256 or
                    self.r.sdk.download(base + 256 + size, 256) != b"\xa5" * 256):
                raise ValueError("MoE SIMT scratch guard changed")


def run(root, package, sdk, point, repeats, samples):
    tokens, m, merged = point["tokens"], point["tokens"] * 8, point["merged"]
    r, endpoints, graph = Resources(sdk), [], None
    try:
        inp, ids = Buffer(r, tokens * 512 * 4), Buffer(r, m * 4)
        middle = Buffer(r, m * 512 * 4)
        slot_out = Buffer(r, m * 512 * 4)
        final, routing_weights = Buffer(r, tokens * 512 * 4), Buffer(r, m * 4)
        roles = (0, 2) if merged else (0, 1, 2)
        mask = 0 if point["kind"] == "tc" else (5 if merged else 7) if point["kind"] == "simt" else (1 if merged else 2)
        for role in roles:
            n = 1024 if role == 0 and merged else 512
            q = point.get("down_q", point["q"]) if role == 2 else point["q"]
            w = weights(q, n, 512, 256, 910 + role * 37)
            output = slot_out if role == 2 else Buffer(r, m * n * 4)
            endpoint = Endpoint(root, package, sdk, r, w, point, role,
                middle.ptr if role == 2 else inp.ptr, ids, bool(mask & (1 << role)), output)
            endpoints.append(endpoint)
        by_role = dict(zip(roles, endpoints))
        plan = MoePlan(version=1, size=C.sizeof(MoePlan), merged=int(merged),
            gate=by_role[0].projection, up=by_role[1].projection if not merged else Projection(),
            down=by_role[2].projection)
        typed = MoePlanV2(2, C.sizeof(MoePlanV2), plan, 1)
        mixed = MixedPlanV2(2, C.sizeof(MixedPlanV2), plan, mask, 1)
        helper = load(root, package["moe"])
        stage_fn = function(helper, "quactlize_kpack_moe_mixed_stage_v2",
                            [C.POINTER(MixedPlanV2), C.c_int, C.c_void_p])
        finish_fn = function(helper, "quactlize_kpack_moe_weighted_finish_v2",
                            [C.POINTER(MixedPlanV2), C.POINTER(MoeFinish), C.c_void_p])
        finish = MoeFinish(1, C.sizeof(MoeFinish), 8, 512, routing_weights.ptr, final.ptr)
        def stage(phase):
            if mask:
                return stage_fn(C.byref(mixed), phase, r.stream)
            gate = by_role[0].tc
            return gate.stage(gate.handle, C.byref(typed), phase, r.stream)
        def execute():
            checked(stage(0), "MoE prepare")
            for role in roles[:-1]:
                checked(by_role[role].produce(typed), "MoE gate/up producer")
            checked(stage(2), "MoE BF16 SwiGLU")
            checked(by_role[2].produce(typed), "MoE down producer")
            return finish_fn(C.byref(mixed), C.byref(finish), r.stream)
        graph = Replay(sdk, r.stream, execute, 1)
        proofs = []
        for repeat in range(repeats):
            act = source(tokens, 512, 720 + repeat)
            routing = ((np.arange(8)[None, :] * 17 + (np.arange(tokens)[:, None] * 11 if repeat % 2 == 0 else 0)
                       + 7 * repeat) % 256).astype("i4")
            if routing.shape[0] == 1 and tokens > 1:
                routing = np.repeat(routing, tokens, axis=0)
            router = np.tile(np.array([1, 2, 3, 4, 5, 6, 7, 8], "f4") / 36, (tokens, 1))
            inp.upload(act)
            ids.upload(routing)
            routing_weights.upload(router)
            owners = routing.ravel()
            order = np.argsort(owners, kind="stable")
            expanded = np.repeat(act, 8, axis=0)
            checked(stage(0), "inspect MoE prepare")
            sdk.synchronize(r.stream)
            for endpoint in endpoints:
                p = endpoint.projection
                got_rows = np.frombuffer(sdk.download(p.io.row_ids, m * 4), dtype="i4")
                if not np.array_equal(got_rows, order):
                    raise ValueError("MoE prepare expert-order row mapping differs")
                header = np.frombuffer(sdk.download(p.directory_header, 16), dtype="i4")
                if header[1] != 0:
                    raise ValueError("MoE prepare rejected valid routing")
                if endpoint.role != 2 and not endpoint.simt:
                    got_a = np.frombuffer(sdk.download(p.a, m * 512 * 2), dtype="<u2").reshape(m, 512)
                    if not np.array_equal(got_a, bf16_bits(expanded[order])):
                        raise ValueError("MoE prepare did not emit exact BF16 activation bits")
            projections, oracle = {}, {}
            for role in roles[:-1]:
                ep = by_role[role]
                checked(ep.produce(typed), "inspect gate/up producer")
                sdk.synchronize(r.stream)
                actual = ep.values()
                gold, denom = ep.w.dot(expanded, owners, "bf16", output_compute=True)
                projections[role] = compare(actual, gold, denom)
                oracle[role] = gold
            actual_gate = by_role[0].values()
            if merged:
                physical_middle = swiglu(actual_gate[:, :512], actual_gate[:, 512:])
                expected_middle = swiglu(oracle[0][:, :512], oracle[0][:, 512:])
            else:
                physical_middle = swiglu(actual_gate, by_role[1].values())
                expected_middle = swiglu(oracle[0], oracle[1])
            checked(stage(2), "inspect BF16 SwiGLU")
            sdk.synchronize(r.stream)
            down = by_role[2]
            if down.simt:
                got_middle = round_compute(np.frombuffer(sdk.download(down.projection.a, m * 512 * 4), "<f4").reshape(m, 512), "bf16")
            else:
                raw = np.frombuffer(sdk.download(down.projection.a, m * 512 * 2), "<u2").reshape(m, 512)
                got_middle = np.empty((m, 512), "f4")
                got_middle[order] = bf16_float(raw)
            ulps = np.abs(bf16_bits(got_middle).astype("i4") - bf16_bits(physical_middle).astype("i4"))
            ulps[got_middle == physical_middle] = 0
            # expf implementations can straddle one BF16 rounding boundary.
            # Exact conversion is checked independently below at exp(-482)=0.
            if np.any(ulps > 1):
                raise ValueError("actual SwiGLU/down boundary differs by more than one BF16 ULP")
            checked(down.produce(typed), "inspect down producer")
            sdk.synchronize(r.stream)
            actual_down = down.values()
            gold_down, denom = down.w.dot(physical_middle, owners, "bf16", output_compute=True)
            projections[2] = compare(actual_down, gold_down, denom)
            checked(finish_fn(C.byref(mixed), C.byref(finish), r.stream), "inspect weighted finish")
            sdk.synchronize(r.stream)
            finished = final.read("<f4", (tokens, 512))
            physical_finish = np.zeros_like(finished)
            for slot in range(8):
                value = actual_down.reshape(tokens, 8, 512)[:, slot] * router[:, slot, None]
                physical_finish = value if slot == 0 else physical_finish + value
            if not np.array_equal(finished, physical_finish):
                raise ValueError("weighted finish order or BF16 projection boundary differs")
            full_down, full_denom = down.w.dot(expected_middle, owners, "bf16", output_compute=True)
            full_gold = (full_down.reshape(tokens, 8, 512) * router[:, :, None]).sum(axis=1)
            full_bound = (full_denom.reshape(tokens, 8, 512) * router[:, :, None]).sum(axis=1)
            complete = compare(finished, full_gold, full_bound, tolerance=0.02)
            checked(graph(), "changed-input complete MoE graph")
            sdk.synchronize(r.stream)
            compare(final.read("<f4", (tokens, 512)), full_gold, full_bound, tolerance=0.02)
            proofs.append(dict(projections=projections, complete=complete,
                prepare_bf16_bits="PASS", swiglu_max_bf16_ulp=int(ulps.max()), weighted_order="PASS",
                input_sha256=digest(act), ids_sha256=digest(routing)))
        range_proof = None
        if point["q"] == 14 and tokens == 1 and not merged and point["kind"] in ("tc", "simt"):
            range_plan = MoePlan.from_buffer_copy(plan)
            gate_value, up_value = np.float32(482.842712), np.float32(504.063690)
            held = []
            for role, value in ((0, gate_value), (1, up_value)):
                p = range_plan.gate if role == 0 else range_plan.up
                p.splits = 1
                host = np.full((m, 512), value, dtype="f4")
                buffer = Buffer(r, host.size * (4 if mask else 2))
                buffer.upload(host if mask else bf16_bits(host))
                p.output = buffer.ptr
                held.append(buffer)
            range_typed = MoePlanV2(2, C.sizeof(MoePlanV2), range_plan, 1)
            range_mixed = MixedPlanV2(2, C.sizeof(MixedPlanV2), range_plan, mask, 1)
            if mask:
                checked(stage_fn(C.byref(range_mixed), 2, r.stream), "MoE wide finite SwiGLU")
                sdk.synchronize(r.stream)
                raw = np.frombuffer(sdk.download(range_plan.down.a, m * 512 * 4), "<f4")
                got = bf16_bits(raw)
            else:
                gate = by_role[0].tc
                checked(gate.stage(gate.handle, C.byref(range_typed), 2, r.stream), "MoE wide finite SwiGLU")
                sdk.synchronize(r.stream)
                got = np.frombuffer(sdk.download(range_plan.down.a, m * 512 * 2), "<u2")
            expected = bf16_bits(round_compute(gate_value, "bf16") * round_compute(up_value, "bf16")).item()
            if np.any(got != expected) or not np.isfinite(bf16_float(got)).all() or bf16_float(np.array([expected], "u2"))[0] <= 65504:
                raise ValueError("MoE SwiGLU/down failed the finite BF16 range seam")
            range_proof = dict(status="PASS", expected_bits=f"0x{expected:04x}", cells=int(got.size), clipped=False)
        checked(graph(), "excluded MoE warmup")
        sdk.synchronize(r.stream)
        timing = r.samples(graph, samples) if samples else []
        return dict(status="PASS", simt_mask=mask, proofs=proofs, range_proof=range_proof, samples_us=timing,
                    fixtures=[ep.w.record() for ep in endpoints])
    finally:
        sdk.synchronize(r.stream)
        if graph:
            graph.close()
        for endpoint in reversed(endpoints):
            endpoint.close()
        r.close()

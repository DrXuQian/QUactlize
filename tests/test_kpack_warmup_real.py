import json
from dataclasses import asdict
import struct
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from reference import gguf_kpack as ref
from quactlize.runtime.tuning import Request, Tactic, digest, UnsupportedTactic
from tools.kpack_warmup_fixture import prepare_expert, Weights, activation_values
from tools.kpack_warmup_real_plan import make_plan, census
from tools.run_kpack_warmup_real import (
    analyze_confirmation,
    save,
    completed,
    case_path,
    run_case,
)


def raw_expert(q, n, k, e):
    spec = ref.SPECS[q]
    rng = np.random.default_rng(np.random.SeedSequence([95171, q, n, k, e]))
    raw = rng.integers(0, 256, size=(n * (k // 256), spec.raw_bytes), dtype=np.uint8)
    for offset in (spec.d_offset, spec.dmin_offset):
        if offset >= 0:
            h = (rng.random(raw.shape[0]) * 0.025 + 0.005).astype("<f2")
            raw[:, offset : offset + 2] = h.view(np.uint8).reshape(-1, 2)
    return raw


@pytest.mark.parametrize("q", range(10, 15))
def test_batched_fixture_planes_equal_scalar_reference_across_n_and_units(q):
    n, k = 512, 1024
    raw = raw_expert(q, n, k, 1)
    actual = prepare_expert(raw, q, n, k)
    expected = ref.prepare_dense(torch.from_numpy(raw), n, k, q)
    assert torch.equal(ref.recover_raw_blocks(expected), torch.from_numpy(raw))
    for name in ("low", "high", "units"):
        assert actual[name].tobytes() == getattr(expected, name).numpy().tobytes(), (
            q,
            name,
        )
    assert np.isfinite(actual["scale"]).all() and np.isfinite(actual["zero"]).all()
    # Include signed Q3/Q6 scale codes, half rounding and both N chunks.
    spec = ref.SPECS[q]
    blob = raw.tobytes()
    for col in (0, 255, 256, 511):
        for sb in range(k // 256):
            base = (col * (k // 256) + sb) * spec.raw_bytes
            d = struct.unpack_from("<e", blob, base + spec.d_offset)[0]
            dmin = (
                struct.unpack_from("<e", blob, base + spec.dmin_offset)[0]
                if spec.has_min
                else 0
            )
            for g in range(spec.groups):
                sc, mn = ref._metadata_codes(blob, base, spec, g)
                if q == 11:
                    sc -= 32
                if q == 14 and sc >= 128:
                    sc -= 256
                s = np.float16(d * sc)
                z = np.float16(-np.float16(dmin * mn))
                zmul = {10: 0, 11: -4, 12: 8, 13: 8, 14: -24}[q]
                z = np.float16(np.float16(zmul * s) + z) if zmul else z
                for name, want in (("scale", s), ("zero", z)):
                    assert actual[name][sb * spec.groups + g, col].view(
                        np.uint16
                    ) == want.view(np.uint16)


@pytest.mark.parametrize("q", range(10, 15))
def test_factorized_oracle_equals_independent_full_gemm_and_detects_zero_low(q):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    n, k, e = 256, 512, 2
    w = Weights(q, n, k, e)
    req = Request("fq-grouped", q, n, k, 26, (9, 17))
    a, golden, denom = w.activation(req)
    offset = 0
    errors = []
    for expert, rows in enumerate(req.rows):
        raw = raw_expert(q, n, k, expert)
        official = (
            dequantize(raw.reshape(-1), GGMLQuantizationType(q))
            .astype(np.float64)
            .reshape(n, k)
        )
        act = a[offset : offset + rows].astype(np.float64)
        expected = act @ official.T
        expected_denominator = np.abs(act) @ np.abs(official).T
        np.testing.assert_allclose(
            golden[offset : offset + rows], expected, rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            denom[offset : offset + rows], expected_denominator, rtol=0, atol=1e-10
        )
        # Independent raw-format fault; do not invert our own placed map here.
        begin, length = {
            10: (16, 64),
            11: (32, 64),
            12: (16, 128),
            13: (48, 128),
            14: (0, 128),
        }[q]
        raw[:, begin : begin + length] = 0
        bad = (
            dequantize(raw.reshape(-1), GGMLQuantizationType(q))
            .astype(np.float64)
            .reshape(n, k)
        )
        errors.append(np.max(np.abs(act @ bad.T - expected) / expected_denominator))
        offset += rows
    assert max(errors) > 5e-3


def test_row_tags_are_fp16_exact_nonzero_and_distinct():
    values = activation_values(np.arange(32768))
    assert np.array_equal(values, values.astype(np.float16).astype(np.float64))
    assert np.all(np.any(values != 0, axis=1))
    assert len(np.unique(values, axis=0)) == len(values)


def test_real_plan_keeps_exact_incumbents_and_bounded_denominator():
    plan = make_plan()
    assert plan["digest"] == digest({k: v for k, v in plan.items() if k != "digest"})
    stats = census(plan)
    assert stats == dict(
        contexts=90,
        parents=173,
        max_candidates=15,
        candidate_contexts=868,
        exact_incumbents=85,
        transfer_controls=5,
        route_contexts={
            "fq-dense": 30,
            "sf-dense": 20,
            "fq-grouped": 20,
            "sf-grouped": 20,
        },
    )
    for c in plan["cases"]:
        assert json.loads(json.dumps(c)) == c
        assert c["candidates"][0] == c["incumbent"]
        assert len({t["parent"] for t in c["candidates"]}) <= 5
        if c["incumbent_evidence"] == "TRANSFER_CONTROL":
            assert c["request"]["m"] == 3072 and c["incumbent_source_key"][5] == 2048


def test_new_m_and_router_controls_reuse_the_intended_buckets():
    plan = make_plan([12])
    cases = plan["cases"]

    def req(c):
        return Request(**(c["request"] | {"rows": tuple(c["request"]["rows"])}))

    anchor = next(
        c for c in cases if c["role"] == "PREFILL" and c["request"]["n"] == 1024
    )
    control = next(c for c in cases if c["role"] == "NEW_M_BUCKET_CONTROL")
    assert req(anchor).bucket == req(control).bucket
    for route in ("fq-grouped", "sf-grouped"):
        a = next(
            c
            for c in cases
            if c["role"] == "ROUTER_BOUNDARY" and c["request"]["route"] == route
        )
        b = next(
            c
            for c in cases
            if c["role"] == "CHANGED_ROUTER" and c["request"]["route"] == route
        )
        assert req(a).bucket == req(b).bucket and req(a).exact_key != req(b).exact_key


def test_confirmation_uses_same_run_best_and_distinguishes_noise():
    a, b = Tactic("a", "TC_S1"), Tactic("b", "TC_S1")
    clean = {a.key: [10, 10.1], b.key: [9.8, 9.9]}
    assert analyze_confirmation(clean, a, b)["verdict"] == "WITHIN_BOUNDED_POOL_5PCT"
    slow = {a.key: [12, 12.1], b.key: [9.8, 9.9]}
    assert analyze_confirmation(slow, a, b)["verdict"] == "BOUNDED_POOL_GAP"
    noisy = {a.key: [12, 10.1], b.key: [9.8, 9.9]}
    assert analyze_confirmation(noisy, a, b)["verdict"] == "TIMING_NOISE_REVIEW"
    for invalid in ({a.key: [10]}, {a.key: [0, 0]}, {a.key: [float("nan"), 10]}):
        with pytest.raises(ValueError):
            analyze_confirmation(invalid, a, a)
    with pytest.raises(ValueError):
        analyze_confirmation(clean, Tactic("missing", "TC_S1"), a)


def test_resume_never_accepts_changed_authority_or_incomplete_write(tmp_path):
    a = Tactic("parent", "TC_S1")
    case = {"id": "case", "request": {"m": 1, "rows": []}, "incumbent": asdict(a)}
    assert completed(tmp_path, case, "authority") is None
    timing = {a.key: [10, 10.1]}
    result = dict(
        case=case,
        authority="authority",
        status="PASS",
        cache_replay=True,
        numeric=dict(
            checks=3, max_error=1e-4, zero_low_rejected=True, planted_error=0.1
        ),
        warmup=dict(status="MEASURED_EXACT", tactic=asdict(a)),
        confirmation=dict(samples_us=timing, **analyze_confirmation(timing, a, a)),
    )
    result["digest"] = digest(result)
    save(case_path(tmp_path, case), result)
    assert completed(tmp_path, case, "authority") == result
    with pytest.raises(ValueError):
        completed(tmp_path, case, "other")
    result["numeric"]["zero_low_rejected"] = False
    result["digest"] = digest({k: v for k, v in result.items() if k != "digest"})
    save(case_path(tmp_path, case), result)
    with pytest.raises(ValueError, match="incomplete numeric"):
        completed(tmp_path, case, "authority")
    result["numeric"]["zero_low_rejected"] = True
    with pytest.raises(ValueError):
        save(case_path(tmp_path, case), result)
        completed(tmp_path, case, "authority")


def test_real_case_orchestration_roundtrip_with_host_backend(tmp_path):
    # Exercise the actual driver, not only its isolated digest/stat helpers.
    a, b, rejected = (Tactic(n, "TC_S1") for n in ("a", "b", "rejected"))
    r = Request("fq-dense", 12, 256, 512, 4)
    case = dict(
        id=r.exact_key,
        request=asdict(r) | {"rows": []},
        incumbent=asdict(a),
        candidates=[asdict(t) for t in (a, b, rejected)],
    )

    class Memory:
        def __init__(self):
            self.cells, self.next = {}, 1

        def upload(self, data):
            p = self.allocate(len(data))
            self.cells[p] = data
            return p

        def allocate(self, size):
            p, self.next = self.next, self.next + 1
            self.cells[p] = bytes(size)
            return p

        def download(self, p, size):
            assert len(self.cells[p]) == size
            return self.cells[p]

        def free(self, p):
            del self.cells[p]

    class Backend:
        identity = dict(
            device="host-test",
            compute_units=1,
            sdk="test",
            kernel="test",
            inventory="test",
        )

        def __init__(self):
            self.sdk, self.buffers = Memory(), {}

        def is_capturing(self):
            return False

        def admissible(self, request, tactic):
            return tactic != rejected

        def arguments(self, request, tactic):
            self.active_tactic = tactic
            if tactic == rejected:
                raise UnsupportedTactic()
            return (
                None,
                None,
                SimpleNamespace(grid=1),
                SimpleNamespace(workspace_bytes=0, shared_bytes=16, occupancy=1),
                None,
            )

        def prepare(self, request, tactic):
            self.arguments(request, tactic)
            return tactic

        def run(self, handle):
            value = 0 if self.buffers["low"] == 999 else 1
            self.sdk.cells[self.buffers["output"]] = np.full(
                (r.m, r.n), value, dtype="<f2"
            ).tobytes()

        def check(self, handle):
            self.run(handle)
            if not self.correctness(self):
                raise RuntimeError("candidate correctness check failed")

        def measure(self, handle, repeats):
            return 10 if handle == a else 8

        def synchronize(self):
            pass

        def close(self, handle):
            pass

    backend = Backend()
    weights = SimpleNamespace(
        activation=lambda request: (
            np.ones((r.m, r.k), dtype="<f2"),
            np.ones((r.m, r.n)),
            np.ones((r.m, r.n)),
        )
    )
    pointers = dict(low=998, high=0, units=997, zero_low=999)
    options = dict(
        max_candidates=15,
        budget_ms=5000,
        warmups=2,
        repeats=5,
        samples=3,
        improve_pct=5,
    )
    result = run_case(case, backend, weights, pointers, tmp_path, options)
    assert result["warmup"]["tactic"] == asdict(b)
    assert result["confirmation"]["rejected"] == [rejected.key]
    assert len(result["confirmation"]["samples_us"]) == 2
    assert result["numeric"]["checks"] == 6 and result["numeric"]["planted_error"] == 1
    assert not backend.sdk.cells and not backend.buffers
    receipt = dict(case=case, authority="test", **result)
    receipt["digest"] = digest(receipt)
    save(case_path(tmp_path, case), receipt)
    assert completed(tmp_path, case, "test") == receipt

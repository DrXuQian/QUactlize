"""Prove the large-input oracle's sensitivity without assuming PPU's order."""

import numpy as np
import pytest
from types import SimpleNamespace

from dev.bf16_compute.fixture import raw_weight, round_compute, bf16_bits
from tools.diagnose_gate_up_rounding import activate, normalized_error, serial_dot
from tools.diagnose_gate_up_rounding import observe
from tests.test_gate_up_paired import HostMemory
from tools.gate_up_rounding_oracle import certify_output, projection_bins


@pytest.fixture(scope="module")
def weights():
    return {
        q: [raw_weight(q, 256, 2048, 17 + side * 419)[1] for side in (0, 1)]
        for q in (11, 14)
    }


def test_mixed_large_legal_f32_order_crosses_bf16_midpoint(weights):
    a = np.random.default_rng(135).integers(-7, 8, (8, 2056)).astype("f4") / 32
    a[:, 0] = np.float32(243383.484)
    a = round_compute(a[:, :2048], "bf16")
    wide = [(a.astype("f8") @ w.astype("f8").T).astype("f4") for w in weights[11]]
    serial = [serial_dot(a, w) for w in weights[11]]
    at = (5, 88)
    assert float(wide[0][at]) < 1428 < float(serial[0][at])
    assert int(bf16_bits(wide[0][at])) == 0x44B2
    assert int(bf16_bits(serial[0][at])) == 0x44B3
    error = normalized_error(activate(*serial, 1), activate(*wide, 1))
    assert error == pytest.approx(0.00558659217877095, abs=1e-12)
    assert error > 0.005
    # Without projection rounding the small F32 accumulation difference
    # remains small; there is no F16 overflow in either CPU computation.
    assert normalized_error(activate(*serial, 0), activate(*wide, 0)) < 1e-5


@pytest.mark.parametrize("q", [11, 14])
def test_sparse_large_control_is_exact_across_f32_orders(weights, q):
    a = np.zeros((1, 2048), dtype="f4")
    a[:, 0] = np.float32(243383.484)
    a = round_compute(a, "bf16")
    assert a[0, 0] > 65504
    wide = [(a.astype("f8") @ w.astype("f8").T).astype("f4") for w in weights[q]]
    serial = [serial_dot(a, w) for w in weights[q]]
    for got, want in zip(serial, wide):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(activate(*serial, 1), activate(*wide, 1))


def test_diagnostic_recovers_paired_split_projections(tmp_path):
    rt = HostMemory()
    m, n, split = 2, 8, 8
    gate = np.arange(m * split * n, dtype="f4").reshape(m, split, n) / 1024
    up = -gate - np.float32(0.125)
    physical = np.stack(
        (gate.reshape(m, split, n // 4, 4), up.reshape(m, split, n // 4, 4)), axis=3
    )
    g, u = np.zeros((m, n), dtype="f4"), np.zeros((m, n), dtype="f4")
    for s in range(split):
        g += gate[:, s]
        u += up[:, s]
    want = activate(g, u, 0)
    storage = np.full(m * (n + 8) * 4 + 32, 0xA5, dtype="u1")
    storage[16:-16].view("f4").reshape(m, n + 8)[:, :n] = want
    work = np.full(physical.nbytes + 32, 0xA5, dtype="u1")
    work[16:-16] = physical.reshape(-1).view("u1")
    bench = SimpleNamespace(
        rt=rt,
        w=SimpleNamespace(n=n),
        rows=m,
        output=rt.upload(storage),
        output_bytes=storage.nbytes,
        work=rt.upload(work),
        gold=want,
        dot=[g, u],
        host_a=np.zeros((m, 1), dtype="f4"),
        rounding=0,
        check=lambda split: None,
    )
    path = tmp_path / "projections.npz"
    record = observe(bench, split, path)
    assert record["error"] == 0
    assert record["projections"] == dict(
        gate_error=0, up_error=0, activation_of_device_projections_error=0
    )
    with np.load(path) as arrays:
        np.testing.assert_array_equal(arrays["gate_device"], g)
        np.testing.assert_array_equal(arrays["up_device"], u)


def large_case(weights, q):
    m = 8 if q == 11 else 1
    a = np.random.default_rng(135).integers(-7, 8, (m, 2056)).astype("f4") / 32
    a[:, 0] = np.float32(243383.484)
    a = round_compute(a[:, :2048], "bf16")
    wide = [(a.astype("f8") @ w.astype("f8").T).astype("f4") for w in weights[q]]
    gold = activate(*wide, 1)
    # Exact returned S1 coordinate. Do not emulate PPU's unknown sum order.
    at, value = ((1, 88), -5422592.0) if q == 11 else ((0, 184), -325173248.0)
    got = gold.copy()
    got[at] = value
    return a, gold, got, at


def certificate(weights, q, a, gold, got, split=1):
    return certify_output(
        got,
        gold,
        a,
        [w[None] for w in weights[q]],
        np.zeros(len(a), dtype=int),
        np.arange(len(a)),
        split,
    )


@pytest.mark.parametrize("q", [11, 14])
def test_returned_s1_value_is_one_exact_discrete_bf16_product(weights, q):
    a, gold, got, at = large_case(weights, q)
    report = certificate(weights, q, a, gold, got)
    assert report["status"] == "CERTIFIED_DISCRETE_ROUNDING"
    assert report["original_verdict"] == "FAIL"
    assert report["original_error"] == pytest.approx(0.00558659217877095)
    assert report["original_threshold"] == 0.005
    assert len(report["exceptions"]) == 1
    exception = report["exceptions"][0]
    assert (exception["row"], exception["column"]) == at
    assert len(set(exception["permitted_outputs"])) == 2


def test_legal_cpu_f32_order_not_forced_to_wide_bf16_rounding(weights):
    a, gold, _, _ = large_case(weights, 11)
    got = activate(*[serial_dot(a, w) for w in weights[11]], 1)
    assert certificate(weights, 11, a, gold, got)["original_error"] > 0.005


@pytest.mark.parametrize("plant", ["zero", "sign", "column", "one_ulp", "inf", "nan"])
@pytest.mark.parametrize("q", [11, 14])
def test_discrete_certificate_rejects_wrong_outputs(weights, q, plant):
    a, gold, got, at = large_case(weights, q)
    if plant == "zero":
        got[at] = 0
    elif plant == "sign":
        got[at] *= -1
    elif plant == "column":
        got[at] = got[at[0], at[1] + 1]
    elif plant == "one_ulp":
        # Even a single F32 ULP off the allowed BF16 product is rejected.
        got[at] = np.nextafter(got[at], np.float32(np.inf))
    else:
        got[at] = float(plant)
    with pytest.raises(ValueError, match="rounding certificate"):
        certificate(weights, q, a, gold, got)


def test_certificate_does_not_admit_non_saturated_gate(weights):
    a, gold, got, _ = large_case(weights, 11)
    a /= 1024
    with pytest.raises(ValueError, match="not positively saturated"):
        certificate(weights, 11, a, gold / 1024**2, got / 1024**2)


def test_projection_bound_rejects_unrepresentable_weights_and_ambiguous_cancellation():
    with pytest.raises(ValueError, match="exact BF16"):
        projection_bins(np.ones(2, "f4"), np.array([1, 0.1], "f4"), 1)
    with pytest.raises(ValueError, match="too many BF16 bins"):
        projection_bins(np.ones(2, "f4"), np.array([1, -1], "f4"), 1)


def test_projection_bins_cover_negative_bf16_midpoint(weights):
    a, _, _, at = large_case(weights, 14)
    result = projection_bins(a[at[0]], weights[14][1][at[1]], 1)
    assert result["bf16"] == [-22912.0, -22784.0]
    assert (
        result["dot"] - result["f32_bound"]
        < -22848
        < result["dot"] + result["f32_bound"]
    )


def test_grouped_and_indexed_certificate_use_expert_and_input_row(weights):
    a, gold, got, at = large_case(weights, 14)
    a = np.concatenate((a / 2, a))
    original = [np.stack((w / 2, w)) for w in weights[14]]
    kwargs = dict(got=got, gold=gold, activation=a, weights=original, split=1)
    assert (
        certify_output(**kwargs, owners=[1], a_rows=[1])["exceptions"][0]["expert"] == 1
    )
    for owners, a_rows in (([0], [1]), ([1], [0])):
        with pytest.raises(ValueError, match="not an exact permitted"):
            certify_output(**kwargs, owners=owners, a_rows=a_rows)


def test_normal_checker_and_guards_are_not_relaxed(weights):
    from tools.run_kpack_gate_up import Bench

    a, gold, got, _ = large_case(weights, 14)
    bench = Bench.__new__(Bench)
    bench.rt = HostMemory()
    bench.rows, n = gold.shape
    bench.output_bytes, bench.work_bytes = (n + 8) * 4 + 32, n * 2 * 8 * 4
    bench.output = bench.rt.allocate(bench.output_bytes)
    bench.work = bench.rt.allocate(bench.work_bytes + 32)
    bench.w = SimpleNamespace(
        q=14, n=n, k=2048, gate=weights[14][0][None], up=weights[14][1][None]
    )
    bench.gold, bench.host_a = gold, a
    bench.compute = bench.storage = bench.output_type = bench.rounding = 1
    bench.large, bench.rounding_certificates = True, []
    bench.mode, bench.m, bench.channels, bench.profile = 0, 1, 1, "dense"
    bench.owners, bench.a_rows = [0], [0]
    bench.poison()
    host = bench.rt.download(bench.output, bench.output_bytes)
    host[16:-16].view("f4").reshape(1, n + 8)[:, :n] = got
    bench.rt.copy(bench.output, host)
    with pytest.raises(ValueError, match="independent GGUF"):
        bench.check(1)  # Original diagnostic still fails with no opt-in.
    assert bench.check(1, certify_rounding=True) > 0.005
    assert len(bench.rounding_certificates) == 1
    bench.large = False
    with pytest.raises(ValueError, match="independent GGUF"):
        bench.check(1, certify_rounding=True)
    bench.large = True
    bench.rt.fill(bench.output, 1, 0)
    with pytest.raises(ValueError, match="output guard"):
        bench.check(1, certify_rounding=True)


@pytest.mark.parametrize(
    "value,allowed",
    [
        ("", (11, 14)),
        ("11,11", (11, 14)),
        ("12", (11, 14)),
        ("tc,", ("simt", "tc")),
        ("unknown", ("simt", "tc")),
    ],
)
def test_gate_subset_rejects_empty_duplicate_or_unsupported(value, allowed):
    from argparse import ArgumentTypeError
    from tools.run_kpack_gate_up import selected_csv

    with pytest.raises(ArgumentTypeError):
        selected_csv(value, allowed)


def test_gate_subset_selects_exact_requested_parts():
    from tools.run_kpack_gate_up import selected_csv, FORMATS

    assert selected_csv("11,14", FORMATS) == [11, 14]
    assert selected_csv("tc", ("simt", "tc")) == ["tc"]

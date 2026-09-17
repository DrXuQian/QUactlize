"""Prove the large-input oracle's sensitivity without assuming PPU's order."""

import numpy as np
import pytest
from types import SimpleNamespace

from dev.bf16_compute.fixture import raw_weight, round_compute, bf16_bits
from tools.diagnose_gate_up_rounding import activate, normalized_error, serial_dot
from tools.diagnose_gate_up_rounding import observe
from tests.test_gate_up_paired import HostMemory


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

"""Host contract checks for bounded full-call timing; not a GPU timing claim."""

import copy
import ctypes as C
import json
from pathlib import Path
import statistics

import numpy as np
import pytest

from dev.gate_up_perf.plan import points, incumbent, parent, candidates, ring_copies
from dev.gate_up_perf.bench import routing, physical, WeightRing, Buffer, Arm
from dev.gate_up_perf.run import complete
from tests.test_gate_up_paired import HostMemory

ROOT = Path(__file__).resolve().parents[1]


def policies():
    return dict(
        matched=json.loads(
            (ROOT / "policies/kpack_smallm_matched_v1.json").read_text()
        ),
        vector=json.loads((ROOT / "policies/kpack_q8_vector_v1.json").read_text()),
    )


def test_inventory_keeps_exact_incumbents_including_tc_and_hoist():
    cohort = [p | dict(incumbent=incumbent(p, **policies())) for p in points()]
    assert len(cohort) == 16 and len({p["key"] for p in cohort}) == 16
    tc = [p for p in cohort if p["incumbent"]["config"]["kind"] == "tc"]
    assert {(p["q"], p["tokens"]) for p in tc} == {(8, 7), (8, 8), (12, 8)}
    assert len({parent(p["incumbent"])["symbol"] for p in tc}) == 2
    first = cohort[0]["incumbent"]
    assert first["vector_override"] and first["config"] == dict(
        kind="simt", variant=5, columns=4, warps=8, values=4, split=1
    )
    assert all(p["rounding"] == int(p["q"] == 12) for p in cohort)
    assert len(candidates()) == 16


def test_missing_duplicate_or_changed_policy_is_not_generic_fallback():
    original = policies()
    key = [8, 0, 512, 2048, 1, 1, 1, 1, 0]
    for plant in ("missing", "duplicate", "changed"):
        data = copy.deepcopy(original)
        row = next(r for r in data["matched"]["exact"] if r["key"] == key)
        if plant == "missing":
            data["matched"]["exact"].remove(row)
        elif plant == "duplicate":
            data["matched"]["exact"].append(row)
        else:
            row["config"]["warps"] = 2
        with pytest.raises(ValueError):
            incumbent(points()[0], **data)


def test_routing_distinct_per_token_and_correct_input_ownership():
    for p in points():
        ids, owners, rows = routing(p)
        if p["mode"]:
            assert ids.shape == (p["tokens"], 8) and all(
                len(set(row)) == 8 for row in ids
            )
            assert ids.min() >= 0 and ids.max() < 256
            np.testing.assert_array_equal(rows, np.repeat(np.arange(p["tokens"]), 8))
            np.testing.assert_array_equal(owners, ids.reshape(-1))


def test_pairing_moves_whole_rows_and_differs_from_concat():
    g = np.arange(16 * 3, dtype="u1").reshape(16, 3)
    u = g + 100
    paired = physical(g, u, True)
    concat = physical(g, u, False)
    np.testing.assert_array_equal(paired[:8], np.concatenate((g[:4], u[:4])))
    assert not np.array_equal(paired, concat)
    restored = paired.reshape(4, 2, 4, 3)
    np.testing.assert_array_equal(restored[:, 0].reshape(16, 3), g)
    np.testing.assert_array_equal(restored[:, 1].reshape(16, 3), u)


class Memory(HostMemory):
    def MemcpyAsync(self, to, source, size, kind, stream):
        assert kind == 3
        C.memmove(to, source, size)
        return 0


def test_cold_ring_real_sparse_expert_slices_and_guards():
    rt = Memory()
    data = {
        e: {
            "low": np.full(8, e + 9, "u1"),
            "high": np.empty(0, "u1"),
            "units": np.full(2, e + 11, "u1"),
        }
        for e in (1, 3)
    }
    ring = WeightRing(rt, data, 5, 3)
    assert all(b.ptr % 256 == 0 for b in ring.buffers.values())
    for copy in range(3):
        at = ring.at(copy)
        assert at["high"] is None
        for e in range(5):
            np.testing.assert_array_equal(
                rt.download(at["low"] + 8 * e, 8),
                data[e]["low"] if e in data else np.zeros(8, "u1"),
            )
    ring.buffers["low"].check_guard()
    assert ring_copies(64 * 1024**2, 9437184) * 9437184 >= 2.25 * 64 * 1024**2
    for l2, size in ((0, 1), (1, 0), (-1, 2)):
        with pytest.raises(ValueError):
            ring_copies(l2, size)


def test_complete_call_stops_on_failure_and_includes_postop():
    calls = []

    def op(i, rc=0):
        def run():
            calls.append(i)
            return rc

        return run

    assert Arm.combine([op("gate"), op("up"), op("swiglu")])() == 0
    assert calls == ["gate", "up", "swiglu"]
    calls.clear()
    assert Arm.combine([op("producer", 4), op("postop")])() == 4
    assert calls == ["producer"]


def receipt():
    ident = dict(rounds=4, samples=11)
    records = [
        dict(
            arm=dict(key=k),
            samples_us=[2.0, 3.0, 4.0],
            median_us=3.0,
            proof=dict(errors=[0.0, 0.0], zero_a_negative="RED"),
        )
        for k in ["incumbent", *candidates()]
    ]
    win = {
        b: next(k for k in candidates() if k.startswith(b + "-"))
        for b in ("simt", "tc")
    }
    confirmation = [
        dict(round=i, key=k, samples_us=[3.0] * 11)
        for i in range(4)
        for k in ["incumbent", *win.values()]
    ]
    return dict(
        status="PASS",
        point=points()[0],
        identity=ident,
        screen=records,
        winners=win,
        confirmation=confirmation,
        fixture=dict(
            copies=3, weight_bytes=1024, cold_weight_bytes=3072, l2_bytes=1024
        ),
        metrics={
            k: dict(
                median_us=3.0,
                delta_pct=0.0,
                modeled_weight_mbu_pct=100 * 1024 / (3.0 * 2700 * 1000),
            )
            for k in ["incumbent", *win.values()]
        },
    )


@pytest.mark.parametrize(
    "plant",
    [
        "duplicate",
        "nonfinite",
        "wrong_median",
        "missing_baseline",
        "missing_tc",
        "short_ring",
        "no_negative",
        "missing_round",
    ],
)
def test_failed_or_incomplete_results_cannot_resume(plant):
    v = receipt()
    assert complete(v, v["point"], v["identity"])
    if plant == "duplicate":
        v["screen"][-1] = v["screen"][0]
    elif plant == "nonfinite":
        v["confirmation"][0]["samples_us"][0] = float("nan")
    elif plant == "wrong_median":
        v["screen"][0]["median_us"] = 2.0
    elif plant == "missing_baseline":
        v["screen"].pop(0)
    elif plant == "missing_tc":
        v["winners"]["tc"] = "none"
    elif plant == "short_ring":
        v["fixture"]["copies"] = 1
    elif plant == "no_negative":
        v["screen"][0]["proof"]["zero_a_negative"] = "GREEN"
    else:
        v["confirmation"].pop()
    assert not complete(v, v["point"], v["identity"])

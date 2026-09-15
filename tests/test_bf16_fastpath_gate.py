"""Independent selected-Q4 typed gate contracts; no accelerator required."""
import copy
import ctypes as C
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dev.bf16_fastpath.gate_plan import Library, plan, ROOT
from dev.bf16_fastpath.gate_fixture import Weights, OUTLIER, storage
from dev.bf16_fastpath.gate import rejected, summarize
from dev.bf16_compute.fixture import raw_weight, round_compute, bf16_float


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    out = tmp_path_factory.mktemp("bf16-q4-gate-host") / "selector.so"
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", f"-I{ROOT}",
        f"-I{ROOT}/quactlize/include", f"-I{ROOT}/third_party/actlize/include",
        str(ROOT / "quactlize/execution/q4_decode.cpp"), str(ROOT / "tests/q4_decode_policy_host.cpp"),
        "-o", str(out)], check=True)
    return Library(out)


def test_denominator_and_actual_selector(library):
    p = plan()
    assert p["denominator"] == dict(requests=192, selected_requests=129, declined_requests=63,
        bf16_cells=258, f16_controls=129, overflow_negatives=129, compiled_recipes=40, covered_recipes=40)
    for point in p["cases"]:
        for kind in (1, 2):
            selected = library.select(point, kind)
            assert bool(selected) == (point["expected"] == "SELECTED")
            if selected:
                call, config, _ = selected
                assert library.run2(C.byref(call), C.byref(config), C.byref(library.arr)) == 124
                if kind == 1:
                    call.compute_type = 0
                    assert library.run2(C.byref(call), C.byref(config), C.byref(library.arr)) == 123


def test_inventory_drift_cannot_silently_change_recipe(library):
    point = copy.deepcopy(next(r for r in plan()["cases"] if r["expected"] == "SELECTED"))
    point["recipe"][2] += 1
    with pytest.raises(ValueError, match="recipe differs"):
        library.select(point)


@pytest.fixture(scope="module")
def weights():
    return Weights(256, 512, 256)


@pytest.mark.parametrize("mode,channels,tokens", [(0, 1, 8), (2, 1, 1), (2, 8, 8)])
@pytest.mark.parametrize("large", [False, True])
def test_factorized_oracle_matches_full_official_dot(weights, mode, channels, tokens, large):
    point = dict(mode=mode, channels=channels, tokens=tokens, rows=tokens*(8 if mode == 2 else 1))
    image, coeff, ids = weights.inputs(point, 2, large)
    gold, denom = weights.truth(point, coeff, ids, "bf16")
    actual = []
    row_stride, token_stride = 520, 520*channels+8 if mode == 2 else 520
    for row in range(point["rows"]):
        token, slot = divmod(row, 8) if mode == 2 else (row, 0)
        expert = int(ids[token, slot]) if ids is not None else 0
        _, w = raw_weight(12, 256, 512, 1709+(expert%8)*31)
        at = token*token_stride+(slot%channels)*row_stride
        a = round_compute(image[at:at+512], "bf16").astype("f8")
        actual.append(w.astype("f8") @ a)
    assert np.allclose(gold, actual, rtol=1e-14, atol=1e-13)
    assert np.all(denom >= np.abs(gold))
    assert rejected(np.zeros_like(gold), gold, denom)
    assert np.array_equal(bf16_float(storage(image, 2)), round_compute(image, "bf16"))


def test_large_range_and_changed_expert_negatives(weights):
    point = dict(mode=2, channels=8, tokens=8, rows=64)
    image, coeff, ids = weights.inputs(point, 2, True)
    assert np.max(np.abs(image)) == OUTLIER
    assert np.isfinite(round_compute(image, "bf16")).all()
    assert not np.isfinite(round_compute(image, "f16")).all()
    assert np.all(np.diff(np.sort(ids[:, :8], axis=1), axis=1) > 0)
    gold, denom = weights.truth(point, coeff, ids, "bf16")
    changed = ids.copy()
    changed[:, :8] = np.roll(changed[:, :8], 1, axis=1)
    wrong, wrong_denom = weights.truth(point, coeff, changed, "bf16")
    assert rejected(gold, wrong, wrong_denom)
    assert weights.pool == 8 and len(weights.samples) == 8
    assert all(x["low"].nbytes == 256*512//2 for x in weights.samples)


def fake_results():
    p = plan()
    out = []
    for point in p["cases"]:
        if point["expected"] != "SELECTED":
            out.append(dict(id=point["id"], status="EXPECTED_QKG_SHAPE"))
            continue
        proofs = []
        for kind in ("F32", "BF16"):
            proofs.append(dict(storage=kind, eager=dict(bad=0), zero_a=True, zero_output_negative=True,
                invalid_ids=True, graph=[dict(repeat=r, large=l, oracle=dict(bad=0), wrong_expert_negative=True)
                for r, l in ((1, False), (2, True), (0, False))]))
        out.append(dict(id=point["id"], status="PASS", bf16=proofs, controls=dict(
            f16_nominal=dict(bad=0), f16_v1_v2_equal=True, f16_overflow_rejected=True, f16_overflow_nonfinite=1)))
    return {"plan": p}, [dict(status="PASS", cases=out)]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "storage", "graph", "overflow", "ids", "process"])
def test_partial_or_failed_device_receipts_do_not_admit(fault):
    manifest, results = fake_results()
    assert summarize(manifest, results)["status"] == "PASS"
    rows = results[0]["cases"]
    first = next(r for r in rows if r["status"] == "PASS")
    if fault == "missing": rows.pop()
    if fault == "duplicate": rows.append(rows[0])
    if fault == "storage": first["bf16"].pop()
    if fault == "graph": first["bf16"][0]["graph"].pop()
    if fault == "overflow": first["controls"]["f16_overflow_nonfinite"] = 0
    if fault == "ids":
        next(r for r in rows if r["id"].startswith("indexed") and r["status"] == "PASS")["bf16"][0]["invalid_ids"] = False
    if fault == "process": results[0]["status"] = "FAIL"
    with pytest.raises(ValueError): summarize(manifest, results)


def test_runner_uses_production_typed_exports_not_generic_sweep():
    source = (ROOT / "dev/bf16_fastpath/gate_plan.py").read_text()
    assert "quactlize_kpack_q4_decode_select_v" in source
    assert "quactlize_kpack_q4_decode_run_v" in source
    assert "simt_query_v2" not in source and "sweep_run" not in source

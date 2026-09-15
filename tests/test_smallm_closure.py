"""The measurement harness must reject missing or misleading evidence."""
from copy import deepcopy
import json

import pytest

from dev.smallm_closure import plan, results
from dev.smallm_closure.fixture import copies_for_l2


def protocol():
    return dict(screen=3, samples=11, rounds=4, threshold_pct=5)


def rows():
    p = dict(id="point", candidates=["old", "new", "tc", "near"], mandatory=["old"])
    c = {k: dict(kind="tc" if k == "tc" else "simt") for k in p["candidates"]}
    screen = {k: dict(candidate=k, status="MEASURED", samples_us=[v] * 3)
              for k, v in (("old", 20.), ("new", 10.), ("tc", 12.), ("near", 10.1))}
    confirmation = {k: dict(status="MEASURED", rounds=[[v] * 11 for _ in range(4)])
                    for k, v in (("old", 20.), ("new", 10.), ("tc", 12.))}
    return p, c, screen, confirmation


def test_competitive_extra_cannot_be_silently_dropped():
    p, c, s, f = rows()
    r = results.adjudicate(p, c, s, f, protocol())
    assert r["status"] == "OPEN"
    assert any(i["reason"] == "COMPETITIVE_UNCONFIRMED" for i in r["issues"])
    f["near"] = dict(status="MEASURED", rounds=[[10.1] * 11 for _ in range(4)])
    r = results.adjudicate(p, c, s, f, protocol())
    assert r["status"] == "MEASURED_POOL_CLOSED"
    assert r["winner"]["candidate"] == "new"


@pytest.mark.parametrize("plant", ("nan", "missing", "short", "unstable", "failed", "tc"))
def test_false_green_plants(plant):
    p, c, s, f = rows()
    f["near"] = dict(status="MEASURED", rounds=[[10.1] * 11 for _ in range(4)])
    if plant == "nan":
        f["new"]["rounds"][0][0] = float("nan")
    elif plant == "missing":
        del s["old"]
    elif plant == "short":
        f["new"]["rounds"][0].pop()
    elif plant == "unstable":
        f["new"]["rounds"][0] = [1.] * 11
    elif plant == "failed":
        s["old"]["status"] = "FAIL"
    else:
        del f["tc"]
    assert results.adjudicate(p, c, s, f, protocol())["status"] == "OPEN"


def test_incumbent_always_confirmed_even_if_screen_slow():
    p, c, s, _ = rows()
    assert "old" in results.shortlist(p, c, s)


def test_receipt_and_resume_identity(tmp_path):
    path = tmp_path / "row.json"
    row = results.seal(dict(authority={"code": 1}, status="MEASURED", samples_us=[3.]))
    results.write(path, row)
    assert results.read(path, {"code": 1}) == row
    with pytest.raises(ValueError):
        results.read(path, {"code": 2})
    data = json.loads(path.read_text())
    data["samples_us"] = [1.]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        results.read(path, {"code": 1})


def test_active_bytes_not_all_experts():
    assert copies_for_l2(1024, 8, 65536) == 18
    assert copies_for_l2(1024, 256, 65536) == 1
    with pytest.raises(ValueError):
        copies_for_l2(1024, 8, 0)


@pytest.fixture(scope="module")
def frozen():
    return plan.make_plan(dict(matrices=[dict(q=14, mode=0, n=248320, k=2048, experts=1,
        topk=1, channels=1, model="model", tensor="output.weight")], exclusions=[], receipts=[]))


def test_all_m_compute_and_incumbents(frozen):
    plan.validate(frozen)
    head = [p for p in frozen["points"] if p["n"] == 248320]
    assert {(p["compute"], p["tokens"]) for p in head} == {(c, m) for c in ("f16", "bf16") for m in range(1, 9)}
    assert all(p["mandatory"] for p in head if p["compute"] == "f16")
    assert all(p["production_bucket_distance"] > 0 for p in head)
    assert sum(len(p["candidates"]) for p in frozen["points"]) < 100000


def test_historical_config_never_pruned_by_shortlist_cap(frozen):
    history = plan.historical()
    for p in frozen["points"]:
        if p["compute"] != "f16":
            continue
        key = tuple(p[f] for f in ("q", "mode", "n", "k", "tokens", "channels"))
        for c in history.get(key, []):
            assert plan.candidate_id(c) in p["mandatory"]


def test_missing_candidate_and_dtype_tamper_rejected(frozen):
    p = deepcopy(frozen)
    p["points"][0]["candidates"] = []
    p["plan_sha256"] = plan.digest({k: v for k, v in p.items() if k != "plan_sha256"})
    with pytest.raises(ValueError):
        plan.validate(p)
    p = deepcopy(frozen)
    p["computes"] = ["f16"]
    with pytest.raises(ValueError):
        plan.validate(p)


def test_audit_keeps_approximations_visible():
    a = plan.evidence_audit()
    assert a["simt_contexts"] == 200
    assert a["overlap"] == 65
    assert len(a["simt_without_tc"]) == 135
    assert {"BF16", "Q8_PROFILED", "FP32_ENDPOINT", "COMPONENT_SUM", "MODEL_FUSION", "LEGACY_BODY"} <= {r["id"] for r in a["issues"]}


def test_profile_sensitive_fast_oracle_is_not_a_production_rule():
    spec = dict(q=12, mode=2, n=1024, k=2048, experts=256, topk=8,
                channels=1, tokens=8, compute="f16")
    rows = []
    for profile, times in (("spread", (10., 20.)), ("cluster", (20., 10.))):
        fast = "a" if times[0] < times[1] else "b"
        rows.append(dict(point_spec=spec | dict(router=profile),
                         winner=dict(candidate=fast, median_us=10.),
                         confirmed=[dict(candidate=c, median_us=t) for c, t in zip(("a", "b"), times)]))
    review = results.policy_review(rows)
    assert review[0]["status"] == "ROUTER_SENSITIVE_PARETO"
    assert review[0]["minimax"]["worst_regret_pct"] == 100
    assert not review[0]["missing_cross_confirm"]


def test_tpot_projection_sum_is_not_labeled_e2e():
    spec = dict(q=8, mode=0, n=512, k=2048, experts=1, topk=1,
                channels=1, tokens=1, compute="f16", router="dense")
    row = dict(point_spec=spec, winner=dict(candidate="a", median_us=3.),
               confirmed=[dict(candidate="a", median_us=3.)])
    inventory = dict(matrices=[spec | dict(model="model", tensor=f"layer{i}", fused=False) for i in range(4)])
    estimates = results.tpot_estimate(inventory, [row], results.policy_review([row]))
    assert estimates[0]["projection_sum_us"] == 12.
    assert estimates[0]["scope"] == "SUM_OF_ISOLATED_PROJECTIONS_NOT_MEASURED_TPOT"
    assert estimates[2]["status"] == "PARTIAL"  # BF16 was not measured.


def test_grouped_indexed_binding_has_own_row_map():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "dev/smallm_closure/bench.py").read_text()
    assert "self.row_ids = Buffer(self.r, b.rows * 4)" in source
    assert "row_ids=self.row_ids.ptr if grouped else None" in source


@pytest.mark.parametrize("q", (8, 10, 11, 12, 13, 14))
def test_chunked_fixture_uses_official_dot(q):
    import numpy as np
    from dev.smallm_closure.fixture import Weights
    from dev.bf16_compute.fixture import raw_weight
    w = Weights(q, 512, 512, 1)
    p = dict(tokens=2, channels=1, topk=1, mode=0, compute="bf16")
    d = w.inputs(p)
    _, g0 = raw_weight(q, 256, 512, 710 + q * 1009)
    _, g1 = raw_weight(q, 256, 512, 710 + q * 1009 + 256)
    gold = np.concatenate((g0, g1)).astype("f8")
    np.testing.assert_allclose(d["gold"], d["a"].astype("f8") @ gold.T, atol=1e-12, rtol=1e-12)


def test_model_headers_include_fused_and_tied_output(tmp_path):
    from tools.gguf_internal_shape_inventory import _synthetic_gguf
    file = tmp_path / "model.gguf"
    file.write_bytes(_synthetic_gguf([("general.architecture", "qwen"), ("qwen.expert_used_count", 8)], [
        ("token_embd.weight", (512, 1024), 14),
        ("blk.0.ffn_gate_exps.weight", (512, 256, 256), 12),
        ("blk.0.ffn_up_exps.weight", (512, 256, 256), 12),
        ("blk.0.ffn_down_exps.weight", (256, 512, 256), 13),
        ("blk.0.attn_q.weight", (512, 512), 8)]))
    model_plan = tmp_path / "models.json"
    model_plan.write_text(json.dumps(dict(models=[dict(name=n, path=str(file)) for n in ("qwen35-35b-q4km", "qwen3-32b-q4km")])))
    inventory = plan.model_inventory(model_plan, None)
    shapes = {(r["tensor"], r["n"], r["k"], r["channels"]) for r in inventory["matrices"]}
    assert ("output.weight", 1024, 512, 1) in shapes
    assert ("blk.0.ffn_gate_up_exps.weight", 512, 512, 1) in shapes
    assert ("blk.0.ffn_down_exps.weight", 512, 256, 8) in shapes
    assert all(r["tensor"] != "token_embd.weight" for r in inventory["matrices"])

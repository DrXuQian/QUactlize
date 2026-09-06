import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kpack_overnight_search as search
import run_kpack_overnight as night
import run_kpack_minimal_probe as probe


@pytest.fixture(scope="module")
def base():
    return search.fix_workers(probe.make_plan(), 2)


def cell(symbol, us=10.0, count=3, grid=0):
    return {
        "symbol": symbol,
        "status": "MEASURED",
        "median_us": us,
        "samples_us": [us] * count,
        "algorithm": "TC_S1",
        "split": 1,
        "grid": grid,
    }


def dataset(plan, count=3):
    return {
        r["id"]: [cell(s, 10 + i * 0.1, count) for i, s in enumerate(r["symbols"])]
        for r in plan["requests"]
    }


def test_stratified_calibration_and_fixed_owners(base):
    p = search.calibration_plan(base)
    assert {r["route"] for r in p["requests"]} == set(probe.tuning.ROUTES)
    for r in p["requests"]:
        original = next(o for o in base["requests"] if o["id"] == r["id"])
        assert r["worker_index"] == original["worker_index"]
        assert set(r["symbols"]) <= set(original["symbols"])
    with pytest.raises(ValueError):
        search.fix_workers(base, 0)
    with pytest.raises(ValueError, match="unknown"):
        search.make_stage(base, {"missing": ["bad"]}, "test")


def test_bounded_exploration_keeps_incumbent(base):
    pool = dataset(base)
    compiled = {c["symbol"] for c in base["candidates"]}
    plan = search.explore_plan(base, pool, compiled, new_limit=4)
    assert len({c["symbol"] for c in plan["candidates"]} - compiled) <= 4
    assert plan["stage_details"]["compile_cap_omissions"] > 0
    for r in plan["requests"]:
        assert plan["stage_details"]["incumbents"][r["id"]] in r["symbols"]
    expired = search.explore_plan(base, pool, compiled, deadline=0)
    assert (
        not expired["requests"]
        and expired["stage_details"]["planning_deadline_reached"]
    )


def test_audit_requires_paired_incumbent(base):
    r = copy.deepcopy(base["requests"][0])
    p = {"requests": [r], "stage_details": {"incumbents": {r["id"]: "a"}}}
    assert not search.audit_gains(p, {r["id"]: [cell("b", 9)]})
    assert search.audit_gains(p, {r["id"]: [cell("a", 10), cell("b", 9)]})
    assert not search.audit_gains(p, {r["id"]: [cell("a", 10), cell("b", 9.9)]})


def test_confirmed_rounds_seeds_and_exact_denominator(base):
    pool = dataset(base)
    plan = search.confirmation_plan(base, pool, pool)
    rounds = [search.round_plan(plan, n) for n in (1, 2, 3)]
    assert len({p["requests"][0]["schedule_salt"] for p in rounds}) == 3
    assert len({p["requests"][0]["worker_index"] for p in rounds}) == 1
    data = [dataset(p, 11) for p in rounds]
    assert all(
        r["status"] == "CONFIRMED_SELECTED_SET"
        for r in search.confirmed_rows(base, plan, data, pool)
    )
    bad = copy.deepcopy(data)
    rid = plan["requests"][0]["id"]
    bad[2][rid][0]["samples_us"].pop()
    assert (
        search.confirmed_rows(base, plan, bad, pool)[0]["status"]
        != "CONFIRMED_SELECTED_SET"
    )
    bad = copy.deepcopy(data)
    bad[1][rid].append(copy.deepcopy(bad[1][rid][0]))
    with pytest.raises(ValueError, match="duplicate"):
        search.confirmed_rows(base, plan, bad, pool)
    bad = copy.deepcopy(data)
    bad[0][rid][0]["samples_us"][0] = float("nan")
    with pytest.raises(ValueError, match="sample"):
        search.confirmed_rows(base, plan, bad, pool)


def test_missing_grid_variant_is_not_complete(base):
    data = dataset(base, 11)
    rid = base["requests"][0]["id"]
    data[rid].append(cell(data[rid][0]["symbol"], 11, 11, grid=72))
    rounds = [copy.deepcopy(data) for _ in range(3)]
    rounds[-1][rid].pop()
    assert (
        search.confirmed_rows(base, base, rounds, dataset(base))[0]["status"]
        == "PARTIAL_CONFIRMATION"
    )


def test_noise_is_explicit(base):
    data = [dataset(base, 11) for _ in range(3)]
    rid = base["requests"][0]["id"]
    data[-1][rid][0]["samples_us"] = [15.0] * 11
    row = search.confirmed_rows(base, base, data, dataset(base))[0]
    assert row["status"] == "NOISY_CONFIRMATION"


def test_cost_model_separates_decode_from_prefill(base, tmp_path):
    p = {"requests": [base["requests"][0]], "candidates": base["candidates"]}
    r = p["requests"][0]
    (tmp_path / "results").mkdir()
    log = tmp_path / "raw.log"
    log.write_text(
        "KPACK_TUNER_PHASE phase=fixture seconds=0.2\nKPACK_TUNER_END id=x rc=0 wall_seconds=1.0\n"
    )
    record = {"logs": [{"path": str(log)}], "cells": [cell(r["symbols"][0])]}
    (tmp_path / "results" / (r["id"] + ".json")).write_text(json.dumps(record))
    model = night.CostModel(p, tmp_path, 45, 2)
    assert model.run_seconds(p, 11) > model.run_seconds(p, 3) > 0
    other = copy.deepcopy(p)
    other["requests"][0]["problem"]["m"] = 2048
    with pytest.raises(ValueError, match="calibration"):
        model.run_seconds(other, 11)


def test_command_deadline_preserves_log(tmp_path):
    campaign = object.__new__(night.Campaign)
    campaign.interrupted = False
    campaign.active = None
    campaign.end = time.monotonic() + 5
    rc, elapsed = campaign.command(
        [sys.executable, "-u", "-c", "print('started'); import time; time.sleep(9)"],
        tmp_path / "command.log",
        time.monotonic() + 0.1,
    )
    assert rc == 124 and elapsed < 3
    assert "started" in (tmp_path / "command.log").read_text()


@pytest.mark.parametrize("admit", (True, False))
def test_campaign_pipeline_or_admission_rejection(base, tmp_path, monkeypatch, admit):
    a = SimpleNamespace(
        output=tmp_path,
        build_cache=None,
        sdk=tmp_path,
        hours=1,
        jobs=2,
        devices="0,1",
        new_parents=0,
    )
    monkeypatch.setattr(night.build, "source_identity", lambda: "source")
    monkeypatch.setattr(night.build, "sdk_identity", lambda _: {})

    class Model:
        safety = 2

        def __init__(self, *args):
            pass

        def run_seconds(self, *args):
            return 1 if admit else 1e9

        def compile_seconds(self, *args):
            return 1

    monkeypatch.setattr(night, "CostModel", Model)
    (tmp_path / "base-plan.json").write_text(json.dumps(base))
    campaign = night.Campaign(a)
    called = []

    def phase(name, plan, cutoff, iterations=3):
        called.append(name)
        campaign.phases.append({"name": name})
        return dataset(plan, iterations)

    monkeypatch.setattr(campaign, "phase", phase)
    rc = campaign.execute()
    report = json.loads((tmp_path / "results/summary.json").read_text())
    assert report["global_5pct_bound_proven"] is False
    if admit:
        assert rc == 0 and report["confirmed_requests"] == len(base["requests"])
        assert called[-3:] == ["confirm-1", "confirm-2", "confirm-3"]
    else:
        assert rc == 2 and called == ["calibration"]
        assert report["confirmed_requests"] == 0

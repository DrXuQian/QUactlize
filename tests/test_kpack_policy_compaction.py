import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_kpack_policy as host_check
import compact_kpack_policy as compact
import fit_kpack_tuner_policy as fit
import generate_kpack_policy_header as export
import kpack_policy as p
import kpack_tuning_plan as tuning
import plan_kpack_policy_validation as planner
import refine_kpack_grid_policy as grids
import run_kpack_policy_validation as box


def problem(m=8, n=1024):
    return dict(qtype=12, m=m, n=n, k=5120, group_size=32)


def config(symbol="a", route="fq-dense", **kwargs):
    c = dict(
        symbol=symbol,
        route=route,
        qtype=12,
        tm=16,
        tn=64,
        tk=256,
        wm=16,
        wn=16,
        stages=2,
        ap=0,
        dn=64,
        persistent=-1,
        algorithm="TC_S1",
        split=1,
        grid=0,
        grid_mode="implicit",
        layout=1,
        mapping_id=p.mapping(12),
    )
    c.update(kwargs)
    return c


def observation(rid, m, costs, n=1024, route="fq-dense"):
    return dict(
        id=rid,
        cell_key=rid,
        route=route,
        problem=problem(m, n),
        status="CONFIRMED_SELECTED_SET",
        costs={
            cid: {
                "regret_pct": value,
                "spread_pct": 0,
                "median_us": 10 * (1 + value / 100),
            }
            for cid, value in costs.items()
        },
        winner_median_us=10,
    )


def test_grid_recipes_match_the_actual_host_source(tmp_path):
    header = (
        Path(__file__).resolve().parents[1]
        / "quactlize/include/scalefirst_persistent_policy.hpp"
    )
    source = f"""#include "{header}"
#include <iostream>
int main() {{ unsigned long long q; int b;
while(std::cin >> q >> b) std::cout << quactlize::scalefirst_policy::capacity_grid(q,72,b) << ' '
 << quactlize::scalefirst_policy::balanced_grid(q,72,b) << '\\n'; }}"""
    binary = tmp_path / "grid"
    subprocess.run(
        ["c++", "-x", "c++", "-std=c++17", "-", "-o", str(binary)],
        input=source,
        text=True,
        check=True,
        capture_output=True,
    )
    cases = [
        (q, b)
        for q in (1, 7, 64, 71, 72, 73, 127, 512, 1024, 2048, 8192)
        for b in (1, 2, 8, 12)
    ]
    run = subprocess.run(
        [str(binary)],
        input="".join(f"{q} {b}\n" for q, b in cases),
        text=True,
        check=True,
        capture_output=True,
    )
    expected = []
    for q, b in cases:
        c = config(route="sf-dense", tm=1, tn=16, grid_b=b, occupancy=12)
        pr = problem(q, 16)
        expected.append(
            " ".join(
                str(p.resolve_grid(dict(c, grid_mode=mode), pr))
                for mode in ("capacity", "balanced")
            )
        )
    assert run.stdout.splitlines() == expected


def test_grid_mask_witness_and_negatives():
    c = config(
        route="sf-dense",
        tm=1,
        tn=16,
        algorithm="PERSISTENT",
        grid_mode="fixed",
        grid=512,
    )
    pr = problem(2048, 16)
    actual = grids.witnesses(c, pr, (8, 0, 1 << 8))
    assert len(actual) == 1 and actual[0]["grid_mode"] == "balanced"
    for meta in ((8, 1 << 8, 0), (7, 0, 1 << 8), (8, 0, 0), (0, 0, 0)):
        with pytest.raises(ValueError):
            grids.witnesses(c, pr, meta)
    with pytest.raises(ValueError, match="exact dense"):
        p.resolve_grid(dict(actual[0], route="sf-grouped"), pr)


def test_cached_tree_fit_equals_independent_fits_and_rejects_changed_costs():
    points = [
        dict(features=(i,), costs=cost, blocked="", rows=[])
        for i, cost in (
            (1, {"a": (0, 0)}),
            (2, {"a": (2, 2), "b": (0, 0)}),
            (4, {"b": (0, 0)}),
            (8, {"b": (1, 1)}),
        )
    ]
    cached = p.tree_fitter(points)
    for i in range(len(points)):
        subset = points[:i] + points[i + 1 :]
        assert cached(subset) == p.fit_tree(subset)
    changed = copy.deepcopy(points)
    changed[0]["costs"]["a"] = (99, 99)
    with pytest.raises(ValueError, match="different costs"):
        cached(changed)


def test_pooling_merges_small_differences_across_N_without_expanding_family_authority():
    configs = {"a": config("a"), "b": config("b"), "common": config("common")}
    observations = [
        observation("x", 8, {"a": 0, "common": 2}),
        observation("y", 8, {"b": 0, "common": 3}, n=2048),
        observation("x2", 32, {"common": 0}),
        observation("y2", 32, {"common": 0}, n=2048),
    ]
    baseline = fit.make_policy(observations, configs, {})[0]
    model, report, _ = compact.fit(observations, configs, baseline, True)
    assert (
        report["rules"] == 1
        and report["parent_symbols"] == 1
        and report["shape_guards"] == 2
    )
    assert report["max_round_regret_pct"] == 3
    assert host_check.check(model)["status"] == "PASS"
    for n in (1024, 2048):
        choice = p.select(
            model,
            "fq-dense",
            problem(8, n),
            device_name="PPU-ZW810",
            compute_units=72,
            mapping_id=p.mapping(12),
        )
        assert choice["config_id"] == "common"
    assert (
        p.select(
            model,
            "fq-dense",
            problem(8, 1536),
            device_name="PPU-ZW810",
            compute_units=72,
            mapping_id=p.mapping(12),
        )["reason"]
        == "UNMEASURED_FAMILY"
    )


def test_pooling_never_merges_missing_or_over_budget_costs():
    configs = {"a": config("a"), "b": config("b")}
    for second in ({"b": 0}, {"a": 6, "b": 0}):
        observations = [
            observation("x", 8, {"a": 0}),
            observation("y", 8, second, n=2048),
        ]
        baseline = fit.make_policy(observations, configs, {})[0]
        _, report, _ = compact.fit(observations, configs, baseline, True)
        assert report["rules"] == 2


@pytest.fixture
def suite():
    parent = next(
        c
        for c in tuning.candidates(12, "fq-dense")
        if c.tm == 16 and c.ap == 0 and c.tk == 256
    )
    c = p.runtime_config(
        asdict(parent), dict(algorithm="TC_S1", split=1, grid=0), problem()
    )
    rid = p.digest(["test", "fq-dense"])
    request = dict(
        id=rid,
        cell_key="test",
        route="fq-dense",
        qtype=12,
        problem=problem(),
        worker_index=0,
        symbols=[parent.symbol],
    )
    return dict(
        schema=planner.SCHEMA,
        plan=dict(requests=[request], candidates=[asdict(parent)]),
        roles={rid: ["adjacent-merge"]},
        targets={rid: [dict(config_id=p.digest(c), config=c)]},
        limits=dict(rounds=3, iterations_per_round=11, correctness_repeats=1),
        denominator=dict(route_workloads=1),
        policy_sha256="policy",
        authority={},
    )


def test_suite_validation_rejects_changed_parent_and_identity(suite):
    box.validate_suite(suite)
    rid = next(iter(suite["targets"]))
    planted = copy.deepcopy(suite)
    c = planted["targets"][rid][0]
    c["config"]["tn"] *= 2
    c["config_id"] = p.digest(c["config"])
    with pytest.raises(ValueError, match="identity"):
        box.validate_suite(planted)
    planted = copy.deepcopy(suite)
    planted["limits"]["rounds"] = 1
    with pytest.raises(ValueError, match="controls"):
        box.validate_suite(planted)


def test_proposed_grid_is_not_silently_replaced_and_failed_requests_cannot_pass(suite):
    rid = next(iter(suite["targets"]))
    c = suite["targets"][rid][0]["config"]
    cell = dict(
        symbol=c["symbol"],
        algorithm="TC_S1",
        split=1,
        grid=0,
        status="MEASURED",
        samples_us=[10.0] * 11,
        median_us=10.0,
    )
    rounds = [{rid: [copy.deepcopy(cell)]} for _ in range(3)]
    assert box.assess(suite, rounds, set())["targets"][0]["status"] == "WITHIN_BUDGET"
    assert box.assess(suite, rounds, {rid})["status"] == "INCOMPLETE"
    assert (
        box.assess(suite, rounds, {rid})["targets"][0]["status"]
        == "INCOMPLETE_OR_REJECTED"
    )
    suite["targets"][rid][0]["config"]["grid"] = 72
    assert (
        box.assess(suite, rounds, set())["targets"][0]["status"]
        == "PROPOSED_RUNTIME_UNAVAILABLE"
    )


def test_box_preflight_has_no_compilation_path(tmp_path, suite, monkeypatch):
    identity = dict(kernel_source="kernel", sdk={"runtime": "sdk"})
    suite["authority"] = {"campaign_identity": identity}
    (tmp_path / "campaign-identity.json").write_text(json.dumps(identity))
    folder = tmp_path / "phases/confirm-1"
    folder.mkdir(parents=True)
    (folder / "bundle.json").write_text(
        json.dumps({"identity": {"source": "kernel", "sdk": identity["sdk"]}})
    )
    monkeypatch.setattr(box.build, "source_identity", lambda: "kernel")
    monkeypatch.setattr(box.build, "sdk_identity", lambda sdk: identity["sdk"])
    monkeypatch.setattr(
        box.build, "build", lambda *a, **k: pytest.fail("compile forbidden")
    )
    monkeypatch.setattr(box.overnight, "subset_bundle", lambda b, p: {"reused": True})
    assert box.preflight(suite, tmp_path, tmp_path) == {"reused": True}
    monkeypatch.setattr(box.overnight, "subset_bundle", lambda b, p: None)
    with pytest.raises(ValueError, match="no compile"):
        box.preflight(suite, tmp_path, tmp_path)


def test_published_compact_policy_and_suite_are_one_frozen_selection():
    root = Path(__file__).resolve().parents[1]
    read = lambda path: json.loads((root / path).read_text())
    model = read("policies/kpack_zw810_compact.json")
    report = read("policies/kpack_zw810_compact.report.json")
    suite = read("policies/kpack_zw810_compact.validation.json")
    box.validate_suite(suite)
    assert p.digest(model) == suite["policy_sha256"]
    assert suite["authority"] == model["authority"]
    assert report["max_round_regret_pct"] <= model["regret_budget_pct"] == 5
    assert len(model["rule_groups"]) == report["rule_groups"] == 20
    assert len(model["families"]) == report["shape_guards"] == 152
    assert (
        sum(p.leaf_count(g["tree"]) for g in model["rule_groups"])
        == report["rules"]
        == 788
    )
    assert (
        len({c["symbol"] for c in model["configurations"].values()})
        == report["parent_symbols"]
        == 219
    )
    assert len(model["configurations"]) == report["runtime_variants"] == 312
    assert suite["denominator"]["route_workloads"] == 315
    assert suite["denominator"]["new_parent_union"] == 0
    assert (root / "policies/kpack_zw810_compact.hpp").read_text() == export.generate(
        model
    )

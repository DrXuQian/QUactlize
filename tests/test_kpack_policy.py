import copy
import json
from pathlib import Path
import sys
import re

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kpack_policy as p
import fit_kpack_tuner_policy as fit


def problem(m=8, route="fq-dense", **kwargs):
    result = dict(qtype=12, n=1024, k=5120, group_size=32)
    (
        result.update(m=m)
        if route.endswith("dense")
        else result.update(total_rows=m, max_rows=4, experts=256)
    )
    result.update(kwargs)
    return result


def config(name="a", route="fq-dense", **kwargs):
    result = dict(
        symbol=name,
        qtype=12,
        route=route,
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
    result.update(kwargs)
    return result


def obs(
    m, costs, *, route="fq-dense", status="CONFIRMED_SELECTED_SET", name=None, **kwargs
):
    return dict(
        id=name or str(m),
        route=route,
        cell_key=f"{route}:{name or m}",
        problem=problem(m, route, **kwargs),
        status=status,
        winner_median_us=10,
        costs={
            c: {"regret_pct": r, "spread_pct": 0, "median_us": 10 * (1 + r / 100)}
            for c, r in costs.items()
        },
    )


def query(model, m=8, route="fq-dense", **kwargs):
    return p.select(
        model,
        route,
        problem(m, route, **kwargs),
        device_name="PPU-ZW810",
        compute_units=72,
        mapping_id=p.mapping(12),
    )


def test_cost_based_compression_and_interpolation_not_exact_winner_cache():
    observations = [
        obs(8, {"a": 0, "b": 2}),
        obs(16, {"b": 0, "a": 3}),
        obs(32, {"b": 0}),
    ]
    model, report, replay, holdout, _ = fit.make_policy(
        observations, {"a": config(), "b": config("b")}, {}
    )
    assert report["rules"] == 1
    assert query(model)["config_id"] == "b"
    assert query(model, 12)["status"] == "INTERPOLATED_PROPOSAL"
    assert not query(model, 12)["performance_validated_at_query"]
    assert query(model, 8)["status"] == "MEASURED_POLICY"
    assert query(model, 64)["status"] == "NO_MEASURED_POLICY"
    assert report["training_max_round_regret_pct"] == 2
    assert not model["compiled_default"] and not model["production_policy_updated"]
    assert sum(r["status"] == "WITHIN_BUDGET" for r in holdout) == 1


def test_missing_measurements_are_unknown_not_fast():
    model, report, _, holdout, followup = fit.make_policy(
        [obs(8, {"a": 0}), obs(16, {"b": 0}), obs(32, {"a": 0})],
        {"a": config(), "b": config("b")},
        {},
    )
    assert report["rules"] == 3
    assert query(model, 16)["config_id"] == "b"
    middle = next(h for h in holdout if h["problem"]["m"] == 16)
    assert middle["status"] == "UNMEASURED_CANDIDATE"
    assert middle in followup["holdout"]


def test_noise_is_not_silently_covered_by_an_interval():
    model, report, _, _, _ = fit.make_policy(
        [
            obs(8, {"a": 0}),
            obs(16, {"a": 0}, status="NOISY_CONFIRMATION"),
            obs(32, {"a": 0}),
        ],
        {"a": config()},
        {},
    )
    assert query(model, 16)["reason"] == "NOISY_OR_INCOMPLETE_CONFIRMATION"
    assert report["selection_status"] == {"MEASURED_POLICY": 2, "NO_MEASURED_POLICY": 1}


def test_grouped_alias_minimax_uses_no_router_features_and_holdout_has_no_leak():
    route = "fq-grouped"
    observations = [
        obs(16, {"a": 0, "b": 2}, route=route, name="permutation-a"),
        obs(16, {"a": 8, "b": 3}, route=route, name="permutation-b"),
        obs(8, {"b": 0}, route=route),
        obs(32, {"b": 0}, route=route),
    ]
    model, report, _, holdout, _ = fit.make_policy(
        observations, {"a": config(route=route), "b": config("b", route=route)}, {}
    )
    assert report["public_points"] == 3 and report["alias_points"] == 1
    assert query(model, 16, route)["config_id"] == "b"
    assert query(model, 16, route)["scope"] == "OBSERVED_ROUTER_FIXTURES"
    assert len(holdout) == 3
    assert next(h for h in holdout if len(h["ids"]) == 2)["status"] == "WITHIN_BUDGET"
    serialized = json.dumps(model)
    assert "permutation-a" not in serialized and "rows_hash" not in serialized


def test_grouped_conflict_fails_closed_even_when_individual_points_are_green():
    route = "fq-grouped"
    observations = [
        obs(16, {"a": 0}, route=route, name="a"),
        obs(16, {"b": 0}, route=route, name="b"),
    ]
    model, report, *_ = fit.make_policy(
        observations, {"a": config(route=route), "b": config("b", route=route)}, {}
    )
    assert query(model, 16, route)["reason"] == "PUBLIC_FEATURE_CONFLICT"
    assert report["blocked_reasons"] == {"PUBLIC_FEATURE_CONFLICT": 2}


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"ap": 1}, "INTERPOLATED_CONFIG_INADMISSIBLE"),
        ({"tk": 512, "split": 4}, "INTERPOLATED_CONFIG_INADMISSIBLE"),
    ],
)
def test_interpolated_config_validity_is_checked(change, reason):
    family = p.fit_family(
        p.family_key("fq-dense", problem()),
        [
            {"features": (1,), "rows": [], "costs": {"a": (0, 0)}, "blocked": ""},
            {"features": (32,), "rows": [], "costs": {"a": (0, 0)}, "blocked": ""},
        ],
    )
    assert (
        p.select_family(family, {"a": config(**change)}, problem(8))["reason"] == reason
    )


def test_tm8_domains_are_route_specific():
    assert p.admissible(config(tm=8), problem(64))
    assert not p.admissible(config(tm=8), problem(65))
    assert not p.admissible(config(tm=8, route="sf-dense"), problem(8))
    assert p.admissible(
        config(tm=8, route="fq-grouped"), problem(512, "fq-grouped", max_rows=129)
    )


def test_runtime_key_keeps_provider_delivery_split_and_persistent_grid():
    base = config(route="sf-dense")
    candidate = {
        k: base[k]
        for k in (
            "symbol",
            "route",
            "qtype",
            "tm",
            "tn",
            "tk",
            "wm",
            "wn",
            "stages",
            "ap",
            "dn",
            "persistent",
        )
    }
    cell = dict(algorithm="NONPERSISTENT", split=1, grid=16)
    a = p.runtime_config(candidate, cell, problem(8))
    b = p.runtime_config(candidate, dict(cell, grid=32), problem(32))
    assert p.digest(a) == p.digest(b) and a["grid_mode"] == "ordinary"
    with pytest.raises(ValueError, match="grid"):
        p.runtime_config(candidate, dict(cell, grid=17), problem(8))
    a = p.runtime_config(
        candidate, dict(cell, algorithm="PERSISTENT", grid=72), problem(8)
    )
    b = p.runtime_config(
        candidate, dict(cell, algorithm="PERSISTENT", grid=144), problem(8)
    )
    assert p.digest(a) != p.digest(b)
    for field, value in (("ap", 1), ("dn", 32)):
        modified = p.runtime_config(
            dict(candidate, **{field: value}),
            dict(cell, algorithm="PERSISTENT", grid=72),
            problem(1),
        )
        assert p.digest(a) != p.digest(modified)


@pytest.mark.parametrize(
    "fields",
    [
        {"m": True},
        {"m": 0},
        {"group_size": 16},
        {"k": 5000},
        {"profile": "balanced"},
        {"m": 1.5},
    ],
)
def test_invalid_public_queries_rejected(fields):
    with pytest.raises(ValueError):
        p.validate_problem("fq-dense", problem(**fields))


def test_device_mapping_and_family_misses_do_not_default():
    model, *_ = fit.make_policy([obs(8, {"a": 0})], {"a": config()}, {})
    assert query(model, n=2048)["reason"] == "UNMEASURED_FAMILY"
    assert (
        p.select(
            model,
            "fq-dense",
            problem(),
            device_name="PPU-ZW810",
            compute_units=1,
            mapping_id=p.mapping(12),
        )["reason"]
        == "DEVICE_MISMATCH"
    )
    assert (
        p.select(
            model,
            "fq-dense",
            problem(),
            device_name="PPU-ZW810",
            compute_units=72,
            mapping_id="0x0",
        )["reason"]
        == "MAPPING_MISMATCH"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 6])
def test_budget_cannot_silently_relax_confirmation(value):
    with pytest.raises(ValueError):
        fit.make_policy([], {}, {}, value)


def test_raw_path_relocation_does_not_mutate_evidence_or_escape(tmp_path, monkeypatch):
    campaign = fit.Campaign(tmp_path)
    record = {
        "logs": [
            {"path": "/box/run/phases/confirm-1/run/logs/x.log", "sha256": "receipt"}
        ]
    }
    saved = copy.deepcopy(record)
    calls = []
    monkeypatch.setattr(fit, "verify_result", lambda r, q, n: calls.append((r, q, n)))
    campaign.replay_record(record, {}, "confirm-1")
    assert record == saved
    assert calls[0][0]["logs"][0]["path"] == str(
        tmp_path / "phases/confirm-1/run/logs/x.log"
    )
    record["logs"][0]["path"] = "/box/run/phases/confirm-1/run/logs/../../../../escape"
    with pytest.raises(ValueError):
        campaign.replay_record(record, {}, "confirm-1")


def test_changed_original_parser_fails_before_fitting(tmp_path):
    files = {
        "base-plan.json": {},
        "results/summary.json": {"schema": "quactlize.kpack-overnight.v1"},
        "campaign-identity.json": {
            "orchestrator": [["run_kpack_tuner.py", "bad-hash"]]
        },
        "device-identity.json": [],
        "phases/confirmation-input/plan.json": {},
    }
    for name, data in files.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="parser/orchestrator changed"):
        fit.Campaign(tmp_path).load()


def test_group_sizes_match_the_canonical_registry():
    registry = (fit.ROOT / "quactlize/include/ppu_format_config.inc").read_text()
    rows = re.findall(r'X\(Q\d_K, "Q\d_K", (\d+), \d+, \d+, (\d+),', registry)
    assert {int(q): int(gs) for q, gs in rows} == p.GROUP_SIZE


def test_native_header_matches_python_at_points_and_boundaries():
    import check_kpack_policy as check

    observations = [
        obs(1, {"ap1": 0}),
        obs(8, {"a": 0}),
        obs(16, {"a": 0}),
        obs(32, {"a": 0}, status="NOISY_CONFIRMATION"),
        obs(64, {"b": 0}),
    ]
    configurations = {"ap1": config("ap1", ap=1), "a": config(), "b": config("b")}
    route = "sf-dense"
    observations.extend([obs(m, {"sf": 0}, route=route) for m in (8, 64)])
    configurations["sf"] = config(
        "sf", route=route, grid_mode="ordinary", algorithm="NONPERSISTENT"
    )
    route = "fq-grouped"
    observations.extend(
        [obs(m, {"g": 0}, route=route, max_rows=mx) for m, mx in ((16, 1), (512, 129))]
    )
    configurations["g"] = config(
        "g", route=route, algorithm="GROUPED_PERSISTENT", grid_mode="fixed", grid=72
    )
    model, *_ = fit.make_policy(observations, configurations, {})
    result = check.check(model)
    assert result["status"] == "PASS" and result["queries"] > 40


def test_recheck_unions_alias_candidates_without_a_new_compile_space():
    import kpack_tuning_plan as tuning
    from dataclasses import asdict

    candidates = list(tuning.candidates(12, "fq-grouped"))[:2]
    requests = []
    for i, candidate in enumerate(candidates):
        requests.append(
            {
                "id": str(i),
                "cell_key": f"grouped-{i}",
                "route": "fq-grouped",
                "qtype": 12,
                "problem": problem(16, "fq-grouped"),
                "symbols": [candidate.symbol],
            }
        )
    base = {"requests": requests, "candidates": [asdict(c) for c in candidates]}
    replay = [{"id": r["id"], "config_id": None} for r in requests]
    plan = fit.recheck_plan(base, base, {}, replay)
    union = {c.symbol for c in candidates}
    assert len(plan["requests"]) == 2
    assert all(set(r["symbols"]) == union for r in plan["requests"])
    assert plan["stage_details"]["new_parent_union"] == 0
    assert plan["denominator"]["selected_parent_workloads"] == 4
    with pytest.raises(ValueError, match="unknown workload"):
        fit.recheck_plan(base, base, {}, [{"id": "absent", "config_id": None}])


def test_checked_in_cpp_is_the_exact_policy_export():
    import generate_kpack_policy_header as header

    root = Path(__file__).resolve().parents[1]
    model = json.loads((root / "policies/kpack_zw810_v1.json").read_text())
    assert header.generate(model) == (root / "policies/kpack_zw810_v1.hpp").read_text()
    assert all(p.digest(c) == cid for cid, c in model["configurations"].items())
    assert {c["qtype"] for c in model["configurations"].values()} == set(range(10, 15))
    assert {c["route"] for c in model["configurations"].values()} == set(p.ROUTES)

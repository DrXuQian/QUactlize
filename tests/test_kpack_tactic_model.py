import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fit_kpack_tactic_model as fit
import kpack_policy as policy
import kpack_tactic_evidence as evidence
import kpack_tactic_model as model


def context(route="fq-dense", m=8, rows=None, **kwargs):
    p = dict(qtype=12, n=64, k=512, group_size=32)
    p.update(
        dict(m=m)
        if route.endswith("dense")
        else dict(experts=len(rows), total_rows=sum(rows), max_rows=max(rows))
    )
    p.update(kwargs)
    return model.context(route, p, rows)


def parent(symbol="p", route="fq-dense", **kwargs):
    p = dict(
        symbol=symbol,
        route=route,
        qtype=12,
        tm=8,
        tn=64,
        tk=64,
        wm=8,
        wn=16,
        stages=2,
        ap=0,
        dn=16,
        persistent=-1,
        occupancy=4,
        shipping_smem=11264,
        low_bits=4,
        high_bits=0,
        metadata_bytes_per_superblock=16,
    )
    p.update(kwargs)
    return p


def scorer(*parents):
    return dict(
        schema=model.SCHEMA,
        device=dict(name="PPU-ZW810", compute_units=72),
        feature_names=dict(
            parent=list(model.PARENT_FEATURES), runtime=list(model.RUNTIME_FEATURES)
        ),
        parent_weights={
            f"{p['qtype']}/{p['route']}": [0.0] * len(model.PARENT_FEATURES)
            for p in parents
        },
        runtime_weights={
            p["route"]: [0.0] * len(model.RUNTIME_FEATURES) for p in parents
        },
        parents={p["symbol"]: p for p in parents},
        cache={},
    )


def cell(parent, ctx, *, split=1, regret=0, spread=0, time=10, index=None):
    choices = model.runtime_choices(parent, ctx)
    choice = (
        choices[index]
        if index is not None
        else next(t for t in choices if t["split"] == split)
    )
    return dict(
        tactic=choice,
        cost=dict(
            regret_pct=regret,
            spread_pct=spread,
            median_us=time,
            round_medians_us=[time] * 3,
        ),
    )


def observation(ctx, cells, name="a", epoch="old", status=fit.STABLE):
    return dict(
        id=name, epoch=epoch, request_id=name, context=ctx, cells=cells, status=status
    )


def test_grouped_grid_uses_actual_rows_not_just_sum_and_max():
    # Identical public features, unequal CTA counts for TM8.
    a, b = context("fq-grouped", rows=[9, 9, 0, 0]), context(
        "fq-grouped", rows=[9, 4, 3, 2]
    )
    p = parent(route="fq-grouped", persistent=1)
    assert a["problem"] == b["problem"]
    assert model.tile_count(p, a) == 4
    assert model.tile_count(p, b) == 5
    assert {t["grid"] for t in model.runtime_choices(p, a)} == {4}
    assert {t["grid"] for t in model.runtime_choices(p, b)} == {5}
    assert model.cache_key(a) != model.cache_key(b)
    assert fit.fold(a) == fit.fold(b)


def test_grid_capacity_balanced_masks_deduplicate_but_preserve_witnesses():
    for q in (1, 4, 71, 72, 73, 129, 576, 1025):
        choices = model.grid_choices(q, 72, 12)
        assert len({x["grid"] for x in choices}) == len(choices)
        for b in range(1, 13):
            cap = min(q, 72 * b)
            waves = (q + 72 * b - 1) // (72 * b)
            balanced = (q + waves - 1) // waves
            assert next(x for x in choices if x["grid"] == cap)["capacity_b_mask"] & (
                1 << b
            )
            assert next(x for x in choices if x["grid"] == balanced)[
                "balanced_b_mask"
            ] & (1 << b)


def test_grid_formula_matches_source_examples():
    rows = model.grid_choices(1024, 72, 8)
    balanced = next(x for x in rows if x["grid"] == 512)
    assert balanced["balanced_b_mask"] & (1 << 8)
    assert next(x for x in rows if x["grid"] == 576)["capacity_b_mask"] & (1 << 8)


def test_fq_split_and_parent_admission_boundaries():
    p = parent()
    assert {t["split"] for t in model.runtime_choices(p, context(m=48))} == {1, 2, 4, 8}
    assert {t["split"] for t in model.runtime_choices(p, context(m=64))} == {1}
    assert model.runtime_choices(p, context(m=65)) == []
    assert model.runtime_choices(parent(ap=1), context(m=2)) == []
    assert model.runtime_choices(parent(ap=1), context(m=1))
    # S1 doesn't inherit the Split-K minimum partition depth guard.
    assert {
        t["split"] for t in model.runtime_choices(parent(stages=12), context(k=512))
    } == {1}


def test_sf_m8_boundary_and_grouped_no_total_m8_limit():
    p = parent(route="sf-dense")
    assert model.runtime_choices(p, context("sf-dense", m=7))
    assert not model.runtime_choices(p, context("sf-dense", m=8))
    p = parent(route="sf-grouped")
    assert model.runtime_choices(p, context("sf-grouped", rows=[129, 128, 0]))


@pytest.mark.parametrize("route", policy.ROUTES)
def test_features_finite_and_names_match(route):
    ctx = context(route, rows=[129, 0, 8] if route.endswith("grouped") else None)
    p = parent(route=route, tm=16, wm=16)
    assert len(model.parent_features(p, ctx)) == len(model.PARENT_FEATURES)
    assert np.isfinite(model.parent_features(p, ctx)).all()
    for t in model.runtime_choices(p, ctx):
        assert len(model.runtime_features(p, ctx, t)) == len(model.RUNTIME_FEATURES)
        assert np.isfinite(model.runtime_features(p, ctx, t)).all()


def test_metadata_registry_bytes_are_not_a_paired_unit_overcount():
    q3 = parent(qtype=11, low_bits=2, high_bits=1, metadata_bytes_per_superblock=14)
    q6 = parent(qtype=14, low_bits=4, high_bits=2, metadata_bytes_per_superblock=18)
    c3, c6 = context(qtype=11, group_size=16), context(qtype=14, group_size=16)
    assert model.geometry(q3, c3)["metadata_bytes"] == 1 * 64 * 2 * 14
    assert model.geometry(q6, c6)["metadata_bytes"] == 1 * 64 * 2 * 18


def test_context_rejects_nonmatching_router_rows():
    ctx = context("fq-grouped", rows=[9, 0, 1])
    with pytest.raises(ValueError, match="actual"):
        model.context("fq-grouped", ctx["problem"], [8, 1, 1])
    with pytest.raises(ValueError, match="must not"):
        model.context("fq-dense", context()["problem"], [8])


def test_fold_keeps_qtypes_routes_epochs_and_router_permutations_together():
    c = context("fq-grouped", rows=[129, 8, 0])
    aliases = [
        context(route, rows=rows, qtype=q, group_size=policy.GROUP_SIZE[q])
        for route in ("fq-grouped", "sf-grouped")
        for q in range(10, 15)
        for rows in ([129, 8, 0], [0, 129, 8])
    ]
    assert {fit.fold(x) for x in aliases} == {fit.fold(c)}


def test_centered_ridge_is_invariant_to_query_time_scale_and_candidate_order():
    groups = [([[1.0, 0], [0.0, 1], [2.0, -1]], [1.0, 2.0, 0.0], 1.0)]
    w = fit.ridge(groups, 2)
    shifted = [
        ([*reversed(groups[0][0])], [x + 20 for x in reversed(groups[0][1])], 1.0)
    ]
    assert np.allclose(w, fit.ridge(shifted, 2))
    assert model.linear_score(w, [2.0, -1]) < model.linear_score(w, [0.0, 1])
    with pytest.raises(ValueError, match="invalid"):
        fit.ridge([([[float("nan")]], [1.0], 1.0)], 1)


def test_training_does_not_mix_runtime_target_with_parent_target():
    a, b, ctx = parent("a"), parent("b", tn=32), context(m=48)
    data = dict(parents={"a": a, "b": b})
    o = observation(
        ctx,
        [
            cell(a, ctx, split=1, time=32),
            cell(a, ctx, split=4, time=18),
            cell(b, ctx, time=22),
        ],
    )
    pg, rg = fit.training_groups(data, [o])
    assert np.allclose(pg["12/fq-dense"][0][1], np.log([18.0, 22.0]))
    assert np.allclose(rg["fq-dense"][0][1], np.log([32.0, 18.0]))


def test_missing_costs_do_not_pass_or_claim_proven_loss():
    a, b, ctx = parent("a"), parent("b"), context()
    m = scorer(a, b)
    # Scores tie: only a is recommended first; only b has a measured cost.
    o = observation(ctx, [cell(b, ctx, regret=20)])
    r = fit.replay(m, o)
    assert r["top1"] == "UNKNOWN"
    assert r["pool_parent_top1"] == "UNKNOWN"
    assert r["pool_parent_top5"] == "UNKNOWN"
    assert r["conditional_parent_top1"] == "KNOWN_OUTSIDE_5PCT"


def test_replay_keeps_historical_winner_as_denominator():
    a, b, ctx = parent("a"), parent("b"), context()
    m = scorer(a, b)
    r = fit.replay(
        m, observation(ctx, [cell(a, ctx, regret=60), cell(b, ctx, regret=0)])
    )
    assert r["top1"] == "KNOWN_OUTSIDE_5PCT"
    assert r["top1_regret_pct"] == 60
    assert r["pool_parent_top3"] == "WITHIN_5PCT"


def test_recommendation_is_unmeasured_and_kpack_only_no_compiled_default():
    a, ctx = parent(), context()
    m = scorer(a)
    result = model.recommend(m, ctx)
    assert result["status"] == "UNMEASURED_SHORTLIST"
    assert not result["performance_admitted"]
    assert result["fallback"] == "CALLER_ADMITTED_KPACK_TACTIC_REQUIRED"
    assert len(result["candidates"][0]["tactics"]) <= 3
    with pytest.raises(ValueError, match="device"):
        model.recommend(m, ctx, compute_units=1)
    with pytest.raises(ValueError, match="arrangement"):
        model.recommend(m, ctx, mapping_id="0x0")


def test_cache_requires_exact_rows_and_every_epoch_within_both_limits():
    p, ctx = parent(), context()
    o = observation(ctx, [cell(p, ctx)])
    data = dict(parents={"p": p}, observations=[o])
    cache, blocked = fit.measured_cache(data)
    assert len(cache) == 1 and not blocked
    m = scorer(p)
    m["cache"] = cache
    result = model.recommend(m, ctx)
    assert result["status"] == "MEASURED_TACTIC"
    assert (
        result["runtime_validation_required"]
        and not result["production_policy_updated"]
    )
    assert model.recommend(m, context(m=9))["status"] == "UNMEASURED_SHORTLIST"
    for later in (
        observation(ctx, [cell(p, ctx, regret=6)], name="b", epoch="new"),
        observation(ctx, [cell(p, ctx, spread=6)], name="b", epoch="new"),
        observation(
            ctx, [cell(p, ctx)], name="b", epoch="new", status="NOISY_CONFIRMATION"
        ),
    ):
        data["observations"] = [o, later]
        assert fit.measured_cache(data)[0] == {}
    m["cache"][model.cache_key(ctx)]["tactic"]["grid"] = 999
    with pytest.raises(ValueError, match="cached"):
        model.recommend(m, ctx)


def test_model_digest_feature_order_and_nan_fail_closed():
    m = scorer(parent())
    m["model_digest"] = policy.digest(m)
    model.validate_model(m)
    bad = copy.deepcopy(m)
    bad["parents"]["p"]["tm"] = 16
    with pytest.raises(ValueError, match="digest"):
        model.validate_model(bad)
    with pytest.raises(ValueError, match="invalid"):
        model.linear_score([float("nan")], [1.0])
    bad = copy.deepcopy(m)
    bad["feature_names"]["parent"].reverse()
    bad["model_digest"] = policy.digest(
        {k: v for k, v in bad.items() if k != "model_digest"}
    )
    with pytest.raises(ValueError, match="feature"):
        model.validate_model(bad)


def test_exact_router_receipt_is_digest_of_rows_not_text_file():
    rows = [9, 0, 1]
    ctx = context("fq-grouped", rows=rows)
    r = dict(
        route="fq-grouped",
        problem=ctx["problem"],
        grouped=dict(rows_file="rows.txt", rows_sha256=policy.digest(rows)),
    )
    plan = dict(router_files={"rows.txt": "9\n0\n1\n"})
    assert evidence.request_context(r, plan) == ctx
    plan["router_files"]["rows.txt"] = "1\n0\n9\n"
    with pytest.raises(ValueError, match="digest"):
        evidence.request_context(r, plan)


def test_evidence_cache_cannot_be_changed_without_replay():
    d = dict(
        schema="quactlize.kpack-tactic-evidence.v1",
        parents={},
        observations=[],
        authority={},
    )
    d["evidence_digest"] = policy.digest(d)
    assert evidence.combine(d)["observations"] == []
    d["observations"].append({})
    with pytest.raises(ValueError, match="evidence digest"):
        evidence.combine(d)


def test_family_hints_use_only_training_points_and_never_copy_split():
    a, b = parent("a"), parent("b")
    low, high = context(m=32), context(m=64)
    observations = [
        observation(low, [cell(a, low, split=4)], name="low"),
        observation(high, [cell(b, high)], name="high"),
    ]
    m = scorer(a, b)
    m["anchors"] = fit.measured_anchors(observations)
    query = context(m=48)
    assert model.nearby_parents(m, query) == ["a", "b"]
    assert {t["split"] for t in model.runtime_choices(b, query)} == {1, 2, 4, 8}
    assert model.nearby_parents(m, context(m=48, n=128)) == []
    m["anchors"] = fit.measured_anchors(observations[:1])
    assert model.nearby_parents(m, query) == ["a"]


def test_hint_family_alias_conflict_is_not_a_hidden_label_feature():
    a = parent("a", route="fq-grouped")
    b = parent("b", route="fq-grouped")
    ca = context("fq-grouped", rows=[9, 9, 0, 0])
    cb = context("fq-grouped", rows=[9, 4, 3, 2])
    assert (
        fit.measured_anchors(
            [
                observation(ca, [cell(a, ca)]),
                observation(cb, [cell(b, cb)], name="alias"),
            ]
        )
        == {}
    )


def test_validation_targets_freeze_destination_runtime_not_nearby_grid():
    import plan_kpack_tactic_validation as plan

    p = parent(route="fq-grouped", persistent=1)
    ctx = context("fq-grouped", rows=[129, 8, 0])
    raw_parent = {k: p[k] for k in model.PARENT_FIELDS}
    t = model.runtime_choices(p, ctx)[0]
    c = plan.target_config(raw_parent, t, ctx["problem"])
    assert policy.resolve_grid(c, ctx["problem"]) == model.tile_count(p, ctx)
    assert c["symbol"] == p["symbol"]


def test_published_tactic_suite_and_model_are_frozen_and_compile_free():
    import run_kpack_policy_validation as runner

    root = Path(__file__).resolve().parents[1]
    mp = root / "policies/kpack_zw810_tactics.json"
    m = json.loads(mp.read_text())
    s = json.loads((root / "policies/kpack_zw810_tactics.validation.json").read_text())
    model.validate_model(m)
    runner.validate_suite(s)
    assert s["policy_sha256"] == policy.digest(m)
    assert s["model_digest"] == m["model_digest"]
    assert s["denominator"]["new_parent_union"] == 0
    assert s["denominator"]["cache_misses"] == 102
    assert s["denominator"]["new_M_requests"] == 110
    assert s["denominator"]["route_workloads"] == 212
    for r in s["plan"]["requests"]:
        ctx = evidence.request_context(r, s["plan"])
        for target in s["targets"].get(r["id"], []):
            runtime = s["runtime_recipes"][r["id"]][target["config_id"]]
            assert runtime in model.runtime_choices(
                m["parents"][runtime["symbol"]], ctx
            )

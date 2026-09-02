"""Contract tests for the canonical K-pack two-route optimum planner."""

from __future__ import annotations

import copy
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plan_fq_kpack_route_optimal as planner  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    value = planner.materialize()
    planner.validate_plan(value)
    return value


@pytest.fixture(scope="module")
def discovery(plan):
    value = planner._synthetic_static_discovery(plan)
    planner.validate_static_discovery(plan, value)
    return value


@pytest.fixture(scope="module")
def census(plan, discovery):
    value = planner._synthetic_census(plan, discovery)
    planner.validate_census(plan, discovery, value)
    return value


@pytest.fixture(scope="module")
def replay(plan, discovery, census):
    value = planner._synthetic_shipping_replay(plan, census)
    planner.validate_shipping_replay(plan, discovery, census, value)
    return value


def test_exact_five_format_real_and_historical_denominator(plan):
    assert [row["qtype"] for row in plan["formats"]] == [10, 11, 12, 13, 14]
    assert plan["routes"] == ["scalefirst", "fully-quantized"]
    assert plan["workload_denominator"] == {
        "dense_real": 143,
        "grouped_real": 52,
        "grouped_router_controls": 24,
        "q4_historical_anchor_only": 286,
        "workloads_per_format": 219,
        "format_workload_cells": 1381,
        "route_inventory_queries": 2762,
        "dense_families": 11,
        "grouped_families": 4,
        "dense_m": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        "grouped_tokens": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        "grouped_control_profiles": [
            "balanced", "hot-skewed", "permutation-a", "permutation-b",
            "sparse-empty", "tilem-boundary",
        ],
    }
    assert sum(cell["source_class"] == "historical-anchor"
               for cell in plan["cells"]) == 286


def test_only_canonical_kpack_descriptors_are_present(plan):
    arrangements = {row["qtype"]: row["arrangement"] for row in plan["formats"]}
    assert arrangements[12]["layout"] == 1
    assert arrangements[12]["mapping_id"] == planner.MAPPING_Q4
    for qtype in (10, 11, 13, 14):
        assert arrangements[qtype]["layout"] == 2
        assert arrangements[qtype]["mapping_id"] == planner.MAPPING_KPACK
    assert all(row["artifact_tile_k"] == 0 for row in arrangements.values())


def test_generator_not_shipping_inventory_owns_discovery(plan):
    contract = plan["candidate_discovery_contract"]
    assert contract["candidate_names_in_plan"] is False
    assert contract["candidate_authority"] == "EXHAUSTIVE_GENERATOR_MANIFEST"
    assert contract["shipping_inventory_is_not_discovery_authority"] is True
    assert contract["static_axis_denominator"] == {
        "all_cartesian_points_recorded": True,
        "constraint_failures_are_static_reject_rows": True,
        "axis_values_and_constraint_program_are_hashed": True,
    }
    assert plan["shipping_replay_contract"]["candidate_source"] == (
        "SELECTED_FROM_COMPILED_DISCOVERY_ONLY")


def test_safe_screen_reducer_model_and_prepass_are_explicit(plan):
    screen = plan["measurement_contract"]["screen"]
    assert screen["all_runtime_candidates"] is True
    assert screen["top_n"] is None
    assert screen["point_estimate_pruning"] is False
    assert screen["uncertain_action"] == "RETAIN"
    split = plan["measurement_contract"]["split_k"]
    assert split == {
        "producer": "MEASURED_PER_CANDIDATE",
        "reducer": "ONE_VERSIONED_MODEL_PER_M_N_S_DTYPE",
        "model_uncertainty": "INCLUDED_IN_REGRET_INTERVAL",
        "per_candidate_reducer_model": False,
        "selected_rows_real_reducer_sanity": True,
        "workspace_init": "MEASURE_IF_REQUIRED_BY_EACH_CALL",
    }
    assert plan["analysis_contract"]["model_uncertainty_is_additive"] is True
    accounting = plan["measurement_contract"]["prepass_accounting"]
    assert accounting["cache_ready_is_explicit"] is True
    assert "PREPASS_US" in accounting["break_even_reuse"]


def test_grouped_hidden_features_are_diagnostics_only(plan):
    public = set(plan["selector_contract"]["features"]["grouped"])
    forbidden = set(plan["selector_contract"]["forbidden_grouped_features"])
    assert public.isdisjoint(forbidden)
    controls = [cell for cell in plan["cells"]
                if cell["operator"] == "grouped" and
                cell["source_class"] == "router-control"]
    assert len(controls) == 5 * 24
    assert all("active" not in cell["public_problem"] for cell in controls)


@pytest.mark.parametrize("plant", [
    "xplane", "missing-cell", "top-n", "hidden", "reducer", "output",
])
def test_plan_negative_controls_fail_closed(plan, plant):
    value = copy.deepcopy(plan)
    if plant == "xplane":
        value["formats"][0]["arrangement"]["layout"] = 0
    elif plant == "missing-cell":
        value["cells"].pop()
    elif plant == "top-n":
        value["measurement_contract"]["screen"]["top_n"] = 8
    elif plant == "hidden":
        value["selector_contract"]["features"]["grouped"].append("rows_hash")
    elif plant == "reducer":
        value["measurement_contract"]["split_k"]["per_candidate_reducer_model"] = True
    else:
        value["output_contract"]["required"].pop()
    with pytest.raises(planner.PlanError):
        planner.validate_plan(value)


def test_static_discovery_is_cartesian_and_anchor_complete(plan, discovery):
    generated = planner.validate_static_discovery(plan, discovery)
    assert len(discovery["classes"]) == 20
    assert all(len(ids) == 5 for ids in generated.values())
    assert all(item["reported_raw_count"] == 6 for item in discovery["classes"])
    campaigns = {row["campaign"]: row
                 for row in plan["historical_anchor_contract"]["campaigns"]}
    assert campaigns["fq-five-format-dense"]["qtypes"] == [10, 11, 13, 14]
    assert campaigns["a04-q4-dense-policy"]["qtypes"] == [12]
    q4_sf_contract = campaigns["q4-scalefirst-real-shapes"]
    assert q4_sf_contract["required_role_counts_per_cell"] == {
        "HISTORICAL_CANDIDATE_ANCHOR": 1}
    assert q4_sf_contract["evidence_status"] == "MISSING_RAW_RESULTS_REMEASURE_REQUIRED"
    assert "raw Q4 ScaleFirst" in q4_sf_contract["missing_source_todo"]
    for campaign in discovery["anchor_campaigns"]:
        contract = campaigns[campaign["campaign"]]
        by_cell = {}
        for row in campaign["anchors"]:
            by_cell.setdefault(row["cell_key"], Counter())[row["role"]] += 1
        assert len(by_cell) == contract["expected_source_cell_count"]
        assert all(value == Counter(contract["required_role_counts_per_cell"])
                   for value in by_cell.values())
        assert campaign["anchor_set_sha256"] == planner.digest(campaign["anchors"])
    a04 = next(row for row in discovery["anchor_campaigns"]
               if row["campaign"] == "a04-q4-dense-policy")
    assert {row["role"] for row in a04["anchors"]} == {
        "HISTORICAL_CANDIDATE_ANCHOR"}


@pytest.mark.parametrize("plant", [
    "missing-class", "cartesian-count", "authority", "missing-anchor",
    "static-reject-anchor", "false-q12-measured-policy", "duplicate-axis",
    "middle-anchor-shift",
])
def test_static_discovery_negative_controls_fail_closed(plan, discovery, plant):
    value = copy.deepcopy(discovery)
    if plant == "missing-class":
        value["classes"].pop()
    elif plant == "cartesian-count":
        value["classes"][0]["reported_raw_count"] -= 1
    elif plant == "authority":
        value["generator_authorities"][0]["sha256"] = "0" * 64
    elif plant == "missing-anchor":
        value["anchor_campaigns"][0]["anchors"].pop()
    elif plant == "static-reject-anchor":
        campaign = value["anchor_campaigns"][0]
        anchor = campaign["anchors"][0]
        parent = planner._static_candidate_id(
            anchor["qtype"], anchor["operator"], anchor["route"], "static-negative")
        anchor["static_candidate_id"] = parent
        anchor["candidate_id"] = f"{parent}::{anchor['runtime_variant_id']}"
        campaign["anchor_set_sha256"] = planner.digest(campaign["anchors"])
    elif plant == "false-q12-measured-policy":
        campaign = next(row for row in value["anchor_campaigns"]
                        if row["campaign"] == "fq-five-format-dense")
        anchor = campaign["anchors"][0]
        cell = next(row for row in plan["cells"]
                    if row["qtype"] == 12 and row["operator"] == "dense")
        anchor.update({
            "cell_key": cell["cell_key"], "qtype": 12,
            "static_candidate_id": planner._static_candidate_id(
                12, "dense", "fully-quantized", "candidate-0"),
        })
        anchor["candidate_id"] = (
            f"{anchor['static_candidate_id']}::{anchor['runtime_variant_id']}")
        campaign["anchor_set_sha256"] = planner.digest(campaign["anchors"])
    elif plant == "duplicate-axis":
        item = value["classes"][0]
        item["rows"][1]["axes"] = copy.deepcopy(item["rows"][0]["axes"])
    else:
        campaign = next(row for row in value["anchor_campaigns"]
                        if row["campaign"] == "q4-scalefirst-real-shapes")
        runner = campaign["anchors"][len(campaign["anchors"]) // 2]
        other = next(row for row in campaign["anchors"]
                     if row["cell_key"] != runner["cell_key"])
        runner.update({
            "cell_key": other["cell_key"], "qtype": other["qtype"],
            "operator": other["operator"],
        })
        runner["source_record_key"] += "/shifted"
        campaign["source_record_keys_sha256"] = planner.digest(sorted(
            row["source_record_key"] for row in campaign["anchors"]))
        campaign["anchor_set_sha256"] = planner.digest(campaign["anchors"])
    with pytest.raises(planner.PlanError):
        planner.validate_static_discovery(plan, value)


def test_compiled_discovery_accounts_for_every_generated_candidate(
        plan, discovery, census):
    planner.validate_census(plan, discovery, census)
    assert len(census["cells"]) == 2762
    assert all(row["generated_candidate_count"] == 5 for row in census["cells"])
    assert all(len(row["runtime_variant_expansions"]) + len(row["rejections"]) == 5
               for row in census["cells"])
    assert all(len(row["candidates"]) == 6 for row in census["cells"])

    unavailable = copy.deepcopy(census)
    anchored_pairs = {
        (anchor["cell_key"], anchor["route"])
        for campaign in discovery["anchor_campaigns"]
        for anchor in campaign["anchors"]
    }
    row = next(item for item in unavailable["cells"]
               if (item["cell_key"], item["route"]) not in anchored_pairs)
    rejected = [{"static_candidate_id": item["static_candidate_id"],
                 "stage": "CAN_IMPLEMENT", "reason": "STRUCTURAL_TEST_REJECT"}
                for item in row["runtime_variant_expansions"]]
    row.update({
        "availability": "STRUCTURAL_UNAVAILABLE",
        "unavailable_reason": "NO_CAN_IMPLEMENT_CANDIDATE",
        "first_query_count": 0, "capacity": 0, "second_query_count": 0,
        "response_sha256": planner.digest([]), "candidates": [],
        "runtime_variant_expansions": [], "rejections": rejected,
    })
    planner.validate_census(plan, discovery, unavailable)


@pytest.mark.parametrize("plant", [
    "missing", "truncated", "hidden-arg", "lost-generated", "response",
    "bad-rejection", "missing-grid-variant",
])
def test_compiled_discovery_negative_controls_fail_closed(
        plan, discovery, census, plant):
    value = copy.deepcopy(census)
    row = value["cells"][0]
    if plant == "missing":
        value["cells"].pop()
    elif plant == "truncated":
        row["second_query_count"] -= 1
    elif plant == "hidden-arg":
        row["query_arguments"]["public_problem"]["active"] = 1
    elif plant in ("lost-generated", "missing-grid-variant"):
        row["candidates"].pop()
        count = len(row["candidates"])
        row["first_query_count"] = row["capacity"] = row["second_query_count"] = count
        row["response_sha256"] = planner.digest(row["candidates"])
    elif plant == "response":
        row["response_sha256"] = "0" * 64
    else:
        row["rejections"] = [{
            "static_candidate_id": "not-generated", "stage": "CAN_IMPLEMENT",
            "reason": "plant",
        }]
    with pytest.raises(planner.PlanError):
        planner.validate_census(plan, discovery, value)


def test_historical_anchor_can_implement_failure_is_fatal(plan, discovery, census):
    value = copy.deepcopy(census)
    anchor = discovery["anchor_campaigns"][0]["anchors"][0]
    row = next(item for item in value["cells"]
               if item["cell_key"] == anchor["cell_key"] and
               item["route"] == anchor["route"])
    candidate = next(item for item in row["candidates"]
                     if item["candidate_id"] == anchor["candidate_id"])
    row["candidates"].remove(candidate)
    expansion = next(item for item in row["runtime_variant_expansions"]
                     if item["static_candidate_id"] == anchor["static_candidate_id"])
    point = next(item for item in expansion["points"]
                 if item["runtime_variant_id"] == anchor["runtime_variant_id"])
    point["admission"] = "CAN_IMPLEMENT_REJECT"
    point["reason"] = "anchor plant"
    expansion["runtime_space_authority"]["response_sha256"] = planner.digest(
        expansion["points"])
    row["first_query_count"] = row["capacity"] = row["second_query_count"] = len(row["candidates"])
    row["response_sha256"] = planner.digest(row["candidates"])
    with pytest.raises(planner.PlanError, match="historical anchor"):
        planner.validate_census(plan, discovery, value)


def test_splitk_candidates_share_one_model_per_m_n_s_dtype(plan, discovery, census):
    value = copy.deepcopy(census)
    cell_by_key = {cell["cell_key"]: cell for cell in plan["cells"]}
    rows = [item for item in value["cells"]
            if item["operator"] == "dense" and item["route"] == "fully-quantized"]
    first = rows[0]
    first_problem = cell_by_key[first["cell_key"]]["public_problem"]
    m, n = first_problem["m"], first_problem["n"]
    second = next(item for item in rows[1:]
                  if cell_by_key[item["cell_key"]]["public_problem"]["m"] == m and
                  cell_by_key[item["cell_key"]]["public_problem"]["n"] == n)
    model = {
        "mode": "VERSIONED_SHARED_MODEL",
        "model_key": f"m{m}-n{n}-s4-float32",
        "model_version": "reducer-v1",
        "model_sha256": "c" * 64,
        "uncertainty_in_regret": True,
        "selected_real_reducer_sanity_required": True,
    }
    for row in (first, second):
        for candidate in row["candidates"]:
            candidate.update({
                "algorithm": "FULLY_QUANTIZED_SPLITK",
                "split_k_slices": 4,
                "timing_scope": "MEASURED_PRODUCER_PLUS_VERSIONED_REDUCER_MODEL",
                "reducer": copy.deepcopy(model),
            })
        row["response_sha256"] = planner.digest(row["candidates"])
    planner.validate_census(plan, discovery, value)

    second["candidates"][0]["reducer"]["model_sha256"] = "d" * 64
    second["response_sha256"] = planner.digest(second["candidates"])
    with pytest.raises(planner.PlanError, match="per-candidate reducer"):
        planner.validate_census(plan, discovery, value)


def test_shipping_replay_uses_product_inventory_only(plan, discovery, census, replay):
    planner.validate_shipping_replay(plan, discovery, census, replay)
    value = copy.deepcopy(replay)
    cell = next(row for row in plan["cells"] if row["cell_key"] == value["cells"][0]["cell_key"])
    value["cells"][0]["query_symbol"] = planner.DISCOVERY_QUERY_SYMBOLS[
        (cell["operator"], value["cells"][0]["route"])]
    with pytest.raises(planner.PlanError):
        planner.validate_shipping_replay(plan, discovery, census, value)


def test_cli_materialize_and_three_stage_validate(
        tmp_path, plan, discovery, census, replay):
    plan_path = tmp_path / "plan.json"
    discovery_path = tmp_path / "static.json"
    census_path = tmp_path / "compiled.json"
    replay_path = tmp_path / "replay.json"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/plan_fq_kpack_route_optimal.py"),
         "materialize", "--output", str(plan_path)], check=True)
    discovery_path.write_text(json.dumps(discovery))
    census_path.write_text(json.dumps(census))
    replay_path.write_text(json.dumps(replay))
    for command, extra in (
        ("validate-static", ["--discovery", str(discovery_path)]),
        ("validate-compiled", ["--discovery", str(discovery_path),
                               "--census", str(census_path)]),
        ("validate-shipping", ["--discovery", str(discovery_path),
                               "--census", str(census_path),
                               "--replay", str(replay_path)]),
    ):
        subprocess.run(
            [sys.executable, str(ROOT / "tools/plan_fq_kpack_route_optimal.py"),
             command, "--plan", str(plan_path), *extra], check=True)
    assert json.loads(plan_path.read_text())["schema"] == planner.PLAN_SCHEMA

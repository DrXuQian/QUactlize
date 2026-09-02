#!/usr/bin/env python3
"""Plan and validate an exhaustive canonical K-pack route sweep.

This module owns contracts, not device execution.  It deliberately contains
no configuration names.  An exhaustive generator manifest owns the static
denominator, generated discovery shards account for build/canImplement
admission, and the product runtime inventory is used only to replay a selected
policy.  Consequently a newly added provider, scheduler, grid, or Split-K row
cannot disappear behind a stale shipping library or planner constant.

"Optimal" below always means optimal inside the complete, manifest-bound
discovery denominator for one exact binary/device/workload denominator.  A
noisy screen may retain extra candidates, but may never prune by rank or top-N.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fq_grouped_multi_router as router_controls  # noqa: E402
import plan_fq_kquant_kpack_perf as real_inventory  # noqa: E402
import plan_fq_kquant_policy_v2_real_families as a04_inventory  # noqa: E402


PLAN_SCHEMA = "quactlize.fq-kpack-route-optimal-plan.v1"
STATIC_DISCOVERY_SCHEMA = "quactlize.fq-kpack-route-static-discovery.v1"
COMPILED_CENSUS_SCHEMA = "quactlize.fq-kpack-route-compiled-discovery-census.v1"
SHIPPING_REPLAY_SCHEMA = "quactlize.fq-kpack-route-shipping-replay.v1"
PLAN_PROFILE = "canonical-kpack-scalefirst-vs-fully-quantized"
ROUTES = ("scalefirst", "fully-quantized")
QTYPE_ORDER = (10, 11, 12, 13, 14)
MAPPING_KPACK = "0x514b504b54000001"
MAPPING_Q4 = "0x51344b5034540001"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")

FORMATS = {
    10: ("Q2_K", 2, 0, 16, 128, 256, 2, 2, 128, MAPPING_KPACK),
    11: ("Q3_K", 2, 1, 16, 256, 256, 3, 2, 256, MAPPING_KPACK),
    12: ("Q4_K", 4, 0, 32, 64, 256, 0, 1, 64, MAPPING_Q4),
    13: ("Q5_K", 4, 1, 32, 256, 256, 1, 2, 256, MAPPING_KPACK),
    14: ("Q6_K", 4, 2, 16, 128, 128, 4, 2, 128, MAPPING_KPACK),
}

DISCOVERY_QUERY_SYMBOLS = {
    ("dense", "scalefirst"):
        "fq_discovery_list_valid_dense_scalefirst_candidates_v1",
    ("dense", "fully-quantized"):
        "fq_discovery_list_valid_dense_fully_quantized_candidates_v1",
    ("grouped", "scalefirst"):
        "fq_discovery_list_valid_grouped_scalefirst_candidates_v1",
    ("grouped", "fully-quantized"):
        "fq_discovery_list_valid_grouped_fully_quantized_candidates_v1",
}
SHIPPING_QUERY_SYMBOLS = {
    ("dense", "scalefirst"):
        "quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2",
    ("dense", "fully-quantized"):
        "quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2",
    ("grouped", "scalefirst"):
        "quactlize_ppu_grouped_lowbit_config_valid_for_arrangement_v2",
    ("grouped", "fully-quantized"):
        "quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2",
}

SELECTOR_FEATURES = {
    "dense": ["qtype", "m", "n", "k", "group_size", "cache_ready", "reuse_class"],
    "grouped": [
        "qtype", "total_rows", "max_rows", "experts", "n", "k",
        "group_size", "cache_ready", "reuse_class",
    ],
}
FORBIDDEN_GROUPED_FEATURES = [
    "active", "zero", "router", "profile", "rows", "rows_hash",
    "rows_sha256", "work_tm16", "work_tm32", "work_tm128",
]

ALGORITHMS = {
    "SCALEFIRST_NONPERSISTENT",
    "SCALEFIRST_PERSISTENT",
    "SCALEFIRST_CUDA",
    "FULLY_QUANTIZED_S1",
    "FULLY_QUANTIZED_SPLITK",
    "FULLY_QUANTIZED_BC",
}
FAMILIES = {"TENSOR_CORE", "CUDA_CORE"}
TIMING_SCOPES = {
    "FULL_PRODUCT_E2E",
    "MEASURED_PRODUCER_PLUS_VERSIONED_REDUCER_MODEL",
}

ANCHOR_CAMPAIGNS = {
    "q4-scalefirst-real-shapes": {
        "operator": "dense",
        "route": "scalefirst",
        "qtypes": [12],
        # Only the policies and one declared anchor symbol are tracked.  The
        # raw result bundles that would prove per-cell winner/runner are not in
        # the repository, so this is a remeasurement challenge, not a timing
        # claim.
        "required_role_counts_per_cell": {"HISTORICAL_CANDIDATE_ANCHOR": 1},
        "expected_source_cell_count": 16,
        "source_cell_selector": "Q4_DENSE_M64_2048_4096_X_FIVE_FAMILIES_PLUS_LEGACY_4096",
        "source_result_schema": "POLICY_ONLY_NO_RESULT_AUTHORITY",
        "extractor": "extract_scalefirst_q4_declared_anchor_geometry_v1",
        "evidence_status": "MISSING_RAW_RESULTS_REMEASURE_REQUIRED",
        "missing_source_todo": (
            "Import the raw Q4 ScaleFirst confirm bundles, bind every source "
            "file hash, and only then promote per-cell KPACK winner/runner labels."),
        "authorities": [
            "benchmarks/scalefirst_q4k_pruned_policy.json",
            "benchmarks/scalefirst_q4k_real_shapes_pruned_policy.json",
        ],
    },
    "fq-five-format-dense": {
        "operator": "dense",
        "route": "fully-quantized",
        # The measured dense policy has 44 families: eleven families for each
        # of Q2/Q3/Q5/Q6.  Q4 dense was intentionally absent and is covered by
        # the separate A04 campaign below.
        "qtypes": [10, 11, 13, 14],
        "required_role_counts_per_cell": {"HISTORICAL_GEOMETRY_ANCHOR": 1},
        "expected_source_cell_count": 572,
        "source_cell_selector": "DENSE_REAL_143_X_Q2_Q3_Q5_Q6",
        "source_result_schema": "quactlize.fq-kquant-kpack-perf-result.v3",
        "extractor": "extract_fq_kpack_dense_geometry_per_row_v3",
        "evidence_status": "GEOMETRY_ONLY_NO_KPACK_RUNNER_CLAIM",
        "missing_source_todo": None,
        "authorities": [
            "quactlize/include/ppu_kquant_measured_policy_data.inc",
            "tools/generate_fq_kquant_measured_policy.py",
        ],
    },
    "fq-five-format-grouped": {
        "operator": "grouped",
        "route": "fully-quantized",
        "qtypes": list(QTYPE_ORDER),
        "required_role_counts_per_cell": {"HISTORICAL_GEOMETRY_ANCHOR": 1},
        "expected_source_cell_count": 260,
        "source_cell_selector": "GROUPED_REAL_52_X_FIVE_FORMATS",
        "source_result_schema": "quactlize.fq-kquant-kpack-perf-result.v3",
        "extractor": "extract_fq_kpack_grouped_geometry_per_row_v3",
        "evidence_status": "GEOMETRY_ONLY_NO_KPACK_RUNNER_CLAIM",
        "missing_source_todo": None,
        "authorities": [
            "tools/plan_fq_kquant_kpack_perf.py",
            "tools/fit_fq_kquant_config_heuristic.py",
        ],
    },
    "a04-q4-dense-policy": {
        "operator": "dense",
        "route": "fully-quantized",
        "qtypes": [12],
        # A04 only compiled the five proposed rows; its real-family box sweep
        # was never run, so none may be labelled winner or runner.
        "required_role_counts_per_cell": {"HISTORICAL_CANDIDATE_ANCHOR": 5},
        "expected_source_cell_count": 320,
        "source_cell_selector": "Q4_DENSE_FIVE_FAMILIES_X_M1_THROUGH_64",
        "source_result_schema": a04_inventory.SCHEMA,
        "extractor": "extract_a04_all_planned_candidate_geometries_v1",
        "evidence_status": "PLAN_ONLY_ALL_CANDIDATES_REQUIRE_REMEASUREMENT",
        "missing_source_todo": None,
        "authorities": [
            "tools/plan_fq_kquant_policy_v2.py",
            "tools/plan_fq_kquant_policy_v2_real_families.py",
            "tools/adjudicate_fq_kquant_policy_v2.py",
        ],
    },
}


class PlanError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _format_rows() -> list[dict[str, Any]]:
    rows = []
    for qtype in QTYPE_ORDER:
        (name, low, high, group, sf_tk, fq_tk, packed, layout,
         transport, mapping) = FORMATS[qtype]
        rows.append({
            "qtype": qtype,
            "name": name,
            "low_bits": low,
            "high_bits": high,
            "group_size": group,
            "scale_first_tile_k": sf_tk,
            "fully_quantized_tile_k": fq_tk,
            "packed_format": packed,
            "arrangement": {
                "version": 2,
                "layout": layout,
                "bits": low,
                "high_bits": high,
                "artifact_tile_k": 0,
                "transport_tile_k": transport,
                "group_size": group,
                "reserved": 0,
                "mapping_id": mapping,
            },
        })
    return rows


def _dense_workloads(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in source["dense"]:
        rows.append({
            "workload_key": item["key"],
            "operator": "dense",
            "public_problem": {
                "m": int(item["m"]), "n": int(item["n"]),
                "k": int(item["k"]),
            },
            "diagnostics": {"sources": list(item["sources"])},
            "source_class": "real-inventory",
        })
    return rows


def _grouped_real_workloads(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in source["grouped"]:
        rows.append({
            "workload_key": item["key"],
            "operator": "grouped",
            "public_problem": {
                "total_rows": int(item["total_rows"]),
                "max_rows": int(item["max_rows"]),
                "experts": int(item["experts"]),
                "n": int(item["n"]), "k": int(item["k"]),
            },
            "diagnostics": {
                "tokens": int(item["tokens"]),
                "topk": int(item["topk"]),
                "active": int(item["active"]),
                "zero": int(item["zero"]),
                "router": item["router"],
                "sources": list(item["sources"]),
            },
            "source_class": "real-inventory",
        })
    return rows


def _grouped_control_workloads() -> list[dict[str, Any]]:
    _, grouped_families = real_inventory.source_families()
    profiles = router_controls.materialize()
    rows = []
    for (n, k), sources in sorted(grouped_families.items()):
        for name, profile in profiles.items():
            rows.append({
                "workload_key": f"grouped_control_{name}_n{n}_k{k}_e{profile['experts']}",
                "operator": "grouped",
                "public_problem": {
                    "total_rows": int(profile["total_rows"]),
                    "max_rows": int(profile["max_rows"]),
                    "experts": int(profile["experts"]),
                    "n": int(n), "k": int(k),
                },
                "diagnostics": {
                    "profile": name,
                    "active": int(profile["active"]),
                    "zero": int(profile["zero"]),
                    "work_tm16": int(profile["work_tm16"]),
                    "work_tm32": int(profile["work_tm32"]),
                    "work_tm128": int(profile["work_tm128"]),
                    "rows_sha256": profile["rows_sha256"],
                    "rows_hash": profile["rows_hash"],
                    "sources": sorted(sources),
                },
                "source_class": "router-control",
            })
    return rows


def _q4_historical_anchor_workloads(
        existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add historical Q4 cells that are outside the power-of-two inventory.

    A04 measured every integer M in [1,64] for five real families, whereas the
    general heuristic inventory contains only its thirteen registered M
    values.  The original ScaleFirst anchor also owns one 4096x4096 shape not
    present in those eleven families.  Historical anchors are a denominator
    subset, so their workload cannot be left for candidate matching to invent.
    """
    present = {
        (row["operator"], tuple(row["public_problem"].get(x) for x in ("m", "n", "k")))
        for row in existing if row["operator"] == "dense"
    }
    rows = []
    for n, k in a04_inventory.FAMILIES:
        for m in a04_inventory.M_VALUES:
            identity = ("dense", (m, n, k))
            if identity in present:
                continue
            present.add(identity)
            rows.append({
                "workload_key": f"a04_dense_m{m}_n{n}_k{k}",
                "operator": "dense",
                "public_problem": {"m": m, "n": n, "k": k},
                "diagnostics": {
                    "historical_campaign": "a04-q4-dense-policy",
                    "source_family": a04_inventory.family_identity(n, k),
                },
                "source_class": "historical-anchor",
            })
    sf_policy = json.loads(
        (ROOT / "benchmarks/scalefirst_q4k_pruned_policy.json").read_text(
            encoding="utf-8"))
    m, n, k = map(int, sf_policy["shape"])
    identity = ("dense", (m, n, k))
    if identity not in present:
        rows.append({
            "workload_key": f"q4_sf_historical_m{m}_n{n}_k{k}",
            "operator": "dense",
            "public_problem": {"m": m, "n": n, "k": k},
            "diagnostics": {
                "historical_campaign": "q4-scalefirst-real-shapes",
                "anchor_symbol": sf_policy["anchor_symbol"],
            },
            "source_class": "historical-anchor",
        })
    return rows


def _cells(formats: list[dict[str, Any]], workloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for fmt in formats:
        for workload in workloads:
            public = dict(workload["public_problem"])
            public.update({"qtype": fmt["qtype"], "group_size": fmt["group_size"]})
            result.append({
                "cell_key": f"q{fmt['qtype']}/{workload['operator']}/{workload['workload_key']}",
                "qtype": fmt["qtype"],
                "format": fmt["name"],
                "operator": workload["operator"],
                "workload_key": workload["workload_key"],
                "source_class": workload["source_class"],
                "public_problem": public,
                "diagnostics": copy.deepcopy(workload["diagnostics"]),
                "arrangement": copy.deepcopy(fmt["arrangement"]),
            })
    return result


def _historical_cell_keys(
        cells: list[dict[str, Any]], campaign: str) -> list[str]:
    """Return the exact historical source denominator for one campaign."""
    if campaign == "q4-scalefirst-real-shapes":
        families = set(a04_inventory.FAMILIES)
        selected = [
            cell for cell in cells
            if cell["qtype"] == 12 and cell["operator"] == "dense" and (
                (cell["public_problem"]["m"] in (64, 2048, 4096) and
                 (cell["public_problem"]["n"], cell["public_problem"]["k"])
                 in families) or
                cell["diagnostics"].get("historical_campaign") == campaign)
        ]
    elif campaign == "fq-five-format-dense":
        selected = [
            cell for cell in cells
            if cell["qtype"] in (10, 11, 13, 14) and
            cell["operator"] == "dense" and cell["source_class"] == "real-inventory"
        ]
    elif campaign == "fq-five-format-grouped":
        selected = [
            cell for cell in cells
            if cell["qtype"] in QTYPE_ORDER and cell["operator"] == "grouped" and
            cell["source_class"] == "real-inventory"
        ]
    elif campaign == "a04-q4-dense-policy":
        families = set(a04_inventory.FAMILIES)
        selected = [
            cell for cell in cells
            if cell["qtype"] == 12 and cell["operator"] == "dense" and
            1 <= cell["public_problem"]["m"] <= 64 and
            (cell["public_problem"]["n"], cell["public_problem"]["k"]) in families
        ]
    else:
        raise PlanError(f"unknown historical campaign {campaign}")
    keys = sorted(cell["cell_key"] for cell in selected)
    if len(keys) != len(set(keys)):
        raise PlanError(f"historical campaign {campaign} has duplicate cells")
    return keys


def materialize() -> dict[str, Any]:
    source = real_inventory.materialize("heuristic")
    formats = _format_rows()
    dense = _dense_workloads(source)
    grouped_real = _grouped_real_workloads(source)
    grouped_controls = _grouped_control_workloads()
    workloads = dense + grouped_real + grouped_controls
    cells = _cells(formats, workloads)
    q4 = next(row for row in formats if row["qtype"] == 12)
    q4_anchor_workloads = _q4_historical_anchor_workloads(workloads)
    cells.extend(_cells([q4], q4_anchor_workloads))
    anchor_campaigns = []
    for campaign, contract in sorted(ANCHOR_CAMPAIGNS.items()):
        historical_cells = _historical_cell_keys(cells, campaign)
        if len(historical_cells) != contract["expected_source_cell_count"]:
            raise PlanError(f"historical campaign {campaign} source denominator differs")
        anchor_campaigns.append({
            "campaign": campaign,
            "operator": contract["operator"],
            "route": contract["route"],
            "qtypes": contract["qtypes"],
            "source_authorities": [
                {"path": path, "sha256": file_sha(ROOT / path)}
                for path in contract["authorities"]
            ],
            "source_cell_selector": contract["source_cell_selector"],
            "source_result_schema": contract["source_result_schema"],
            "extractor": contract["extractor"],
            "evidence_status": contract["evidence_status"],
            "missing_source_todo": contract["missing_source_todo"],
            "expected_source_cell_count": contract["expected_source_cell_count"],
            "expected_source_cells_sha256": digest(historical_cells),
            "required_role_counts_per_cell": contract["required_role_counts_per_cell"],
            "expected_anchor_count": (
                contract["expected_source_cell_count"] *
                sum(contract["required_role_counts_per_cell"].values())),
        })
    return {
        "schema": PLAN_SCHEMA,
        "profile": PLAN_PROFILE,
        "scope": "five-format-canonical-kpack-dense-grouped-route-optimum",
        "authorities": {
            "tools/plan_fq_kquant_kpack_perf.py": file_sha(
                ROOT / "tools/plan_fq_kquant_kpack_perf.py"),
            "tools/fq_grouped_multi_router.py": file_sha(
                ROOT / "tools/fq_grouped_multi_router.py"),
            "benchmarks/workloads.py": file_sha(ROOT / "benchmarks/workloads.py"),
            "benchmarks/moe_router_fixture.hpp": file_sha(
                ROOT / "benchmarks/moe_router_fixture.hpp"),
            "quactlize/include/ppu_format_config.inc": file_sha(
                ROOT / "quactlize/include/ppu_format_config.inc"),
        },
        "formats": formats,
        "routes": list(ROUTES),
        "workload_denominator": {
            "dense_real": len(dense),
            "grouped_real": len(grouped_real),
            "grouped_router_controls": len(grouped_controls),
            "q4_historical_anchor_only": len(q4_anchor_workloads),
            "workloads_per_format": len(workloads),
            "format_workload_cells": len(cells),
            "route_inventory_queries": len(cells) * len(ROUTES),
            "dense_families": source["dense_families"],
            "grouped_families": source["grouped_families"],
            "dense_m": source["dense_m"],
            "grouped_tokens": source["grouped_tokens"],
            "grouped_control_profiles": sorted(router_controls.materialize()),
        },
        "cells": cells,
        "candidate_discovery_contract": {
            "static_schema": STATIC_DISCOVERY_SCHEMA,
            "compiled_schema": COMPILED_CENSUS_SCHEMA,
            "candidate_names_in_plan": False,
            "candidate_authority": "EXHAUSTIVE_GENERATOR_MANIFEST",
            "shipping_inventory_is_not_discovery_authority": True,
            "static_generator_authorities": {
                "scalefirst": [
                    "tools/scalefirst_internal_matrix.py",
                    "tools/emit_scalefirst_internal_superset.cpp",
                ],
                "fully-quantized": [
                    "tools/fully_quantized_kpack_discovery_matrix.py",
                    "tools/gen_fully_quantized_kpack_discovery_units.py",
                    "tools/gen_fully_quantized_grouped_kpack_units.py",
                    "tools/emit_scalefirst_internal_superset.cpp",
                ],
                "shared_axes": ["quactlize/include/ppu_tactic_space.hpp"],
            },
            "static_statuses": ["GENERATED", "STATIC_REJECT"],
            "static_axis_denominator": {
                "all_cartesian_points_recorded": True,
                "constraint_failures_are_static_reject_rows": True,
                "axis_values_and_constraint_program_are_hashed": True,
            },
            "compiled_rejection_stages": ["TYPE_BUILD", "CAN_IMPLEMENT"],
            "runtime_variant_contract": {
                "parent_identity": "static_candidate_id",
                "concrete_identity": "runtime_variant_id",
                "shape_device_occupancy_expansion_is_two_call": True,
                "all_grid_provider_scheduler_variants_recorded": True,
                "variant_admission_statuses": ["ADMITTED", "CAN_IMPLEMENT_REJECT"],
                "runtime_space_authority_and_response_are_hashed": True,
            },
            "complete_two_call_query_required": True,
            "empty_available_inventory_is_error": True,
            "structural_unavailability_must_be_named": True,
            "discovery_query_symbols": {
                f"{operator}/{route}": symbol
                for (operator, route), symbol in sorted(DISCOVERY_QUERY_SYMBOLS.items())
            },
            "candidate_identity_fields": [
                "runtime_ordinal", "candidate_id", "static_candidate_id",
                "runtime_variant_id", "config_name", "family",
                "algorithm", "provider", "bchunk", "metadata_policy",
                "resolved_delivery_n",
                "scheduler", "grid_policy", "grid", "split_k_slices",
                "timing_scope", "partial_dtype", "output_dtype", "reducer",
                "runtime_valid",
            ],
        },
        "shipping_replay_contract": {
            "schema": SHIPPING_REPLAY_SCHEMA,
            "candidate_source": "SELECTED_FROM_COMPILED_DISCOVERY_ONLY",
            "query_symbols": {
                f"{operator}/{route}": symbol
                for (operator, route), symbol in sorted(SHIPPING_QUERY_SYMBOLS.items())
            },
            "unknown_policy_is_explicit": True,
            "selected_candidate_must_replay": True,
            "selected_real_reducer_sanity_required": True,
        },
        "historical_anchor_contract": {
            "campaigns": anchor_campaigns,
            "anchors_are_discovery_denominator_subset": True,
            "anchors_are_not_the_complete_candidate_set": True,
            "every_authority_anchor_required": True,
            "required_states": {
                "generated": True,
                "compiled": True,
                "static_filter_pass": True,
                "can_implement_pass": True,
                "runtime_census_match": True,
            },
            "missing_anchor_verdict": "FAIL_BEFORE_HEURISTIC",
        },
        "selector_contract": {
            "features": SELECTOR_FEATURES,
            "forbidden_grouped_features": FORBIDDEN_GROUPED_FEATURES,
            "categorical_config_only": True,
            "numeric_config_interpolation": False,
            "same_public_features_share_one_policy_leaf": True,
            "unknown_cell": "NO_MEASURED_POLICY",
        },
        "measurement_contract": {
            "correctness": {
                "screen": "FULL_OUTPUT_RAW_BITS_ZERO_BAD",
                "finalist_repeats": 256,
                "shipping_repeats": 8192,
                "prepass": "RAW_BIT_EXACT_SCALE_ZERO_FROM_SAME_PACKED_UNITS",
                "fault_controls": ["low", "high", "units", "scale", "zero"],
            },
            "screen": {
                "all_runtime_candidates": True,
                "minimum_paired_samples": 5,
                "confidence": 0.99,
                "elimination_rule": "CANDIDATE_LOWER_BOUND_GT_INCUMBENT_UPPER_BOUND_PLUS_MARGIN",
                "uncertain_action": "RETAIN",
                "top_n": None,
                "point_estimate_pruning": False,
            },
            "confirm": {
                "rounds": 3,
                "samples_per_round": 11,
                "warmups_per_round": 3,
                "order": "RECORDED_AB_BA_LATIN_SQUARE",
                "paired_log_ratio_confidence": 0.99,
                "unresolved_if_intervals_overlap": True,
            },
            "timing_boards": {
                "steady": "CACHE_READY_COMPUTE_ONLY",
                "cold_compute": "EXPLICIT_L2_FLUSH_THEN_COMPUTE",
                "first_use": "PREPASS_PLUS_COMPUTE",
                "prepass": "ONE_TIME_PER_RESIDENT_WEIGHT_TENSOR",
            },
            "prepass_accounting": {
                "offline_pack_and_h2d": "EXCLUDED_BOTH_ROUTES",
                "extra_plane_bytes": "REPORTED",
                "break_even_reuse": "CEIL(PREPASS_US/(FQ_US-SF_US))_WHEN_SF_IS_FASTER",
                "cache_ready_is_explicit": True,
            },
            "split_k": {
                "producer": "MEASURED_PER_CANDIDATE",
                "reducer": "ONE_VERSIONED_MODEL_PER_M_N_S_DTYPE",
                "model_uncertainty": "INCLUDED_IN_REGRET_INTERVAL",
                "per_candidate_reducer_model": False,
                "selected_rows_real_reducer_sanity": True,
                "workspace_init": "MEASURE_IF_REQUIRED_BY_EACH_CALL",
            },
            "cache_and_order_controls": {
                "single_device": True,
                "single_stream": True,
                "fixed_clock_identity": "RECORDED",
                "aa_same_binary_control": "REQUIRED",
                "candidate_order_seed": "RECORDED",
            },
        },
        "analysis_contract": {
            "regret_threshold_pct": 3.0,
            "regret_statistic": "MAX_99PCT_UPPER_CONFIDENCE_BOUND_PER_POLICY_LEAF",
            "model_uncertainty_is_additive": True,
            "cross_family_extrapolation": False,
            "compiled_default_is_measured_policy": False,
            "possible_verdicts": [
                "SCALEFIRST_STEADY", "FULLY_QUANTIZED",
                "CONDITIONAL_ON_CACHE", "UNRESOLVED_KEEP_CURRENT",
                "STRUCTURAL_UNAVAILABLE",
            ],
        },
        "output_contract": {
            "required": [
                "inputs/plan.json", "inputs/static-discovery.json",
                "inputs/compiled-discovery-census.json",
                "inputs/shipping-replay.json",
                "inputs/build-authority.json", "results/candidate-census.tsv",
                "results/correctness.tsv", "results/prepass.tsv",
                "results/timing-raw.tsv", "results/break-even.tsv",
                "results/summary.json", "results/summary.tsv",
                "results/route-heuristic.json", "results/unresolved.tsv",
                "results/selected-real-reducer-sanity.tsv",
                "results/result-authority.json",
            ],
            "raw_round_logs_hashed": True,
            "result_binds_binary_device_sdk": True,
        },
    }


def _assert_exact_plan_shape(value: dict[str, Any]) -> None:
    expected = materialize()
    if value != expected:
        raise PlanError("plan differs from canonical workload/format/measurement authority")
    denominator = value["workload_denominator"]
    if denominator != {
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
    }:
        raise PlanError("workload denominator differs")
    cells = value["cells"]
    if len(cells) != 1381 or len({row["cell_key"] for row in cells}) != 1381:
        raise PlanError("format/workload cell identities differ")
    if not all(
        any(row["qtype"] == q and row["operator"] == operator for row in cells)
        for q in QTYPE_ORDER for operator in ("dense", "grouped")
    ):
        raise PlanError("one format/operator denominator is absent")
    if "candidate_names" in value or value["candidate_discovery_contract"]["candidate_names_in_plan"]:
        raise PlanError("candidate names must come only from the generator manifest")
    screen = value["measurement_contract"]["screen"]
    if screen["top_n"] is not None or screen["point_estimate_pruning"]:
        raise PlanError("unsafe top-N or point-estimate screen enabled")
    split = value["measurement_contract"]["split_k"]
    if (split["reducer"] != "ONE_VERSIONED_MODEL_PER_M_N_S_DTYPE" or
            split["per_candidate_reducer_model"] or
            split["model_uncertainty"] != "INCLUDED_IN_REGRET_INTERVAL" or
            not split["selected_rows_real_reducer_sanity"]):
        raise PlanError("Split-K reducer model contract differs")
    features = value["selector_contract"]["features"]["grouped"]
    if set(features) & set(FORBIDDEN_GROUPED_FEATURES):
        raise PlanError("hidden grouped feature entered selector policy")


def validate_plan(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or value.get("schema") != PLAN_SCHEMA:
        raise PlanError("plan schema differs")
    _assert_exact_plan_shape(value)


def _candidate_fields(plan: dict[str, Any]) -> set[str]:
    return set(plan["candidate_discovery_contract"]["candidate_identity_fields"])


def _expected_query_arguments(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "public_problem": copy.deepcopy(cell["public_problem"]),
        "arrangement": copy.deepcopy(cell["arrangement"]),
    }


def _reducer_identity(
        cell: dict[str, Any], split_k_slices: int, partial_dtype: str
        ) -> tuple[tuple[int, int, int, str], str]:
    problem = cell["public_problem"]
    m = int(problem.get("m", problem.get("total_rows")))
    n = int(problem["n"])
    key = (m, n, split_k_slices, partial_dtype)
    return key, f"m{m}-n{n}-s{split_k_slices}-{partial_dtype}"


def _expected_discovery_classes() -> set[tuple[int, str, str]]:
    return {
        (qtype, operator, route)
        for qtype in QTYPE_ORDER
        for operator in ("dense", "grouped")
        for route in ROUTES
    }


def validate_static_discovery(
        plan: dict[str, Any], discovery: dict[str, Any]) -> dict[tuple[int, str, str], set[str]]:
    """Validate the exhaustive generator denominator before any device build."""
    validate_plan(plan)
    required = {
        "schema", "plan_sha256", "source_commit", "generator_authorities",
        "classes", "anchor_campaigns",
    }
    if not isinstance(discovery, dict) or set(discovery) != required:
        raise PlanError("static discovery top-level fields differ")
    if (discovery["schema"] != STATIC_DISCOVERY_SCHEMA or
            discovery["plan_sha256"] != digest(plan)):
        raise PlanError("static discovery plan authority differs")
    if (not isinstance(discovery["source_commit"], str) or
            not GIT_RE.fullmatch(discovery["source_commit"])):
        raise PlanError("static discovery source commit is malformed")
    expected_generator_paths = sorted(
        path
        for paths in plan["candidate_discovery_contract"]["static_generator_authorities"].values()
        for path in paths
    )
    authorities = discovery["generator_authorities"]
    if (not isinstance(authorities, list) or
            [row.get("path") for row in authorities if isinstance(row, dict)] !=
            expected_generator_paths):
        raise PlanError("static discovery generator authorities differ")
    for row in authorities:
        if (set(row) != {"path", "sha256"} or
                row["sha256"] != file_sha(ROOT / row["path"])):
            raise PlanError("static discovery generator digest differs")

    classes = discovery["classes"]
    if not isinstance(classes, list):
        raise PlanError("static discovery classes must be a list")
    observed = {(row.get("qtype"), row.get("operator"), row.get("route"))
                for row in classes if isinstance(row, dict)}
    if observed != _expected_discovery_classes() or len(classes) != len(observed):
        raise PlanError("static discovery class denominator differs")
    generated: dict[tuple[int, str, str], set[str]] = {}
    row_by_candidate: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for item in classes:
        if set(item) != {
                "qtype", "operator", "route", "axis_denominator",
                "reported_raw_count", "rows"}:
            raise PlanError("static discovery class fields differ")
        key = (item["qtype"], item["operator"], item["route"])
        rows = item["rows"]
        axis_denominator = item["axis_denominator"]
        if (not isinstance(axis_denominator, dict) or set(axis_denominator) != {
                "axis_names", "axis_values", "cartesian_product_count",
                "constraint_program_sha256", "generator_manifest_sha256",
        }):
            raise PlanError("static discovery axis denominator fields differ")
        names = axis_denominator["axis_names"]
        values = axis_denominator["axis_values"]
        if (not isinstance(names, list) or not names or
                len(names) != len(set(names)) or
                any(not isinstance(name, str) or not name for name in names) or
                not isinstance(values, dict) or list(values) != names or
                any(not isinstance(axis, list) or not axis or
                    len({canonical(value) for value in axis}) != len(axis)
                    for axis in values.values())):
            raise PlanError("static discovery axes are malformed")
        cartesian = 1
        for name in names:
            cartesian *= len(values[name])
        manifest_payload = {
            "qtype": item["qtype"],
            "operator": item["operator"],
            "route": item["route"],
            "axis_names": names,
            "axis_values": values,
            "constraint_program_sha256": axis_denominator["constraint_program_sha256"],
        }
        if (not SHA256_RE.fullmatch(str(axis_denominator["constraint_program_sha256"])) or
                axis_denominator["cartesian_product_count"] != cartesian or
                axis_denominator["generator_manifest_sha256"] != digest(manifest_payload)):
            raise PlanError("static discovery Cartesian authority differs")
        if (not isinstance(item["reported_raw_count"], int) or
                item["reported_raw_count"] <= 0 or
                item["reported_raw_count"] != cartesian or
                not isinstance(rows, list) or len(rows) != item["reported_raw_count"]):
            raise PlanError("static discovery raw denominator differs")
        ids: set[str] = set()
        ordinals: set[int] = set()
        axis_points: set[bytes] = set()
        generated[key] = set()
        for row in rows:
            if set(row) != {
                "generator_ordinal", "static_candidate_id", "axes",
                "static_status", "reason"
            }:
                raise PlanError("static discovery row fields differ")
            if (not isinstance(row["generator_ordinal"], int) or row["generator_ordinal"] < 0 or
                    not isinstance(row["static_candidate_id"], str) or
                    not row["static_candidate_id"] or
                    not isinstance(row["axes"], dict) or list(row["axes"]) != names or
                    any(row["axes"][name] not in values[name] for name in names)):
                raise PlanError("static discovery candidate identity is malformed")
            if (row["generator_ordinal"] in ordinals or
                    row["static_candidate_id"] in ids):
                raise PlanError("static discovery candidate identity is duplicated")
            axis_point = canonical(row["axes"])
            if axis_point in axis_points:
                raise PlanError("static discovery Cartesian point is duplicated")
            axis_points.add(axis_point)
            ordinals.add(row["generator_ordinal"])
            ids.add(row["static_candidate_id"])
            if row["static_status"] == "GENERATED":
                if row["reason"] is not None:
                    raise PlanError("generated candidate carries a rejection reason")
                generated[key].add(row["static_candidate_id"])
            elif row["static_status"] == "STATIC_REJECT":
                if not isinstance(row["reason"], str) or not row["reason"]:
                    raise PlanError("static rejection is unnamed")
            else:
                raise PlanError("static discovery status is unknown")
            row_by_candidate[(*key, row["static_candidate_id"])] = row
        if len(axis_points) != cartesian:
            raise PlanError("static discovery Cartesian denominator is incomplete")

    plan_campaigns = {
        row["campaign"]: row
        for row in plan["historical_anchor_contract"]["campaigns"]
    }
    campaigns = discovery["anchor_campaigns"]
    if (not isinstance(campaigns, list) or
            {row.get("campaign") for row in campaigns if isinstance(row, dict)} !=
            set(plan_campaigns) or len(campaigns) != len(plan_campaigns)):
        raise PlanError("historical anchor campaign denominator differs")
    plan_cells = {row["cell_key"]: row for row in plan["cells"]}
    anchor_ids: set[str] = set()
    for campaign in campaigns:
        if set(campaign) != {
            "campaign", "result_authority", "policy_authorities",
            "expected_anchor_count", "source_record_keys_sha256",
            "anchor_set_sha256", "anchors",
        }:
            raise PlanError("historical anchor campaign fields differ")
        contract = plan_campaigns[campaign["campaign"]]
        if campaign["policy_authorities"] != contract["source_authorities"]:
            raise PlanError("historical anchor policy authority differs")
        expected_cells = _historical_cell_keys(plan["cells"], campaign["campaign"])
        if (len(expected_cells) != contract["expected_source_cell_count"] or
                digest(expected_cells) != contract["expected_source_cells_sha256"]):
            raise PlanError("historical source cell denominator differs")
        authority = campaign["result_authority"]
        if (not isinstance(authority, dict) or set(authority) != {
                "locator", "sha256", "result_schema", "source_commit",
                "winner_runner_extractor", "expected_record_count",
        } or
                not isinstance(authority["locator"], str) or not authority["locator"] or
                authority["result_schema"] != contract["source_result_schema"] or
                authority["winner_runner_extractor"] != contract["extractor"] or
                authority["expected_record_count"] != contract["expected_anchor_count"] or
                not SHA256_RE.fullmatch(str(authority["sha256"])) or
                not GIT_RE.fullmatch(str(authority["source_commit"]))):
            raise PlanError("historical result authority is malformed")
        anchors = campaign["anchors"]
        if (campaign["expected_anchor_count"] != contract["expected_anchor_count"] or
                not isinstance(anchors, list) or
                len(anchors) != campaign["expected_anchor_count"]):
            raise PlanError("historical anchor count differs from its result authority")
        if campaign["anchor_set_sha256"] != digest(anchors):
            raise PlanError("historical winner/runner import digest differs")
        roles_by_cell: dict[str, Counter[str]] = {
            key: Counter() for key in expected_cells
        }
        source_record_keys: set[str] = set()
        for anchor in anchors:
            if set(anchor) != {
                "anchor_id", "role", "cell_key", "qtype", "operator", "route",
                "static_candidate_id", "runtime_variant_id", "candidate_id",
                "source_record_key", "source_config_name", "historical_anchor",
                "generated", "static_filter_pass",
            }:
                raise PlanError("historical anchor fields differ")
            if (anchor["anchor_id"] in anchor_ids or
                    not isinstance(anchor["anchor_id"], str) or not anchor["anchor_id"]):
                raise PlanError("historical anchor identity is duplicated")
            if (not isinstance(anchor["source_record_key"], str) or
                    not anchor["source_record_key"] or
                    anchor["source_record_key"] in source_record_keys or
                    not isinstance(anchor["source_config_name"], str) or
                    not anchor["source_config_name"]):
                raise PlanError("historical source record identity is malformed")
            source_record_keys.add(anchor["source_record_key"])
            anchor_ids.add(anchor["anchor_id"])
            if anchor["cell_key"] not in roles_by_cell:
                raise PlanError("historical anchor is outside its exact source denominator")
            roles_by_cell[anchor["cell_key"]][anchor["role"]] += 1
            cell = plan_cells.get(anchor["cell_key"])
            key = (anchor["qtype"], anchor["operator"], anchor["route"])
            static_row = row_by_candidate.get((*key, anchor["static_candidate_id"]))
            if (cell is None or cell["qtype"] != anchor["qtype"] or
                    cell["operator"] != anchor["operator"] or
                    anchor["operator"] != contract["operator"] or
                    anchor["route"] != contract["route"] or
                    anchor["qtype"] not in contract["qtypes"] or
                    not isinstance(anchor["runtime_variant_id"], str) or
                    not anchor["runtime_variant_id"] or
                    not isinstance(anchor["candidate_id"], str) or
                    not anchor["candidate_id"] or
                    anchor["historical_anchor"] is not True or
                    anchor["generated"] is not True or
                    anchor["static_filter_pass"] is not True or
                    static_row is None or static_row["static_status"] != "GENERATED"):
                raise PlanError("historical anchor was absent or statically rejected")
        expected_roles = Counter(contract["required_role_counts_per_cell"])
        if any(roles != expected_roles for roles in roles_by_cell.values()):
            raise PlanError("historical per-cell geometry/winner/runner denominator differs")
        if campaign["source_record_keys_sha256"] != digest(sorted(source_record_keys)):
            raise PlanError("historical deterministic import record set differs")
    return generated


def _validate_candidate(candidate: Any, fields: set[str], route: str) -> None:
    if not isinstance(candidate, dict) or set(candidate) != fields:
        raise PlanError("candidate identity fields differ")
    if not isinstance(candidate["runtime_ordinal"], int) or candidate["runtime_ordinal"] < 0:
        raise PlanError("candidate runtime ordinal is invalid")
    for name in ("candidate_id", "static_candidate_id", "runtime_variant_id",
                 "config_name", "provider", "metadata_policy", "scheduler",
                 "grid_policy", "partial_dtype", "output_dtype"):
        if not isinstance(candidate[name], str) or not candidate[name]:
            raise PlanError(f"candidate {name} is invalid")
    if candidate["family"] not in FAMILIES or candidate["algorithm"] not in ALGORITHMS:
        raise PlanError("candidate family/algorithm is unknown")
    if not isinstance(candidate["runtime_valid"], bool) or not candidate["runtime_valid"]:
        raise PlanError("runtime inventory returned a non-valid candidate")
    if candidate["bchunk"] is not None and candidate["bchunk"] not in (0, 1):
        raise PlanError("candidate BChunk is invalid")
    if candidate["grid"] is not None and (
            isinstance(candidate["grid"], bool) or not isinstance(candidate["grid"], int) or
            candidate["grid"] <= 0):
        raise PlanError("candidate grid is invalid")
    split = candidate["split_k_slices"]
    if isinstance(split, bool) or not isinstance(split, int) or split not in (1, 2, 4, 8):
        raise PlanError("candidate Split-K is invalid")
    if candidate["timing_scope"] not in TIMING_SCOPES:
        raise PlanError("candidate timing scope is invalid")
    reducer = candidate["reducer"]
    if not isinstance(reducer, dict):
        raise PlanError("candidate reducer contract is malformed")
    if split == 1:
        if candidate["timing_scope"] != "FULL_PRODUCT_E2E" or reducer != {"mode": "NOT_APPLICABLE"}:
            raise PlanError("S1 candidate cannot carry a reducer model")
    else:
        expected_keys = {
            "mode", "model_key", "model_version", "model_sha256",
            "uncertainty_in_regret", "selected_real_reducer_sanity_required",
        }
        if (candidate["timing_scope"] !=
                "MEASURED_PRODUCER_PLUS_VERSIONED_REDUCER_MODEL" or
                set(reducer) != expected_keys or
                reducer["mode"] != "VERSIONED_SHARED_MODEL" or
                not isinstance(reducer["model_key"], str) or not reducer["model_key"] or
                not isinstance(reducer["model_version"], str) or not reducer["model_version"] or
                not isinstance(reducer["model_sha256"], str) or
                not SHA256_RE.fullmatch(reducer["model_sha256"]) or
                reducer["uncertainty_in_regret"] is not True or
                reducer["selected_real_reducer_sanity_required"] is not True):
            raise PlanError("Split-K candidate lacks the shared versioned reducer model")
    if route == "scalefirst" and not candidate["algorithm"].startswith("SCALEFIRST_"):
        raise PlanError("ScaleFirst census contains a FullyQuantized algorithm")
    if route == "fully-quantized" and not candidate["algorithm"].startswith("FULLY_QUANTIZED_"):
        raise PlanError("FullyQuantized census contains a ScaleFirst algorithm")


def validate_census(
        plan: dict[str, Any], discovery: dict[str, Any], census: dict[str, Any]) -> None:
    """Validate compiled discovery coverage against every generated row."""
    generated = validate_static_discovery(plan, discovery)
    required_top = {
        "schema", "plan_sha256", "static_discovery_sha256", "source_commit",
        "bundle_manifest_sha256", "inventory_origin", "cells", "anchor_results",
    }
    if not isinstance(census, dict) or set(census) != required_top:
        raise PlanError("candidate census top-level fields differ")
    if (census["schema"] != COMPILED_CENSUS_SCHEMA or
            census["plan_sha256"] != digest(plan) or
            census["static_discovery_sha256"] != digest(discovery)):
        raise PlanError("candidate census plan authority differs")
    if not isinstance(census["source_commit"], str) or not GIT_RE.fullmatch(census["source_commit"]):
        raise PlanError("candidate census source commit is malformed")
    if (not isinstance(census["bundle_manifest_sha256"], str) or
            not SHA256_RE.fullmatch(census["bundle_manifest_sha256"])):
        raise PlanError("candidate census bundle authority is malformed")
    if census["inventory_origin"] != "COMPILED_DISCOVERY_TWO_CALL_QUERY":
        raise PlanError("candidate census was not produced by compiled discovery binaries")
    plan_cells = {row["cell_key"]: row for row in plan["cells"]}
    expected_pairs = {(key, route) for key in plan_cells for route in ROUTES}
    rows = census["cells"]
    if not isinstance(rows, list):
        raise PlanError("candidate census cells must be a list")
    observed_pairs = {(row.get("cell_key"), row.get("route")) for row in rows
                      if isinstance(row, dict)}
    if observed_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise PlanError("candidate census omitted or duplicated a cell/route query")
    fields = _candidate_fields(plan)
    shared_reducers: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for row in rows:
        required = {
            "cell_key", "qtype", "operator", "route", "availability",
            "unavailable_reason", "query_symbol", "query_arguments",
            "first_query_count", "capacity", "second_query_count",
            "response_sha256", "generated_candidate_count", "candidates",
            "runtime_variant_expansions", "rejections",
        }
        if set(row) != required:
            raise PlanError("candidate census cell fields differ")
        cell = plan_cells[row["cell_key"]]
        route = row["route"]
        if row["qtype"] != cell["qtype"] or row["operator"] != cell["operator"]:
            raise PlanError("candidate census cell identity differs")
        if row["query_symbol"] != DISCOVERY_QUERY_SYMBOLS[(cell["operator"], route)]:
            raise PlanError("candidate census used the wrong runtime query")
        if row["query_arguments"] != _expected_query_arguments(cell):
            raise PlanError("candidate census query used hidden or stale arguments")
        candidates = row["candidates"]
        expansions = row["runtime_variant_expansions"]
        rejections = row["rejections"]
        if (not isinstance(candidates, list) or not isinstance(expansions, list) or
                not isinstance(rejections, list)):
            raise PlanError("candidate census records are malformed")
        class_key = (cell["qtype"], cell["operator"], route)
        expected_generated = generated[class_key]
        if row["generated_candidate_count"] != len(expected_generated):
            raise PlanError("compiled census generated denominator differs")
        counts = (row["first_query_count"], row["capacity"], row["second_query_count"])
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts):
            raise PlanError("candidate inventory count is invalid")
        if counts != (len(candidates),) * 3:
            raise PlanError("runtime two-call inventory was truncated or changed")
        if row["response_sha256"] != digest(candidates):
            raise PlanError("runtime candidate response digest differs")
        if row["availability"] == "AVAILABLE":
            if row["unavailable_reason"] is not None or not candidates:
                raise PlanError("available route has no complete runtime candidates")
        elif row["availability"] == "STRUCTURAL_UNAVAILABLE":
            if candidates or not isinstance(row["unavailable_reason"], str) or not row["unavailable_reason"]:
                raise PlanError("structural route absence is unnamed or nonempty")
        else:
            raise PlanError("route availability is unknown")
        ids: set[str] = set()
        concrete_keys: set[tuple[str, str]] = set()
        ordinals: set[int] = set()
        for candidate in candidates:
            _validate_candidate(candidate, fields, route)
            concrete_key = (
                candidate["static_candidate_id"], candidate["runtime_variant_id"])
            if (candidate["candidate_id"] in ids or concrete_key in concrete_keys or
                    candidate["runtime_ordinal"] in ordinals):
                raise PlanError("runtime candidate identity is duplicated")
            ids.add(candidate["candidate_id"])
            concrete_keys.add(concrete_key)
            ordinals.add(candidate["runtime_ordinal"])
            if candidate["split_k_slices"] > 1:
                key, model_key = _reducer_identity(
                    cell, candidate["split_k_slices"], candidate["partial_dtype"])
                reducer = candidate["reducer"]
                if reducer["model_key"] != model_key:
                    raise PlanError("Split-K reducer model key is not M/N/S/dtype")
                old = shared_reducers.setdefault(key, reducer)
                if old != reducer:
                    raise PlanError("one M/N/S/dtype cell used per-candidate reducer models")
        expanded_parents: set[str] = set()
        admitted_points: dict[tuple[str, str], str] = {}
        for expansion in expansions:
            if (not isinstance(expansion, dict) or set(expansion) != {
                    "static_candidate_id", "runtime_space_authority",
                    "first_query_count", "capacity", "second_query_count",
                    "reported_variant_count", "points",
            }):
                raise PlanError("runtime variant expansion fields differ")
            parent = expansion["static_candidate_id"]
            if (not isinstance(parent, str) or not parent or
                    parent in expanded_parents):
                raise PlanError("runtime variant parent identity is duplicated")
            expanded_parents.add(parent)
            authority = expansion["runtime_space_authority"]
            if (not isinstance(authority, dict) or set(authority) != {
                    "expander_symbol", "source_sha256", "device_identity_sha256",
                    "workload_sha256", "response_sha256",
            } or
                    authority["expander_symbol"] !=
                    f"{DISCOVERY_QUERY_SYMBOLS[(cell['operator'], route)]}::variant-space" or
                    not SHA256_RE.fullmatch(str(authority["source_sha256"])) or
                    not SHA256_RE.fullmatch(str(authority["device_identity_sha256"])) or
                    authority["workload_sha256"] != digest(_expected_query_arguments(cell))):
                raise PlanError("runtime grid/provider variant authority differs")
            points = expansion["points"]
            variant_counts = (
                expansion["first_query_count"], expansion["capacity"],
                expansion["second_query_count"], expansion["reported_variant_count"])
            if (not isinstance(points, list) or not points or
                    any(isinstance(count, bool) or not isinstance(count, int) or count < 0
                        for count in variant_counts) or
                    variant_counts != (len(points),) * 4 or
                    authority["response_sha256"] != digest(points)):
                raise PlanError("runtime variant two-call denominator was truncated or changed")
            seen_variants: set[str] = set()
            seen_concrete: set[str] = set()
            for point in points:
                if (not isinstance(point, dict) or set(point) != {
                        "runtime_variant_id", "candidate_id", "axes", "admission", "reason"
                } or not isinstance(point["runtime_variant_id"], str) or
                        not point["runtime_variant_id"] or
                        not isinstance(point["candidate_id"], str) or
                        not point["candidate_id"] or
                        not isinstance(point["axes"], dict) or not point["axes"] or
                        point["runtime_variant_id"] in seen_variants or
                        point["candidate_id"] in seen_concrete):
                    raise PlanError("runtime variant point is malformed or duplicated")
                seen_variants.add(point["runtime_variant_id"])
                seen_concrete.add(point["candidate_id"])
                if point["admission"] == "ADMITTED":
                    if point["reason"] is not None:
                        raise PlanError("admitted runtime variant carries rejection reason")
                    admitted_points[(parent, point["runtime_variant_id"])] = point["candidate_id"]
                elif point["admission"] == "CAN_IMPLEMENT_REJECT":
                    if not isinstance(point["reason"], str) or not point["reason"]:
                        raise PlanError("runtime variant rejection is unnamed")
                else:
                    raise PlanError("runtime variant admission state is unknown")
        if set(admitted_points) != concrete_keys or any(
                admitted_points[key] != candidate["candidate_id"]
                for key, candidate in {
                    (candidate["static_candidate_id"], candidate["runtime_variant_id"]): candidate
                    for candidate in candidates
                }.items()):
            raise PlanError("concrete candidate bypassed or was lost from variant expansion")

        rejected_ids: set[str] = set()
        for rejected in rejections:
            if (not isinstance(rejected, dict) or set(rejected) !=
                    {"static_candidate_id", "stage", "reason"} or
                    rejected["stage"] not in ("TYPE_BUILD", "CAN_IMPLEMENT") or
                    not isinstance(rejected["static_candidate_id"], str) or
                    not isinstance(rejected["reason"], str) or not rejected["reason"] or
                    rejected["static_candidate_id"] in rejected_ids):
                raise PlanError("compiled discovery rejection is malformed")
            rejected_ids.add(rejected["static_candidate_id"])
        if (expanded_parents & rejected_ids or
                expanded_parents | rejected_ids != expected_generated):
            raise PlanError("compiled discovery lost or invented a static parent topology")

    anchors = {
        anchor["anchor_id"]: anchor
        for campaign in discovery["anchor_campaigns"]
        for anchor in campaign["anchors"]
    }
    results = census["anchor_results"]
    if (not isinstance(results, list) or
            {row.get("anchor_id") for row in results if isinstance(row, dict)} != set(anchors) or
            len(results) != len(anchors)):
        raise PlanError("compiled historical anchor result denominator differs")
    cell_rows = {(row["cell_key"], row["route"]): row for row in rows}
    for result in results:
        if set(result) != {
            "anchor_id", "historical_anchor", "compiled", "static_filter_pass",
            "can_implement_pass", "runtime_census_match",
        }:
            raise PlanError("compiled historical anchor result fields differ")
        anchor = anchors[result["anchor_id"]]
        candidate_keys = {
            (row["static_candidate_id"], row["runtime_variant_id"], row["candidate_id"])
            for row in cell_rows[(anchor["cell_key"], anchor["route"])]["candidates"]
        }
        if (result != {
                "anchor_id": result["anchor_id"],
                "historical_anchor": True,
                "compiled": True,
                "static_filter_pass": True,
                "can_implement_pass": True,
                "runtime_census_match": True,
            } or (
                anchor["static_candidate_id"], anchor["runtime_variant_id"],
                anchor["candidate_id"]) not in candidate_keys):
            raise PlanError("historical anchor failed generation/build/canImplement/census admission")


def validate_shipping_replay(
        plan: dict[str, Any], discovery: dict[str, Any], compiled: dict[str, Any],
        replay: dict[str, Any]) -> None:
    """Validate that a generated shipping policy replays in product DSOs."""
    validate_census(plan, discovery, compiled)
    required = {
        "schema", "plan_sha256", "compiled_census_sha256", "source_commit",
        "bundle_manifest_sha256", "cells",
    }
    if not isinstance(replay, dict) or set(replay) != required:
        raise PlanError("shipping replay top-level fields differ")
    if (replay["schema"] != SHIPPING_REPLAY_SCHEMA or
            replay["plan_sha256"] != digest(plan) or
            replay["compiled_census_sha256"] != digest(compiled) or
            not GIT_RE.fullmatch(str(replay["source_commit"])) or
            not SHA256_RE.fullmatch(str(replay["bundle_manifest_sha256"]))):
        raise PlanError("shipping replay authority differs")
    plan_cells = {row["cell_key"]: row for row in plan["cells"]}
    rows = replay["cells"]
    if (not isinstance(rows, list) or len(rows) != len(plan_cells) or
            {row.get("cell_key") for row in rows if isinstance(row, dict)} != set(plan_cells)):
        raise PlanError("shipping replay workload denominator differs")
    compiled_rows = {(row["cell_key"], row["route"]): row for row in compiled["cells"]}
    for row in rows:
        if set(row) != {
            "cell_key", "selection", "route", "candidate_id", "query_symbol",
            "replay_status", "selected_real_reducer_sanity",
        }:
            raise PlanError("shipping replay cell fields differ")
        cell = plan_cells[row["cell_key"]]
        if row["selection"] == "NO_MEASURED_POLICY":
            if (row["route"] is not None or row["candidate_id"] is not None or
                    row["query_symbol"] is not None or row["replay_status"] != "EXPLICIT_MISS" or
                    row["selected_real_reducer_sanity"] != "NOT_SELECTED"):
                raise PlanError("shipping replay policy miss is not explicit")
            continue
        if row["selection"] != "SELECTED" or row["route"] not in ROUTES:
            raise PlanError("shipping replay selection state is invalid")
        if row["query_symbol"] != SHIPPING_QUERY_SYMBOLS[(cell["operator"], row["route"])]:
            raise PlanError("shipping replay used a discovery or wrong product query")
        candidates = compiled_rows[(row["cell_key"], row["route"])]["candidates"]
        selected = next((c for c in candidates if c["candidate_id"] == row["candidate_id"]), None)
        if selected is None or row["replay_status"] != "PASS":
            raise PlanError("shipping selection was not discovered or did not replay")
        required_sanity = ("PASS" if selected["split_k_slices"] > 1 else "NOT_APPLICABLE")
        if row["selected_real_reducer_sanity"] != required_sanity:
            raise PlanError("selected Split-K row lacks its small real-reducer sanity")


def _candidate(
        route: str, ordinal: int = 0, static_candidate_id: str | None = None,
        runtime_variant_id: str = "grid-64", candidate_id: str | None = None
        ) -> dict[str, Any]:
    parent = static_candidate_id or f"{route}-topology-{ordinal}"
    concrete = candidate_id or f"{parent}::{runtime_variant_id}"
    return {
        "runtime_ordinal": ordinal,
        "candidate_id": concrete,
        "static_candidate_id": parent,
        "runtime_variant_id": runtime_variant_id,
        "config_name": f"discovery-config-{ordinal}",
        "family": "TENSOR_CORE",
        "algorithm": ("SCALEFIRST_NONPERSISTENT" if route == "scalefirst"
                      else "FULLY_QUANTIZED_S1"),
        "provider": "runtime-provider",
        "bchunk": 0,
        "metadata_policy": "runtime-metadata",
        "resolved_delivery_n": 64,
        "scheduler": "runtime-scheduler",
        "grid_policy": "runtime-grid",
        "grid": int(runtime_variant_id.removeprefix("grid-")),
        "split_k_slices": 1,
        "timing_scope": "FULL_PRODUCT_E2E",
        "partial_dtype": "float32",
        "output_dtype": "float16",
        "reducer": {"mode": "NOT_APPLICABLE"},
        "runtime_valid": True,
    }


def _static_candidate_id(
        qtype: int, operator: str, route: str, variant: str) -> str:
    return f"q{qtype}-{operator}-{route}-{variant}"


def _synthetic_static_discovery(plan: dict[str, Any]) -> dict[str, Any]:
    """Build a small but fully exhaustive generator manifest for tests."""
    classes = []
    for qtype, operator, route in sorted(_expected_discovery_classes()):
        axis_names = ["variant"]
        axis_values = {
            "variant": [f"candidate-{index}" for index in range(5)] +
            ["static-negative"]
        }
        constraint_sha = file_sha(
            ROOT / ("tools/emit_scalefirst_internal_superset.cpp"
                    if route == "scalefirst"
                    else "tools/emit_fully_quantized_splitk_superset.cpp"))
        manifest_payload = {
            "qtype": qtype,
            "operator": operator,
            "route": route,
            "axis_names": axis_names,
            "axis_values": axis_values,
            "constraint_program_sha256": constraint_sha,
        }
        rows = []
        for ordinal, variant in enumerate(axis_values["variant"]):
            rejected = variant == "static-negative"
            rows.append({
                "generator_ordinal": ordinal,
                "static_candidate_id": _static_candidate_id(
                    qtype, operator, route, variant),
                "axes": {"variant": variant},
                "static_status": "STATIC_REJECT" if rejected else "GENERATED",
                "reason": "SYNTHETIC_CONSTRAINT_REJECT" if rejected else None,
            })
        classes.append({
            "qtype": qtype,
            "operator": operator,
            "route": route,
            "axis_denominator": {
                "axis_names": axis_names,
                "axis_values": axis_values,
                "cartesian_product_count": len(rows),
                "constraint_program_sha256": constraint_sha,
                "generator_manifest_sha256": digest(manifest_payload),
            },
            "reported_raw_count": len(rows),
            "rows": rows,
        })

    cell_by_key = {cell["cell_key"]: cell for cell in plan["cells"]}
    campaigns = []
    for contract in plan["historical_anchor_contract"]["campaigns"]:
        anchors = []
        source_keys = []
        for cell_key in _historical_cell_keys(plan["cells"], contract["campaign"]):
            cell = cell_by_key[cell_key]
            for role, count in contract["required_role_counts_per_cell"].items():
                for role_index in range(count):
                    variant_index = (
                        role_index if role == "HISTORICAL_CANDIDATE_ANCHOR"
                        else 1 if role == "KPACK_RUNNER" else 0)
                    variant = f"candidate-{variant_index}"
                    parent = _static_candidate_id(
                        cell["qtype"], cell["operator"], contract["route"], variant)
                    runtime_variant = "grid-64"
                    concrete = f"{parent}::{runtime_variant}"
                    source_key = (
                        f"{contract['campaign']}/{cell_key}/{role}/{role_index}")
                    source_keys.append(source_key)
                    if role == "HISTORICAL_CANDIDATE_ANCHOR" and count == 5:
                        source_config = a04_inventory.CANDIDATES[role_index]
                    elif contract["campaign"] == "q4-scalefirst-real-shapes":
                        policy = json.loads(
                            (ROOT / "benchmarks/scalefirst_q4k_pruned_policy.json").read_text(
                                encoding="utf-8"))
                        source_config = policy["anchor_symbol"]
                    else:
                        source_config = f"synthetic-{role.lower()}-{role_index}"
                    anchors.append({
                        "anchor_id": source_key,
                        "role": role,
                        "cell_key": cell_key,
                        "qtype": cell["qtype"],
                        "operator": cell["operator"],
                        "route": contract["route"],
                        "static_candidate_id": parent,
                        "runtime_variant_id": runtime_variant,
                        "candidate_id": concrete,
                        "source_record_key": source_key,
                        "source_config_name": source_config,
                        "historical_anchor": True,
                        "generated": True,
                        "static_filter_pass": True,
                    })
        campaigns.append({
            "campaign": contract["campaign"],
            "result_authority": {
                "locator": f"synthetic://{contract['campaign']}",
                "sha256": digest({"campaign": contract["campaign"]}),
                "result_schema": contract["source_result_schema"],
                "source_commit": "a" * 40,
                "winner_runner_extractor": contract["extractor"],
                "expected_record_count": len(anchors),
            },
            "policy_authorities": copy.deepcopy(contract["source_authorities"]),
            "expected_anchor_count": len(anchors),
            "source_record_keys_sha256": digest(sorted(source_keys)),
            "anchor_set_sha256": digest(anchors),
            "anchors": anchors,
        })
    generator_paths = sorted(
        path
        for paths in plan["candidate_discovery_contract"]["static_generator_authorities"].values()
        for path in paths
    )
    return {
        "schema": STATIC_DISCOVERY_SCHEMA,
        "plan_sha256": digest(plan),
        "source_commit": "a" * 40,
        "generator_authorities": [
            {"path": path, "sha256": file_sha(ROOT / path)}
            for path in generator_paths
        ],
        "classes": classes,
        "anchor_campaigns": campaigns,
    }


def _synthetic_census(
        plan: dict[str, Any], discovery: dict[str, Any] | None = None
        ) -> dict[str, Any]:
    discovery = discovery or _synthetic_static_discovery(plan)
    generated = validate_static_discovery(plan, discovery)
    rows = []
    for cell in plan["cells"]:
        for route in ROUTES:
            parent_ids = sorted(generated[(cell["qtype"], cell["operator"], route)])
            candidates = []
            expansions = []
            runtime_ordinal = 0
            for parent_ordinal, parent in enumerate(parent_ids):
                grids = (64, 128) if parent_ordinal == 0 else (64,)
                points = []
                for grid in grids:
                    runtime_variant = f"grid-{grid}"
                    concrete = f"{parent}::{runtime_variant}"
                    points.append({
                        "runtime_variant_id": runtime_variant,
                        "candidate_id": concrete,
                        "axes": {"grid": grid, "provider": "standard-aiu"},
                        "admission": "ADMITTED",
                        "reason": None,
                    })
                    candidates.append(_candidate(
                        route, runtime_ordinal, parent, runtime_variant, concrete))
                    runtime_ordinal += 1
                source_sha = file_sha(
                    ROOT / ("tools/emit_scalefirst_internal_superset.cpp"
                            if route == "scalefirst"
                            else "tools/emit_fully_quantized_splitk_superset.cpp"))
                expansions.append({
                    "static_candidate_id": parent,
                    "runtime_space_authority": {
                        "expander_symbol": (
                            f"{DISCOVERY_QUERY_SYMBOLS[(cell['operator'], route)]}"
                            "::variant-space"),
                        "source_sha256": source_sha,
                        "device_identity_sha256": "d" * 64,
                        "workload_sha256": digest(_expected_query_arguments(cell)),
                        "response_sha256": digest(points),
                    },
                    "first_query_count": len(points),
                    "capacity": len(points),
                    "second_query_count": len(points),
                    "reported_variant_count": len(points),
                    "points": points,
                })
            rows.append({
                "cell_key": cell["cell_key"],
                "qtype": cell["qtype"],
                "operator": cell["operator"],
                "route": route,
                "availability": "AVAILABLE",
                "unavailable_reason": None,
                "query_symbol": DISCOVERY_QUERY_SYMBOLS[(cell["operator"], route)],
                "query_arguments": _expected_query_arguments(cell),
                "first_query_count": len(candidates),
                "capacity": len(candidates),
                "second_query_count": len(candidates),
                "response_sha256": digest(candidates),
                "generated_candidate_count": len(parent_ids),
                "candidates": candidates,
                "runtime_variant_expansions": expansions,
                "rejections": [],
            })
    anchors = [
        anchor
        for campaign in discovery["anchor_campaigns"]
        for anchor in campaign["anchors"]
    ]
    return {
        "schema": COMPILED_CENSUS_SCHEMA,
        "plan_sha256": digest(plan),
        "static_discovery_sha256": digest(discovery),
        "source_commit": "a" * 40,
        "bundle_manifest_sha256": "b" * 64,
        "inventory_origin": "COMPILED_DISCOVERY_TWO_CALL_QUERY",
        "cells": rows,
        "anchor_results": [
            {
                "anchor_id": anchor["anchor_id"],
                "historical_anchor": True,
                "compiled": True,
                "static_filter_pass": True,
                "can_implement_pass": True,
                "runtime_census_match": True,
            }
            for anchor in anchors
        ],
    }


def _synthetic_shipping_replay(
        plan: dict[str, Any], compiled: dict[str, Any]) -> dict[str, Any]:
    compiled_rows = {(row["cell_key"], row["route"]): row for row in compiled["cells"]}
    rows = []
    for cell in plan["cells"]:
        route = "fully-quantized"
        candidate = compiled_rows[(cell["cell_key"], route)]["candidates"][0]
        rows.append({
            "cell_key": cell["cell_key"],
            "selection": "SELECTED",
            "route": route,
            "candidate_id": candidate["candidate_id"],
            "query_symbol": SHIPPING_QUERY_SYMBOLS[(cell["operator"], route)],
            "replay_status": "PASS",
            "selected_real_reducer_sanity": (
                "PASS" if candidate["split_k_slices"] > 1 else "NOT_APPLICABLE"),
        })
    return {
        "schema": SHIPPING_REPLAY_SCHEMA,
        "plan_sha256": digest(plan),
        "compiled_census_sha256": digest(compiled),
        "source_commit": "a" * 40,
        "bundle_manifest_sha256": "c" * 64,
        "cells": rows,
    }


def self_test() -> None:
    plan = materialize()
    validate_plan(plan)
    discovery = _synthetic_static_discovery(plan)
    validate_static_discovery(plan, discovery)
    census = _synthetic_census(plan, discovery)
    validate_census(plan, discovery, census)
    replay = _synthetic_shipping_replay(plan, census)
    validate_shipping_replay(plan, discovery, census, replay)

    plan_plants = []
    broken = copy.deepcopy(plan); broken["formats"][0]["arrangement"]["layout"] = 0; plan_plants.append(broken)
    broken = copy.deepcopy(plan); broken["cells"].pop(); plan_plants.append(broken)
    broken = copy.deepcopy(plan); broken["measurement_contract"]["screen"]["top_n"] = 8; plan_plants.append(broken)
    broken = copy.deepcopy(plan); broken["selector_contract"]["features"]["grouped"].append("active"); plan_plants.append(broken)
    broken = copy.deepcopy(plan); broken["measurement_contract"]["split_k"]["per_candidate_reducer_model"] = True; plan_plants.append(broken)
    broken = copy.deepcopy(plan); broken["output_contract"]["required"].pop(); plan_plants.append(broken)
    for broken in plan_plants:
        try:
            validate_plan(broken)
        except PlanError:
            pass
        else:
            raise AssertionError("plan negative stayed green")

    discovery_plants = []
    broken = copy.deepcopy(discovery); broken["classes"].pop(); discovery_plants.append(broken)
    broken = copy.deepcopy(discovery); broken["classes"][0]["reported_raw_count"] -= 1; discovery_plants.append(broken)
    broken = copy.deepcopy(discovery); broken["generator_authorities"][0]["sha256"] = "0" * 64; discovery_plants.append(broken)
    broken = copy.deepcopy(discovery); broken["anchor_campaigns"][0]["anchors"].pop(); discovery_plants.append(broken)
    broken = copy.deepcopy(discovery)
    campaign = next(row for row in broken["anchor_campaigns"]
                    if row["campaign"] == "fq-five-format-dense")
    anchor = campaign["anchors"][0]
    q12_cell = next(cell for cell in plan["cells"]
                    if cell["qtype"] == 12 and cell["operator"] == "dense")
    anchor.update({
        "cell_key": q12_cell["cell_key"], "qtype": 12,
        "candidate_id": _static_candidate_id(
            12, "dense", "fully-quantized", "winner"),
    })
    campaign["anchor_set_sha256"] = digest(campaign["anchors"])
    discovery_plants.append(broken)
    for index, broken in enumerate(discovery_plants):
        try:
            validate_static_discovery(plan, broken)
        except PlanError:
            pass
        else:
            raise AssertionError(f"static discovery negative {index} stayed green")

    first = census["cells"][0]
    census_plants = []
    broken = copy.deepcopy(census); broken["cells"].pop(); census_plants.append(broken)
    broken = copy.deepcopy(census); broken["cells"][0]["second_query_count"] = 0; census_plants.append(broken)
    broken = copy.deepcopy(census); broken["cells"][0]["query_arguments"]["public_problem"]["active"] = 1; census_plants.append(broken)
    broken = copy.deepcopy(census); broken["cells"][0]["candidates"] = []; broken["cells"][0]["response_sha256"] = digest([]); broken["cells"][0]["first_query_count"] = broken["cells"][0]["capacity"] = broken["cells"][0]["second_query_count"] = 0; census_plants.append(broken)
    broken = copy.deepcopy(census); broken["cells"][0]["candidates"][0]["candidate_id"] = "hardcoded"; census_plants.append(broken)
    for index, broken in enumerate(census_plants):
        try:
            validate_census(plan, discovery, broken)
        except PlanError:
            pass
        else:
            raise AssertionError(f"candidate census negative {index} stayed green")
    assert first["query_arguments"] == _expected_query_arguments(plan["cells"][0])
    broken_replay = copy.deepcopy(replay)
    broken_replay["cells"][0]["query_symbol"] = DISCOVERY_QUERY_SYMBOLS[
        (plan["cells"][0]["operator"], "fully-quantized")]
    try:
        validate_shipping_replay(plan, discovery, census, broken_replay)
    except PlanError:
        pass
    else:
        raise AssertionError("shipping replay negative stayed green")
    print(
        "[fq-kpack-route-optimal-plan:self-test] PASS formats=5 dense=143 "
        "grouped=52+24 q4_anchor_only=286 cells=1381 route_queries=2762 "
        "static-generator/compiled-discovery/shipping-replay safe-elimination "
        "prepass/cold/warm break-even shared-reducer-model historical-anchors; "
        "eighteen plants RED"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    emit = commands.add_parser("materialize")
    emit.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate-plan")
    check.add_argument("--plan", type=Path, required=True)
    static_check = commands.add_parser("validate-static")
    static_check.add_argument("--plan", type=Path, required=True)
    static_check.add_argument("--discovery", type=Path, required=True)
    census_check = commands.add_parser("validate-compiled")
    census_check.add_argument("--plan", type=Path, required=True)
    census_check.add_argument("--discovery", type=Path, required=True)
    census_check.add_argument("--census", type=Path, required=True)
    replay_check = commands.add_parser("validate-shipping")
    replay_check.add_argument("--plan", type=Path, required=True)
    replay_check.add_argument("--discovery", type=Path, required=True)
    replay_check.add_argument("--census", type=Path, required=True)
    replay_check.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "materialize":
            value = materialize()
            validate_plan(value)
            atomic_json(args.output, value)
            print(
                f"[fq-kpack-route-optimal-plan] PASS cells={len(value['cells'])} "
                f"route_queries={value['workload_denominator']['route_inventory_queries']} "
                f"output={args.output}"
            )
        elif args.command == "validate-plan":
            validate_plan(json.loads(args.plan.read_text(encoding="utf-8")))
            print(f"[fq-kpack-route-optimal-plan] PASS validated={args.plan}")
        elif args.command == "validate-static":
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
            validate_static_discovery(plan, discovery)
            print(f"[fq-kpack-route-optimal-static] PASS validated={args.discovery}")
        elif args.command == "validate-compiled":
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
            census = json.loads(args.census.read_text(encoding="utf-8"))
            validate_census(plan, discovery, census)
            print(f"[fq-kpack-route-optimal-compiled] PASS validated={args.census}")
        else:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
            census = json.loads(args.census.read_text(encoding="utf-8"))
            replay = json.loads(args.replay.read_text(encoding="utf-8"))
            validate_shipping_replay(plan, discovery, census, replay)
            print(f"[fq-kpack-route-optimal-shipping] PASS validated={args.replay}")
        return 0
    except (AssertionError, KeyError, OSError, PlanError, TypeError, ValueError) as error:
        print(f"[fq-kpack-route-optimal-plan] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

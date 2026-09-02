from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import plan_fq_splitk_reducer_lookup as reducer_plan
import plan_kpack_confidence_followup as followup


IDENTITY = (10, "dense", "dense_test")
PUBLIC = {"qtype": 10, "m": 1, "n": 256, "k": 3072, "group_size": 16}
EXPECTED = {IDENTITY: {"public_problem": PUBLIC, "source_class": "test"}}


def _samples(values):
    values = list(values)
    assert len(values) == 33
    return values


def _row(symbol: str, values, *, work_item: str = "a" * 64,
         route: str = "scalefirst", split: int = 1):
    values = _samples(values)
    runs = []
    for round_index in range(3):
        part = values[round_index * 11:(round_index + 1) * 11]
        runs.append({
            "round": round_index + 1,
            "order": ("FORWARD", "REVERSE", "HASHED")[round_index],
            "schedule_seed": f"0x{round_index + 1:016x}",
            "source_log_sha256": str(round_index + 1) * 64,
            "samples": [{"sample_index": index, "us": value}
                        for index, value in enumerate(part)],
        })
    return {
        "classification": "MEASURED", "work_item_id": work_item,
        "route": route, "operator": "dense", "qtype": 10,
        "workload_key": "dense_test", "source_class": "test",
        "public_problem": dict(PUBLIC), "parent_id": symbol,
        "static_candidate_id": f"static-{symbol}", "symbol": symbol,
        "static": {"tile_m": 8, "tile_n": 64, "tactic_tile_k": 128},
        "runtime": {"algorithm": f"TC_S{split}", "split": split},
        "timing": {"confirm_runs": runs},
        "authority": {
            "device_identity_sha256": "d" * 64,
            "runtime_linkage_sha256": "e" * 64,
        },
    }


def _authority():
    return {
        "route_plan_canonical_sha256": "1" * 64,
        "discovery": {"summary_sha256": "2" * 64},
        "reducer": {"summary_sha256": "3" * 64},
    }


def _reducer(values):
    implementation = reducer_plan.expected_implementation(1, 256)
    key = (1, 256, 2, "fp32", "fp16", implementation)
    raw = _samples(values)
    return {key: {
        "case_id": "m1-n256-s2-fp32-fp16", "samples": raw,
        "samples_sha256": followup.digest(raw),
        "implementation": implementation,
    }}


def _write_reducer_result(root: Path):
    plan = reducer_plan.materialize()
    samples = [f"{1 + index / 1000:.9f}" for index in range(33)]
    ordered = sorted(samples, key=float)
    round_medians = [samples[index + 5] for index in (0, 11, 22)]
    rows = []
    for case in plan["cases"]:
        rows.append({
            "ordinal": case["ordinal"], "case_id": case["case_id"],
            "m": case["m"], "n": case["n"], "split": case["split"],
            "partial_dtype": "fp32", "output_dtype": "fp16",
            "implementation": case["expected_implementation"],
            "workspace_bytes": case["workspace_bytes"],
            "output_bytes": case["output_bytes"],
            "grid_ctas": 1, "block_threads": 256,
            "round_medians_us": round_medians, "samples_us": samples,
            "sample_count": 33, "median_us": ordered[16],
            "min_us": ordered[0], "max_us": ordered[-1],
            "raw_bit_correctness": "PASS",
        })
    summary_authority = {
        "execution_authority_sha256": "e" * 64,
        "source_commit": "source", "source_tree": "tree",
        "sdk_authority_sha256": "c" * 64,
        "binary_sha256": "b" * 64,
        "build_authority_sha256": "a" * 64,
        "device_identity_sha256": "d" * 64,
        "device_homogeneity_sha256": "f" * 64,
        "device_homogeneity_key": "9" * 64,
        "runtime_linkage_sha256": "8" * 64,
    }
    summary = {
        "schema": "quactlize.fq-splitk-reducer-lookup-result.v1",
        "verdict": "EXACT_REDUCER_LOOKUP_MEASURED",
        "plan_sha256": reducer_plan.digest(plan),
        "bundle_manifest_sha256": "7" * 64,
        "authorities": summary_authority,
        "measurement": {
            "rounds": plan["measurement"]["rounds"],
            "warmups_per_round": 3, "samples_per_round": 11,
            "samples_per_case": 33,
            "case_order": plan["measurement"]["case_order"],
            "top_n": None, "point_estimate_pruning": False,
            "raw_bit_correctness":
                "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
        },
        "denominator": {"cases": 1035, "unique_m_n": 345,
                        "splits": [2, 4, 8]},
        "rows": rows,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tsv_path = root / "summary.tsv"
    tsv_path.write_text("fixture\n")
    analyzer_sha = hashlib.sha256(
        (ROOT / "tools/analyze_fq_splitk_reducer_lookup.py").read_bytes()).hexdigest()
    authority = {
        "schema": "quactlize.fq-splitk-reducer-result-authority.v1",
        "execution_authority": {"path": "execution-authority.json",
                                "sha256": "e" * 64},
        "plan": {"path": "reducer-plan.json", "file_sha256": "6" * 64,
                 "canonical_sha256": reducer_plan.digest(plan)},
        "bundle_manifest": {"path": "manifest.json", "sha256": "7" * 64},
        "binary": {"sha256": "b" * 64,
                   "build_authority_sha256": "a" * 64,
                   "sdk_authority_sha256": "c" * 64},
        "device": {"worker_id": 0, "device_identity_sha256": "d" * 64,
                   "device_homogeneity_sha256": "f" * 64,
                   "device_homogeneity_key": "9" * 64,
                   "runtime_linkage_sha256": "8" * 64},
        "runs": [{"round": index, "path": f"round-{index}.log",
                  "sha256": str(index) * 64} for index in (1, 2, 3)],
        "outputs": {
            "summary.json": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "summary.tsv": hashlib.sha256(tsv_path.read_bytes()).hexdigest(),
        },
        "analyzer": {"path": "tools/analyze_fq_splitk_reducer_lookup.py",
                     "sha256": analyzer_sha},
    }
    (root / "result-authority.json").write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n")
    discovery = {
        "source_sha": "source", "source_tree": "tree",
        "device_homogeneity_sha256": "f" * 64,
        "device_workers": {"d" * 64: {
            "worker_id": 0, "runtime_linkage_sha256": "8" * 64}},
    }
    return discovery


def test_distribution_free_intervals_have_at_least_99pct_coverage():
    ratio = followup.paired_log_ratio_interval([2.0] * 33, [1.0] * 33)
    assert ratio["order_rank"] == 9
    assert ratio["coverage"] >= 0.99
    assert ratio["lower_ratio"] == pytest.approx(2.0)
    absolute = followup.order_statistic_interval(
        range(1, 34), followup.ABSOLUTE_COMPONENT_CONFIDENCE)
    assert absolute["order_rank"] == 8
    assert absolute["coverage"] >= followup.ABSOLUTE_COMPONENT_CONFIDENCE


def test_no_numeric_cap_retains_every_overlapping_candidate():
    rows = [_row(f"candidate-{index:03d}", [10.0] * 33)
            for index in range(137)]
    result = followup.derive_followup(rows, {}, EXPECTED, _authority())
    assert result["contract"]["top_n"] is None
    assert result["contract"]["numeric_candidate_cap"] is None
    assert result["denominator"]["input_product_candidates"] == 137
    assert result["denominator"]["retained_candidates"] == 137
    assert result["denominator"]["excluded_candidates"] == 0
    assert all(job["candidate_count"] == 137 for job in result["measurement_jobs"]
               if job["board"] != "prepass")
    followup.validate_plan(result)


def test_cleanly_separated_paired_candidate_is_excluded():
    rows = [_row("fast", [10.0] * 33), _row("slow", [20.0] * 33)]
    result = followup.derive_followup(rows, {}, EXPECTED, _authority())
    workload = result["workloads"][0]
    assert workload["retained_count"] == 1
    assert workload["excluded_count"] == 1
    proof = workload["excluded"][0]["proof"]
    assert proof["absolute_envelope"]["proved_slower"] is True
    assert proof["paired_log_ratio"]["lower_ratio"] > 1.03


def test_absolute_separation_without_paired_ratio_proof_is_retained():
    # The marginal intervals separate: only seven high witness samples and
    # seven low candidate samples lie outside their central absolute bands.
    # Pair those two tails disjointly, however, and fourteen ratios fail the
    # margin.  The second gate therefore does real work and must retain.
    witness = [100.0] * 7 + [1.0] * 26
    candidate = [2.0] * 7 + [0.5] * 7 + [2.0] * 19
    rows = [_row("witness", witness), _row("candidate", candidate)]
    normalized = [followup._candidate(row, {}) for row in rows]
    assert normalized[1]["envelope_lower_us"] > (
        normalized[0]["envelope_upper_us"] * 1.03)
    ratio = followup.paired_log_ratio_interval(
        normalized[1]["total_samples"], normalized[0]["total_samples"])
    assert ratio["lower_ratio"] <= 1.03
    result = followup.derive_followup(rows, {}, EXPECTED, _authority())
    assert result["denominator"]["retained_candidates"] == 2


def test_noisy_overlapping_candidate_is_retained():
    fast = [10.0] * 26 + [30.0] * 7
    noisy = [9.0] * 8 + [13.0] * 25
    result = followup.derive_followup(
        [_row("fast", fast), _row("noisy", noisy)], {}, EXPECTED, _authority())
    assert result["workloads"][0]["retained_count"] == 2
    assert result["workloads"][0]["excluded_count"] == 0


def test_reducer_uncertainty_is_added_and_can_prevent_elimination():
    fast = _row("fast", [1.0] * 33, route="fully-quantized", split=2)
    slow = _row("slow", [2.0] * 33, route="fully-quantized", split=2)
    narrow = _reducer([1.0] * 33)
    narrow_result = followup.derive_followup(
        [fast, slow], narrow, EXPECTED, _authority())
    assert narrow_result["workloads"][0]["excluded_count"] == 1

    wide_values = [0.1] * 7 + [1.0] * 18 + [100.0] * 8
    wide_result = followup.derive_followup(
        [fast, slow], _reducer(wide_values), EXPECTED, _authority())
    assert wide_result["workloads"][0]["retained_count"] == 2
    for candidate in wide_result["workloads"][0]["retained_candidates"]:
        assert candidate["steady_evidence"]["reducer_component_interval"] is not None


def test_missing_exact_reducer_fails_closed():
    row = _row("split", [1.0] * 33, route="fully-quantized", split=2)
    with pytest.raises(followup.PlannerError, match="no exact reducer") as failure:
        followup.derive_followup([row], {}, EXPECTED, _authority())
    assert failure.value.code == "MISSING_REDUCER"
    denied = followup.denial(failure.value, Path("/absent-a"), Path("/absent-b"))
    assert denied["verdict"] == "NO_MEASURED_POLICY"
    assert denied["compiled_default_fallback"] is False
    assert denied["measurement_jobs"] == []


def test_no_e2e_public_candidate_fails_closed():
    diagnostic = _row("producer-only", [1.0] * 33, split=4)
    with pytest.raises(followup.PlannerError) as failure:
        followup.derive_followup([diagnostic], {}, EXPECTED, _authority())
    assert failure.value.code == "NO_PUBLIC_ADMISSIBLE_CANDIDATE"


def test_tampered_output_payload_is_rejected():
    result = followup.derive_followup(
        [_row("one", [10.0] * 33)], {}, EXPECTED, _authority())
    broken = copy.deepcopy(result)
    broken["workloads"][0]["retained_candidates"][0]["symbol"] = "tampered"
    with pytest.raises(followup.PlannerError, match="payload hash"):
        followup.validate_plan(broken)


def test_reducer_result_authority_loads_and_tampering_fails(tmp_path):
    discovery = _write_reducer_result(tmp_path)
    result, authority = followup.load_reducer(tmp_path, discovery)
    assert len(result) == 1035
    assert authority["device_identity_sha256"] == "d" * 64

    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["rows"][0]["samples_us"][0] = "9.000000000"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with pytest.raises(followup.PlannerError) as failure:
        followup.load_reducer(tmp_path, discovery)
    assert failure.value.code == "REDUCER_AUTHORITY_MISMATCH"


def test_cross_work_item_candidate_is_not_falsely_paired():
    rows = [_row("fast", [1.0] * 33, work_item="a" * 64),
            _row("slow", [100.0] * 33, work_item="b" * 64)]
    result = followup.derive_followup(rows, {}, EXPECTED, _authority())
    assert result["denominator"]["retained_candidates"] == 2

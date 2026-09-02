from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_fq_splitk_reducer_lookup as analyzer
import kpack_discovery_worker_plan as worker_plan
import plan_fq_splitk_reducer_lookup as reducer_plan


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _round_log(plan, round_number: int, seed: int) -> str:
    order = analyzer.seeded_order(len(plan["cases"]), seed)
    order_hash = analyzer.order_hash(order)
    plan_sha = reducer_plan.digest(plan)
    lines = [
        "FQ_REDUCER_LOOKUP_RUN "
        f"schema={reducer_plan.SCHEMA} plan_sha256={plan_sha} total_cases=1035 "
        f"case_begin=0 case_end=1035 selected_cases=1035 round={round_number} "
        f"warmups=3 samples=11 schedule_seed=0x{seed:016x} "
        f"order_hash=0x{order_hash:016x} partial_dtype=fp32 output_dtype=fp16 "
        "reducer=M1FastReductionE2 fixture=period31-plus-rounding257-v1 "
        "plant_output_fault=0 status=BEGIN"
    ]
    for execution, ordinal in enumerate(order):
        row = plan["cases"][ordinal]
        base = ordinal + round_number
        samples = [f"{base + sample + 1}.000000000" for sample in range(11)]
        for sample, value in enumerate(samples):
            lines.append(
                "FQ_REDUCER_LOOKUP_SAMPLE "
                f"ordinal={ordinal} execution_ordinal={execution} "
                f"case_id={row['case_id']} round={round_number} sample={sample} "
                f"M={row['m']} N={row['n']} S={row['split']} "
                f"implementation={row['expected_implementation']} us={value}")
        lines.append(
            "FQ_REDUCER_LOOKUP_CASE "
            f"ordinal={ordinal} execution_ordinal={execution} case_id={row['case_id']} "
            f"M={row['m']} N={row['n']} S={row['split']} partial_dtype=fp32 "
            f"output_dtype=fp16 implementation={row['expected_implementation']} "
            f"workspace_bytes={row['workspace_bytes']} output_bytes={row['output_bytes']} "
            f"grid_ctas=1 block_threads=256 round={round_number} warmups=3 samples=11 "
            f"raw_bad=0 first_bad=4294967295 median_us={samples[5]} "
            f"min_us={samples[0]} max_us={samples[-1]} status=PASS")
    lines.append(
        "FQ_REDUCER_LOOKUP_DONE "
        f"plan_sha256={plan_sha} selected_cases=1035 measured=1035 failures=0 "
        f"round={round_number} warmups=3 samples=11 schedule_seed=0x{seed:016x} "
        f"order_hash=0x{order_hash:016x} status=PASS")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("reducer-evidence")
    runs = root / "runs"; runs.mkdir()
    plan = reducer_plan.materialize()
    plan_path = root / "reducer-plan.json"; _write_json(plan_path, plan)
    manifest = {
        "schema": "quactlize.fq-splitk-reducer-prebuilt.v1",
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "sdk": {"test": True},
        "build": {"plan_sha256": reducer_plan.digest(plan),
                  "build_authority_sha256": "c" * 64},
        "artifacts": {"binary": {"sha256": "d" * 64}},
    }
    manifest_path = root / "bundle-manifest.json"; _write_json(manifest_path, manifest)
    linkage = [["libhggc.so.13", "/runtime/libhggc.so.13",
                "/runtime/libhggc.so.13.0", 1, "e" * 64]]
    execution = {
        "schema": analyzer.EXECUTION_SCHEMA,
        "source_commit": "a" * 40, "source_tree": "b" * 40,
        "sdk_authority_sha256": analyzer.canonical_sha(manifest["sdk"]),
        "bundle_manifest_sha256": analyzer.sha256_file(manifest_path),
        "binary_sha256": "d" * 64, "build_authority_sha256": "c" * 64,
        "plan_file_sha256": analyzer.sha256_file(plan_path),
        "plan_sha256": reducer_plan.digest(plan), "worker_id": 0,
        "visible_device_ordinal": "0", "device_identity_sha256": "f" * 64,
        "device_homogeneity_sha256": "1" * 64,
        "device_homogeneity_key": "2" * 64,
        "runtime_linkage": linkage,
        "runtime_linkage_sha256": worker_plan.digest(linkage),
        "measurement": {
            "rounds": plan["measurement"]["rounds"], "warmups": 3,
            "samples": 11, "case_order": plan["measurement"]["case_order"],
            "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
            "top_n": None, "point_estimate_pruning": False,
        },
    }
    execution_path = root / "execution-authority.json"
    _write_json(execution_path, execution)
    for round_number, seed in enumerate(reducer_plan.ROUND_SEEDS, 1):
        (runs / f"round-{round_number}.log").write_text(
            _round_log(plan, round_number, seed))
    return root, plan, plan_path, manifest_path, execution_path, runs


def test_exact_three_round_lookup_is_emitted(evidence, tmp_path):
    _root, _plan, plan_path, manifest_path, execution_path, runs = evidence
    output = tmp_path / "results"
    result = analyzer.analyze(
        plan_path, manifest_path, execution_path, runs, output)
    assert result["verdict"] == "EXACT_REDUCER_LOOKUP_MEASURED"
    assert result["denominator"] == {
        "cases": 1035, "unique_m_n": 345, "splits": [2, 4, 8]}
    assert len(result["rows"]) == 1035
    assert all(row["sample_count"] == 33 and
               row["raw_bit_correctness"] == "PASS"
               for row in result["rows"])
    authority = json.loads((output / "result-authority.json").read_text())
    assert set(authority) == {
        "schema", "execution_authority", "plan", "bundle_manifest",
        "binary", "device", "runs", "outputs", "analyzer"}
    assert authority["outputs"]["summary.json"] == analyzer.sha256_file(
        output / "summary.json")
    assert result["authorities"]["device_homogeneity_sha256"] == "1" * 64


@pytest.mark.parametrize("plant,pattern", [
    ("missing-sample", "lacks 11 exact samples"),
    ("wrong-seed", "schedule_seed differs"),
    ("wrong-order", "sample execution order differs"),
    ("raw-bad", "failed correctness/resource closure"),
    ("median", "sample statistics differ"),
])
def test_round_tampering_fails_closed(evidence, tmp_path, plant, pattern):
    _root, plan, _plan_path, _manifest, _execution, runs = evidence
    source = (runs / "round-1.log").read_text()
    if plant == "missing-sample":
        lines = source.splitlines()
        del lines[next(index for index, line in enumerate(lines)
                       if line.startswith("FQ_REDUCER_LOOKUP_SAMPLE "))]
        source = "\n".join(lines) + "\n"
    elif plant == "wrong-seed":
        source = source.replace(
            "schedule_seed=0x6a09e667f3bcc909",
            "schedule_seed=0x6a09e667f3bcc908", 1)
    elif plant == "wrong-order":
        source = source.replace("execution_ordinal=0", "execution_ordinal=1", 1)
    elif plant == "raw-bad":
        source = source.replace("raw_bad=0", "raw_bad=1", 1)
    else:
        source = source.replace("median_us=", "median_us=999", 1)
    path = tmp_path / f"{plant}.log"; path.write_text(source)
    with pytest.raises(analyzer.AnalyzeError, match=pattern):
        analyzer.parse_round(path, plan, 1, reducer_plan.ROUND_SEEDS[0])


def test_execution_authority_tampering_fails_closed(evidence):
    _root, plan, _plan_path, manifest_path, execution_path, _runs = evidence
    manifest = json.loads(manifest_path.read_text())
    value = json.loads(execution_path.read_text())
    value["runtime_linkage_sha256"] = "0" * 64
    with pytest.raises(analyzer.AnalyzeError, match="binding differs"):
        analyzer.validate_execution_authority(value, manifest, plan)

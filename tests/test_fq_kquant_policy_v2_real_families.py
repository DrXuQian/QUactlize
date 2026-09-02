"""Tests for the compile-free five-family A04 policy extension."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_fq_kquant_policy_v2_real_families as analyzer  # noqa: E402
import plan_fq_kquant_policy_v2_real_families as planner  # noqa: E402


def make_run(tmp_path, timing=analyzer._base_timing):
    root = tmp_path / "run"
    analyzer._write_synthetic_run(root, timing)
    return root


def test_plan_is_exact_five_family_categorical_authority():
    value = planner.materialize()
    planner.validate(value)
    assert [(row["n"], row["k"]) for row in value["families"]] == list(planner.FAMILIES)
    assert all([cell["m"] for cell in row["dense"]] == list(range(1, 65)) for row in value["families"])
    assert tuple(value["candidate_names"]) == planner.pilot.CANDIDATES
    assert value["measurement"] == {
        "execution_unit": "one-family-one-round",
        "rounds": 3,
        "samples_per_round": 11,
        "warmups_per_round": 3,
        "dense_cases_per_execution": 64,
        "categorical_candidates_per_case": 5,
        "correctness": "full-output-raw-bit-device-compare",
    }
    assert value["policy"]["compiled_default_is_measured_policy"] is False
    assert value["policy"]["cross_family_extrapolation"] is False
    assert value["outside_scope"]["m_greater_than_64"] == "A07_SCALEFIRST"


@pytest.mark.parametrize(
    "plant",
    ["family", "m", "candidate", "mapping", "rounds", "default", "cross-family", "m-gt64"],
)
def test_plan_plants_fail_closed(plant):
    value = copy.deepcopy(planner.materialize())
    if plant == "family":
        value["families"].pop()
    elif plant == "m":
        value["families"][0]["dense"].pop()
    elif plant == "candidate":
        value["candidate_names"].pop()
    elif plant == "mapping":
        value["layout"]["mapping_id"] = "0x0"
    elif plant == "rounds":
        value["measurement"]["rounds"] = 2
    elif plant == "default":
        value["policy"]["compiled_default_is_measured_policy"] = True
    elif plant == "cross-family":
        value["policy"]["cross_family_extrapolation"] = True
    else:
        value["outside_scope"]["m_greater_than_64"] = "COMPILED_DEFAULT"
    with pytest.raises(planner.PlanError):
        planner.validate(value)


def test_analyzer_emits_independent_family_keyed_leaves(tmp_path):
    result = analyzer.analyze(make_run(tmp_path), tmp_path / "out")
    assert result["verdict"] == "FIVE_REAL_FAMILY_BOUNDED_REGRET_TABLES"
    assert [(row["family"]["n"], row["family"]["k"]) for row in result["families"]] == list(planner.FAMILIES)
    assert len({row["cells"][0]["point_winner"] for row in result["families"]}) == 5
    assert all(row["leaves"] and not row["unadjudicated_gaps"] for row in result["families"])
    assert result["measurement"] == {
        "families": 5,
        "m_values_per_family": 64,
        "categorical_candidates": 5,
        "rounds": 3,
        "samples_per_round": 11,
        "warmups_per_round": 3,
        "raw_bad_required": 0,
        "regret_threshold_pct": 3.0,
    }
    assert result["policy_contract"]["unknown_n_k"] == "NO_MEASURED_POLICY"
    assert result["policy_contract"]["m_greater_than_64"] == "A07_SCALEFIRST"
    assert result["policy_contract"]["compiled_default_is_measured_policy"] is False
    assert result["policy_contract"]["cross_family_extrapolation"] is False
    assert (tmp_path / "out/adjudication.json").is_file()
    lines = (tmp_path / "out/adjudication.tsv").read_text().splitlines()
    assert lines[0].startswith("N\tK\tkind")
    assert {tuple(map(int, line.split("\t")[:2])) for line in lines[1:]} == set(planner.FAMILIES)


@pytest.mark.parametrize(
    "plant",
    [
        "missing-log", "extra-log", "bad-checksum", "raw-bad", "wrong-map",
        "wrong-provider", "roundtrip", "wrong-family", "unknown-config",
        "duplicate-row", "short-samples", "wrong-median", "completion",
    ],
)
def test_raw_authority_plants_fail_closed(tmp_path, plant):
    root = make_run(tmp_path)
    log = root / "runs/q12-n1024-k5120-round3.log"
    text = log.read_text()
    first = next(line for line in text.splitlines() if line.startswith("FQ_KQUANT_LAYOUT_DENSE "))
    if plant == "missing-log":
        log.unlink(); log.with_suffix(".log.sha256").unlink()
    elif plant == "extra-log":
        extra = root / "runs/q12-n1-k1-round1.log"; extra.write_text("extra\n")
    elif plant == "bad-checksum":
        log.with_suffix(".log.sha256").write_text(f"{'f' * 64}  {log}\n")
    else:
        if plant == "raw-bad":
            text = text.replace("raw_bad=0", "raw_bad=1", 1)
        elif plant == "wrong-map":
            text = text.replace(planner.pilot.MAPPING_ID, "0x0000000000000000", 1)
        elif plant == "wrong-provider":
            text = text.replace("provider=standard-aiu", "provider=packed-row", 1)
        elif plant == "roundtrip":
            text = text.replace("roundtrip=PASS", "roundtrip=FAIL", 1)
        elif plant == "wrong-family":
            text = text.replace("shape=1x1024x5120", "shape=1x5120x8192", 1)
        elif plant == "unknown-config":
            text = text.replace(planner.CANDIDATES[0], "compiled-default", 1)
        elif plant == "duplicate-row":
            text = text.replace(first + "\n", first + "\n" + first + "\n", 1)
        elif plant == "short-samples":
            begin = first.index("samples=[") + len("samples=[")
            end = first.index("]", begin)
            vector = first[begin:end].split(",")[:-1]
            text = text.replace(first, first[:begin] + ",".join(vector) + first[end:], 1)
        elif plant == "wrong-median":
            text = text.replace("median_us=", "median_us=999", 1)
        else:
            text = text.replace("dense_cases=64", "dense_cases=63", 1)
        log.write_text(text)
        analyzer.refresh_sidecar(log)
    with pytest.raises(analyzer.AnalysisError):
        analyzer.analyze(root, tmp_path / "out")


def test_one_family_overlap_remains_a_local_gap(tmp_path):
    def timing(family, m, candidate, round_id):
        if family == 3 and m == 17:
            return 10.0 + candidate * 0.01 + (0.5 if round_id == 2 else 0.0)
        return analyzer._base_timing(family, m, candidate, round_id)

    result = analyzer.analyze(make_run(tmp_path, timing), tmp_path / "out")
    assert result["verdict"] == "UNADJUDICATED_GAPS"
    assert all(not row["unadjudicated_gaps"] for row in result["families"][:3])
    assert any(gap["m_min"] <= 17 <= gap["m_max"] for gap in result["families"][3]["unadjudicated_gaps"])
    assert not result["families"][4]["unadjudicated_gaps"]


def test_box_runner_is_execute_only_and_binds_both_sources():
    runner = (ROOT / "tools/run_fq_kquant_policy_v2_real_families_box.sh").read_text()
    assert "build.sh" not in runner and "PPU_BUILD_DIR" not in runner and "TARGET=" not in runner
    assert "425198f5d52377faf85eae4160cd44826e7f4388" in runner
    assert "FQ_KQUANT_POLICY_V2_BUILD_SOURCE" in runner
    assert 'runner_source="$(git -C "$root" rev-parse HEAD)"' in runner
    assert "$build_source/tools/fq_kquant_policy_v2_prebuilt.py" in runner
    assert "--execution-sdk-compatible" in runner
    assert 'probe["device_count"] == 1' in runner
    assert "for family in" in runner and "for round in 1 2 3" in runner
    assert "dense_cases=64" in runner and "raw_bad_required" in runner
    assert 'sha256sum "$log" >"$log.sha256"' in runner
    subprocess.run(["bash", "-n", str(ROOT / "tools/run_fq_kquant_policy_v2_real_families_box.sh")], check=True)

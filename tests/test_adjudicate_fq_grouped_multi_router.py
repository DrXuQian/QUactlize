"""Independent A05 grouped-selector adjudication tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "adjudicate_fq_grouped_multi_router.py"
SPEC = importlib.util.spec_from_file_location("fq_grouped_adjudicator", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def fixture(tmp_path: Path, scenario: str = "one") -> tuple[Path, Path]:
    plan = module.planner.materialize()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    runs = tmp_path / "runs"
    module._synthetic_runs(runs, plan, scenario)
    return plan_path, runs


@pytest.mark.parametrize(
    ("scenario", "verdict"),
    (
        ("one", "ONE_TACTIC_WITHIN_THRESHOLD"),
        ("abi", "ROUTE_FEATURES_ABI_SUFFICIENT"),
        ("insufficient", "ROUTE_FEATURES_INSUFFICIENT"),
        ("unresolved", "UNRESOLVED_OVERLAPPING_INTERVAL"),
    ),
)
def test_exact_verdicts_and_complete_outputs(
    tmp_path: Path, scenario: str, verdict: str
):
    plan, runs = fixture(tmp_path, scenario)
    output = tmp_path / "output"
    result = module.adjudicate(plan, runs, output, 2, 3, 1)
    assert result["verdict"] == verdict
    assert result["measurement_status"] == "COMPLETE"
    assert len(result["qtype_results"]) == 5
    assert result["selector_abi"]["measured_equivalence_fields"] == [
        "n",
        "k",
        "experts",
        "total_rows",
        "max_rows",
    ]
    assert "active" in result["selector_abi"]["unavailable_histogram_fields"]
    assert all(
        len(row["route_feature_classes"]) == 5 for row in result["qtype_results"]
    )
    assert len(result["profile_candidates"]) == 60
    assert {row["sample_count"] for row in result["profile_candidates"]} == {6}
    assert (output / "summary.json").is_file()
    assert len((output / "summary.tsv").read_text().splitlines()) == 6
    assert len((output / "candidates.tsv").read_text().splitlines()) == 61
    assert len((output / "common-candidates.tsv").read_text().splitlines()) == 21
    assert len(result["input_authority"]["raw_log_sha256"]) == 10


def test_unresolved_is_normal_exit_but_never_printed_as_pass(tmp_path: Path):
    plan, runs = fixture(tmp_path, "unresolved")
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "adjudicate",
            "--plan",
            str(plan),
            "--runs",
            str(runs),
            "--output",
            str(tmp_path / "cli-output"),
            "--rounds",
            "2",
            "--iterations",
            "3",
            "--warmups",
            "1",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "verdict=UNRESOLVED_OVERLAPPING_INTERVAL" in completed.stdout
    assert " PASS" not in completed.stdout


@pytest.mark.parametrize(
    ("before", "after"),
    (
        ("raw_bad=0", "raw_bad=1"),
        ("provider=standard-aiu", "provider=packed-row"),
        ("layout=kpack", "layout=xplane"),
        (" status=PASS", " status=FAIL"),
        ("samples=[", "samples=[99,"),
        ("config=B", "config=A"),
    ),
)
def test_planted_raw_log_negatives_are_rejected(
    tmp_path: Path, before: str, after: str
):
    plan, runs = fixture(tmp_path)
    target = runs / "q10-round2.log"
    body = target.read_text(encoding="utf-8")
    assert before in body
    target.write_text(body.replace(before, after, 1), encoding="utf-8")
    with pytest.raises(module.AdjudicationError):
        module.adjudicate(plan, runs, tmp_path / "output", 2, 3, 1)


def test_candidate_set_drift_and_log_denominator_are_rejected(tmp_path: Path):
    plan, runs = fixture(tmp_path)
    target = runs / "q10-round2.log"
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text(
        "\n".join(
            line
            for line in lines
            if not (
                line.startswith("FQ_GROUPED_ROUTER_CELL ")
                and "profile=balanced" in line
                and "config=B" in line
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(module.AdjudicationError, match="candidate set differs"):
        module.adjudicate(plan, runs, tmp_path / "drift", 2, 3, 1)

    module._synthetic_runs(runs, json.loads(plan.read_text()), "one")
    (runs / "q15-round1.log").write_text("extra\n", encoding="utf-8")
    with pytest.raises(module.AdjudicationError, match="raw log denominator"):
        module.adjudicate(plan, runs, tmp_path / "extra", 2, 3, 1)


def test_plan_is_authority_not_an_old_summary(tmp_path: Path):
    plan, runs = fixture(tmp_path)
    (runs.parent / "summary.json").write_text(
        json.dumps({"verdict": "PILOT_COMPLETE", "winner": "fabricated"}),
        encoding="utf-8",
    )
    result = module.adjudicate(plan, runs, tmp_path / "output", 2, 3, 1)
    assert result["verdict"] == "ONE_TACTIC_WITHIN_THRESHOLD"

    document = json.loads(plan.read_text(encoding="utf-8"))
    document["cells"][0]["mapping_id"] = "0x0"
    plan.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(module.AdjudicationError, match="plan authority differs"):
        module.adjudicate(plan, runs, tmp_path / "red", 2, 3, 1)


def test_builtin_self_test():
    module.self_test()

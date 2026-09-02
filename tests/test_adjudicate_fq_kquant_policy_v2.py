"""Tests for conservative A04 Q4 policy-v2 adjudication."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
MODULE_PATH = ROOT / "tools/adjudicate_fq_kquant_policy_v2.py"
SPEC = importlib.util.spec_from_file_location("fq_policy_adjudication", MODULE_PATH)
adjudicator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(adjudicator)


def make_run(tmp_path, timing=adjudicator._base_timing):
    root = tmp_path / "run"
    adjudicator._write_synthetic_run(root, timing)
    return root


def refresh_sidecar(log):
    log.with_suffix(".log.sha256").write_text(f"{adjudicator.sha256(log)}  {log}\n")


def test_adjudicates_only_measured_categorical_leaves(tmp_path):
    result = adjudicator.adjudicate(make_run(tmp_path), tmp_path / "out")
    assert result["verdict"] == "BOUNDED_REGRET_CATEGORICAL_LEAVES"
    assert [
        (leaf["m_min"], leaf["m_max"], leaf["config"]) for leaf in result["leaves"]
    ] == [
        (1, 8, adjudicator.planner.CANDIDATES[0]),
        (9, 32, adjudicator.planner.CANDIDATES[1]),
        (33, 64, adjudicator.planner.CANDIDATES[4]),
    ]
    assert not result["unadjudicated_gaps"]
    assert result["measurement"]["rounds"] == 3
    assert result["measurement"]["samples_per_round"] == 11
    assert result["policy_contract"] == {
        "kind": "categorical-measured-leaves",
        "config_names_are_categorical": True,
        "compiled_default_is_measured_policy": False,
        "outside_measured_scope": "NO_MEASURED_POLICY",
        "interval_semantics": "inclusive contiguous integers; every M in every leaf was measured",
    }
    assert result["scope"]["verdict"] == "SINGLE_REAL_FAMILY_M1_TO_M64_ONLY"
    assert len(result["scope"]["unproven_real_families"]) == 4
    assert result["scope"]["m_greater_than_64_proven"] is False
    assert (tmp_path / "out/adjudication.json").is_file()
    assert (tmp_path / "out/adjudication.tsv").read_text().count("\n") == 4


@pytest.mark.parametrize(
    "plant",
    [
        "missing-m",
        "missing-candidate",
        "duplicate-candidate",
        "raw-bad",
        "wrong-map",
        "wrong-provider",
        "short-samples",
        "wrong-median",
        "wrong-round-count",
        "bad-checksum",
    ],
)
def test_raw_authority_plants_fail_closed(tmp_path, plant):
    root = make_run(tmp_path)
    log = root / "runs/q12-round3.log"
    text = log.read_text()
    first = next(
        line for line in text.splitlines() if line.startswith("FQ_KQUANT_LAYOUT_DENSE ")
    )
    if plant == "missing-m":
        text = (
            "\n".join(
                line for line in text.splitlines() if " shape=64x1024x5120 " not in line
            )
            + "\n"
        )
    elif plant == "missing-candidate":
        text = text.replace(first + "\n", "", 1)
    elif plant == "duplicate-candidate":
        text = text.replace(first + "\n", first + "\n" + first + "\n", 1)
    elif plant == "raw-bad":
        text = text.replace("raw_bad=0", "raw_bad=1", 1)
    elif plant == "wrong-map":
        text = text.replace(adjudicator.planner.MAPPING_ID, "0x0", 1)
    elif plant == "wrong-provider":
        text = text.replace("provider=standard-aiu", "provider=packed-row", 1)
    elif plant == "short-samples":
        start = first.index("samples=[") + len("samples=[")
        end = first.index("]", start)
        vector = first[start:end].split(",")[:-1]
        text = text.replace(first, first[:start] + ",".join(vector) + first[end:], 1)
    elif plant == "wrong-median":
        text = text.replace("median_us=", "median_us=999", 1)
    elif plant == "wrong-round-count":
        (root / "runs/q12-round2.log").unlink()
        (root / "runs/q12-round2.log.sha256").unlink()
    elif plant == "bad-checksum":
        log.with_suffix(".log.sha256").write_text(f"{'f' * 64}  {log}\n")
    if plant not in ("wrong-round-count", "bad-checksum"):
        log.write_text(text)
        refresh_sidecar(log)
    with pytest.raises(adjudicator.AdjudicationError):
        adjudicator.adjudicate(root, tmp_path / "out")


def test_near_neighbour_overlap_is_not_promoted_to_policy(tmp_path):
    def timing(m, candidate, round_id):
        if m == 9:
            return 10.0 + candidate * 0.01 + (0.5 if round_id == 2 else 0.0)
        return adjudicator._base_timing(m, candidate, round_id)

    result = adjudicator.adjudicate(make_run(tmp_path, timing), tmp_path / "out")
    assert result["verdict"] == "UNADJUDICATED_GAPS"
    assert result["cells"][8]["m"] == 9
    assert result["cells"][8]["point_winner_resolution"] == "OVERLAPPING_ENVELOPES"
    gap = next(
        gap for gap in result["unadjudicated_gaps"] if gap["m_min"] <= 9 <= gap["m_max"]
    )
    assert gap["reason"] == "NO_CANDIDATE_PROVEN_WITHIN_BOUNDED_REGRET"
    assert not any(leaf["m_min"] <= 9 <= leaf["m_max"] for leaf in result["leaves"])


def test_overlapping_winners_can_still_have_a_proven_bounded_regret_leaf(tmp_path):
    def timing(m, candidate, round_id):
        if m == 9 and candidate in (0, 1):
            return 10.0 + candidate * 0.001 + (round_id - 2) * 0.002
        return adjudicator._base_timing(m, candidate, round_id)

    result = adjudicator.adjudicate(make_run(tmp_path, timing), tmp_path / "out")
    cell = result["cells"][8]
    assert cell["point_winner_resolution"] == "OVERLAPPING_ENVELOPES"
    assert cell["admissible_configs"]
    assert any(leaf["m_min"] <= 9 <= leaf["m_max"] for leaf in result["leaves"])
    assert not result["unadjudicated_gaps"]

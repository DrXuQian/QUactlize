from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import plan_kpack_screen_retention as retention  # noqa: E402


def fixture() -> dict:
    return retention._fixture()


def retained_ids(result: dict) -> set[str]:
    return {row["candidate_id"] for row in result["retained"]}


def excluded_ids(result: dict) -> set[str]:
    return {row["candidate_id"] for row in result["excluded"]}


def test_global_envelope_excludes_only_cleanly_separated_candidate() -> None:
    result = retention.derive(fixture())
    slow = f"{2:064x}"
    assert excluded_ids(result) == {slow}
    proof = result["excluded"][0]["proof"]
    assert proof["witness_candidate_id"] == f"{1:064x}"
    assert proof["candidate_lower_us"] > (
        proof["witness_upper_us"] * proof["margin_multiplier"])
    assert result["policy"]["top_n"] is None
    assert result["policy"]["candidate_cap"] is None


def test_boundary_and_noise_are_retained() -> None:
    value = fixture()
    # 10.1 * 1.10 = 11.11: strict separation is required, equality/overlap stays.
    value["candidates"][1]["timing"]["samples_us"] = [11.11, 11.11]
    result = retention.derive(value)
    assert f"{2:064x}" in retained_ids(result)
    reasons = {row["candidate_id"]: row["reasons"] for row in result["retained"]}
    assert "OVERLAPPING_OBSERVED_ENVELOPE" in reasons[f"{2:064x}"]
    assert "UNCERTAIN_SCREEN_SPREAD" in reasons[f"{3:064x}"]


def test_historical_missing_reducer_and_noncomparable_are_never_pruned() -> None:
    result = retention.derive(fixture())
    reasons = {row["candidate_id"]: set(row["reasons"])
               for row in result["retained"]}
    assert "HISTORICAL_ANCHOR" in reasons[f"{4:064x}"]
    assert "MISSING_EXACT_REDUCER" in reasons[f"{5:064x}"]
    assert "NONCOMPARABLE_E2E" in reasons[f"{6:064x}"]


def test_family_and_axis_sentinels_preserve_unexplored_geometry() -> None:
    value = fixture()
    work = value["work_items"][0]
    candidate = copy.deepcopy(value["candidates"][1])
    candidate["candidate_id"] = f"{7:064x}"
    candidate["symbol"] = "unique-geometry"
    candidate["timing"]["samples_us"] = [100.0, 100.1]
    candidate["family_key"] = "new-provider"
    candidate["axes"] = {"tile_m": 1024, "tile_n": 64}
    value["candidates"].append(candidate)
    work["candidate_ids"].append(candidate["candidate_id"])
    result = retention.derive(value)
    reasons = next(row["reasons"] for row in result["retained"]
                   if row["candidate_id"] == candidate["candidate_id"])
    assert "FAMILY_SENTINEL:new-provider" in reasons
    assert "AXIS_SENTINEL:tile_m=1024" in reasons


def test_symbol_granularity_closes_over_runtime_siblings() -> None:
    value = fixture()
    sibling = copy.deepcopy(value["candidates"][1])
    sibling["candidate_id"] = f"{8:064x}"
    sibling["symbol"] = "anchor"
    sibling["timing"]["samples_us"] = [100.0, 100.1]
    value["candidates"].append(sibling)
    value["work_items"][0]["candidate_ids"].append(sibling["candidate_id"])
    result = retention.derive(value)
    reasons = next(row["reasons"] for row in result["retained"]
                   if row["candidate_id"] == sibling["candidate_id"])
    assert reasons == ["SYMBOL_GRANULARITY_SIBLING"]
    assert sibling["candidate_id"] not in excluded_ids(result)


def test_available_reducer_uncertainty_participates_in_envelope() -> None:
    value = fixture()
    fast, slow = value["candidates"][0], value["candidates"][1]
    authority = "9" * 64
    fast["reducer"] = {"status": "AVAILABLE", "lower_us": 1.0,
                       "upper_us": 5.0, "authority_sha256": authority}
    slow["reducer"] = {"status": "AVAILABLE", "lower_us": 1.0,
                       "upper_us": 1.0, "authority_sha256": authority}
    # Fast producer's conservative total upper is now 15.1, so the producer-
    # slower candidate is no longer proved slower end to end.
    result = retention.derive(value)
    assert slow["candidate_id"] in retained_ids(result)


def test_structural_work_item_gets_an_explicit_empty_selection() -> None:
    value = fixture()
    work_id = "c" * 64
    terminal = copy.deepcopy(value["candidates"][0])
    terminal.update({
        "candidate_id": f"{9:064x}", "work_item_id": work_id,
        "symbol": "structural", "historical_anchor": False,
        "classification": "STRUCTURAL_UNAVAILABLE",
        "terminal_reason": "INADMISSIBLE_SHARED_STORAGE",
        "timing": {"samples_us": []},
        "comparison": {"status": "NONCOMPARABLE", "key": None},
        "reducer": {"status": "NOT_REQUIRED"},
    })
    value["candidates"].append(terminal)
    value["work_items"].append({
        "work_item_id": work_id, "qtype": 12, "operator": "dense",
        "workload_key": "m8-n5120-k8192", "route": "scalefirst",
        "candidate_ids": [terminal["candidate_id"]],
    })
    value["workloads"][0]["work_item_ids"].append(work_id)
    result = retention.derive(value)
    selected = next(row for row in result["work_items"]
                    if row["work_item_id"] == work_id)
    assert selected["selected_symbols"] == []
    assert selected["structural_candidate_ids"] == [terminal["candidate_id"]]
    assert result["denominator"]["empty_work_items"] == 1


@pytest.mark.parametrize("plant", ["missing-candidate", "duplicate-candidate",
                                    "missing-work-item", "duplicate-work-item"])
def test_ownership_plants_fail_closed(plant: str) -> None:
    value = fixture()
    if plant == "missing-candidate":
        value["work_items"][0]["candidate_ids"].pop()
    elif plant == "duplicate-candidate":
        value["work_items"][1]["candidate_ids"].append(
            value["work_items"][0]["candidate_ids"][0])
    elif plant == "missing-work-item":
        value["workloads"][0]["work_item_ids"].pop()
    else:
        value["workloads"][0]["work_item_ids"].append(
            value["workloads"][0]["work_item_ids"][0])
    with pytest.raises(retention.RetentionError, match="ownership"):
        retention.derive(value)


def test_output_is_bound_to_exact_input_file(tmp_path: Path) -> None:
    value = fixture()
    source = tmp_path / "screen.json"
    source.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    result = retention.derive(value, input_sha256=retention.file_sha(source))
    retention.validate_output(
        result, value, input_sha256=retention.file_sha(source))
    changed = copy.deepcopy(value)
    changed["candidates"][0]["timing"]["samples_us"][0] = 9.9
    with pytest.raises(retention.RetentionError, match="differs"):
        retention.validate_output(
            result, changed, input_sha256=retention.file_sha(source))

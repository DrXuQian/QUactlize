from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import plan_fq_splitk_reducer_lookup as reducer_plan


@pytest.fixture(scope="module")
def plan():
    return reducer_plan.materialize()


def test_exact_dense_reducer_denominator(plan):
    assert plan["denominator"] == {
        "source_dense_cells": 1001,
        "unique_m_n": 345,
        "splits": [2, 4, 8],
        "cases": 1035,
        "grouped_splitk": "STRUCTURAL_UNAVAILABLE",
        "grouped_splitk_structural_cells": 15,
    }
    assert [(row["m"], row["n"], row["split"])
            for row in plan["cases"]] == sorted({
                (row["m"], row["n"], row["split"])
                for row in plan["cases"]})
    assert [row["ordinal"] for row in plan["cases"]] == list(range(1035))


def test_bytes_and_shipping_dispatch_identity(plan):
    for row in plan["cases"]:
        assert row["workspace_bytes"] == (
            row["m"] * row["n"] * row["split"] * 4)
        assert row["output_bytes"] == row["m"] * row["n"] * 2
        expected = (reducer_plan.FAST_IMPLEMENTATION
                    if row["m"] == 1 else reducer_plan.GENERIC_IMPLEMENTATION)
        assert row["expected_implementation"] == expected
    assert sum(row["expected_implementation"] ==
               reducer_plan.FAST_IMPLEMENTATION
               for row in plan["cases"]) == 27
    assert plan["measurement"]["rounds"] == [
        {"round": index + 1, "schedule_seed": f"0x{seed:016x}"}
        for index, seed in enumerate(reducer_plan.ROUND_SEEDS)]
    assert plan["measurement"]["warmups"] == 3
    assert plan["measurement"]["samples"] == 11
    assert plan["measurement"]["top_n"] is None
    assert plan["measurement"]["point_estimate_pruning"] is False


def test_generated_cpp_include_is_complete(plan):
    text = reducer_plan.cpp_include(plan).decode("ascii")
    assert text.count("  X(") == 1035
    assert reducer_plan.digest(plan) in text
    assert reducer_plan.FQ_INCLUDE_SENTINEL in text
    assert "top_n" not in text


@pytest.mark.parametrize("mutation", [
    lambda value: value["cases"].pop(),
    lambda value: value["cases"][0].update(split=3),
    lambda value: value["cases"][0].update(workspace_bytes=4),
    lambda value: value["measurement"].update(top_n=1),
    lambda value: value["denominator"].update(cases=1034),
])
def test_plan_mutations_fail_closed(plan, mutation):
    broken = copy.deepcopy(plan)
    mutation(broken)
    with pytest.raises(reducer_plan.PlanError):
        reducer_plan.validate_plan(broken, plan)

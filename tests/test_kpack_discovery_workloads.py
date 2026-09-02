"""Contract tests for the complete K-pack discovery workload projection."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import materialize_kpack_discovery_workloads as workloads  # noqa: E402
import plan_fq_kpack_route_optimal as plan_module  # noqa: E402


def write_plan(path: Path) -> dict:
    document = plan_module.materialize()
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def test_complete_projection_is_frozen_and_validated(tmp_path: Path):
    plan = tmp_path / "plan.json"
    write_plan(plan)
    output = tmp_path / "workloads"
    first = workloads.materialize(plan, output)
    second = workloads.materialize(plan, output)
    assert first == second == workloads.validate(plan, output)
    assert (first["format_cells"], first["dense_cells"],
            first["grouped_cells"], first["router_control_cells"]) == (
                1381, 1001, 380, 120)


def test_router_permutations_remain_distinct(tmp_path: Path):
    plan = tmp_path / "plan.json"
    write_plan(plan)
    output = tmp_path / "workloads"
    workloads.materialize(plan, output)
    a = (output / "router-rows/permutation-a.txt").read_bytes()
    b = (output / "router-rows/permutation-b.txt").read_bytes()
    assert a != b
    (output / "router-rows/permutation-b.txt").write_bytes(a)
    with pytest.raises(workloads.WorkloadError):
        workloads.validate(plan, output)


@pytest.mark.parametrize("source_class", ["router-control", "historical-anchor"])
def test_missing_non_real_cell_fails_closed(tmp_path: Path, source_class: str):
    plan_path = tmp_path / "plan.json"
    document = write_plan(plan_path)
    document["cells"] = [
        row for row in document["cells"]
        if row["source_class"] != source_class
    ]
    plan_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with pytest.raises(workloads.WorkloadError):
        workloads.materialize(plan_path, tmp_path / "workloads")

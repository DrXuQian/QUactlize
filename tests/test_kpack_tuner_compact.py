import copy
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import export_kpack_tuner_compact as compact


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compact.encoded(data))


def cell(symbol, us, grid=0):
    return {
        "symbol": symbol,
        "algorithm": "TC_S1",
        "split": 1,
        "grid": grid,
        "status": "MEASURED",
        "median_us": us,
        "samples_us": [us] * 3,
    }


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "campaign"
    source.mkdir()
    (source / "night.lock").touch()
    request = {
        "cell_key": "q12-m1-n1024-k5120",
        "route": "fq-dense",
        "qtype": 12,
        "problem": {"m": 1, "n": 1024, "k": 5120},
        "source_class": "real-inventory",
        "workload_key": "dense_m1_n1024_k5120",
        "grouped": None,
        "worker_index": 0,
        "symbols": ["fast", "middle", "winner"],
        "reasons": {},
    }
    request["id"] = compact.digest([request["cell_key"], request["route"]])
    candidates = [{"symbol": s, "tm": 8} for s in request["symbols"]]
    plan = {"requests": [request], "candidates": candidates, "router_files": {}}
    final = {
        "schema": "quactlize.kpack-overnight.v1",
        "status": "INCOMPLETE",
        "requests": 1,
        "confirmed_requests": 0,
        "global_5pct_bound_proven": False,
        "rows": [
            {
                "cell_key": request["cell_key"],
                "route": request["route"],
                "status": "NOISY_CONFIRMATION",
                "symbol": "winner",
                "algorithm": "TC_S1",
                "split": 1,
                "grid": 9,
                "median_us": 11,
                "round_medians_us": [10, 11, 12],
            }
        ],
    }
    write(source / "base-plan.json", plan)
    write(source / "results/summary.json", final)
    write(source / "campaign-identity.json", {"kernel_source": "test-source"})
    prefix = source / "phases/screen"
    write(prefix / "plan.json", plan)
    bundle = {"identity": {"source": "test-source"}, "payloads": {}}
    write(prefix / "bundle.json", bundle)
    write(
        prefix / "run/epoch.json",
        {
            "devices": [0],
            "iterations": 3,
            "plan_sha256": compact.digest(plan),
            "bundle_sha256": compact.digest(bundle),
            "assignment": [[request["id"]]],
        },
    )
    write(
        prefix / "run/results" / (request["id"] + ".json"),
        {
            "request_sha256": compact.digest(request),
            "device": 0,
            "status": "MEASURED",
            "cells": [cell("fast", 9), cell("middle", 10), cell("winner", 11, 9)],
            "rejected": {"bad": {"rc": 1, "log": "box-only/raw-failure.log"}},
        },
    )
    raw = prefix / "run/logs/raw.log"
    raw.parent.mkdir(parents=True)
    raw.write_text("RAW_DO_NOT_INCLUDE\n")
    return source


def test_small_export_preserves_status_identity_final_and_failure_index(source):
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = compact.Exporter(source, 1).export()
    assert result["final_summary"]["status"] == "INCOMPLETE"
    row = result["workloads"][0]
    assert row["final"]["status"] == "NOISY_CONFIRMATION"
    assert row["request"]["problem"] == {"m": 1, "n": 1024, "k": 5120}
    stage = row["stages"]["screen"]
    assert [c["symbol"] for c in stage["top"]] == ["fast", "winner"]
    assert stage["omitted_measured_variants"] == 1
    assert set(result["configurations"]) == {"fast", "winner"}
    assert result["failures"][0]["rejected"]["bad"]["log"] == "box-only/raw-failure.log"
    assert result["phases"]["confirm-1"]["status"] == "NOT_PLANNED"
    assert not result["raw_logs_replayed"]
    encoded = compact.encoded(result)
    assert b"RAW_DO_NOT_INCLUDE" not in encoded and b"samples_us" not in encoded
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_exact_variant_and_distinct_parent_topk():
    cells = [cell("a", 1), cell("a", 2, 1), cell("b", 3), cell("c", 4)]
    result = compact.top_cells(cells, {"a", "b", "c"}, 3, 2, cell("c", 4))
    assert [c["symbol"] for c in result["top"]] == ["a", "b", "c"]
    assert result["measured_variants"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cells: cells[0]["samples_us"].pop(),
        lambda cells: cells[0]["samples_us"].__setitem__(0, float("nan")),
        lambda cells: cells[0].__setitem__("median_us", 99),
        lambda cells: cells.append(copy.deepcopy(cells[0])),
        lambda cells: cells[0].__setitem__("symbol", "foreign"),
    ],
)
def test_invalid_cells_cannot_become_measured(mutation):
    cells = [cell("a", 1)]
    mutation(cells)
    with pytest.raises(ValueError):
        compact.top_cells(cells, {"a"}, 3, 1, {})


def test_missing_receipt_is_explicit_not_pass(source):
    # Repoint the fixture through a fresh phase with no result receipt.
    plan = json.loads((source / "base-plan.json").read_text())
    write(source / "phases/audit/plan.json", plan)
    result = compact.Exporter(source, 3).export()
    assert result["workloads"][0]["stages"]["audit"]["status"] == "MISSING"
    assert result["phases"]["audit"]["request_status_counts"] == {"MISSING": 1}


def test_denominator_and_wrong_epoch_rejected(source):
    epoch_path = source / "phases/screen/run/epoch.json"
    epoch = json.loads(epoch_path.read_text())
    epoch["plan_sha256"] = "wrong"
    write(epoch_path, epoch)
    with pytest.raises(ValueError, match="receipt"):
        compact.Exporter(source, 3).export()
    final_path = source / "results/summary.json"
    final = json.loads(final_path.read_text())
    final["requests"] = 2
    write(final_path, final)
    with pytest.raises(ValueError, match="denominator"):
        compact.Exporter(source, 3).export()


def test_cli_size_cap_and_no_overwrite(source, tmp_path, monkeypatch):
    output = tmp_path / "compact.json.gz"
    argv = ["export", "--source", str(source), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv + ["--max-mib", "0.00001"])
    with pytest.raises(SystemExit):
        compact.main()
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", argv)
    compact.main()
    compressed = output.read_bytes()
    assert (
        json.loads(gzip.decompress(compressed))["schema"]
        == "quactlize.kpack-compact-review.v1"
    )
    with pytest.raises(SystemExit):
        compact.main()
    assert output.read_bytes() == compressed

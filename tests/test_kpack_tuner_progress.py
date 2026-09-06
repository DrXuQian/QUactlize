import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import watch_kpack_tuner as watch


def observation(counts, phase="screen/test", status="RUNNING"):
    return {
        "phase": phase,
        "status": status,
        "counters": counts,
        "completed": sum(c[0] for c in counts.values()),
        "total": sum(c[1] for c in counts.values()),
        "failed": 0,
    }


def test_eta_uses_slowest_worker_not_global_average():
    eta = watch.RollingETA()
    assert eta.update(observation({"0": (0, 100), "1": (0, 20)}), 0) is None
    assert eta.update(observation({"0": (80, 100), "1": (1, 20)}), 30) is None
    # The first burst could be cached results; it is not a runtime sample.
    assert eta.update(observation({"0": (100, 100), "1": (2, 20)}), 60) == 540


def test_stalled_worker_is_unknown_not_zero():
    eta = watch.RollingETA()
    state = observation({"0": (0, 10), "1": (0, 10)})
    eta.update(state, 0)
    eta.update(state, 30)
    assert eta.update(observation({"0": (9, 10), "1": (0, 10)}), 60) is None


def test_reset_phase_resume_and_finished():
    eta = watch.RollingETA()
    eta.update(observation({"compile": (0, 100)}, "screen/compile"), 0)
    eta.update(observation({"compile": (80, 100)}, "screen/compile"), 30)
    assert eta.update(observation({"compile": (90, 100)}, "screen/compile"), 60) == 30
    # Resumption/reduced counters invalidates the old rate.
    assert eta.update(observation({"compile": (0, 100)}, "screen/compile"), 90) is None
    assert eta.update(observation({"0": (0, 20)}, "screen/test"), 120) is None
    assert eta.update(observation({"0": (20, 20)}, "screen/test"), 150) == 0
    assert eta.update(observation({"0": (20, 20)}, status="STOPPED"), 160) is None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_snapshot_build_link_and_test_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "campaign_running", lambda _: True)
    folder = tmp_path / "phases/screen"
    write_json(
        folder / "plan.json",
        {
            "requests": [
                {"id": "a", "worker_index": 0},
                {"id": "b", "worker_index": 1},
            ]
        },
    )
    (folder / "build.log").write_text(
        "KPACK_TUNER_BUILD completed=9/10 status=PASS unit=x\n"
    )
    reader = watch.Reader()
    state = watch.snapshot(tmp_path, reader)
    assert state["phase"] == "screen/compile"
    assert state["counters"] == {"compile": (9, 10)}
    with (folder / "build.log").open("a") as stream:
        stream.write("KPACK_TUNER_BUILD completed=10/10 status=PASS unit=y\n")
    state = watch.snapshot(tmp_path, reader)
    assert state["phase"] == "screen/link" and not state["counters"]
    assert "remaining_minutes=ESTIMATING_LINK" in watch.render(state, None, 30)
    write_json(folder / "bundle.json", {})
    write_json(folder / "run/results/a.json", {"status": "MEASURED", "cells": [1]})
    write_json(folder / "run/results/b.json", {"cells": [1]})
    state = watch.snapshot(tmp_path, reader)
    assert state["phase"] == "screen/test"
    assert state["counters"] == {"0": (1, 1), "1": (0, 1)}
    assert state["completed"] == 1
    write_json(folder / "run/results/b.json", {"status": "NO_VALID_CANDIDATE"})
    state = watch.snapshot(tmp_path, reader)
    assert state["completed"] == 2 and state["failed"] == 1


def test_monitor_never_changes_artifacts_and_ignores_old_summary_when_running(
    tmp_path, monkeypatch
):
    write_json(
        tmp_path / "results/summary.json",
        {
            "status": "INCOMPLETE",
            "confirmed_requests": 0,
            "requests": 5,
        },
    )
    files = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(watch, "campaign_running", lambda _: True)
    assert watch.snapshot(tmp_path, watch.Reader())["phase"] == "planning"
    monkeypatch.setattr(watch, "campaign_running", lambda _: False)
    state = watch.snapshot(tmp_path, watch.Reader())
    assert state["phase"] == "finished" and state["status"] == "INCOMPLETE"
    assert "remaining_minutes=NOT_RUNNING" in watch.render(state, None, 0)
    assert files == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_json_cache_handles_partial_receipts_and_only_caches_status(tmp_path):
    reader, path = watch.Reader(), tmp_path / "result.json"
    path.write_text("{")
    assert reader.json(path) is None
    write_json(path, {"status": "MEASURED", "cells": list(range(10))})
    assert reader.json(path, ("status",)) == {"status": "MEASURED"}
    assert next(iter(reader.cache.values()))[1] == {"status": "MEASURED"}


def test_display_eta_is_not_budget_or_whole_campaign():
    text = watch.render(observation({"0": (5, 10)}), 120, 60)
    assert "remaining_minutes=2.0" in text
    assert "eta_scope=CURRENT_PHASE" in text
    assert "campaign_remaining_minutes=UNKNOWN_ADAPTIVE_STAGES" in text
    assert "advisory_only=1" in text


def test_missing_output_is_not_created(tmp_path, monkeypatch):
    output = tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", ["watch", "--output", str(output), "--once"])
    with pytest.raises(SystemExit):
        watch.main()
    assert not output.exists()

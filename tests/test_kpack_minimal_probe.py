import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_kpack_minimal_probe as probe


def test_exact_minimal_probe_scope():
    plan = probe.make_plan()
    assert plan["denominator"] == {
        "workloads": 4,
        "route_workloads": 8,
        "selected_parent_workloads": 8,
        "compile_parent_union": 4,
    }
    assert {r["route"] for r in plan["requests"]} == set(probe.tuning.ROUTES)
    assert all(r["qtype"] == 12 and len(r["symbols"]) == 1 for r in plan["requests"])
    assert {
        r["problem"]["m"] for r in plan["requests"] if r["route"].endswith("dense")
    } == {1, 2048}
    assert {r["grouped"]["tokens"] for r in plan["requests"] if r["grouped"]} == {
        "1",
        "2048",
    }
    assert all(
        tuple(c[k] for k in ("tm", "tn", "tk", "wm", "wn", "stages"))
        == (16, 64, 64, 16, 16, 2)
        for c in plan["candidates"]
    )


def test_timed_command_failure_and_timeout():
    rc, seconds = probe.timed_command([sys.executable, "-c", "raise SystemExit(7)"], 3)
    assert rc == 7 and seconds < 3
    rc, seconds = probe.timed_command(
        [sys.executable, "-c", "import time; time.sleep(10)"], 0.1
    )
    assert rc == 124 and seconds < 3

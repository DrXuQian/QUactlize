#!/usr/bin/env python3
"""Minimal Q4 compile + device wall-time probe. Never starts a full sweep."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import kpack_tuning_plan as tuning


def make_plan():
    plan = tuning.make_plan(qtypes=(12,), pilot=True)
    union = {}
    for r in plan["requests"]:
        rows = [
            c
            for c in tuning.candidates(12, r["route"])
            if c.geometry == (16, 64, 64, 16, 16, 2)
            and c.ap == 0
            and c.dn == 16
            and c.persistent != 0
        ]
        if len(rows) != 1 or not tuning.admissible(rows[0], r["problem"]):
            raise ValueError(f"minimal probe type is absent/ambiguous for {r['route']}")
        c = rows[0]
        r["symbols"] = [c.symbol]
        r["reasons"] = {c.symbol: "MINIMAL_TIMING_PROBE_NOT_TUNING"}
        r.pop("anchor_recall", None)
        union[c.symbol] = asdict(c)
    if len(plan["requests"]) != 8 or len(union) != 4:
        raise ValueError(
            "minimal probe must have four parents and eight route/workloads"
        )
    plan.update(
        scope="minimal-compile-and-runtime-probe",
        budget_parents=1,
        selection="ONE_FIXED_TYPE_PER_ROUTE_NOT_A_HEURISTIC",
        candidates=[union[s] for s in sorted(union)],
        denominator={
            "workloads": 4,
            "route_workloads": 8,
            "selected_parent_workloads": 8,
            "compile_parent_union": 4,
        },
    )
    return plan


def timed_command(argv, seconds):
    start = time.monotonic()
    proc = subprocess.Popen(argv, start_new_session=True)
    try:
        rc = proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        # The builder/runner handles TERM and preserves completed artifacts.
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        rc = 124
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=60)
        rc = 130
    return rc, time.monotonic() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=192)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--phase-timeout-minutes", type=float, default=30)
    p.add_argument("--build-only", action="store_true")
    a = p.parse_args()
    if a.jobs < 1 or a.device < 0 or not 0 < a.phase_timeout_minutes <= 60:
        p.error("jobs positive, device nonnegative, phase timeout in (0,60] required")
    if not a.output.is_absolute() or a.output.is_symlink():
        p.error("output must be absolute and not a symlink")
    a.output, a.sdk = a.output.resolve(), a.sdk.resolve()
    if str(a.output) in ("/", "/root", "/workspace", "/root/autodl-tmp"):
        p.error("output must be a dedicated child directory")
    if not (a.sdk / "bin/hgcc").is_file() or not (a.sdk / "bin/hgobjdump").is_file():
        p.error("SDK must contain bin/hgcc and bin/hgobjdump")
    os.environ["PATH"] = str(a.sdk / "bin") + ":" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(str(a.sdk / d) for d in ("lib", "lib64", "targets/x86_64-linux/lib"))
        + ":"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output / "probe.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = a.output / "plan.json"
    start = time.monotonic()
    plan = make_plan()
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        p.error("existing plan differs; use a new output")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    planning_seconds = time.monotonic() - start
    print(
        "KPACK_MINIMAL_PLAN parents=4 drivers=4 compile_units=9 route_workloads=8 "
        "format=Q4_K config=16x64x64_w16x16_s2_ap0_dn16 full_sweep=DISABLED",
        flush=True,
    )
    prefix = [sys.executable, "-u", "-B"]
    cache_existed = (a.output / "build/bundle.json").exists()
    rc, build_seconds = timed_command(
        prefix
        + [
            str(tuning.ROOT / "tools/build_kpack_tuner.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(a.output / "build"),
            "--sdk",
            str(a.sdk),
            "--jobs",
            str(a.jobs),
            "--parents-per-module",
            "1",
        ],
        a.phase_timeout_minutes * 60,
    )
    print(
        f"KPACK_MINIMAL_BUILD rc={rc} seconds={build_seconds:.3f} jobs={a.jobs}",
        flush=True,
    )
    report = {
        "plan_seconds": planning_seconds,
        "build_seconds": build_seconds,
        "build_rc": rc,
        "runtime_seconds": None,
        "runtime_rc": None,
        "device_validated": False,
        "full_sweep_started": False,
        "full_campaign_duration_proven": False,
        "build_cache_existed": cache_existed,
        "logical_cpus": os.cpu_count(),
        "compile_units": 9,
        "requested_jobs": a.jobs,
    }
    if rc == 0 and not a.build_only:
        rc, elapsed = timed_command(
            prefix
            + [
                str(tuning.ROOT / "tools/run_kpack_tuner.py"),
                "--plan",
                str(plan_path),
                "--bundle",
                str(a.output / "build/bundle.json"),
                "--output",
                str(a.output / "run"),
                "--sdk",
                str(a.sdk),
                "--devices",
                str(a.device),
                "--iterations",
                "3",
            ],
            a.phase_timeout_minutes * 60,
        )
        report.update(runtime_seconds=elapsed, runtime_rc=rc, device_validated=rc == 0)
        print(
            f"KPACK_MINIMAL_RUN rc={rc} seconds={elapsed:.3f} correctness_repeats=1 samples=3",
            flush=True,
        )
    (a.output / "timing.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"KPACK_MINIMAL_DONE rc={rc} timing={a.output/'timing.json'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Executable controls for benchmarks/sweep_gemv_perf.py.

This is intentionally standalone until the C++ raw writer lands.  It exercises
the analyser's planted lattice cases and the driver's dry-run, timeout, resume,
private-output validation, and progress contracts with a temporary fake child.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent.parent
SWEEP = ROOT / "benchmarks" / "sweep_gemv_perf.py"


FAKE = r'''#!/usr/bin/env python3
import json, os, struct, sys, time
job = sys.argv[1]
marker = os.environ["FAKE_MARKER"]
with open(marker, "a") as h:
    h.write(job + "\n")
if job == "slow" and os.environ.get("ALLOW_SLOW") != "1":
    time.sleep(2)
out = os.environ["GEMV_SWEEP_JSONL"]
run = os.environ["GEMV_SWEEP_RUN_ID"]
attempt = os.environ["GEMV_SWEEP_ATTEMPT"]
shape = {"m": 1, "n": 2048 if job == "fast" else 4096, "k": 2048,
         "experts": 0, "active": 1, "format":"int4", "route":"dense"}
sid = "shape-fast" if job == "fast" else "shape-slow"
cfg = {"format":"int4", "layout":"native", "tile_size_k":64, "step_k":16,
       "threads":64, "route":"dense", "cta_m":1, "cta_n":8, "chunk":1}
cid = "int4-native-tk64-s16-t64-dense-m1-n8-c1"
base = {"schema":"gemv-sweep-raw-v1", "run_id":run, "shape_id":sid,
        "shape":shape, "format":"int4", "config_id":cid, "config":cfg,
        "attempt_id":attempt, "pass":0}
records = [
  {"rec":"run", "schema":"gemv-sweep-raw-v1", "run_id":run,
   "build":"fake-build", "space_id":"fake-space", "partial_space":False},
  dict(base, rec="attempt", expected_samples=3),
]
for i, us in enumerate((6.144, 8.192, 10.240)):
    bits = struct.unpack("<I", struct.pack("<f", us / 1000.0))[0]
    records.append(dict(base, rec="sample", launch_index=i,
                        event_ms_bits=bits, event_us=f"{us:.3f}"))
with open(out, "a") as h:
    for record in records:
        h.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
'''


def run(command, *, env=None, want=0):
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.returncode != want:
        raise AssertionError(
            f"rc={result.returncode}, want={want}\ncmd={command}\nstdout={result.stdout}\nstderr={result.stderr}")
    return result


def config():
    return {"format": "int4", "layout": "native", "tile_size_k": 64,
            "step_k": 16, "threads": 64, "route": "dense",
            "cta_m": 1, "cta_n": 8, "chunk": 1}


def main() -> int:
    run([sys.executable, str(SWEEP), "--self-test"])
    with tempfile.TemporaryDirectory(prefix="gemv-sweep-driver-") as td:
        root = pathlib.Path(td)
        fake = root / "fake.py"
        fake.write_text(FAKE)
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        marker = root / "marker"
        raw, progress = root / "raw.jsonl", root / "progress.jsonl"
        jobs = []
        for name, n in (("fast", 2048), ("slow", 4096)):
            jobs.append({
                "job_id": f"suite/{name}",
                "shape_id": f"shape-{name}",
                "shape": {"m": 1, "n": n, "k": 2048, "experts": 0, "active": 1,
                          "format": "int4", "route": "dense"},
                "formats": ["int4"],
                "argv": [name],
                "env": {"FAKE_MARKER": str(marker)},
                "expected": [{"format": "int4",
                              "config_id": "int4-native-tk64-s16-t64-dense-m1-n8-c1",
                              "config": config()}],
            })
        manifest = {
            "schema": "gemv-sweep-manifest-v1",
            "space_id": "fake-space",
            "partial_space": False,
            "counts": {"total": 2, "legal": 2, "pruned": 0, "prune_reasons": {}},
            "jobs": jobs,
        }
        plan = root / "plan.json"
        plan.write_text(json.dumps(manifest))
        dry = root / "dry.jsonl"
        command = [sys.executable, str(SWEEP), "run", str(plan), "--bin", str(fake),
                   "--raw", str(raw), "--progress", str(progress),
                   "--build-id", "fake-build"]
        run(command + ["--dry-run", "--dry-run-manifest", str(dry)])
        lines = [json.loads(x) for x in dry.read_text().splitlines()]
        if [x["job_id"] for x in lines] != ["suite/fast", "suite/slow"] or marker.exists():
            raise AssertionError("dry-run did not emit exactly the pending manifest without launching")
        if any(x["env"].get("GEMV_SWEEP_SAMPLES") != "20" or
               x["env"].get("GEMV_SWEEP_BUILD") != "fake-build" for x in lines):
            raise AssertionError("dry-run omitted the fixed sample/build protocol identity")

        # Fast completes; slow times out before writing.  rc=3 is bounded
        # incompleteness, not a malformed invocation.
        run(command + ["--shape-timeout", "0.15", "--deadline-seconds", "5"], want=3)
        first = [json.loads(x) for x in progress.read_text().splitlines()]
        if [x["status"] for x in first] != ["complete", "timeout"]:
            raise AssertionError(f"timeout progress mismatch: {first}")

        # Simulate death after raw append+fsync but before the complete progress
        # record.  Durable raw, not the advisory progress file, must own fast.
        progress.write_text(json.dumps(first[1], sort_keys=True) + "\n")

        # Resume must not launch fast again.  An external condition makes the
        # exact same slow job complete without changing the manifest hash.
        env = dict(os.environ, ALLOW_SLOW="1")
        run(command + ["--shape-timeout", "1", "--deadline-seconds", "5", "--resume"], env=env)
        launches = marker.read_text().splitlines()
        if launches != ["fast", "slow", "slow"]:
            raise AssertionError(f"resume relaunched a completed job: {launches}")
        states = [json.loads(x) for x in progress.read_text().splitlines()]
        if [(x["job_id"], x["attempt"], x["status"]) for x in states] != [
                ("suite/slow", 0, "timeout"), ("suite/slow", 1, "complete")]:
            raise AssertionError(f"attempt/progress sequence mismatch: {states}")

        wrong_build = command[:-2] + ["--build-id", "different-build", "--resume"]
        mismatch = run(wrong_build, env=env, want=2)
        if "build/space/protocol identity differs" not in mismatch.stderr:
            raise AssertionError("resume accepted durable raw from a different binary/build identity")

        result = root / "result.json"
        run([sys.executable, str(SWEEP), "analyse", str(raw), "--manifest", str(plan),
             "--output", str(result)])
        analysed = json.loads(result.read_text())
        if not analysed["complete"] or analysed["manifest_coverage"]["missing_outcomes"] != 0:
            raise AssertionError(f"completed raw failed exact manifest coverage: {analysed}")

    print("[gemv-sweep-driver] PASS: analyser plants + dry-run + raw-owned crash resume + "
          "build identity rejection + exact coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

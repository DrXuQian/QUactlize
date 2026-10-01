"""Physical-device PDL chain correctness, not performance or simulation.

Uses the library/application admitted by check.py. All cases compare ordinary
and PDL execution, direct stream submission and graph replay, against a CPU
oracle and against each other's raw output. Failed cases remain in the report.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from check_device import sha
from run_benchmark import snapshot, require_idle


def cases(suite):
    if suite not in ("smoke", "coverage"):
        raise ValueError("unknown PDL suite")
    rows = []
    formats = (12, 14) if suite == "smoke" else (8, 10, 11, 12, 13, 14)
    configs = (0, 2, 5) if suite == "smoke" else range(6)

    def add(q, compute, config, route, grouped, m=1, n=256, k=512, storage=None):
        rows.append(dict(q=q, compute=compute, config=config, route=route, grouped=grouped,
                         m=m, n=n, k=k, input=compute if storage is None else storage,
                         output=2 if storage is None else storage))

    for q in formats:
        for compute in (0, 1):
            for config in configs:
                for route in (0, 1):
                    for grouped in (False, True):
                        add(q, compute, config, route, grouped)
    for compute in (0, 1):
        for config in ((7,) if suite == "smoke" else range(6, 11)):
            for route in (0, 1):
                add(12, compute, config, route, True, n=257)
    if suite == "coverage":
        for q in formats:
            for compute in (0, 1):
                for m in (2, 3, 4, 5, 6, 7, 8, 65):
                    add(q, compute, 5, 0, False, m=m, n=256 if q == 8 else 48, k=1024)
                for storage in (0, 1, 2):
                    for config in (0, 2):
                        add(q, compute, config, 1, True, storage=storage)
    return rows


def arguments(application, point):
    result = [str(application)]
    for field, flag in (("q", "qtype"), ("compute", "compute"), ("input", "input"),
                        ("output", "output"), ("config", "config"), ("m", "m"), ("n", "n"), ("k", "k")):
        result += ["--" + flag, str(point[field])]
    if point["route"]:
        result.append("--sf")
    if point["grouped"]:
        # Include empty experts and both sides of the 64-row TC tile seam.
        result += ["--rows", "0,1,7,8,0" if point["config"] >= 6 else "0,1,8,65,0"]
    return result


def run_case(command, env, log, device, uuid, timeout):
    initial = snapshot(device)
    if initial["uuid"] != uuid or initial["processes"]:
        require_idle(initial, uuid)
    telemetry = [initial]
    error = None
    started = time.monotonic()
    with log.open("x") as output:
        output.write(json.dumps(command) + "\n")
        output.flush()
        child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                current = snapshot(device)
                telemetry.append(current)
                if current["uuid"] != uuid:
                    raise ValueError("GPU UUID changed")
                if any(p["pid"] != child.pid for p in current["processes"]):
                    raise ValueError("another GPU process appeared; only this test is stopped")
                if time.monotonic() - started > timeout:
                    raise ValueError("chain timeout")
                time.sleep(.25)
        except Exception as failure:
            error = str(failure)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait()
    return dict(returncode=child.returncode, error=error, telemetry=telemetry,
                wall_seconds=time.monotonic() - started, log=str(log), log_sha256=sha(log))


def validate_result(text, point, replays=3):
    lines = [line for line in text.splitlines() if line.startswith("PPU17_PDL_CHAIN PASS ")]
    if len(lines) != 1:
        raise ValueError("missing/duplicate chain result")
    fields = dict(token.split("=", 1) for token in lines[0].split()[2:])
    expected = {**{key: str(point[key]) for key in ("q", "compute", "input", "output", "config", "route")},
                "grouped": str(int(point["grouped"])), "comparisons": str(8 * replays),
                "raw_equal": "1", "cpu_oracle": "PASS", "stream": "PASS", "graph": "PASS", "timing": "NOT_MEASURED"}
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError("chain result differs: " + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True, help="check.py output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--suite", choices=("smoke", "coverage"), default="smoke")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    bundle, validation = args.bundle.resolve(), args.validation.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    gate = json.loads((validation / "validation.json").read_text())
    app = validation / "pdl-chain"
    library = bundle / "libquactlize_ppu17.so"
    if not manifest.get("pdl_compiled") or not gate.get("pdl_compiled") or gate.get("status") != "PASS":
        raise ValueError("PDL local gate has not passed")
    if sha(library) != manifest["library_sha256"] or sha(library) != gate["library_sha256"]:
        raise ValueError("library differs from local gate")
    if sha(app) != gate["pdl_chain_sha256"]:
        raise ValueError("chain application differs from local gate")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial = snapshot(args.device)
    require_idle(initial, args.expected_uuid)
    points = cases(args.suite)
    receipt = dict(scope="PHYSICAL_HOPPER_PDL_CORRECTNESS_NOT_NATIVE_PPU17",
                   timing="NOT_MEASURED", status="INCOMPLETE", gpu=initial,
                   application_sha256=sha(app), library_sha256=sha(library),
                   build_key=manifest["build_key"], suite=args.suite, cases=points, results=[])
    if manifest["target"] == "native":
        receipt["scope"] = "PHYSICAL_PPU17_PDL_CORRECTNESS"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.device,
           "LD_LIBRARY_PATH": str(bundle) + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}

    def save():
        (out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    started = time.monotonic()
    interrupted = False
    for index, point in enumerate(points):
        log = out / f"{index:04d}.log"
        try:
            result = run_case(arguments(app, point), env, log, args.device, args.expected_uuid, args.timeout)
        except (ValueError, subprocess.SubprocessError) as failure:
            receipt["interruption"] = str(failure)
            interrupted = True
            break
        try:
            if result["error"] or result["returncode"]:
                raise ValueError(result["error"] or "process failed")
            validate_result(log.read_text(), point)
            status = "PASS"
        except ValueError as error:
            status = "FAIL"
            result["error"] = str(error)
        receipt["results"].append(dict(index=index, point=point, status=status, **result))
        save()
        print(f"PPU17_PDL_PROGRESS completed={index+1}/{len(points)} status={status} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        if result["telemetry"][-1]["uuid"] != args.expected_uuid or "another GPU" in (result["error"] or ""):
            interrupted = True
            break
    negative = {"status": "NOT_RUN"}
    if not interrupted:
        try:
            negative = run_case([str(app), "--negative-missing-wait"], env, out / "negative.log",
                                args.device, args.expected_uuid, args.timeout)
            negative_text = (out / "negative.log").read_text()
            negative["status"] = ("EXPECTED_RED" if not negative["error"] and negative["returncode"] == 0 and
                                  "missing_wait=EXPECTED_RED" in negative_text and "waited_control=PASS producer=PASS" in negative_text
                                  else "UNRESOLVED_OR_FAILED")
        except (ValueError, subprocess.SubprocessError) as failure:
            negative = {"status": "NOT_RUN", "error": str(failure)}
    receipt["negative"] = negative
    receipt["correctness"] = ("PASS" if len(receipt["results"]) == len(points) and
                               all(row["status"] == "PASS" for row in receipt["results"]) else "FAIL_OR_INCOMPLETE")
    receipt["status"] = "PASS" if receipt["correctness"] == "PASS" and negative["status"] == "EXPECTED_RED" else "INCOMPLETE_OR_FAILED"
    receipt["elapsed_seconds"] = time.monotonic() - started
    save()
    print(f"PPU17_PDL_DONE status={receipt['status']} correctness={receipt['correctness']} negative={negative['status']} results={out}", flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

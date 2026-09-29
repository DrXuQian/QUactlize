"""Serial, fresh-process numerical checks on an explicitly supplied GPU.

Every application process submits one target kernel and uses a CPU oracle.
No timings here are admitted as kernel performance measurements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_identity(device):
    fields = "index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu"
    text = subprocess.check_output(["nvidia-smi", f"--id={device}", f"--query-gpu={fields}",
                                    "--format=csv,noheader,nounits"], text=True)
    return dict(zip(fields.split(","), [v.strip() for v in next(csv.reader(text.splitlines()))]))


def gpu_processes(device):
    text = subprocess.check_output(["nvidia-smi", f"--id={device}",
        "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    return [line.strip() for line in text.splitlines() if line.strip()]


def cases(suite):
    result = []

    def add(q, compute, config, route=0, rows=None, m=9, n=256, k=512, input_type=None, output=2, phase="gemm"):
        result.append(dict(q=q, compute=compute, config=config, route=route, rows=rows,
            m=m, n=n, k=k, input=compute if input_type is None else input_type, output=output, phase=phase))

    if suite == "cached-decode":
        for compute in (0, 1):
            for config in (6, 7, 8, 9, 10):
                for route in (0, 1):
                    for m in range(1, 9):
                        add(12, compute, config, route, m=m, n=257)
                    for storage in (0, 1, 2):
                        add(12, compute, config, route, rows=[0,1,2,3,4,5,6,7,8,0],
                            n=257, k=1024, input_type=storage, output=storage)
        return result

    for q in (8, 10, 11, 12, 13, 14):
        for compute in (0, 1):
            for config in range(6):
                for route in (0, 1):
                    for grouped in (False, True):
                        add(q, compute, config, route, [0, 1, 9, 65] if grouped else None)
            add(q, compute, 2, rows=[0, 1, 9, 65], phase="scales")
    if suite == "extended":
        for q in (8, 10, 11, 12, 13, 14):
            for compute in (0, 1):
                n = 256 if q == 8 else 48
                for config in (0, 1, 2, 3, 4, 5):
                    # K1024 wraps the two-stage TMA epoch repeatedly.
                    for route in (0, 1):
                        add(q, compute, config, route, rows=[0, 1, 8, 9, 63, 64, 65, 0], n=n, k=1024)
                for m in (1, 2, 3, 4, 5, 6, 7, 8, 63, 64, 65, 129):
                    for config in (0, 2, 5):
                        add(q, compute, config, m=m, n=n)
                for config in (0, 2):
                    for route in (0, 1):
                        for storage in (0, 1, 2):
                            add(q, compute, config, route, rows=[0, 2, 0, 8], n=n,
                                input_type=storage, output=storage)
    return result


def arguments(application, point):
    argv = [str(application), "--qtype", str(point["q"]), "--compute", str(point["compute"]),
            "--input", str(point["input"]), "--output", str(point["output"]),
            "--config", str(point["config"]), "--m", str(point["m"]),
            "--n", str(point["n"]), "--k", str(point["k"])]
    if point["rows"] is not None: argv += ["--rows", ",".join(map(str, point["rows"]))]
    if point["route"]: argv.append("--sf")
    if point["phase"] == "scales": argv.append("--expand")
    return argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--application", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-uuid", required=True)
    p.add_argument("--suite", choices=("coverage", "extended", "cached-decode"), default="coverage")
    p.add_argument("--timeout", type=int, default=45)
    p.add_argument("--max-failures", type=int, default=8)
    p.add_argument("--resume-from", type=Path, help="reuse checked PASS rows from an interrupted summary")
    a = p.parse_args()
    if a.timeout <= 0 or a.max_failures <= 0: p.error("timeout/max-failures must be positive")
    bundle, application, out = a.bundle.resolve(), a.application.resolve(), a.output.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    library = bundle / "libquactlize_ppu17.so"
    if manifest["library_sha256"] != sha(library): raise ValueError("library identity differs")
    identity = gpu_identity(a.device)
    if identity["uuid"] != a.expected_uuid: raise ValueError("selected GPU UUID differs")
    if gpu_processes(a.device): raise ValueError("GPU has existing compute jobs; do not start")
    points = cases(a.suite)
    if out.exists() and any(out.iterdir()): raise ValueError("result directory must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    receipt = {"gpu": identity, "application_sha256": sha(application), "library_sha256": sha(library),
        "suite": a.suite, "cases": points, "scope": "PHYSICAL_HOPPER_NUMERICS_NOT_PPU17", "timing_admission": "NOT_MEASURED"}
    reused = {}
    if a.resume_from:
        previous = json.loads(a.resume_from.read_text())
        for key in ("application_sha256", "library_sha256", "suite", "cases", "scope"):
            if previous[key] != receipt[key]: raise ValueError("resume identity differs: " + key)
        if previous["gpu"]["uuid"] != identity["uuid"]: raise ValueError("resume GPU differs")
        for row in previous["results"]:
            if row["status"] != "PASS": continue
            index, log = row["index"], Path(row["log"])
            if index in reused or points[index] != row["point"]: raise ValueError("resume point differs")
            if row["returncode"] or sha(log) != row["log_sha256"] or "PPU17_CASE PASS target_launches=1" not in log.read_text():
                raise ValueError("resume log differs")
            reused[index] = {**row, "reused_from": str(a.resume_from.resolve())}
        receipt["resume_summary_sha256"] = sha(a.resume_from)
        receipt["reused"] = len(reused)
    (out / "inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.device,
           "LD_LIBRARY_PATH": str(bundle) + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
    results = []
    started = time.monotonic()
    failures = 0
    for index, point in enumerate(points):
        if index in reused:
            results.append(reused[index])
            continue
        # Numerical cases are serialized; don't overlap another user's job.
        external = gpu_processes(a.device)
        if external:
            receipt["interruption"] = {"reason": "GPU_BECAME_BUSY", "processes": external}
            break
        argv = arguments(application, point)
        key = hashlib.sha256(json.dumps(point, sort_keys=True).encode()).hexdigest()[:12]
        log = logs / f"{index:04d}-{key}.log"
        with log.open("w") as f:
            f.write(json.dumps(argv) + "\n"); f.flush()
            try:
                rc = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        text = log.read_text()
        status = "PASS" if rc == 0 and "PPU17_CASE PASS target_launches=1" in text else "FAIL"
        failures += status == "FAIL"
        results.append({"index": index, "point": point, "status": status, "returncode": rc, "log": str(log), "log_sha256": sha(log)})
        if status == "FAIL": print("PPU17_DEVICE_FAIL " + json.dumps(results[-1]), flush=True)
        if index % 20 == 0 or status == "FAIL" or index + 1 == len(points):
            print(f"PPU17_DEVICE_PROGRESS completed={len(results)}/{len(points)} failures={failures} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        if failures >= a.max_failures:
            receipt["interruption"] = {"reason": "FAILURE_BUDGET_REACHED"}
            break
    receipt.update(status="PASS" if len(results) == len(points) and failures == 0 else "INCOMPLETE_OR_FAILED",
                   completed=len(results), failed=failures, results=results, elapsed_seconds=time.monotonic()-started,
                   gpu_after=gpu_identity(a.device), processes_after=gpu_processes(a.device))
    (out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PPU17_DEVICE_DONE status={receipt['status']} completed={len(results)}/{len(points)} failures={failures} results={out}", flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

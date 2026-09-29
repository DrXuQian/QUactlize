"""Bounded physical-Hopper baseline; never a simulator or production tuner.

Refuse a busy GPU, verify our live GPU PID, and monitor all reported GPU
processes. Any observed foreign job invalidates the run. This is sampled
process isolation, not a driver-enforced reservation; do not schedule other
GPU work on this device while the runner is active.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

from check_device import sha


def snapshot(device):
    xml = subprocess.check_output(["nvidia-smi", "--id=" + device, "-q", "-x"],
                                  text=True, timeout=10)
    root = ET.fromstring(xml)
    nodes = root.findall("gpu")
    if len(nodes) != 1: raise ValueError("exactly one physical GPU is required")
    gpu = nodes[0]
    processes = gpu.find("processes")
    if processes is None or (processes.text or "").strip() not in ("", "None"):
        raise ValueError("GPU process inventory is unavailable")
    return {
        "utc": datetime.now(timezone.utc).isoformat(), "uuid": gpu.findtext("uuid"),
        "name": gpu.findtext("product_name"), "driver": root.findtext("driver_version"),
        "processes": [{"pid": int(p.findtext("pid")), "type": p.findtext("type"),
                       "name": p.findtext("process_name")} for p in processes.findall("process_info")],
        "gpu_util": int(gpu.findtext("utilization/gpu_util").split()[0]),
        "clocks": {x.tag: x.text for x in gpu.findall("clocks/*")},
        "temperature": gpu.findtext("temperature/gpu_temp"),
        "power": gpu.findtext("gpu_power_readings/instant_power_draw"),
        "pstate": gpu.findtext("performance_state"),
    }


def require_idle(sample, uuid):
    if sample["uuid"] != uuid: raise ValueError("GPU UUID changed")
    if sample["processes"] or sample["gpu_util"]:
        raise ValueError("GPU_NOT_IDLE: " + json.dumps(sample))


def validate_result(row):
    if row.get("correctness") != "CPU_DOT_AND_GUARDS_PASS": raise ValueError("correctness did not pass")
    if row.get("scope") != "PHYSICAL_HOPPER_GRAPH_AVERAGE_NOT_PPU17": raise ValueError("wrong timing scope")
    samples = row.get("samples_us", [])
    if len(samples) != 11 or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("invalid timing samples")
    if abs(sorted(samples)[5] - row["median_us"]) > 1.e-6: raise ValueError("median differs from samples")
    if row["launches_per_graph"] % row["weight_copies"]: raise ValueError("incomplete rotating cycle")
    if row["cache"] == "rotating" and row["weight_bytes"] * row["weight_copies"] <= 2 * row["l2_bytes"]:
        raise ValueError("rotating working set does not exceed two L2 capacities")


def benchmark_points(suite, configs=None, reverse=False):
    selected = list(range(6)) if configs is None else list(configs)
    if not selected or len(set(selected)) != len(selected) or any(c not in range(6) for c in selected):
        raise ValueError("configs must be a nonempty unique subset of 0..5")
    shapes = [(1, False), (128, False), (65, True)] if suite == "baseline" else [(9, True)]
    points = [(m, grouped, route, config) for m, grouped in shapes
              for route in (0, 1) for config in selected]
    return list(reversed(points)) if reverse else points


def run_point(command, env, log, device, uuid, timeout):
    telemetry = []
    quiet = 0
    deadline = time.monotonic() + 15
    while quiet < 3:
        current = snapshot(device); telemetry.append(current)
        if current["uuid"] != uuid or current["processes"]: require_idle(current, uuid)
        quiet = quiet + 1 if current["gpu_util"] == 0 else 0
        if time.monotonic() >= deadline: raise ValueError("GPU_DID_NOT_BECOME_IDLE")
        time.sleep(.25)
    ready = threading.Event()
    received = []
    with log.open("w") as output:
        output.write(json.dumps(command) + "\n"); output.flush()
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
        def reader():
            for line in child.stdout:
                output.write(line); output.flush()
                if line.strip() == "PPU17_BENCH_READY": ready.set()
                if line.startswith("PPU17_BENCH "): received.append(json.loads(line.split(" ", 1)[1]))
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        began = time.monotonic()
        admitted = False
        failure = None
        try:
            while child.poll() is None:
                current = snapshot(device); telemetry.append(current)
                if current["uuid"] != uuid: raise ValueError("GPU UUID changed")
                foreign = [p for p in current["processes"] if p["pid"] != child.pid]
                if foreign: raise ValueError("CONTAMINATED_BY_OTHER_GPU_JOB: " + json.dumps(foreign))
                if ready.is_set() and not admitted:
                    if not any(p["pid"] == child.pid for p in current["processes"]):
                        raise ValueError("cannot confirm benchmark GPU PID; possible PID namespace mismatch")
                    child.stdin.write("R\n"); child.stdin.flush()
                    admitted = True
                if time.monotonic() - began > timeout: raise ValueError("benchmark timeout")
                time.sleep(.1)
        except Exception as error:
            failure = str(error)
            # Only terminate our direct child; never interfere with the foreign job.
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
        finally:
            child.wait(); thread.join()
            child.stdin.close(); child.stdout.close()
        current = snapshot(device); telemetry.append(current)
        if current["uuid"] != uuid: failure = "GPU_UUID_CHANGED_AFTER_RUN"
        foreign = [p for p in current["processes"] if p["pid"] != child.pid]
        if foreign: failure = "CONTAMINATED_BY_OTHER_GPU_JOB_AFTER_RUN"
        if not failure and (not admitted or child.returncode or len(received) != 1):
            failure = "benchmark incomplete: rc=" + str(child.returncode)
        if not failure:
            try: validate_result(received[0])
            except ValueError as error: failure = str(error)
        return {"status": "INVALID" if failure else "PASS", "error": failure,
                "returncode": child.returncode, "pid": child.pid, "result": received,
                "telemetry": telemetry, "wall_seconds": time.monotonic()-began}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--qtype", type=int, choices=(8,10,11,12,13,14), default=12)
    parser.add_argument("--compute", type=int, choices=(0,1), default=1)
    parser.add_argument("--cache", choices=("warm","rotating"), default="rotating")
    parser.add_argument("--suite", choices=("smoke","baseline"), default="smoke")
    parser.add_argument("--configs", type=int, nargs="+", choices=range(6),
                        help="bounded reconfirmation subset; default: all six configs")
    parser.add_argument("--reverse", action="store_true", help="reverse point order for reconfirmation")
    args = parser.parse_args()
    app, bundle, out = args.application.resolve(), args.bundle.resolve(), args.output.resolve()
    manifest = json.loads((bundle/"manifest.json").read_text())
    lib = bundle/"libquactlize_ppu17.so"
    if sha(lib) != manifest["library_sha256"]: raise ValueError("library hash differs")
    if out.exists() and any(out.iterdir()): raise ValueError("result directory must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.device, "Q17_BENCH_GUARDED": "1",
           "LD_LIBRARY_PATH": str(bundle) + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
    points = benchmark_points(args.suite, args.configs, args.reverse)
    receipt = {"application_sha256":sha(app), "library_sha256":sha(lib), "runner_sha256":sha(__file__),
               "gpu_uuid":args.expected_uuid, "points":points, "results":[], "scope":"HOPPER_BASELINE_NOT_PRODUCTION_SELECTION",
               "isolation":"PRECHECK_AND_100MS_SAMPLED_PROCESS_MONITOR_NOT_DRIVER_RESERVATION"}
    for i,(m,grouped,route,config) in enumerate(points):
        command=[str(app),"--qtype",str(args.qtype),"--compute",str(args.compute),"--config",str(config),"--m",str(m)]
        if grouped: command.append("--grouped")
        if route: command.append("--sf")
        if args.cache == "rotating": command.append("--rotating")
        try: row=run_point(command,env,out/f"{i:03d}.log",args.device,args.expected_uuid,args.timeout)
        except Exception as error: row={"status":"INVALID","error":str(error)}
        if row["status"] == "PASS":
            expected = {"q":args.qtype, "compute":args.compute, "config":config, "route":route,
                        "m":m+9 if grouped else m, "n":512, "k":2048,
                        "experts":4 if grouped else 1, "cache":args.cache}
            if any(row["result"][0].get(key) != value for key,value in expected.items()):
                row.update(status="INVALID",error="measured point differs from requested point")
        row.update(command=command,point=points[i]); receipt["results"].append(row)
        print(f"PPU17_TIMING_PROGRESS completed={i+1}/{len(points)} status={row['status']}",flush=True)
        receipt["status"]="PASS" if i+1==len(points) and row["status"]=="PASS" else "INCOMPLETE_OR_INVALID"
        (out/"summary.json").write_text(json.dumps(receipt,indent=2)+"\n")
        if row["status"] != "PASS": print("PPU17_TIMING_STOP "+str(row.get("error")),flush=True); return 1
    print("PPU17_TIMING_DONE status=PASS results="+str(out),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

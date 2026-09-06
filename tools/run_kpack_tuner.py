#!/usr/bin/env python3
"""Run budgeted K-pack profiling, one active process per device.

Requests of the same qtype/route/N/K/expert geometry share a process and its
host weight cache. Every request shares its fixture/H2D/workspace across all
selected modules. Raw correctness is mandatory; a failing candidate is logged,
the poisoned process is discarded, and only unfinished candidates are retried.
Results are exact-shape measurements, not a global-optimality certificate.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import signal
import statistics
import subprocess
import threading
import time
from typing import Any

from kpack_tuning_plan import SCHEMA, digest
from build_kpack_tuner import sha, sdk_identity

STRUCTURAL = {
    "SHIPPING_SHARED_STORAGE",
    "SPLIT_SHARED_STORAGE",
    "SPLIT_PARTITION",
    "INADMISSIBLE_PIPELINE_DEPTH",
    "M8_DECODE_ONLY_M_GE_8",
    "PACKED_A_DECODE_ONLY_M_NOT_1",
    "REAL_CAN_IMPLEMENT",
    "INADMISSIBLE_SHARED_STORAGE",
    "INADMISSIBLE_OCCUPANCY",
}
COMPLETE = (
    "FQ_SHAPE_DONE ",
    "SF_COMPLETE ",
    "FQ_GROUPED_KPACK_COMPLETE ",
    "SF_GROUPED_COMPLETE ",
)
STOP = threading.Event()
DRIVERS_LOCK = threading.RLock()
DRIVERS = set()


def kv(line: str) -> dict[str, str]:
    result = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            if key in result:
                raise ValueError(f"duplicate field {key}")
            result[key] = value
    return result


def parse_cells(
    text: str,
    request: dict,
    expected_symbols: set[str],
    iterations: int,
    complete: bool = True,
) -> list[dict]:
    """Normalize only correct full-output cells with exact finite samples."""
    cells = {}
    seen_symbols = set()
    route = request["route"]
    for line in text.splitlines():
        if line.startswith("SF_CELL "):
            row = json.loads(line[len("SF_CELL ") :])
            q = int(row["qtype"])
            symbol, state = row["symbol"], row["status"]
            algorithm, split, grid = (
                row["algorithm"],
                int(row["split"]),
                int(row["grid"]),
            )
            samples = [float(row["sample_us"])] if state == "MEASURED" else []
            scope = row["metric_scope"]
            index = int(row["sample"])
            structural = state == "INADMISSIBLE" and str(row["reason"]).startswith(
                "INADMISSIBLE_"
            )
        elif line.startswith(
            ("FQ_TC_CELL ", "FQ_GROUPED_KPACK_CELL ", "SF_GROUPED_CELL ")
        ):
            row = kv(line)
            q = int(row["q"])
            symbol, state = row["symbol"], row["state"]
            split, grid = int(row.get("S", 1)), int(row.get("grid", 0))
            algorithm = row.get("algorithm", f"TC_S{split}")
            samples = json.loads(row["samples"])
            scope = row.get("scope", "FULL_OUTPUT")
            structural = state in STRUCTURAL
            index = 0
        else:
            continue
        if symbol not in expected_symbols or q != request["qtype"]:
            raise ValueError("foreign candidate/qtype in measurement")
        if route.endswith("dense"):
            p = request["problem"]
            if row.get("shape") != f"{p['m']}x{p['n']}x{p['k']}":
                raise ValueError("measurement shape differs")
        if int(row.get("raw_bad", -1)) != 0:
            if complete:
                raise ValueError("raw-bit mismatch cannot be timed")
            continue
        if state != "MEASURED" and not structural:
            if complete:
                raise ValueError(f"nonterminal failure {state}")
            continue
        if state == "MEASURED" and scope != "FULL_OUTPUT":
            raise ValueError("producer-only measurement cannot rank as full output")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in samples
        ):
            raise ValueError("invalid timing sample")
        key = symbol, algorithm, split, grid
        if line.startswith("SF_CELL ") and key in cells:
            if index != len(cells[key]["samples_us"]):
                raise ValueError("duplicate/missing ScaleFirst sample")
            cells[key]["samples_us"].extend(samples)
        else:
            if key in cells or index != 0:
                raise ValueError("duplicate runtime cell")
            cells[key] = {
                "symbol": symbol,
                "algorithm": algorithm,
                "split": split,
                "grid": grid,
                "status": "MEASURED" if state == "MEASURED" else "STRUCTURAL",
                "reason": row.get("reason", state),
                "samples_us": samples,
            }
        seen_symbols.add(symbol)
    good = []
    for cell in cells.values():
        if cell["status"] == "MEASURED":
            if len(cell["samples_us"]) != iterations:
                if complete:
                    raise ValueError("timing sample denominator differs")
                continue
            cell["median_us"] = statistics.median(cell["samples_us"])
        good.append(cell)
    if complete:
        if seen_symbols != expected_symbols:
            raise ValueError("missing candidate cells")
        if not any(
            line.startswith(COMPLETE)
            and ("status=PASS" in line or "status=COMPLETE" in line)
            for line in text.splitlines()
        ):
            raise ValueError("missing successful request completion")
        if route == "fq-dense":
            splits = {1} if request["problem"]["m"] >= 64 else {1, 2, 4, 8}
            for symbol in expected_symbols:
                if {r["split"] for r in good if r["symbol"] == symbol} != splits:
                    raise ValueError("FQ Split-K denominator differs")
        if route == "sf-dense":
            for symbol in expected_symbols:
                if {r["algorithm"] for r in good if r["symbol"] == symbol} != {
                    "NONPERSISTENT",
                    "PERSISTENT",
                }:
                    raise ValueError(
                        "ScaleFirst full-output algorithm denominator differs"
                    )
        if route.endswith("grouped"):
            prefix = (
                "FQ_GROUPED_KPACK_SHARD "
                if route == "fq-grouped"
                else "SF_GROUPED_SHARD "
            )
            headers = [
                kv(line) for line in text.splitlines() if line.startswith(prefix)
            ]
            p = request["problem"]
            if (
                len(headers) != 1
                or any(
                    int(headers[0].get(k, -1)) != p[k]
                    for k in ("total_rows", "max_rows", "experts")
                )
                or headers[0].get("workload") != request["workload_key"]
                or headers[0].get("roundtrip") != "PASS"
            ):
                raise ValueError("grouped shape/router/roundtrip header differs")
    return good


def verify_result(record: dict, request: dict, iterations: int) -> None:
    observed = []
    for log in record["logs"]:
        path = Path(log["path"])
        if sha(path) != log["sha256"]:
            raise ValueError("result log changed")
        text = path.read_text()
        wanted = set(log["symbols"])
        if not wanted <= set(request["symbols"]):
            raise ValueError("foreign replay symbols")
        if log["rc"] == 0:
            observed.extend(parse_cells(text, request, wanted, iterations))
        else:
            attempts = [
                kv(line)["symbol"]
                for line in text.splitlines()
                if line.startswith("KPACK_TUNER_ROW_BEGIN ")
            ]
            if attempts:
                observed.extend(
                    c
                    for c in parse_cells(text, request, wanted, iterations, False)
                    if c["symbol"] != attempts[-1]
                )
    if observed != record["cells"]:
        raise ValueError("cached cells differ from immutable raw logs")


def request_argv(
    request: dict, symbols: Path, rows_root: Path, iterations: int, seed: int
) -> list[str]:
    p, route = request["problem"], request["route"]
    args = [
        f"--iterations={iterations}",
        "--correctness-repeats=1",
        f"--schedule-seed={seed}",
    ]
    if route.endswith("dense"):
        args.append(f"--shape={p['m']}x{p['n']}x{p['k']}")
        if route == "fq-dense":
            args += ["--bc-mode=skip", "--tm8-max-m=64", f"--symbols-file={symbols}"]
            if p["m"] >= 64:
                args += ["--only-split=1"]
        else:
            args += ["--algorithm=full-output", f"--symbol-file={symbols}"]
    else:
        row = request["grouped"]
        args += [
            f"--n={p['n']}",
            f"--k={p['k']}",
            f"--experts={p['experts']}",
            f"--workload-key={request['workload_key']}",
            f"--router-profile={row['profile']}",
            "--warmups=1",
            f"--symbol-file={symbols}",
        ]
        if row["rows_file"] == "-":
            args += [f"--tokens={row['tokens']}", f"--topk={row['topk']}"]
        else:
            args += [f"--rows-file={rows_root/row['rows_file']}"]
    return args


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


class Driver:
    def __init__(self, binary: str, device: int, sdk: Path):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        env["LD_LIBRARY_PATH"] = (
            ":".join(str(sdk / p) for p in ("lib", "lib64", "targets/x86_64-linux/lib"))
            + ":"
            + env.get("LD_LIBRARY_PATH", "")
        )
        self.process = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        with DRIVERS_LOCK:
            DRIVERS.add(self)
        self.lines = queue.Queue()

        def read():
            try:
                for line in self.process.stdout:
                    self.lines.put(line)
            finally:
                self.lines.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def run(
        self, request_id: str, modules: Path, argv: list[str], log: Path, timeout: float
    ) -> tuple[int, str]:
        fields = [request_id, str(modules), *argv]
        if any(not x or any(c in x for c in "\r\n\t") for x in fields):
            raise ValueError("invalid request field")
        collected = []
        deadline = time.monotonic() + timeout
        try:
            self.process.stdin.write("\t".join(fields) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            pass  # Drain startup diagnostics below instead of hiding them.
        with log.open("w") as stream:
            while True:
                try:
                    line = self.lines.get(
                        timeout=max(0.01, deadline - time.monotonic())
                    )
                except queue.Empty:
                    stream.write("KPACK_TUNER_TIMEOUT\n")
                    return 124, "".join(collected)
                if line is None:
                    return self.process.wait() or 2, "".join(collected)
                stream.write(line)
                stream.flush()
                collected.append(line)
                if line.startswith("KPACK_TUNER_END "):
                    row = kv(line)
                    if row.get("id") != request_id:
                        raise ValueError("driver response ID differs")
                    return int(row["rc"]), "".join(collected)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.reader.join(timeout=1)
        with DRIVERS_LOCK:
            DRIVERS.discard(self)


def probe_devices(bundle: dict, devices: list[int], sdk: Path) -> list[dict]:
    binary = next(iter(bundle["pairs"].values()))["driver"]

    def probe(device):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        env["LD_LIBRARY_PATH"] = (
            ":".join(str(sdk / p) for p in ("lib", "lib64", "targets/x86_64-linux/lib"))
            + ":"
            + env.get("LD_LIBRARY_PATH", "")
        )
        result = subprocess.run(
            [binary],
            input="",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=60,
        )
        rows = [
            kv(line)
            for line in result.stdout.splitlines()
            if line.startswith("KPACK_TUNER_DEVICE ")
        ]
        if result.returncode or len(rows) != 1 or rows[0].get("visible") != "1":
            raise ValueError(f"device {device} probe failed: {result.stdout[-3000:]}")
        return {"ordinal": device, **rows[0]}

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        rows = list(pool.map(probe, devices))
    if len({r["pci"] for r in rows}) != len(devices):
        raise ValueError("workers refer to the same physical device")
    if len({(r["name"], r["cu"]) for r in rows}) != 1:
        raise ValueError("heterogeneous device pool")
    return rows


def run_group(
    requests: list[dict],
    device: int,
    bundle: dict,
    output: Path,
    sdk: Path,
    iterations: int,
    timeout: float,
    retry: bool,
    progress,
) -> None:
    first = requests[0]
    pair = bundle["pairs"][f"q{first['qtype']}-{first['route']}"]
    driver = None
    try:
        for request in sorted(
            requests,
            key=lambda r: (
                r["problem"].get("m", r["problem"].get("total_rows")),
                r["id"],
            ),
        ):
            if STOP.is_set():
                return
            identity = digest(request)
            result_path = output / "results" / (request["id"] + ".json")
            result = {
                "request_sha256": identity,
                "cells": [],
                "rejected": {},
                "logs": [],
                "device": device,
            }
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["request_sha256"] != identity or result["device"] != device:
                    raise ValueError("existing result request/device differs")
                verify_result(result, request, iterations)
            done = {c["symbol"] for c in result["cells"]}
            if retry:
                result["rejected"] = {}
            pending = set(request["symbols"]) - done - result["rejected"].keys()
            attempt = len(result["logs"])
            while pending:
                if STOP.is_set():
                    return
                attempt += 1
                tag = f"{request['id']}.{attempt:03d}"
                # Keep interrupted, unsealed attempt logs; never overwrite
                # one merely because it was absent from the result receipt.
                while (output / "logs" / (tag + ".log")).exists():
                    attempt += 1
                    tag = f"{request['id']}.{attempt:03d}"
                symbol_file = output / "inputs" / (tag + ".symbols")
                symbol_file.write_text("\n".join(sorted(pending)) + "\n")
                selected_modules = [
                    m["path"]
                    for m in pair["modules"]
                    if pending.intersection(m["symbols"])
                ]
                if not selected_modules:
                    raise ValueError("missing selected modules")
                module_file = output / "inputs" / (tag + ".modules")
                module_file.write_text("\n".join(selected_modules) + "\n")
                log = output / "logs" / (tag + ".log")
                args = request_argv(
                    request,
                    symbol_file,
                    output / "inputs",
                    iterations,
                    int(request["id"][:16], 16)
                    ^ attempt
                    ^ int(request.get("schedule_salt", 0)),
                )
                if driver is None:
                    driver = Driver(pair["driver"], device, sdk)
                print(
                    f"KPACK_TUNER_REQUEST device={device} route={request['route']} workload={request['workload_key']} candidates={len(pending)}",
                    flush=True,
                )
                rc, text = driver.run(request["id"], module_file, args, log, timeout)
                if STOP.is_set():
                    return
                record = {
                    "path": str(log),
                    "sha256": sha(log),
                    "rc": rc,
                    "symbols": sorted(pending),
                }
                result["logs"].append(record)
                if rc == 0:
                    result["cells"].extend(
                        parse_cells(text, request, pending, iterations)
                    )
                    pending.clear()
                else:
                    driver.close()
                    driver = None
                    attempts = [
                        kv(line)["symbol"]
                        for line in text.splitlines()
                        if line.startswith("KPACK_TUNER_ROW_BEGIN ")
                    ]
                    if not attempts or attempts[-1] not in pending:
                        result["infrastructure_failure"] = {"rc": rc, "log": str(log)}
                        atomic_json(result_path, result)
                        raise ValueError(
                            f"fixture/module/device failure before candidate; see {log}"
                        )
                    bad = attempts[-1]
                    partial = parse_cells(
                        text, request, pending, iterations, complete=False
                    )
                    # The last attempted row is never accepted from a failed
                    # request, even if some of its runtime variants passed.
                    clean = {r["symbol"] for r in partial} - {bad}
                    result["cells"].extend(r for r in partial if r["symbol"] in clean)
                    result["rejected"][bad] = {"rc": rc, "log": str(log)}
                    pending -= clean | {bad}
                    print(
                        f"KPACK_TUNER_REJECT device={device} symbol={bad} remaining={len(pending)} log={log}",
                        flush=True,
                    )
                result.pop("infrastructure_failure", None)
                atomic_json(result_path, result)
            measured = [c for c in result["cells"] if c["status"] == "MEASURED"]
            result["status"] = "MEASURED" if measured else "NO_VALID_CANDIDATE"
            result["best"] = (
                min(measured, key=lambda c: c["median_us"]) if measured else None
            )
            result["scope"] = "BEST_MEASURED_SELECTED_SET_NOT_GLOBAL_OPTIMUM"
            atomic_json(result_path, result)
            progress(request)
    finally:
        if driver:
            driver.close()


def summarize(plan: dict, output: Path) -> dict:
    rows = []
    for request in plan["requests"]:
        path = output / "results" / (request["id"] + ".json")
        record = json.loads(path.read_text()) if path.exists() else {}
        best = record.get("best") or {}
        rows.append(
            {
                "cell_key": request["cell_key"],
                "route": request["route"],
                "status": record.get("status", "INCOMPLETE"),
                "symbol": best.get("symbol", ""),
                "algorithm": best.get("algorithm", ""),
                "split": best.get("split", ""),
                "grid": best.get("grid", ""),
                "median_us": best.get("median_us", ""),
                "measured_runtime_cells": sum(
                    c["status"] == "MEASURED" for c in record.get("cells", [])
                ),
                "rejected_parents": len(record.get("rejected", {})),
                "scope": "BEST_MEASURED_SELECTED_SET",
            }
        )
    with (output / "summary.tsv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    complete = all(r["status"] == "MEASURED" for r in rows)
    report = {
        "schema": "quactlize.kpack-tuner-result.v1",
        "plan_sha256": digest(plan),
        "status": "SCREEN_COMPLETE" if complete else "INCOMPLETE",
        "global_5pct_bound_proven": False,
        "production_policy_updated": False,
        "requests": rows,
        "rejected_parents": sum(r["rejected_parents"] for r in rows),
    }
    atomic_json(output / "summary.json", report)
    atomic_json(
        output / "heuristic-input.json",
        {
            "schema": "quactlize.kpack-measured-exact-candidates.v1",
            "scope": "PER_ROUTE_KERNEL_FULL_OUTPUT_NO_PREPASS_AMORTIZATION",
            "quality": "PROVISIONAL_SCREEN_NOT_GLOBAL_5PCT_PROOF",
            "rows": [
                {"request": r, "measurement": m} for r, m in zip(plan["requests"], rows)
            ],
        },
    )
    return report


def run(
    plan: dict,
    bundle: dict,
    output: Path,
    sdk: Path,
    devices: list[int],
    iterations: int = 2,
    timeout: float = 1800,
    retry: bool = False,
) -> dict:
    if (
        plan.get("schema") != SCHEMA
        or bundle.get("schema") != "quactlize.kpack-tuner-build.v1"
    ):
        raise ValueError("plan/bundle schema differs")
    if bundle["plan_sha256"] != digest(plan):
        raise ValueError("bundle plan differs")
    if (
        iterations < 2
        or not devices
        or len(set(devices)) != len(devices)
        or min(devices) < 0
    ):
        raise ValueError("invalid measurement/device controls")
    if bundle["identity"]["sdk"] != sdk_identity(sdk):
        raise ValueError("SDK differs from compiled modules")
    for path, want in bundle["payloads"].items():
        if sha(Path(path)) != want:
            raise ValueError(f"payload changed: {path}")
    groups = defaultdict(list)
    for request in plan["requests"]:
        p = request["problem"]
        groups[
            request["qtype"], request["route"], p["n"], p["k"], p.get("experts", 1)
        ].append(request)
    assigned = [[] for _ in devices]
    costs = [0.0 for _ in devices]

    def cost(group):
        first = group[0]["problem"]
        weight = first["n"] * first["k"] * first.get("experts", 1)
        flops = sum(
            r["problem"]["n"]
            * r["problem"]["k"]
            * r["problem"].get("m", r["problem"].get("total_rows"))
            * len(r["symbols"])
            for r in group
        )
        return weight + flops / 128

    for group in sorted(groups.values(), key=lambda g: (-cost(g), g[0]["id"])):
        owners = {r.get("worker_index") for r in group}
        if owners == {None}:
            owner = min(range(len(devices)), key=lambda i: (costs[i], i))
        elif len(owners) == 1 and all(
            type(i) is int and 0 <= i < len(devices) for i in owners
        ):
            owner = next(iter(owners))
        else:
            raise ValueError(
                "weight group has inconsistent/faulty fixed worker assignment"
            )
        assigned[owner].append(group)
        costs[owner] += cost(group)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("inputs", "results", "logs"):
        (output / directory).mkdir(exist_ok=True)
    # Advisory lock prevents accidental simultaneous reuse of one epoch.
    lock = (output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    epoch = {
        "plan_sha256": digest(plan),
        "bundle_sha256": digest(bundle),
        "iterations": iterations,
        "correctness_repeats": 1,
        "devices": devices,
        "device_identity": probe_devices(bundle, devices, sdk),
        "assignment": [[r["id"] for g in groups_ for r in g] for groups_ in assigned],
    }
    if any(
        int(r["cu"]) != plan["compute_units_model"] for r in epoch["device_identity"]
    ):
        raise ValueError(
            "device compute units differ from the candidate model; replan with --cu"
        )
    path = output / "epoch.json"
    if path.exists() and json.loads(path.read_text()) != epoch:
        raise ValueError("timing epoch changed; use a fresh output directory")
    atomic_json(path, epoch)
    for relative, text in plan["router_files"].items():
        path = output / "inputs" / relative
        path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_text() != text:
            raise ValueError("router rows changed")
        path.write_text(text)
    progress_lock = threading.Lock()
    completed = 0
    start = time.monotonic()

    def progress(request):
        nonlocal completed
        with progress_lock:
            completed += 1
            print(
                f"KPACK_TUNER_PROGRESS completed={completed}/{len(plan['requests'])} elapsed_minutes={(time.monotonic()-start)/60:.1f} last={request['cell_key']}/{request['route']}",
                flush=True,
            )

    failures = []

    def worker(index):
        for group in assigned[index]:
            if STOP.is_set():
                return
            try:
                run_group(
                    group,
                    devices[index],
                    bundle,
                    output,
                    sdk,
                    iterations,
                    timeout,
                    retry,
                    progress,
                )
            except Exception as error:
                with progress_lock:
                    failures.append(str(error))
                print(
                    f"KPACK_TUNER_GROUP_FAIL device={devices[index]} reason={error}",
                    flush=True,
                )

    STOP.clear()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():

        def interrupt(signum, _frame):
            STOP.set()
            with DRIVERS_LOCK:
                for driver in list(DRIVERS):
                    if driver.process.poll() is None:
                        driver.process.terminate()
            print(
                f"KPACK_TUNER_INTERRUPTED signal={signum} completed_results_preserved=1",
                flush=True,
            )

        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            for future in as_completed(
                [pool.submit(worker, i) for i in range(len(devices))]
            ):
                future.result()
        report = summarize(plan, output)
        atomic_json(output / "failures.json", failures)
        print(
            f"KPACK_TUNER_DONE status={report['status']} failures={len(failures)} output={output}",
            flush=True,
        )
        return report
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--retry-failures", action="store_true")
    a = p.parse_args()
    report = run(
        json.loads(a.plan.read_text()),
        json.loads(a.bundle.read_text()),
        a.output.resolve(),
        a.sdk.resolve(),
        list(map(int, a.devices.split(","))),
        a.iterations,
        a.timeout,
        a.retry_failures,
    )
    return 0 if report["status"] == "SCREEN_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Budget-admitted overnight K-pack tuning; compile time is inside the budget.

No exhaustive fallback. Calibration must predict full coverage plus 3x11
confirmation within the available window before the main build is admitted.
Deadlines preserve completed work; incomplete work never becomes a PASS.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

import build_kpack_tuner as build
import kpack_overnight_search as search
import kpack_tuning_plan as tuning
import run_kpack_tuner as runner

ROOT = tuning.ROOT


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"immutable phase inputs changed: {path}; use a fresh OUT")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        runner.atomic_json(path, value)


def subset_bundle(bundle, plan):
    """Reuse registered modules, never alter their code/SDK/source identity."""
    pairs, payloads = {}, {}
    for c in plan["candidates"]:
        key = f"q{c['qtype']}-{c['route']}"
        pair = bundle["pairs"].get(key)
        if pair is None:
            return None
        modules = [m for m in pair["modules"] if c["symbol"] in m["symbols"]]
        if len(modules) != 1:
            return None
        target = pairs.setdefault(key, {**pair, "modules": []})
        if modules[0] not in target["modules"]:
            target["modules"].append(modules[0])
    for pair in pairs.values():
        for path in [pair["driver"]] + [m["path"] for m in pair["modules"]]:
            if (
                path not in bundle["payloads"]
                or build.sha(Path(path)) != bundle["payloads"][path]
            ):
                raise ValueError(f"changed module payload: {path}")
            payloads[path] = bundle["payloads"][path]
    return {
        **bundle,
        "plan_sha256": tuning.digest(plan),
        "pairs": pairs,
        "payloads": payloads,
    }


def collect(plan, directory):
    data = {}
    epoch = directory / "epoch.json"
    if not epoch.exists():
        return data
    identity = json.loads(epoch.read_text())
    if identity["plan_sha256"] != tuning.digest(plan):
        raise ValueError("measurement plan identity differs")
    bundle = directory.parent / "bundle.json"
    if bundle.exists() and identity["bundle_sha256"] != tuning.digest(
        json.loads(bundle.read_text())
    ):
        raise ValueError("measurement bundle identity differs")
    for r in plan["requests"]:
        path = directory / "results" / (r["id"] + ".json")
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if (
            record["request_sha256"] != tuning.digest(r)
            or record["device"] != identity["devices"][r["worker_index"]]
        ):
            raise ValueError("measurement request/device differs")
        runner.verify_result(record, r, identity["iterations"])
        data[r["id"]] = record["cells"]
    return data


def measured_count(data):
    return sum(bool(search.parent_ranking(cells)[0]) for cells in data.values())


def merged(*datasets):
    result = {}
    for data in datasets:
        for rid, cells in data.items():
            result.setdefault(rid, []).extend(cells)
    return result


def calibration_build_seconds(folder: Path, jobs: int) -> float:
    """A cached calibration still owes compilation time for unseen parents."""
    cost_path = folder / "build-cost.json"
    elapsed = (
        float(json.loads(cost_path.read_text())["seconds"])
        if cost_path.exists()
        else 0.0
    )
    bundle_path = folder / "bundle.json"
    durations = []
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text())
        for pair in bundle["pairs"].values():
            for module in pair["modules"]:
                receipt = Path(module["path"]).with_suffix(".receipt.json")
                if receipt.exists():
                    value = float(json.loads(receipt.read_text())["seconds"])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("invalid cached compile duration")
                    durations.append(value)
    if durations:
        elapsed = max(elapsed, max(durations), sum(durations) / jobs)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(
            "cold compile costs unavailable; cannot admit unseen builds at zero cost"
        )
    return elapsed


class CostModel:
    """Conservative measured wall-time scaling, not kernel-count-only scaling."""

    def __init__(self, plan, directory, build_seconds, workers, safety=2.0):
        self.rates, self.workers, self.safety = {}, workers, safety
        self.compile_per_parent = build_seconds / max(1, len(plan["candidates"]))
        for r in plan["requests"]:
            path = directory / "results" / (r["id"] + ".json")
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            elapsed, fixture = 0.0, 0.0
            for entry in record["logs"]:
                text = Path(entry["path"]).read_text()
                elapsed += sum(
                    float(x)
                    for x in re.findall(
                        r"KPACK_TUNER_END .*?wall_seconds=([\d.]+)", text
                    )
                )
                fixture += sum(
                    float(x)
                    for x in re.findall(
                        r"KPACK_TUNER_PHASE phase=fixture seconds=([\d.]+)", text
                    )
                )
            if elapsed <= 0 or not search.parent_ranking(record["cells"])[0]:
                continue
            timed = (
                sum(
                    sum(c["samples_us"])
                    for c in record["cells"]
                    if c["status"] == "MEASURED"
                )
                / 1e6
            )
            p = r["problem"]
            feature = p["n"] * p["k"] * p.get("experts", 1) + search.work(r) / 256
            parents = len(r["symbols"])
            rate = (
                max(fixture, 0.01) / feature,
                max(elapsed - fixture - timed, 0.01) / parents,
                timed / max(1, search.work(r) * parents * 3),
            )
            key = self.key(r)
            old = self.rates.get(key, (0, 0, 0))
            self.rates[key] = tuple(max(a, b) for a, b in zip(old, rate))

    @staticmethod
    def key(r):
        # Do not extrapolate decode's mostly fixed latency linearly to M4096.
        p = r["problem"]
        return (
            r["qtype"],
            r["route"],
            "decode" if p.get("m", p.get("max_rows", 1)) <= 8 else "prefill",
        )

    def run_seconds(self, plan, iterations):
        costs = [0.0] * self.workers
        for r in plan["requests"]:
            key = self.key(r)
            if key not in self.rates:
                raise ValueError(f"calibration has no measured wall cost for {key}")
            a, b, c = self.rates[key]
            p, parents = r["problem"], len(r["symbols"])
            feature = p["n"] * p["k"] * p.get("experts", 1) + search.work(r) / 256
            seconds = (
                0.1
                + a * feature
                + b * parents * max(1, iterations / 3)
                + c * search.work(r) * parents * iterations
            )
            costs[r["worker_index"]] += seconds
        return max(costs, default=0) * self.safety

    def compile_seconds(self, count):
        return self.compile_per_parent * max(count, 0) * self.safety


class Campaign:
    def __init__(self, args):
        self.a = args
        self.output, self.cache = args.output, args.build_cache or args.output / "build"
        self.start = time.monotonic()
        self.end = self.start + args.hours * 3600
        self.active = None
        self.interrupted = False
        self.phases = []
        self.build_seconds = 0.0
        self.kernel_source = build.source_identity()
        self.sdk_identity = build.sdk_identity(args.sdk)

    def validate_bundle(self, bundle):
        identity = bundle["identity"]
        if (
            identity["source"] != self.kernel_source
            or identity["sdk"] != self.sdk_identity
            or identity["flags"] != build.FLAGS
        ):
            raise ValueError(
                "cached bundle source/SDK/flags differ; use a fresh build cache"
            )

    def command(self, argv, log, cutoff):
        if self.interrupted or time.monotonic() >= cutoff:
            return 124, 0.0
        start = time.monotonic()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write("ARGV " + json.dumps(argv) + "\n")
            stream.flush()
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self.active = proc
            lines = queue.Queue()

            def read():
                try:
                    for line in proc.stdout:
                        lines.put(line)
                finally:
                    lines.put(None)

            thread = threading.Thread(target=read, daemon=True)
            thread.start()
            stopped, stop_time, last = False, None, start
            try:
                while True:
                    now = time.monotonic()
                    if (self.interrupted or now >= cutoff) and not stopped:
                        proc.terminate()
                        stopped, stop_time = True, now
                        print(
                            f"KPACK_OVERNIGHT_DEADLINE log={log} completed_work_preserved=1",
                            flush=True,
                        )
                    if stopped and now - stop_time > 60 and proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    try:
                        line = lines.get(timeout=0.2)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        break
                    if line:
                        stream.write(line)
                        stream.flush()
                        print(line, end="", flush=True)
                    if now - last >= 30:
                        print(
                            f"KPACK_OVERNIGHT_BUDGET remaining_minutes={max(0,(self.end-now)/60):.1f} phase_remaining_minutes={max(0,(cutoff-now)/60):.1f}",
                            flush=True,
                        )
                        last = now
                rc = proc.wait()
                return (124 if stopped else rc), time.monotonic() - start
            finally:
                self.active = None
                if proc.poll() is None:
                    proc.terminate()
                thread.join(timeout=1)

    def phase(self, name, plan, cutoff, iterations=3):
        folder = self.output / "phases" / name
        folder.mkdir(parents=True, exist_ok=True)
        freeze(folder / "plan.json", plan)
        if not plan["requests"]:
            self.phases.append({"name": name, "status": "EMPTY"})
            return {}
        bundle_path = folder / "bundle.json"
        if not bundle_path.exists():
            current = self.cache / "bundle.json"
            bundle = None
            if current.exists():
                cached = json.loads(current.read_text())
                self.validate_bundle(cached)
                bundle = subset_bundle(cached, plan)
            if bundle is None:
                rc, elapsed = self.command(
                    [
                        sys.executable,
                        "-u",
                        "-B",
                        str(ROOT / "tools/build_kpack_tuner.py"),
                        "--plan",
                        str(folder / "plan.json"),
                        "--output",
                        str(self.cache),
                        "--sdk",
                        str(self.a.sdk),
                        "--jobs",
                        str(self.a.jobs),
                        "--parents-per-module",
                        "1",
                    ],
                    folder / "build.log",
                    cutoff,
                )
                self.build_seconds += elapsed
                runner.atomic_json(folder / "build-cost.json", {"seconds": elapsed})
                if rc:
                    self.phases.append(
                        {"name": name, "status": "BUILD_FAILED_OR_BUDGET", "rc": rc}
                    )
                    return collect(plan, folder / "run")
                bundle = json.loads(current.read_text())
            freeze(bundle_path, bundle)
        self.validate_bundle(json.loads(bundle_path.read_text()))
        rc, seconds = self.command(
            [
                sys.executable,
                "-u",
                "-B",
                str(ROOT / "tools/run_kpack_tuner.py"),
                "--plan",
                str(folder / "plan.json"),
                "--bundle",
                str(bundle_path),
                "--output",
                str(folder / "run"),
                "--sdk",
                str(self.a.sdk),
                "--devices",
                self.a.devices,
                "--iterations",
                str(iterations),
            ],
            folder / "run.log",
            cutoff,
        )
        data = collect(plan, folder / "run")
        epoch_path = folder / "run/epoch.json"
        if epoch_path.exists():
            device_identity = json.loads(epoch_path.read_text())["device_identity"]
            freeze(self.output / "device-identity.json", device_identity)
        self.phases.append(
            {
                "name": name,
                "rc": rc,
                "seconds": seconds,
                "measured_requests": measured_count(data),
                "expected_requests": len(plan["requests"]),
            }
        )
        return data

    def frozen_stage(self, name, generate):
        path = self.output / "phases" / name / "plan.json"
        return json.loads(path.read_text()) if path.exists() else generate()

    def finish(self, base, confirm, rounds, initial, reason, audit=None):
        rows = search.confirmed_rows(base, confirm, rounds, initial)
        complete = all(r["status"] == "CONFIRMED_SELECTED_SET" for r in rows)
        output = self.output / "results"
        output.mkdir(exist_ok=True)
        # Equal columns on missing/noisy/complete rows; do not silently erase holes.
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (output / "summary.tsv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "schema": "quactlize.kpack-overnight.v1",
            "status": "CONFIRMED_SELECTED_SET" if complete else "INCOMPLETE",
            "reason": reason,
            "budget_hours": self.a.hours,
            "elapsed_hours": (time.monotonic() - self.start) / 3600,
            "requests": len(rows),
            "confirmed_requests": sum(
                r["status"] == "CONFIRMED_SELECTED_SET" for r in rows
            ),
            "global_5pct_bound_proven": False,
            "production_policy_updated": False,
            "phases": self.phases,
            "audit": audit or {},
            "rows": rows,
        }
        runner.atomic_json(output / "summary.json", report)
        runner.atomic_json(
            output / "heuristic-input.json",
            {
                "schema": "quactlize.kpack-measured-exact-candidates.v1",
                "scope": "PER_ROUTE_FULL_OUTPUT_NO_PREPASS_AMORTIZATION",
                "quality": report["status"],
                "rows": [
                    {"request": r, "measurement": m}
                    for r, m in zip(base["requests"], rows)
                ],
            },
        )
        print(
            f"KPACK_OVERNIGHT_DONE status={report['status']} confirmed={report['confirmed_requests']}/{len(rows)} reason={reason} results={output}",
            flush=True,
        )
        return 0 if complete else 2

    def execute(self):
        a = self.a
        base_path = self.output / "base-plan.json"
        if not base_path.exists():
            raw = self.output / "source-plan.json"
            rc, _ = self.command(
                [
                    sys.executable,
                    "-u",
                    "-B",
                    str(ROOT / "tools/kpack_tuning_plan.py"),
                    "--output",
                    str(raw),
                    "--budget",
                    str(a.budget),
                    "--qtypes",
                    a.qtypes,
                ],
                self.output / "planning.log",
                self.end,
            )
            if rc:
                raise ValueError("base planning failed or exceeded budget")
            freeze(
                base_path,
                search.fix_workers(
                    json.loads(raw.read_text()), len(a.devices.split(","))
                ),
            )
        base = json.loads(base_path.read_text())
        empty = search.make_stage(base, {}, "not-started")
        # First measure every format/route. Cap admission work to 20% of the
        # night, instead of discovering an impossible estimate after hours.
        calibration = self.frozen_stage(
            "calibration", lambda: search.calibration_plan(base)
        )
        probe = self.phase(
            "calibration", calibration, min(self.end, self.start + a.hours * 720)
        )
        if measured_count(probe) != len(calibration["requests"]):
            return self.finish(
                base, empty, [], {}, "CALIBRATION_INCOMPLETE_NO_FULL_CAMPAIGN_STARTED"
            )
        build_cost = calibration_build_seconds(
            self.output / "phases/calibration", a.jobs
        )
        model = CostModel(
            calibration,
            self.output / "phases/calibration/run",
            build_cost,
            len(a.devices.split(",")),
        )
        # Reserve for up to 9 parents: top 4 + 4 near ties + initial incumbent.
        reservation = copy.deepcopy(base)
        for r in reservation["requests"]:
            r["symbols"] = (r["symbols"] * 9)[:9]
        confirm_seconds = 3 * model.run_seconds(reservation, 11)
        compiled = {c["symbol"] for c in calibration["candidates"]}
        missing = len({c["symbol"] for c in base["candidates"]} - compiled)
        estimate = (
            model.compile_seconds(missing)
            + model.run_seconds(base, 3)
            + confirm_seconds
            + 300
        )
        admission = {
            "source": "MEASURED_WALL_FIXTURE_AND_KERNEL_SCALING",
            "safety_factor": model.safety,
            "remaining_seconds": self.end - time.monotonic(),
            "mandatory_estimate_seconds": estimate,
            "compile_estimate_seconds": model.compile_seconds(missing),
            "screen_estimate_seconds": model.run_seconds(base, 3),
            "confirmation_reserve_seconds": confirm_seconds,
            "verdict": "ADMIT" if estimate <= self.end - time.monotonic() else "REJECT",
        }
        runner.atomic_json(self.output / "budget-admission.json", admission)
        print(
            "KPACK_OVERNIGHT_ADMISSION " + json.dumps(admission, sort_keys=True),
            flush=True,
        )
        if admission["verdict"] != "ADMIT":
            return self.finish(
                base,
                empty,
                [],
                {},
                "PREDICTED_BUDGET_EXCEEDED_NO_FULL_CAMPAIGN_STARTED",
            )
        search_end = self.end - confirm_seconds - 300
        initial = self.phase("screen", base, search_end)
        if measured_count(initial) != len(base["requests"]):
            return self.finish(
                base, empty, [], initial, "SCREEN_INCOMPLETE_NO_CONFIRMED_POLICY"
            )
        pool = merged(initial)
        compiled.update(c["symbol"] for c in base["candidates"])
        # Optional work uses only spare search time, never the reserved 3x11.
        neighbors = self.frozen_stage(
            "neighbors",
            lambda: search.explore_plan(
                base, pool, compiled, new_limit=a.new_parents, deadline=search_end
            ),
        )
        extra = self.phase(
            "neighbors",
            neighbors,
            time.monotonic() + max(0, search_end - time.monotonic()) * 0.45,
        )
        pool = merged(pool, extra)
        compiled.update(
            c["symbol"]
            for c in neighbors["candidates"]
            if c["symbol"] in {x["symbol"] for cells in extra.values() for x in cells}
        )
        audit_plan = self.frozen_stage(
            "audit",
            lambda: search.explore_plan(
                base,
                pool,
                compiled,
                audit=True,
                new_limit=a.new_parents,
                deadline=search_end,
            ),
        )
        audit_data = self.phase(
            "audit",
            audit_plan,
            time.monotonic() + max(0, search_end - time.monotonic()) * 0.65,
        )
        gains = search.audit_gains(audit_plan, audit_data)
        pool = merged(pool, audit_data)
        propagation = self.frozen_stage(
            "family-propagation", lambda: search.propagation_plan(base, pool, gains)
        )
        spread = self.phase("family-propagation", propagation, search_end)
        pool = merged(pool, spread)
        confirm = self.frozen_stage(
            "confirmation-input", lambda: search.confirmation_plan(base, pool, initial)
        )
        freeze(self.output / "phases/confirmation-input/plan.json", confirm)
        rounds = []
        for number in (1, 2, 3):
            rounds.append(
                self.phase(
                    f"confirm-{number}",
                    search.round_plan(confirm, number),
                    self.end - 120,
                    iterations=11,
                )
            )
        audit_report = {
            "selected_requests": len(audit_plan["requests"]),
            "measured_requests": measured_count(audit_data),
            "gain_families": [
                {"family": list(k), "winners": sorted(v)}
                for k, v in sorted(gains.items())
            ],
            "propagation_requests": len(propagation["requests"]),
            "propagation_measured": measured_count(spread),
            "neighborhood_compile_capped": neighbors["stage_details"].get(
                "compile_cap_omissions", 0
            ),
            "audit_compile_capped": audit_plan["stage_details"].get(
                "compile_cap_omissions", 0
            ),
        }
        return self.finish(
            base,
            confirm,
            rounds,
            initial,
            "FINISHED_OR_DEADLINE_WITH_EXPLICIT_COVERAGE",
            audit_report,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--build-cache", type=Path)
    p.add_argument("--hours", type=float, default=10)
    p.add_argument("--jobs", type=int, default=192)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--qtypes", default="10,11,12,13,14")
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--new-parents", type=int, default=256)
    a = p.parse_args()
    if (
        not math.isfinite(a.hours)
        or not 0.1 <= a.hours <= 24
        or a.jobs < 1
        or a.new_parents < 0
    ):
        p.error("hours must be 0.1..24; jobs positive; new-parents nonnegative")
    if (
        not a.output.is_absolute()
        or a.output.is_symlink()
        or str(a.output) in ("/", "/root", "/workspace", "/root/autodl-tmp")
    ):
        p.error("OUT must be an absolute dedicated child directory, not a symlink")
    a.output, a.sdk = a.output.resolve(), a.sdk.resolve()
    if str(a.output) in ("/", "/root", "/workspace", "/root/autodl-tmp"):
        p.error("resolved OUT must be a dedicated child directory")
    devices = [int(x) for x in a.devices.split(",")]
    qtypes = [int(x) for x in a.qtypes.split(",")]
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        p.error("unique nonnegative device ordinals required")
    if (
        not qtypes
        or not set(qtypes) <= {10, 11, 12, 13, 14}
        or len(set(qtypes)) != len(qtypes)
    ):
        p.error("unique qtypes from 10..14 required")
    if not 4 <= a.budget <= 128:
        p.error("budget must be 4..128")
    os.environ["PATH"] = str(a.sdk / "bin") + ":" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(str(a.sdk / d) for d in ("lib", "lib64", "targets/x86_64-linux/lib"))
        + ":"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    if a.build_cache:
        a.build_cache = a.build_cache.resolve()
        if a.build_cache == a.output or str(a.build_cache) in (
            "/",
            "/root",
            "/workspace",
            "/root/autodl-tmp",
        ):
            p.error("build cache must be a dedicated child directory")
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "night.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "kernel_source": build.source_identity(),
            "sdk": build.sdk_identity(a.sdk),
            "orchestrator": [
                (name, build.sha(ROOT / "tools" / name))
                for name in (
                    "run_kpack_overnight.py",
                    "kpack_overnight_search.py",
                    "run_kpack_tuner.py",
                    "kpack_tuning_plan.py",
                    "probe_box_identity.py",
                    "box_identity_probe.cpp",
                    "box_identity_schema.py",
                )
            ],
            "devices": a.devices,
            "qtypes": a.qtypes,
            "budget": a.budget,
            "new_parents": a.new_parents,
        }
        freeze(a.output / "campaign-identity.json", identity)
        campaign = Campaign(a)

        def stop(signum, _frame):
            campaign.interrupted = True
            if campaign.active and campaign.active.poll() is None:
                campaign.active.terminate()
            print(f"KPACK_OVERNIGHT_INTERRUPTED signal={signum}", flush=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        return campaign.execute()


if __name__ == "__main__":
    raise SystemExit(main())

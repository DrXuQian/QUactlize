#!/usr/bin/env python3
"""Read-only rolling ETA for an already running K-pack overnight campaign.

No SDK, device access, campaign restart, or receipt changes. Remaining time is
for the CURRENT phase: future adaptive plans are not a known denominator yet.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import fcntl
import json
import math
from pathlib import Path
import re
import time

PHASES = (
    "calibration",
    "screen",
    "neighbors",
    "audit",
    "family-propagation",
    "confirm-1",
    "confirm-2",
    "confirm-3",
)


class Reader:
    """Cache unchanged JSON; never re-read all large result files every tick."""

    def __init__(self):
        self.cache = {}

    def json(self, path, fields=None):
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            key = (path, fields)
            old = self.cache.get(key)
            if old and old[0] == stamp:
                return old[1]
            value = json.loads(path.read_text())
            if fields is not None:
                value = {field: value.get(field) for field in fields}
            self.cache[key] = (stamp, value)
            return value
        except (OSError, ValueError):
            # A receipt may be between creation and close. It is not complete.
            return None


def build_log(path):
    # This coordinator log contains one short row per unit, not compiler stderr.
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def campaign_running(output):
    try:
        with (output / "night.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock, fcntl.LOCK_UN)
    except FileNotFoundError:
        pass
    return False


def snapshot(output, reader):
    running = campaign_running(output)
    final = reader.json(output / "results/summary.json")
    if not running and final:
        return {
            "phase": "finished",
            "status": final["status"],
            "counters": {},
            "completed": final.get("confirmed_requests", 0),
            "total": final.get("requests", 0),
            "failed": 0,
        }
    name = next(
        (n for n in reversed(PHASES) if (output / "phases" / n / "plan.json").exists()),
        None,
    )
    if name is None:
        return {
            "phase": "planning",
            "status": "RUNNING" if running else "IDLE",
            "counters": {},
            "completed": 0,
            "total": 0,
            "failed": 0,
        }
    folder = output / "phases" / name
    plan = reader.json(folder / "plan.json")
    result = {
        "phase": name,
        "status": "RUNNING" if running else "STOPPED",
        "counters": {},
        "completed": 0,
        "total": 0,
        "failed": 0,
    }
    if plan is None:
        return result
    # This includes downtime on resumption, hence 'age', not active runtime.
    result["phase_age_minutes"] = max(
        0, (time.time() - (folder / "plan.json").stat().st_mtime) / 60
    )
    if not plan["requests"]:
        result["phase"] += "/empty"
        return result
    if not (folder / "bundle.json").exists():
        result["phase"] += "/compile"
        text = build_log(folder / "build.log")
        lines = re.findall(
            r"KPACK_TUNER_BUILD(?:_PROGRESS)? completed=(\d+)/(\d+)", text
        )
        if lines:
            done, total = map(int, lines[-1])
            result.update(
                completed=done, total=total, counters={"compile": (done, total)}
            )
            if done == total:
                # Object completion is not link/receipt completion.
                result.update(phase=name + "/link", counters={})
        result["failed"] = len(re.findall(r"KPACK_TUNER_BUILD .*status=FAIL", text))
        return result
    result["phase"] += "/test"
    counters = defaultdict(lambda: [0, 0])
    for request in plan["requests"]:
        worker = str(request["worker_index"])
        counters[worker][1] += 1
        record = reader.json(
            folder / "run/results" / (request["id"] + ".json"), ("status",)
        )
        if record and record.get("status") in ("MEASURED", "NO_VALID_CANDIDATE"):
            counters[worker][0] += 1
            result["failed"] += record["status"] == "NO_VALID_CANDIDATE"
    result["counters"] = {w: tuple(counts) for w, counts in counters.items()}
    result["completed"] = sum(c[0] for c in counters.values())
    result["total"] = sum(c[1] for c in counters.values())
    return result


class RollingETA:
    """Slowest unfinished worker, using recent completion rate, not a deadline.

    The first interval establishes a baseline, excluding the initial burst of
    cached builds/results. Unknown or stalled workers never count as zero cost.
    """

    def __init__(self, window=300, warmup=30):
        self.window, self.warmup = window, warmup
        self.phase = None
        self.status = None
        self.started = 0.0
        self.baseline_ready = False
        self.samples = deque()

    def update(self, observation, now):
        phase, counts = observation["phase"], observation["counters"]
        if (
            phase != self.phase
            or observation["status"] != self.status
            or (
                self.samples
                and (
                    counts.keys() != self.samples[-1][1].keys()
                    or any(
                        counts[w][1] != self.samples[-1][1][w][1]
                        or counts[w][0] < self.samples[-1][1][w][0]
                        for w in counts
                    )
                )
            )
        ):
            self.phase, self.started, self.baseline_ready = phase, now, False
            self.status = observation["status"]
            self.samples.clear()
        if observation["status"] != "RUNNING" or not counts:
            return None
        pending = [w for w, (done, total) in counts.items() if done < total]
        if not pending:
            return 0.0
        if not self.baseline_ready:
            if now - self.started >= self.warmup:
                self.baseline_ready = True
                self.samples.append((now, counts))
            return None
        self.samples.append((now, counts))
        while len(self.samples) > 2 and self.samples[1][0] <= now - self.window:
            self.samples.popleft()
        start, baseline = self.samples[0]
        elapsed = now - start
        if elapsed < self.warmup:
            return None
        estimates = []
        for worker in pending:
            done, total = counts[worker]
            delta = done - baseline[worker][0]
            if delta <= 0:
                return None
            estimates.append((total - done) * elapsed / delta)
        return max(estimates)


def render(observation, remaining, observed_seconds):
    value = "ESTIMATING" if remaining is None else f"{remaining / 60:.1f}"
    if observation["status"] != "RUNNING":
        value = "NOT_RUNNING"
    elif observation["phase"].endswith("/link"):
        value = "ESTIMATING_LINK"
    campaign_remaining = (
        "NOT_RUNNING"
        if observation["status"] != "RUNNING"
        else "UNKNOWN_ADAPTIVE_STAGES"
    )
    return (
        f"KPACK_TUNER_ETA phase={observation['phase']} status={observation['status']} "
        f"completed={observation['completed']}/{observation['total']} "
        f"failed={observation['failed']} observed_minutes={observed_seconds/60:.1f} "
        f"phase_age_minutes={observation.get('phase_age_minutes', 0):.1f} "
        f"remaining_minutes={value} eta_scope=CURRENT_PHASE "
        f"campaign_remaining_minutes={campaign_remaining} advisory_only=1 "
        "method=ROLLING_SLOWEST_WORKER"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.output.is_dir():
        parser.error("existing campaign output directory required")
    if not math.isfinite(args.interval) or not 1 <= args.interval <= 60:
        parser.error("interval must be 1..60 seconds")
    reader, eta, started = Reader(), RollingETA(), time.monotonic()
    try:
        while True:
            state = snapshot(args.output, reader)
            now = time.monotonic()
            print(render(state, eta.update(state, now), now - started), flush=True)
            if args.once or state["phase"] == "finished":
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

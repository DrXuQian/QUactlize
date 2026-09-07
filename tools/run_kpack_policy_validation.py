#!/usr/bin/env python3
"""Run the compact policy's fixed validation suite using existing PPU modules.

No compilation path. Immutable input receipts, 3x11 fresh confirmation rounds,
failure isolation and resumption use the existing tuner. Failed/absent proposed
grids remain explicit; this runner never silently substitutes a measured winner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import statistics
import threading
import time

import build_kpack_tuner as build
import kpack_overnight_search as search
import kpack_policy as policy
from plan_kpack_policy_validation import SCHEMA
import run_kpack_overnight as overnight
import run_kpack_tuner as runner

ROOT = Path(__file__).resolve().parents[1]


def validate_suite(suite):
    if suite.get("schema") != SCHEMA:
        raise ValueError("unsupported validation suite")
    plan = suite["plan"]
    requests = {r["id"]: r for r in plan["requests"]}
    definitions = {c["symbol"]: c for c in plan["candidates"]}
    if (
        len(requests) != len(plan["requests"])
        or len(definitions) != len(plan["candidates"])
        or set(suite["roles"]) != set(requests)
        or not set(suite["targets"]) <= set(requests)
        or suite["limits"]["rounds"] != 3
        or suite["limits"]["iterations_per_round"] != 11
        or suite["limits"]["correctness_repeats"] != 1
    ):
        raise ValueError("validation denominator or fixed measurement controls differ")
    union = set()
    for rid, request in requests.items():
        policy.validate_problem(request["route"], request["problem"])
        if rid != policy.digest([request["cell_key"], request["route"]]):
            raise ValueError("noncanonical validation request ID")
        if not 0 <= request["worker_index"] < 8 or len(set(request["symbols"])) != len(
            request["symbols"]
        ):
            raise ValueError("invalid assignment or duplicate selected parent")
        union.update(request["symbols"])
        seen = set()
        for target in suite["targets"].get(rid, []):
            c = target["config"]
            if (
                target["config_id"] != policy.digest(c)
                or target["config_id"] in seen
                or c["symbol"] not in request["symbols"]
                or c["route"] != request["route"]
                or c["symbol"] not in definitions
                or any(c[k] != v for k, v in definitions[c["symbol"]].items())
                or not policy.admissible(c, request["problem"])
            ):
                raise ValueError(
                    "proposal runtime identity differs from its compiled parent"
                )
            seen.add(target["config_id"])
            policy.resolve_grid(c, request["problem"])
    if union != set(definitions) or suite["denominator"]["route_workloads"] != len(
        requests
    ):
        raise ValueError("compiled parent or request union differs")


def preflight(suite, campaign, sdk):
    validate_suite(suite)
    identity = suite["authority"]["campaign_identity"]
    if json.loads((campaign / "campaign-identity.json").read_text()) != identity:
        raise ValueError("source campaign identity differs")
    if build.source_identity() != identity["kernel_source"]:
        raise ValueError(
            "kernel source changed; old compiled modules cannot validate this source"
        )
    if build.sdk_identity(sdk) != identity["sdk"]:
        raise ValueError("SDK differs from the original compiled module runtime")
    bundle = json.loads((campaign / "phases/confirm-1/bundle.json").read_text())
    if (
        bundle["identity"]["source"] != identity["kernel_source"]
        or bundle["identity"]["sdk"] != identity["sdk"]
    ):
        raise ValueError("compiled bundle source/SDK differs")
    subset = overnight.subset_bundle(bundle, suite["plan"])
    if subset is None:
        raise ValueError(
            "original confirmation bundle lacks selected parents; no compile was attempted"
        )
    return subset


@contextmanager
def progress(folder, total, number, previous_seconds):
    start = time.monotonic()
    initial = len(list((folder / "run/results").glob("*.json")))
    stop = threading.Event()

    def report():
        while not stop.wait(30):
            completed = len(list((folder / "run/results").glob("*.json")))
            elapsed = time.monotonic() - start
            new = completed - initial
            remaining = elapsed / new * max(0, total - completed) if new >= 4 else None
            round_eta = f"{remaining/60:.1f}" if remaining is not None else "UNKNOWN"
            # Future rounds have a fixed workload here, unlike adaptive search.
            future = (
                statistics.median(previous_seconds) * (3 - number)
                if previous_seconds
                else None
            )
            full = (
                f"{(remaining+future)/60:.1f}"
                if remaining is not None and future is not None
                else "UNKNOWN"
            )
            print(
                f"KPACK_POLICY_VALIDATION_PROGRESS round={number}/3 completed={completed}/{total} "
                f"elapsed_minutes={elapsed/60:.1f} round_remaining_minutes={round_eta} "
                f"campaign_remaining_minutes={full} advisory_only=1 compile=0",
                flush=True,
            )

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def assess(suite, rounds, failed):
    records, request_states = [], []
    for r in suite["plan"]["requests"]:
        rid = r["id"]
        maps = [
            {
                search.runtime_key(c): c
                for c in data.get(rid, [])
                if c["status"] == "MEASURED"
            }
            for data in rounds
        ]
        common = set.intersection(*(set(m) for m in maps)) if len(maps) == 3 else set()
        clean = rid not in failed and bool(common)
        request_states.append(
            {
                "id": rid,
                "roles": suite["roles"][rid],
                "status": "MEASURED" if clean else "INCOMPLETE_OR_REJECTED",
            }
        )
        references = (
            [min(m[c]["median_us"] for c in common) for m in maps] if clean else []
        )
        for target in suite["targets"].get(rid, []):
            config = target["config"]
            key = (
                config["symbol"],
                config["algorithm"],
                config["split"],
                policy.resolve_grid(config, r["problem"]),
            )
            row = {
                "id": rid,
                "route": r["route"],
                "problem": r["problem"],
                "roles": suite["roles"][rid],
                "config_id": target["config_id"],
                "symbol": key[0],
                "algorithm": key[1],
                "split": key[2],
                "grid": key[3],
                "status": (
                    "INCOMPLETE_OR_REJECTED"
                    if not clean
                    else "PROPOSED_RUNTIME_UNAVAILABLE"
                ),
            }
            if clean and key in common:
                cells = [m[key] for m in maps]
                if any(len(c["samples_us"]) != 11 for c in cells):
                    raise ValueError("proposal confirmation sample denominator differs")
                times = [c["median_us"] for c in cells]
                regret = max(a / b - 1 for a, b in zip(times, references)) * 100
                spread = (max(times) / min(times) - 1) * 100
                row.update(
                    status=(
                        "WITHIN_BUDGET"
                        if max(regret, spread) <= 5
                        else "OUTSIDE_BUDGET"
                    ),
                    median_us=statistics.median(
                        x for c in cells for x in c["samples_us"]
                    ),
                    round_medians_us=times,
                    max_round_regret_pct=regret,
                    spread_pct=spread,
                )
            records.append(row)
    winner_rows = search.confirmed_rows(suite["plan"], suite["plan"], rounds, {})
    for row, request in zip(winner_rows, suite["plan"]["requests"]):
        if request["id"] in failed:
            row["status"] = "INCOMPLETE_OR_REJECTED"
    return {
        "schema": "quactlize.kpack-policy-validation-result.v1",
        "status": (
            "MEASUREMENT_COMPLETE"
            if all(r["status"] == "MEASURED" for r in request_states)
            else "INCOMPLETE"
        ),
        "suite_sha256": policy.digest(suite),
        "policy_sha256": suite["policy_sha256"],
        "requests": request_states,
        "targets": records,
        "winner_rows": winner_rows,
        "target_status": dict(Counter(r["status"] for r in records)),
        "winner_status": dict(Counter(r["status"] for r in winner_rows)),
        "failed_requests": sorted(failed),
        "production_policy_updated": False,
        "scope": "FROZEN_PROPOSALS_VS_SAME_RUN_SELECTED_SET_NOT_GLOBAL_OPTIMALITY",
    }


def run(suite, campaign, output, sdk, devices, retry=False):
    if output.is_symlink():
        raise ValueError("output must not be a symlink")
    bundle = preflight(suite, campaign, sdk)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        overnight.freeze(output / "suite.json", suite)
        rounds, durations, fresh_durations, failed = [], [], [], set()
        for number in (1, 2, 3):
            folder = output / "phases" / f"confirm-{number}"
            plan = search.round_plan(suite["plan"], 100 + number)
            phase_bundle = {**bundle, "plan_sha256": policy.digest(plan)}
            overnight.freeze(folder / "plan.json", plan)
            overnight.freeze(folder / "bundle.json", phase_bundle)
            fresh = not any((folder / "run/results").glob("*.json"))
            started = time.monotonic()
            print(
                f"KPACK_POLICY_VALIDATION_ROUND round={number}/3 requests={len(plan['requests'])} compile=0 iterations=11",
                flush=True,
            )
            with progress(folder, len(plan["requests"]), number, fresh_durations):
                runner.run(
                    plan, phase_bundle, folder / "run", sdk, devices, 11, retry=retry
                )
            elapsed = time.monotonic() - started
            durations.append(elapsed)
            data = overnight.collect(plan, folder / "run")
            for request in plan["requests"]:
                path = folder / "run/results" / (request["id"] + ".json")
                record = json.loads(path.read_text()) if path.exists() else {}
                if (
                    not record
                    or record.get("rejected")
                    or record.get("infrastructure_failure")
                    or record.get("status") != "MEASURED"
                ):
                    failed.add(request["id"])
            if fresh and len(data) == len(plan["requests"]):
                fresh_durations.append(elapsed)
            rounds.append(data)
            print(
                f"KPACK_POLICY_VALIDATION_ROUND_DONE round={number} seconds={elapsed:.3f}",
                flush=True,
            )
        result = assess(suite, rounds, failed)
        result["round_seconds"] = durations
        (output / "results").mkdir(exist_ok=True)
        runner.atomic_json(output / "results/summary.json", result)
        print(
            "KPACK_POLICY_VALIDATION_DONE "
            + json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "target_status",
                        "winner_status",
                        "round_seconds",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "policies/kpack_zw810_compact.validation.json",
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    try:
        suite = json.loads(args.suite.read_text())
        devices = list(map(int, args.devices.split(",")))
        if len(devices) != 8 or len(set(devices)) != 8 or min(devices) < 0:
            raise ValueError(
                "this fixed validation suite requires eight distinct device ordinals"
            )
        result = run(
            suite,
            args.campaign.resolve(),
            args.output,
            args.sdk.resolve(),
            devices,
            args.retry_failures,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0 if result["status"] == "MEASUREMENT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit/query-independent K-pack rules from an uploaded overnight campaign.

Reads original 3x11 confirmation logs, validates receipts and replays the final
decisions. The generated JSON is a host policy, not a new device library.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics

import kpack_overnight_search as search
import kpack_policy as policy
from export_kpack_tuner_compact import REQUEST_FIELDS, top_cells
from run_kpack_tuner import verify_result

ROOT = Path(__file__).resolve().parents[1]


class Campaign:
    def __init__(self, root):
        self.root = root.resolve()
        self.receipts = {}
        self.log_count = 0

    def read(self, relative):
        raw = (self.root / relative).read_bytes()
        self.receipts[str(relative)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def replay_record(self, record, request, phase):
        # Relocation affects an in-memory copy only. Require the exact phase
        # suffix; a relocated archive cannot redirect the parser outside root.
        replay = copy.deepcopy(record)
        prefix = f"phases/{phase}/run/logs/"
        for log in replay["logs"]:
            old = PurePosixPath(log["path"])
            if ".." in old.parts or str(old).count(prefix) != 1:
                raise ValueError("raw log path has no unique phase suffix")
            relative = prefix + str(old).split(prefix, 1)[1]
            path = self.root / relative
            if not path.resolve().is_relative_to(self.root):
                raise ValueError("raw log escapes the campaign directory")
            log["path"] = str(path)
            self.receipts[relative] = log["sha256"]  # verified by original parser below
            self.log_count += 1
        verify_result(replay, request, 11)

    def load(self):
        base = self.read("base-plan.json")
        final = self.read("results/summary.json")
        identity = self.read("campaign-identity.json")
        devices = self.read("device-identity.json")
        confirm = self.read("phases/confirmation-input/plan.json")
        if final.get("schema") != "quactlize.kpack-overnight.v1":
            raise ValueError(
                "expected overnight results, not an exhaustive cost matrix"
            )
        for filename, expected in identity["orchestrator"]:
            if (
                hashlib.sha256((ROOT / "tools" / filename).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError(f"original parser/orchestrator changed: {filename}")
        if not devices or len({d["pci"] for d in devices}) != len(devices):
            raise ValueError("ambiguous physical device identity")
        if any(d["name"] != "PPU-ZW810" or int(d["cu"]) != 72 for d in devices):
            raise ValueError("unsupported policy device")
        requests = {r["id"]: r for r in base["requests"]}
        if len(requests) != len(base["requests"]) or final["requests"] != len(requests):
            raise ValueError("workload denominator differs")
        for rid, r in requests.items():
            if rid != policy.digest([r["cell_key"], r["route"]]):
                raise ValueError("noncanonical request identity")
        definitions = {c["symbol"]: c for c in confirm["candidates"]}
        if len(definitions) != len(confirm["candidates"]):
            raise ValueError("duplicate candidate definition")
        rounds = []
        for name in ("confirm-1", "confirm-2", "confirm-3"):
            prefix = f"phases/{name}"
            plan = self.read(f"{prefix}/plan.json")
            epoch = self.read(f"{prefix}/run/epoch.json")
            bundle = self.read(f"{prefix}/bundle.json")
            if (
                epoch["plan_sha256"] != policy.digest(plan)
                or epoch["bundle_sha256"] != policy.digest(bundle)
                or epoch["iterations"] != 11
                or epoch["correctness_repeats"] != 1
                or epoch["device_identity"] != devices
                or bundle["identity"]["source"] != identity["kernel_source"]
                or bundle["identity"]["sdk"] != identity["sdk"]
            ):
                raise ValueError(f"{name}: source/SDK/plan/epoch receipt differs")
            if plan["candidates"] != confirm["candidates"]:
                raise ValueError("confirmation configuration definitions changed")
            expected_assignment = [[] for _ in epoch["devices"]]
            for r in plan["requests"]:
                expected_assignment[r["worker_index"]].append(r["id"])
            # Runtime orders weight groups by cost. Membership/ownership, not
            # base-plan list order, is the assignment invariant.
            if list(map(sorted, epoch["assignment"])) != list(
                map(sorted, expected_assignment)
            ):
                raise ValueError("confirmation worker assignment differs")
            data = {}
            for r in plan["requests"]:
                rid = r["id"]
                if (
                    rid in data
                    or rid not in requests
                    or any(r[k] != requests[rid][k] for k in REQUEST_FIELDS)
                ):
                    raise ValueError("confirmation workload changed or duplicated")
                record = self.read(f"{prefix}/run/results/{rid}.json")
                if (
                    record["request_sha256"] != policy.digest(r)
                    or record["device"] != epoch["devices"][r["worker_index"]]
                    or record.get("rejected")
                    or record.get("infrastructure_failure")
                    or record["status"] != "MEASURED"
                ):
                    raise ValueError("request/device/status receipt differs")
                # This validates all finite samples, exact medians, duplicate
                # identities and structural exclusions, not just the winner.
                top_cells(record["cells"], set(r["symbols"]), 11, 1, {})
                self.replay_record(record, r, name)
                data[rid] = record["cells"]
            if set(data) != set(requests):
                raise ValueError("confirmation phase is incomplete")
            rounds.append(data)
            print(
                f"KPACK_POLICY_REPLAY phase={name} requests={len(data)} raw_logs={self.log_count}",
                flush=True,
            )
        reconstructed = search.confirmed_rows(base, confirm, rounds, {})
        if reconstructed != final["rows"] or final["confirmed_requests"] != sum(
            r["status"] == "CONFIRMED_SELECTED_SET" for r in reconstructed
        ):
            raise ValueError("final decisions differ from original 3x11 measurements")
        finals = {(r["cell_key"], r["route"]): r for r in reconstructed}
        configurations, observations = {}, []
        for rid, request in requests.items():
            maps = [
                {
                    search.runtime_key(c): c
                    for c in data[rid]
                    if c["status"] == "MEASURED"
                }
                for data in rounds
            ]
            common = set.intersection(*(set(m) for m in maps))
            costs = {}
            references = [min(m[key]["median_us"] for key in common) for m in maps]
            for key in sorted(common):
                cells = [m[key] for m in maps]
                config = policy.runtime_config(
                    definitions[key[0]], cells[0], request["problem"]
                )
                if config["route"] != request["route"]:
                    raise ValueError("configuration route differs from request")
                cid = policy.digest(config)
                if cid in costs or (
                    cid in configurations and configurations[cid] != config
                ):
                    raise ValueError("runtime identity collision")
                configurations[cid] = config
                times = [c["median_us"] for c in cells]
                costs[cid] = {
                    "regret_pct": max(a / b - 1 for a, b in zip(times, references))
                    * 100,
                    "spread_pct": (max(times) / min(times) - 1) * 100,
                    "median_us": statistics.median(
                        x for c in cells for x in c["samples_us"]
                    ),
                    "round_medians_us": times,
                }
            f = finals[request["cell_key"], request["route"]]
            observations.append(
                {
                    "id": rid,
                    "cell_key": request["cell_key"],
                    "route": request["route"],
                    "problem": request["problem"],
                    "status": f["status"],
                    "costs": costs,
                    "winner_median_us": f["median_us"],
                }
            )
        return (
            observations,
            configurations,
            {
                "campaign_identity": identity,
                "final_summary_sha256": self.receipts["results/summary.json"],
                "receipt_digest": policy.digest(self.receipts),
                "raw_confirmation_logs_replayed": self.log_count,
                "final_decisions_replayed": len(observations),
                "binary_payloads_inspected": False,
                "scope": "THREE_ROUND_CONFIRMATION_SELECTED_SET_NOT_ALL_CONFIGS",
            },
        )


def evaluate(families, configurations, threshold):
    """Training replay and leave-one-PUBLIC-feature-point-out diagnostics."""
    models, replay, holdout = [], [], []
    for key, points in sorted(families.items()):
        model = policy.fit_family(key, points)
        models.append(model)
        for point in points:
            first = point["rows"][0]
            choice = policy.select_family(model, configurations, first["problem"])
            for o in point["rows"]:
                cid = choice.get("config_id")
                cost = o["costs"].get(cid)
                if cid and (
                    cost is None
                    or max(cost["regret_pct"], cost["spread_pct"]) > threshold
                ):
                    raise ValueError("fitted rule violated a training cost constraint")
                replay.append(
                    {
                        "id": o["id"],
                        "qtype": key[0],
                        "route": key[1],
                        "problem": o["problem"],
                        "measurement_status": o["status"],
                        "selection_status": choice["status"],
                        "reason": choice.get("reason", ""),
                        "config_id": cid,
                        "regret_pct": cost["regret_pct"] if cost else None,
                        "median_delta_pct": (
                            (cost["median_us"] / o["winner_median_us"] - 1) * 100
                            if cost
                            else None
                        ),
                    }
                )
            if point["blocked"]:
                continue
            # Remove the entire alias class: otherwise permutation-a would
            # leak the held-out permutation-b public feature into training.
            reduced = policy.fit_family(key, [p for p in points if p is not point])
            guess = policy.select_family(reduced, configurations, first["problem"])
            cid = guess.get("config_id")
            costs = [o["costs"].get(cid) for o in point["rows"]]
            regret = max(c["regret_pct"] for c in costs) if cid and all(costs) else None
            spread = max(c["spread_pct"] for c in costs) if cid and all(costs) else None
            status = "ABSTAINED" if not cid else "UNMEASURED_CANDIDATE"
            if regret is not None:
                status = (
                    "WITHIN_BUDGET"
                    if max(regret, spread) <= threshold
                    else "OUTSIDE_BUDGET"
                )
            holdout.append(
                {
                    "qtype": key[0],
                    "route": key[1],
                    "problem": first["problem"],
                    "ids": [o["id"] for o in point["rows"]],
                    "status": status,
                    "reason": guess.get("reason", ""),
                    "config_id": cid,
                    "regret_pct": regret,
                    "spread_pct": spread,
                }
            )
    return models, replay, holdout


def make_policy(observations, configurations, authority, threshold=5.0):
    if not 0 <= threshold <= 5:
        raise ValueError("regret budget must be in [0,5]; source confirmation uses 5%")
    families = policy.public_points(observations, threshold)
    models, replay, holdout = evaluate(families, configurations, threshold)
    used = set().union(*(policy.tree_configs(m["tree"]) for m in models))
    selected = [r for r in replay if r["config_id"]]
    report = {
        "schema": "quactlize.kpack-policy-fit-report.v1",
        "requests": len(observations),
        "measurement_status": dict(Counter(o["status"] for o in observations)),
        "selection_status": dict(Counter(r["selection_status"] for r in replay)),
        "blocked_reasons": dict(
            Counter(r["reason"] for r in replay if not r["config_id"])
        ),
        "public_points": sum(map(len, families.values())),
        "alias_points": sum(len(p["rows"]) > 1 for ps in families.values() for p in ps),
        "families": len(models),
        "rules": sum(policy.leaf_count(m["tree"]) for m in models),
        "runtime_variants": len(used),
        "parent_symbols": len({configurations[c]["symbol"] for c in used}),
        "training_max_round_regret_pct": max(
            (r["regret_pct"] for r in selected), default=None
        ),
        "training_mean_median_delta_pct": (
            statistics.mean(r["median_delta_pct"] for r in selected)
            if selected
            else None
        ),
        "holdout_status": dict(Counter(r["status"] for r in holdout)),
        "holdout_scope": "LEAVE_ONE_PUBLIC_POINT_OUT; sparse missing costs are UNKNOWN, not PASS",
        "training_bound_scope": "MEASURED_CONFIRMATION_SET_ONLY; not global optimum or interpolation proof",
        "production_policy_updated": False,
    }
    model = {
        "schema": policy.SCHEMA,
        "regret_budget_pct": threshold,
        "device": {"name": "PPU-ZW810", "compute_units": 72},
        "authority": authority,
        "scope": "PER_ROUTE_FULL_OUTPUT; route/prepass amortization is not selected",
        "interpolation": "PROPOSAL_ONLY_WITHIN_OBSERVED_FAMILY_RANGE",
        "grouped_scope": "COMMON_CHOICE_ACROSS_OBSERVED_PUBLIC_FEATURE_ALIASES_NOT_ALL_ROUTER_DISTRIBUTIONS",
        "compiled_default": False,
        "production_policy_updated": False,
        "configurations": {c: configurations[c] for c in sorted(used)},
        "families": models,
    }
    followup = {
        "schema": "quactlize.kpack-policy-followup.v1",
        "scope": "TARGETED_EVIDENCE_GAPS_NOT_A_FULL_SWEEP_PLAN",
        "blocked": [r for r in replay if not r["config_id"]],
        "holdout": [r for r in holdout if r["status"] != "WITHIN_BUDGET"],
        "required_interpolation_validation": "Unmeasured M/router queries remain proposals; validate chosen config against same-run challenges before admission.",
    }
    return model, report, replay, holdout, followup


def recheck_plan(base, confirmation, model, replay):
    """Remeasure blocked public keys, including both sides of router aliases.

    Use the union of their already confirmed parents, so a missing cross-router
    cost can be measured without introducing another compile/search space.
    This is not the interpolation or production shipping gate.
    """
    blocked = {r["id"] for r in replay if not r["config_id"]}
    requests = {r["id"]: r for r in confirmation["requests"]}
    if not blocked <= requests.keys():
        raise ValueError("policy recheck requested an unknown workload")
    choices = {}
    for rid in blocked:
        r = requests[rid]
        key = policy.family_key(r["route"], r["problem"]), policy.features(
            r["route"], r["problem"]
        )
        choices.setdefault(key, set()).update(r["symbols"])
    selections = {}
    for rid in blocked:
        r = requests[rid]
        key = policy.family_key(r["route"], r["problem"]), policy.features(
            r["route"], r["problem"]
        )
        selections[rid] = choices[key]
    result = search.make_stage(
        base,
        selections,
        "policy-recheck",
        salt=100,
        details={
            "policy_sha256": policy.digest(model),
            "reason": "BLOCKED_PUBLIC_KEYS_ONLY",
            "new_parent_union": 0,
            "interpolation_admission": False,
            "production_shipping_gate": False,
        },
    )
    if not {c["symbol"] for c in result["candidates"]} <= {
        c["symbol"] for c in confirmation["candidates"]
    }:
        raise ValueError("recheck unexpectedly requires a new parent")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regret-pct", type=float, default=5.0)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("output already exists; choose a fresh directory")
    try:
        campaign = Campaign(args.source)
        observations, configs, authority = campaign.load()
        model, report, replay, holdout, followup = make_policy(
            observations, configs, authority, args.regret_pct
        )
        args.out.mkdir(parents=True)
        outputs = {
            "policy.json": model,
            "report.json": report,
            "replay.json": replay,
            "holdout.json": holdout,
            "followup.json": followup,
            "source-receipts.json": campaign.receipts,
            "recheck-plan.json": recheck_plan(
                campaign.read("base-plan.json"),
                campaign.read("phases/confirmation-input/plan.json"),
                model,
                replay,
            ),
        }
        for filename, data in outputs.items():
            (args.out / filename).write_text(
                json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        with (args.out / "replay.tsv").open("w") as f:
            fields = (
                "qtype",
                "route",
                "id",
                "measurement_status",
                "selection_status",
                "regret_pct",
                "median_delta_pct",
                "config_id",
                "reason",
            )
            writer = csv.DictWriter(
                f, fieldnames=fields, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(replay)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    print("KPACK_POLICY_FIT " + json.dumps(report, sort_keys=True), flush=True)
    print(f"KPACK_POLICY_OUTPUT path={args.out}", flush=True)


if __name__ == "__main__":
    main()

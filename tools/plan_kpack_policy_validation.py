#!/usr/bin/env python3
"""Bounded, no-new-parent validation of merges and interpolation boundaries."""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import kpack_overnight_search as search
import kpack_policy as policy
import refine_kpack_grid_policy as grids

SCHEMA = "quactlize.kpack-policy-validation.v1"


def make_suite(base, confirmation, observations, configurations, model, per_route=8):
    requests = {r["id"]: copy.deepcopy(r) for r in base["requests"]}
    families = policy.public_points(observations, 5)
    selections, targets, roles = {}, defaultdict(dict), defaultdict(set)
    compiled = {c["symbol"] for c in confirmation["candidates"]}
    confirm_requests = {r["id"]: r for r in confirmation["requests"]}

    def add(rid, cid, role):
        c = configurations[cid]
        if c["symbol"] not in compiled or not policy.admissible(
            c, requests[rid]["problem"]
        ):
            raise ValueError("validation requires a new or inadmissible parent")
        selections.setdefault(rid, set()).add(c["symbol"])
        targets[rid][cid] = c
        roles[rid].add(role)

    # All original blockers; include the other observed alias of a noisy key.
    for key, points in families.items():
        for point in points:
            if not point["blocked"]:
                continue
            parents = set().union(
                *(set(confirm_requests[o["id"]]["symbols"]) for o in point["rows"])
            )
            for o in point["rows"]:
                selections[o["id"]] = set(parents)
                roles[o["id"]].add("blocked-recheck")

    # Propose adjacent-point merges only when no known measured cost rejects
    # the proposed choice. Missing costs stay missing until this device run.
    proposals = defaultdict(lambda: defaultdict(dict))
    for key, points in families.items():
        ordered = sorted(
            (p for p in points if not p["blocked"]), key=lambda p: p["features"]
        )
        for left, right in zip(ordered, ordered[1:]):
            for source, destination in ((left, right), (right, left)):
                for cid in source["costs"]:
                    c = configurations[cid]
                    if not policy.admissible(c, destination["rows"][0]["problem"]):
                        continue
                    known = [
                        o["costs"][cid]
                        for o in destination["rows"]
                        if cid in o["costs"]
                    ]
                    if any(max(v["regret_pct"], v["spread_pct"]) > 5 for v in known):
                        continue
                    for o in destination["rows"]:
                        if cid not in o["costs"]:
                            proposals[key[:2]][o["id"]][cid] = source["costs"][cid][0]
    proposal_count = sum(len(rows) for rows in proposals.values())
    for route_key, rows in sorted(proposals.items()):
        # Prefer high-opportunity requests; deterministic, format/route balanced.
        chosen = sorted(rows, key=lambda rid: (-len(rows[rid]), rid))[:per_route]
        for rid in chosen:
            ranked = sorted(rows[rid], key=lambda c: (rows[rid][c], c))
            used = set()
            for cid in ranked:
                symbol = configurations[cid]["symbol"]
                if symbol in used:
                    continue
                add(rid, cid, "adjacent-merge")
                used.add(symbol)
                if len(used) == 2:
                    break
            r = requests[rid]
            incumbent = policy.select(
                model,
                r["route"],
                r["problem"],
                device_name="PPU-ZW810",
                compute_units=72,
                mapping_id=policy.mapping(r["qtype"]),
            )
            if "config_id" in incumbent:
                add(rid, incumbent["config_id"], "same-run-incumbent")

    # One previously unmeasured M per dense N/K family. Compare the frozen
    # proposal with both adjacent measured-rule parents; no training on results.
    new_count = 0
    for key, points in sorted(families.items()):
        if not key[1].endswith("dense"):
            continue
        ordered = sorted(
            (p for p in points if not p["blocked"]), key=lambda p: p["features"]
        )
        all_m = {p["features"][0] for p in points}
        options = []
        for a, b in zip(ordered, ordered[1:]):
            lo, hi = a["features"][0], b["features"][0]
            if hi - lo < 2:
                continue
            for m in {(lo + hi) // 2, lo + 1, hi - 1} - all_m:
                problem = dict(a["rows"][0]["problem"], m=m)
                guess = policy.select(
                    model,
                    key[1],
                    problem,
                    device_name="PPU-ZW810",
                    compute_units=72,
                    mapping_id=policy.mapping(key[0]),
                )
                if guess["status"] != "INTERPOLATED_PROPOSAL":
                    continue
                ends = [
                    policy.select(
                        model,
                        key[1],
                        p["rows"][0]["problem"],
                        device_name="PPU-ZW810",
                        compute_units=72,
                        mapping_id=policy.mapping(key[0]),
                    )
                    for p in (a, b)
                ]
                different = ends[0].get("config_id") != ends[1].get("config_id")
                # Prefer a transition, then a true midpoint over endpoint-adjacent probes.
                options.append(
                    (
                        (not different, abs(m - (lo + hi) / 2), lo, m),
                        a,
                        problem,
                        guess,
                        ends,
                    )
                )
        if not options:
            continue
        _, anchor, problem, guess, ends = min(options, key=lambda o: o[0])
        r = copy.deepcopy(requests[anchor["rows"][0]["id"]])
        r["problem"] = problem
        r["workload_key"] = (
            f"policy_boundary_m{problem['m']}_n{problem['n']}_k{problem['k']}"
        )
        r["cell_key"] = f"q{key[0]}/dense/{r['workload_key']}"
        r["id"] = policy.digest([r["cell_key"], key[1]])
        r["source_class"] = "policy-interpolation-boundary"
        if r["id"] in requests:
            raise ValueError("new boundary duplicates a measured request")
        requests[r["id"]] = r
        add(r["id"], guess["config_id"], "unmeasured-M-boundary")
        for endpoint in ends:
            cid = endpoint.get("config_id")
            if cid and policy.admissible(configurations[cid], problem):
                selections[r["id"]].add(configurations[cid]["symbol"])
        new_count += 1
    full = dict(base, requests=list(requests.values()))
    plan = search.make_stage(full, selections, "policy-merge-validation", salt=200)
    # Balance this small suite independently while keeping weight families whole.
    plan = search.fix_workers(plan, 8)
    if not {c["symbol"] for c in plan["candidates"]} <= compiled:
        raise ValueError("validation unexpectedly added a compiled type")
    suite = {
        "schema": SCHEMA,
        "policy_sha256": policy.digest(model),
        "authority": model["authority"],
        "plan": plan,
        "targets": {
            rid: [{"config_id": cid, "config": c} for cid, c in sorted(cs.items())]
            for rid, cs in targets.items()
        },
        "roles": {rid: sorted(rs) for rid, rs in roles.items()},
        "limits": {
            "merge_requests_per_qtype_route": per_route,
            "parents_per_merge_request": 3,
            "new_M_per_dense_family": 1,
            "iterations_per_round": 11,
            "rounds": 3,
            "correctness_repeats": 1,
        },
        "denominator": {
            **plan["denominator"],
            "available_merge_requests": proposal_count,
            "selected_merge_requests": sum(
                "adjacent-merge" in r for r in roles.values()
            ),
            "blocked_requests": sum("blocked-recheck" in r for r in roles.values()),
            "new_M_requests": new_count,
            "new_parent_union": 0,
        },
        "scope": "PRIORITY_MERGES_AND_BOUNDARIES_NOT_ALL_INTERPOLATION_OR_GLOBAL_5PCT_PROOF",
    }
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists")
    read = lambda path: json.loads(path.read_text())
    data = read(args.evidence / "measured-costs.json")
    metadata = {
        tuple(k.split("/")): tuple(v)
        for k, v in read(args.evidence / "grid-metadata.json").items()
    }
    observations, configs = grids.expand(
        data["observations"], data["configurations"], metadata
    )
    suite = make_suite(
        read(args.campaign / "base-plan.json"),
        read(args.campaign / "phases/confirmation-input/plan.json"),
        observations,
        configs,
        read(args.policy),
    )
    with args.output.open("x") as stream:
        json.dump(suite, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(
        "KPACK_POLICY_VALIDATION_PLAN "
        + json.dumps(suite["denominator"], sort_keys=True)
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pool N/K families and merge near-equal measured K-pack choices.

No performance threshold is relaxed. Only common, measured in-budget choices
may share a leaf. A greedy parent cover is compared with unrestricted pooling;
it is a maintenance tradeoff, not an asserted minimum set-cover solution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics

import kpack_policy as policy
import refine_kpack_grid_policy as grids

SCHEMA = policy.POOLED_SCHEMA


def group_key(key):
    return key[0], key[1], key[4], key[5]


def parent_cover(points, configurations):
    coverage = defaultdict(int)
    for i, point in enumerate(points):
        for cid in point["costs"]:
            coverage[configurations[cid]["symbol"]] |= 1 << i
    remaining = (1 << len(points)) - 1
    selected = set()
    while remaining:
        best = min(coverage, key=lambda c: (-(coverage[c] & remaining).bit_count(), c))
        if not coverage[best] & remaining:
            raise ValueError("parent cover left an uncovered measured point")
        selected.add(best)
        remaining &= ~coverage[best]
    return selected


def fit_tree(points, configurations, restrict_parents=False):
    parents = parent_cover(points, configurations) if restrict_parents else None
    coverage = defaultdict(int)
    point_costs = []
    features = (*policy.axes(points[0]["rows"][0]["route"]), "n", "k")
    buckets = {axis: defaultdict(int) for axis in features}
    for i, point in enumerate(points):
        costs = {
            c: v
            for c, v in point["costs"].items()
            if parents is None or configurations[c]["symbol"] in parents
        }
        if not costs:
            raise ValueError("pruning removed all eligible candidates")
        point_costs.append(costs)
        for cid in costs:
            coverage[cid] |= 1 << i
        for axis in features:
            buckets[axis][point["rows"][0]["problem"][axis]] |= 1 << i
    full = (1 << len(points)) - 1

    @lru_cache(None)
    def indices(mask):
        return [i for i in range(len(points)) if mask & (1 << i)]

    @lru_cache(None)
    def best_coverage(mask):
        return max((mask & bits).bit_count() for bits in coverage.values())

    def solve(mask):
        common = [c for c, bits in coverage.items() if bits & mask == mask]
        if common:
            # Prefer a broadly reusable in-budget choice before sub-percent
            # differences; all costs here already satisfy the unchanged gate.
            best = min(
                common,
                key=lambda c: (
                    -coverage[c].bit_count(),
                    max(point_costs[i][c][0] for i in indices(mask)),
                    sum(point_costs[i][c][1] for i in indices(mask)),
                    c,
                ),
            )
            return {"config": best}
        splits = []
        for axis in features:
            active = [
                (value, bits & mask)
                for value, bits in sorted(buckets[axis].items())
                if bits & mask
            ]
            left = 0
            for (value, bits), (next_value, _) in zip(active, active[1:]):
                left |= bits
                right = mask & ~left
                ln, rn = left.bit_count(), right.bit_count()
                lb, rb = best_coverage(left), best_coverage(right)
                # Lower-bound leaf estimate, then uncovered points and balance.
                # This is a deterministic greedy tree, not a Cartesian search.
                score = (
                    math.ceil(ln / lb) + math.ceil(rn / rb),
                    ln - lb + rn - rb,
                    max(ln, rn),
                    features.index(axis),
                    value,
                )
                splits.append(
                    (score, axis, math.isqrt(value * next_value), left, right)
                )
        if not splits:
            raise ValueError("conflicting uncoalesced public features")
        _, axis, cut, left, right = min(splits, key=lambda s: s[0])
        return {"feature": axis, "le": cut, "left": solve(left), "right": solve(right)}

    return solve(full)


def choose(tree, problem):
    while "config" not in tree:
        tree = tree["left"] if problem[tree["feature"]] <= tree["le"] else tree["right"]
    return tree["config"]


def fit(observations, configurations, baseline, restrict_parents=False):
    families = policy.public_points(observations, 5)
    groups = defaultdict(list)
    for key, points in families.items():
        groups[group_key(key)].extend(p for p in points if not p["blocked"])
    trees = [
        {"key": list(key), "tree": fit_tree(points, configurations, restrict_parents)}
        for key, points in sorted(groups.items())
    ]
    lookup = {tuple(g["key"]): i for i, g in enumerate(trees)}
    result = copy.deepcopy(baseline)
    result["schema"] = SCHEMA
    result["rule_groups"] = trees
    for family in result["families"]:
        family.pop("tree", None)
        family["rule_group"] = lookup[group_key(family["key"])]
    selected = set().union(*(policy.tree_configs(g["tree"]) for g in trees))
    result["configurations"] = {c: configurations[c] for c in sorted(selected)}
    result["compaction"] = "POOLED_PUBLIC_FEATURES" + (
        "_PARENT_COVER" if restrict_parents else ""
    )
    rows = []
    for key, points in families.items():
        tree = trees[lookup[group_key(key)]]["tree"]
        for point in points:
            if point["blocked"]:
                continue
            cid = choose(tree, point["rows"][0]["problem"])
            for o in point["rows"]:
                c = o["costs"].get(cid)
                if c is None or max(c["regret_pct"], c["spread_pct"]) > 5:
                    raise ValueError(
                        "merged rule lacks a within-budget measured witness"
                    )
                rows.append(
                    {
                        "id": o["id"],
                        "route": o["route"],
                        "problem": o["problem"],
                        "config_id": cid,
                        "regret_pct": c["regret_pct"],
                        "median_delta_pct": (c["median_us"] / o["winner_median_us"] - 1)
                        * 100,
                    }
                )
    report = {
        "compaction": result["compaction"],
        "rules": sum(policy.leaf_count(g["tree"]) for g in trees),
        "shape_guards": len(result["families"]),
        "rule_groups": len(trees),
        "runtime_variants": len(selected),
        "parent_symbols": len({configurations[c]["symbol"] for c in selected}),
        "measured_requests": len(rows),
        "max_round_regret_pct": max(r["regret_pct"] for r in rows),
        "mean_median_delta_pct": statistics.mean(r["median_delta_pct"] for r in rows),
        "regret_budget_pct": 5,
        "production_policy_updated": False,
        "interpolation_admitted": False,
    }
    return result, report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("output exists")
    data = json.loads((args.evidence / "measured-costs.json").read_text())
    metadata = {
        tuple(k.split("/")): tuple(v)
        for k, v in json.loads(
            (args.evidence / "grid-metadata.json").read_text()
        ).items()
    }
    observations, configurations = grids.expand(
        data["observations"], data["configurations"], metadata
    )
    baseline = json.loads((args.evidence / "policy.json").read_text())
    args.out.mkdir(parents=True)
    for label, restrict in (("pooled", False), ("covered", True)):
        result, report, rows = fit(observations, configurations, baseline, restrict)
        for name, value in (("policy", result), ("report", report), ("replay", rows)):
            with (args.out / f"{label}-{name}.json").open("x") as stream:
                json.dump(
                    value,
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
        print("KPACK_POLICY_COMPACT " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

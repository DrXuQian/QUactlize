#!/usr/bin/env python3
"""Small, public-feature K-pack policy; no SDK, compiler or compiled default.

Fitting compresses a sparse measured cost matrix, not winner labels. Each leaf
must have a common, measured configuration within the regret/spread limits at
every training point. Queries between points are explicitly proposals.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "quactlize.kpack-policy.v1"
ROUTES = ("fq-dense", "sf-dense", "fq-grouped", "sf-grouped")
GROUP_SIZE = {10: 16, 11: 16, 12: 32, 13: 32, 14: 16}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def family_key(route, p):
    return (p["qtype"], route, p["n"], p["k"], p["group_size"], p.get("experts", 1))


def axes(route):
    return ("m",) if route.endswith("dense") else ("total_rows", "max_rows")


def features(route, p):
    return tuple(p[a] for a in axes(route))


def mapping(q):
    return "0x51344b5034540001" if q == 12 else "0x514b504b54000001"


def validate_problem(route, p):
    if route not in ROUTES or p.get("qtype") not in GROUP_SIZE:
        raise ValueError("unknown route or format")
    allowed = {"qtype", "n", "k", "group_size", *axes(route)}
    if route.endswith("grouped"):
        allowed.add("experts")
    if set(p) != allowed:
        raise ValueError("query must contain only the public route features")
    if any(type(v) is not int or v <= 0 for v in p.values()):
        raise ValueError("positive integer features required")
    if p["group_size"] != GROUP_SIZE[p["qtype"]] or p["k"] % 256 or p["n"] % 16:
        raise ValueError("canonical group size / K-superblock / N alignment differs")
    if route.endswith("grouped") and not (
        p["max_rows"] <= p["total_rows"] <= p["max_rows"] * p["experts"]
    ):
        raise ValueError("inconsistent grouped row counts")


def admissible(config, p):
    """Known query restrictions, NOT a replacement for device can_implement."""
    if config["qtype"] != p["qtype"] or config["mapping_id"] != mapping(p["qtype"]):
        return False
    if p["k"] % (config["tk"] * config["split"]):
        return False
    if config["ap"] and p.get("m") != 1:
        return False
    if config["route"] == "sf-dense" and config["tm"] == 8 and p["m"] >= 8:
        return False
    if config["route"] == "fq-dense":
        if config["tm"] == 8 and p["m"] > 64:
            return False
        if config["split"] != 1 and p["m"] >= 64:
            return False
    return True


def runtime_config(candidate, cell, problem):
    """Keep all tunable identity; only ordinary SF grid is shape-derived."""
    result = dict(candidate)
    result.update({k: cell[k] for k in ("algorithm", "split", "grid")})
    result["mapping_id"] = mapping(candidate["qtype"])
    result["layout"] = 1 if candidate["qtype"] == 12 else 2
    result["grid_mode"] = "fixed"
    algorithm, route = cell["algorithm"], candidate["route"]
    if route == "fq-dense":
        if (
            algorithm != f"TC_S{cell['split']}"
            or cell["split"] not in (1, 2, 4, 8)
            or cell["grid"] != 0
        ):
            raise ValueError("FQ dense algorithm/split/grid identity differs")
        result["grid_mode"] = "implicit"
    else:
        allowed = (
            ("NONPERSISTENT", "PERSISTENT")
            if route == "sf-dense"
            else ("GROUPED_NONPERSISTENT", "GROUPED_PERSISTENT")
        )
        if algorithm not in allowed or cell["split"] != 1:
            raise ValueError("algorithm is outside the measured route")
        persistent = algorithm in ("PERSISTENT", "GROUPED_PERSISTENT")
        if route == "fq-grouped" and candidate["persistent"] != int(persistent):
            raise ValueError("grouped parent/scheduler identity differs")
        if persistent and cell["grid"] <= 0:
            raise ValueError("persistent grid is missing")
        if not persistent:
            expected = 0
            if route == "sf-dense":
                expected = math.ceil(problem["m"] / candidate["tm"]) * math.ceil(
                    problem["n"] / candidate["tn"]
                )
                result["grid_mode"] = "ordinary"
            if cell["grid"] != expected:
                raise ValueError("ordinary grid does not match its shape formula")
            result["grid"] = 0
    if not admissible(result, problem):
        raise ValueError("measured candidate violates the query restrictions")
    return result


def public_points(observations, threshold):
    grouped = defaultdict(list)
    for o in observations:
        validate_problem(o["route"], o["problem"])
        grouped[
            family_key(o["route"], o["problem"]), features(o["route"], o["problem"])
        ].append(o)
    families = defaultdict(list)
    for (family, point), rows in sorted(grouped.items()):
        common = set.intersection(*(set(o["costs"]) for o in rows))
        eligible = {
            c
            for c in common
            if all(
                max(o["costs"][c]["regret_pct"], o["costs"][c]["spread_pct"])
                <= threshold
                for o in rows
            )
        }
        reason = ""
        if any(o["status"] != "CONFIRMED_SELECTED_SET" for o in rows):
            reason = "NOISY_OR_INCOMPLETE_CONFIRMATION"
        elif not eligible:
            reason = (
                "PUBLIC_FEATURE_CONFLICT"
                if len(rows) > 1
                else "NO_WITHIN_BUDGET_CANDIDATE"
            )
        costs = {
            c: (
                max(o["costs"][c]["regret_pct"] for o in rows),
                sum(o["costs"][c]["regret_pct"] for o in rows),
            )
            for c in eligible
        }
        families[family].append(
            {"features": point, "rows": rows, "costs": costs, "blocked": reason}
        )
    return families


def fit_tree(points):
    """Minimum-leaf axis-aligned partition, then minimum summed regret.

    One axis is an interval dynamic program. Two axes enumerate guillotine
    partitions with memoization. Missing costs are never imputed as fast.
    """

    @lru_cache(None)
    def solve(indices):
        common = set.intersection(*(set(points[i]["costs"]) for i in indices))
        if common:
            best = min(
                common,
                key=lambda c: (
                    max(points[i]["costs"][c][0] for i in indices),
                    sum(points[i]["costs"][c][1] for i in indices),
                    c,
                ),
            )
            return (
                1,
                sum(points[i]["costs"][best][1] for i in indices),
                {"config": best},
            )
        choices = []
        for axis in range(len(points[0]["features"])):
            values = sorted({points[i]["features"][axis] for i in indices})
            for a, b in zip(values, values[1:]):
                # Midpoint in log space respects the powers-of-two M sampling.
                cut = math.isqrt(a * b)
                left = tuple(i for i in indices if points[i]["features"][axis] <= cut)
                right = tuple(i for i in indices if points[i]["features"][axis] > cut)
                l, r = solve(left), solve(right)
                choices.append((l[0] + r[0], l[1] + r[1], axis, cut, l[2], r[2]))
        if not choices:
            raise ValueError("uncoalesced or infeasible public feature point")
        best = min(choices, key=lambda c: c[:4])
        return (
            best[0],
            best[1],
            {"axis": best[2], "le": best[3], "left": best[4], "right": best[5]},
        )

    return solve(tuple(range(len(points))))[2] if points else None


def tree_configs(tree):
    if tree is None:
        return set()
    if "config" in tree:
        return {tree["config"]}
    return tree_configs(tree["left"]) | tree_configs(tree["right"])


def leaf_count(tree):
    if tree is None:
        return 0
    return (
        1 if "config" in tree else leaf_count(tree["left"]) + leaf_count(tree["right"])
    )


def fit_family(key, points):
    admitted = [p for p in points if not p["blocked"]]
    return {
        "key": list(key),
        "axes": list(axes(key[1])),
        "observed": [list(p["features"]) for p in admitted],
        "blocked": [
            {"features": list(p["features"]), "reason": p["blocked"]}
            for p in points
            if p["blocked"]
        ],
        "bounds": (
            [
                [
                    min(p["features"][i] for p in admitted),
                    max(p["features"][i] for p in admitted),
                ]
                for i in range(len(axes(key[1])))
            ]
            if admitted
            else []
        ),
        "tree": fit_tree(admitted),
    }


def select_family(family, configs, problem):
    point = features(family["key"][1], problem)
    blocked = next(
        (p for p in family["blocked"] if tuple(p["features"]) == point), None
    )
    if blocked:
        return {"status": "NO_MEASURED_POLICY", "reason": blocked["reason"]}
    if family["tree"] is None or any(
        not lo <= x <= hi for x, (lo, hi) in zip(point, family["bounds"])
    ):
        return {
            "status": "NO_MEASURED_POLICY",
            "reason": "OUTSIDE_MEASURED_FAMILY_RANGE",
        }
    node = family["tree"]
    while "config" not in node:
        node = node["left"] if point[node["axis"]] <= node["le"] else node["right"]
    cid = node["config"]
    config = configs[cid]
    if not admissible(config, problem):
        return {
            "status": "NO_MEASURED_POLICY",
            "reason": "INTERPOLATED_CONFIG_INADMISSIBLE",
        }
    observed = list(point) in family["observed"]
    resolved = dict(config)
    if config["grid_mode"] == "ordinary":
        resolved["grid"] = math.ceil(problem["m"] / config["tm"]) * math.ceil(
            problem["n"] / config["tn"]
        )
    return {
        "status": "MEASURED_POLICY" if observed else "INTERPOLATED_PROPOSAL",
        "config_id": cid,
        "config": resolved,
        "scope": (
            "OBSERVED_ROUTER_FIXTURES"
            if family["key"][1].endswith("grouped")
            else "OBSERVED_SHAPES"
        ),
        "runtime_validation_required": True,
        "performance_validated_at_query": observed,
    }


def select(policy, route, problem, *, device_name, compute_units, mapping_id):
    if policy.get("schema") != SCHEMA:
        raise ValueError("unsupported policy schema")
    validate_problem(route, problem)
    if (
        device_name != policy["device"]["name"]
        or compute_units != policy["device"]["compute_units"]
    ):
        return {"status": "NO_MEASURED_POLICY", "reason": "DEVICE_MISMATCH"}
    if mapping_id != mapping(problem["qtype"]):
        return {"status": "NO_MEASURED_POLICY", "reason": "MAPPING_MISMATCH"}
    key = list(family_key(route, problem))
    family = next((f for f in policy["families"] if f["key"] == key), None)
    if family is None:
        return {"status": "NO_MEASURED_POLICY", "reason": "UNMEASURED_FAMILY"}
    return select_family(family, policy["configurations"], problem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--qtype", type=int, required=True)
    for arg in ("m", "n", "k", "total-rows", "max-rows", "experts"):
        parser.add_argument("--" + arg, type=int, required=arg in ("n", "k"))
    parser.add_argument("--device-name", default="PPU-ZW810")
    parser.add_argument("--compute-units", type=int, default=72)
    parser.add_argument("--mapping-id")
    args = parser.parse_args()
    try:
        p = {
            a: getattr(args, a)
            for a in ("m", "n", "k", "total_rows", "max_rows", "experts")
            if getattr(args, a) is not None
        }
        p.update(qtype=args.qtype, group_size=GROUP_SIZE[args.qtype])
        result = select(
            json.loads(args.policy.read_text()),
            args.route,
            p,
            device_name=args.device_name,
            compute_units=args.compute_units,
            mapping_id=args.mapping_id or mapping(args.qtype),
        )
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

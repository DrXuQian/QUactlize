#!/usr/bin/env python3
"""Exact measured K-pack tactic lookup; no model fitting or online search.

This is the Python reference for the generated SDK-free C++17 selector.
Unknown inputs require the caller's admitted K-pack fallback. An exception
is numerically measured but does not carry the cross-epoch 5% timing bound.
"""

import argparse
from bisect import bisect_left
import json
from pathlib import Path

import kpack_policy as shared

SCHEMA = "quactlize.kpack-runtime-policy.v1"
ROUTES = shared.ROUTES
WITHIN = "MEASURED_WITHIN_5PCT"
EXCEPTION = "MEASURED_EXCEPTION"


def rows_hash(rows):
    value = 14695981039346656037
    for row in rows:
        for shift in (0, 8, 16, 24):
            value = ((value ^ ((row >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
    return value


def query_key(route, problem, rows=None):
    shared.validate_problem(route, problem)
    if any(x > 2147483647 for x in problem.values()):
        raise ValueError("query exceeds the int32 runtime ABI")
    p = problem
    if route.endswith("dense"):
        if rows is not None:
            raise ValueError("dense query must not supply expert rows")
        return [p["qtype"], ROUTES.index(route), p["n"], p["k"], 1, p["m"], 0, 0]
    if (
        rows is None
        or len(rows) != p["experts"]
        or any(type(x) is not int or x < 0 or x > 2147483647 for x in rows)
        or sum(rows) != p["total_rows"]
        or max(rows) != p["max_rows"]
    ):
        raise ValueError("grouped lookup requires the matching actual expert rows")
    return [
        p["qtype"],
        ROUTES.index(route),
        p["n"],
        p["k"],
        p["experts"],
        p["total_rows"],
        p["max_rows"],
        rows_hash(rows),
    ]


def resolve_grid(config, problem, rows=None):
    mode = config["grid_mode"]
    if mode == "implicit":
        return 0
    ms = [problem["m"]] if "m" in problem else rows
    tiles = sum((m + config["tm"] - 1) // config["tm"] for m in ms) * (
        (problem["n"] + config["tn"] - 1) // config["tn"]
    )
    if mode == "ordinary":
        return tiles
    if (
        mode not in ("capacity", "balanced")
        or not 1 <= config["grid_b"] <= config["occupancy"]
    ):
        raise ValueError("invalid source-owned grid recipe")
    capacity = 72 * config["grid_b"]
    return (
        min(tiles, capacity)
        if mode == "capacity"
        else (tiles + (tiles + capacity - 1) // capacity - 1)
        // ((tiles + capacity - 1) // capacity)
    )


class Selector:
    def __init__(self, model):
        if model.get("schema") != SCHEMA or model.get("policy_digest") != shared.digest(
            {k: v for k, v in model.items() if k != "policy_digest"}
        ):
            raise ValueError("runtime policy schema/digest differs")
        self.model = model
        self.keys = [entry["key"] for entry in model["entries"]]
        if self.keys != sorted(self.keys):
            raise ValueError("runtime table is not sorted")

    def select(
        self,
        route,
        problem,
        rows=None,
        *,
        device_name="PPU-ZW810",
        compute_units=72,
        mapping_id=None,
        kernel_source=None,
        sdk_digest=None,
    ):
        try:
            key = query_key(route, problem, rows)
        except ValueError:
            return dict(status="INVALID_QUERY")
        binding = self.model["required_binding"]
        if (
            device_name != binding["name"]
            or compute_units != binding["compute_units"]
            or mapping_id != binding["mappings"].get(str(problem.get("qtype")))
            or kernel_source != binding["kernel_source"]
            or sdk_digest != binding["sdk_digest"]
        ):
            return dict(status="BINDING_MISMATCH")
        i = bisect_left(self.keys, key)
        while i < len(self.keys) and self.keys[i] == key:
            entry = self.model["entries"][i]
            if rows is None or self.model["row_vectors"][entry["row_vector"]] == rows:
                c = self.model["configurations"][entry["config_id"]]
                return dict(
                    status=entry["status"],
                    config_id=entry["config_id"],
                    config=c,
                    grid=resolve_grid(c, problem, rows),
                    reasons=entry["reasons"],
                    max_measured_regret_pct=entry["max_measured_regret_pct"],
                    max_measured_spread_pct=entry["max_measured_spread_pct"],
                    runtime_admission_required=True,
                )
            i += 1
        return dict(
            status="FALLBACK_REQUIRED",
            fallback="CALLER_ADMITTED_KPACK_TACTIC",
            performance_bound=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--qtype", type=int, required=True)
    for name in ("m", "n", "k", "experts", "total-rows", "max-rows"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--rows-file", type=Path)
    args = parser.parse_args()
    try:
        model = json.loads(args.policy.read_text())
        binding = model["required_binding"]
        p = dict(
            qtype=args.qtype,
            n=args.n,
            k=args.k,
            group_size=shared.GROUP_SIZE.get(args.qtype),
        )
        p.update({a: getattr(args, a) for a in shared.axes(args.route)})
        if args.route.endswith("grouped"):
            p["experts"] = args.experts
        rows = (
            [int(x) for x in args.rows_file.read_text().splitlines()]
            if args.rows_file
            else None
        )
        result = Selector(model).select(
            args.route,
            p,
            rows,
            mapping_id=shared.mapping(args.qtype),
            kernel_source=binding["kernel_source"],
            sdk_digest=binding["sdk_digest"],
        )
        result["binding_scope"] = (
            "REFERENCE_QUERY_USING_POLICY_IDENTITY_NOT_LOADED_DSO_VERIFICATION"
        )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

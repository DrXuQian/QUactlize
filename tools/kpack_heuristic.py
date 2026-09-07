#!/usr/bin/env python3
"""Deterministic single-tactic selection; no compilation or online timing.

Recent measured tactics override historical data. Otherwise retain a measured
exact hit, or propose the nearest eligible profile in the same weight family.
A transfer is never labelled as measured on the query. All results require the
consuming module's actual query/can_implement before launch.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import kpack_policy as common
import kpack_runtime_policy as measured

SCHEMA = "quactlize.kpack-heuristic.v1"
RECENT = "MEASURED_RECENT"
HISTORICAL = "MEASURED_HISTORICAL"
PREDICTED = "HEURISTIC_UNVALIDATED"


def eligible(c, route, p):
    """Cheap module restrictions, not a substitute for the device query."""
    grouped = route.endswith("grouped")
    m = p["total_rows"] if grouped else p["m"]
    return (
        c["route"] == route
        and c["qtype"] == p["qtype"]
        and p["k"] % (c["tk"] * c["split"]) == 0
        and p["k"] // (c["tk"] * c["split"]) >= c["stages"] - 1
        and (not c["ap"] or (not grouped and m == 1))
        and (grouped or c["tm"] != 8 or m <= (64 if route == "fq-dense" else 7))
        and (c["split"] == 1 or (not grouped and m < 64))
        and (
            c["grid_mode"] not in ("capacity", "balanced")
            or 1 <= c["grid_b"] <= c["occupancy"]
        )
    )


def profile(route, problem, rows):
    if route.endswith("dense"):
        return (problem["m"], 0, 0)
    return (problem["total_rows"], problem["max_rows"], sum(r > 0 for r in rows))


def distance(a, b):
    # Fixed-point relative distance is reproducible in Python/C++ and respects
    # scale. The unit is only ranking precision, not a performance threshold.
    return sum(1000000 * abs(x - y) // max(x, y, 1) for x, y in zip(a, b))


def family(route, problem):
    return (
        problem["qtype"],
        measured.ROUTES.index(route),
        problem["n"],
        problem["k"],
        problem.get("experts", 1),
    )


def launch_grid(config, problem, rows):
    # The resident module ABI uses zero for nonpersistent launch. Historical
    # SF logs used the logical CTA count for "ordinary"; do not pass that as a
    # persistent grid to the new module API.
    if config["grid_mode"] in ("implicit", "ordinary"):
        return 0
    return measured.resolve_grid(config, problem, rows)


def validate(model, base):
    measured.Selector(base)
    if (
        model.get("schema") != SCHEMA
        or model.get("model_digest")
        != common.digest({k: v for k, v in model.items() if k != "model_digest"})
        or model.get("base_policy_digest") != base["policy_digest"]
    ):
        raise ValueError("heuristic schema/digest/base policy differs")
    configs = base["configurations"] | model["configurations"]
    binding = model["required_binding"]
    contract = model["authority"]["receipt"]["compiler"]
    if (
        binding["device"] != "PPU-ZW810"
        or binding["compute_units"] != 72
        or any(not binding[k] or binding[k] != contract[k] for k in ("sdk", "kernel"))
    ):
        raise ValueError("heuristic module binding differs")
    if any(cid != common.digest(c) for cid, c in model["configurations"].items()):
        raise ValueError("recent config identity differs")
    keys = []
    for entry in model["entries"]:
        ctx = entry["context"]
        p, route, rows = ctx["problem"], ctx["route"], ctx["rows"]
        if route.endswith("dense") and rows:
            raise ValueError("recent dense context cannot contain expert rows")
        key = measured.query_key(route, p, rows if route.endswith("grouped") else None)
        c = configs.get(entry["config_id"])
        if (
            not c
            or not eligible(c, route, p)
            or c["mapping_id"] != common.mapping(p["qtype"])
        ):
            raise ValueError("recent tactic is not eligible for its measured context")
        if entry["key"] != key:
            raise ValueError("recent query identity differs")
        keys.append((tuple(key), tuple(rows)))
    if keys != sorted(set(keys)):
        raise ValueError("recent contexts are not sorted/unique")


class Selector:
    def __init__(self, model, base):
        validate(model, base)
        self.model, self.base = model, base
        self.configs = base["configurations"] | model["configurations"]
        self.entries, self.families = {}, defaultdict(list)
        for e in base["entries"]:
            key = e["key"]
            rows = base["row_vectors"][e["row_vector"]] if e["row_vector"] >= 0 else []
            value = dict(
                e,
                rows=rows,
                source=HISTORICAL,
                profile=(key[5], key[6], sum(r > 0 for r in rows)),
            )
            self.entries[(tuple(key), tuple(rows))] = value
        for e in model["entries"]:
            ctx = e["context"]
            value = dict(
                e,
                rows=ctx["rows"],
                source=RECENT,
                profile=profile(ctx["route"], ctx["problem"], ctx["rows"]),
            )
            self.entries[(tuple(e["key"]), tuple(ctx["rows"]))] = value
        for identity, e in sorted(self.entries.items()):
            self.families[identity[0][:5]].append(e)

    def select(
        self,
        route,
        problem,
        rows=None,
        *,
        device_name,
        compute_units,
        mapping_id,
        kernel_source,
        sdk_digest,
        allow_prediction=False,
    ):
        try:
            key = measured.query_key(route, problem, rows)
        except (ValueError, KeyError, TypeError):
            return dict(status="INVALID_QUERY")
        if problem["n"] % 256 or problem["k"] % (
            512 if problem["qtype"] in (11, 14) else 256
        ):
            return dict(status="INVALID_QUERY")
        b = self.model["required_binding"]
        if (
            device_name != b["device"]
            or compute_units != b["compute_units"]
            or mapping_id != common.mapping(problem["qtype"])
            or kernel_source != b["kernel"]
            or sdk_digest != b["sdk"]
        ):
            return dict(status="BINDING_MISMATCH")
        e = self.entries.get((tuple(key), tuple(rows or [])))
        source = e["source"] if e else None
        if e is None and allow_prediction:
            target = profile(route, problem, rows)
            options = [
                e
                for e in self.families.get(family(route, problem), ())
                if eligible(self.configs[e["config_id"]], route, problem)
            ]
            if options:
                e = min(
                    options,
                    key=lambda e: (
                        distance(target, e["profile"]),
                        e["source"] != RECENT,
                        e["config_id"],
                        e["key"],
                        e["rows"],
                    ),
                )
                source = PREDICTED
        if e is None or not eligible(self.configs[e["config_id"]], route, problem):
            return dict(status="FALLBACK_REQUIRED", reason="NO_ADMITTED_EXACT_TACTIC")
        c = self.configs[e["config_id"]]
        result = dict(
            status=source,
            config_id=e["config_id"],
            config=c,
            grid=launch_grid(c, problem, rows),
            model_digest=self.model["model_digest"],
            module_contract=self.model["authority"]["receipt"]["compiler"],
            request=dict(
                route=route,
                qtype=problem["qtype"],
                n=problem["n"],
                k=problem["k"],
                m=problem.get("m", problem.get("total_rows")),
                rows=list(rows or []),
            ),
            runtime_admission_required=True,
            performance_bound=False,
        )
        if source == RECENT:
            result["recent_regret_pct"] = e["regret_pct"]
            result["recent_spread_pct"] = e["spread_pct"]
        elif source == HISTORICAL:
            result["historical_status"] = e["status"]
        else:
            result["numerical_validation_required"] = True
            result["source_key"] = e["key"]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--route", choices=measured.ROUTES, required=True)
    parser.add_argument("--qtype", type=int, required=True)
    for name in ("m", "n", "k"):
        parser.add_argument("--" + name, type=int, required=name in ("n", "k"))
    parser.add_argument(
        "--rows-file", type=Path, help="one actual expert row count per line"
    )
    parser.add_argument("--allow-prediction", action="store_true")
    args = parser.parse_args()
    try:
        base = json.loads(args.base.read_text())
        model = json.loads(args.model.read_text())
        rows = (
            [int(x) for x in args.rows_file.read_text().splitlines()]
            if args.rows_file
            else None
        )
        p = dict(
            qtype=args.qtype,
            n=args.n,
            k=args.k,
            group_size=common.GROUP_SIZE[args.qtype],
        )
        if args.route.endswith("grouped"):
            if not rows or args.m is not None:
                raise ValueError("grouped requires rows-file, not M")
            p.update(experts=len(rows), total_rows=sum(rows), max_rows=max(rows))
        else:
            p["m"] = args.m
        b = model["required_binding"]
        result = Selector(model, base).select(
            args.route,
            p,
            rows,
            allow_prediction=args.allow_prediction,
            device_name=b["device"],
            compute_units=b["compute_units"],
            mapping_id=common.mapping(args.qtype),
            kernel_source=b["kernel"],
            sdk_digest=b["sdk"],
        )
        result["binding_scope"] = "HOST_REFERENCE_NOT_LOADED_MODULE_ADMISSION"
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

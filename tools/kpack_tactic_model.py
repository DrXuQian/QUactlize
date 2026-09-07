#!/usr/bin/env python3
"""Host-only ranked K-pack tactics: parent geometry and runtime policy separate.

The numeric models rank candidates, not predict an admitted execution time.
An exact measured cache hit is distinct from an unmeasured recommendation.
The caller's compiled inventory/can_implement remains the legality authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import kpack_policy as policy

SCHEMA = "quactlize.kpack-tactic-model.v1"
PARENT_FIELDS = (
    "symbol",
    "route",
    "qtype",
    "tm",
    "tn",
    "tk",
    "wm",
    "wn",
    "stages",
    "ap",
    "dn",
    "persistent",
)
PARENT_FEATURES = (
    "tm",
    "tn",
    "tk",
    "wm",
    "wn",
    "stages",
    "delivery",
    "ap",
    "warps",
    "m_padding",
    "n_padding",
    "m_tiles",
    "n_tiles",
    "k_steps",
    "pipeline_depth",
    "waves",
    "underfill",
    "shared",
    "occupancy",
    "a_traffic",
    "b_traffic",
    "metadata_traffic",
    "padded_flops",
    "short_k_stage",
    "long_k_stage",
    "tk_depth",
    "warp_work",
    "tm8",
    "tm16",
    "tm32",
    "tm64",
    "tm128",
    "tm256",
    "stage2",
    "stage3",
    "stage4",
    "stage6",
    "stage8",
    "stage12",
)
RUNTIME_FEATURES = (
    "persistent",
    "split",
    "log_split",
    "partial_traffic",
    "k_steps",
    "cta_launches",
    "underfill",
    "work_cu1",
    "work_cu2",
    "work_cu4",
    "work_occupancy",
    "serial_tiles",
    "last_grid_fill",
    "grid_capacity",
    "excess_grid",
    "split_output",
    "persistent_small_work",
)


def ceil_div(a, b):
    return (a + b - 1) // b


def context(route, problem, rows=None):
    policy.validate_problem(route, problem)
    if route.endswith("dense"):
        if rows is not None:
            raise ValueError("dense query must not supply expert rows")
        rows = [problem["m"]]
    elif (
        rows is None
        or len(rows) != problem["experts"]
        or any(type(x) is not int or x < 0 for x in rows)
        or sum(rows) != problem["total_rows"]
        or max(rows) != problem["max_rows"]
    ):
        raise ValueError("grouped ranking requires the actual, matching expert rows")
    return {"route": route, "problem": dict(problem), "rows": list(rows)}


def public_fold_key(ctx):
    # Keep every qtype/route/epoch/router alias of the original public point
    # together. A permutation is not a new independent validation sample.
    p = ctx["problem"]
    return (
        "dense" if "m" in p else "grouped",
        p["n"],
        p["k"],
        p.get("experts", 1),
        p.get("m", p.get("total_rows")),
        p.get("max_rows", 0),
    )


def cache_key(ctx):
    # Exact rows are a memoization key only; their hash/name is never a model
    # feature. Matching aggregate rows does not prove equal router performance.
    return policy.digest(ctx)


def anchor_family(ctx):
    return policy.digest(policy.family_key(ctx["route"], ctx["problem"]))


def anchor_features(ctx):
    return list(policy.features(ctx["route"], ctx["problem"]))


def nearby_parents(model, ctx):
    """At most two measured parent hints, never a copied runtime choice.

    Dense uses a lower/upper M bracket. Grouped uses distance in total/max rows;
    actual expert rows still own the CTA count/grid in runtime_choices().
    These hints are training data, not admission on an unmeasured query.
    """
    points = model.get("anchors", {}).get(anchor_family(ctx), [])
    feature = anchor_features(ctx)
    points = [
        x for x in points if parent_admissible(model["parents"][x["parent"]], ctx)
    ]
    distance = lambda x: (
        sum(abs(math.log2(a / b)) for a, b in zip(x["features"], feature)),
        x["features"],
        x["parent"],
    )
    if ctx["route"].endswith("dense"):
        lower = [x for x in points if x["features"][0] <= feature[0]]
        upper = [x for x in points if x["features"][0] >= feature[0]]
        chosen = [min(side, key=distance) for side in (lower, upper) if side]
    else:
        chosen = sorted(points, key=distance)[:2]
    return list(dict.fromkeys(x["parent"] for x in chosen))


def tile_count(parent, ctx):
    return sum(ceil_div(x, parent["tm"]) for x in ctx["rows"]) * ceil_div(
        ctx["problem"]["n"], parent["tn"]
    )


def parent_admissible(parent, ctx):
    p = ctx["problem"]
    return (
        parent["qtype"] == p["qtype"]
        and parent["route"] == ctx["route"]
        and p["k"] % parent["tk"] == 0
        and (not parent["ap"] or p.get("m") == 1)
        and not (ctx["route"] == "sf-dense" and parent["tm"] == 8 and p["m"] >= 8)
        and not (ctx["route"] == "fq-dense" and parent["tm"] == 8 and p["m"] > 64)
    )


def grid_choices(q, cu, occupancy):
    if q <= 0 or cu <= 0 or not 0 <= occupancy <= 63:
        raise ValueError("invalid tile count/device/occupancy")
    result = {}
    for b in range(1, occupancy + 1):
        for mode, grid in (
            ("capacity", min(q, cu * b)),
            ("balanced", ceil_div(q, ceil_div(q, cu * b))),
        ):
            item = result.setdefault(
                grid,
                dict(
                    grid=grid,
                    grid_mode=mode,
                    grid_b=b,
                    capacity_b_mask=0,
                    balanced_b_mask=0,
                ),
            )
            item[mode + "_b_mask"] |= 1 << b
    return [result[g] for g in sorted(result)]


def runtime_choices(parent, ctx, cu=72):
    if not parent_admissible(parent, ctx):
        return []
    p, route = ctx["problem"], ctx["route"]
    q = tile_count(parent, ctx)
    base = dict(
        symbol=parent["symbol"], split=1, grid=0, grid_mode="implicit", grid_b=0
    )
    if route == "fq-dense":
        return [
            dict(base, algorithm=f"TC_S{s}", split=s)
            for s in (1, 2, 4, 8)
            if (s == 1 or p["m"] < 64)
            and p["k"] % (parent["tk"] * s) == 0
            and (s == 1 or p["k"] // (parent["tk"] * s) >= parent["stages"] - 1)
        ]
    grouped = route.endswith("grouped")
    prefix = "GROUPED_" if grouped else ""
    result = []
    if parent["persistent"] != 1:
        result.append(
            dict(
                base,
                algorithm=prefix + "NONPERSISTENT",
                grid=0 if grouped else q,
                grid_mode="implicit" if grouped else "ordinary",
            )
        )
    if parent["persistent"] != 0:
        for g in grid_choices(q, cu, parent.get("occupancy", 0)):
            result.append(
                dict(
                    base,
                    algorithm=prefix + "PERSISTENT",
                    **{k: g[k] for k in ("grid", "grid_mode", "grid_b")},
                )
            )
    return result


def runtime_key(tactic):
    return tactic["symbol"], tactic["algorithm"], tactic["split"], tactic["grid"]


def geometry(parent, ctx):
    p, rows = ctx["problem"], ctx["rows"]
    total = sum(rows)
    mt = sum(ceil_div(x, parent["tm"]) for x in rows)
    nt = ceil_div(p["n"], parent["tn"])
    # Registry-owned plane widths are embedded by the evidence loader. Shared
    # estimates rank only when the raw record has no actual resource value.
    bits = parent["low_bits"] + parent["high_bits"]
    metadata_per_k = (
        parent["metadata_bytes_per_superblock"] / 256
        if ctx["route"].startswith("fq")
        else 4 / p["group_size"]
    )
    a_bytes = 2 * total * p["k"] * nt
    b_bytes = mt * p["n"] * p["k"] * bits / 8
    metadata_bytes = mt * p["n"] * p["k"] * metadata_per_k
    smem_est = parent["stages"] * (
        2 * parent["tm"] * parent["tk"]
        + parent["tn"] * parent["tk"] * (bits / 8 + 4 / p["group_size"])
    )
    return dict(
        q=mt * nt,
        total=total,
        mt=mt,
        nt=nt,
        a_bytes=a_bytes,
        b_bytes=b_bytes,
        metadata_bytes=metadata_bytes,
        smem=parent.get("shipping_smem", 0) or smem_est,
    )


def parent_features(parent, ctx, cu=72):
    p, g, c = ctx["problem"], geometry(parent, ctx), parent
    log = math.log2
    steps = p["k"] / c["tk"]
    values = [log(c[x]) for x in ("tm", "tn", "tk", "wm", "wn", "stages", "dn")]
    values += [
        float(c["ap"]),
        log((c["tm"] // c["wm"]) * (c["tn"] // c["wn"])),
        log(g["mt"] * c["tm"] / g["total"]),
        log(g["nt"] * c["tn"] / p["n"]),
        log(g["mt"]),
        log(g["nt"]),
        log(steps),
        log(steps / max(1, c["stages"] - 1)),
        log(max(1, g["q"] / cu)),
        log(max(1, cu / g["q"])),
        log(g["smem"] / 16384),
        log(max(1, c.get("occupancy", 0))),
        math.log2(1 + g["a_bytes"] / 1e6),
        math.log2(1 + g["b_bytes"] / 1e6),
        math.log2(1 + g["metadata_bytes"] / 1e6),
        math.log2(1 + 2 * g["q"] * c["tm"] * c["tn"] * p["k"] / 1e9),
        c["stages"] / max(1, steps),
        log(steps) * log(c["stages"]),
        log(steps) * log(c["tk"] / 64),
        log(c["wm"] * c["wn"] * c["tk"]),
    ]
    values += [float(c["tm"] == x) for x in (8, 16, 32, 64, 128, 256)]
    values += [float(c["stages"] == x) for x in (2, 3, 4, 6, 8, 12)]
    assert len(values) == len(PARENT_FEATURES)
    return values


def runtime_features(parent, ctx, tactic, cu=72):
    p = ctx["problem"]
    q = tile_count(parent, ctx)
    s = tactic["split"]
    persistent = tactic["algorithm"] in ("PERSISTENT", "GROUPED_PERSISTENT")
    grid = tactic["grid"] if persistent else q * s
    steps = p["k"] / parent["tk"] / s
    occupancy = max(1, parent.get("occupancy", 0))
    total = sum(ctx["rows"])
    # Three capacity proxies are calibrated rather than asserting a measured
    # FQ occupancy that its raw records do not provide.
    work = [max(1, q * s / (cu * b)) * steps for b in (1, 2, 4)]
    serial = ceil_div(q, grid) if persistent else 1
    return [
        float(persistent),
        float(s > 1),
        math.log2(s),
        math.log2(1 + (8 * total * p["n"] * s if s > 1 else 0) / 1e6),
        math.log2(steps),
        math.log2(grid),
        math.log2(max(1, cu / grid)),
        *[math.log2(x) for x in work],
        math.log2(max(1, grid / (cu * occupancy)) * serial * steps),
        math.log2(serial),
        math.log2(q / (grid * serial)) if persistent else 0,
        math.log2(max(1, grid / (cu * occupancy))),
        math.log2(max(1, grid / q)),
        float(s > 1) * math.log2(1 + total * p["n"] / 1e4),
        float(persistent) / max(1, q / cu),
    ]


def linear_score(weights, values):
    if len(weights) != len(values) or any(
        not math.isfinite(float(x)) for x in (*weights, *values)
    ):
        raise ValueError("invalid calibrated model")
    return sum(a * b for a, b in zip(weights, values))


def validate_model(model):
    """Validate serialized host output once, not on every score evaluation."""
    if model.get("model_digest") != policy.digest(
        {k: v for k, v in model.items() if k != "model_digest"}
    ):
        raise ValueError("model digest differs")
    if (
        "host_scorer_sha256" in model
        and model["host_scorer_sha256"]
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("host scoring source differs")
    if model.get("feature_names") != {
        "parent": list(PARENT_FEATURES),
        "runtime": list(RUNTIME_FEATURES),
    }:
        raise ValueError("feature schema differs")
    for field, names in (
        ("parent_weights", PARENT_FEATURES),
        ("runtime_weights", RUNTIME_FEATURES),
    ):
        for weights in model[field].values():
            linear_score(weights, [0.0] * len(names))
    for symbol, parent in model["parents"].items():
        if parent["symbol"] != symbol or parent["route"] not in policy.ROUTES:
            raise ValueError("invalid parent identity")
        if any(
            type(parent[k]) is not int or parent[k] <= 0
            for k in ("tm", "tn", "tk", "wm", "wn", "stages", "dn")
        ):
            raise ValueError("invalid parent geometry")
        if parent["persistent"] not in (-1, 0, 1) or parent["ap"] not in (0, 1):
            raise ValueError("invalid parent provider/schedule")
        if not 0 <= parent["occupancy"] <= 63 or parent["shipping_smem"] < 0:
            raise ValueError("invalid parent resources")


def recommend(
    model,
    ctx,
    count=5,
    *,
    use_cache=True,
    device_name="PPU-ZW810",
    compute_units=72,
    mapping_id=None,
):
    if model.get("schema") != SCHEMA or model.get("device") != {
        "name": "PPU-ZW810",
        "compute_units": 72,
    }:
        raise ValueError("unsupported model/device")
    if {"name": device_name, "compute_units": compute_units} != model["device"]:
        raise ValueError("query device differs from calibrated device")
    checked = context(
        ctx["route"],
        ctx["problem"],
        ctx["rows"] if ctx["route"].endswith("grouped") else None,
    )
    if checked != ctx:
        raise ValueError("query context differs")
    if not 1 <= count <= 32:
        raise ValueError("candidate count must be in [1,32]")
    route, qtype = ctx["route"], ctx["problem"]["qtype"]
    if mapping_id is not None and mapping_id != policy.mapping(qtype):
        raise ValueError("query is not the canonical K-pack arrangement")
    if use_cache and cache_key(ctx) in model.get("cache", {}):
        entry = model["cache"][cache_key(ctx)]
        parent = model["parents"][entry["tactic"]["symbol"]]
        if entry["mapping_id"] != policy.mapping(qtype) or entry[
            "tactic"
        ] not in runtime_choices(parent, ctx):
            raise ValueError("cached tactic is not admitted by the query recipes")
        return dict(
            status="MEASURED_TACTIC",
            choice=dict(entry, parent=parent),
            required_binding=model.get("required_binding"),
            runtime_validation_required=True,
            performance_scope="RECORDED_CONTEXTS_AND_TIMING_EPOCHS_ONLY",
            performance_admitted=False,
            production_policy_updated=False,
        )
    key = f"{qtype}/{route}"
    if key not in model["parent_weights"] or route not in model["runtime_weights"]:
        return dict(
            status="NO_CALIBRATED_MODEL",
            fallback="CALLER_ADMITTED_KPACK_TACTIC_REQUIRED",
        )
    parents = [
        (linear_score(model["parent_weights"][key], parent_features(c, ctx)), c)
        for c in model["parents"].values()
        if parent_admissible(c, ctx)
    ]
    hints = nearby_parents(model, ctx)
    order = {s: i for i, s in enumerate(hints)}
    ranked = []
    for score, c in sorted(
        parents,
        key=lambda x: (order.get(x[1]["symbol"], len(hints)), x[0], x[1]["symbol"]),
    ):
        options = runtime_choices(c, ctx)
        if not options:
            continue
        runtimes = sorted(
            options,
            key=lambda t: (
                linear_score(
                    model["runtime_weights"][route], runtime_features(c, ctx, t)
                ),
                runtime_key(t),
            ),
        )
        ranked.append(
            dict(
                parent=c["symbol"],
                parent_score=score,
                origin=(
                    "MEASURED_FAMILY_HINT"
                    if c["symbol"] in order
                    else "CALIBRATED_GEOMETRY"
                ),
                tactics=runtimes[:3],
            )
        )
        if len(ranked) == count:
            break
    return dict(
        status="UNMEASURED_SHORTLIST",
        candidates=ranked,
        fallback="CALLER_ADMITTED_KPACK_TACTIC_REQUIRED",
        runtime_validation_required=True,
        performance_admitted=False,
        production_policy_updated=False,
        required_binding=model.get("required_binding"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--route", choices=policy.ROUTES, required=True)
    parser.add_argument("--qtype", type=int, required=True)
    for axis in ("m", "n", "k", "experts", "total-rows", "max-rows"):
        parser.add_argument("--" + axis, type=int)
    parser.add_argument("--rows-file", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device-name", default="PPU-ZW810")
    parser.add_argument("--compute-units", type=int, default=72)
    parser.add_argument("--mapping-id")
    args = parser.parse_args()
    p = dict(
        qtype=args.qtype,
        n=args.n,
        k=args.k,
        group_size=policy.GROUP_SIZE.get(args.qtype),
    )
    p.update({x: getattr(args, x) for x in policy.axes(args.route)})
    if args.route.endswith("grouped"):
        p["experts"] = args.experts
    try:
        model = json.loads(args.model.read_text())
        validate_model(model)
        ctx = context(
            args.route,
            p,
            (
                [int(x) for x in args.rows_file.read_text().splitlines()]
                if args.rows_file
                else None
            ),
        )
        print(
            json.dumps(
                recommend(
                    model,
                    ctx,
                    args.top,
                    use_cache=not args.no_cache,
                    device_name=args.device_name,
                    compute_units=args.compute_units,
                    mapping_id=args.mapping_id,
                ),
                indent=2,
                allow_nan=False,
            )
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

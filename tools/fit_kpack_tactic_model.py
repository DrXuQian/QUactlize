#!/usr/bin/env python3
"""Calibrate a small parent/runtime ranker from verified sparse K-pack costs.

Grouped five-fold validation holds all formats, routes, router aliases and
timing epochs of a public point together. Missing costs are unknown, never
imputed passes. This produces a host experiment, not production admission.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import kpack_policy as policy
import kpack_tactic_evidence as evidence
import kpack_tactic_model as tactic

STABLE = "CONFIRMED_SELECTED_SET"
FOLDS = 5
RIDGE = 0.01


def fold(ctx):
    return int(policy.digest(tactic.public_fold_key(ctx)), 16) % FOLDS


def good(cost, threshold=5.0):
    return cost["regret_pct"] <= threshold and cost["spread_pct"] <= threshold


def training_groups(data, observations):
    """Center only within the same timing epoch/query or epoch/query/parent."""
    parent_groups, runtime_groups = defaultdict(list), defaultdict(list)
    for o in observations:
        if o["status"] != STABLE:
            continue
        ctx = o["context"]
        groups = defaultdict(list)
        for c in o["cells"]:
            if c["cost"]["spread_pct"] <= 5:
                groups[c["tactic"]["symbol"]].append(c)
        px, py = [], []
        for symbol, cells in sorted(groups.items()):
            parent = data["parents"][symbol]
            px.append(tactic.parent_features(parent, ctx))
            py.append(math.log(min(c["cost"]["median_us"] for c in cells)))
            if len(cells) > 1:
                runtime_groups[ctx["route"]].append(
                    (
                        [
                            tactic.runtime_features(parent, ctx, c["tactic"])
                            for c in cells
                        ],
                        [math.log(c["cost"]["median_us"]) for c in cells],
                        1 / max(1, len(groups)),
                    )
                )
        if len(px) > 1:
            parent_groups[f"{ctx['problem']['qtype']}/{ctx['route']}"].append(
                (px, py, 1)
            )
    return parent_groups, runtime_groups


def ridge(groups, dimensions, alpha=RIDGE):
    """Equal query weight; standardize centered features on training data only.

    Centering a query's log times removes its clock/fixture scale. The equivalent
    all-pairs differences avoid choosing a special reference parent or grid.
    """
    gram, rhs = np.zeros((dimensions, dimensions)), np.zeros(dimensions)
    total_weight = 0.0
    for features, targets, weight in groups:
        x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
        if (
            x.shape != (len(y), dimensions)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("invalid training features/targets")
        if len(y) < 2 or not math.isfinite(weight) or weight <= 0:
            raise ValueError("training groups require positive weight and pairs")
        x, y = x - x.mean(axis=0), y - y.mean()
        gram += weight / len(y) * x.T @ x
        rhs += weight / len(y) * x.T @ y
        total_weight += weight
    if total_weight == 0:
        return None
    gram, rhs = gram / total_weight, rhs / total_weight
    scale = np.sqrt(np.maximum(np.diag(gram), 1e-12))
    normalized = gram / scale[:, None] / scale[None, :]
    weights = (
        np.linalg.solve(normalized + alpha * np.eye(dimensions), rhs / scale) / scale
    )
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite fitted coefficient")
    return weights.tolist()


def fit(data, observations):
    parents, runtimes = training_groups(data, observations)
    return dict(
        schema=tactic.SCHEMA,
        device=dict(name="PPU-ZW810", compute_units=72),
        feature_names=dict(
            parent=list(tactic.PARENT_FEATURES), runtime=list(tactic.RUNTIME_FEATURES)
        ),
        parent_weights={
            key: ridge(g, len(tactic.PARENT_FEATURES))
            for key, g in sorted(parents.items())
        },
        runtime_weights={
            key: ridge(g, len(tactic.RUNTIME_FEATURES))
            for key, g in sorted(runtimes.items())
        },
        parents=data["parents"],
        cache={},
        anchors=measured_anchors(observations),
    )


def measured_anchors(observations):
    """Conservative parent hints from training points, not held-out labels.

    Public router aliases may disagree on runtime. A hint requires a common
    good parent, but its runtime is always resolved anew on the destination.
    """
    groups = defaultdict(list)
    for o in observations:
        ctx = o["context"]
        groups[tactic.anchor_family(ctx), tuple(tactic.anchor_features(ctx))].append(o)
    result = defaultdict(list)
    for (family, feature), rows in sorted(groups.items()):
        if any(o["status"] != STABLE for o in rows):
            continue
        costs = []
        for o in rows:
            pc = {}
            for cell in o["cells"]:
                if good(cell["cost"]):
                    s = cell["tactic"]["symbol"]
                    pc[s] = min(pc.get(s, float("inf")), cell["cost"]["regret_pct"])
            costs.append(pc)
        common = set.intersection(*(set(c) for c in costs))
        if common:
            selected = min(common, key=lambda s: (max(c[s] for c in costs), s))
            result[family].append(dict(features=list(feature), parent=selected))
    return dict(result)


def cost_verdict(cells, threshold=5):
    if not cells:
        return "UNKNOWN"
    return (
        "WITHIN_5PCT"
        if any(good(c["cost"], threshold) for c in cells)
        else "KNOWN_OUTSIDE_5PCT"
    )


def replay(model, observation):
    ctx = observation["context"]
    result = tactic.recommend(model, ctx, 5, use_cache=False)
    cells = observation["cells"]
    by_runtime = {tactic.runtime_key(c["tactic"]): c for c in cells}
    by_parent = defaultdict(list)
    for c in cells:
        by_parent[c["tactic"]["symbol"]].append(c)
    row = dict(
        id=observation["id"],
        epoch=observation["epoch"],
        route=ctx["route"],
        qtype=ctx["problem"]["qtype"],
        status=observation["status"],
        fold=fold(ctx),
        model_status=result["status"],
    )
    candidates = result.get("candidates", [])
    row["candidates"] = candidates
    row["measured_parents"] = len(by_parent)
    # Conditional replay tests the score only on already timed parents. It is
    # NOT evidence that the actual global-pool recommendation was measured.
    key = f"{ctx['problem']['qtype']}/{ctx['route']}"
    conditional = (
        sorted(
            by_parent,
            key=lambda s: (
                tactic.linear_score(
                    model["parent_weights"].get(key, [0] * len(tactic.PARENT_FEATURES)),
                    tactic.parent_features(model["parents"][s], ctx),
                ),
                s,
            ),
        )
        if key in model["parent_weights"]
        else []
    )
    for count in (1, 3, 5):
        chosen = candidates[:count]
        pcells = [c for x in chosen for c in by_parent.get(x["parent"], [])]
        tcells = [
            by_runtime[tactic.runtime_key(t)]
            for x in chosen
            for t in x["tactics"]
            if tactic.runtime_key(t) in by_runtime
        ]
        # A partial miss may conceal a fast candidate: never label it a proven
        # loss just because a different, measured candidate is slow.
        for name, selected, expected, available in (
            (
                "pool_parent",
                pcells,
                len(chosen),
                sum(x["parent"] in by_parent for x in chosen),
            ),
            (
                "pool_tactic",
                tcells,
                sum(len(x["tactics"]) for x in chosen),
                len(tcells),
            ),
        ):
            verdict = cost_verdict(selected)
            if verdict != "WITHIN_5PCT" and (not expected or available < expected):
                verdict = "UNKNOWN"
            row[f"{name}_top{count}"] = verdict
        row[f"conditional_parent_top{count}"] = cost_verdict(
            [c for s in conditional[:count] for c in by_parent[s]]
        )
    top = candidates[0]["tactics"][0] if candidates else None
    actual = by_runtime.get(tactic.runtime_key(top)) if top else None
    row["top1"] = cost_verdict([actual] if actual else [])
    row["top1_regret_pct"] = actual["cost"]["regret_pct"] if actual else None
    row["top1_spread_pct"] = actual["cost"]["spread_pct"] if actual else None
    return row


def census(rows):
    stable = [r for r in rows if r["status"] == STABLE]
    columns = [
        f"{mode}_top{k}"
        for mode in ("conditional_parent", "pool_parent", "pool_tactic")
        for k in (1, 3, 5)
    ] + ["top1"]
    return dict(
        total=len(rows),
        stable=len(stable),
        noisy_or_unconfirmed=len(rows) - len(stable),
        measured_parent_count=dict(Counter(r["measured_parents"] for r in stable)),
        conditional_top5_trivial=len([r for r in stable if r["measured_parents"] <= 5]),
        decisions={k: dict(Counter(r[k] for r in stable)) for k in columns},
        known_top1_max_regret_pct=max(
            (r["top1_regret_pct"] for r in stable if r["top1_regret_pct"] is not None),
            default=None,
        ),
    )


def measured_cache(data):
    """One exact context must have one good runtime in EVERY supplied epoch.

    Cache entries retain complete runtime identity and measurement receipts.
    A stale/noisy alias is not erased by a later good observation.
    """
    groups = defaultdict(list)
    for o in data["observations"]:
        groups[tactic.cache_key(o["context"])].append(o)
    result, blocked = {}, Counter()
    for key, observations in sorted(groups.items()):
        if any(o["status"] != STABLE for o in observations):
            blocked["NOISY_OBSERVATION"] += 1
            continue
        maps = [
            {tactic.runtime_key(c["tactic"]): c for c in o["cells"] if good(c["cost"])}
            for o in observations
        ]
        common = set.intersection(*(set(m) for m in maps))
        if not common:
            blocked["NO_COMMON_MEASURED_RUNTIME_WITHIN_5PCT"] += 1
            continue
        choice = min(
            common, key=lambda k: (max(m[k]["cost"]["regret_pct"] for m in maps), k)
        )
        parent = data["parents"][choice[0]]
        result[key] = dict(
            tactic=maps[0][choice]["tactic"],
            mapping_id=policy.mapping(parent["qtype"]),
            observations=[o["id"] for o in observations],
            max_measured_regret_pct=max(m[choice]["cost"]["regret_pct"] for m in maps),
            max_measured_spread_pct=max(m[choice]["cost"]["spread_pct"] for m in maps),
        )
    return result, dict(blocked)


def calibrate(data):
    rows, geometry_only = [], []
    observations = data["observations"]
    for held in range(FOLDS):
        train = [o for o in observations if fold(o["context"]) != held]
        test = [o for o in observations if fold(o["context"]) == held]
        model = fit(data, train)
        rows.extend(replay(model, o) for o in test)
        anchors = model.pop("anchors")
        geometry_only.extend(replay(model, o) for o in test)
        model["anchors"] = anchors
        print(
            f"KPACK_TACTIC_FOLD fold={held} train={len(train)} held_out={len(test)}",
            flush=True,
        )
    model = fit(data, observations)
    model["cache"], blocked = measured_cache(data)
    model["authority"] = data["authority"]
    identities = [a["campaign_identity"] for a in data["authority"].values()]
    bindings = [
        {"kernel_source": i["kernel_source"], "sdk_digest": policy.digest(i["sdk"])}
        for i in identities
    ]
    if not bindings or any(b != bindings[0] for b in bindings):
        raise ValueError("cannot mix different kernel/SDK calibration epochs")
    model["required_binding"] = dict(
        bindings[0],
        **model["device"],
        mappings={str(q): policy.mapping(q) for q in policy.GROUP_SIZE},
    )
    model["calibration"] = dict(
        folds=FOLDS,
        ridge=RIDGE,
        target="WITHIN_QUERY_LOG_FULL_OUTPUT_TIME",
        holdout="PUBLIC_POINT_ALL_FORMATS_ROUTES_ALIASES_EPOCHS",
        production_admitted=False,
        evaluation_scope="INTERNAL_GROUPED_CV_USED_FOR_DESIGN; FRESH_CHALLENGE_REQUIRED",
    )
    report = dict(
        schema="quactlize.kpack-tactic-calibration.v1",
        authority=data["authority"],
        calibration=model["calibration"],
        overall=census(rows),
        geometry_only_ablation=census(geometry_only),
        by_route={
            route: census([r for r in rows if r["route"] == route])
            for route in policy.ROUTES
        },
        by_epoch={
            epoch: census([r for r in rows if r["epoch"] == epoch])
            for epoch in sorted(data["authority"])
        },
        cache=dict(entries=len(model["cache"]), blocked=blocked),
        model=dict(
            parent_scoring_blocks=len(model["parent_weights"]),
            runtime_scoring_blocks=len(model["runtime_weights"]),
            coefficients=sum(map(len, model["parent_weights"].values()))
            + sum(map(len, model["runtime_weights"].values())),
            compiled_parent_pool=len(model["parents"]),
            branch_tree_rules=0,
            measured_family_hints=sum(map(len, model["anchors"].values())),
        ),
        scope="SPARSE_SELECTED_SET_NOT_GLOBAL_OPTIMUM; UNKNOWN_COST_IS_NOT_A_PASS",
    )
    return model, report, sorted(rows, key=lambda r: r["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--publish-prefix",
        type=Path,
        help="also emit new compact JSON/model report under this prefix",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    published = (
        []
        if args.publish_prefix is None
        else [
            Path(str(args.publish_prefix) + suffix)
            for suffix in (".json", ".report.json")
        ]
    )
    if any(p.exists() or p.is_symlink() for p in published):
        parser.error("published output already exists")
    data = evidence.combine(*(json.loads(p.read_text()) for p in args.evidence))
    model, report, replay_rows = calibrate(data)
    report["input_evidence_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.evidence
    }
    model["evidence_digest"] = policy.digest(data)
    model["host_scorer_sha256"] = hashlib.sha256(
        Path(tactic.__file__).read_bytes()
    ).hexdigest()
    model["model_digest"] = policy.digest(model)
    report["model_digest"] = model["model_digest"]
    args.output.mkdir(parents=True)
    for name, value in (
        ("model.json", model),
        ("report.json", report),
        ("replay.json", replay_rows),
    ):
        (args.output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    for path, value in zip(published, (model, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(
                value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            stream.write("\n")
    print(
        "KPACK_TACTIC_CALIBRATION " + json.dumps(report["overall"], sort_keys=True),
        flush=True,
    )
    print(f"KPACK_TACTIC_CALIBRATION output={args.output} production_updated=0")


if __name__ == "__main__":
    main()

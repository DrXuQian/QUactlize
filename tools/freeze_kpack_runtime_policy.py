#!/usr/bin/env python3
"""Freeze measured tactics without another rule tree or performance model.

All recorded timing epochs remain evidence. Minimize the parent cover within
the unchanged cross-epoch 5% bounds. Where no such tactic exists, retain the
newest epoch's measured choice with explicit historical/missing-cost reasons.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import kpack_policy as shared
import kpack_runtime_policy as runtime
import kpack_tactic_evidence as evidence
import kpack_tactic_model as recipes


def rank(key, maps):
    costs = [m[key]["cost"] for m in maps if key in m]
    regret = max(c["regret_pct"] for c in costs)
    spread = max(c["spread_pct"] for c in costs)
    return max(regret, spread), regret, spread, key


def parent_cover(options):
    supports = defaultdict(set)
    for point, choices in options.items():
        for key in choices:
            supports[key[0]].add(point)
    uncovered, selected = set(options), set()
    while uncovered:
        parent = min(supports, key=lambda s: (-len(supports[s] & uncovered), s))
        coverage = supports[parent] & uncovered
        if not coverage:
            raise ValueError("uncovered measured query has no parent")
        selected.add(parent)
        uncovered -= coverage
    return selected


def freeze(data):
    groups = defaultdict(list)
    for o in data["observations"]:
        if o["status"] not in ("CONFIRMED_SELECTED_SET", "NOISY_CONFIRMATION"):
            raise ValueError("only raw-verified numerical results may enter policy")
        if not o["cells"] or len(
            {recipes.runtime_key(c["tactic"]) for c in o["cells"]}
        ) != len(o["cells"]):
            raise ValueError("empty or duplicated measured cells")
        if any(
            not math.isfinite(c["cost"][k]) or c["cost"][k] < 0
            for c in o["cells"]
            for k in ("regret_pct", "spread_pct")
        ):
            raise ValueError("invalid measured cost")
        groups[recipes.cache_key(o["context"])].append(o)
    if not groups:
        raise ValueError("empty evidence")
    maps, acceptable = {}, {}
    for key, observations in groups.items():
        maps[key] = [
            {recipes.runtime_key(c["tactic"]): c for c in o["cells"]}
            for o in observations
        ]
        sets = [
            {
                t
                for t, c in m.items()
                if c["cost"]["regret_pct"] <= 5 and c["cost"]["spread_pct"] <= 5
            }
            for m in maps[key]
        ]
        choices = set.intersection(*sets)
        if choices:
            acceptable[key] = choices
    parents = parent_cover(acceptable)
    rows, row_index, configs, entries = [], {}, {}, []
    for key, observations in groups.items():
        ms = maps[key]
        reasons = []
        if key in acceptable:
            choices = {t for t in acceptable[key] if t[0] in parents}
            selected = min(choices, key=lambda t: rank(t, ms))
            status = runtime.WITHIN
        else:
            # Missing old costs do not justify retaining a known slow shared
            # incumbent over a freshly measured faster tactic. Nor do they
            # become a cross-epoch performance pass: this stays an exception.
            latest_good = {
                t
                for t, c in ms[-1].items()
                if c["cost"]["regret_pct"] <= 5 and c["cost"]["spread_pct"] <= 5
            }
            choices = latest_good or set(ms[-1])
            if not choices:
                raise ValueError("no numerically measured exception exists")
            selected = min(
                choices,
                key=lambda t: (
                    ms[-1][t]["cost"]["regret_pct"],
                    ms[-1][t]["cost"]["spread_pct"],
                    t,
                ),
            )
            if any(selected not in m for m in ms):
                reasons.append("MISSING_CROSS_EPOCH_COSTS")
            status = runtime.EXCEPTION
        _, regret, spread, _ = rank(selected, ms)
        if regret > 5:
            reasons.append("REGRET_ABOVE_5PCT")
        if spread > 5:
            reasons.append("ROUND_SPREAD_ABOVE_5PCT")
        source_cell = next(m[selected] for m in reversed(ms) if selected in m)
        tactic = source_cell["tactic"]
        parent = data["parents"][selected[0]]
        config = dict(
            parent,
            algorithm=tactic["algorithm"],
            split=tactic["split"],
            grid_mode=tactic["grid_mode"],
            grid_b=tactic["grid_b"],
            mapping_id=shared.mapping(parent["qtype"]),
        )
        cid = shared.digest(config)
        configs[cid] = config
        ctx = observations[0]["context"]
        vector = -1
        if ctx["route"].endswith("grouped"):
            row_key = tuple(ctx["rows"])
            if row_key not in row_index:
                row_index[row_key] = len(rows)
                rows.append(list(row_key))
            vector = row_index[row_key]
        entries.append(
            dict(
                key=runtime.query_key(
                    ctx["route"], ctx["problem"], ctx["rows"] if vector >= 0 else None
                ),
                row_vector=vector,
                config_id=cid,
                status=status,
                reasons=reasons,
                max_measured_regret_pct=regret,
                max_measured_spread_pct=spread,
                latest_epoch=observations[-1]["epoch"],
                latest_regret_pct=ms[-1][selected]["cost"]["regret_pct"],
                latest_spread_pct=ms[-1][selected]["cost"]["spread_pct"],
                observation_ids=[o["id"] for o in observations],
                measured_epochs=[
                    o["epoch"] for o, m in zip(observations, ms) if selected in m
                ],
            )
        )
    identities = [a["campaign_identity"] for a in data["authority"].values()]
    binding = dict(
        kernel_source=identities[0]["kernel_source"],
        sdk_digest=shared.digest(identities[0]["sdk"]),
        name="PPU-ZW810",
        compute_units=72,
        mappings={str(q): shared.mapping(q) for q in shared.GROUP_SIZE},
    )
    if any(
        i["kernel_source"] != binding["kernel_source"]
        or shared.digest(i["sdk"]) != binding["sdk_digest"]
        for i in identities
    ):
        raise ValueError("kernel/SDK changed across evidence epochs")
    model = dict(
        schema=runtime.SCHEMA,
        required_binding=binding,
        configurations=configs,
        entries=sorted(entries, key=lambda e: (e["key"], e["row_vector"])),
        row_vectors=rows,
        authority=data["authority"],
        evidence_digest=shared.digest(data),
        scope="RECORDED_FP16_FULL_OUTPUT_CONTEXTS_ONLY; UNKNOWN_REQUIRES_ADMITTED_KPACK_FALLBACK",
        production_library_updated=False,
    )
    model["policy_digest"] = shared.digest(model)
    report = dict(
        schema="quactlize.kpack-runtime-freeze.v1",
        policy_digest=model["policy_digest"],
        contexts=len(groups),
        observations=len(data["observations"]),
        status=dict(Counter(e["status"] for e in entries)),
        reasons=dict(Counter(r for e in entries for r in e["reasons"])),
        parents=len({c["symbol"] for c in configs.values()}),
        runtime_recipes=len(configs),
        row_vectors=len(rows),
        handwritten_shape_branches=0,
        runtime_model_coefficients=0,
        bounded_parent_cover=len(parents),
        within5_max_regret_pct=max(
            (
                e["max_measured_regret_pct"]
                for e in entries
                if e["status"] == runtime.WITHIN
            ),
            default=None,
        ),
        exception_max_measured_regret_pct=max(
            (
                e["max_measured_regret_pct"]
                for e in entries
                if e["status"] == runtime.EXCEPTION
            ),
            default=0,
        ),
        exception_max_measured_spread_pct=max(
            (
                e["max_measured_spread_pct"]
                for e in entries
                if e["status"] == runtime.EXCEPTION
            ),
            default=0,
        ),
        latest_epoch_within_both_5pct=sum(
            e["latest_regret_pct"] <= 5 and e["latest_spread_pct"] <= 5 for e in entries
        ),
        latest_epoch_max_regret_pct=max(e["latest_regret_pct"] for e in entries),
    )
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, action="append", required=True, help="oldest to newest"
    )
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        Path(str(args.prefix) + suffix) for suffix in (".json", ".report.json", ".hpp")
    ]
    if any(p.exists() or p.is_symlink() for p in paths):
        parser.error("output exists")
    data = evidence.combine(*(json.loads(p.read_text()) for p in args.evidence))
    model, report = freeze(data)
    from generate_kpack_runtime_header import generate

    outputs = [
        json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for x in (model, report)
    ]
    outputs.append(generate(model))
    for path, text in zip(paths, outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(text)
    print("KPACK_RUNTIME_FREEZE " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

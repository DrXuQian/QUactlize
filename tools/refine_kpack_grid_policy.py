#!/usr/bin/env python3
"""Replace fixed dense persistent grids with source-owned, measured recipes.

No kernel code changes. Every recipe witness comes from raw occupancy and
capacity/balanced masks, and is checked against the actual emitted grid.
Grouped grids stay fixed: public row summaries do not determine tile count.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import fit_kpack_tuner_policy as fit
import kpack_policy as policy
import kpack_overnight_search as search


def witnesses(config, problem, metadata):
    occupancy, cap, balanced = metadata
    if not 1 <= occupancy <= 63 or cap < 0 or balanced < 0:
        raise ValueError("invalid grid occupancy/mask")
    recipes, expected = [], [0, 0]
    for b in range(1, occupancy + 1):
        for index, mode in enumerate(("capacity", "balanced")):
            candidate = dict(
                config, grid_mode=mode, grid_b=b, occupancy=occupancy, grid=0
            )
            if policy.resolve_grid(candidate, problem) == config["grid"]:
                expected[index] |= 1 << b
                recipes.append(candidate)
    if expected != [cap, balanced] or not recipes:
        raise ValueError("raw grid masks disagree with the source-owned recipes")
    return recipes


def load_metadata(campaign, observations, configurations):
    metadata = {}
    occupancy_by_parent = {}
    for phase in ("confirm-1", "confirm-2", "confirm-3"):
        count = 0
        for o in observations:
            if o["route"] != "sf-dense":
                continue
            record = campaign.read(f"phases/{phase}/run/results/{o['id']}.json")
            observed = {}
            for receipt in record["logs"]:
                prefix = f"phases/{phase}/run/logs/"
                relative = prefix + receipt["path"].split(prefix, 1)[1]
                raw = (campaign.root / relative).read_bytes()
                if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                    raise ValueError("grid metadata log changed after numeric replay")
                for line in raw.decode().splitlines():
                    if not line.startswith("SF_CELL "):
                        continue
                    c = json.loads(line[len("SF_CELL ") :])
                    if c["status"] != "MEASURED" or c["algorithm"] != "PERSISTENT":
                        continue
                    key = search.runtime_key(c)
                    values = (
                        int(c["occupancy"]),
                        int(c["capacity_b_mask"], 16),
                        int(c["balanced_b_mask"], 16),
                    )
                    if key in observed and observed[key] != values:
                        raise ValueError("grid metadata differs between samples")
                    observed[key] = values
            for cid in o["costs"]:
                config = configurations[cid]
                if config["algorithm"] != "PERSISTENT":
                    continue
                key = search.runtime_key(config)
                if key not in observed:
                    raise ValueError(
                        "confirmed persistent grid has no raw recipe witness"
                    )
                values = observed[key]
                witnesses(config, o["problem"], values)
                pair = o["id"], cid
                if pair in metadata and metadata[pair] != values:
                    raise ValueError(
                        "grid metadata changed between confirmation rounds"
                    )
                if (
                    config["symbol"] in occupancy_by_parent
                    and occupancy_by_parent[config["symbol"]] != values[0]
                ):
                    raise ValueError("same parent has inconsistent measured occupancy")
                metadata[pair] = values
                occupancy_by_parent[config["symbol"]] = values[0]
                count += 1
        print(
            f"KPACK_GRID_METADATA phase={phase} variants={count} status=PASS",
            flush=True,
        )
    return metadata


def expand(observations, configurations, metadata):
    rows, configs = [], {}
    for o in observations:
        row = {**o, "costs": {}}
        for cid, cost in o["costs"].items():
            config = configurations[cid]
            candidates = [config]
            if config["route"] == "sf-dense" and config["algorithm"] == "PERSISTENT":
                candidates = witnesses(config, o["problem"], metadata[o["id"], cid])
            for candidate in candidates:
                key = policy.digest(candidate)
                if key in row["costs"]:
                    raise ValueError("recipe collapsed distinct runtime measurements")
                configs[key] = candidate
                row["costs"][key] = cost
        rows.append(row)
    return rows, configs


def comparison(before, after):
    key = lambda row: tuple(row["ids"])
    old = {key(r): r for r in before}
    if set(old) != {key(r) for r in after}:
        raise ValueError("holdout public-point denominator changed")
    result = []
    for r in after:
        b = old[key(r)]
        result.append(
            {
                "route": r["route"],
                "problem": r["problem"],
                "ids": r["ids"],
                "before": b["status"],
                "after": r["status"],
                "before_regret_pct": b["regret_pct"],
                "after_regret_pct": r["regret_pct"],
                "config_id": r["config_id"],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("output exists; use a fresh directory")
    try:
        campaign = fit.Campaign(args.source)
        observations, configurations, authority = campaign.load()
        print("KPACK_GRID_FIT phase=fixed-grid-baseline", flush=True)
        before = fit.make_policy(observations, configurations, authority)
        metadata = load_metadata(campaign, observations, configurations)
        expanded, configs = expand(observations, configurations, metadata)
        recipe_authority = {
            **authority,
            "grid_policy_source_sha256": hashlib.sha256(
                (
                    fit.ROOT / "quactlize/include/scalefirst_persistent_policy.hpp"
                ).read_bytes()
            ).hexdigest(),
        }
        print("KPACK_GRID_FIT phase=grid-recipes", flush=True)
        after = fit.make_policy(expanded, configs, recipe_authority)
        model, report, replay, holdout, followup = after
        changes = comparison(before[3], holdout)
        print(
            "KPACK_GRID_COMPARISON "
            + json.dumps(
                {
                    "before": before[1],
                    "after": report,
                    "transitions": dict(
                        Counter(r["before"] + "->" + r["after"] for r in changes)
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        args.out.mkdir(parents=True)
        outputs = {
            "policy.json": model,
            "report.json": report,
            "replay.json": replay,
            "holdout.json": holdout,
            "followup.json": followup,
            "baseline-report.json": before[1],
            "baseline-holdout.json": before[3],
            "comparison.json": changes,
            # All costs retained for local model diagnostics, not a runtime asset.
            "measured-costs.json": {
                "authority": authority,
                "observations": observations,
                "configurations": configurations,
            },
            "grid-metadata.json": {
                rid + "/" + cid: values for (rid, cid), values in metadata.items()
            },
        }
        for name, data in outputs.items():
            with (args.out / name).open("x") as stream:
                json.dump(
                    data, stream, sort_keys=True, allow_nan=False, separators=(",", ":")
                )
                stream.write("\n")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(
        f"KPACK_GRID_OUTPUT path={args.out} kernel_code_changed=0 production_policy_updated=0",
        flush=True,
    )


if __name__ == "__main__":
    main()

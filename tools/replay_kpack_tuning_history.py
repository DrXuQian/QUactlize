#!/usr/bin/env python3
"""Check a budgeted plan against an older measured K-pack shortlist.

This replays CTA/warp/TK/stage geometry coverage only. Old timing samples do
not enter the new tuner or prove identical provider/grid/kernel performance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

from kpack_tuning_plan import SCHEMA, digest
from fully_quantized_kpack_discovery_matrix import FQ_TILE_K


def load_summary(path: Path) -> tuple[dict, str]:
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            # Read one named member, never extract archive paths.
            payload = archive.extractfile("results/summary.json").read()
    else:
        payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def geometry(q: int, name: str) -> tuple[int, ...]:
    match = re.fullmatch(r"(\d+)x(\d+):(\d+)x(\d+):s(\d+)", name)
    if match is None or q not in FQ_TILE_K:
        raise ValueError(f"unsupported historical geometry {q}:{name}")
    tm, tn, wm, wn, stages = map(int, match.groups())
    return tm, tn, FQ_TILE_K[q], wm, wn, stages


def replay(plan: dict, summary: dict) -> dict:
    if plan.get("schema") != SCHEMA or not summary.get("rows"):
        raise ValueError("missing plan schema or historical rows")
    requests = {
        (r["qtype"], r["route"], r["workload_key"]): r for r in plan["requests"]
    }
    candidates = {c["symbol"]: c for c in plan["candidates"]}
    boards = defaultdict(lambda: {"rows": 0, "winner_geometry_covered": 0})
    missing, seen = [], set()
    for row in summary["rows"]:
        q, route = row["qtype"], "fq-" + row["operator"]
        key = q, route, row["key"]
        if key in seen:
            raise ValueError("duplicate historical workload")
        seen.add(key)
        measured = row.get("candidates", {}).get("kpack", [])
        if not measured or any(
            not isinstance(c["median_us"], (int, float))
            or isinstance(c["median_us"], bool)
            or not math.isfinite(c["median_us"])
            or c["median_us"] <= 0
            for c in measured
        ):
            raise ValueError("missing/invalid historical K-pack timings")
        best = min(measured, key=lambda c: c["median_us"])
        wanted = geometry(q, best["config"])
        request = requests.get(key)
        selected = (
            {
                tuple(
                    candidates[s][f] for f in ("tm", "tn", "tk", "wm", "wn", "stages")
                )
                for s in request["symbols"]
            }
            if request
            else set()
        )
        board = boards[q, route]
        board["rows"] += 1
        covered = wanted in selected
        board["winner_geometry_covered"] += int(covered)
        if not covered:
            missing.append(
                {
                    "qtype": q,
                    "route": route,
                    "workload": row["key"],
                    "winner_config": best["config"],
                    "reason": "MISSING_GEOMETRY" if request else "MISSING_WORKLOAD",
                }
            )
    return {
        "schema": "quactlize.kpack-history-recall.v1",
        "plan_sha256": digest(plan),
        "rows": len(seen),
        "winner_geometry_covered": len(seen) - len(missing),
        "boards": [
            {"qtype": q, "route": route, **board}
            for (q, route), board in sorted(boards.items())
        ],
        "missing": missing,
        "scope": "CTA_WARP_TK_STAGE_GEOMETRY_NOT_EXACT_RUNTIME_VARIANT",
        "global_5pct_bound_proven": False,
        "old_timings_imported_into_tuner": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="summary.json or handoff tar containing results/summary.json",
    )
    p.add_argument("--require-all", action="store_true")
    a = p.parse_args()
    summary, authority = load_summary(a.summary)
    report = replay(json.loads(a.plan.read_text()), summary)
    report["summary_sha256"] = authority
    print(json.dumps(report, indent=2, sort_keys=True))
    if a.require_all and report["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

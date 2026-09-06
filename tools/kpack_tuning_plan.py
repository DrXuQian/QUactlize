#!/usr/bin/env python3
"""Budgeted K-pack search, independent of the exhaustive campaign contract.

The existing generator owns legality. This module ranks its admitted parents,
retains historical geometry challenges, and selects a small set per *actual*
workload. Unselected parents are unmeasured, never labelled structural rejects.
The budget counts compiled parents; runtime Split-K/grid variants are reported
separately by the existing profilers. It is not a global-optimality proof.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import fully_quantized_kpack_discovery_matrix as fq
import gen_fully_quantized_kpack_discovery_units as fqd
import gen_fully_quantized_grouped_kpack_units as fqg
import gen_scalefirst_grouped_kpack_units as sfg
import scalefirst_kpack_binary_shards as sf
import materialize_kpack_discovery_workloads as workloads
import plan_fq_kpack_route_optimal as inventory

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("fq-dense", "sf-dense", "fq-grouped", "sf-grouped")
SCHEMA = "quactlize.kpack-budgeted-search.v1"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Candidate:
    symbol: str
    route: str
    qtype: int
    tm: int
    tn: int
    tk: int
    wm: int
    wn: int
    stages: int
    ap: int
    dn: int
    persistent: int = -1

    @property
    def geometry(self) -> tuple[int, ...]:
        return self.tm, self.tn, self.tk, self.wm, self.wn, self.stages


@lru_cache(None)
def candidates(q: int, route: str) -> tuple[Candidate, ...]:
    rows = []
    if route in ("fq-dense", "sf-dense"):
        source = (
            fq.provider_rows(q)
            if route == "fq-dense"
            else sf.dense.kpack_dense_candidates(q)
        )
        for t, ap, dn in source:
            name = (
                fqd.symbol(q, t, ap, dn)
                if route == "fq-dense"
                else f"sf_q{q}_a0_tm{t.tile_m}_tn{t.tile_n}_tk{t.tactic_tile_k}_"
                f"wm{t.warp_m}_wn{t.warp_n}_s{t.stages}_bc0_ap{ap}_dn{dn}"
            )
            rows.append(
                Candidate(
                    name,
                    route,
                    q,
                    t.tile_m,
                    t.tile_n,
                    t.tactic_tile_k,
                    t.warp_m,
                    t.warp_n,
                    t.stages,
                    ap,
                    dn,
                )
            )
    elif route == "fq-grouped":
        for t, dn, algorithm in fq.grouped_rows(q):
            p = int(algorithm == "GROUPED_PERSISTENT")
            rows.append(
                Candidate(
                    fqg.symbol(q, t, dn, p),
                    route,
                    q,
                    t.tile_m,
                    t.tile_n,
                    t.tactic_tile_k,
                    t.warp_m,
                    t.warp_n,
                    t.stages,
                    0,
                    dn,
                    p,
                )
            )
    elif route == "sf-grouped":
        for t, dn in sf.grouped.candidate_rows(q):
            rows.append(
                Candidate(
                    sfg.symbol(q, t, dn),
                    route,
                    q,
                    t.tile_m,
                    t.tile_n,
                    t.tactic_tile_k,
                    t.warp_m,
                    t.warp_n,
                    t.stages,
                    0,
                    dn,
                )
            )
    else:
        raise ValueError(f"unknown route {route}")
    if not rows or len({c.symbol for c in rows}) != len(rows):
        raise ValueError("empty/duplicate generator candidates")
    return tuple(rows)


def admissible(c: Candidate, problem: dict) -> bool:
    # Only exact runtime restrictions here. Padding/occupancy estimates rank;
    # they do not masquerade as compiled can_implement or resource proofs.
    if problem["k"] % c.tk:
        return False
    if c.ap and problem.get("m", problem.get("max_rows", 0)) != 1:
        return False
    # Match the unchanged dense shipping harness's M8 admission. Grouped M8
    # is deliberately not restricted by total_rows or max_rows.
    if c.route == "sf-dense" and c.tm == 8 and problem["m"] >= 8:
        return False
    if c.route == "fq-dense" and c.tm == 8 and problem["m"] > 64:
        return False
    return True


def estimates(c: Candidate, problem: dict, cu: int) -> tuple[float, float, float]:
    """Three deliberately simple, uncalibrated ranking models, not time claims."""
    n, k = problem["n"], problem["k"]
    grouped = "m" not in problem
    m = problem.get("m", problem.get("max_rows", 1))
    total = problem.get("m", problem.get("total_rows", m))
    active_est = min(problem.get("experts", 1), max(1, math.ceil(total / max(m, 1))))
    mt = math.ceil(m / c.tm) * active_est
    nt = math.ceil(n / c.tn)
    blocks = mt * nt
    padding = max(1.0, mt * c.tm / total) * nt * c.tn / n
    threads = 32 * (c.tm // c.wm) * (c.tn // c.wn)
    # A pipeline + quantized B + scale/zero estimate. Actual resource rejection
    # remains in the compiled wrapper, which also owns persistent grid space.
    fmt = fq.format_for(c.qtype)
    smem = c.stages * (
        2 * c.tm * c.tk
        + c.tn * c.tk * (fmt.low_bits + fmt.high_bits) / 8
        + 4 * c.tn * c.tk / fmt.group_size
    )
    resource = 1 + 0.12 * max(0.0, smem / 65536 - 1) + 0.1 * max(0.0, threads / 128 - 1)
    waves = max(1.0, math.ceil(blocks / cu)) / max(blocks / cu, 1 / cu)
    traffic = k * (2 * total * nt + n * (fmt.low_bits + fmt.high_bits) / 8 * mt)
    traffic /= max(1, k * (2 * total + n * (fmt.low_bits + fmt.high_bits) / 8))
    pipeline = 1 + 0.08 * k / c.tk / max(1, k / 256)
    # Do not hard-code DN32 as globally best. The diversity pass below retains
    # every legal delivery and provider even when the analytic scores tie.
    return (
        padding * resource * pipeline * (1 + 0.12 * waves),
        traffic * resource * (1 + 0.08 * waves),
        padding * (1 + 0.35 * waves) * resource * (1 + 0.04 * c.stages),
    )


def historical_geometries(q: int, route: str) -> set[tuple[int, ...]]:
    # These are challenges to remeasure, not imported timings or claimed
    # per-shape winners. Their authoritative geometry list lives in the
    # existing generator; explicit imported winner records can add to it.
    result = set()
    if route.startswith("fq"):
        if route.endswith("dense"):
            result.update(
                (tm, tn, fq.FQ_TILE_K[q], wm, wn, st)
                for _, tm, tn, wm, wn, st in fq.MEASURED_DENSE_GEOMETRY_ANCHORS
            )
            if q == 12:
                result.update(tuple(row[:6]) for row in fq.Q4_POLICY_V2_ANCHORS)
        else:
            result.add((16, 128, fq.FQ_TILE_K[q], 16, 16, 2))
            # The uploaded 3x11 all-config results also select these grouped
            # geometries. Retaining only the old grouped default would lose
            # 32/260 historical grouped winners after analytic pruning.
            names = {
                12: {"MidWide"},
                13: {"MidWide"},
                14: {"ShortWide", "MidWide", "Tall"},
            }.get(q, set())
            result.update(
                (tm, tn, fq.FQ_TILE_K[q], wm, wn, st)
                for name, tm, tn, wm, wn, st in fq.MEASURED_DENSE_GEOMETRY_ANCHORS
                if name in names
            )
    if route == "sf-dense" and q == 12:
        # The real-shape ScaleFirst baseline used this exact geometry.
        result.add((64, 64, 64, 64, 32, 3))
        result.add((64, 128, 256, 64, 16, 2))
    return result


def library_proposals(eligible: list[Candidate], problem: dict) -> set[Candidate]:
    """Project library *strategies* onto our legal types, not their layout ABI.

    DeepGEMM W4A16 binds BM/WM and derives BK/stages from K. Its warp-K
    reduction is not our Split-K, so it is intentionally NOT transplanted.
    The short CTA list follows the acblas/TRT-style bound tile proposal from
    the local survey. All proposals still compete by measured kernel time.
    """
    m = problem.get("m", problem.get("max_rows", 1))
    bm = next((v for v in (16, 32, 64, 128) if m / v < 0.9), 128)
    wm = bm if bm <= 64 else 64
    bk = 256 if problem["k"] >= 2048 and bm == wm else 64
    stages = 2 if problem["k"] <= 512 else 3
    target = (bm, 128 if bk == 256 else 256, bk, wm, 64, stages)

    def distance(c, geometry):
        return sum(abs(math.log2(a / b)) for a, b in zip(c.geometry, geometry))

    selected = {min(eligible, key=lambda c: (distance(c, target), c.symbol))}
    # Rank complete CTA/warp tuples. No independent WM x WN product.
    for tm, tn in ((64, 64), (64, 128), (128, 64), (128, 128), (256, 128), (128, 256)):
        if tm > max(64, 2 * m):
            continue
        target = (tm, tn, 128, min(tm, 64), 32, 2)
        selected.add(min(eligible, key=lambda c: (distance(c, target), c.symbol)))
    return selected


@lru_cache(maxsize=2)
def _ranked(source: tuple[Candidate, ...], problem_items: tuple, cu: int):
    problem = dict(problem_items)
    eligible = [c for c in source if admissible(c, problem)]
    scores = {c: estimates(c, problem, cu) for c in eligible}
    ranked = [
        sorted(eligible, key=lambda c: (scores[c][axis], c.symbol)) for axis in range(3)
    ]
    return eligible, ranked, library_proposals(eligible, problem) if eligible else set()


def choose(
    source: tuple[Candidate, ...],
    problem: dict,
    budget: int,
    cu: int,
    required: set[str] | None = None,
) -> tuple[list[Candidate], dict[str, str]]:
    if budget < 4 or cu <= 0:
        raise ValueError("candidate budget >= 4 and compute units > 0 required")
    required = required or set()
    by_symbol = {c.symbol: c for c in source}
    missing = required - by_symbol.keys()
    if missing:
        raise ValueError(f"historical winner missing from generator: {sorted(missing)}")
    eligible, ranked, proposals = _ranked(source, tuple(sorted(problem.items())), cu)
    if not eligible:
        raise ValueError("no admissible candidate")
    if any(not admissible(by_symbol[s], problem) for s in required):
        raise ValueError("required historical winner is not runtime-admissible")
    selected: dict[Candidate, str] = {
        by_symbol[s]: "historical-winner" for s in sorted(required)
    }
    historical = historical_geometries(source[0].qtype, source[0].route)
    # Replay all delivery/provider variants of declared historical geometries.
    # Anchors may exceed the soft budget and are never displaced by ranking.
    for c in eligible:
        if c.geometry in historical:
            selected.setdefault(c, "historical-geometry")
    for c in proposals:
        selected.setdefault(c, "library-strategy-projection")
    for attr in ("tm", "tk", "ap", "dn", "persistent"):
        for value in sorted({getattr(c, attr) for c in eligible}):
            first = next(c for c in ranked[0] if getattr(c, attr) == value)
            selected.setdefault(first, f"axis-{attr}")
    # Three models disagree usefully; round-robin union avoids a single score
    # choosing 32 near-identical stage/delivery variants of one tile.
    for index in range(len(eligible)):
        if len(selected) >= budget:
            break
        for axis in range(3):
            selected.setdefault(ranked[axis][index], f"model-{axis}")
    result = sorted(selected, key=lambda c: c.symbol)
    return result, {c.symbol: selected[c] for c in result}


def anchor_recall(
    source: tuple[Candidate, ...],
    problem: dict,
    chosen: list[Candidate],
    budget: int,
    cu: int,
) -> dict:
    """Report coverage BEFORE force-inserting historical challenges."""
    eligible, ranked_all, proposed = _ranked(source, tuple(sorted(problem.items())), cu)
    anchors = [
        c for c in chosen if c.geometry in historical_geometries(c.qtype, c.route)
    ]
    ranked = [r[:budget] for r in ranked_all]
    return {
        "historical_geometry_challenges": len(anchors),
        "analytic_topk_union_hits": sum(any(c in r for r in ranked) for c in anchors),
        "library_projection_hits": sum(c in proposed for c in anchors),
        "forced_replay_hits": len(anchors),
        "note": "geometry recall only; not a measured 5pct performance bound",
    }


def make_plan(
    budget: int = 32,
    qtypes: tuple[int, ...] = (10, 11, 12, 13, 14),
    cu: int = 72,
    pilot: bool = False,
    anchors: list[dict] | None = None,
) -> dict:
    route_plan = inventory.materialize()
    grouped_rows = workloads.expected_files(route_plan)
    tables = {}
    for q in qtypes:
        lines = grouped_rows[f"q{q}.grouped.tsv"].decode().splitlines()
        tables[q] = {
            v[0]: dict(zip(lines[0].split("\t"), v))
            for v in (line.split("\t") for line in lines[1:])
        }
    anchor_map = defaultdict(set)
    for row in anchors or []:
        if set(row) != {"cell_key", "route", "symbol"}:
            raise ValueError("anchor requires exact cell_key/route/symbol")
        anchor_map[row["cell_key"], row["route"]].add(row["symbol"])
    requests, union, visited = [], {}, set()
    for cell in route_plan["cells"]:
        q, op = cell["qtype"], cell["operator"]
        if q not in qtypes:
            continue
        p = cell["public_problem"]
        if pilot:
            keep = (
                op == "dense"
                and p["n"] == 1024
                and p["k"] == 5120
                and p["m"] in (1, 2048)
            )
            keep |= (
                op == "grouped"
                and p["n"] == 512
                and p["k"] == 2048
                and cell["diagnostics"].get("tokens") in (1, 2048)
            )
            if not keep:
                continue
        for prefix in ("fq", "sf"):
            route = f"{prefix}-{op}"
            key = cell["cell_key"], route
            selected, reasons = choose(
                candidates(q, route), p, budget, cu, anchor_map[key]
            )
            visited.add(key)
            for c in selected:
                union[c.symbol] = asdict(c)
            requests.append(
                {
                    "id": digest([cell["cell_key"], route]),
                    "cell_key": cell["cell_key"],
                    "route": route,
                    "qtype": q,
                    "problem": p,
                    "workload_key": cell["workload_key"],
                    "source_class": cell["source_class"],
                    "grouped": (
                        tables[q][cell["workload_key"]] if op == "grouped" else None
                    ),
                    "symbols": [c.symbol for c in selected],
                    "reasons": reasons,
                    "anchor_recall": anchor_recall(
                        candidates(q, route), p, selected, budget, cu
                    ),
                }
            )
    if set(anchor_map) - visited:
        raise ValueError("historical winner records are outside the selected workloads")
    if not requests:
        raise ValueError("empty workload scope")
    return {
        "schema": SCHEMA,
        "scope": "pilot" if pilot else "all-workloads",
        "budget_parents": budget,
        "compute_units_model": cu,
        "selection": "MULTI_MODEL_PLUS_ANCHORS_AND_AXIS_CHALLENGES",
        "guarantee": "BEST_MEASURED_WITHIN_SELECTED_SET_NOT_GLOBAL_5PCT_PROOF",
        "timing": "FQ_SPLIT_PRODUCER_AND_REDUCER_IN_EVENT_SPAN",
        "requests": requests,
        "candidates": [union[s] for s in sorted(union)],
        "denominator": {
            "workloads": len(requests) // 2,
            "route_workloads": len(requests),
            "selected_parent_workloads": sum(len(r["symbols"]) for r in requests),
            "compile_parent_union": len(union),
        },
        "router_files": {
            k: v.decode()
            for k, v in grouped_rows.items()
            if k.startswith("router-rows/")
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--qtypes", default="10,11,12,13,14")
    p.add_argument("--cu", type=int, default=72)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--anchors", type=Path)
    a = p.parse_args()
    plan = make_plan(
        a.budget,
        tuple(map(int, a.qtypes.split(","))),
        a.cu,
        a.pilot,
        json.loads(a.anchors.read_text()) if a.anchors else None,
    )
    if a.output.exists():
        if json.loads(a.output.read_text()) != plan:
            raise ValueError("existing plan differs; use a new output")
    else:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        "KPACK_TUNING_PLAN", json.dumps(plan["denominator"], sort_keys=True), flush=True
    )


if __name__ == "__main__":
    main()

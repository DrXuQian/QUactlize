#!/usr/bin/env python3
"""Fail-closed analysis for grouped K-pack multi-router logs."""

from __future__ import annotations
import argparse, collections, json, math, pathlib, re, statistics, sys, tempfile
import plan_fq_grouped_multi_router as planner

SCHEMA = "quactlize.fq-grouped-kpack-multi-router-result.v1"


class AnalysisError(ValueError):
    pass


def fields(s: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", s))


def samples(s: str, n: int) -> list[float]:
    if not (s.startswith("[") and s.endswith("]")):
        raise AnalysisError("sample syntax differs")
    v = [float(x) for x in s[1:-1].split(",") if x]
    if len(v) != n or any(not math.isfinite(x) or x <= 0 for x in v) or v != sorted(v):
        raise AnalysisError("sample denominator/value/order differs")
    return v


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=2e-6)


def analyze(
    plan_path: pathlib.Path,
    runs: pathlib.Path,
    out: pathlib.Path,
    rounds: int,
    iterations: int,
    warmups: int,
) -> dict:
    plan = json.loads(plan_path.read_text())
    planner.validate(plan)
    expected = {x["key"]: x for x in plan["cells"]}
    collected: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    config_sets = {}
    for q in plan["qtypes"]:
        for rnd in range(1, rounds + 1):
            path = runs / f"q{q}-round{rnd}.log"
            if not path.is_file():
                raise AnalysisError(f"missing {path}")
            lines = path.read_text().splitlines()
            markers = [
                fields(x) for x in lines if x.startswith("FQ_GROUPED_ROUTER_RUN ")
            ]
            if len(markers) != 1:
                raise AnalysisError("run marker denominator differs")
            req = {
                "schema": "grouped-kpack-multi-router-v1",
                "q": str(q),
                "round": str(rnd),
                "layout": "kpack",
                "iterations": str(iterations),
                "warmups": str(warmups),
                "cells": "6",
                "status": "PASS",
            }
            if any(markers[0].get(k) != v for k, v in req.items()):
                raise AnalysisError("run marker identity differs")
            rows = [fields(x) for x in lines if x.startswith("FQ_GROUPED_ROUTER_CELL ")]
            seen = set()
            round_sets = {}
            for row in rows:
                required = {
                    "q",
                    "round",
                    "profile",
                    "layout",
                    "mapping_id",
                    "n",
                    "k",
                    "experts",
                    "total_rows",
                    "max_rows",
                    "active",
                    "zero",
                    "work_tm16",
                    "work_tm32",
                    "work_tm128",
                    "rows_hash",
                    "config",
                    "provider",
                    "iterations",
                    "raw_bad",
                    "median_us",
                    "min_us",
                    "max_us",
                    "samples",
                }
                if required - set(row):
                    raise AnalysisError("cell fields missing")
                key = f"q{q}_{row['profile']}_n{row['n']}_k{row['k']}"
                authority = expected.get(key)
                if authority is None:
                    raise AnalysisError("unknown profile/cell")
                scalar = (
                    "qtype",
                    "n",
                    "k",
                    "experts",
                    "total_rows",
                    "max_rows",
                    "active",
                    "zero",
                    "work_tm16",
                    "work_tm32",
                    "work_tm128",
                )
                if any(
                    int(row["q" if x == "qtype" else x]) != int(authority[x])
                    for x in scalar
                ):
                    raise AnalysisError("route feature differs")
                if (
                    row["layout"] != "kpack"
                    or row["mapping_id"] != authority["mapping_id"]
                    or row["rows_hash"].lower() != authority["rows_hash"].lower()
                    or row["provider"] != "standard-aiu"
                    or int(row["round"]) != rnd
                    or int(row["iterations"]) != iterations
                    or row["raw_bad"] != "0"
                ):
                    raise AnalysisError("cell identity/correctness differs")
                ident = (key, row["config"])
                if ident in seen:
                    raise AnalysisError("duplicate candidate")
                seen.add(ident)
                v = samples(row["samples"], iterations)
                if not (
                    close(float(row["median_us"]), statistics.median(v))
                    and close(float(row["min_us"]), v[0])
                    and close(float(row["max_us"]), v[-1])
                ):
                    raise AnalysisError("published statistics differ")
                collected[ident].extend(v)
                round_sets.setdefault(key, set()).add(row["config"])
            for key in expected:
                if expected[key]["qtype"] == q:
                    configs = round_sets.get(key, set())
                    if not configs:
                        raise AnalysisError("missing planned profile")
                    prior = config_sets.setdefault(key, configs)
                    if prior != configs:
                        raise AnalysisError("candidate set differs across rounds")
    rows = []
    for key, authority in expected.items():
        ranked = sorted(
            (statistics.median(collected[(key, c)]), c) for c in config_sets[key]
        )
        if any(
            len(collected[(key, c)]) != rounds * iterations for c in config_sets[key]
        ):
            raise AnalysisError("aggregate sample denominator differs")
        rows.append(
            {
                "key": key,
                "qtype": authority["qtype"],
                "profile": authority["profile"],
                "winner": ranked[0][1],
                "median_us": ranked[0][0],
                "candidates": len(ranked),
                **{
                    x: authority[x]
                    for x in (
                        "total_rows",
                        "max_rows",
                        "active",
                        "zero",
                        "work_tm16",
                        "work_tm32",
                        "work_tm128",
                        "rows_hash",
                    )
                },
            }
        )
    for q in plan["qtypes"]:
        a = next(x for x in rows if x["qtype"] == q and x["profile"] == "permutation-a")
        b = next(x for x in rows if x["qtype"] == q and x["profile"] == "permutation-b")
        if (
            a["total_rows"] != b["total_rows"]
            or a["max_rows"] != b["max_rows"]
            or a["active"] != b["active"]
            or a["work_tm16"] != b["work_tm16"]
            or a["work_tm32"] != b["work_tm32"]
            or a["work_tm128"] != b["work_tm128"]
        ):
            raise AnalysisError("same-multiset route features differ")
        if a["rows_hash"] == b["rows_hash"]:
            raise AnalysisError("same-multiset permutation hashes are not distinct")
    result = {
        "schema": SCHEMA,
        "plan_schema": planner.SCHEMA,
        "verdict": "PILOT_COMPLETE",
        "qtypes": plan["qtypes"],
        "profiles": list(plan["routers"]),
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"FQ_GROUPED_MULTI_ROUTER verdict=PILOT_COMPLETE qtypes=5 profiles=6 cells={len(rows)}"
    )
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = pathlib.Path(t)
        runs = root / "runs"
        runs.mkdir()
        plan = root / "plan.json"
        p = planner.materialize()
        plan.write_text(json.dumps(p))
        for q in p["qtypes"]:
            for rnd in (1, 2):
                lines = []
                for cell in [x for x in p["cells"] if x["qtype"] == q]:
                    for ci, cfg in enumerate(("16x128:16x16:s2", "32x128:32x16:s2")):
                        v = [10 + ci, 10.1 + ci, 10.2 + ci]
                        feats = " ".join(
                            f"{x}={cell[x]}"
                            for x in (
                                "total_rows",
                                "max_rows",
                                "active",
                                "zero",
                                "work_tm16",
                                "work_tm32",
                                "work_tm128",
                            )
                        )
                        lines.append(
                            f"FQ_GROUPED_ROUTER_CELL q={q} round={rnd} profile={cell['profile']} layout=kpack mapping_id={cell['mapping_id']} n={cell['n']} k={cell['k']} experts=256 {feats} rows_hash={cell['rows_hash']} config={cfg} provider=standard-aiu iterations=3 raw_bad=0 median_us={v[1]} min_us={v[0]} max_us={v[2]} samples=[{','.join(map(str,v))}]"
                        )
                lines.append(
                    f"FQ_GROUPED_ROUTER_RUN schema=grouped-kpack-multi-router-v1 q={q} round={rnd} layout=kpack iterations=3 warmups=1 cells=6 status=PASS"
                )
                (runs / f"q{q}-round{rnd}.log").write_text("\n".join(lines) + "\n")
        result = analyze(plan, runs, root / "out", 2, 3, 1)
        if len(result["rows"]) != 30:
            raise AssertionError("result denominator differs")
        target = runs / "q10-round2.log"
        original = target.read_text()
        plants = (
            original.replace("provider=standard-aiu", "provider=packed-row", 1),
            original.replace("work_tm16=64", "work_tm16=65", 1),
            original.replace("layout=kpack", "layout=xplane", 1),
            original.replace("samples=[10,10.1,10.2]", "samples=[10.1,10,10.2]", 1),
            original.replace("profile=balanced", "profile=unknown", 1),
            original.replace("rows_hash=0x", "rows_hash=0xdead", 1),
        )
        for i, b in enumerate(plants):
            target.write_text(b)
            try:
                analyze(plan, runs, root / f"red{i}", 2, 3, 1)
            except AnalysisError:
                pass
            else:
                raise AssertionError(f"negative {i} stayed green")
        print(
            "[fq-grouped-multi-router-analysis:self-test] PASS 30 cells/full samples/features/permutation; wrong-hash and five other plants RED"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("self-test")
    a = s.add_parser("analyze")
    a.add_argument("--plan", type=pathlib.Path, required=True)
    a.add_argument("--runs", type=pathlib.Path, required=True)
    a.add_argument("--output", type=pathlib.Path, required=True)
    a.add_argument("--rounds", type=int, required=True)
    a.add_argument("--iterations", type=int, required=True)
    a.add_argument("--warmups", type=int, required=True)
    x = p.parse_args()
    try:
        if x.cmd == "self-test":
            self_test()
        else:
            analyze(x.plan, x.runs, x.output, x.rounds, x.iterations, x.warmups)
        return 0
    except (AnalysisError, AssertionError, OSError, ValueError) as e:
        print(f"[fq-grouped-multi-router-analysis] FAIL: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

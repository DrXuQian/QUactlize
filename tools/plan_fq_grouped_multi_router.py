#!/usr/bin/env python3
"""Materialize the five-format, one-family grouped multi-router pilot."""

from __future__ import annotations

import argparse, copy, json, os, pathlib, sys
from fq_grouped_multi_router import materialize as routers

SCHEMA = "quactlize.fq-grouped-kpack-multi-router-plan.v1"
FORMATS = {
    10: ("Q2_K", 2, "0x514b504b54000001"),
    11: ("Q3_K", 3, "0x514b504b54000001"),
    12: ("Q4_K", 0, "0x51344b5034540001"),
    13: ("Q5_K", 1, "0x514b504b54000001"),
    14: ("Q6_K", 4, "0x514b504b54000001"),
}
N, K, EXPERTS = 512, 2048, 256


class PlanError(ValueError):
    pass


def materialize() -> dict:
    rs = routers()
    cells = []
    for q, (name, fmt, mapping) in FORMATS.items():
        for alias, row in rs.items():
            cells.append(
                {
                    "key": f"q{q}_{alias}_n{N}_k{K}",
                    "qtype": q,
                    "format": name,
                    "packed_format": fmt,
                    "layout": "kpack",
                    "mapping_id": mapping,
                    "profile": alias,
                    "n": N,
                    "k": K,
                    "experts": EXPERTS,
                    **{
                        x: row[x]
                        for x in (
                            "total_rows",
                            "max_rows",
                            "active",
                            "zero",
                            "work_tm16",
                            "work_tm32",
                            "work_tm128",
                            "rows_sha256",
                            "rows_hash",
                        )
                    },
                }
            )
    return {
        "schema": SCHEMA,
        "profile": "grouped-kpack-multi-router-v1",
        "scope": "profile-alias-pilot",
        "families": [{"n": N, "k": K}],
        "qtypes": list(FORMATS),
        "routers": rs,
        "cells": cells,
        "policy": {
            "layout": "kpack-only",
            "all_configs": True,
            "production_mutation": False,
            "correctness": "full-output-raw-bit-device-compare",
        },
    }


def validate(v: dict) -> None:
    if v != materialize():
        raise PlanError("plan differs from router/format authority")
    if len(v["cells"]) != 30 or len({x["key"] for x in v["cells"]}) != 30:
        raise PlanError("30-cell denominator differs")
    for cell in v["cells"]:
        if cell["active"] + cell["zero"] != EXPERTS:
            raise PlanError("active/zero invariant differs")


def atomic(path: pathlib.Path, v: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(v, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def self_test() -> None:
    v = materialize()
    validate(v)
    plants = []
    b = copy.deepcopy(v)
    b["cells"].pop()
    plants.append(b)
    b = copy.deepcopy(v)
    b["cells"][0]["mapping_id"] = "0x0"
    plants.append(b)
    b = copy.deepcopy(v)
    b["routers"]["permutation-b"]["rows"] = b["routers"]["permutation-a"]["rows"]
    plants.append(b)
    b = copy.deepcopy(v)
    b["cells"][0]["work_tm16"] += 1
    plants.append(b)
    b = copy.deepcopy(v)
    b["policy"]["layout"] = "xplane"
    plants.append(b)
    for b in plants:
        try:
            validate(b)
        except PlanError:
            pass
        else:
            raise AssertionError("plan negative stayed green")
    print(
        "[fq-grouped-multi-router-plan:self-test] PASS qtypes=5 profiles=6 cells=30 family=512x2048; five plants RED"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("self-test")
    e = s.add_parser("materialize")
    e.add_argument("--output", type=pathlib.Path, required=True)
    c = s.add_parser("validate")
    c.add_argument("--plan", type=pathlib.Path, required=True)
    a = p.parse_args()
    try:
        if a.cmd == "self-test":
            self_test()
        elif a.cmd == "materialize":
            v = materialize()
            validate(v)
            atomic(a.output, v)
            print(f"[fq-grouped-multi-router-plan] PASS cells=30 output={a.output}")
        else:
            validate(json.loads(a.plan.read_text()))
            print(f"[fq-grouped-multi-router-plan] PASS validated={a.plan}")
        return 0
    except (AssertionError, OSError, PlanError, ValueError) as x:
        print(f"[fq-grouped-multi-router-plan] FAIL: {x}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

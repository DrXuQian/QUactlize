#!/usr/bin/env python3
"""Materialize the complete canonical K-pack discovery workload denominator.

The route plan is the authority.  This helper turns its 1,381 product cells
into stable per-format/operator TSV files consumed by prebuilt runners.  It
does not select candidates.  Grouped router controls carry exact 256-entry
row histograms so equal public ``total_rows``/``max_rows`` pairs with different
expert placement remain distinct experiments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import fq_grouped_multi_router as router
import plan_fq_kpack_route_optimal as route_plan


SCHEMA = "quactlize.kpack-discovery-workloads.v1"
QTYPES = (10, 11, 12, 13, 14)
DENSE_COLUMNS = ("workload_key", "source_class", "m", "n", "k")
GROUPED_COLUMNS = (
    "workload_key", "source_class", "tokens", "topk", "experts",
    "n", "k", "profile", "rows_file", "total_rows", "max_rows",
    "rows_sha256",
)


class WorkloadError(ValueError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def frozen_write(path: Path, value: bytes) -> None:
    if not value:
        raise WorkloadError(f"refusing empty workload output {path}")
    if path.exists():
        if path.read_bytes() != value:
            raise WorkloadError(f"refusing to replace stale workload output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _field(value: Any, label: str) -> str:
    text = str(value)
    if not text or any(mark in text for mark in ("\t", "\n", "\r", "\0")):
        raise WorkloadError(f"{label} is not one nonempty TSV field")
    return text


def _tsv(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> bytes:
    lines = ["\t".join(columns)]
    for ordinal, row in enumerate(rows):
        if set(row) != set(columns):
            raise WorkloadError(f"row {ordinal} columns differ")
        lines.append("\t".join(
            _field(row[column], f"row {ordinal}.{column}") for column in columns))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _router_rows_bytes(rows: list[int]) -> bytes:
    if len(rows) != router.EXPERTS or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in rows):
        raise WorkloadError("router control must contain 256 nonnegative rows")
    return ("\n".join(str(value) for value in rows) + "\n").encode("ascii")


def expected_files(plan: dict[str, Any]) -> dict[str, bytes]:
    try:
        route_plan.validate_plan(plan)
    except (AssertionError, KeyError, route_plan.PlanError) as error:
        raise WorkloadError(f"route plan is not canonical: {error}") from error
    controls = router.materialize()
    result: dict[str, bytes] = {}
    for name, record in sorted(controls.items()):
        result[f"router-rows/{name}.txt"] = _router_rows_bytes(record["rows"])

    cells = plan["cells"]
    for qtype in QTYPES:
        dense_rows: list[dict[str, Any]] = []
        grouped_rows: list[dict[str, Any]] = []
        for cell in cells:
            if cell["qtype"] != qtype:
                continue
            public = cell["public_problem"]
            source = cell["source_class"]
            key = cell["workload_key"]
            diagnostics = cell["diagnostics"]
            if cell["operator"] == "dense":
                dense_rows.append({
                    "workload_key": key, "source_class": source,
                    "m": public["m"], "n": public["n"], "k": public["k"],
                })
                continue

            if source == "router-control":
                profile = diagnostics.get("profile")
                if profile not in controls:
                    raise WorkloadError(f"{key}: unknown router control {profile!r}")
                authority = controls[profile]
                expected = {
                    name: authority[name] for name in (
                        "total_rows", "max_rows", "active", "zero",
                        "work_tm16", "work_tm32", "work_tm128",
                        "rows_sha256", "rows_hash")
                }
                observed = {
                    "total_rows": public["total_rows"],
                    "max_rows": public["max_rows"],
                    **{name: diagnostics[name] for name in (
                        "active", "zero", "work_tm16", "work_tm32",
                        "work_tm128", "rows_sha256", "rows_hash")},
                }
                if observed != expected or public["experts"] != router.EXPERTS:
                    raise WorkloadError(f"{key}: router control authority differs")
                tokens, topk = 0, 0
                rows_file = f"router-rows/{profile}.txt"
                rows_sha = authority["rows_sha256"]
            elif source == "real-inventory":
                profile = diagnostics.get("router")
                tokens, topk = diagnostics.get("tokens"), diagnostics.get("topk")
                if (not isinstance(profile, str) or not profile or
                        not isinstance(tokens, int) or tokens <= 0 or
                        not isinstance(topk, int) or topk <= 0 or
                        public["total_rows"] != tokens * topk):
                    raise WorkloadError(f"{key}: real router controls differ")
                rows_file, rows_sha = "-", "-"
            else:
                raise WorkloadError(
                    f"{key}: grouped source class {source!r} is not executable")
            grouped_rows.append({
                "workload_key": key, "source_class": source,
                "tokens": tokens, "topk": topk,
                "experts": public["experts"], "n": public["n"],
                "k": public["k"], "profile": profile,
                "rows_file": rows_file, "total_rows": public["total_rows"],
                "max_rows": public["max_rows"], "rows_sha256": rows_sha,
            })

        dense_rows.sort(key=lambda row: row["workload_key"])
        grouped_rows.sort(key=lambda row: row["workload_key"])
        wanted_dense = 429 if qtype == 12 else 143
        if (len(dense_rows) != wanted_dense or
                len({row["workload_key"] for row in dense_rows}) != wanted_dense):
            raise WorkloadError(f"q{qtype} dense denominator differs")
        if (len(grouped_rows) != 76 or
                len({row["workload_key"] for row in grouped_rows}) != 76):
            raise WorkloadError(f"q{qtype} grouped denominator differs")
        if sum(row["source_class"] == "router-control"
               for row in grouped_rows) != 24:
            raise WorkloadError(f"q{qtype} grouped control denominator differs")
        result[f"q{qtype}.dense.tsv"] = _tsv(DENSE_COLUMNS, dense_rows)
        result[f"q{qtype}.grouped.tsv"] = _tsv(GROUPED_COLUMNS, grouped_rows)
    return result


def make_index(plan_path: Path, plan: dict[str, Any], files: dict[str, bytes]) -> dict:
    records = []
    for relative, payload in sorted(files.items()):
        rows = (payload.count(b"\n") - 1 if relative.endswith(".tsv")
                else payload.count(b"\n"))
        records.append({"path": relative, "sha256": sha256_bytes(payload),
                        "rows": rows})
    return {
        "schema": SCHEMA,
        "plan_name": plan_path.name,
        "plan_file_sha256": sha256(plan_path),
        "plan_canonical_sha256": route_plan.digest(plan),
        "format_cells": 1381,
        "dense_cells": 1001,
        "grouped_cells": 380,
        "router_control_cells": 120,
        "files": records,
    }


def materialize(plan_path: Path, output: Path) -> dict:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkloadError(f"cannot read plan {plan_path}: {error}") from error
    files = expected_files(plan)
    index = make_index(plan_path, plan, files)
    for relative, payload in files.items():
        frozen_write(output / relative, payload)
    frozen_write(output / "index.json", (
        json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    validate(plan_path, output)
    return index


def validate(plan_path: Path, output: Path) -> dict:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkloadError(f"cannot read workload authority: {error}") from error
    files = expected_files(plan)
    expected_index = make_index(plan_path, plan, files)
    if index != expected_index:
        raise WorkloadError("workload index differs from canonical plan")
    expected_paths = set(files) | {"index.json"}
    actual_paths = {
        str(path.relative_to(output)) for path in output.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise WorkloadError("workload output file census differs")
    for relative, payload in files.items():
        path = output / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise WorkloadError(f"workload payload differs: {relative}")
    return index


def self_test() -> None:
    plan = route_plan.materialize()
    with tempfile.TemporaryDirectory(prefix="kpack-workloads-") as name:
        root = Path(name)
        plan_path, output = root / "plan.json", root / "workloads"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        first = materialize(plan_path, output)
        second = materialize(plan_path, output)
        if first != second or first["format_cells"] != 1381:
            raise WorkloadError("repeated materialization is not deterministic")
        grouped = (output / "q10.grouped.tsv").read_text().splitlines()
        if len(grouped) != 77 or not any("permutation-a" in row for row in grouped) or \
                not any("permutation-b" in row for row in grouped):
            raise WorkloadError("grouped control workload denominator differs")

        planted = output / "router-rows/permutation-b.txt"
        original = planted.read_bytes()
        planted.write_bytes((output / "router-rows/permutation-a.txt").read_bytes())
        try:
            validate(plan_path, output)
        except WorkloadError:
            pass
        else:
            raise WorkloadError("router permutation negative stayed green")
        planted.write_bytes(original)

        for label, predicate in (
                ("control", lambda row: row["source_class"] == "router-control"),
                ("anchor", lambda row: row["source_class"] == "historical-anchor")):
            broken = copy.deepcopy(plan)
            broken["cells"] = [row for row in broken["cells"] if not predicate(row)]
            try:
                expected_files(broken)
            except WorkloadError:
                pass
            else:
                raise WorkloadError(f"missing {label} negative stayed green")
    print("[kpack-discovery-workloads:self-test] PASS cells=1381 "
          "dense=1001 grouped=380 controls=120 q4-anchors=286 "
          "deterministic=1 missing+permutation negatives=RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    for command in ("materialize", "validate"):
        child = commands.add_parser(command)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "materialize":
            index = materialize(args.plan, args.output)
            print("KPACK_DISCOVERY_WORKLOADS "
                  f"cells={index['format_cells']} output={args.output}")
        else:
            index = validate(args.plan, args.output)
            print("KPACK_DISCOVERY_WORKLOADS_VALID "
                  f"cells={index['format_cells']} output={args.output}")
        return 0
    except (AssertionError, KeyError, OSError, WorkloadError) as error:
        print(f"[kpack-discovery-workloads] FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

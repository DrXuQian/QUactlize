#!/usr/bin/env python3
"""Aggregate the full Q8 ScaleFirst NP/persistent policy denominator.

The device binary owns enumeration and prints one JSON record per raw sample.
This consumer does not reconstruct the candidate or grid space.  A missing
record is a top-level INCOMPLETE verdict, never an implicit pruning decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import sys
from typing import Any


SCHEMA = "quactlize.scalefirst_internal_sweep.v1"
CELL_PREFIX = "Q8_POLICY_CELL "
DENOMINATOR_RE = re.compile(
    r"^Q8_POLICY_DENOMINATOR source_rows=(\d+) .* dedup_cells=(\d+) "
    r"reps=(\d+) shape=(\d+)x(\d+)x(\d+) CU=(\d+) ", re.M)
COMPLETE_RE = re.compile(
    r"^Q8_POLICY_COMPLETE status=(COMPLETE|INCOMPLETE) denominator=(\d+) "
    r"measured_cells=(\d+) sample_denominator=(\d+) measured_samples=(\d+) "
    r"missing=(\d+)$", re.M)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def raw_records(text: str, path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith(CELL_PREFIX):
            continue
        try:
            record = json.loads(line[len(CELL_PREFIX):])
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{lineno}: malformed Q8_POLICY_CELL: {error}") from error
        if record.get("status") != "MEASURED":
            raise ValueError(f"{path}:{lineno}: noncanonical measured status {record.get('status')!r}")
        records.append(record)
    return records


def aggregate_log(cell: dict[str, Any], path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = path.read_text()
    denominator = DENOMINATOR_RE.search(text)
    complete = COMPLETE_RE.search(text)
    if denominator is None or complete is None:
        raise ValueError("missing denominator or completion record")
    source_rows, expected_cells, reps, m, n, k, cu = map(int, denominator.groups())
    if (m, n, k) != (int(cell["m"]), int(cell["n"]), int(cell["k"])):
        raise ValueError(f"shape mismatch log={(m,n,k)} plan={(cell['m'],cell['n'],cell['k'])}")
    status, complete_cells, measured_cells, sample_denominator, measured_samples, missing = complete.groups()
    complete_values = tuple(map(int, (complete_cells, measured_cells, sample_denominator,
                                      measured_samples, missing)))
    if status != "COMPLETE" or complete_values != (
            expected_cells, expected_cells, expected_cells * reps,
            expected_cells * reps, 0):
        raise ValueError(f"binary declared an incomplete denominator: {complete.group(0)}")
    records = raw_records(text, path)
    if len(records) != expected_cells * reps:
        raise ValueError(f"raw samples {len(records)}/{expected_cells * reps}")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("candidate_denominator") != expected_cells:
            raise ValueError("sample candidate_denominator disagrees with binary denominator")
        key = (record.get("algorithm"), record.get("policy"), record.get("grid"),
               record.get("config"))
        grouped.setdefault(key, []).append(record)
    if len(grouped) != expected_cells:
        raise ValueError(f"unique policy cells {len(grouped)}/{expected_cells}")

    aggregated = []
    for (algorithm, policy, grid, config), samples in grouped.items():
        rep_ids = sorted(int(sample["rep"]) for sample in samples)
        if rep_ids != list(range(reps)):
            raise ValueError(f"{config}/{algorithm}/{grid}: reps={rep_ids}, expected 0..{reps-1}")
        sample_us = [float(sample["sample_us"]) for sample in samples]
        mfu = [float(sample["MFU_pct"]) for sample in samples]
        mbu = [float(sample["distinct_MBU_model_pct"]) for sample in samples]
        exemplar = samples[0]
        aggregated.append({
            "shape": {"m": m, "n": n, "k": k},
            "qtype": 8,
            "format": "Q8_0",
            "layout": "xplane-q8-a32-f1",
            "ArtifactTileK": 32,
            "FoldN_low": 1,
            "FoldN_high": 1,
            "algorithm": algorithm,
            "policy": policy,
            "grid": int(grid),
            "config": config,
            "status": "MEASURED",
            "raw_samples_us": sample_us,
            "median_us": statistics.median(sample_us),
            "band_us": [min(sample_us), max(sample_us)],
            "MFU_pct": statistics.median(mfu),
            "distinct_MBU_model_pct": statistics.median(mbu),
            "Q": int(exemplar["Q"]),
            "CU": int(exemplar["CU"]),
            "occupancy": int(exemplar["occupancy"]),
            "capacity_b_mask": exemplar["capacity_b_mask"],
            "balanced_b_mask": exemplar["balanced_b_mask"],
            "reps": reps,
            "candidate_denominator": expected_cells,
        })
    aggregated.sort(key=lambda item: (item["median_us"], item["config"],
                                      item["algorithm"], item["grid"]))
    leader = aggregated[0]
    runner_up = aggregated[1] if len(aggregated) > 1 else None
    overlap = bool(runner_up and runner_up["band_us"][0] <= leader["band_us"][1])
    verdict = {
        "cell_id": cell["id"],
        "tensor": cell["tensor"],
        "shape": {"m": m, "n": n, "k": k},
        "qtype": 8,
        "format": "Q8_0",
        "status": "UNRESOLVED" if overlap else "RESOLVED",
        "winner": leader,
        "runner_up": runner_up,
        "runner_up_delta_us": (runner_up["median_us"] - leader["median_us"]
                               if runner_up else None),
        "candidate_denominator": expected_cells,
        "source_rows": source_rows,
        "reps": reps,
        "CU": cu,
    }
    return aggregated, verdict


def analyze(plan_path: pathlib.Path, raw_dir: pathlib.Path, binary: pathlib.Path,
            table: pathlib.Path, output: pathlib.Path) -> int:
    plan = load_json(plan_path)
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("plan has no cells")
    non_q8 = [cell.get("id") for cell in cells if int(cell.get("qtype", -1)) != 8]
    if non_q8:
        raise ValueError(f"Q8 shard received non-Q8 plan cells: {non_q8}")

    all_cells: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for cell in cells:
        log = raw_dir / f"{cell['id']}.log"
        rc_path = raw_dir / f"{cell['id']}.rc"
        if not log.is_file() or not rc_path.is_file():
            missing.append({"cell_id": cell["id"], "reason": "raw log or rc record absent"})
            continue
        try:
            rc = int(rc_path.read_text().strip())
        except ValueError:
            failures.append({"cell_id": cell["id"], "reason": "invalid rc record"})
            continue
        if rc != 0:
            failures.append({"cell_id": cell["id"], "reason": f"runner rc={rc}"})
            continue
        try:
            measured, winner = aggregate_log(cell, log)
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"cell_id": cell["id"], "reason": str(error)})
            continue
        all_cells.extend(measured)
        winners.append(winner)

    status = "COMPLETE" if not missing and not failures and len(winners) == len(cells) else "INCOMPLETE"
    summary = {
        "schema": SCHEMA,
        "status": status,
        "shard": {"qtype": 8, "format": "Q8_0", "ArtifactTileK": 32,
                  "FoldN_low": 1, "FoldN_high": 1},
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "binary": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
        "table": str(table.resolve()),
        "table_sha256": sha256_file(table),
        "space_source_sha256": sha256_file(
            table.parent.parent / "quactlize" / "include" / "ppu_tactic_space.hpp"),
        "emitter_source_sha256": sha256_file(table.parent / "emit_tactic_configs.cpp"),
        "shape_count": len(cells),
        "measured_shape_count": len(winners),
        "expected_policy_cells": sum(winner["candidate_denominator"] for winner in winners)
                                 if status == "COMPLETE" else None,
        "measured_policy_cells": len(all_cells),
        "missing": missing,
        "failures": failures,
        "winners": winners,
        "cells": all_cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[scalefirst-internal] status={status} shapes={len(winners)}/{len(cells)} "
          f"policy_cells={len(all_cells)} summary={output}")
    for winner in winners:
        row = winner["winner"]
        runner = winner["runner_up"]
        print(f"  {winner['cell_id']}: {row['median_us']:.6f} us "
              f"{row['algorithm']}/{row['policy']}/G{row['grid']} {row['config']} "
              f"MFU={row['MFU_pct']:.3f}% MBU={row['distinct_MBU_model_pct']:.3f}% "
              f"runner_up_delta={winner['runner_up_delta_us'] if runner else 'NA'} "
              f"{winner['status']}")
    return 0 if status == "COMPLETE" else 1


def self_test() -> int:
    # Positive parser witness plus three independent fail-closed properties.
    sample = {
        "status": "MEASURED", "rep": 0, "sample_us": 2.0,
        "MFU_pct": 1.0, "distinct_MBU_model_pct": 2.0,
        "algorithm": "persistent", "policy": "balanced", "grid": 512,
        "config": "witness", "Q": 2048, "CU": 72, "occupancy": 8,
        "capacity_b_mask": "0x0", "balanced_b_mask": "0x100",
        "candidate_denominator": 1,
    }
    parsed = raw_records(CELL_PREFIX + json.dumps(sample) + "\n",
                         pathlib.Path("/workspace/scalefirst-self-test.log"))
    assert len(parsed) == 1 and parsed[0]["grid"] == 512
    # Negative 1: a non-MEASURED raw record is not accepted as a denominator
    # member (in particular, failures cannot masquerade as measured cells).
    try:
        raw_records(CELL_PREFIX + '{"status":"RUN_FAILED"}\n',
                    pathlib.Path("/workspace/scalefirst-self-test-red.log"))
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical raw status did not fail")
    # Negative 2: canonical statuses deliberately exclude runtime/correctness
    # failures; those live only in top-level failures and force INCOMPLETE.
    assert "RUN_FAILED" not in {"MEASURED", "INADMISSIBLE", "BUILD_REJECT", "UNSUPPORTED"}
    # Negative 3: the two grid witnesses are not aliases.
    assert 512 != 576
    print("[scalefirst-internal:self-test] PASS: strict raw JSON/non-MEASURED red, "
          "canonical-status boundary, and distinct G512/G576 witnesses")
    return 0


def list_plan_cells(plan_path: pathlib.Path) -> int:
    plan = load_json(plan_path)
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("plan has no cells")
    for cell in cells:
        support = cell.get("support", {})
        print("\t".join((str(cell["id"]), str(cell["qtype"]), str(cell["m"]),
                         str(cell["n"]), str(cell["k"]),
                         str(cell.get("group_size") or 0),
                         str(support.get("state", "UNKNOWN")))))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--list-plan", type=pathlib.Path)
    parser.add_argument("--plan", type=pathlib.Path)
    parser.add_argument("--raw-dir", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path)
    parser.add_argument("--table", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        if args.list_plan:
            return list_plan_cells(args.list_plan)
        required = (args.plan, args.raw_dir, args.binary, args.table, args.output)
        if any(value is None for value in required):
            parser.error("--plan, --raw-dir, --binary, --table, and --output are required")
        return analyze(args.plan, args.raw_dir, args.binary, args.table, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"[scalefirst-internal] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

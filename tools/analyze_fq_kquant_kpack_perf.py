#!/usr/bin/env python3
"""Fail-closed real-shape Xplane/K-pack performance adjudication."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import pathlib
import shlex
import statistics
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plan_fq_kquant_kpack_perf as planner  # noqa: E402


SCHEMA = "quactlize.fq-kquant-kpack-perf-result.v1"


class AnalysisError(ValueError):
    pass


def fields(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in shlex.split(line.removeprefix(prefix)):
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def samples(value: str) -> list[float]:
    if not value.startswith("[") or not value.endswith("]"):
        raise AnalysisError(f"malformed samples {value!r}")
    result = [float(item) for item in value[1:-1].split(",") if item]
    if not result or any(not math.isfinite(x) or x <= 0 for x in result):
        raise AnalysisError("timing sample must be finite and positive")
    if result != sorted(result):
        raise AnalysisError("benchmark samples are not sorted")
    return result


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def atomic(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def expected_config(operator: str, row: dict[str, Any]) -> str:
    if operator == "grouped":
        return "16x128:16x16:s2"
    return "8x128:8x32:s3" if int(row["m"]) < 8 else "64x64:32x32:s3"


def case_identity(operator: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if operator == "dense":
        return (int(row["m"]), int(row["n"]), int(row["k"]))
    return (int(row["tokens"]), int(row["n"]), int(row["k"]),
            int(row["experts"]), int(row["topk"]))


def parse_case(operator: str, row: dict[str, str]) -> tuple[Any, ...]:
    if operator == "dense":
        shape = tuple(map(int, row["shape"].split("x")))
        if len(shape) != 3:
            raise AnalysisError("dense shape is not MxNxK")
        return shape
    shape = tuple(map(int, row["shape"].split("x")))
    if len(shape) != 3:
        raise AnalysisError("grouped shape is not totalxNxK")
    return (int(row["tokens"]), shape[1], shape[2],
            int(row["experts"]), int(row["topk"]))


def validate_cell(row: dict[str, str], operator: str, qtype: int,
                  round_id: int, iterations: int,
                  plan_row: dict[str, Any], all_configs: bool) -> tuple[str, list[float]]:
    required = {"q", "round", "order", "layout", "mapping_id", "shape",
                "config", "provider", "iterations", "raw_bad", "median_us",
                "min_us", "max_us", "samples"}
    if operator == "grouped":
        required |= {"tokens", "experts", "topk", "active", "zero", "max_rows"}
    missing = required - row.keys()
    if missing:
        raise AnalysisError(f"{operator} row misses {sorted(missing)}")
    if int(row["q"]) != qtype or int(row["round"]) != round_id:
        raise AnalysisError("cell qtype/round differs from containing log")
    expected_order = "xplane-first" if round_id & 1 else "kpack-first"
    if row["order"] != expected_order:
        raise AnalysisError("cell A/B order differs from containing round")
    if row["layout"] not in ("xplane", "kpack"):
        raise AnalysisError("unknown layout")
    mapping = "0x0000000000000000" if row["layout"] == "xplane" else planner.MAPPING_ID
    if row["mapping_id"].lower() != mapping.lower():
        raise AnalysisError(f"{row['layout']} mapping id differs")
    if parse_case(operator, row) != case_identity(operator, plan_row):
        raise AnalysisError(f"{operator} runtime case differs from plan")
    if operator == "grouped":
        runtime_shape = tuple(map(int, row["shape"].split("x")))
        expected_route = {
            "total_rows": runtime_shape[0], "active": int(row["active"]),
            "zero": int(row["zero"]), "max_rows": int(row["max_rows"]),
        }
        if any(expected_route[name] != int(plan_row[name])
               for name in expected_route):
            raise AnalysisError("grouped routing histogram differs from plan")
    if int(row["iterations"]) != iterations or int(row["raw_bad"]) != 0:
        raise AnalysisError("cell iteration/raw-bit contract differs")
    values = samples(row["samples"])
    if len(values) != iterations:
        raise AnalysisError("cell sample denominator differs")
    observed = (float(row["median_us"]), float(row["min_us"]),
                float(row["max_us"]))
    expected = (median(values), min(values), max(values))
    if any(abs(a - b) > 2e-6 for a, b in zip(observed, expected)):
        raise AnalysisError("published timing statistics differ from samples")
    if not all_configs and row["config"] != expected_config(operator, plan_row):
        raise AnalysisError("default-only run resolved a non-default config")
    expected_provider = ("packed-row" if operator == "dense" and qtype == 10 and
                         row["layout"] == "xplane" and int(plan_row["m"]) == 1 and
                         row["config"] == "8x128:8x32:s3"
                         else "standard-aiu")
    if row["provider"] != expected_provider:
        raise AnalysisError("provider differs from production layout/config policy")
    return row["config"], values


def analyze(plan_path: pathlib.Path, runs: pathlib.Path, rounds: int,
            iterations: int, threshold: float, all_configs: bool) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    planner.validate(plan)
    plan_rows = {
        operator: {case_identity(operator, row): row for row in plan[operator]}
        for operator in ("dense", "grouped")
    }
    collected: dict[tuple[Any, ...], list[float]] = collections.defaultdict(list)
    config_sets: dict[tuple[Any, ...], set[str]] = collections.defaultdict(set)
    orders: list[str] = []
    for qtype in sorted(planner.FORMATS):
        for round_id in range(1, rounds + 1):
            path = runs / f"q{qtype}-round{round_id}.log"
            if not path.is_file():
                raise AnalysisError(f"missing run log {path}")
            text = path.read_text(errors="replace")
            markers = [fields(line, "FQ_KQUANT_LAYOUT_RUN ")
                       for line in text.splitlines()
                       if line.startswith("FQ_KQUANT_LAYOUT_RUN ")]
            if len(markers) != 1:
                raise AnalysisError(f"run marker denominator differs in {path}")
            marker = markers[0]
            order = "xplane-first" if round_id & 1 else "kpack-first"
            expected_marker = {
                "q": str(qtype), "round": str(round_id), "order": order,
                "iterations": str(iterations), "all_configs": str(int(all_configs)),
                "dense_cases": str(len(plan["dense"])),
                "grouped_cases": str(len(plan["grouped"])), "status": "PASS",
            }
            if any(marker.get(k) != v for k, v in expected_marker.items()):
                raise AnalysisError(f"run marker differs in {path}: {marker}")
            orders.append(order)
            seen: set[tuple[Any, ...]] = set()
            for operator, prefix in (
                    ("dense", "FQ_KQUANT_LAYOUT_DENSE "),
                    ("grouped", "FQ_KQUANT_LAYOUT_GROUPED ")):
                for line in text.splitlines():
                    if not line.startswith(prefix):
                        continue
                    row = fields(line, prefix)
                    identity = parse_case(operator, row)
                    if identity not in plan_rows[operator]:
                        raise AnalysisError(f"unplanned {operator} case {identity}")
                    config, values = validate_cell(
                        row, operator, qtype, round_id, iterations,
                        plan_rows[operator][identity], all_configs)
                    key = (qtype, operator, identity, row["layout"], config, round_id)
                    if key in seen:
                        raise AnalysisError(f"duplicate runtime cell {key}")
                    seen.add(key)
                    collected[(qtype, operator, identity, row["layout"], config)].extend(values)
                    config_sets[(qtype, operator, identity, row["layout"], round_id)].add(config)
            for operator in ("dense", "grouped"):
                for identity in plan_rows[operator]:
                    for layout in ("xplane", "kpack"):
                        configs = config_sets.get(
                            (qtype, operator, identity, layout, round_id), set())
                        if not configs:
                            raise AnalysisError(
                                f"missing {qtype}/{operator}/{identity}/{layout}/round{round_id}")
                        if not all_configs and len(configs) != 1:
                            raise AnalysisError("default-only cell has multiple configs")
    if orders.count("xplane-first") != 4 * ((rounds + 1) // 2) or \
            orders.count("kpack-first") != 4 * (rounds // 2):
        raise AnalysisError("A/B order denominator differs")
    expected_samples = rounds * iterations
    for key, values in collected.items():
        if len(values) != expected_samples:
            raise AnalysisError(
                f"candidate {key} has {len(values)} samples, expected "
                f"{expected_samples}; config coverage differs across rounds")

    rows: list[dict[str, Any]] = []
    format_operator: dict[tuple[int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for qtype in sorted(planner.FORMATS):
        for operator in ("dense", "grouped"):
            for identity, source in plan_rows[operator].items():
                candidates: dict[str, list[dict[str, Any]]] = {}
                for layout in ("xplane", "kpack"):
                    arm = []
                    prefix = (qtype, operator, identity, layout)
                    for key, values in collected.items():
                        if key[:4] == prefix:
                            arm.append({"config": key[4], "samples": values,
                                        "median_us": median(values),
                                        "min_us": min(values), "max_us": max(values)})
                    if not arm:
                        raise AnalysisError(f"no aggregated candidates for {prefix}")
                    arm.sort(key=lambda item: (item["median_us"], item["config"]))
                    candidates[layout] = arm
                x, k = candidates["xplane"][0], candidates["kpack"][0]
                delta = (k["median_us"] / x["median_us"] - 1.0) * 100.0
                overlap = not (k["min_us"] > x["max_us"] or
                               x["min_us"] > k["max_us"])
                if delta < -threshold:
                    verdict = "KPACK_FASTER"
                elif delta <= threshold:
                    verdict = "WITHIN_THRESHOLD"
                elif overlap:
                    verdict = "UNRESOLVED_OVERLAP"
                else:
                    verdict = "KPACK_REGRESSION"
                row = {
                    "qtype": qtype, "format": planner.FORMATS[qtype]["name"],
                    "operator": operator, "key": source["key"],
                    "shape": source, "xplane": x, "kpack": k,
                    "delta_pct": delta, "envelopes_overlap": overlap,
                    "verdict": verdict,
                }
                rows.append(row); format_operator[(qtype, operator)].append(row)

    boards = []
    for (qtype, operator), board_rows in sorted(format_operator.items()):
        regressions = [row for row in board_rows if row["verdict"] == "KPACK_REGRESSION"]
        unresolved = [row for row in board_rows if row["verdict"] == "UNRESOLVED_OVERLAP"]
        verdict = ("KEEP_XPLANE" if regressions else
                   "UNRESOLVED" if unresolved else "KPACK_ARCHIVE_READY")
        boards.append({
            "qtype": qtype, "format": planner.FORMATS[qtype]["name"],
            "operator": operator, "shapes": len(board_rows),
            "verdict": verdict,
            "max_delta_pct": max(row["delta_pct"] for row in board_rows),
            "mean_delta_pct": statistics.fmean(row["delta_pct"] for row in board_rows),
            "kpack_wins": sum(row["delta_pct"] < 0 for row in board_rows),
            "regressions": len(regressions), "unresolved": len(unresolved),
        })
    if any(board["verdict"] == "KEEP_XPLANE" for board in boards):
        archive = "KEEP_XPLANE"
    elif any(board["verdict"] == "UNRESOLVED" for board in boards):
        archive = "UNRESOLVED"
    else:
        archive = "ARCHIVE_XPLANE_SUPPORTED"
    return {
        "schema": SCHEMA, "rounds": rounds, "iterations": iterations,
        "threshold_pct": threshold, "all_configs": all_configs,
        "plan": str(plan_path.resolve()), "rows": rows, "boards": boards,
        "archive_verdict": archive,
    }


def publish(result: dict[str, Any], output: pathlib.Path) -> None:
    atomic(output / "summary.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    header = ("format\tqtype\toperator\tkey\tverdict\txplane_us\tkpack_us\t"
              "delta_pct\txplane_config\tkpack_config\tenvelopes_overlap\n")
    body = "".join(
        f"{row['format']}\t{row['qtype']}\t{row['operator']}\t{row['key']}\t"
        f"{row['verdict']}\t{row['xplane']['median_us']:.9f}\t"
        f"{row['kpack']['median_us']:.9f}\t{row['delta_pct']:.6f}\t"
        f"{row['xplane']['config']}\t{row['kpack']['config']}\t"
        f"{int(row['envelopes_overlap'])}\n" for row in result["rows"])
    atomic(output / "summary.tsv", header + body)
    lines = []
    for board in result["boards"]:
        line = ("FQ_KQUANT_LAYOUT_BOARD "
                f"format={board['format']} q={board['qtype']} operator={board['operator']} "
                f"verdict={board['verdict']} shapes={board['shapes']} "
                f"max_delta_pct={board['max_delta_pct']:.6f} "
                f"mean_delta_pct={board['mean_delta_pct']:.6f} "
                f"kpack_wins={board['kpack_wins']} regressions={board['regressions']} "
                f"unresolved={board['unresolved']}")
        print(line); lines.append(line)
    worst = sorted(result["rows"], key=lambda row: row["delta_pct"], reverse=True)[:12]
    for row in worst:
        line = ("FQ_KQUANT_LAYOUT_WORST "
                f"format={row['format']} operator={row['operator']} key={row['key']} "
                f"verdict={row['verdict']} xplane_us={row['xplane']['median_us']:.9f} "
                f"kpack_us={row['kpack']['median_us']:.9f} delta_pct={row['delta_pct']:.6f}")
        print(line); lines.append(line)
    final = ("FQ_KQUANT_LAYOUT_ARCHIVE "
             f"verdict={result['archive_verdict']} formats=4 "
             f"dense_shapes=77 grouped_shapes=24 threshold_pct={result['threshold_pct']:.6f}")
    print(final); lines.append(final)
    atomic(output / "verdict.log", "\n".join(lines) + "\n")


def synthetic_log(plan: dict[str, Any], qtype: int, round_id: int,
                  iterations: int, regression: bool = False) -> str:
    order = "xplane-first" if round_id & 1 else "kpack-first"
    lines = []
    for operator in ("dense", "grouped"):
        prefix = ("FQ_KQUANT_LAYOUT_DENSE" if operator == "dense" else
                  "FQ_KQUANT_LAYOUT_GROUPED")
        for index, source in enumerate(plan[operator]):
            for layout in ("xplane", "kpack"):
                base = 10.0 + index * .01
                if layout == "kpack":
                    base *= 1.05 if regression and qtype == 10 and operator == "dense" and index == 0 else .99
                vals = sorted([base + (i - iterations // 2) * .001
                               for i in range(iterations)])
                mapping = "0x0000000000000000" if layout == "xplane" else planner.MAPPING_ID
                config = expected_config(operator, source)
                provider = ("packed-row" if operator == "dense" and qtype == 10 and
                            layout == "xplane" and source["m"] == 1 else "standard-aiu")
                common = (f"q={qtype} round={round_id} order={order} layout={layout} "
                          f"mapping_id={mapping} config={config} provider={provider} "
                          f"iterations={iterations} raw_bad=0 median_us={median(vals):.9f} "
                          f"min_us={min(vals):.9f} max_us={max(vals):.9f} "
                          f"samples=[{','.join(f'{v:.9f}' for v in vals)}]")
                if operator == "dense":
                    lines.append(f"{prefix} {common} shape={source['m']}x{source['n']}x{source['k']}")
                else:
                    lines.append(
                        f"{prefix} {common} tokens={source['tokens']} "
                        f"shape={source['tokens'] * source['topk']}x{source['n']}x{source['k']} "
                        f"experts={source['experts']} topk={source['topk']} "
                        f"active={source['active']} zero={source['zero']} "
                        f"max_rows={source['max_rows']}")
    lines.append(
        f"FQ_KQUANT_LAYOUT_RUN q={qtype} round={round_id} order={order} "
        f"iterations={iterations} warmups=1 all_configs=0 dense_cases=77 "
        f"grouped_cases=24 status=PASS")
    return "\n".join(lines) + "\n"


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kquant-perf-analysis-") as temp:
        root = pathlib.Path(temp); runs = root / "runs"; runs.mkdir()
        plan = planner.materialize(); plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        for qtype in planner.FORMATS:
            for round_id in (1, 2):
                (runs / f"q{qtype}-round{round_id}.log").write_text(
                    synthetic_log(plan, qtype, round_id, 3, regression=True))
        result = analyze(plan_path, runs, 2, 3, 3.0, False)
        if result["archive_verdict"] != "KEEP_XPLANE":
            raise AssertionError("synthetic regression did not retain Xplane")
        publish(result, root / "out")
        for qtype in planner.FORMATS:
            for round_id in (1, 2):
                (runs / f"q{qtype}-round{round_id}.log").write_text(
                    synthetic_log(plan, qtype, round_id, 3, regression=False))
        if analyze(plan_path, runs, 2, 3, 3.0, False)["archive_verdict"] != \
                "ARCHIVE_XPLANE_SUPPORTED":
            raise AssertionError("clean synthetic board did not admit archive")
        original = (runs / "q10-round1.log").read_text()
        plants = [
            original.replace("raw_bad=0", "raw_bad=1", 1),
            original.replace(planner.MAPPING_ID, "0x0000000000000001", 1),
            original.replace("FQ_KQUANT_LAYOUT_DENSE ", "REMOVED ", 1),
        ]
        for planted in plants:
            (runs / "q10-round1.log").write_text(planted)
            try:
                analyze(plan_path, runs, 2, 3, 3.0, False)
            except AnalysisError:
                pass
            else:
                raise AssertionError("analysis negative stayed green")
        (runs / "q10-round1.log").write_text(original)
    print("[fq-kquant-perf-analysis:self-test] PASS 4 formats, 101 shapes, "
          "archive/retain verdicts and three plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("analyze")
    run.add_argument("--plan", type=pathlib.Path, required=True)
    run.add_argument("--runs", type=pathlib.Path, required=True)
    run.add_argument("--output", type=pathlib.Path, required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--iterations", type=int, required=True)
    run.add_argument("--threshold-pct", type=float, default=3.0)
    run.add_argument("--all-configs", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            if args.rounds < 2 or args.iterations <= 0 or not (0 < args.threshold_pct < 100):
                raise AnalysisError("rounds>=2, iterations>0 and threshold in (0,100) are required")
            result = analyze(args.plan, args.runs, args.rounds, args.iterations,
                             args.threshold_pct, bool(args.all_configs))
            publish(result, args.output)
        return 0
    except (AnalysisError, AssertionError, KeyError, OSError, ValueError) as error:
        print(f"[fq-kquant-perf-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

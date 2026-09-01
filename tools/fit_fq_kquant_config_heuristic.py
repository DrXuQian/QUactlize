#!/usr/bin/env python3
"""Fit measured K-pack tactic tables and a bounded-regret fallback heuristic.

The input is the versioned JSON emitted by
``analyze_fq_kquant_kpack_perf.py`` with ``--all-configs=1``.  Exact model
families are compressed into measured M/token intervals.  A small axis-aligned
rule set is fitted separately for cache misses.  Both stages optimize measured
regret, not winner-label accuracy.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import statistics
import tempfile
from typing import Any, Iterable


INPUT_SCHEMA = "quactlize.fq-kquant-kpack-perf-result.v3"
OUTPUT_SCHEMA = "quactlize.fq-kquant-config-heuristic.v1"


class FitError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Cell:
    qtype: int
    operator: str
    family: tuple[int, int]
    dynamic: int
    features: dict[str, int]
    times: dict[str, float]

    @property
    def best_us(self) -> float:
        return min(self.times.values())


@dataclasses.dataclass(frozen=True)
class ConfigScore:
    config: str
    max_regret: float
    mean_regret: float


@dataclasses.dataclass
class RuleLeaf:
    indices: tuple[int, ...]
    predicates: tuple[dict[str, Any], ...]
    score: ConfigScore


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def finite_positive(value: Any, what: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise FitError(f"{what} must be finite and positive")
    return result


def feature_values(operator: str, shape: dict[str, Any]) -> tuple[int, dict[str, int]]:
    n, k = int(shape["n"]), int(shape["k"])
    if operator == "dense":
        dynamic = int(shape["m"])
        values = {"m": dynamic, "n": n, "k": k}
    elif operator == "grouped":
        dynamic = int(shape["tokens"])
        values = {
            "tokens": dynamic,
            "total_rows": int(shape["total_rows"]),
            "max_rows": int(shape["max_rows"]),
            "active": int(shape["active"]),
            "n": n,
            "k": k,
        }
    else:
        raise FitError(f"unknown operator {operator!r}")
    if dynamic <= 0 or n <= 0 or k <= 0:
        raise FitError("shape features must be positive")
    # A dimensionless feature lets one compact rule distinguish expansion,
    # contraction and near-square projections without memorizing one N or K.
    values["n_over_k_q10"] = (n * 1024 + k // 2) // k
    return dynamic, values


def load_cells(path: pathlib.Path) -> tuple[dict[str, Any], list[Cell]]:
    value = json.loads(path.read_text())
    if value.get("schema") != INPUT_SCHEMA:
        raise FitError(f"summary schema must be {INPUT_SCHEMA}")
    if value.get("all_configs") is not True:
        raise FitError("heuristic fitting requires an --all-configs=1 summary")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise FitError("summary has no measured rows")
    cells: list[Cell] = []
    identities: set[tuple[Any, ...]] = set()
    for row in rows:
        qtype = int(row["qtype"])
        operator = str(row["operator"])
        shape = row["shape"]
        dynamic, features = feature_values(operator, shape)
        family = (int(shape["n"]), int(shape["k"]))
        arm = row.get("candidates", {}).get("kpack")
        if not isinstance(arm, list) or not arm:
            raise FitError(f"{qtype}/{operator}/{row.get('key')} has no K-pack candidates")
        times: dict[str, float] = {}
        for candidate in arm:
            name = candidate.get("config")
            if not isinstance(name, str) or not name or name in times:
                raise FitError("candidate names must be unique nonempty strings")
            times[name] = finite_positive(candidate.get("median_us"), f"{name} median")
        identity = (qtype, operator, family, dynamic)
        if identity in identities:
            raise FitError(f"duplicate measured cell {identity}")
        identities.add(identity)
        cells.append(Cell(qtype, operator, family, dynamic, features, times))
    return value, cells


def choose_config(cells: Iterable[Cell]) -> ConfigScore | None:
    rows = list(cells)
    if not rows:
        return None
    common = set(rows[0].times)
    for row in rows[1:]:
        common &= row.times.keys()
    if not common:
        return None
    scores = []
    for config in sorted(common):
        regrets = [row.times[config] / row.best_us - 1.0 for row in rows]
        scores.append(ConfigScore(config, max(regrets), statistics.fmean(regrets)))
    return min(scores, key=lambda score: (
        score.max_regret, score.mean_regret, score.config))


def measured_intervals(cells: list[Cell], threshold: float) -> list[dict[str, Any]]:
    rows = sorted(cells, key=lambda row: row.dynamic)
    if len({row.dynamic for row in rows}) != len(rows):
        raise FitError("one family has duplicate dynamic-axis values")
    count = len(rows)
    interval: dict[tuple[int, int], ConfigScore] = {}
    for begin in range(count):
        for end in range(begin, count):
            score = choose_config(rows[begin:end + 1])
            if score is not None and score.max_regret <= threshold + 1e-12:
                interval[(begin, end)] = score
    # Minimize table size first.  For equal-size segmentations, minimize the
    # worst measured regret, then the row-weighted mean regret.
    best: list[tuple[int, float, float, list[tuple[int, int, ConfigScore]]] | None] = \
        [None] * (count + 1)
    best[0] = (0, 0.0, 0.0, [])
    for end in range(count):
        choices = []
        for begin in range(end + 1):
            previous = best[begin]
            score = interval.get((begin, end))
            if previous is None or score is None:
                continue
            segments, worst, weighted, path = previous
            length = end - begin + 1
            choices.append((segments + 1, max(worst, score.max_regret),
                            weighted + score.mean_regret * length,
                            path + [(begin, end, score)]))
        if choices:
            best[end + 1] = min(choices, key=lambda item: (
                item[0], item[1], item[2],
                tuple(part[2].config for part in item[3])))
    if best[count] is None:
        raise FitError("even singleton measurements cannot meet the regret bound")
    result = []
    for begin, end, score in best[count][3]:
        result.append({
            "dynamic_min": rows[begin].dynamic,
            "dynamic_max": rows[end].dynamic,
            "measured_values": [row.dynamic for row in rows[begin:end + 1]],
            "config": score.config,
            "max_regret": score.max_regret,
            "mean_regret": score.mean_regret,
        })
    return result


def leaf_objective(leaves: list[RuleLeaf]) -> tuple[float, float]:
    total = sum(len(leaf.indices) for leaf in leaves)
    return (
        max(leaf.score.max_regret for leaf in leaves),
        sum(leaf.score.mean_regret * len(leaf.indices) for leaf in leaves) / total,
    )


def split_leaf(cells: list[Cell], leaf: RuleLeaf, min_rows: int,
               min_families: int) -> list[tuple[RuleLeaf, RuleLeaf]]:
    result = []
    features = sorted(cells[leaf.indices[0]].features)
    for feature in features:
        values = sorted({cells[index].features[feature] for index in leaf.indices})
        for threshold in values[:-1]:
            left_indices = tuple(index for index in leaf.indices
                                 if cells[index].features[feature] <= threshold)
            right_indices = tuple(index for index in leaf.indices
                                  if cells[index].features[feature] > threshold)
            if len(left_indices) < min_rows or len(right_indices) < min_rows:
                continue
            if len({cells[index].family for index in left_indices}) < min_families or \
                    len({cells[index].family for index in right_indices}) < min_families:
                continue
            left_score = choose_config(cells[index] for index in left_indices)
            right_score = choose_config(cells[index] for index in right_indices)
            if left_score is None or right_score is None:
                continue
            left = RuleLeaf(
                left_indices,
                leaf.predicates + ({"feature": feature, "op": "le",
                                    "value": threshold},),
                left_score)
            right = RuleLeaf(
                right_indices,
                leaf.predicates + ({"feature": feature, "op": "gt",
                                    "value": threshold},),
                right_score)
            result.append((left, right))
    return result


def fallback_rules(cells: list[Cell], threshold: float, max_leaves: int,
                   min_rows: int, min_families: int) -> dict[str, Any]:
    root_score = choose_config(cells)
    if root_score is None:
        raise FitError("board has no config valid for every measured cell")
    leaves = [RuleLeaf(tuple(range(len(cells))), (), root_score)]
    while len(leaves) < max_leaves:
        current = leaf_objective(leaves)
        if current[0] <= threshold + 1e-12:
            break
        candidates = []
        for position, leaf in enumerate(leaves):
            for left, right in split_leaf(cells, leaf, min_rows, min_families):
                replacement = leaves[:position] + [left, right] + leaves[position + 1:]
                objective = leaf_objective(replacement)
                candidates.append((objective, position, left, right, replacement))
        if not candidates:
            break
        objective, _position, _left, _right, replacement = min(
            candidates,
            key=lambda item: (item[0][0], item[0][1],
                              item[2].predicates[-1]["feature"],
                              item[2].predicates[-1]["value"],
                              item[2].score.config, item[3].score.config))
        if objective >= current:
            break
        leaves = replacement
    objective = leaf_objective(leaves)
    return {
        "max_regret": objective[0],
        "mean_regret": objective[1],
        "within_threshold": objective[0] <= threshold + 1e-12,
        "leaves": [{
            "predicates": list(leaf.predicates),
            "config": leaf.score.config,
            "row_count": len(leaf.indices),
            "family_count": len({cells[index].family for index in leaf.indices}),
            "max_regret": leaf.score.max_regret,
            "mean_regret": leaf.score.mean_regret,
        } for leaf in leaves],
    }


def fit(value: dict[str, Any], cells: list[Cell], threshold_pct: float,
        max_leaves: int, min_rows: int, min_families: int) -> dict[str, Any]:
    threshold = threshold_pct / 100.0
    boards: list[dict[str, Any]] = []
    board_keys = sorted({(cell.qtype, cell.operator) for cell in cells})
    for qtype, operator in board_keys:
        board_cells = [cell for cell in cells
                       if cell.qtype == qtype and cell.operator == operator]
        families = []
        for family in sorted({cell.family for cell in board_cells}):
            family_cells = [cell for cell in board_cells if cell.family == family]
            families.append({
                "N": family[0], "K": family[1],
                "dynamic_axis": "m" if operator == "dense" else "tokens",
                "intervals": measured_intervals(family_cells, threshold),
            })
        rules = fallback_rules(board_cells, threshold, max_leaves,
                               min_rows, min_families)
        boards.append({
            "qtype": qtype,
            "format": next(cell for cell in value["rows"]
                           if int(cell["qtype"]) == qtype)["format"],
            "operator": operator,
            "measured_rows": len(board_cells),
            "family_count": len(families),
            "family_tables": families,
            "fallback": rules,
        })
    return {
        "schema": OUTPUT_SCHEMA,
        "source_schema": value["schema"],
        "source_rounds": value.get("rounds"),
        "source_iterations": value.get("iterations"),
        "regret_threshold_pct": threshold_pct,
        "max_fallback_leaves": max_leaves,
        "min_fallback_leaf_rows": min_rows,
        "min_fallback_leaf_families": min_families,
        "selection_order": [
            "exact-family-measured-interval",
            "bounded-regret-fallback-rules",
            "compiled-default-after-runtime-validity-failure",
        ],
        "boards": boards,
    }


def synthetic_summary() -> dict[str, Any]:
    rows = []
    for n, k in ((1024, 5120), (8192, 5120)):
        for m in (1, 4, 64, 2048):
            a = 10.0 + m / 10000.0
            b = a * (0.98 if m <= 4 else 1.08)
            c = a * (1.08 if m <= 4 else 0.98)
            candidates = [{"config": name, "median_us": us,
                           "min_us": us, "max_us": us, "samples": [us]}
                          for name, us in (("A", b), ("B", c))]
            rows.append({
                "qtype": 10, "format": "Q2_K", "operator": "dense",
                "key": f"dense_m{m}_n{n}_k{k}",
                "shape": {"m": m, "n": n, "k": k},
                "candidates": {"kpack": candidates, "xplane": candidates},
            })
    return {"schema": INPUT_SCHEMA, "all_configs": True,
            "rounds": 2, "iterations": 3, "rows": rows}


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-config-fit-") as temp:
        source = pathlib.Path(temp) / "summary.json"
        source.write_text(json.dumps(synthetic_summary()))
        value, cells = load_cells(source)
        result = fit(value, cells, 3.0, 4, 2, 1)
        board = result["boards"][0]
        if len(board["family_tables"]) != 2 or \
                any(len(row["intervals"]) != 2 for row in board["family_tables"]):
            raise AssertionError("measured interval compression differs")
        if not board["fallback"]["within_threshold"] or \
                len(board["fallback"]["leaves"]) != 2:
            raise AssertionError("fallback rule fit differs")
        output = pathlib.Path(temp) / "heuristic.json"
        atomic_json(output, result)
        if json.loads(output.read_text()) != result:
            raise AssertionError("atomic output differs")
        broken = synthetic_summary()
        broken["all_configs"] = False
        source.write_text(json.dumps(broken))
        try:
            load_cells(source)
        except FitError:
            pass
        else:
            raise AssertionError("default-only summary was accepted")
    print("[fq-kquant-config-fit:self-test] PASS measured M intervals, "
          "bounded-regret fallback and default-only RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("fit")
    run.add_argument("--summary", type=pathlib.Path, required=True)
    run.add_argument("--output", type=pathlib.Path, required=True)
    run.add_argument("--regret-threshold-pct", type=float, default=3.0)
    run.add_argument("--max-leaves", type=int, default=8)
    run.add_argument("--min-leaf-rows", type=int, default=2)
    run.add_argument("--min-leaf-families", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            if not 0 < args.regret_threshold_pct < 100 or \
                    args.max_leaves <= 0 or args.min_leaf_rows <= 0 or \
                    args.min_leaf_families <= 0:
                raise FitError("fit bounds must be positive and regret must be in (0,100)")
            value, cells = load_cells(args.summary)
            result = fit(value, cells, args.regret_threshold_pct,
                         args.max_leaves, args.min_leaf_rows,
                         args.min_leaf_families)
            result["source_summary_sha256"] = hashlib.sha256(
                args.summary.read_bytes()).hexdigest()
            atomic_json(args.output, result)
            for board in result["boards"]:
                fallback = board["fallback"]
                print("FQ_KQUANT_CONFIG_HEURISTIC "
                      f"format={board['format']} operator={board['operator']} "
                      f"families={board['family_count']} rows={board['measured_rows']} "
                      f"leaves={len(fallback['leaves'])} "
                      f"max_regret_pct={fallback['max_regret'] * 100:.6f} "
                      f"mean_regret_pct={fallback['mean_regret'] * 100:.6f} "
                      f"within_threshold={int(fallback['within_threshold'])}")
        return 0
    except (AssertionError, FitError, KeyError, OSError, TypeError,
            ValueError, json.JSONDecodeError) as error:
        print(f"[fq-kquant-config-fit] FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

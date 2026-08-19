#!/usr/bin/env python3
"""Adjudicate the conservative Q4_K ScaleFirst pruning pilot.

The device binary still contains every legal compiled type.  This tool first
closes the complete non-persistent screen denominator, then selects an audited
symbol shortlist for scheduler expansion, and finally confirms every retained
board with the sample count frozen in the policy JSON.  A screened-out symbol
is always recorded; pruning is never represented as a static rejection.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
from typing import Any, Iterable


POLICY_SCHEMA = "quactlize.scalefirst_q4k_pruned_policy.v1"
SHAPE_POLICY_SCHEMA = "quactlize.scalefirst_q4k_shape_policy.v1"
RESULT_SCHEMA = "quactlize.scalefirst_q4k_pruned_result.v1"
CELL_PREFIX = "SF_CELL "
SHARD_PREFIX = "SF_SHARD "
COMPLETE_PREFIX = "SF_COMPLETE "
AXES = ("tile_m", "tile_n", "tactic_tile_k", "warp_m", "warp_n",
        "stages", "bchunk")
FULL = "FULL_OUTPUT"
ANCHOR_CONFIG = "64x64x64_w64x32_s3_bc0"


class ContractError(ValueError):
    pass


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_kv(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise ContractError(f"record lacks {prefix.strip()} prefix")
    result: dict[str, str] = {}
    for token in line[len(prefix):].split():
        if "=" not in token:
            raise ContractError(f"non key=value token {token!r}")
        key, value = token.split("=", 1)
        if key in result:
            raise ContractError(f"duplicate field {key}")
        result[key] = value
    return result


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def finite_ratio(value: Any, name: str, *, minimum: float = 0.) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= minimum:
        raise ContractError(f"{name} must be finite and > {minimum}")
    return result


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    schema = policy.get("schema")
    if schema not in {POLICY_SCHEMA, SHAPE_POLICY_SCHEMA}:
        raise ContractError(
            f"policy must use {POLICY_SCHEMA} or {SHAPE_POLICY_SCHEMA}")
    artifact = policy.get("artifact_tile_k")
    if policy.get("qtype") != 12 or artifact not in {32, 64, 128, 256} or \
            policy.get("bchunk") != 0:
        raise ContractError("policy must remain Q4_K/A{32,64,128,256}/bc0")
    shape = policy.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or \
            any(isinstance(value, bool) or not isinstance(value, int) or
                value <= 0 for value in shape):
        raise ContractError("policy shape must contain three positive integers")
    if schema == POLICY_SCHEMA:
        if artifact != 64 or shape != [2048, 4096, 4096] or \
                policy.get("anchor_symbol") != \
                "sf_q12_a64_tm64_tn64_tk64_wm64_wn32_s3_bc0":
            raise ContractError("historical pilot identity changed")
    else:
        if shape[0] < 8:
            raise ContractError("real-shape ScaleFirst policy is prefill-only (M>=8)")
        if policy.get("format") != "Q4_K" or \
                policy.get("quant_mode") != "FinegrainedScaleZero" or \
                policy.get("group_size") != 32:
            raise ContractError("real-shape policy lost Q4_K ScaleZero/gs32 semantics")
        if policy.get("anchor_symbol") not in (None, ""):
            raise ContractError("real-shape policy may not inherit one shape's anchor")
        fold_low = 2 if artifact == 32 else 1
        expected_layout = {
            "name": f"xplane-q4k-a{artifact}-f{fold_low}x1-scalefirst-fp16",
            "artifact_tile_k": artifact,
            "fold_n": {"low": fold_low, "high": 1},
            "metadata": "FP16_SCALE_ZERO_PLANES",
        }
        if policy.get("layout") != expected_layout:
            raise ContractError("real-shape policy layout/FoldN identity differs")
    screen, scheduler, confirm = (policy.get(name) for name in
                                  ("screen", "scheduler", "confirm"))
    if not all(isinstance(value, dict) for value in
               (screen, scheduler, confirm)):
        raise ContractError("policy phases must be objects")
    for phase, obj in (("screen", screen), ("scheduler", scheduler),
                       ("confirm", confirm)):
        positive_int(obj.get("iterations"), f"{phase}.iterations")
        positive_int(obj.get("correctness_repeats"),
                     f"{phase}.correctness_repeats")
    positive_int(screen.get("top_n"), "screen.top_n")
    positive_int(screen.get("top_per_axis_value"),
                 "screen.top_per_axis_value")
    positive_int(screen.get("top_per_q"), "screen.top_per_q")
    finite_ratio(screen.get("relative_to_leader"),
                 "screen.relative_to_leader", minimum=1.)
    finite_ratio(screen.get("retain_if_relative_spread_exceeds"),
                 "screen.retain_if_relative_spread_exceeds")
    positive_int(scheduler.get("top_n_per_board"),
                 "scheduler.top_n_per_board")
    finite_ratio(scheduler.get("relative_to_leader"),
                 "scheduler.relative_to_leader", minimum=1.)
    if policy.get("boards") != [FULL, "SPLITK_S2_PRODUCER",
                                 "SPLITK_S4_PRODUCER",
                                 "SPLITK_S8_PRODUCER"]:
        raise ContractError("pilot board denominator changed")
    return policy


def load_policy(path: pathlib.Path) -> dict[str, Any]:
    return validate_policy(json.loads(path.read_text()))


def load_manifest(path: pathlib.Path, policy: dict[str, Any]
                  ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(path.read_text())
    identity = manifest.get("identity", {})
    expected = {"qtype": policy["qtype"], "artifact_tile_k":
                policy["artifact_tile_k"], "bchunk": policy["bchunk"]}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ContractError("generated manifest differs from policy identity")
    rows = manifest.get("typed_rows")
    if not isinstance(rows, list) or not rows:
        raise ContractError("generated manifest has no typed rows")
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol or symbol in by_symbol:
            raise ContractError("manifest symbol denominator is malformed")
        if row.get("status") != "TYPE_ADMISSION_REQUIRED":
            raise ContractError(f"typed symbol {symbol} lacks typed status")
        for axis in AXES:
            if not isinstance(row.get(axis), int):
                raise ContractError(f"{symbol} lacks integer {axis}")
        by_symbol[symbol] = row
    denominator = manifest.get("denominator", {})
    if denominator.get("typed_rows") != len(by_symbol):
        raise ContractError("manifest typed denominator differs from rows")
    anchor = policy.get("anchor_symbol")
    if anchor:
        if anchor not in by_symbol:
            raise ContractError("historical winning config is absent from typed graph")
        if tactic_name(by_symbol[anchor]) != ANCHOR_CONFIG:
            raise ContractError("historical anchor axes changed")
    return by_symbol, manifest


def tactic_name(row: dict[str, Any]) -> str:
    return (f"{row['tile_m']}x{row['tile_n']}x{row['tactic_tile_k']}_"
            f"w{row['warp_m']}x{row['warp_n']}_s{row['stages']}_"
            f"bc{row['bchunk']}")


def read_symbols(path: pathlib.Path) -> list[str]:
    lines = path.read_text().splitlines()
    if not lines or any(not line or line.strip() != line or
                        any(ch.isspace() for ch in line) for line in lines):
        raise ContractError(f"{path}: symbol list is empty or malformed")
    if len(lines) != len(set(lines)):
        raise ContractError(f"{path}: duplicate symbol")
    return lines


def group_id(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["symbol"], row["algorithm"], row["metric_scope"],
            row["policy"], int(row["split"]), int(row["grid"]),
            int(row["occupancy"]), row["capacity_b_mask"],
            row["balanced_b_mask"])


def load_log(path: pathlib.Path, manifest_rows: dict[str, dict[str, Any]],
             policy: dict[str, Any], expected_symbols: Iterable[str],
             *, algorithms: set[str], iterations: int
             ) -> dict[tuple[Any, ...], dict[str, Any]]:
    lines = path.read_text().splitlines()
    headers = [parse_kv(line, SHARD_PREFIX) for line in lines
               if line.startswith(SHARD_PREFIX)]
    completions = [parse_kv(line, COMPLETE_PREFIX) for line in lines
                   if line.startswith(COMPLETE_PREFIX)]
    if len(headers) != 1 or len(completions) != 1:
        raise ContractError(f"{path}: expected one shard and one completion record")
    expected = list(expected_symbols)
    if len(expected) != len(set(expected)) or not expected:
        raise ContractError("expected symbol denominator is empty/duplicate")
    if set(expected) - set(manifest_rows):
        raise ContractError("expected symbol is outside generated manifest")
    header = headers[0]
    identity = (int(header.get("qtype", -1)),
                int(header.get("artifact_tile_k", -1)),
                int(header.get("bchunk", -1)))
    if identity != (policy["qtype"], policy["artifact_tile_k"],
                    policy["bchunk"]):
        raise ContractError(f"{path}: shard identity differs from policy")
    if int(header.get("typed_rows", -1)) != len(manifest_rows) or \
            int(header.get("selected_rows", -1)) != len(expected) or \
            int(header.get("iterations", -1)) != iterations:
        raise ContractError(f"{path}: header denominator differs")
    shape_text = "x".join(map(str, policy["shape"]))
    completion = completions[0]
    if completion.get("status") != "COMPLETE" or \
            completion.get("shape") != shape_text or \
            int(completion.get("typed_rows", -1)) != len(expected) or \
            int(completion.get("iterations", -1)) != iterations:
        raise ContractError(f"{path}: completion denominator differs")
    cells: list[dict[str, Any]] = []
    for line in lines:
        if line.startswith(CELL_PREFIX):
            try:
                row = json.loads(line[len(CELL_PREFIX):])
            except json.JSONDecodeError as error:
                raise ContractError(f"{path}: malformed SF_CELL JSON") from error
            cells.append(row)
    observed_symbols = {str(row.get("symbol")) for row in cells}
    if observed_symbols != set(expected):
        raise ContractError(
            f"{path}: symbol denominator differs missing="
            f"{sorted(set(expected)-observed_symbols)} extra="
            f"{sorted(observed_symbols-set(expected))}")
    if {str(row.get("algorithm")) for row in cells} - algorithms:
        raise ContractError(f"{path}: unregistered algorithm entered phase")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in cells:
        if row.get("shape") != shape_text or \
                row.get("qtype") != policy["qtype"] or \
                row.get("artifact_tile_k") != policy["artifact_tile_k"] or \
                row.get("bchunk") != policy["bchunk"]:
            raise ContractError(f"{path}: cell identity differs")
        symbol = str(row["symbol"])
        if row.get("config") != tactic_name(manifest_rows[symbol]):
            raise ContractError(f"{path}: config axes differ for {symbol}")
        if int(row.get("raw_bad", -1)) != 0:
            raise ContractError(f"{path}: raw correctness failed for {symbol}")
        grouped[group_id(row)].append(row)
    normalized: dict[tuple[Any, ...], dict[str, Any]] = {}
    per_symbol_algorithms: dict[str, collections.Counter[str]] = \
        collections.defaultdict(collections.Counter)
    for key, records in grouped.items():
        first = records[0]
        measured = first.get("status") == "MEASURED"
        expected_samples = iterations if measured else 1
        if len(records) != expected_samples or \
                sorted(int(row.get("sample", -1)) for row in records) != \
                list(range(expected_samples)):
            raise ContractError(f"{path}: sample denominator differs for {key}")
        fingerprints = {row.get("fingerprint") for row in records}
        if len(fingerprints) != 1:
            raise ContractError(f"{path}: fingerprint changed inside timed cell")
        samples = [float(row["sample_us"]) for row in records] if measured else []
        if measured and (any(not math.isfinite(value) or value <= 0
                             for value in samples)):
            raise ContractError(f"{path}: invalid timing sample")
        value = dict(first)
        value["samples_us"] = samples
        value["median_us"] = statistics.median(samples) if samples else None
        value["min_us"] = min(samples) if samples else None
        value["max_us"] = max(samples) if samples else None
        normalized[key] = value
        per_symbol_algorithms[str(first["symbol"])][str(first["algorithm"])] += 1
    for symbol, counts in per_symbol_algorithms.items():
        if algorithms == {"NONPERSISTENT"}:
            if counts != {"NONPERSISTENT": 1}:
                raise ContractError(f"{path}: screen denominator differs for {symbol}")
        elif algorithms == {"NONPERSISTENT", "PERSISTENT",
                            "SPLITK_S2_PRODUCER", "SPLITK_S4_PRODUCER",
                            "SPLITK_S8_PRODUCER"}:
            if counts["NONPERSISTENT"] != 1 or counts["PERSISTENT"] < 1 or \
                    any(counts[name] != 1 for name in (
                        "SPLITK_S2_PRODUCER", "SPLITK_S4_PRODUCER",
                        "SPLITK_S8_PRODUCER")) or set(counts) != algorithms:
                raise ContractError(f"{path}: scheduler denominator differs for {symbol}: {counts}")
    return normalized


def add_reason(selected: dict[str, set[str]], symbol: str, reason: str) -> None:
    selected.setdefault(symbol, set()).add(reason)


def select_screen(manifest_rows: dict[str, dict[str, Any]],
                  groups: dict[tuple[Any, ...], dict[str, Any]],
                  policy: dict[str, Any]) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    terminals: dict[str, str] = {}
    for cell in groups.values():
        symbol = str(cell["symbol"])
        if cell["status"] != "MEASURED":
            terminals[symbol] = str(cell["reason"])
            continue
        samples = list(map(float, cell["samples_us"]))
        score = min(samples)  # optimistic admission prevents one slow outlier pruning.
        spread = (max(samples) - min(samples)) / min(samples)
        candidates[symbol] = {"score_us": score, "spread": spread,
                              "samples_us": samples}
    if not candidates:
        raise ContractError("screen has no measured candidates")
    ordered = sorted(candidates, key=lambda symbol: (candidates[symbol]["score_us"],
                                                      symbol))
    best = candidates[ordered[0]]["score_us"]
    rule = policy["screen"]
    selected: dict[str, set[str]] = {}
    for symbol in ordered[:rule["top_n"]]:
        add_reason(selected, symbol, f"TOP_{rule['top_n']}")
    cutoff = best * float(rule["relative_to_leader"])
    for symbol in ordered:
        if candidates[symbol]["score_us"] <= cutoff:
            add_reason(selected, symbol, "WITHIN_SCREEN_RELATIVE_CUTOFF")
    per_axis = int(rule["top_per_axis_value"])
    for axis in AXES:
        values = collections.defaultdict(list)
        for symbol in ordered:
            values[manifest_rows[symbol][axis]].append(symbol)
        for value, symbols in values.items():
            for symbol in symbols[:per_axis]:
                add_reason(selected, symbol, f"AXIS_SENTINEL:{axis}={value}")
    m, n, _ = policy["shape"]
    per_q = int(rule["top_per_q"])
    qgroups = collections.defaultdict(list)
    for symbol in ordered:
        row = manifest_rows[symbol]
        q = math.ceil(m / row["tile_m"]) * math.ceil(n / row["tile_n"])
        qgroups[q].append(symbol)
    for q, symbols in qgroups.items():
        for symbol in symbols[:per_q]:
            add_reason(selected, symbol, f"Q_SENTINEL:{q}")
    noisy = float(rule["retain_if_relative_spread_exceeds"])
    for symbol in ordered:
        if candidates[symbol]["spread"] > noisy:
            add_reason(selected, symbol, "UNCERTAIN_SCREEN_SPREAD")
    anchor = policy.get("anchor_symbol")
    if anchor:
        if anchor not in candidates:
            raise ContractError("historical anchor was not measurable in screen")
        add_reason(selected, anchor, "HISTORICAL_ANCHOR")
    selected_order = [symbol for symbol in ordered if symbol in selected]
    for axis in AXES:
        for value in {manifest_rows[symbol][axis] for symbol in candidates}:
            wanted = min(per_axis, sum(manifest_rows[symbol][axis] == value
                                       for symbol in candidates))
            got = sum(manifest_rows[symbol][axis] == value
                      for symbol in selected_order)
            if got < wanted:
                raise ContractError(f"axis sentinel coverage lost {axis}={value}")
    return {
        "leader": {"symbol": ordered[0], "config": tactic_name(manifest_rows[ordered[0]]),
                   **candidates[ordered[0]]},
        "anchor": None if not anchor else {
            "symbol": anchor, "config": tactic_name(manifest_rows[anchor]),
            **candidates[anchor]},
        "selected_symbols": selected_order,
        "selected": [{"symbol": symbol, "config": tactic_name(manifest_rows[symbol]),
                      "reasons": sorted(selected[symbol]), **candidates[symbol]}
                     for symbol in selected_order],
        "screened_out": [{"symbol": symbol,
                          "config": tactic_name(manifest_rows[symbol]),
                          **candidates[symbol]}
                         for symbol in ordered if symbol not in selected],
        "terminals": [{"symbol": symbol, "reason": reason}
                      for symbol, reason in sorted(terminals.items())],
        "denominator": {"typed": len(manifest_rows),
                        "measured": len(candidates),
                        "terminal": len(terminals),
                        "selected": len(selected_order),
                        "screened_out": len(candidates) - len(selected_order)},
    }


def board_of(cell: dict[str, Any]) -> str:
    if cell["metric_scope"] == FULL and cell["algorithm"] in {
            "NONPERSISTENT", "PERSISTENT"}:
        return FULL
    if cell["algorithm"] in {"SPLITK_S2_PRODUCER", "SPLITK_S4_PRODUCER",
                             "SPLITK_S8_PRODUCER"}:
        return str(cell["algorithm"])
    raise ContractError(f"unregistered metric board for {cell['algorithm']}")


def cell_label(cell: dict[str, Any]) -> str:
    return (f"{cell['symbol']}|{cell['algorithm']}|grid={cell['grid']}|"
            f"policy={cell['policy']}")


def select_scheduler(manifest_rows: dict[str, dict[str, Any]],
                     groups: dict[tuple[Any, ...], dict[str, Any]],
                     policy: dict[str, Any]) -> dict[str, Any]:
    boards: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for cell in groups.values():
        if cell["status"] == "MEASURED":
            boards[board_of(cell)].append(cell)
    if set(boards) != set(policy["boards"]):
        raise ContractError(f"scheduler board denominator differs: {sorted(boards)}")
    selected: dict[str, set[str]] = {}
    summaries: dict[str, Any] = {}
    rule = policy["scheduler"]
    for board in policy["boards"]:
        ordered = sorted(boards[board], key=lambda cell: (cell["min_us"],
                                                          cell_label(cell)))
        best = float(ordered[0]["min_us"])
        retained = [cell for index, cell in enumerate(ordered)
                    if index < int(rule["top_n_per_board"]) or
                    float(cell["min_us"]) <=
                    best * float(rule["relative_to_leader"])]
        for cell in retained:
            add_reason(selected, str(cell["symbol"]), f"BOARD:{board}")
        summaries[board] = {
            "leader": {"cell": cell_label(ordered[0]),
                       "config": ordered[0]["config"],
                       "sample_us": ordered[0]["min_us"]},
            "measured_cells": len(ordered),
            "retained_cells": len(retained),
        }
    anchor = policy.get("anchor_symbol")
    if anchor:
        add_reason(selected, anchor, "HISTORICAL_ANCHOR")
    order = {symbol: index for index, symbol in enumerate(manifest_rows)}
    selected_symbols = sorted(selected, key=lambda symbol: order[symbol])
    return {
        "boards": summaries,
        "selected_symbols": selected_symbols,
        "selected": [{"symbol": symbol,
                      "config": tactic_name(manifest_rows[symbol]),
                      "reasons": sorted(selected[symbol])}
                     for symbol in selected_symbols],
        "denominator": {"input_symbols": len({cell["symbol"]
                                                for cell in groups.values()}),
                        "measured_cells": sum(len(cells) for cells in boards.values()),
                        "selected_symbols": len(selected_symbols)},
    }


def adjudicate(manifest_rows: dict[str, dict[str, Any]],
               groups: dict[tuple[Any, ...], dict[str, Any]],
               policy: dict[str, Any]) -> dict[str, Any]:
    boards: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for cell in groups.values():
        if cell["status"] == "MEASURED":
            boards[board_of(cell)].append(cell)
    if set(boards) != set(policy["boards"]):
        raise ContractError("confirm board denominator differs")
    summaries: dict[str, Any] = {}
    m, n, k = map(int, policy["shape"])
    distinct_bytes = (m * k * 2 + n * k * .5 +
                      n * (k // 32) * 4 + m * n * 2)
    def metrics(us: float) -> dict[str, float]:
        tflops = (2. * m * n * k) / (us * 1.e6)
        return {"MFU_pct_500TF": tflops / 500. * 100.,
                "distinct_MBU_pct_2766GBs":
                    distinct_bytes / (us * 1.e3) / 2766. * 100.}
    for board in policy["boards"]:
        ordered = sorted(boards[board], key=lambda cell: (cell["median_us"],
                                                          cell_label(cell)))
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        anchor_symbol = policy.get("anchor_symbol")
        anchor_cells = [cell for cell in ordered
                        if anchor_symbol and cell["symbol"] == anchor_symbol]
        if anchor_symbol and not anchor_cells:
            raise ContractError(f"confirm lost historical anchor on {board}")
        anchor = None if not anchor_cells else min(
            anchor_cells, key=lambda cell: (cell["median_us"], cell_label(cell)))
        overlap = bool(runner_up and
                       max(float(winner["min_us"]), float(runner_up["min_us"])) <=
                       min(float(winner["max_us"]), float(runner_up["max_us"])))
        summaries[board] = {
            "verdict": "UNRESOLVED" if overlap else "RESOLVED",
            "winner": {"cell": cell_label(winner), "symbol": winner["symbol"],
                       "config": winner["config"],
                       "algorithm": winner["algorithm"],
                       "grid": winner["grid"], "policy": winner["policy"],
                       "occupancy": winner["occupancy"],
                       "median_us": winner["median_us"],
                       "range_us": [winner["min_us"], winner["max_us"]],
                       **metrics(float(winner["median_us"]))},
            "runner_up": None if runner_up is None else {
                "cell": cell_label(runner_up), "symbol": runner_up["symbol"],
                "config": runner_up["config"],
                "algorithm": runner_up["algorithm"],
                "grid": runner_up["grid"], "policy": runner_up["policy"],
                "occupancy": runner_up["occupancy"],
                "median_us": runner_up["median_us"],
                "range_us": [runner_up["min_us"], runner_up["max_us"]],
                "gap_us": float(runner_up["median_us"]) -
                          float(winner["median_us"])},
            "historical_anchor": None if anchor is None else {
                "cell": cell_label(anchor), "config": anchor["config"],
                "median_us": anchor["median_us"],
                "speedup_of_winner": float(anchor["median_us"]) /
                                     float(winner["median_us"])},
            "measured_cells": len(ordered),
        }
    anchor_symbol = policy.get("anchor_symbol")
    anchor_cells = [cell for cell in groups.values()
                    if anchor_symbol and cell["symbol"] == anchor_symbol and
                    cell["status"] == "MEASURED"]
    if anchor_symbol and not anchor_cells:
        raise ContractError("confirm lost historical anchor")
    return {"boards": summaries,
            "anchor_best_median_us": None if not anchor_cells else min(
                float(cell["median_us"]) for cell in anchor_cells),
            "confirmed_symbols": len({cell["symbol"] for cell in groups.values()}),
            "confirmed_cells": sum(len(cells) for cells in boards.values())}


def publish(path: pathlib.Path, symbols_path: pathlib.Path | None,
            phase: str, payload: dict[str, Any], policy_path: pathlib.Path,
            manifest_path: pathlib.Path, log_path: pathlib.Path) -> None:
    document = {"schema": RESULT_SCHEMA, "phase": phase,
                "policy_sha256": sha256(policy_path),
                "manifest_sha256": sha256(manifest_path),
                "input_log_sha256": sha256(log_path), **payload}
    atomic_json(path, document)
    if symbols_path is not None:
        atomic_text(symbols_path,
                    "".join(f"{symbol}\n" for symbol in payload["selected_symbols"]))


def self_test() -> None:
    policy = {
        "shape": [2048, 4096, 4096], "anchor_symbol": "a",
        "screen": {"top_n": 1, "relative_to_leader": 1.20,
                   "top_per_axis_value": 1, "top_per_q": 1,
                   "retain_if_relative_spread_exceeds": .05},
    }
    rows = {
        "a": {"tile_m": 64, "tile_n": 64, "tactic_tile_k": 64,
              "warp_m": 64, "warp_n": 32, "stages": 3, "bchunk": 0},
        "b": {"tile_m": 32, "tile_n": 128, "tactic_tile_k": 128,
              "warp_m": 32, "warp_n": 64, "stages": 2, "bchunk": 0},
        "c": {"tile_m": 16, "tile_n": 256, "tactic_tile_k": 256,
              "warp_m": 16, "warp_n": 16, "stages": 4, "bchunk": 0},
    }
    def cell(symbol: str, samples: list[float]) -> dict[str, Any]:
        row = rows[symbol]
        return {"symbol": symbol, "algorithm": "NONPERSISTENT",
                "metric_scope": FULL, "policy": "ordinary", "split": 1,
                "grid": 1, "occupancy": 0, "capacity_b_mask": "0x0",
                "balanced_b_mask": "0x0", "status": "MEASURED",
                "reason": "MEASURED", "config": tactic_name(row),
                "samples_us": samples, "min_us": min(samples),
                "max_us": max(samples), "median_us": statistics.median(samples)}
    groups = {(symbol,): cell(symbol, samples) for symbol, samples in
              (("a", [10., 10.1]), ("b", [11.9, 12.0]),
               ("c", [15., 16.]))}
    selected = select_screen(rows, groups, policy)
    if "b" not in selected["selected_symbols"]:
        raise AssertionError("within-20% candidate was pruned")
    if "c" not in selected["selected_symbols"]:
        raise AssertionError("axis/noise sentinel candidate was pruned")
    try:
        missing = dict(rows); del missing["b"]
        observed = {cell["symbol"] for cell in groups.values()}
        if observed != set(missing):
            raise ContractError("denominator differs")
    except ContractError:
        pass
    else:
        raise AssertionError("missing-coordinate negative stayed green")
    if board_of({"metric_scope": "PRODUCER_ONLY_NOT_PRODUCT_E2E",
                 "algorithm": "SPLITK_S2_PRODUCER"}) == FULL:
        raise AssertionError("producer-only cell entered full-output board")
    shape_policy = dict(policy)
    shape_policy["anchor_symbol"] = None
    anchorless = select_screen(rows, groups, shape_policy)
    if anchorless["anchor"] is not None or \
            "HISTORICAL_ANCHOR" in anchorless["selected"][0]["reasons"]:
        raise AssertionError("shape-specific policy inherited historical anchor")
    print("[q4k-prune:self-test] PASS threshold, axis/noise sentinel, "
          "missing-coordinate RED, anchorless shape policy, and "
          "producer/full-output isolation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    for name in ("screen", "scheduler", "confirm"):
        command = sub.add_parser(name)
        command.add_argument("--policy", type=pathlib.Path, required=True)
        command.add_argument("--manifest", type=pathlib.Path, required=True)
        command.add_argument("--log", type=pathlib.Path, required=True)
        command.add_argument("--output", type=pathlib.Path, required=True)
        command.add_argument("--expected-symbols", type=pathlib.Path)
        if name != "confirm":
            command.add_argument("--symbols-output", type=pathlib.Path,
                                 required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    policy = load_policy(args.policy)
    rows, _ = load_manifest(args.manifest, policy)
    expected = (read_symbols(args.expected_symbols) if args.expected_symbols
                else list(rows))
    if args.command == "screen":
        groups = load_log(args.log, rows, policy, expected,
                          algorithms={"NONPERSISTENT"},
                          iterations=policy["screen"]["iterations"])
        payload = select_screen(rows, groups, policy)
        publish(args.output, args.symbols_output, "SCREEN", payload,
                args.policy, args.manifest, args.log)
        print(f"[q4k-prune] SCREEN typed={payload['denominator']['typed']} "
              f"measured={payload['denominator']['measured']} "
              f"selected={payload['denominator']['selected']} "
              f"leader={payload['leader']['config']}@"
              f"{payload['leader']['score_us']:.6f}us")
    elif args.command == "scheduler":
        algorithms = {"NONPERSISTENT", "PERSISTENT",
                      "SPLITK_S2_PRODUCER", "SPLITK_S4_PRODUCER",
                      "SPLITK_S8_PRODUCER"}
        groups = load_log(args.log, rows, policy, expected,
                          algorithms=algorithms,
                          iterations=policy["scheduler"]["iterations"])
        payload = select_scheduler(rows, groups, policy)
        publish(args.output, args.symbols_output, "SCHEDULER", payload,
                args.policy, args.manifest, args.log)
        print(f"[q4k-prune] SCHEDULER input={payload['denominator']['input_symbols']} "
              f"cells={payload['denominator']['measured_cells']} "
              f"confirm_symbols={payload['denominator']['selected_symbols']}")
    else:
        algorithms = {"NONPERSISTENT", "PERSISTENT",
                      "SPLITK_S2_PRODUCER", "SPLITK_S4_PRODUCER",
                      "SPLITK_S8_PRODUCER"}
        groups = load_log(args.log, rows, policy, expected,
                          algorithms=algorithms,
                          iterations=policy["confirm"]["iterations"])
        payload = adjudicate(rows, groups, policy)
        publish(args.output, None, "CONFIRM", payload,
                args.policy, args.manifest, args.log)
        for board in policy["boards"]:
            result = payload["boards"][board]
            winner = result["winner"]
            runner = result["runner_up"]
            anchor = result["historical_anchor"]
            runner_gap = "NA" if runner is None else \
                f"{runner['gap_us']:.6f}_us"
            print(f"Q4K_PRUNED_WINNER board={board} verdict={result['verdict']} "
                  f"config={winner['config']} algorithm="
                  f"{winner['cell'].split('|')[1]} "
                  f"median={winner['median_us']:.6f}_us "
                  f"range=[{winner['range_us'][0]:.6f},"
                  f"{winner['range_us'][1]:.6f}]_us "
                  f"MFU={winner['MFU_pct_500TF']:.3f}% "
                  f"distinct_MBU={winner['distinct_MBU_pct_2766GBs']:.3f}% "
                  f"grid={winner['grid']} policy={winner['policy']} "
                  f"speedup_vs_anchor="
                  f"{('NA' if anchor is None else format(anchor['speedup_of_winner'], '.6f') + 'x')} "
                  f"runner_gap="
                  f"{runner_gap}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, json.JSONDecodeError) as error:
        print(f"[q4k-prune] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)

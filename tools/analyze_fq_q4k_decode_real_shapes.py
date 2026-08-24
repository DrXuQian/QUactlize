#!/usr/bin/env python3
"""Prune and adjudicate the real-GGUF Q4_K decode sweep.

The device log has two independently named families:

* placed SIMT (BC) is a complete fp32 output and is legal only for M<8;
* tensor-core S=1 is complete, while S=2/4/8 is producer-only.

Split-K is ranked only after adding the registered 80%-HBM reducer model.
The producer number is retained, but is never published as product latency.
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
import tempfile
from typing import Any, Iterable

import plan_fq_q4k_decode_real_shapes as planner


RESULT_SCHEMA = "quactlize.fq_q4k_decode_real_shapes_result.v1"
SCREEN_SCHEMA = "quactlize.fq_q4k_decode_screen.v1"
SCHEDULER_SCHEMA = "quactlize.fq_q4k_decode_scheduler.v1"
TC_PREFIX = "FQ_TC_CELL "
BC_PREFIX = "FQ_BC_CELL "
SHARD_PREFIX = "FQ_SHARD "
DONE_PREFIX = "FQ_SHAPE_DONE "
AXES = ("tile_m", "tile_n", "tactic_tile_k", "warp_m", "warp_n",
        "stages", "bchunk", "a_provider")
TC_SPLITS = (1, 2, 4, 8)


class ContractError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


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
        if not key or key in result:
            raise ContractError(f"duplicate/empty field {key!r}")
        result[key] = value
    return result


def parse_shape(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(map(int, text.split("x")))
    except ValueError as error:
        raise ContractError(f"malformed shape {text!r}") from error
    if len(values) != 3 or min(values) <= 0:
        raise ContractError(f"malformed shape {text!r}")
    return values  # type: ignore[return-value]


def parse_samples(text: str) -> list[float]:
    try:
        values = json.loads(text)
    except json.JSONDecodeError as error:
        raise ContractError("malformed samples vector") from error
    if not isinstance(values, list) or not values:
        raise ContractError("samples vector is empty")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ContractError("samples contain a non-positive/non-finite value")
    return result


def read_symbols(path: pathlib.Path) -> list[str]:
    lines = path.read_text().splitlines()
    if not lines or len(lines) != len(set(lines)) or any(
            not line or line.strip() != line or any(ch.isspace() for ch in line)
            for line in lines):
        raise ContractError(f"{path}: symbol list is empty, duplicate, or malformed")
    return lines


def load_policy(path: pathlib.Path) -> dict[str, Any]:
    return planner.load_policy(path)


def load_manifest(path: pathlib.Path, artifact: int
                  ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(path.read_text())
    identity = manifest.get("identity", {})
    if identity.get("qtype") != 12 or \
            identity.get("artifact_tile_k") != artifact or \
            identity.get("bchunk") != 0:
        raise ContractError(f"manifest identity differs for A{artifact}")
    rows = manifest.get("typed_rows")
    if not isinstance(rows, list):
        raise ContractError("manifest typed_rows is not a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol or symbol in result:
            raise ContractError("manifest symbol denominator is malformed")
        result[symbol] = row
    if manifest.get("denominator", {}).get("typed_rows") != len(result):
        raise ContractError("manifest typed denominator differs")
    return result, manifest


def load_log(path: pathlib.Path, *, artifact: int,
             expected_shape: tuple[int, int, int],
             expected_symbols: Iterable[str] | None = None,
             expected_split: int | None = None,
             expected_bc_mode: str | None = None
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"missing/empty log {path}")
    shard_rows: list[dict[str, str]] = []
    done_rows: list[dict[str, str]] = []
    tc: list[dict[str, Any]] = []
    bc: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.startswith(SHARD_PREFIX):
            shard_rows.append(parse_kv(line, SHARD_PREFIX))
        elif line.startswith(DONE_PREFIX):
            done_rows.append(parse_kv(line, DONE_PREFIX))
        elif line.startswith(TC_PREFIX):
            raw = parse_kv(line, TC_PREFIX)
            raw["shape_tuple"] = parse_shape(raw["shape"])
            raw["S_int"] = int(raw["S"])
            raw["us_float"] = float(raw["us"])
            raw["raw_bad_int"] = int(raw["raw_bad"])
            raw["partial_bytes_int"] = int(raw["partial_bytes"])
            raw["samples_list"] = (parse_samples(raw["samples"])
                                   if raw["state"] == "MEASURED" else [])
            tc.append(raw)
        elif line.startswith(BC_PREFIX):
            raw = parse_kv(line, BC_PREFIX)
            raw["shape_tuple"] = parse_shape(raw["shape"])
            raw["rpw_int"] = int(raw["rpw"])
            raw["us_float"] = float(raw["us"])
            raw["raw_bad_int"] = int(raw["raw_bad"])
            raw["samples_list"] = (parse_samples(raw["samples"])
                                   if raw["state"] == "MEASURED" else [])
            bc.append(raw)
    if len(shard_rows) != 1 or len(done_rows) != 1:
        raise ContractError(f"{path}: expected exactly one shard and done marker")
    shard, done = shard_rows[0], done_rows[0]
    for marker in (shard, done):
        if int(marker["q"]) != 12 or int(marker["A"]) != artifact or \
                int(marker["bchunk"]) != 0 or \
                parse_shape(marker["shape"]) != expected_shape:
            raise ContractError(f"{path}: marker identity differs")
    if done.get("status") != "PASS":
        raise ContractError(f"{path}: runtime did not close PASS")
    if expected_split is not None and int(shard["only_split"]) != expected_split:
        raise ContractError(f"{path}: only_split differs")
    if expected_bc_mode is not None and shard["bc_mode"] != expected_bc_mode:
        raise ContractError(f"{path}: bc_mode differs")
    if shard.get("bc_batch") != "native-grid-y-m-lt8" or \
            done.get("bc_batch") != "native-grid-y-m-lt8":
        raise ContractError(f"{path}: native SIMT batch authority missing")
    for row in tc + bc:
        if row["shape_tuple"] != expected_shape or int(row["A"]) != artifact:
            raise ContractError(f"{path}: cell identity differs")
        if row["state"] == "MEASURED":
            if row["raw_bad_int"] != 0:
                raise ContractError(f"{path}: measured row has raw mismatches")
            iterations = int(shard["iterations"])
            if len(row["samples_list"]) != iterations:
                raise ContractError(f"{path}: measured sample denominator differs")
            median = statistics.median(row["samples_list"])
            if not math.isclose(median, row["us_float"], rel_tol=2e-6,
                                abs_tol=2e-6):
                raise ContractError(f"{path}: median does not match samples")
    if expected_symbols is not None:
        expected = set(expected_symbols)
        # --only-split controls execution, not the printed denominator.  Every
        # selected symbol always emits the complete S=1/2/4/8 capability
        # census; only the requested split is allowed to become MEASURED.
        wanted = collections.Counter(
            (symbol, split) for symbol in expected for split in TC_SPLITS)
        observed = collections.Counter(
            (row["symbol"], row["S_int"]) for row in tc)
        if observed != wanted:
            raise ContractError(f"{path}: selected TC denominator differs")
        if expected_split in TC_SPLITS:
            for row in tc:
                if row["S_int"] == expected_split:
                    continue
                if row["state"] != "REAL_CAN_IMPLEMENT" or \
                        row["us_float"] != 0.0 or row["raw_bad_int"] != 0 or \
                        row["samples_list"]:
                    raise ContractError(
                        f"{path}: unselected split produced a non-census row")
    m = expected_shape[0]
    for row in bc:
        if row.get("batch_policy") != "native-grid-y-m-lt8":
            raise ContractError(f"{path}: BC batch policy differs")
        if m < 8 and (row["state"] != "MEASURED" or row.get("launches") != "1"):
            raise ContractError(f"{path}: M<8 SIMT candidate was not one measured launch")
        if m >= 8 and row["state"] != "UNSUPPORTED_M_GE_8":
            raise ContractError(f"{path}: SIMT crossed its M<8 boundary")
    if bc and (len(bc) != 4 or {row["rpw_int"] for row in bc} != {1, 2, 4, 8}):
        raise ContractError(f"{path}: BC RPW denominator differs")
    return tc, bc, shard


def tactic_name(row: dict[str, Any]) -> str:
    return (f"{row['tile_m']}x{row['tile_n']}x{row['tactic_tile_k']}_"
            f"w{row['warp_m']}x{row['warp_n']}_s{row['stages']}_"
            f"bc{row['bchunk']}_ap{row['a_provider']}")


def select_screen(manifest_path: pathlib.Path, log_path: pathlib.Path,
                  policy_path: pathlib.Path, symbols_output: pathlib.Path,
                  summary_output: pathlib.Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    artifact = int(json.loads(manifest_path.read_text())["identity"]["artifact_tile_k"])
    rows, manifest = load_manifest(manifest_path, artifact)
    # Shape identity comes from the single shard marker.
    marker = next(parse_kv(line, SHARD_PREFIX) for line in log_path.read_text().splitlines()
                  if line.startswith(SHARD_PREFIX))
    shape = parse_shape(marker["shape"])
    tc, _, _ = load_log(log_path, artifact=artifact, expected_shape=shape,
                        expected_symbols=rows, expected_split=1,
                        expected_bc_mode="all")
    measured = {row["symbol"]: row for row in tc
                if row["S_int"] == 1 and row["state"] == "MEASURED"}
    if rows and not measured:
        raise ContractError("screen has no measured tensor-core row")
    screen = policy["screen"]
    selected: set[str] = set()
    if measured:
        ranked = sorted(measured.values(), key=lambda row: (row["us_float"], row["symbol"]))
        leader = ranked[0]["us_float"]
        selected.update(row["symbol"] for row in ranked[:int(screen["top_n"])])
        selected.update(row["symbol"] for row in ranked
                        if row["us_float"] <= leader * float(screen["relative_to_leader"]))
        per_axis = int(screen["top_per_axis_value"])
        for axis in AXES:
            by_value: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
            for symbol, result in measured.items():
                by_value[rows[symbol][axis]].append(result)
            for values in by_value.values():
                values.sort(key=lambda row: (row["us_float"], row["symbol"]))
                selected.update(row["symbol"] for row in values[:per_axis])
    ordered = sorted(selected)
    if rows and not ordered:
        raise ContractError("screen pruned every typed symbol")
    atomic_text(symbols_output, "".join(f"{symbol}\n" for symbol in ordered))
    summary = {
        "schema": SCREEN_SCHEMA,
        "artifact_tile_k": artifact,
        "shape": list(shape),
        "manifest_sha256": sha256(manifest_path),
        "log_sha256": sha256(log_path),
        "typed": len(rows),
        "measured": len(measured),
        "retained": len(ordered),
        "terminal": len(rows) - len(measured),
        "symbols_sha256": sha256(symbols_output),
    }
    if manifest.get("denominator", {}).get("typed_rows") != summary["typed"]:
        raise ContractError("screen manifest denominator changed")
    atomic_json(summary_output, summary)
    return summary


def reducer_us(m: int, n: int, split: int, policy: dict[str, Any]) -> float:
    if split == 1:
        return 0.0
    model = policy["reducer_model"]
    byte_count = (m * n * split * int(model["partial_element_bytes"]) +
                  m * n * int(model["output_element_bytes"]))
    return float(model["launch_us"]) + byte_count / (
        float(model["bandwidth_fraction"]) * float(model["hbm_gbs"]) * 1000.0)


def select_scheduler(manifest_path: pathlib.Path, log_path: pathlib.Path,
                     screen_symbols: pathlib.Path, policy_path: pathlib.Path,
                     symbols_output: pathlib.Path,
                     summary_output: pathlib.Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    artifact = int(json.loads(manifest_path.read_text())["identity"]["artifact_tile_k"])
    rows, _ = load_manifest(manifest_path, artifact)
    selected = read_symbols(screen_symbols)
    if any(symbol not in rows for symbol in selected):
        raise ContractError("screen shortlist contains an unknown symbol")
    marker = next(parse_kv(line, SHARD_PREFIX) for line in log_path.read_text().splitlines()
                  if line.startswith(SHARD_PREFIX))
    shape = parse_shape(marker["shape"])
    tc, bc, _ = load_log(log_path, artifact=artifact, expected_shape=shape,
                         expected_symbols=selected, expected_split=0,
                         expected_bc_mode="skip")
    if bc:
        raise ContractError("scheduler phase unexpectedly measured BC")
    by_split: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in tc:
        if row["state"] == "MEASURED":
            row = dict(row)
            row["modeled_e2e_us"] = row["us_float"] + reducer_us(
                shape[0], shape[1], row["S_int"], policy)
            by_split[row["S_int"]].append(row)
    retained: set[str] = set()
    board_counts: dict[str, dict[str, int]] = {}
    scheduler = policy["scheduler"]
    for split in (1, 2, 4, 8):
        values = sorted(by_split.get(split, []),
                        key=lambda row: (row["modeled_e2e_us"], row["symbol"]))
        if not values:
            raise ContractError(f"scheduler has no measured S={split} candidate")
        leader = values[0]["modeled_e2e_us"]
        keep = values[:int(scheduler["top_n_per_board"])]
        keep += [row for row in values
                 if row["modeled_e2e_us"] <= leader * float(scheduler["relative_to_leader"])]
        names = {row["symbol"] for row in keep}
        retained.update(names)
        board_counts[f"S{split}"] = {"measured": len(values), "retained": len(names)}
    ordered = sorted(retained)
    atomic_text(symbols_output, "".join(f"{symbol}\n" for symbol in ordered))
    summary = {
        "schema": SCHEDULER_SCHEMA,
        "artifact_tile_k": artifact,
        "shape": list(shape),
        "manifest_sha256": sha256(manifest_path),
        "screen_symbols_sha256": sha256(screen_symbols),
        "log_sha256": sha256(log_path),
        "input_symbols": len(selected),
        "retained_symbols": len(ordered),
        "boards": board_counts,
        "symbols_sha256": sha256(symbols_output),
    }
    atomic_json(summary_output, summary)
    return summary


def candidate_from_tc(row: dict[str, Any], shape: tuple[int, int, int],
                      policy: dict[str, Any], manifest_rows: dict[str, dict[str, Any]],
                      artifact: int) -> dict[str, Any]:
    split = row["S_int"]
    reduce = reducer_us(shape[0], shape[1], split, policy)
    samples = [value + reduce for value in row["samples_list"]]
    meta = manifest_rows[row["symbol"]]
    return {
        "family": "TENSOR_CORE",
        "algorithm": "TC_S1_FULL_OUTPUT" if split == 1 else f"TC_SPLITK_S{split}_MODELED_E2E",
        "metric_scope": "FULL_OUTPUT" if split == 1 else "PRODUCER_PLUS_MODELED_REDUCER",
        "artifact_tile_k": artifact,
        "physical_layout_class": planner.layout_class(artifact),
        "symbol": row["symbol"],
        "config": tactic_name(meta),
        "split": split,
        "producer_median_us": row["us_float"],
        "modeled_reducer_us": reduce,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def candidate_from_bc(row: dict[str, Any], shape: tuple[int, int, int],
                      artifact: int) -> dict[str, Any]:
    if shape[0] >= 8:
        raise ContractError("M>=8 SIMT row cannot become a candidate")
    samples = row["samples_list"]
    return {
        "family": "SIMT_BC",
        "algorithm": f"SIMT_BC_RPW{row['rpw_int']}",
        "metric_scope": "FULL_OUTPUT",
        "artifact_tile_k": artifact,
        "physical_layout_class": planner.layout_class(artifact),
        "symbol": None,
        "config": f"bc-rpw{row['rpw_int']}-threads{row['threads']}",
        "split": 1,
        "producer_median_us": row["us_float"],
        "modeled_reducer_us": 0.0,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
        "launches": 1,
        "batch_policy": "native-grid-y-m-lt8",
    }


def metrics(candidate: dict[str, Any], shape: tuple[int, int, int],
            policy: dict[str, Any]) -> dict[str, float]:
    m, n, k = shape
    us = candidate["median_us"]
    flops = 2.0 * m * n * k
    # Q4_K is 144 resident bytes per 256 weights. Activation/output are fp16.
    distinct_bytes = m * k * 2 + n * k * 144 / 256 + m * n * 2
    return {
        "MFU_pct": flops / (us * 1e-6) / (500.0 * 1e12) * 100.0,
        "distinct_MBU_pct": distinct_bytes / (us * 1e-6) /
                            (float(policy["reducer_model"]["hbm_gbs"]) * 1e9) * 100.0,
        "distinct_bytes": distinct_bytes,
    }


def adjudicate(candidates: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    if not candidates:
        raise ContractError("cell has no measured candidate")
    ranked = sorted(candidates, key=lambda row: (row["median_us"], row["algorithm"],
                                                 row["config"]))
    winner = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    verdict = "RESOLVED"
    if runner is not None and winner["max_us"] >= runner["min_us"]:
        verdict = "UNRESOLVED_OVERLAPPING_ENVELOPES"
    return verdict, winner, runner


def finalize(plan_path: pathlib.Path, policy_path: pathlib.Path,
             raw_root: pathlib.Path, generated_root: pathlib.Path,
             output_dir: pathlib.Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    plan = json.loads(plan_path.read_text())
    planner.validate_plan(plan)
    shapes = {shape["shape_key"]: shape for shape in plan["shapes"]}
    manifests: dict[int, tuple[dict[str, dict[str, Any]], pathlib.Path]] = {}
    for artifact in planner.ARTIFACTS:
        path = generated_root / f"a{artifact}" / "manifest.json"
        rows, _ = load_manifest(path, artifact)
        manifests[artifact] = (rows, path)
    cell_results: dict[tuple[int, str], dict[str, Any]] = {}
    for cell in plan["cells"]:
        artifact = int(cell["artifact_tile_k"])
        shape_key = cell["shape_key"]
        shape_obj = shapes[shape_key]
        shape = (int(shape_obj["m"]), int(shape_obj["n"]), int(shape_obj["k"]))
        directory = raw_root / f"a{artifact}" / shape_key
        manifest_rows, manifest_path = manifests[artifact]
        candidates: list[dict[str, Any]] = []
        symbols_path = directory / "confirm-symbols.txt"
        if manifest_rows:
            symbols = read_symbols(symbols_path)
            if any(symbol not in manifest_rows for symbol in symbols):
                raise ContractError(f"{shape_key}/A{artifact}: unknown confirm symbol")
            tc, bc, _ = load_log(directory / "confirm-tc.log", artifact=artifact,
                                 expected_shape=shape, expected_symbols=symbols,
                                 expected_split=0, expected_bc_mode="skip")
            if bc:
                raise ContractError("TC confirmation unexpectedly contains BC")
            candidates.extend(candidate_from_tc(row, shape, policy, manifest_rows, artifact)
                              for row in tc if row["state"] == "MEASURED")
        bc_tc, bc_rows, _ = load_log(
            directory / "confirm-bc.log", artifact=artifact,
            expected_shape=shape, expected_symbols=[], expected_split=0,
            expected_bc_mode="only")
        if bc_tc:
            raise ContractError("BC confirmation unexpectedly contains TC")
        candidates.extend(candidate_from_bc(row, shape, artifact)
                          for row in bc_rows if row["state"] == "MEASURED")
        verdict, winner, runner = adjudicate(candidates)
        winner = dict(winner)
        winner.update(metrics(winner, shape, policy))
        result = {
            "cell_key": cell["cell_key"],
            "shape_key": shape_key,
            "shape": list(shape),
            "artifact_tile_k": artifact,
            "physical_layout_class": cell["physical_layout_class"],
            "verdict": verdict,
            "winner": winner,
            "runner_up": runner,
            "candidate_count": len(candidates),
            "manifest_sha256": sha256(manifest_path),
        }
        cell_results[(artifact, shape_key)] = result

    per_shape = []
    for shape_key, shape_obj in sorted(shapes.items()):
        # The global runner-up may be the second candidate from the same
        # ArtifactTileK as the winner.  Keeping only four per-artifact winners
        # would make overlapping confirmation envelopes look separated.
        candidates = []
        for artifact in planner.ARTIFACTS:
            result = cell_results[(artifact, shape_key)]
            candidates.append(result["winner"])
            if result["runner_up"] is not None:
                candidates.append(result["runner_up"])
        verdict, winner, runner = adjudicate(candidates)
        winner = dict(winner)
        winner.update(metrics(winner, tuple(shape_obj[key] for key in ("m", "n", "k")), policy))
        per_shape.append({
            "shape_key": shape_key,
            "shape": [shape_obj["m"], shape_obj["n"], shape_obj["k"]],
            "references": shape_obj["references"],
            "verdict": verdict,
            "winner": winner,
            "runner_up": runner,
        })

    by_family: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in per_shape:
        by_family[(row["shape"][1], row["shape"][2])].append(row)
    layout_decisions = []
    for (n, k), family_rows in sorted(by_family.items()):
        global_best = {row["shape"][0]: row["winner"]["median_us"] for row in family_rows}
        classes: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
        for artifact in planner.ARTIFACTS:
            klass = planner.layout_class(artifact)["name"]
            for row in family_rows:
                m = row["shape"][0]
                winner = cell_results[(artifact, row["shape_key"])]["winner"]
                previous = classes[klass].get(m)
                if previous is None or winner["median_us"] < previous["median_us"]:
                    classes[klass][m] = winner
        scores = []
        for name, per_m in sorted(classes.items()):
            if set(per_m) != set(planner.DECODE_M):
                scores.append({"physical_layout_class": name, "available": False,
                               "covered_m": sorted(per_m)})
                continue
            rows = []
            for m in planner.DECODE_M:
                selected = per_m[m]
                regret = selected["median_us"] / global_best[m] - 1.0
                rows.append({"M": m, "reader_artifact_tile_k": selected["artifact_tile_k"],
                             "algorithm": selected["algorithm"], "config": selected["config"],
                             "median_us": selected["median_us"], "regret": regret})
            scores.append({"physical_layout_class": name, "available": True,
                           "max_regret": max(row["regret"] for row in rows),
                           "mean_regret": statistics.mean(row["regret"] for row in rows),
                           "per_m": rows})
        available = sorted((row for row in scores if row["available"]),
                           key=lambda row: (row["max_regret"], row["mean_regret"],
                                            row["physical_layout_class"]))
        if not available:
            raise ContractError(f"family {n}x{k} has no layout covering every decode M")
        references: dict[str, dict[str, Any]] = {}
        for row in family_rows:
            for ref in row["references"]:
                key = canonical({name: ref[name] for name in (
                    "model_id", "tensor", "role", "tp_world", "tp_rank", "tp_partition")})
                references[key] = ref
        layout_decisions.append({
            "N": n, "K": k, "M_values": list(planner.DECODE_M),
            "verdict": "RESOLVED_BY_ALL_M_COVERAGE" if len(available) == 1 else "RESOLVED_MINIMAX",
            "selected": available[0],
            "runner_up": available[1] if len(available) > 1 else None,
            "all_scores": scores,
            "references": sorted(references.values(), key=canonical),
        })

    output = {
        "schema": RESULT_SCHEMA,
        "plan_sha256": sha256(plan_path),
        "policy_sha256": sha256(policy_path),
        "metric_scope": {
            "SIMT_BC": "FULL_OUTPUT_ONE_LAUNCH",
            "TC_S1": "FULL_OUTPUT",
            "TC_SPLITK": "PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH",
        },
        "cell_count": len(cell_results),
        "shape_count": len(per_shape),
        "cells": [cell_results[key] for key in sorted(cell_results)],
        "shape_winners": per_shape,
        "layout_decisions": layout_decisions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "summary.json", output)
    rows = ["shape\tM\tN\tK\tverdict\tfamily\talgorithm\tA\tconfig\tmedian_us\tMFU_pct\tdistinct_MBU_pct"]
    for row in per_shape:
        w = row["winner"]
        rows.append("\t".join(map(str, (row["shape_key"], *row["shape"], row["verdict"],
                                         w["family"], w["algorithm"], w["artifact_tile_k"],
                                         w["config"], w["median_us"], w["MFU_pct"],
                                         w["distinct_MBU_pct"]))))
    atomic_text(output_dir / "summary.tsv", "\n".join(rows) + "\n")
    registry = []
    for decision in layout_decisions:
        for ref in decision["references"]:
            registry.append({"layer": ref, "N": decision["N"], "K": decision["K"],
                             "decode_m": decision["M_values"], "layout": decision["selected"]})
    atomic_json(output_dir / "winner-registry.json", {
        "schema": "quactlize.fq_q4k_decode_winner_registry.v1", "rows": registry})
    for row in per_shape:
        models = sorted({ref["model_id"] for ref in row["references"]})
        for model in models:
            model_row = dict(row)
            model_row["references"] = [ref for ref in row["references"]
                                       if ref["model_id"] == model]
            atomic_json(output_dir / "models" / model / row["shape_key"] / "winner.json",
                        model_row)
    return output


def self_test() -> None:
    policy = load_policy(pathlib.Path(__file__).resolve().parents[1] /
                         "benchmarks/fq_q4k_decode_real_shapes_policy.json")
    value = reducer_us(4, 1024, 8, policy)
    expected = (4 * 1024 * 8 * 4 + 4 * 1024 * 2) / (.8 * 2766 * 1000)
    if not math.isclose(value, expected):
        raise AssertionError("reducer byte model differs")
    candidates = [
        {"median_us": 10., "min_us": 9.8, "max_us": 10.1,
         "algorithm": "a", "config": "a"},
        {"median_us": 11., "min_us": 10.5, "max_us": 11.2,
         "algorithm": "b", "config": "b"},
    ]
    if adjudicate(candidates)[0] != "RESOLVED":
        raise AssertionError("separated confirmation envelopes unresolved")
    candidates[1]["min_us"] = 10.0
    if adjudicate(candidates)[0] != "UNRESOLVED_OVERLAPPING_ENVELOPES":
        raise AssertionError("overlapping confirmation envelopes resolved")
    if planner.layout_class(64) != planner.layout_class(256) or \
            planner.layout_class(32) == planner.layout_class(64):
        raise AssertionError("canonical layout classes differ")
    def bc_log(m: int, state: str) -> str:
        measured = state == "MEASURED"
        samples = "[1.000000000,1.200000000]" if measured else "[]"
        us = "1.100000000" if measured else "0.000000000"
        launches = "1" if measured else "0"
        lines = [
            f"FQ_SHARD q=12 A=32 bchunk=0 shape={m}x1024x5120 "
            "typed_rows=0 selected_rows=0 only_split=1 bc_mode=all "
            "bc_batch=native-grid-y-m-lt8 iterations=2 correctness_repeats=1"
        ]
        for rpw in (1, 2, 4, 8):
            lines.append(
                f"FQ_BC_CELL q=12 A=32 shape={m}x1024x5120 rpw={rpw} "
                f"threads=256 scope=FULL_OUTPUT launches={launches} "
                f"batch_policy=native-grid-y-m-lt8 state={state} us={us} "
                f"raw_bad=0 samples={samples}")
        lines.append(
            f"FQ_SHAPE_DONE q=12 A=32 bchunk=0 shape={m}x1024x5120 "
            "typed_rows=0 selected_rows=0 only_split=1 bc_mode=all "
            "bc_batch=native-grid-y-m-lt8 iterations=2 status=PASS")
        return "\n".join(lines) + "\n"
    def tc_screen_log(*, drop: tuple[str, int] | None = None,
                      duplicate: tuple[str, int] | None = None,
                      measure_extra: tuple[str, int] | None = None) -> str:
        symbols = ("tc_alpha", "tc_beta")
        lines = [
            "FQ_SHARD q=12 A=64 bchunk=0 shape=1x1024x5120 "
            "typed_rows=2 selected_rows=2 only_split=1 bc_mode=all "
            "bc_batch=native-grid-y-m-lt8 iterations=2 correctness_repeats=1"
        ]
        for symbol in symbols:
            for split in TC_SPLITS:
                if drop == (symbol, split):
                    continue
                measured = split == 1 or measure_extra == (symbol, split)
                state = "MEASURED" if measured else "REAL_CAN_IMPLEMENT"
                us = "1.100000000" if measured else "0.000000000"
                samples = "[1.000000000,1.200000000]" if measured else "[]"
                row = (
                    "FQ_TC_CELL q=12 A=64 bchunk=0 shape=1x1024x5120 "
                    f"symbol={symbol} tm=8 tn=64 tk=256 wm=8 wn=64 stages=3 "
                    f"provider=standard-aiu S={split} scope="
                    f"{'FULL_OUTPUT' if split == 1 else 'PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS'} "
                    f"provider_capacity_rows=0 state={state} us={us} raw_bad=0 "
                    "reducer_untimed=0 failure_step=NONE failure_repeat=-1 "
                    "first_bad=18446744073709551615 first_want=0x0000 "
                    "first_got=0x0000 shipping_smem=1 split_smem=1 "
                    f"partial_bytes=0 samples={samples}")
                lines.append(row)
                if duplicate == (symbol, split):
                    lines.append(row)
        lines.append(
            "FQ_SHAPE_DONE q=12 A=64 bchunk=0 shape=1x1024x5120 "
            "typed_rows=2 selected_rows=2 only_split=1 bc_mode=all "
            "bc_batch=native-grid-y-m-lt8 iterations=2 status=PASS")
        return "\n".join(lines) + "\n"
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        path = root / "bc.log"
        path.write_text(bc_log(4, "MEASURED"))
        load_log(path, artifact=32, expected_shape=(4, 1024, 5120),
                 expected_symbols=[], expected_split=1, expected_bc_mode="all")
        path.write_text(bc_log(8, "MEASURED"))
        try:
            load_log(path, artifact=32, expected_shape=(8, 1024, 5120),
                     expected_symbols=[], expected_split=1, expected_bc_mode="all")
        except ContractError:
            pass
        else:
            raise AssertionError("M=8 SIMT negative control stayed green")
        tc_path = root / "tc.log"
        symbols = ("tc_alpha", "tc_beta")
        tc_path.write_text(tc_screen_log())
        tc_rows, _, _ = load_log(
            tc_path, artifact=64, expected_shape=(1, 1024, 5120),
            expected_symbols=symbols, expected_split=1, expected_bc_mode="all")
        if len(tc_rows) != len(symbols) * len(TC_SPLITS) or \
                sum(row["state"] == "MEASURED" for row in tc_rows) != len(symbols):
            raise AssertionError("TC census and measured denominators were mixed")
        negative_logs = (
            tc_screen_log(drop=("tc_alpha", 8)),
            tc_screen_log(duplicate=("tc_alpha", 8)),
            tc_screen_log(measure_extra=("tc_alpha", 2)),
        )
        for index, log in enumerate(negative_logs):
            tc_path.write_text(log)
            try:
                load_log(tc_path, artifact=64, expected_shape=(1, 1024, 5120),
                         expected_symbols=symbols, expected_split=1,
                         expected_bc_mode="all")
            except ContractError:
                pass
            else:
                raise AssertionError(f"TC denominator negative {index} stayed green")
    print("[fq-q4k-decode-analysis:self-test] PASS: 80%-HBM reducer bytes, "
          "confirmation-envelope fail-close, native M4 SIMT, M8 negative, "
          "complete S1/S2/S4/S8 census separated from S1 measurement, three "
          "TC denominator negatives, and canonical layout classes")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    screen = sub.add_parser("screen")
    scheduler = sub.add_parser("scheduler")
    for command in (screen, scheduler):
        command.add_argument("--manifest", type=pathlib.Path, required=True)
        command.add_argument("--log", type=pathlib.Path, required=True)
        command.add_argument("--policy", type=pathlib.Path, required=True)
        command.add_argument("--symbols-output", type=pathlib.Path, required=True)
        command.add_argument("--summary-output", type=pathlib.Path, required=True)
    scheduler.add_argument("--screen-symbols", type=pathlib.Path, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--plan", type=pathlib.Path, required=True)
    final.add_argument("--policy", type=pathlib.Path, required=True)
    final.add_argument("--raw-root", type=pathlib.Path, required=True)
    final.add_argument("--generated-root", type=pathlib.Path, required=True)
    final.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "screen":
            result = select_screen(args.manifest, args.log, args.policy,
                                   args.symbols_output, args.summary_output)
            print(f"[fq-q4k-decode-screen] PASS typed={result['typed']} "
                  f"measured={result['measured']} retained={result['retained']}")
        elif args.command == "scheduler":
            result = select_scheduler(args.manifest, args.log, args.screen_symbols,
                                      args.policy, args.symbols_output,
                                      args.summary_output)
            print(f"[fq-q4k-decode-scheduler] PASS input={result['input_symbols']} "
                  f"retained={result['retained_symbols']}")
        else:
            result = finalize(args.plan, args.policy, args.raw_root,
                              args.generated_root, args.output_dir)
            print(f"[fq-q4k-decode-final] PASS shapes={result['shape_count']} "
                  f"cells={result['cell_count']} output={args.output_dir}")
        return 0
    except (ContractError, KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        print(f"[fq-q4k-decode-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Adjudicate the complete inventory-owned Q4_K K-pack4 prefill sweep."""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import pathlib
import statistics
import sys
import tempfile
from typing import Any, Iterable

import analyze_fq_q4k_decode_real_shapes as core
import gen_fully_quantized_splitk_producer_units as generator
import plan_scalefirst_q4k_real_shapes as qplan


MAPPING_ID = "0x51344b5034540001"
MANIFEST_TYPED = 918
PREFILL_TYPED = 774
PREFILL_TM = (16, 32, 64, 128, 256)
PREFILL_M = (64, 2048, 4096)
SCHEMA = "quactlize.fq_q4k_kpack4_prefill_shape.v1"
AGGREGATE_SCHEMA = "quactlize.fq_q4k_kpack4_prefill_real_shapes.v1"
SCREEN_SCHEMA = "quactlize.fq_q4k_kpack4_prefill_real_screen.v1"
SCHEDULER_SCHEMA = "quactlize.fq_q4k_kpack4_prefill_real_scheduler.v1"
POLICY_SCHEMA = "quactlize.fq_q4k_kpack4_prefill_real_shapes_policy.v1"
KPACK4_CLASS = {
    "name": "q4-kpack4-transpose-v1",
    "mapping_id": MAPPING_ID,
    "artifact_tile_k_is_not_an_axis": True,
}
PROVIDER_COUNTS = {
    (8, "standard-aiu"): 72,
    (8, "packed-row"): 72,
    (16, "standard-aiu"): 72,
    (32, "standard-aiu"): 144,
    (64, "standard-aiu"): 210,
    (128, "standard-aiu"): 192,
    (256, "standard-aiu"): 156,
}


class PrefillRealError(ValueError):
    pass


def parse_shape(text: str) -> tuple[int, int, int]:
    shape = core.parse_shape(text)
    if shape[0] not in PREFILL_M:
        raise PrefillRealError(f"prefill M is outside {PREFILL_M}: {shape[0]}")
    return shape


def shape_text(shape: tuple[int, int, int]) -> str:
    return "x".join(map(str, shape))


def load_policy(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    exact = {
        "schema": POLICY_SCHEMA,
        "qtype": 12,
        "format": "Q4_K",
        "quant_mode": "FinegrainedScaleZero",
        "group_size": 32,
        "problem_route": "dense",
        "prefill_m": list(PREFILL_M),
        "weight_layout": KPACK4_CLASS["name"],
        "mapping_id": MAPPING_ID,
        "tile_m": list(PREFILL_TM),
        "a_provider": ["standard-aiu"],
        "scheduled_delivery_n": 0,
        "scalezero_fused": False,
        "split_k": list(core.TC_SPLITS),
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise PrefillRealError(f"real prefill policy {key} differs")
    phases = {
        "screen": {"only_split": 1, "iterations": 2,
                   "correctness_repeats": 1, "top_n": 32,
                   "relative_to_leader": 1.2,
                   "top_per_axis_value": 2},
        "scheduler": {"iterations": 1, "correctness_repeats": 1,
                      "top_n_per_board": 8,
                      "relative_to_leader": 1.05},
        "confirm": {"iterations": 7, "correctness_repeats": 2,
                    "unresolved_if_sample_envelopes_overlap": True},
    }
    for phase, expected in phases.items():
        if value.get(phase) != expected:
            raise PrefillRealError(f"real prefill policy {phase} differs")
    model = value.get("reducer_model", {})
    for key, expected in {
            "bandwidth_fraction": .8, "hbm_gbs": 2766.0,
            "launch_us": 0.0, "partial_element_bytes": 4,
            "output_element_bytes": 2}.items():
        if model.get(key) != expected:
            raise PrefillRealError(f"real prefill reducer {key} differs")
    return value


def load_manifest(path: pathlib.Path) -> tuple[
        dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    value = json.loads(path.read_text())
    if value.get("identity") != {
            "qtype": 12, "format": "Q4_K", "artifact_tile_k": 0,
            "bchunk": 0, "weight_layout": "q4-kpack4"}:
        raise PrefillRealError("real prefill manifest identity differs")
    if value.get("weight_mapping") != {
            "layout": KPACK4_CLASS["name"], "mapping_id": MAPPING_ID,
            "artifact_tile_k_is_not_an_axis": True,
            "transport_tile_k": 64, "transport_tile_n": 16}:
        raise PrefillRealError("real prefill manifest mapping differs")
    if value.get("denominator") != {
            "raw_topology_rows": 11520,
            "provider_expanded_rows": 12000,
            "source_typed_rows": MANIFEST_TYPED,
            "typed_rows": MANIFEST_TYPED,
            "selection_reject_rows": 0,
            "static_reject_rows": 11082,
            "runtime_tc_cells": 48000,
            "typed_runtime_tc_cells": 3672}:
        raise PrefillRealError("real prefill manifest denominator differs")
    rows: dict[str, dict[str, Any]] = {}
    for row in value.get("typed_rows", []):
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol or symbol in rows:
            raise PrefillRealError("real prefill manifest symbols are malformed")
        if row.get("tactic_tile_k") != 256 or row.get("artifact_tile_k") != 0:
            raise PrefillRealError("real prefill manifest carries foreign geometry")
        rows[symbol] = row
    counts = collections.Counter(
        (int(row["tile_m"]), str(row["a_provider"]))
        for row in rows.values())
    if len(rows) != MANIFEST_TYPED or dict(counts) != PROVIDER_COUNTS:
        raise PrefillRealError(
            f"real prefill provider/TM denominator differs: {dict(counts)}")
    prefill = {
        symbol: row for symbol, row in rows.items()
        if row["tile_m"] in PREFILL_TM and row["a_provider"] == "standard-aiu"
    }
    if len(prefill) != PREFILL_TYPED:
        raise PrefillRealError("real prefill AP0 denominator differs")
    return rows, prefill, value


def symbols(manifest: pathlib.Path, output: pathlib.Path) -> None:
    _, prefill, _ = load_manifest(manifest)
    core.atomic_text(output, "".join(f"{symbol}\n" for symbol in sorted(prefill)))


def fixture_contract(log: pathlib.Path, shape: tuple[int, int, int]) -> None:
    records = [core.parse_kv(line, "FQ_KPACK4_FIXTURE ")
               for line in log.read_text().splitlines()
               if line.startswith("FQ_KPACK4_FIXTURE ")]
    by_phase = {record.get("phase"): record for record in records}
    if len(records) != 2 or set(by_phase) != {"prepare", "recover"}:
        raise PrefillRealError(f"{log}: K-pack4 fixture denominator differs")
    common = {"q": "12", "shape": shape_text(shape),
              "mapping_id": MAPPING_ID, "direct_rc": "0", "abi_rc": "0",
              "direct_equal": "1"}
    prepare = by_phase["prepare"]
    if any(prepare.get(key) != expected for key, expected in {
            **common, "version": "2", "layout": "1", "bits": "4",
            "high_bits": "0", "artifact_tile_k": "0",
            "transport_tile_k": "64", "group_size": "32",
            "reserved": "0"}.items()):
        raise PrefillRealError(f"{log}: K-pack4 prepare fixture differs")
    recover = by_phase["recover"]
    if any(recover.get(key) != expected for key, expected in {
            **common, "native_equal": "1"}.items()):
        raise PrefillRealError(f"{log}: K-pack4 recover fixture differs")


def terminal_contract(row: dict[str, Any], label: str) -> None:
    if row["state"] not in core.SCHEDULER_TERMINAL_STATES or \
            row["us_float"] != 0.0 or row["raw_bad_int"] != 0 or \
            row["partial_bytes_int"] != 0 or row["samples_list"]:
        raise PrefillRealError(f"{label} carries invalid terminal data: {row}")


def load_phase(log: pathlib.Path, all_rows: dict[str, dict[str, Any]],
               prefill_rows: dict[str, dict[str, Any]],
               selected: Iterable[str], shape: tuple[int, int, int], *,
               only_split: int, bc_mode: str
               ) -> tuple[list[dict[str, Any]], dict[str, str]]:
    chosen = list(selected)
    if len(chosen) != len(set(chosen)) or any(
            symbol not in prefill_rows for symbol in chosen):
        raise PrefillRealError("real prefill symbol selection is not exact AP0")
    fixture_contract(log, shape)
    tc, bc, marker = core.load_log(
        log, artifact=0, expected_shape=shape, expected_symbols=chosen,
        expected_split=only_split, expected_bc_mode=bc_mode)
    if bc:
        raise PrefillRealError("K-pack4 real prefill emitted an Xplane BC row")
    done = [core.parse_kv(line, core.DONE_PREFIX)
            for line in log.read_text().splitlines()
            if line.startswith(core.DONE_PREFIX)]
    if len(done) != 1:
        raise PrefillRealError("real prefill lost its done marker")
    common = {
        "q": "12", "A": "0", "bchunk": "0", "shape": shape_text(shape),
        "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
        "weight_delivery_n": "0", "typed_rows": str(len(all_rows)),
        "selected_rows": str(len(chosen)), "only_split": str(only_split),
        "bc_mode": bc_mode, "bc_batch": "native-grid-y-m-lt8",
    }
    for runtime in (marker, done[0]):
        if any(runtime.get(key) != expected for key, expected in common.items()):
            raise PrefillRealError(f"real prefill marker differs: {runtime}")
    for cell in tc:
        meta = prefill_rows.get(cell.get("symbol"))
        if meta is None:
            raise PrefillRealError("real prefill emitted a non-prefill row")
        expected = {
            "tm": str(meta["tile_m"]), "tn": str(meta["tile_n"]),
            "tk": str(meta["tactic_tile_k"]), "wm": str(meta["warp_m"]),
            "wn": str(meta["warp_n"]), "stages": str(meta["stages"]),
            "provider": "standard-aiu", "provider_capacity_rows": "0",
            "scalezero_fused": "0",
        }
        if any(cell.get(key) != expected_value
               for key, expected_value in expected.items()):
            raise PrefillRealError("real prefill cell/manifest axes differ")
    return tc, marker


def choose_screen(prefill: dict[str, dict[str, Any]],
                  tc: list[dict[str, Any]], policy: dict[str, Any]) -> list[str]:
    measured = {row["symbol"]: row for row in tc
                if row["S_int"] == 1 and row["state"] == "MEASURED"}
    if not measured:
        raise PrefillRealError("real prefill screen has no measured S1 row")
    for row in tc:
        if row["state"] != "MEASURED":
            terminal_contract(row, "screen")
    ranked = sorted(measured.values(), key=lambda row: (
        row["us_float"], row["symbol"]))
    spec = policy["screen"]
    leader = ranked[0]["us_float"]
    selected = {row["symbol"] for row in ranked[:int(spec["top_n"])]}
    selected.update(row["symbol"] for row in ranked
                    if row["us_float"] <=
                    leader * float(spec["relative_to_leader"]))
    for axis in core.AXES:
        by_value: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
        for symbol, result in measured.items():
            by_value[prefill[symbol][axis]].append(result)
        for values in by_value.values():
            values.sort(key=lambda row: (row["us_float"], row["symbol"]))
            selected.update(row["symbol"] for row in
                            values[:int(spec["top_per_axis_value"])])
    return sorted(selected)


def screen(manifest: pathlib.Path, log: pathlib.Path, policy_path: pathlib.Path,
           shape: tuple[int, int, int], symbols_output: pathlib.Path,
           summary_output: pathlib.Path) -> dict[str, Any]:
    all_rows, prefill, _ = load_manifest(manifest)
    policy = load_policy(policy_path)
    tc, marker = load_phase(log, all_rows, prefill, prefill, shape,
                            only_split=1, bc_mode="all")
    if int(marker["iterations"]) != policy["screen"]["iterations"] or \
            int(marker["correctness_repeats"]) != \
            policy["screen"]["correctness_repeats"] or \
            len(tc) != PREFILL_TYPED:
        raise PrefillRealError("real prefill screen denominator differs")
    selected = choose_screen(prefill, tc, policy)
    core.atomic_text(symbols_output, "".join(f"{symbol}\n" for symbol in selected))
    result = {
        "schema": SCREEN_SCHEMA, "shape": list(shape),
        "manifest_typed": MANIFEST_TYPED, "prefill_typed": PREFILL_TYPED,
        "measured": sum(row["state"] == "MEASURED" for row in tc),
        "terminal": sum(row["state"] != "MEASURED" for row in tc),
        "retained": len(selected), "manifest_sha256": core.sha256(manifest),
        "log_sha256": core.sha256(log),
        "symbols_sha256": core.sha256(symbols_output),
    }
    core.atomic_json(summary_output, result)
    return result


def scheduler(manifest: pathlib.Path, log: pathlib.Path,
              screen_symbols: pathlib.Path, policy_path: pathlib.Path,
              shape: tuple[int, int, int], symbols_output: pathlib.Path,
              summary_output: pathlib.Path) -> dict[str, Any]:
    all_rows, prefill, _ = load_manifest(manifest)
    policy = load_policy(policy_path)
    selected = core.read_symbols(screen_symbols)
    tc, marker = load_phase(log, all_rows, prefill, selected, shape,
                            only_split=0, bc_mode="skip")
    if int(marker["iterations"]) != policy["scheduler"]["iterations"] or \
            int(marker["correctness_repeats"]) != \
            policy["scheduler"]["correctness_repeats"] or \
            len(tc) != len(selected) * len(core.TC_SPLITS):
        raise PrefillRealError("real prefill scheduler denominator differs")
    by_split: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    terminals: dict[int, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    for row in tc:
        if row["state"] == "MEASURED":
            item = dict(row)
            item["modeled_e2e_us"] = row["us_float"] + core.reducer_us(
                shape[0], shape[1], row["S_int"], policy)
            by_split[row["S_int"]].append(item)
        else:
            terminal_contract(row, "scheduler")
            terminals[row["S_int"]][row["state"]] += 1
    retained: set[str] = set()
    boards: dict[str, Any] = {}
    spec = policy["scheduler"]
    for split in core.TC_SPLITS:
        values = sorted(by_split[split], key=lambda row: (
            row["modeled_e2e_us"], row["symbol"]))
        if len(values) + sum(terminals[split].values()) != len(selected):
            raise PrefillRealError(f"real prefill S{split} denominator differs")
        if not values:
            if split == 1:
                raise PrefillRealError("real prefill lost every S1 candidate")
            boards[f"S{split}"] = {"status": "UNAVAILABLE", "measured": 0,
                                   "retained": 0,
                                   "terminal_states": dict(terminals[split])}
            continue
        leader = values[0]["modeled_e2e_us"]
        keep = values[:int(spec["top_n_per_board"])]
        keep += [row for row in values if row["modeled_e2e_us"] <=
                 leader * float(spec["relative_to_leader"])]
        names = {row["symbol"] for row in keep}
        retained.update(names)
        boards[f"S{split}"] = {"status": "AVAILABLE",
                               "measured": len(values),
                               "retained": len(names),
                               "terminal_states": dict(terminals[split])}
    ordered = sorted(retained)
    if not ordered:
        raise PrefillRealError("real prefill scheduler pruned every symbol")
    core.atomic_text(symbols_output, "".join(f"{symbol}\n" for symbol in ordered))
    result = {
        "schema": SCHEDULER_SCHEMA, "shape": list(shape),
        "input_symbols": len(selected), "retained_symbols": len(ordered),
        "boards": boards, "manifest_sha256": core.sha256(manifest),
        "screen_symbols_sha256": core.sha256(screen_symbols),
        "log_sha256": core.sha256(log),
        "symbols_sha256": core.sha256(symbols_output),
    }
    core.atomic_json(summary_output, result)
    return result


def measured_candidate(row: dict[str, Any], meta: dict[str, Any],
                       shape: tuple[int, int, int],
                       policy: dict[str, Any]) -> dict[str, Any]:
    split = row["S_int"]
    if split == 1:
        if row.get("scope") != "FULL_OUTPUT" or \
                row.get("reducer_untimed") != "0" or \
                row["partial_bytes_int"] != 0:
            raise PrefillRealError("real prefill S1 scope differs")
    else:
        expected = shape[0] * shape[1] * split * 4
        if row.get("scope") != \
                "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS" or \
                row.get("reducer_untimed") != "1" or \
                row["partial_bytes_int"] != expected:
            raise PrefillRealError(f"real prefill S{split} scope differs")
    reduce = core.reducer_us(shape[0], shape[1], split, policy)
    samples = [sample + reduce for sample in row["samples_list"]]
    return {
        "family": "TENSOR_CORE", "symbol": row["symbol"],
        "config": core.tactic_name(meta), "split": split,
        "artifact_tile_k": 0, "physical_layout_class": KPACK4_CLASS,
        "algorithm": "TC_S1_FULL_OUTPUT" if split == 1 else
                     f"TC_SPLITK_S{split}_MODELED_E2E",
        "metric_scope": "FULL_OUTPUT" if split == 1 else
            "PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH",
        "producer_median_us": row["us_float"],
        "modeled_reducer_us": reduce,
        "median_us": statistics.median(samples), "min_us": min(samples),
        "max_us": max(samples), "samples_us": samples,
    }


def rank(values: list[dict[str, Any]]) -> tuple[
        str, dict[str, Any], dict[str, Any] | None]:
    ordered = sorted(values, key=lambda row: (
        row["median_us"], row["split"], row["symbol"]))
    if not ordered:
        raise PrefillRealError("cannot rank an empty real prefill board")
    winner = ordered[0]
    runner = ordered[1] if len(ordered) > 1 else None
    verdict = "RESOLVED"
    if runner is not None and winner["max_us"] >= runner["min_us"]:
        verdict = "UNRESOLVED_OVERLAPPING_ENVELOPES"
    return verdict, winner, runner


def metrics(candidate: dict[str, Any],
            shape: tuple[int, int, int]) -> dict[str, float]:
    m, n, k = shape
    us = float(candidate["median_us"])
    flops = 2.0 * m * n * k
    distinct = m * k * 2 + n * k * 144 / 256 + m * n * 2
    return {
        "MFU_pct": flops / (us * 1e-6) / (500e12) * 100,
        "distinct_MBU_pct": distinct / (us * 1e-6) / (2766e9) * 100,
        "distinct_bytes": distinct,
    }


def finalize(manifest: pathlib.Path, log: pathlib.Path,
             symbols_path: pathlib.Path, policy_path: pathlib.Path,
             shape: tuple[int, int, int], output_json: pathlib.Path,
             output_tsv: pathlib.Path) -> dict[str, Any]:
    all_rows, prefill, manifest_value = load_manifest(manifest)
    policy = load_policy(policy_path)
    selected = core.read_symbols(symbols_path)
    tc, marker = load_phase(log, all_rows, prefill, selected, shape,
                            only_split=0, bc_mode="skip")
    if int(marker["iterations"]) != policy["confirm"]["iterations"] or \
            int(marker["correctness_repeats"]) != \
            policy["confirm"]["correctness_repeats"] or \
            len(tc) != len(selected) * len(core.TC_SPLITS):
        raise PrefillRealError("real prefill confirmation denominator differs")
    by_split: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    terminals: dict[int, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    for row in tc:
        if row["state"] == "MEASURED":
            by_split[row["S_int"]].append(measured_candidate(
                row, prefill[row["symbol"]], shape, policy))
        else:
            terminal_contract(row, "confirm")
            terminals[row["S_int"]][row["state"]] += 1
    boards: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for split in core.TC_SPLITS:
        values = by_split[split]
        if len(values) + sum(terminals[split].values()) != len(selected):
            raise PrefillRealError(f"real confirm S{split} denominator differs")
        if not values:
            if split == 1:
                raise PrefillRealError("real confirmation lost every S1 candidate")
            boards[f"S{split}"] = {"status": "UNAVAILABLE", "verdict": None,
                                   "measured": 0, "winner": None,
                                   "runner_up": None,
                                   "terminal_states": dict(terminals[split])}
            continue
        verdict, winner, runner = rank(values)
        boards[f"S{split}"] = {"status": "AVAILABLE", "verdict": verdict,
                               "measured": len(values), "winner": winner,
                               "runner_up": runner,
                               "terminal_states": dict(terminals[split])}
        candidates.extend(values)
    verdict, winner, runner = rank(candidates)
    winner = dict(winner)
    winner.update(metrics(winner, shape))
    result = {
        "schema": SCHEMA, "shape": list(shape),
        "tile_m": list(PREFILL_TM), "layout": KPACK4_CLASS,
        "scheduled_delivery_n": 0, "scalezero_fused": False,
        "manifest_typed": MANIFEST_TYPED, "prefill_typed": PREFILL_TYPED,
        "confirmed_symbols": len(selected),
        "manifest_sha256": core.sha256(manifest),
        "manifest_authority": manifest_value["authority_sha256"],
        "policy_sha256": core.sha256(policy_path),
        "log_sha256": core.sha256(log),
        "symbols_sha256": core.sha256(symbols_path),
        "reducer_model": policy["reducer_model"], "boards": boards,
        "global": {"verdict": verdict, "winner": winner,
                   "runner_up": runner},
    }
    core.atomic_json(output_json, result)
    lines = ["board\tstatus\tverdict\tmeasured\talgorithm\tS\tproducer_us\t"
             "reducer_us\te2e_us\tconfig\trunner_e2e_us\tgap_pct"]
    for name in ("S1", "S2", "S4", "S8"):
        board = boards[name]
        best, second = board["winner"], board["runner_up"]
        if best is None:
            lines.append(f"{name}\tUNAVAILABLE\tNONE\t0\tNONE\t{name[1:]}\t"
                         "NONE\tNONE\tNONE\tNONE\tNONE\tNONE")
            continue
        gap = ((second["median_us"] / best["median_us"] - 1) * 100
               if second is not None else "NONE")
        lines.append("\t".join(map(str, (
            name, board["status"], board["verdict"], board["measured"],
            best["algorithm"], best["split"], best["producer_median_us"],
            best["modeled_reducer_us"], best["median_us"], best["config"],
            second["median_us"] if second is not None else "NONE", gap))))
    core.atomic_text(output_tsv, "\n".join(lines) + "\n")
    return result


def aggregate(plan_path: pathlib.Path, raw_root: pathlib.Path,
              policy_path: pathlib.Path, output_json: pathlib.Path,
              output_tsv: pathlib.Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    qplan.validate_plan(plan)
    policy = load_policy(policy_path)
    if plan.get("shape_count") != 15 or plan.get("cell_count") != 60:
        raise PrefillRealError("real prefill plan denominator differs")
    shapes = plan["shapes"]
    if sorted({int(row["m"]) for row in shapes}) != list(PREFILL_M) or \
            len({(int(row["n"]), int(row["k"])) for row in shapes}) != 5:
        raise PrefillRealError("real prefill M/family denominator differs")
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for expected in sorted(shapes, key=lambda row: row["shape_key"]):
        path = raw_root / expected["shape_key"] / "summary.json"
        if not path.is_file():
            raise PrefillRealError(f"real prefill missing {path}")
        value = json.loads(path.read_text())
        shape = [expected["m"], expected["n"], expected["k"]]
        if value.get("schema") != SCHEMA or value.get("shape") != shape or \
                value.get("layout") != KPACK4_CLASS or \
                value.get("manifest_typed") != MANIFEST_TYPED or \
                value.get("prefill_typed") != PREFILL_TYPED or \
                value.get("policy_sha256") != core.sha256(policy_path):
            raise PrefillRealError(f"real prefill result identity differs: {path}")
        winner = value["global"]["winner"]
        if winner.get("metric_scope") not in {
                "FULL_OUTPUT",
                "PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH"}:
            raise PrefillRealError("real prefill aggregate mixed product scopes")
        results.append({"shape_key": expected["shape_key"], **value})
        rows.append({
            "shape": expected["shape_key"], "M": expected["m"],
            "N": expected["n"], "K": expected["k"],
            "verdict": value["global"]["verdict"],
            "algorithm": winner["algorithm"], "S": winner["split"],
            "producer_us": winner["producer_median_us"],
            "reducer_us": winner["modeled_reducer_us"],
            "e2e_us": winner["median_us"], "config": winner["config"],
            "MFU_pct": winner["MFU_pct"],
            "distinct_MBU_pct": winner["distinct_MBU_pct"],
        })
    document = {
        "schema": AGGREGATE_SCHEMA, "shape_count": len(results),
        "family_count": 5, "prefill_m": list(PREFILL_M),
        "layout": KPACK4_CLASS, "manifest_typed": MANIFEST_TYPED,
        "prefill_typed": PREFILL_TYPED,
        "plan_sha256": core.sha256(plan_path),
        "policy_sha256": core.sha256(policy_path),
        "reducer_model": policy["reducer_model"], "shapes": results,
    }
    core.atomic_json(output_json, document)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("FQ_KPACK4_PREFILL_SHAPE " + " ".join(
            f"{key}={value}" for key, value in row.items()))
    print("FQ_KPACK4_PREFILL_CENSUS shapes=15 families=5 "
          "M=64,2048,4096 manifest_typed=918 prefill_AP0=774")
    return document


def fixture_lines(shape: tuple[int, int, int]) -> list[str]:
    text = shape_text(shape)
    return [
        f"FQ_KPACK4_FIXTURE phase=prepare q=12 shape={text} version=2 "
        "layout=1 bits=4 high_bits=0 artifact_tile_k=0 transport_tile_k=64 "
        f"group_size=32 reserved=0 mapping_id={MAPPING_ID} direct_rc=0 "
        "abi_rc=0 direct_equal=1",
        f"FQ_KPACK4_FIXTURE phase=recover q=12 shape={text} "
        f"mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1 "
        "native_equal=1",
    ]


def synthetic_log(prefill: list[dict[str, Any]],
                  shape: tuple[int, int, int], *, only_split: int,
                  iterations: int, repeats: int, bc_mode: str) -> str:
    text = shape_text(shape)
    lines = fixture_lines(shape)
    common = (f"q=12 A=0 bchunk=0 shape={text} weight_layout=1 "
              f"weight_mapping_id={MAPPING_ID} weight_delivery_n=0 "
              f"typed_rows={MANIFEST_TYPED} selected_rows={len(prefill)} "
              f"only_split={only_split} bc_mode={bc_mode} "
              "bc_batch=native-grid-y-m-lt8 split_timing=ordered-close ")
    lines.append(f"FQ_SHARD {common}iterations={iterations} "
                 f"correctness_repeats={repeats}")
    splits = (only_split,) if only_split else core.TC_SPLITS
    for index, meta in enumerate(prefill):
        for split in splits:
            measured = split != 8
            state = "MEASURED" if measured else "SPLIT_PARTITION"
            base = 100.0 + index * .01 + split
            if measured:
                offsets = ([0.0] if iterations == 1 else
                           [-.02, .02] if iterations == 2 else
                           [-.06, -.04, -.02, 0, .02, .04, .06])
                samples = [base + offset for offset in offsets]
                us = statistics.median(samples)
                sample_text = "[" + ",".join(
                    f"{value:.9f}" for value in samples) + "]"
            else:
                us, sample_text = 0.0, "[]"
            scope = "FULL_OUTPUT" if split == 1 else \
                "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS"
            partial = shape[0] * shape[1] * split * 4 \
                if measured and split > 1 else 0
            lines.append(
                "FQ_TC_CELL q=12 A=0 bchunk=0 "
                f"shape={text} symbol={meta['symbol']} tm={meta['tile_m']} "
                f"tn={meta['tile_n']} tk={meta['tactic_tile_k']} "
                f"wm={meta['warp_m']} wn={meta['warp_n']} "
                f"stages={meta['stages']} provider=standard-aiu S={split} "
                f"scope={scope} provider_capacity_rows=0 scalezero_fused=0 "
                f"state={state} us={us:.9f} raw_bad=0 "
                f"reducer_untimed={int(measured and split > 1)} "
                "failure_step=NONE failure_repeat=-1 "
                "first_bad=18446744073709551615 first_want=0x0000 "
                "first_got=0x0000 shipping_smem=1 split_smem=1 "
                f"partial_bytes={partial} samples={sample_text}")
    lines.append(f"FQ_SHAPE_DONE {common}iterations={iterations} status=PASS")
    return "\n".join(lines) + "\n"


def synthetic_plan() -> dict[str, Any]:
    families = ((1024, 5120), (5120, 8192), (5120, 25600),
                (8192, 5120), (25600, 5120))
    cells = []
    for m in PREFILL_M:
        for ordinal, (n, k) in enumerate(families):
            cells.append({
                "qtype": 12, "format": "Q4_K", "group_size": 32,
                "inventory_status": "SUPPORTED", "problem_route": "dense",
                "grouped": None, "m": m, "n": n, "k": k,
                "model_id": "self-test", "shape_id": f"{m}-{ordinal}",
                "source_tensors": [f"tensor-{ordinal}"], "tp_world": 1,
                "tp_rank": 0, "tp_partition": "replicated",
            })
    return qplan.build_plan({"cells": cells},
                            {"minimum_m": 8,
                             "prefill_m": list(PREFILL_M)}, "0" * 64)


def self_test(policy_path: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-prefill-real-") as temporary:
        root = pathlib.Path(temporary)
        generated = root / "generated"
        generator.generate(12, 0, 0, generated, MANIFEST_TYPED, False,
                           None, "q4-kpack4")
        manifest = generated / "manifest.json"
        _, prefill, _ = load_manifest(manifest)
        initial = root / "initial.txt"
        symbols(manifest, initial)
        if len(core.read_symbols(initial)) != PREFILL_TYPED:
            raise PrefillRealError("initial symbol denominator differs")
        shape = (2048, 1024, 5120)
        screen_log = root / "screen.log"
        screen_log.write_text(synthetic_log(
            list(prefill.values()), shape, only_split=1, iterations=2,
            repeats=1, bc_mode="all"))
        shortlist = root / "screen-symbols.txt"
        screen(manifest, screen_log, policy_path, shape, shortlist,
               root / "screen.json")
        selected = [prefill[symbol] for symbol in core.read_symbols(shortlist)]
        scheduler_log = root / "scheduler.log"
        scheduler_log.write_text(synthetic_log(
            selected, shape, only_split=0, iterations=1, repeats=1,
            bc_mode="skip"))
        confirmed_path = root / "confirm-symbols.txt"
        scheduler(manifest, scheduler_log, shortlist, policy_path, shape,
                  confirmed_path, root / "scheduler.json")
        confirmed = [prefill[symbol]
                     for symbol in core.read_symbols(confirmed_path)]
        confirm_log = root / "confirm.log"
        original = synthetic_log(confirmed, shape, only_split=0, iterations=7,
                                 repeats=2, bc_mode="skip")
        confirm_log.write_text(original)
        result = finalize(manifest, confirm_log, confirmed_path, policy_path,
                          shape, root / "one.json", root / "one.tsv")
        if result["manifest_typed"] != MANIFEST_TYPED or \
                result["prefill_typed"] != PREFILL_TYPED:
            raise PrefillRealError("synthetic real prefill did not close")
        for old, new, label in (
                (MAPPING_ID, "0x0", "mapping"),
                ("raw_bad=0", "raw_bad=1", "raw-bit")):
            confirm_log.write_text(original.replace(old, new, 1))
            try:
                finalize(manifest, confirm_log, confirmed_path, policy_path,
                         shape, root / "red.json", root / "red.tsv")
            except (PrefillRealError, core.ContractError):
                pass
            else:
                raise PrefillRealError(f"real prefill {label} negative stayed green")
        plan = synthetic_plan()
        plan_path = root / "plan.json"
        core.atomic_json(plan_path, plan)
        raw = root / "raw"
        for item in plan["shapes"]:
            payload = copy.deepcopy(result)
            payload["shape"] = [item["m"], item["n"], item["k"]]
            core.atomic_json(raw / item["shape_key"] / "summary.json", payload)
        aggregate(plan_path, raw, policy_path, root / "summary.json",
                  root / "summary.tsv")
        missing = raw / plan["shapes"][0]["shape_key"] / "summary.json"
        missing.rename(missing.with_suffix(".missing"))
        try:
            aggregate(plan_path, raw, policy_path, root / "red.json",
                      root / "red.tsv")
        except PrefillRealError:
            pass
        else:
            raise PrefillRealError("real prefill missing-shape negative stayed green")
    print("[fq-kpack4-prefill-real-analysis:self-test] PASS full 918 manifest, "
          "774 AP0 prefill symbols, 15 inventory shapes, product scopes and "
          "three RED plants")


def add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--manifest", type=pathlib.Path, required=True)
    command.add_argument("--policy", type=pathlib.Path, required=True)
    command.add_argument("--shape", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sym = sub.add_parser("symbols")
    sym.add_argument("--manifest", type=pathlib.Path, required=True)
    sym.add_argument("--output", type=pathlib.Path, required=True)
    scr = sub.add_parser("screen")
    add_common(scr)
    scr.add_argument("--log", type=pathlib.Path, required=True)
    scr.add_argument("--symbols-output", type=pathlib.Path, required=True)
    scr.add_argument("--summary-output", type=pathlib.Path, required=True)
    sch = sub.add_parser("scheduler")
    add_common(sch)
    sch.add_argument("--log", type=pathlib.Path, required=True)
    sch.add_argument("--screen-symbols", type=pathlib.Path, required=True)
    sch.add_argument("--symbols-output", type=pathlib.Path, required=True)
    sch.add_argument("--summary-output", type=pathlib.Path, required=True)
    fin = sub.add_parser("finalize")
    add_common(fin)
    fin.add_argument("--log", type=pathlib.Path, required=True)
    fin.add_argument("--symbols", type=pathlib.Path, required=True)
    fin.add_argument("--output-json", type=pathlib.Path, required=True)
    fin.add_argument("--output-tsv", type=pathlib.Path, required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--plan", type=pathlib.Path, required=True)
    agg.add_argument("--raw-root", type=pathlib.Path, required=True)
    agg.add_argument("--policy", type=pathlib.Path, required=True)
    agg.add_argument("--output-json", type=pathlib.Path, required=True)
    agg.add_argument("--output-tsv", type=pathlib.Path, required=True)
    test = sub.add_parser("self-test")
    test.add_argument("--policy", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "symbols":
            symbols(args.manifest, args.output)
            print(f"[fq-kpack4-prefill-real-symbols] PASS rows={PREFILL_TYPED}")
        elif args.command == "screen":
            result = screen(args.manifest, args.log, args.policy,
                            parse_shape(args.shape), args.symbols_output,
                            args.summary_output)
            print(f"[fq-kpack4-prefill-real-screen] PASS "
                  f"measured={result['measured']} retained={result['retained']}")
        elif args.command == "scheduler":
            result = scheduler(args.manifest, args.log, args.screen_symbols,
                               args.policy, parse_shape(args.shape),
                               args.symbols_output, args.summary_output)
            print(f"[fq-kpack4-prefill-real-scheduler] PASS "
                  f"input={result['input_symbols']} "
                  f"retained={result['retained_symbols']}")
        elif args.command == "finalize":
            result = finalize(args.manifest, args.log, args.symbols,
                              args.policy, parse_shape(args.shape),
                              args.output_json, args.output_tsv)
            winner = result["global"]["winner"]
            print("[fq-kpack4-prefill-real-final] PASS "
                  f"shape={args.shape} verdict={result['global']['verdict']} "
                  f"algorithm={winner['algorithm']} config={winner['config']} "
                  f"e2e_us={winner['median_us']:.9f}")
        elif args.command == "aggregate":
            aggregate(args.plan, args.raw_root, args.policy,
                      args.output_json, args.output_tsv)
        else:
            self_test(args.policy)
        return 0
    except (PrefillRealError, core.ContractError, qplan.PlanError,
            AssertionError, KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        print(f"[fq-kpack4-prefill-real-analysis] FAIL: {error}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

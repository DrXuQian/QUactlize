#!/usr/bin/env python3
"""Prune and adjudicate the first native K-pack4 M=2048 prefill pilot."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys
import tempfile
from typing import Any, Iterable

import analyze_fq_q4k_decode_real_shapes as core
import gen_fully_quantized_splitk_producer_units as generator


SHAPE = (2048, 1024, 5120)
SHAPE_TEXT = "2048x1024x5120"
MAPPING_ID = "0x51344b5034540001"
TYPED_ROWS = 630
SOURCE_TYPED_ROWS = 2754
SCHEMA = "quactlize.fq_q4k_kpack4_prefill_pilot.v1"
SCREEN_SCHEMA = "quactlize.fq_q4k_kpack4_prefill_screen.v1"
SCHEDULER_SCHEMA = "quactlize.fq_q4k_kpack4_prefill_scheduler.v1"
POLICY_SCHEMA = "quactlize.fq_q4k_kpack4_prefill_pilot_policy.v1"
KPACK4_CLASS = {
    "name": "q4-kpack4-transpose-v1",
    "mapping_id": MAPPING_ID,
    "artifact_tile_k_is_not_an_axis": True,
}
TERMINAL_STATES = core.SCHEDULER_TERMINAL_STATES


class PrefillError(ValueError):
    pass


def load_policy(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    exact = {
        "schema": POLICY_SCHEMA,
        "qtype": 12,
        "format": "Q4_K",
        "quant_mode": "FinegrainedScaleZero",
        "group_size": 32,
        "shape": list(SHAPE),
        "weight_layout": KPACK4_CLASS["name"],
        "mapping_id": MAPPING_ID,
        "tile_m": [64],
        "a_provider": ["standard-aiu"],
        "scheduled_delivery_n": 0,
        "split_k": list(core.TC_SPLITS),
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise PrefillError(f"prefill policy {key} differs: {value.get(key)!r}")
    phases = {
        "screen": {"only_split": 1, "iterations": 2,
                   "correctness_repeats": 1, "top_n": 32,
                   "relative_to_leader": 1.2, "top_per_axis_value": 2},
        "scheduler": {"iterations": 1, "correctness_repeats": 1,
                      "top_n_per_board": 8, "relative_to_leader": 1.05,
                      "top_per_tactic_tile_k": 2},
        "confirm": {"iterations": 7, "correctness_repeats": 2,
                    "unresolved_if_sample_envelopes_overlap": True},
    }
    for name, expected in phases.items():
        if value.get(name) != expected:
            raise PrefillError(f"prefill policy {name} differs")
    model = value.get("reducer_model", {})
    if any(model.get(key) != expected for key, expected in {
            "bandwidth_fraction": .8, "hbm_gbs": 2766.0,
            "launch_us": 0.0, "partial_element_bytes": 4,
            "output_element_bytes": 2}.items()):
        raise PrefillError("prefill reducer model differs")
    return value


def load_manifest(path: pathlib.Path) -> tuple[dict[str, dict[str, Any]], dict]:
    value = json.loads(path.read_text())
    if value.get("identity") != {
            "qtype": 12, "format": "Q4_K", "artifact_tile_k": 0,
            "bchunk": 0, "tile_m_filter": 64,
            "weight_layout": "q4-kpack4",
            "kpack4_subsuperblocks": True}:
        raise PrefillError("prefill manifest identity differs")
    if value.get("weight_mapping") != {
            "layout": KPACK4_CLASS["name"], "mapping_id": MAPPING_ID,
            "artifact_tile_k_is_not_an_axis": True,
            "transport_tile_k": 64, "transport_tile_n": 16}:
        raise PrefillError("prefill manifest mapping differs")
    if value.get("denominator") != {
            "raw_topology_rows": 11520,
            "provider_expanded_rows": 12000,
            "source_typed_rows": SOURCE_TYPED_ROWS,
            "typed_rows": TYPED_ROWS,
            "selection_reject_rows": 2124,
            "static_reject_rows": 9246,
            "runtime_tc_cells": 48000,
            "typed_runtime_tc_cells": 2520}:
        raise PrefillError("prefill manifest denominator differs")
    rows: dict[str, dict[str, Any]] = {}
    for row in value.get("typed_rows", []):
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol or symbol in rows:
            raise PrefillError("prefill manifest symbols are malformed")
        if (row.get("tile_m"), row.get("a_provider"),
                row.get("a_provider_capacity_rows"),
                row.get("artifact_tile_k")) != \
                (64, "standard-aiu", None, 0) or \
                row.get("tactic_tile_k") not in (64, 128, 256):
            raise PrefillError(f"prefill manifest carries a foreign row: {row}")
        rows[symbol] = row
    if len(rows) != TYPED_ROWS:
        raise PrefillError("prefill typed row count differs")
    return rows, value


def fixture_contract(log: pathlib.Path) -> None:
    records = [core.parse_kv(line, "FQ_KPACK4_FIXTURE ")
               for line in log.read_text().splitlines()
               if line.startswith("FQ_KPACK4_FIXTURE ")]
    by_phase = {record.get("phase"): record for record in records}
    if len(records) != 2 or set(by_phase) != {"prepare", "recover"}:
        raise PrefillError(f"{log}: K-pack4 fixture denominator differs")
    prepare = by_phase["prepare"]
    if any(prepare.get(key) != expected for key, expected in {
            "q": "12", "shape": SHAPE_TEXT, "version": "2",
            "layout": "1", "bits": "4", "high_bits": "0",
            "artifact_tile_k": "0", "transport_tile_k": "64",
            "group_size": "32", "reserved": "0", "mapping_id": MAPPING_ID,
            "direct_rc": "0", "abi_rc": "0", "direct_equal": "1"}.items()):
        raise PrefillError(f"{log}: K-pack4 prepare fixture differs")
    recover = by_phase["recover"]
    if any(recover.get(key) != expected for key, expected in {
            "q": "12", "shape": SHAPE_TEXT, "mapping_id": MAPPING_ID,
            "direct_rc": "0", "abi_rc": "0", "direct_equal": "1",
            "native_equal": "1"}.items()):
        raise PrefillError(f"{log}: K-pack4 recover fixture differs")


def load_phase(log: pathlib.Path, rows: dict[str, dict[str, Any]],
               symbols: Iterable[str], *, only_split: int,
               bc_mode: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    selected = list(symbols)
    fixture_contract(log)
    tc, bc, marker = core.load_log(
        log, artifact=0, expected_shape=SHAPE, expected_symbols=selected,
        expected_split=only_split, expected_bc_mode=bc_mode)
    if bc:
        raise PrefillError("K-pack4 prefill unexpectedly emitted a BC row")
    done = [core.parse_kv(line, core.DONE_PREFIX)
            for line in log.read_text().splitlines()
            if line.startswith(core.DONE_PREFIX)]
    if len(done) != 1:
        raise PrefillError("prefill log lost its done marker")
    common = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE_TEXT,
        "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
        "weight_delivery_n": "0", "typed_rows": str(TYPED_ROWS),
        "selected_rows": str(len(selected)), "only_split": str(only_split),
        "bc_mode": bc_mode, "bc_batch": "native-grid-y-m-lt8",
    }
    for runtime in (marker, done[0]):
        if any(runtime.get(key) != expected for key, expected in common.items()):
            raise PrefillError(f"prefill runtime marker differs: {runtime}")
    for cell in tc:
        meta = rows.get(cell.get("symbol"))
        if meta is None:
            raise PrefillError("prefill log emitted an unknown symbol")
        expected = {
            "tm": str(meta["tile_m"]), "tn": str(meta["tile_n"]),
            "tk": str(meta["tactic_tile_k"]), "wm": str(meta["warp_m"]),
            "wn": str(meta["warp_n"]), "stages": str(meta["stages"]),
            "provider": "standard-aiu", "provider_capacity_rows": "0",
            "scalezero_fused": "0",
        }
        if any(cell.get(key) != value for key, value in expected.items()):
            raise PrefillError(f"prefill cell/manifest axes differ: {cell}")
    return tc, marker


def terminal_contract(row: dict[str, Any], label: str) -> None:
    if row["state"] not in TERMINAL_STATES or row["us_float"] != 0.0 or \
            row["raw_bad_int"] != 0 or row["partial_bytes_int"] != 0 or \
            row["samples_list"]:
        raise PrefillError(f"{label} carries an invalid terminal row: {row}")


def choose_screen(rows: dict[str, dict[str, Any]], tc: list[dict[str, Any]],
                  policy: dict[str, Any]) -> list[str]:
    measured = {row["symbol"]: row for row in tc
                if row["S_int"] == 1 and row["state"] == "MEASURED"}
    if not measured:
        raise PrefillError("prefill screen has no measured S1 row")
    for row in tc:
        if row["state"] != "MEASURED":
            terminal_contract(row, "screen")
    ranked = sorted(measured.values(), key=lambda row: (row["us_float"], row["symbol"]))
    spec = policy["screen"]
    leader = ranked[0]["us_float"]
    selected = {row["symbol"] for row in ranked[:int(spec["top_n"])]}
    selected.update(row["symbol"] for row in ranked
                    if row["us_float"] <= leader * float(spec["relative_to_leader"]))
    for axis in core.AXES:
        by_value: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
        for symbol, result in measured.items():
            by_value[rows[symbol][axis]].append(result)
        for values in by_value.values():
            values.sort(key=lambda row: (row["us_float"], row["symbol"]))
            selected.update(row["symbol"] for row in
                            values[:int(spec["top_per_axis_value"])])
    return sorted(selected)


def screen(manifest: pathlib.Path, log: pathlib.Path, policy_path: pathlib.Path,
           symbols_output: pathlib.Path, summary_output: pathlib.Path) -> dict[str, Any]:
    rows, _ = load_manifest(manifest)
    policy = load_policy(policy_path)
    tc, marker = load_phase(log, rows, rows, only_split=1, bc_mode="all")
    if int(marker["iterations"]) != policy["screen"]["iterations"] or \
            int(marker["correctness_repeats"]) != \
            policy["screen"]["correctness_repeats"] or \
            len([row for row in tc if row["S_int"] == 1]) != TYPED_ROWS:
        raise PrefillError("prefill screen sample/cell denominator differs")
    selected = choose_screen(rows, tc, policy)
    core.atomic_text(symbols_output, "".join(f"{symbol}\n" for symbol in selected))
    measured = sum(row["state"] == "MEASURED" for row in tc)
    result = {
        "schema": SCREEN_SCHEMA, "shape": list(SHAPE),
        "typed": TYPED_ROWS, "measured": measured,
        "terminal": TYPED_ROWS - measured, "retained": len(selected),
        "manifest_sha256": core.sha256(manifest),
        "log_sha256": core.sha256(log),
        "symbols_sha256": core.sha256(symbols_output),
    }
    core.atomic_json(summary_output, result)
    return result


def scheduler(manifest: pathlib.Path, log: pathlib.Path,
              screen_symbols: pathlib.Path, policy_path: pathlib.Path,
              symbols_output: pathlib.Path, summary_output: pathlib.Path) -> dict[str, Any]:
    rows, _ = load_manifest(manifest)
    policy = load_policy(policy_path)
    selected = core.read_symbols(screen_symbols)
    if any(symbol not in rows for symbol in selected):
        raise PrefillError("prefill screen shortlist contains a foreign symbol")
    tc, marker = load_phase(log, rows, selected, only_split=0, bc_mode="skip")
    if int(marker["iterations"]) != policy["scheduler"]["iterations"] or \
            int(marker["correctness_repeats"]) != \
            policy["scheduler"]["correctness_repeats"] or \
            len(tc) != len(selected) * len(core.TC_SPLITS):
        raise PrefillError("prefill scheduler denominator differs")
    by_split: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    terminals: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in tc:
        if row["state"] == "MEASURED":
            item = dict(row)
            item["modeled_e2e_us"] = row["us_float"] + core.reducer_us(
                SHAPE[0], SHAPE[1], row["S_int"], policy)
            by_split[row["S_int"]].append(item)
        else:
            terminal_contract(row, "scheduler")
            terminals[row["S_int"]][row["state"]] += 1
    retained: set[str] = set()
    boards: dict[str, dict[str, Any]] = {}
    spec = policy["scheduler"]
    for split in core.TC_SPLITS:
        values = sorted(by_split[split], key=lambda row: (
            row["modeled_e2e_us"], row["symbol"]))
        if len(values) + sum(terminals[split].values()) != len(selected):
            raise PrefillError(f"prefill scheduler S{split} denominator differs")
        if not values:
            if split == 1:
                raise PrefillError("prefill scheduler lost every S1 candidate")
            boards[f"S{split}"] = {"status": "UNAVAILABLE", "measured": 0,
                                   "retained": 0,
                                   "terminal_states": dict(terminals[split])}
            continue
        leader = values[0]["modeled_e2e_us"]
        keep = values[:int(spec["top_n_per_board"])]
        keep += [row for row in values if row["modeled_e2e_us"] <=
                 leader * float(spec["relative_to_leader"])]
        for tactic_tile_k in (64, 128, 256):
            per_tk = [row for row in values
                      if rows[row["symbol"]]["tactic_tile_k"] == tactic_tile_k]
            keep += per_tk[:int(spec["top_per_tactic_tile_k"])]
        names = {row["symbol"] for row in keep}
        retained.update(names)
        boards[f"S{split}"] = {"status": "AVAILABLE", "measured": len(values),
                               "retained": len(names),
                               "terminal_states": dict(terminals[split])}
    ordered = sorted(retained)
    if not ordered:
        raise PrefillError("prefill scheduler pruned every symbol")
    core.atomic_text(symbols_output, "".join(f"{symbol}\n" for symbol in ordered))
    result = {
        "schema": SCHEDULER_SCHEMA, "shape": list(SHAPE),
        "input_symbols": len(selected), "retained_symbols": len(ordered),
        "boards": boards, "manifest_sha256": core.sha256(manifest),
        "screen_symbols_sha256": core.sha256(screen_symbols),
        "log_sha256": core.sha256(log),
        "symbols_sha256": core.sha256(symbols_output),
    }
    core.atomic_json(summary_output, result)
    return result


def measured_candidate(row: dict[str, Any], meta: dict[str, Any],
                       policy: dict[str, Any]) -> dict[str, Any]:
    split = row["S_int"]
    if split == 1:
        if row.get("scope") != "FULL_OUTPUT" or row.get("reducer_untimed") != "0" or \
                row["partial_bytes_int"] != 0:
            raise PrefillError("prefill S1 scope differs")
    else:
        expected = SHAPE[0] * SHAPE[1] * split * 4
        if row.get("scope") != "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS" or \
                row.get("reducer_untimed") != "1" or \
                row["partial_bytes_int"] != expected:
            raise PrefillError(f"prefill S{split} producer scope differs")
    reduce = core.reducer_us(SHAPE[0], SHAPE[1], split, policy)
    samples = [sample + reduce for sample in row["samples_list"]]
    return {
        "family": "TENSOR_CORE", "symbol": row["symbol"],
        "config": core.tactic_name(meta), "split": split,
        "tactic_tile_k": int(meta["tactic_tile_k"]),
        "artifact_tile_k": 0, "physical_layout_class": KPACK4_CLASS,
        "algorithm": "TC_S1_FULL_OUTPUT" if split == 1 else
                     f"TC_SPLITK_S{split}_MODELED_E2E",
        "metric_scope": "FULL_OUTPUT" if split == 1 else
            "PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH",
        "producer_median_us": row["us_float"], "modeled_reducer_us": reduce,
        "median_us": statistics.median(samples), "min_us": min(samples),
        "max_us": max(samples), "samples_us": samples,
    }


def rank(values: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    ordered = sorted(values, key=lambda row: (
        row["median_us"], row["split"], row["symbol"]))
    if not ordered:
        raise PrefillError("cannot rank an empty prefill board")
    winner = ordered[0]
    runner = ordered[1] if len(ordered) > 1 else None
    verdict = "RESOLVED"
    if runner is not None and winner["max_us"] >= runner["min_us"]:
        verdict = "UNRESOLVED_OVERLAPPING_ENVELOPES"
    return verdict, winner, runner


def metrics(candidate: dict[str, Any]) -> dict[str, float]:
    m, n, k = SHAPE
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
             output_json: pathlib.Path, output_tsv: pathlib.Path) -> dict[str, Any]:
    rows, manifest_value = load_manifest(manifest)
    policy = load_policy(policy_path)
    symbols = core.read_symbols(symbols_path)
    tc, marker = load_phase(log, rows, symbols, only_split=0, bc_mode="skip")
    if int(marker["iterations"]) != policy["confirm"]["iterations"] or \
            int(marker["correctness_repeats"]) != \
            policy["confirm"]["correctness_repeats"] or \
            len(tc) != len(symbols) * len(core.TC_SPLITS):
        raise PrefillError("prefill confirmation denominator differs")
    by_split: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    terminals: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in tc:
        if row["state"] == "MEASURED":
            by_split[row["S_int"]].append(measured_candidate(
                row, rows[row["symbol"]], policy))
        else:
            terminal_contract(row, "confirm")
            terminals[row["S_int"]][row["state"]] += 1
    boards: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for split in core.TC_SPLITS:
        values = by_split[split]
        if len(values) + sum(terminals[split].values()) != len(symbols):
            raise PrefillError(f"prefill confirm S{split} denominator differs")
        if not values:
            if split == 1:
                raise PrefillError("prefill confirmation lost every S1 candidate")
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
    winner.update(metrics(winner))
    by_tactic_tile_k: dict[str, dict[str, Any]] = {}
    for tactic_tile_k in (64, 128, 256):
        values = [row for row in candidates
                  if row["tactic_tile_k"] == tactic_tile_k]
        if not values:
            raise PrefillError(
                f"confirmation lost every TK{tactic_tile_k} candidate")
        tk_verdict, tk_winner, tk_runner = rank(values)
        by_tactic_tile_k[str(tactic_tile_k)] = {
            "verdict": tk_verdict, "winner": tk_winner,
            "runner_up": tk_runner, "candidates": len(values),
        }
    result = {
        "schema": SCHEMA, "shape": list(SHAPE), "tile_m": [64],
        "layout": KPACK4_CLASS, "scheduled_delivery_n": 0,
        "scalezero_fused": False, "typed_rows": TYPED_ROWS,
        "source_typed_rows": SOURCE_TYPED_ROWS,
        "confirmed_symbols": len(symbols),
        "manifest_sha256": core.sha256(manifest),
        "manifest_authority": manifest_value["authority_sha256"],
        "policy_sha256": core.sha256(policy_path),
        "log_sha256": core.sha256(log),
        "symbols_sha256": core.sha256(symbols_path),
        "reducer_model": policy["reducer_model"], "boards": boards,
        "by_tactic_tile_k": by_tactic_tile_k,
        "global": {"verdict": verdict, "winner": winner, "runner_up": runner},
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


def fixture_lines() -> list[str]:
    return [
        f"FQ_KPACK4_FIXTURE phase=prepare q=12 shape={SHAPE_TEXT} version=2 "
        "layout=1 bits=4 high_bits=0 artifact_tile_k=0 transport_tile_k=64 "
        f"group_size=32 reserved=0 mapping_id={MAPPING_ID} direct_rc=0 "
        "abi_rc=0 direct_equal=1",
        f"FQ_KPACK4_FIXTURE phase=recover q=12 shape={SHAPE_TEXT} "
        f"mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1 "
        "native_equal=1",
    ]


def synthetic_log(rows: list[dict[str, Any]], *, only_split: int,
                  iterations: int, repeats: int, bc_mode: str) -> str:
    lines = fixture_lines()
    common = (f"q=12 A=0 bchunk=0 shape={SHAPE_TEXT} weight_layout=1 "
              f"weight_mapping_id={MAPPING_ID} weight_delivery_n=0 "
              f"typed_rows={TYPED_ROWS} selected_rows={len(rows)} "
              f"only_split={only_split} bc_mode={bc_mode} "
              "bc_batch=native-grid-y-m-lt8 split_timing=ordered-close ")
    lines.append(f"FQ_SHARD {common}iterations={iterations} "
                 f"correctness_repeats={repeats}")
    splits = (only_split,) if only_split else core.TC_SPLITS
    for index, meta in enumerate(rows):
        for split in splits:
            measured = split != 8
            state = "MEASURED" if measured else "SPLIT_PARTITION"
            base = 120.0 + index * .02 - (20 if split == 2 else
                                          45 if split == 4 else 0)
            if measured:
                offsets = ([0.0] if iterations == 1 else
                           [x * .02 for x in (-1, 1)] if iterations == 2 else
                           [x * .02 for x in (-3, -2, -1, 0, 1, 2, 3)])
                samples = [base + offset for offset in offsets]
                us = statistics.median(samples)
                sample_text = "[" + ",".join(f"{value:.9f}" for value in samples) + "]"
            else:
                us, sample_text = 0.0, "[]"
            scope = "FULL_OUTPUT" if split == 1 else \
                "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS"
            partial = SHAPE[0] * SHAPE[1] * split * 4 if measured and split > 1 else 0
            lines.append(
                "FQ_TC_CELL q=12 A=0 bchunk=0 "
                f"shape={SHAPE_TEXT} symbol={meta['symbol']} tm=64 "
                f"tn={meta['tile_n']} tk={meta['tactic_tile_k']} "
                f"wm={meta['warp_m']} wn={meta['warp_n']} stages={meta['stages']} "
                f"provider=standard-aiu S={split} scope={scope} "
                "provider_capacity_rows=0 scalezero_fused=0 "
                f"state={state} us={us:.9f} raw_bad=0 "
                f"reducer_untimed={int(measured and split > 1)} "
                "failure_step=NONE failure_repeat=-1 "
                "first_bad=18446744073709551615 first_want=0x0000 first_got=0x0000 "
                "shipping_smem=1 split_smem=1 "
                f"partial_bytes={partial} samples={sample_text}")
    lines.append(f"FQ_SHAPE_DONE {common}iterations={iterations} status=PASS")
    return "\n".join(lines) + "\n"


def self_test(policy_path: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-prefill-") as temporary:
        root = pathlib.Path(temporary)
        generated = root / "generated"
        generator.generate(12, 0, 0, generated, TYPED_ROWS, False, 64,
                           "q4-kpack4", True)
        manifest = generated / "manifest.json"
        rows, _ = load_manifest(manifest)
        screen_log = root / "screen.log"
        screen_log.write_text(synthetic_log(
            list(rows.values()), only_split=1, iterations=2, repeats=1,
            bc_mode="all"))
        screen_symbols = root / "screen-symbols.txt"
        screen(manifest, screen_log, policy_path, screen_symbols,
               root / "screen.json")
        selected = [rows[symbol] for symbol in core.read_symbols(screen_symbols)]
        scheduler_log = root / "scheduler.log"
        scheduler_log.write_text(synthetic_log(
            selected, only_split=0, iterations=1, repeats=1, bc_mode="skip"))
        confirm_symbols = root / "confirm-symbols.txt"
        scheduler(manifest, scheduler_log, screen_symbols, policy_path,
                  confirm_symbols, root / "scheduler.json")
        confirmed = [rows[symbol] for symbol in core.read_symbols(confirm_symbols)]
        confirm_log = root / "confirm.log"
        confirm_log.write_text(synthetic_log(
            confirmed, only_split=0, iterations=7, repeats=2, bc_mode="skip"))
        result = finalize(manifest, confirm_log, confirm_symbols, policy_path,
                          root / "summary.json", root / "summary.tsv")
        if result["typed_rows"] != TYPED_ROWS or \
                result["global"]["winner"]["split"] != 4 or \
                result["boards"]["S8"]["status"] != "UNAVAILABLE" or \
                set(result["by_tactic_tile_k"]) != {"64", "128", "256"}:
            raise PrefillError("synthetic prefill pilot did not close")
        original = confirm_log.read_text()
        confirm_log.write_text(original.replace(MAPPING_ID, "0x0", 1))
        try:
            finalize(manifest, confirm_log, confirm_symbols, policy_path,
                     root / "red.json", root / "red.tsv")
        except (PrefillError, core.ContractError):
            pass
        else:
            raise PrefillError("prefill mapping negative stayed green")
        lines = original.splitlines()
        index = next(i for i, line in enumerate(lines)
                     if line.startswith(core.TC_PREFIX))
        confirm_log.write_text("\n".join(lines[:index] + lines[index + 1:]) + "\n")
        try:
            finalize(manifest, confirm_log, confirm_symbols, policy_path,
                     root / "red.json", root / "red.tsv")
        except (PrefillError, core.ContractError):
            pass
        else:
            raise PrefillError("prefill missing-cell negative stayed green")
        confirm_log.write_text(original.replace("raw_bad=0", "raw_bad=1", 1))
        try:
            finalize(manifest, confirm_log, confirm_symbols, policy_path,
                     root / "red.json", root / "red.tsv")
        except (PrefillError, core.ContractError):
            pass
        else:
            raise PrefillError("prefill raw-bit negative stayed green")
    print("[fq-kpack4-prefill-analysis:self-test] PASS exact TM64=630/2754 "
          "TK64/TK128/TK256 M2048 raw-bit screen, four scheduler boards, "
          "seven-sample confirm and three RED plants")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("screen", "scheduler", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=pathlib.Path, required=True)
        command.add_argument("--log", type=pathlib.Path, required=True)
        command.add_argument("--policy", type=pathlib.Path, required=True)
        if name == "screen":
            command.add_argument("--symbols-output", type=pathlib.Path, required=True)
            command.add_argument("--summary-output", type=pathlib.Path, required=True)
        elif name == "scheduler":
            command.add_argument("--screen-symbols", type=pathlib.Path, required=True)
            command.add_argument("--symbols-output", type=pathlib.Path, required=True)
            command.add_argument("--summary-output", type=pathlib.Path, required=True)
        else:
            command.add_argument("--symbols", type=pathlib.Path, required=True)
            command.add_argument("--output-json", type=pathlib.Path, required=True)
            command.add_argument("--output-tsv", type=pathlib.Path, required=True)
    test = sub.add_parser("self-test")
    test.add_argument("--policy", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "screen":
            result = screen(args.manifest, args.log, args.policy,
                            args.symbols_output, args.summary_output)
            print(f"[fq-kpack4-prefill-screen] PASS typed={result['typed']} "
                  f"measured={result['measured']} retained={result['retained']}")
        elif args.command == "scheduler":
            result = scheduler(args.manifest, args.log, args.screen_symbols,
                               args.policy, args.symbols_output,
                               args.summary_output)
            print(f"[fq-kpack4-prefill-scheduler] PASS "
                  f"input={result['input_symbols']} retained={result['retained_symbols']}")
        elif args.command == "finalize":
            result = finalize(args.manifest, args.log, args.symbols,
                              args.policy, args.output_json, args.output_tsv)
            winner = result["global"]["winner"]
            print("[fq-kpack4-prefill-final] PASS "
                  f"verdict={result['global']['verdict']} "
                  f"algorithm={winner['algorithm']} config={winner['config']} "
                  f"e2e_us={winner['median_us']:.9f} "
                  f"MFU_pct={winner['MFU_pct']:.6f}")
        else:
            self_test(args.policy)
        return 0
    except (PrefillError, core.ContractError, AssertionError, KeyError,
            OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"[fq-kpack4-prefill-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

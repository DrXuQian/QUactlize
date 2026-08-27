#!/usr/bin/env python3
"""Prune and adjudicate the first native Q4_K K-pack4 decode pilot.

The pilot deliberately covers one real shape before the layout is admitted to
the full decode sweep.  It uses the existing conservative three-phase policy:
all 72 TM8 tactics at S1, a scheduler screen over the retained symbols, then a
seven-sample confirmation of the per-board union.  Split-K producer time is
converted to modeled product time with the registered 80%-HBM reducer model;
it is never compared to S1 as producer-only latency.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
import sys
import tempfile
from typing import Any, Iterable

import analyze_fq_q4k_decode_real_shapes as decode
import gen_fully_quantized_splitk_producer_units as generator


SHAPE = (1, 1024, 5120)
SHAPE_TEXT = "1x1024x5120"
MAPPING_ID = "0x51344b5034540001"
SCHEMA = "quactlize.fq_q4k_kpack4_pilot.v1"
TERMINAL_STATES = decode.SCHEDULER_TERMINAL_STATES


class PilotError(ValueError):
    pass


def load_manifest(path: pathlib.Path) -> tuple[dict[str, dict[str, Any]], dict]:
    value = json.loads(path.read_text())
    identity = value.get("identity", {})
    expected_identity = {
        "qtype": 12, "format": "Q4_K", "artifact_tile_k": 0,
        "bchunk": 0, "tile_m_filter": 8,
        "weight_layout": "q4-kpack4",
    }
    if identity != expected_identity:
        raise PilotError(f"K-pack4 pilot manifest identity differs: {identity}")
    if value.get("weight_mapping") != {
            "layout": "q4-kpack4-transpose-v1",
            "mapping_id": MAPPING_ID,
            "artifact_tile_k_is_not_an_axis": True,
            "transport_tile_k": 64,
            "transport_tile_n": 16}:
        raise PilotError("K-pack4 pilot mapping identity differs")
    denominator = value.get("denominator", {})
    if denominator != {
            "raw_topology_rows": 11520,
            "provider_expanded_rows": 11520,
            "source_typed_rows": 846,
            "typed_rows": 72,
            "selection_reject_rows": 774,
            "static_reject_rows": 10674,
            "runtime_tc_cells": 46080,
            "typed_runtime_tc_cells": 288}:
        raise PilotError(f"K-pack4 pilot denominator differs: {denominator}")
    rows: dict[str, dict[str, Any]] = {}
    for row in value.get("typed_rows", []):
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol or symbol in rows:
            raise PilotError("K-pack4 pilot symbol denominator is malformed")
        if (row.get("qtype"), row.get("artifact_tile_k"),
                row.get("tile_m"), row.get("warp_m"),
                row.get("a_provider"), row.get("bchunk")) != \
                (12, 0, 8, 8, "standard-aiu", 0):
            raise PilotError(f"K-pack4 pilot row carries a foreign axis: {row}")
        rows[symbol] = row
    if len(rows) != 72:
        raise PilotError(f"K-pack4 pilot typed denominator is {len(rows)}, expected 72")
    return rows, value


def fixture_contract(log: pathlib.Path) -> None:
    records = [decode.parse_kv(line, "FQ_KPACK4_FIXTURE ")
               for line in log.read_text().splitlines()
               if line.startswith("FQ_KPACK4_FIXTURE ")]
    by_phase = {record.get("phase"): record for record in records}
    if len(records) != 2 or set(by_phase) != {"prepare", "recover"}:
        raise PilotError(f"{log}: K-pack4 fixture denominator differs")
    prepare = by_phase["prepare"]
    if any(prepare.get(key) != value for key, value in {
            "q": "12", "shape": SHAPE_TEXT, "version": "2",
            "layout": "1", "bits": "4", "high_bits": "0",
            "artifact_tile_k": "0", "transport_tile_k": "64",
            "group_size": "32", "reserved": "0",
            "mapping_id": MAPPING_ID, "direct_rc": "0", "abi_rc": "0",
            "direct_equal": "1"}.items()):
        raise PilotError(f"{log}: K-pack4 prepare fixture differs")
    recover = by_phase["recover"]
    if any(recover.get(key) != value for key, value in {
            "q": "12", "shape": SHAPE_TEXT, "mapping_id": MAPPING_ID,
            "direct_rc": "0", "abi_rc": "0", "direct_equal": "1",
            "native_equal": "1"}.items()):
        raise PilotError(f"{log}: K-pack4 recovery fixture differs")


def marker_contract(log: pathlib.Path, *, selected: int,
                    only_split: int, bc_mode: str) -> None:
    fixture_contract(log)
    shard = [decode.parse_kv(line, decode.SHARD_PREFIX)
             for line in log.read_text().splitlines()
             if line.startswith(decode.SHARD_PREFIX)]
    done = [decode.parse_kv(line, decode.DONE_PREFIX)
            for line in log.read_text().splitlines()
            if line.startswith(decode.DONE_PREFIX)]
    if len(shard) != 1 or len(done) != 1:
        raise PilotError(f"{log}: shard/done denominator differs")
    common = {
        "q": "12", "A": "0", "bchunk": "0", "shape": SHAPE_TEXT,
        "weight_layout": "1", "weight_mapping_id": MAPPING_ID,
        "typed_rows": "72", "selected_rows": str(selected),
        "only_split": str(only_split), "bc_mode": bc_mode,
        "bc_batch": "native-grid-y-m-lt8",
    }
    for marker in (shard[0], done[0]):
        if any(marker.get(key) != value for key, value in common.items()):
            raise PilotError(f"{log}: runtime marker identity differs: {marker}")
    if done[0].get("status") != "PASS":
        raise PilotError(f"{log}: runtime did not close PASS")


def load_phase(log: pathlib.Path, rows: dict[str, dict[str, Any]],
               symbols: Iterable[str], *, only_split: int,
               bc_mode: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    selected = list(symbols)
    marker_contract(log, selected=len(selected), only_split=only_split,
                    bc_mode=bc_mode)
    tc, bc, marker = decode.load_log(
        log, artifact=0, expected_shape=SHAPE, expected_symbols=selected,
        expected_split=only_split, expected_bc_mode=bc_mode)
    if bc:
        raise PilotError("K-pack4 pilot unexpectedly emitted an xplane BC row")
    if any(symbol not in rows for symbol in selected):
        raise PilotError("K-pack4 pilot symbol list contains an unknown row")
    for cell in tc:
        symbol = cell.get("symbol")
        if symbol not in rows:
            raise PilotError(f"K-pack4 pilot emitted an unknown symbol: {symbol}")
        meta = rows[symbol]
        expected = {
            "q": "12", "A": "0", "bchunk": "0",
            "tm": str(meta["tile_m"]), "tn": str(meta["tile_n"]),
            "tk": str(meta["tactic_tile_k"]), "wm": str(meta["warp_m"]),
            "wn": str(meta["warp_n"]), "stages": str(meta["stages"]),
            "provider": "standard-aiu", "provider_capacity_rows": "0",
        }
        if any(cell.get(key) != value for key, value in expected.items()):
            raise PilotError(f"K-pack4 cell/manifest axes differ: {cell}")
    return tc, marker


def screen(manifest: pathlib.Path, log: pathlib.Path, policy: pathlib.Path,
           symbols_output: pathlib.Path, summary_output: pathlib.Path) -> dict:
    rows, _ = load_manifest(manifest)
    tc, marker = load_phase(log, rows, rows, only_split=1, bc_mode="all")
    policy_value = decode.load_policy(policy)
    if int(marker["iterations"]) != int(policy_value["screen"]["iterations"]) or \
            int(marker["correctness_repeats"]) != \
            int(policy_value["screen"]["correctness_repeats"]):
        raise PilotError("S1 screen sample/repeat denominator differs from policy")
    if len([row for row in tc if row["S_int"] == 1]) != len(rows):
        raise PilotError("K-pack4 S1 screen denominator differs")
    result = decode.select_screen(
        manifest, log, policy, symbols_output, summary_output)
    if result["typed"] != 72 or result["measured"] <= 0 or \
            result["retained"] <= 0 or result["retained"] > result["measured"]:
        raise PilotError(f"K-pack4 S1 screen result differs: {result}")
    return result


def scheduler(manifest: pathlib.Path, log: pathlib.Path,
              screen_symbols: pathlib.Path, policy: pathlib.Path,
              symbols_output: pathlib.Path,
              summary_output: pathlib.Path) -> dict:
    rows, _ = load_manifest(manifest)
    selected = decode.read_symbols(screen_symbols)
    tc, marker = load_phase(log, rows, selected, only_split=0, bc_mode="skip")
    policy_value = decode.load_policy(policy)
    if int(marker["iterations"]) != int(policy_value["scheduler"]["iterations"]) or \
            int(marker["correctness_repeats"]) != \
            int(policy_value["scheduler"]["correctness_repeats"]):
        raise PilotError("scheduler sample/repeat denominator differs from policy")
    if len(tc) != len(selected) * len(decode.TC_SPLITS):
        raise PilotError("K-pack4 scheduler four-board denominator differs")
    result = decode.select_scheduler(
        manifest, log, screen_symbols, policy, symbols_output, summary_output)
    if result["input_symbols"] != len(selected) or \
            result["retained_symbols"] <= 0:
        raise PilotError(f"K-pack4 scheduler result differs: {result}")
    if result["boards"].get("S8") != {
            "status": "UNAVAILABLE", "measured": 0, "retained": 0,
            "terminal_states": {"SPLIT_PARTITION": len(selected)}}:
        raise PilotError("K-pack4 pilot S8 must be the exact 20-tile partition negative")
    return result


def measured_candidate(row: dict[str, Any], meta: dict[str, Any],
                       policy: dict[str, Any]) -> dict[str, Any]:
    split = row["S_int"]
    if split == 1:
        if row.get("scope") != "FULL_OUTPUT" or \
                row.get("reducer_untimed") != "0" or \
                row["partial_bytes_int"] != 0:
            raise PilotError("S1 candidate is not full-output scope")
    else:
        expected_partial = SHAPE[0] * SHAPE[1] * split * 4
        if row.get("scope") != "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS" or \
                row.get("reducer_untimed") != "1" or \
                row["partial_bytes_int"] != expected_partial:
            raise PilotError(f"S{split} producer/reducer scope differs")
    reduce = decode.reducer_us(SHAPE[0], SHAPE[1], split, policy)
    samples = [sample + reduce for sample in row["samples_list"]]
    return {
        "symbol": row["symbol"],
        "config": decode.tactic_name(meta),
        "split": split,
        "algorithm": "TC_S1_FULL_OUTPUT" if split == 1 else
                     f"TC_SPLITK_S{split}_MODELED_E2E",
        "metric_scope": "FULL_OUTPUT" if split == 1 else
                        "PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH",
        "producer_median_us": row["us_float"],
        "modeled_reducer_us": reduce,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def rank(values: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    ordered = sorted(values, key=lambda row: (
        row["median_us"], row["split"], row["symbol"]))
    if not ordered:
        raise PilotError("cannot rank an empty candidate board")
    winner = ordered[0]
    runner = ordered[1] if len(ordered) > 1 else None
    verdict = "RESOLVED"
    if runner is not None and winner["max_us"] >= runner["min_us"]:
        verdict = "UNRESOLVED_OVERLAPPING_ENVELOPES"
    return verdict, winner, runner


def finalize(manifest: pathlib.Path, log: pathlib.Path,
             symbols_path: pathlib.Path, policy_path: pathlib.Path,
             output_json: pathlib.Path, output_tsv: pathlib.Path) -> dict:
    rows, manifest_value = load_manifest(manifest)
    symbols = decode.read_symbols(symbols_path)
    tc, marker = load_phase(log, rows, symbols, only_split=0, bc_mode="skip")
    policy = decode.load_policy(policy_path)
    if int(marker["iterations"]) != int(policy["confirm"]["iterations"]) or \
            int(marker["correctness_repeats"]) != \
            int(policy["confirm"]["correctness_repeats"]):
        raise PilotError("confirmation sample/repeat denominator differs from policy")
    by_split: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    terminal: dict[int, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    for row in tc:
        split = row["S_int"]
        if row["state"] == "MEASURED":
            by_split[split].append(measured_candidate(
                row, rows[row["symbol"]], policy))
        else:
            if row["state"] not in TERMINAL_STATES or \
                    row["us_float"] != 0.0 or row["raw_bad_int"] != 0 or \
                    row["samples_list"]:
                raise PilotError(f"S{split} has an invalid terminal row: {row}")
            terminal[split][row["state"]] += 1
    boards: dict[str, dict[str, Any]] = {}
    all_candidates: list[dict[str, Any]] = []
    for split in decode.TC_SPLITS:
        values = by_split.get(split, [])
        if len(values) + sum(terminal[split].values()) != len(symbols):
            raise PilotError(f"S{split} confirmation denominator differs")
        if not values:
            if split == 1:
                raise PilotError("confirmation lost every S1 product candidate")
            boards[f"S{split}"] = {
                "status": "UNAVAILABLE",
                "measured": 0,
                "terminal_states": dict(sorted(terminal[split].items())),
                "verdict": None, "winner": None, "runner_up": None,
            }
            continue
        verdict, winner, runner = rank(values)
        boards[f"S{split}"] = {
            "status": "AVAILABLE", "measured": len(values),
            "terminal_states": dict(sorted(terminal[split].items())),
            "verdict": verdict, "winner": winner, "runner_up": runner,
        }
        all_candidates.extend(values)
    s8 = boards["S8"]
    if s8["status"] != "UNAVAILABLE" or \
            s8["terminal_states"] != {"SPLIT_PARTITION": len(symbols)}:
        raise PilotError("S8 is not the exact structural 20-tile negative")
    verdict, winner, runner = rank(all_candidates)
    result = {
        "schema": SCHEMA,
        "shape": list(SHAPE),
        "layout": "q4-kpack4-transpose-v1",
        "weight_mapping_id": MAPPING_ID,
        "typed_rows": len(rows),
        "confirmed_symbols": len(symbols),
        "manifest_sha256": decode.sha256(manifest),
        "manifest_authority": manifest_value["authority_sha256"],
        "log_sha256": decode.sha256(log),
        "symbols_sha256": decode.sha256(symbols_path),
        "reducer_model": policy["reducer_model"],
        "boards": boards,
        "global": {"verdict": verdict, "winner": winner,
                   "runner_up": runner},
    }
    decode.atomic_json(output_json, result)
    lines = [
        "board\tstatus\tverdict\tmeasured\talgorithm\tS\tproducer_us\t"
        "reducer_us\te2e_us\tconfig\trunner_e2e_us\tgap_pct"
    ]
    for board in ("S1", "S2", "S4", "S8"):
        item = boards[board]
        best = item["winner"]
        second = item["runner_up"]
        if best is None:
            lines.append(f"{board}\tUNAVAILABLE\tNONE\t0\tNONE\t{board[1:]}\t"
                         "NONE\tNONE\tNONE\tNONE\tNONE\tNONE")
            continue
        gap = ((second["median_us"] / best["median_us"] - 1.0) * 100.0
               if second is not None else None)
        lines.append("\t".join(map(str, (
            board, item["status"], item["verdict"], item["measured"],
            best["algorithm"], best["split"], best["producer_median_us"],
            best["modeled_reducer_us"], best["median_us"], best["config"],
            second["median_us"] if second is not None else "NONE",
            gap if gap is not None else "NONE"))))
    decode.atomic_text(output_tsv, "\n".join(lines) + "\n")
    return result


def fixture_lines() -> list[str]:
    return [
        f"FQ_KPACK4_FIXTURE phase=prepare q=12 shape={SHAPE_TEXT} version=2 "
        "layout=1 bits=4 high_bits=0 artifact_tile_k=0 "
        "transport_tile_k=64 group_size=32 reserved=0 "
        f"mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1",
        f"FQ_KPACK4_FIXTURE phase=recover q=12 shape={SHAPE_TEXT} "
        f"mapping_id={MAPPING_ID} direct_rc=0 abi_rc=0 direct_equal=1 "
        "native_equal=1",
    ]


def synthetic_log(rows: list[dict[str, Any]], *, only_split: int,
                  iterations: int, repeats: int, bc_mode: str) -> str:
    lines = fixture_lines()
    lines.append(
        f"FQ_SHARD q=12 A=0 bchunk=0 shape={SHAPE_TEXT} weight_layout=1 "
        f"weight_mapping_id={MAPPING_ID} typed_rows=72 "
        f"selected_rows={len(rows)} only_split={only_split} bc_mode={bc_mode} "
        "bc_batch=native-grid-y-m-lt8 split_timing=ordered-close "
        f"iterations={iterations} correctness_repeats={repeats}")
    splits = (only_split,) if only_split else decode.TC_SPLITS
    for index, meta in enumerate(rows):
        for split in splits:
            measured = split != 8
            state = "MEASURED" if measured else "SPLIT_PARTITION"
            base_us = 20.0 + index * 0.05 - (5.0 if split == 2 else
                                             8.0 if split == 4 else 0.0)
            if measured:
                offsets = [0.0] if iterations == 1 else \
                    ([x * 0.02 for x in (-1, 1)] if iterations == 2 else
                     [x * 0.02 for x in (-3, -2, -1, 0, 1, 2, 3)])
                samples = [base_us + value for value in offsets]
                us = statistics.median(samples)
                samples_text = "[" + ",".join(f"{x:.9f}" for x in samples) + "]"
            else:
                us = 0.0
                samples_text = "[]"
            scope = ("FULL_OUTPUT" if split == 1 else
                     "PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS")
            partial = SHAPE[0] * SHAPE[1] * split * 4 if measured and split > 1 else 0
            lines.append(
                "FQ_TC_CELL q=12 A=0 bchunk=0 "
                f"shape={SHAPE_TEXT} symbol={meta['symbol']} tm={meta['tile_m']} "
                f"tn={meta['tile_n']} tk={meta['tactic_tile_k']} "
                f"wm={meta['warp_m']} wn={meta['warp_n']} stages={meta['stages']} "
                f"provider=standard-aiu S={split} scope={scope} "
                f"provider_capacity_rows=0 state={state} us={us:.9f} raw_bad=0 "
                f"reducer_untimed={int(measured and split > 1)} "
                "failure_step=NONE failure_repeat=-1 "
                "first_bad=18446744073709551615 first_want=0x0000 "
                "first_got=0x0000 shipping_smem=1 split_smem=1 "
                f"partial_bytes={partial} samples={samples_text}")
    lines.append(
        f"FQ_SHAPE_DONE q=12 A=0 bchunk=0 shape={SHAPE_TEXT} "
        f"weight_layout=1 weight_mapping_id={MAPPING_ID} typed_rows=72 "
        f"selected_rows={len(rows)} only_split={only_split} bc_mode={bc_mode} "
        "bc_batch=native-grid-y-m-lt8 split_timing=ordered-close "
        f"iterations={iterations} status=PASS")
    return "\n".join(lines) + "\n"


def self_test(policy_path: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-pilot-") as temporary:
        root = pathlib.Path(temporary)
        manifest_dir = root / "generated"
        generator.generate(12, 0, 0, manifest_dir, 4, False, 8,
                           "q4-kpack4")
        manifest = manifest_dir / "manifest.json"
        rows, _ = load_manifest(manifest)
        screen_log = root / "screen.log"
        screen_log.write_text(synthetic_log(
            list(rows.values()), only_split=1, iterations=2, repeats=1,
            bc_mode="all"))
        screen_symbols = root / "screen-symbols.txt"
        screen(manifest, screen_log, policy_path, screen_symbols,
               root / "screen.json")
        selected = [rows[symbol] for symbol in decode.read_symbols(screen_symbols)]
        scheduler_log = root / "scheduler.log"
        scheduler_log.write_text(synthetic_log(
            selected, only_split=0, iterations=1, repeats=1,
            bc_mode="skip"))
        confirm_symbols = root / "confirm-symbols.txt"
        scheduler(manifest, scheduler_log, screen_symbols, policy_path,
                  confirm_symbols, root / "scheduler.json")
        confirmed = [rows[symbol] for symbol in decode.read_symbols(confirm_symbols)]
        confirm_log = root / "confirm.log"
        confirm_log.write_text(synthetic_log(
            confirmed, only_split=0, iterations=7, repeats=2,
            bc_mode="skip"))
        result = finalize(manifest, confirm_log, confirm_symbols, policy_path,
                          root / "summary.json", root / "summary.tsv")
        if result["typed_rows"] != 72 or \
                result["boards"]["S8"]["status"] != "UNAVAILABLE" or \
                result["global"]["winner"]["split"] != 4:
            raise PilotError("synthetic K-pack4 pilot did not close")
        broken = confirm_log.read_text().replace(MAPPING_ID, "0x0", 1)
        confirm_log.write_text(broken)
        try:
            finalize(manifest, confirm_log, confirm_symbols, policy_path,
                     root / "red.json", root / "red.tsv")
        except (PilotError, decode.ContractError):
            pass
        else:
            raise PilotError("mapping-id negative control stayed green")
        confirm_log.write_text(synthetic_log(
            confirmed, only_split=0, iterations=7, repeats=2,
            bc_mode="skip"))
        lines = confirm_log.read_text().splitlines()
        drop = next(i for i, line in enumerate(lines)
                    if line.startswith(decode.TC_PREFIX))
        confirm_log.write_text("\n".join(lines[:drop] + lines[drop + 1:]) + "\n")
        try:
            finalize(manifest, confirm_log, confirm_symbols, policy_path,
                     root / "red.json", root / "red.tsv")
        except (PilotError, decode.ContractError):
            pass
        else:
            raise PilotError("missing confirmation cell stayed green")
    print("[fq-q4k-kpack4-pilot:self-test] PASS native 72-row screen, "
          "per-board scheduler union, seven-sample confirmation, 80%-HBM "
          "reducer and structural S8; mapping/missing-cell negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    self_parser = sub.add_parser("self-test")
    self_parser.add_argument("--policy", type=pathlib.Path, required=True)
    screen_parser = sub.add_parser("screen")
    scheduler_parser = sub.add_parser("scheduler")
    finalize_parser = sub.add_parser("finalize")
    for command in (screen_parser, scheduler_parser, finalize_parser):
        command.add_argument("--manifest", type=pathlib.Path, required=True)
        command.add_argument("--log", type=pathlib.Path, required=True)
        command.add_argument("--policy", type=pathlib.Path, required=True)
    screen_parser.add_argument("--symbols-output", type=pathlib.Path, required=True)
    screen_parser.add_argument("--summary-output", type=pathlib.Path, required=True)
    scheduler_parser.add_argument("--screen-symbols", type=pathlib.Path, required=True)
    scheduler_parser.add_argument("--symbols-output", type=pathlib.Path, required=True)
    scheduler_parser.add_argument("--summary-output", type=pathlib.Path, required=True)
    finalize_parser.add_argument("--symbols", type=pathlib.Path, required=True)
    finalize_parser.add_argument("--output-json", type=pathlib.Path, required=True)
    finalize_parser.add_argument("--output-tsv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test(args.policy)
        elif args.command == "screen":
            result = screen(args.manifest, args.log, args.policy,
                            args.symbols_output, args.summary_output)
            print(f"[fq-q4k-kpack4-pilot-screen] PASS typed={result['typed']} "
                  f"measured={result['measured']} retained={result['retained']}")
        elif args.command == "scheduler":
            result = scheduler(
                args.manifest, args.log, args.screen_symbols, args.policy,
                args.symbols_output, args.summary_output)
            print(f"[fq-q4k-kpack4-pilot-scheduler] PASS "
                  f"input={result['input_symbols']} "
                  f"retained={result['retained_symbols']} "
                  f"boards={json.dumps(result['boards'], sort_keys=True)}")
        else:
            result = finalize(args.manifest, args.log, args.symbols,
                              args.policy, args.output_json, args.output_tsv)
            winner = result["global"]["winner"]
            print("[fq-q4k-kpack4-pilot-final] PASS "
                  f"verdict={result['global']['verdict']} "
                  f"algorithm={winner['algorithm']} "
                  f"config={winner['config']} e2e_us={winner['median_us']:.9f} "
                  f"confirmed={result['confirmed_symbols']}")
        return 0
    except (OSError, KeyError, json.JSONDecodeError, PilotError,
            decode.ContractError, ValueError) as error:
        print(f"[fq-q4k-kpack4-pilot] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed analysis for Q4_K K-pack4 resident-delivery A/B/C.

The offline bytes, tactic, split count and A provider stay fixed.  The only
axis is the compile-time maximum resident N delivered by one matched AIU/TSM
pair: auto64, D32 or D16.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import statistics
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_fq_q4k_kpack4_xplane_isomorphic_ab as base  # noqa: E402


SCHEMA = "quactlize.fq-q4k-kpack4-delivery-ab.v1"
ACU_SCHEMA = "quactlize.fq-q4k-kpack4-delivery-acu.v1"
SHAPE_KEY = "m1_n8192_k5120"
SHAPE = (1, 8192, 5120)
DELIVERIES = (0, 32, 16)
DELIVERY_NAMES = {0: "auto64", 32: "d32", 16: "d16"}
PROVIDERS = (0, 1)


class AnalysisError(ValueError):
    pass


def arm_name(ap: int, delivery: int) -> str:
    return f"kpack4-ap{ap}-{DELIVERY_NAMES[delivery]}"


def load_delivery_run(path: pathlib.Path, arm: dict[str, Any], ap: int,
                      delivery: int, iterations: int) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    expected = str(delivery)
    for prefix in ("FQ_SHARD ", "FQ_SHAPE_DONE "):
        marker = base.exactly_one(text, prefix)
        if marker.get("weight_delivery_n") != expected:
            raise AnalysisError(
                f"{path}: {prefix.strip()} delivery cap differs: {marker}")
    row = base.load_run(path, arm, SHAPE, iterations)
    if arm.get("name") != f"kpack4-ap{ap}":
        raise AnalysisError(f"{path}: provider arm identity differs")
    return row


def load_codegen(path: pathlib.Path, ap: int, delivery: int) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "quactlize.fq-q4k-kpack4-xplane-codegen.v1" or \
            value.get("arm") != f"kpack4-ap{ap}" or \
            value.get("delivery_cap_n") != delivery:
        raise AnalysisError(f"codegen identity differs: {path}")
    focus = value.get("focus_counts", {})
    if focus.get("mma", 0) <= 0 or focus.get("tsm_load", 0) <= 0:
        raise AnalysisError(f"codegen lost MMA/TSM load: {path}")
    return value


def resolved_winner(rows: dict[int, dict[str, Any]], threshold: float) -> tuple[int, str]:
    ordered = sorted(DELIVERIES, key=lambda value: rows[value]["median_us"])
    winner, runner = ordered[:2]
    paired = [
        win / other - 1.0
        for win, other in zip(rows[winner]["run_medians_us"],
                              rows[runner]["run_medians_us"])
    ]
    gap = rows[runner]["median_us"] / rows[winner]["median_us"] - 1.0
    stable = all(value < 0 for value in paired)
    verdict = "RESOLVED" if stable and gap >= threshold else \
        "UNRESOLVED_OVERLAPPING_ROUNDS"
    return winner, verdict


def analyze(master_path: pathlib.Path, runs_root: pathlib.Path,
            codegen_root: pathlib.Path, iterations: int, rounds: int,
            threshold: float, output_json: pathlib.Path,
            output_tsv: pathlib.Path) -> None:
    arms = base.load_master(master_path)
    if set(arms) != set(base.ARMS):
        raise AnalysisError("master denominator differs")
    results = []
    for ap in PROVIDERS:
        manifest_arm = arms[f"kpack4-ap{ap}"]
        delivery_rows: dict[int, dict[str, Any]] = {}
        for delivery in DELIVERIES:
            samples: list[float] = []
            medians: list[float] = []
            resources: set[tuple[int, int, int]] = set()
            for round_index in range(1, rounds + 1):
                path = (runs_root / f"ap{ap}" /
                        f"round-{round_index}-{DELIVERY_NAMES[delivery]}.log")
                row = load_delivery_run(
                    path, manifest_arm, ap, delivery, iterations)
                samples.extend(row["samples"])
                medians.append(row["median_us"])
                resources.add((row["shipping_smem"], row["split_smem"],
                               row["partial_bytes"]))
            if len(resources) != 1:
                raise AnalysisError(
                    f"runtime resource identity drifted: AP{ap}/D{delivery}")
            delivery_rows[delivery] = {
                "delivery_cap_n": delivery,
                "delivery": DELIVERY_NAMES[delivery],
                "median_us": statistics.median(samples),
                "min_us": min(samples), "max_us": max(samples),
                "samples": len(samples), "run_medians_us": medians,
                "resources": list(next(iter(resources))),
                "codegen": load_codegen(
                    codegen_root / f"ap{ap}-{DELIVERY_NAMES[delivery]}.json",
                    ap, delivery),
            }
        if len({tuple(row["resources"]) for row in delivery_rows.values()}) != 1:
            raise AnalysisError(f"shared/workspace ABI changed across AP{ap} deliveries")
        auto = delivery_rows[0]
        for delivery in (32, 16):
            row = delivery_rows[delivery]
            row["delta_vs_auto"] = row["median_us"] / auto["median_us"] - 1.0
            row["paired_vs_auto"] = [
                candidate / control - 1.0
                for control, candidate in zip(auto["run_medians_us"],
                                              row["run_medians_us"])
            ]
        auto["delta_vs_auto"] = 0.0
        auto["paired_vs_auto"] = [0.0] * rounds
        winner, verdict = resolved_winner(delivery_rows, threshold)
        ordered = sorted(DELIVERIES,
                         key=lambda value: delivery_rows[value]["median_us"])
        runner = ordered[1]
        results.append({
            "a_provider": f"AP{ap}", "shape": list(SHAPE),
            "winner_delivery_cap_n": winner,
            "winner": DELIVERY_NAMES[winner], "verdict": verdict,
            "runner": DELIVERY_NAMES[runner],
            "gap": (delivery_rows[runner]["median_us"] /
                    delivery_rows[winner]["median_us"] - 1.0),
            "deliveries": [delivery_rows[value] for value in DELIVERIES],
        })
    value = {
        "schema": SCHEMA, "shape_key": SHAPE_KEY, "shape": list(SHAPE),
        "config": base.CONFIG, "split": base.SPLIT,
        "iterations_per_run": iterations, "rounds": rounds,
        "material_threshold": threshold, "providers": results,
    }
    output_json.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    lines = [
        "provider\tdelivery_cap_n\tdelivery\tmedian_us\trange_us\t"
        "delta_vs_auto_pct\tpaired_vs_auto_pct\tinstructions\tregisters\t"
        "spill\ttsm_load\tmma\tshipping_smem\tsplit_smem\tpartial_bytes"
    ]
    for provider in results:
        for row in provider["deliveries"]:
            codegen = row["codegen"]
            focus = codegen["focus_counts"]
            lines.append("\t".join((
                provider["a_provider"], str(row["delivery_cap_n"]),
                row["delivery"], f"{row['median_us']:.9f}",
                f"[{row['min_us']:.9f},{row['max_us']:.9f}]",
                f"{100 * row['delta_vs_auto']:.6f}",
                ",".join(f"{100 * item:.6f}"
                         for item in row["paired_vs_auto"]),
                str(codegen["instruction_total"]),
                str(codegen["registers"] if codegen["registers"] is not None
                    else "UNKNOWN"), str(codegen["spill_status"]),
                str(focus["tsm_load"]), str(focus["mma"]),
                *(str(item) for item in row["resources"]),
            )))
        print("FQ_KPACK4_DELIVERY_VERDICT "
              f"provider={provider['a_provider']} verdict={provider['verdict']} "
              f"winner={provider['winner']} runner={provider['runner']} "
              f"gap_pct={100 * provider['gap']:.6f}")
    output_tsv.write_text("\n".join(lines) + "\n")
    print(f"[fq-kpack4-delivery-analysis] PASS providers=2 arms=6 "
          f"output={output_json}")


def numeric(value: str) -> float | None:
    text = value.strip().replace(",", "").replace("%", "")
    if not text or text.lower() in {"n/a", "na", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_acu_details(path: pathlib.Path) -> list[dict[str, Any]]:
    text = path.read_text(errors="replace")
    stripped = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[dict[str, str]] = []
    if stripped and all(line.startswith("{") for line in stripped):
        records = [json.loads(line) for line in stripped]
    else:
        start = next((index for index, line in enumerate(text.splitlines())
                      if "Metric Name" in line and "Metric Value" in line), -1)
        if start < 0:
            raise AnalysisError(f"{path}: ACU details header is missing")
        records = list(csv.DictReader(text.splitlines()[start:]))
    result: list[dict[str, Any]] = []
    occurrences: dict[tuple[str, str, str], int] = {}
    for record in records:
        section = str(record.get("Section Name", "")).strip()
        metric = str(record.get("Metric Name", "")).strip()
        unit = str(record.get("Metric Unit", "")).strip()
        value = numeric(str(record.get("Metric Value", "")))
        if not metric or value is None:
            continue
        basic = (section, metric, unit)
        occurrence = occurrences.get(basic, 0)
        occurrences[basic] = occurrence + 1
        result.append({"section": section, "metric": metric, "unit": unit,
                       "occurrence": occurrence, "value": value})
    if not result:
        raise AnalysisError(f"{path}: ACU details exposed no numeric metrics")
    return result


def analyze_acu(index_path: pathlib.Path, output_json: pathlib.Path,
                output_tsv: pathlib.Path) -> None:
    with index_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {(f"AP{ap}", DELIVERY_NAMES[delivery])
                for ap in PROVIDERS for delivery in DELIVERIES}
    actual = {(row.get("provider"), row.get("delivery")) for row in rows}
    if actual != expected or len(rows) != 6:
        raise AnalysisError(f"ACU arm denominator differs: {sorted(actual)}")
    parsed: dict[tuple[str, str], dict[tuple[str, str, str, int], float]] = {}
    for row in rows:
        key = (str(row["provider"]), str(row["delivery"]))
        metrics = parse_acu_details(pathlib.Path(row["details"]))
        raw_path = str(row.get("raw", "NONE"))
        if raw_path != "NONE":
            raw_metrics = parse_acu_details(pathlib.Path(raw_path))
            metrics.extend({**item, "section": "RAW/" + item["section"]}
                           for item in raw_metrics)
        parsed[key] = {
            (item["section"], item["metric"], item["unit"], item["occurrence"]):
                item["value"] for item in metrics
        }
    output_rows = []
    for ap in PROVIDERS:
        provider = f"AP{ap}"
        common = set.intersection(*(
            set(parsed[(provider, DELIVERY_NAMES[value])])
            for value in DELIVERIES))
        for key in sorted(common):
            values = [parsed[(provider, DELIVERY_NAMES[value])][key]
                      for value in DELIVERIES]
            auto = values[0]
            deltas = [None if auto == 0 else value / auto - 1.0
                      for value in values]
            searchable = " ".join(key[:3]).lower()
            highlight = any(token in searchable for token in (
                "bank conflict", "shared load", "duration", "active cycles",
                "no eligible", "warp cycles per"))
            output_rows.append({
                "provider": provider, "section": key[0], "metric": key[1],
                "unit": key[2], "occurrence": key[3],
                "auto64": values[0], "d32": values[1], "d16": values[2],
                "d32_delta": deltas[1], "d16_delta": deltas[2],
                "highlight": highlight,
            })
    value = {"schema": ACU_SCHEMA, "rows": output_rows}
    output_json.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    lines = [
        "provider\tsection\tmetric\tunit\toccurrence\tauto64\td32\td16\t"
        "d32_delta_pct\td16_delta_pct\thighlight"
    ]
    for row in output_rows:
        lines.append("\t".join((
            row["provider"], row["section"], row["metric"], row["unit"],
            str(row["occurrence"]), str(row["auto64"]), str(row["d32"]),
            str(row["d16"]),
            "NA" if row["d32_delta"] is None else
                f"{100 * row['d32_delta']:.6f}",
            "NA" if row["d16_delta"] is None else
                f"{100 * row['d16_delta']:.6f}",
            str(int(row["highlight"])),
        )))
        if row["highlight"]:
            print("FQ_KPACK4_DELIVERY_ACU "
                  f"provider={row['provider']} section={json.dumps(row['section'])} "
                  f"metric={json.dumps(row['metric'])} unit={row['unit']} "
                  f"auto64={row['auto64']} d32={row['d32']} d16={row['d16']} "
                  f"d32_delta_pct={'NA' if row['d32_delta'] is None else 100 * row['d32_delta']} "
                  f"d16_delta_pct={'NA' if row['d16_delta'] is None else 100 * row['d16_delta']}")
    output_tsv.write_text("\n".join(lines) + "\n")
    print(f"[fq-kpack4-delivery-acu] PASS arms=6 common_rows="
          f"{len(output_rows)} output={output_json}")


def synthetic_master(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    arms = []
    for name in base.ARMS:
        kpack = name.startswith("kpack4")
        ap = int(name.endswith("ap1"))
        artifact = 0 if kpack else 64
        provider = "packed-row" if ap else "standard-aiu"
        arms.append({
            "schema": base.ARM_SCHEMA, "name": name,
            "layout": "q4-kpack4" if kpack else "xplane",
            "weight_layout": int(kpack), "artifact_tile_k": artifact,
            "a_provider": provider, "a_provider_id": ap,
            "selection_denominator": 1, "source_typed_denominator": 144,
            "source_global_typed_denominator": 918,
            "row": {"qtype": 12, "artifact_tile_k": artifact,
                    "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
                    "warp_m": 8, "warp_n": 16, "stages": 2,
                    "bchunk": 0, "a_provider": provider,
                    "symbol": f"symbol_{name}"},
        })
    value = {
        "schema": "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab.v1",
        "axes": {"qtype": 12, "tile_m": 8, "tile_n": 64,
                 "tactic_tile_k": 256, "warp_m": 8, "warp_n": 16,
                 "stages": 2, "bchunk": 0, "split": 4},
        "arms": arms,
    }
    path.write_text(json.dumps(value) + "\n")
    return {row["name"]: row for row in arms}


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-delivery-") as temp:
        root = pathlib.Path(temp)
        master, runs, codegen = root / "master.json", root / "runs", root / "codegen"
        arms = synthetic_master(master)
        codegen.mkdir()
        for ap in PROVIDERS:
            directory = runs / f"ap{ap}"
            directory.mkdir(parents=True)
            arm = arms[f"kpack4-ap{ap}"]
            for delivery, median in ((0, 10.0), (32, 9.5), (16, 9.8)):
                name = DELIVERY_NAMES[delivery]
                (codegen / f"ap{ap}-{name}.json").write_text(json.dumps({
                    "schema": "quactlize.fq-q4k-kpack4-xplane-codegen.v1",
                    "arm": f"kpack4-ap{ap}", "delivery_cap_n": delivery,
                    "instruction_total": 100 + delivery,
                    "registers": 80, "spill_status": "ZERO",
                    "focus_counts": {"mma": 16, "tsm_load": 16},
                }) + "\n")
                for round_index in range(1, 4):
                    samples = [median - .1, median, median + .1]
                    common = f"q=12 A=0 bchunk=0 shape=1x8192x5120"
                    text = (
                        f"FQ_SHARD {common} weight_layout=1 "
                        f"weight_mapping_id={base.MAPPING_ID} "
                        f"weight_delivery_n={delivery} typed_rows=1 "
                        "selected_rows=1 only_split=4 bc_mode=skip "
                        "iterations=3 correctness_repeats=8\n"
                        f"FQ_TC_CELL {common} symbol={arm['row']['symbol']} "
                        "tm=8 tn=64 tk=256 wm=8 wn=16 stages=2 "
                        f"provider={arm['a_provider']} S=4 "
                        "scope=PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS "
                        f"provider_capacity_rows={ap} state=MEASURED "
                        f"us={median:.9f} raw_bad=0 reducer_untimed=1 "
                        "failure_step=NONE failure_repeat=-1 shipping_smem=1024 "
                        "split_smem=2048 partial_bytes=131072 "
                        f"samples=[{','.join(str(item) for item in samples)}]\n"
                        f"FQ_SHAPE_DONE {common} weight_layout=1 "
                        f"weight_mapping_id={base.MAPPING_ID} "
                        f"weight_delivery_n={delivery} typed_rows=1 "
                        "selected_rows=1 only_split=4 bc_mode=skip "
                        "iterations=3 status=PASS\n")
                    (directory / f"round-{round_index}-{name}.log").write_text(text)
        analyze(master, runs, codegen, 3, 3, .02,
                root / "summary.json", root / "summary.tsv")
        result = json.loads((root / "summary.json").read_text())
        assert len(result["providers"]) == 2
        assert all(row["winner"] == "d32" and row["verdict"] == "RESOLVED"
                   for row in result["providers"])
        victim = runs / "ap0" / "round-1-d32.log"
        victim.write_text(victim.read_text().replace(
            "weight_delivery_n=32", "weight_delivery_n=64"))
        try:
            analyze(master, runs, codegen, 3, 3, .02,
                    root / "red.json", root / "red.tsv")
        except AnalysisError:
            pass
        else:
            raise AssertionError("delivery marker negative stayed green")

        details = root / "details"
        details.mkdir()
        index = root / "acu-index.tsv"
        index.write_text("provider\tdelivery\tarm\treport\tdetails\traw\n")
        with index.open("a") as handle:
            for ap in PROVIDERS:
                for delivery, value in ((0, 100.0), (32, 70.0), (16, 60.0)):
                    name = DELIVERY_NAMES[delivery]
                    path = details / f"ap{ap}-{name}.csv"
                    path.write_text(
                        "Section Name,Metric Name,Metric Unit,Metric Value\n"
                        f"Shared Memory,Bank Conflicts,conflict,{value}\n"
                        f"Speed Of Light,Duration,nsecond,{value * 2}\n")
                    handle.write(f"AP{ap}\t{name}\t{arm_name(ap, delivery)}\t"
                                 f"unused\t{path}\tNONE\n")
        analyze_acu(index, root / "acu.json", root / "acu.tsv")
        acu = json.loads((root / "acu.json").read_text())
        assert len(acu["rows"]) == 4 and all(row["highlight"] for row in acu["rows"])
    print("[fq-kpack4-delivery-analysis:self-test] PASS exact 2x3x3 timing "
          "denominator, delivery-marker RED and six-arm ACU parser")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("analyze")
    run.add_argument("--master", type=pathlib.Path, required=True)
    run.add_argument("--runs-root", type=pathlib.Path, required=True)
    run.add_argument("--codegen-root", type=pathlib.Path, required=True)
    run.add_argument("--iterations", type=int, required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--threshold", type=float, default=.02)
    run.add_argument("--output-json", type=pathlib.Path, required=True)
    run.add_argument("--output-tsv", type=pathlib.Path, required=True)
    acu = sub.add_parser("acu")
    acu.add_argument("--index", type=pathlib.Path, required=True)
    acu.add_argument("--output-json", type=pathlib.Path, required=True)
    acu.add_argument("--output-tsv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "analyze":
            if args.iterations <= 0 or args.rounds <= 0 or \
                    not 0 < args.threshold < 1:
                raise AnalysisError("iterations/rounds/threshold are invalid")
            analyze(args.master, args.runs_root, args.codegen_root,
                    args.iterations, args.rounds, args.threshold,
                    args.output_json, args.output_tsv)
        else:
            analyze_acu(args.index, args.output_json, args.output_tsv)
    except (AnalysisError, base.AnalysisError, OSError, ValueError,
            KeyError, AssertionError, json.JSONDecodeError) as exc:
        print(f"[fq-kpack4-delivery-analysis] FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

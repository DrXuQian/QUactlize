#!/usr/bin/env python3
"""Analyze a matched K-pack4 D32 plain versus fused scale/zero A/B.

The weight layout, A provider, tactic, Split-K count and resident-B delivery
remain fixed.  The only axis is whether the packed Q4_K metadata decoder
publishes separate fp16 scale/zero planes or one interleaved fp16x2 plane.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_fq_q4k_kpack4_xplane_isomorphic_ab as base  # noqa: E402
import analyze_fq_q4k_kpack4_delivery_ab as delivery  # noqa: E402


SCHEMA = "quactlize.fq-q4k-kpack4-scalezero-ab.v2"
ACU_SCHEMA = "quactlize.fq-q4k-kpack4-scalezero-acu.v2"
SHAPE = (1, 8192, 5120)
VARIANTS = ("plain", "store", "load")
PROVIDERS = (0, 1)
DELIVERY_N = 32


class AnalysisError(ValueError):
    pass


def arm_name(ap: int, variant: str) -> str:
    return f"kpack4-ap{ap}-{variant}"


def load_run(path: pathlib.Path, arm: dict[str, Any], ap: int,
             fused: bool, fused_read: bool, iterations: int) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    for prefix in ("FQ_SHARD ", "FQ_SHAPE_DONE "):
        marker = base.exactly_one(text, prefix)
        if marker.get("weight_delivery_n") != str(DELIVERY_N):
            raise AnalysisError(f"{path}: {prefix.strip()} is not D32")
    cell = base.exactly_one(text, "FQ_TC_CELL ")
    if cell.get("scalezero_fused") != str(int(fused)):
        raise AnalysisError(f"{path}: collective fused marker differs")
    if cell.get("scalezero_fused_read") != str(int(fused_read)):
        raise AnalysisError(f"{path}: collective fused-read marker differs")
    row = base.load_run(path, arm, SHAPE, iterations)
    if arm.get("name") != f"kpack4-ap{ap}":
        raise AnalysisError(f"{path}: provider arm identity differs")
    return row


def load_codegen(path: pathlib.Path, ap: int, fused: bool,
                 fused_read: bool) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "quactlize.fq-q4k-kpack4-xplane-codegen.v1" or \
            value.get("arm") != f"kpack4-ap{ap}" or \
            value.get("delivery_cap_n") != DELIVERY_N or \
            value.get("scalezero_fused") is not fused or \
            value.get("scalezero_fused_read") is not fused_read:
        raise AnalysisError(f"codegen identity differs: {path}")
    focus = value.get("focus_counts", {})
    if focus.get("mma", 0) <= 0 or focus.get("tsm_load", 0) <= 0:
        raise AnalysisError(f"codegen lost MMA/TSM load: {path}")
    return value


def timing_verdict(store: dict[str, Any], load: dict[str, Any],
                   threshold: float) -> str:
    paired = [candidate / control - 1.0 for control, candidate in zip(
        store["run_medians_us"], load["run_medians_us"])]
    delta = load["median_us"] / store["median_us"] - 1.0
    if all(value < 0 for value in paired) and delta <= -threshold:
        return "RESOLVED_FUSED_LOAD_FASTER"
    if all(value > 0 for value in paired) and delta >= threshold:
        return "RESOLVED_FUSED_LOAD_SLOWER"
    return "UNRESOLVED_OVERLAPPING_ROUNDS"


def analyze(master_path: pathlib.Path, runs_root: pathlib.Path,
            codegen_root: pathlib.Path, iterations: int, rounds: int,
            threshold: float, output_json: pathlib.Path,
            output_tsv: pathlib.Path) -> None:
    arms = base.load_master(master_path)
    if set(arms) != set(base.ARMS):
        raise AnalysisError("master denominator differs")
    providers = []
    for ap in PROVIDERS:
        manifest_arm = arms[f"kpack4-ap{ap}"]
        rows: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            fused = variant != "plain"
            fused_read = variant == "load"
            samples: list[float] = []
            medians: list[float] = []
            resources: set[tuple[int, int, int]] = set()
            for round_index in range(1, rounds + 1):
                row = load_run(
                    runs_root / f"ap{ap}" / f"round-{round_index}-{variant}.log",
                    manifest_arm, ap, fused, fused_read, iterations)
                samples.extend(row["samples"])
                medians.append(row["median_us"])
                resources.add((row["shipping_smem"], row["split_smem"],
                               row["partial_bytes"]))
            if len(resources) != 1:
                raise AnalysisError(f"runtime resource drifted: AP{ap}/{variant}")
            rows[variant] = {
                "variant": variant, "fused": fused,
                "fused_read": fused_read,
                "median_us": statistics.median(samples),
                "min_us": min(samples), "max_us": max(samples),
                "samples": len(samples), "run_medians_us": medians,
                "resources": list(next(iter(resources))),
                "codegen": load_codegen(
                    codegen_root / f"ap{ap}-{variant}.json", ap, fused,
                    fused_read),
            }
        if len({tuple(rows[name]["resources"]) for name in VARIANTS}) != 1:
            raise AnalysisError(f"shared/workspace ABI changed across AP{ap}")
        plain, store, load = rows["plain"], rows["store"], rows["load"]
        for row in rows.values():
            row["delta_vs_plain"] = row["median_us"] / plain["median_us"] - 1.0
            row["paired_vs_plain"] = [
                candidate / control - 1.0 for control, candidate in zip(
                    plain["run_medians_us"], row["run_medians_us"])]
        load["delta_vs_store"] = load["median_us"] / store["median_us"] - 1.0
        load["paired_vs_store"] = [
            candidate / control - 1.0 for control, candidate in zip(
                store["run_medians_us"], load["run_medians_us"])]
        verdict = timing_verdict(store, load, threshold)
        winner = min(VARIANTS, key=lambda name: rows[name]["median_us"])
        providers.append({
            "a_provider": f"AP{ap}", "shape": list(SHAPE),
            "delivery_cap_n": DELIVERY_N, "verdict": verdict,
            "winner": winner, "variants": [rows[name] for name in VARIANTS],
        })
        print("FQ_KPACK4_SCALEZERO_VERDICT "
              f"provider=AP{ap} verdict={verdict} winner={winner} "
              f"load_vs_store_delta_pct={100 * load['delta_vs_store']:.6f} "
              f"load_vs_plain_delta_pct={100 * load['delta_vs_plain']:.6f}")
    value = {
        "schema": SCHEMA, "shape": list(SHAPE), "config": base.CONFIG,
        "split": base.SPLIT, "delivery_cap_n": DELIVERY_N,
        "iterations_per_run": iterations, "rounds": rounds,
        "material_threshold": threshold, "providers": providers,
    }
    output_json.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    lines = [
        "provider\tvariant\tmedian_us\trange_us\tdelta_vs_plain_pct\t"
        "delta_vs_store_pct\tpaired_vs_plain_pct\tpaired_vs_store_pct\t"
        "instructions\tregisters\tspill\ttsm_load\t"
        "mma\tshipping_smem\tsplit_smem\tpartial_bytes"
    ]
    for provider in providers:
        for row in provider["variants"]:
            codegen = row["codegen"]
            focus = codegen["focus_counts"]
            lines.append("\t".join((
                provider["a_provider"], row["variant"],
                f"{row['median_us']:.9f}",
                f"[{row['min_us']:.9f},{row['max_us']:.9f}]",
                f"{100 * row['delta_vs_plain']:.6f}",
                "NA" if "delta_vs_store" not in row else
                f"{100 * row['delta_vs_store']:.6f}",
                ",".join(f"{100 * item:.6f}"
                         for item in row["paired_vs_plain"]),
                "NA" if "paired_vs_store" not in row else
                ",".join(f"{100 * item:.6f}"
                         for item in row["paired_vs_store"]),
                str(codegen["instruction_total"]),
                str(codegen["registers"] if codegen["registers"] is not None
                    else "UNKNOWN"), str(codegen["spill_status"]),
                str(focus["tsm_load"]), str(focus["mma"]),
                *(str(item) for item in row["resources"]),
            )))
    output_tsv.write_text("\n".join(lines) + "\n")
    print("[fq-kpack4-scalezero-analysis] PASS providers=2 arms=6 "
          f"output={output_json}")


def analyze_acu(index_path: pathlib.Path, output_json: pathlib.Path,
                output_tsv: pathlib.Path) -> None:
    with index_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {(f"AP{ap}", variant)
                for ap in PROVIDERS for variant in VARIANTS}
    actual = {(row.get("provider"), row.get("variant")) for row in rows}
    if actual != expected or len(rows) != 6:
        raise AnalysisError(f"ACU arm denominator differs: {sorted(actual)}")
    parsed: dict[tuple[str, str], dict[tuple[str, str, str, int], float]] = {}
    for row in rows:
        key = (str(row["provider"]), str(row["variant"]))
        metrics = delivery.parse_acu_details(pathlib.Path(row["details"]))
        raw_path = str(row.get("raw", "NONE"))
        if raw_path != "NONE":
            raw_metrics = delivery.parse_acu_details(pathlib.Path(raw_path))
            metrics.extend({**item, "section": "RAW/" + item["section"]}
                           for item in raw_metrics)
        parsed[key] = {
            (item["section"], item["metric"], item["unit"], item["occurrence"]):
                item["value"] for item in metrics
        }
    output_rows = []
    conflicts = {f"AP{ap}": 0 for ap in PROVIDERS}
    conflict_verdicts = []
    for ap in PROVIDERS:
        provider = f"AP{ap}"
        common = (set(parsed[(provider, "plain")]) &
                  set(parsed[(provider, "store")]) &
                  set(parsed[(provider, "load")]))
        shared_load_conflicts = [key for key in common
                                 if "bank conflict" in " ".join(key[:3]).lower()
                                 and key[3] == 0
                                 and not key[0].startswith("RAW/")]
        if len(shared_load_conflicts) != 1:
            raise AnalysisError(
                f"{provider}: Shared Load bank-conflict denominator differs: "
                f"{shared_load_conflicts}")
        conflict_key = shared_load_conflicts[0]
        conflict_values = {
            variant: parsed[(provider, variant)][conflict_key]
            for variant in VARIANTS
        }
        if conflict_values["load"] < conflict_values["store"]:
            conflict_verdict = "SHARED_LOAD_CONFLICT_REDUCED"
        elif conflict_values["load"] > conflict_values["store"]:
            conflict_verdict = "SHARED_LOAD_CONFLICT_INCREASED"
        else:
            conflict_verdict = "SHARED_LOAD_CONFLICT_UNCHANGED"
        conflict_verdicts.append({
            "provider": provider, "verdict": conflict_verdict,
            "plain": conflict_values["plain"],
            "store": conflict_values["store"],
            "load": conflict_values["load"],
        })
        print("FQ_KPACK4_SCALEZERO_CONFLICT "
              f"provider={provider} verdict={conflict_verdict} "
              f"plain={conflict_values['plain']} store={conflict_values['store']} "
              f"load={conflict_values['load']}")
        for key in sorted(common):
            plain = parsed[(provider, "plain")][key]
            store = parsed[(provider, "store")][key]
            load = parsed[(provider, "load")][key]
            delta_plain = None if plain == 0 else load / plain - 1.0
            delta_store = None if store == 0 else load / store - 1.0
            searchable = " ".join(key[:3]).lower()
            is_conflict = "bank conflict" in searchable
            if is_conflict:
                conflicts[provider] += 1
            highlight = is_conflict or any(token in searchable for token in (
                "shared load", "shared store", "duration", "active cycles",
                "no eligible", "warp cycles per"))
            output_rows.append({
                "provider": provider, "section": key[0], "metric": key[1],
                "unit": key[2], "occurrence": key[3], "plain": plain,
                "store": store, "load": load,
                "delta_plain": delta_plain, "delta_store": delta_store,
                "highlight": highlight,
                "bank_conflict": is_conflict,
            })
    if any(count == 0 for count in conflicts.values()):
        raise AnalysisError(f"ACU exposed no common bank-conflict row: {conflicts}")
    output_json.write_text(json.dumps({
        "schema": ACU_SCHEMA, "bank_conflict_rows": conflicts,
        "conflict_verdicts": conflict_verdicts, "rows": output_rows,
    }, indent=2, sort_keys=True) + "\n")
    lines = [
        "provider\tsection\tmetric\tunit\toccurrence\tplain\tstore\tload\t"
        "load_vs_plain_pct\tload_vs_store_pct\tbank_conflict\thighlight"
    ]
    for row in output_rows:
        lines.append("\t".join((
            row["provider"], row["section"], row["metric"], row["unit"],
            str(row["occurrence"]), str(row["plain"]), str(row["store"]),
            str(row["load"]),
            "NA" if row["delta_plain"] is None else f"{100 * row['delta_plain']:.6f}",
            "NA" if row["delta_store"] is None else f"{100 * row['delta_store']:.6f}",
            str(int(row["bank_conflict"])), str(int(row["highlight"])),
        )))
        if row["highlight"]:
            print("FQ_KPACK4_SCALEZERO_ACU "
                  f"provider={row['provider']} section={json.dumps(row['section'])} "
                  f"metric={json.dumps(row['metric'])} unit={row['unit']} "
                  f"plain={row['plain']} store={row['store']} load={row['load']} "
                  f"load_vs_plain_pct={'NA' if row['delta_plain'] is None else 100 * row['delta_plain']} "
                  f"load_vs_store_pct={'NA' if row['delta_store'] is None else 100 * row['delta_store']}")
    output_tsv.write_text("\n".join(lines) + "\n")
    print("[fq-kpack4-scalezero-acu] PASS arms=6 "
          f"bank_conflict_rows={conflicts} output={output_json}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-scalezero-") as temp:
        root = pathlib.Path(temp)
        master, runs, codegen = root / "master.json", root / "runs", root / "codegen"
        arms = delivery.synthetic_master(master)
        codegen.mkdir()
        for ap in PROVIDERS:
            directory = runs / f"ap{ap}"
            directory.mkdir(parents=True)
            arm = arms[f"kpack4-ap{ap}"]
            for variant, median in (("plain", 10.0), ("store", 9.9),
                                    ("load", 9.5)):
                fused = variant != "plain"
                fused_read = variant == "load"
                (codegen / f"ap{ap}-{variant}.json").write_text(json.dumps({
                    "schema": "quactlize.fq-q4k-kpack4-xplane-codegen.v1",
                    "arm": f"kpack4-ap{ap}", "delivery_cap_n": DELIVERY_N,
                    "scalezero_fused": fused,
                    "scalezero_fused_read": fused_read,
                    "instruction_total": 100, "registers": 80,
                    "spill_status": "ZERO",
                    "focus_counts": {"mma": 16, "tsm_load": 16},
                }) + "\n")
                for round_index in range(1, 5):
                    samples = [median - .1, median, median + .1]
                    common = "q=12 A=0 bchunk=0 shape=1x8192x5120"
                    text = (
                        f"FQ_SHARD {common} weight_layout=1 "
                        f"weight_mapping_id={base.MAPPING_ID} "
                        f"weight_delivery_n={DELIVERY_N} typed_rows=1 "
                        "selected_rows=1 only_split=4 bc_mode=skip "
                        "iterations=3 correctness_repeats=8\n"
                        f"FQ_TC_CELL {common} symbol={arm['row']['symbol']} "
                        "tm=8 tn=64 tk=256 wm=8 wn=16 stages=2 "
                        f"provider={arm['a_provider']} S=4 "
                        "scope=PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS "
                        f"provider_capacity_rows={ap} "
                        f"scalezero_fused={int(fused)} "
                        f"scalezero_fused_read={int(fused_read)} state=MEASURED "
                        f"us={median:.9f} raw_bad=0 reducer_untimed=1 "
                        "failure_step=NONE failure_repeat=-1 shipping_smem=1024 "
                        "split_smem=2048 partial_bytes=131072 "
                        f"samples=[{','.join(str(item) for item in samples)}]\n"
                        f"FQ_SHAPE_DONE {common} weight_layout=1 "
                        f"weight_mapping_id={base.MAPPING_ID} "
                        f"weight_delivery_n={DELIVERY_N} typed_rows=1 "
                        "selected_rows=1 only_split=4 bc_mode=skip "
                        "iterations=3 status=PASS\n")
                    (directory / f"round-{round_index}-{variant}.log").write_text(text)
        analyze(master, runs, codegen, 3, 4, .02,
                root / "summary.json", root / "summary.tsv")
        result = json.loads((root / "summary.json").read_text())
        assert all(row["verdict"] == "RESOLVED_FUSED_LOAD_FASTER"
                   for row in result["providers"])
        victim = runs / "ap0" / "round-1-load.log"
        victim.write_text(victim.read_text().replace(
            "weight_delivery_n=32", "weight_delivery_n=64"))
        try:
            analyze(master, runs, codegen, 3, 4, .02,
                    root / "red.json", root / "red.tsv")
        except AnalysisError:
            pass
        else:
            raise AssertionError("D32 marker negative stayed green")

        details = root / "details"
        details.mkdir()
        index = root / "acu-index.tsv"
        index.write_text("provider\tvariant\tarm\treport\tdetails\traw\n")
        with index.open("a") as handle:
            for ap in PROVIDERS:
                for variant, value in (("plain", 100.0), ("store", 90.0),
                                       ("load", 50.0)):
                    path = details / f"ap{ap}-{variant}.csv"
                    path.write_text(
                        "Section Name,Metric Name,Metric Unit,Metric Value\n"
                        f"Shared Memory,Bank Conflicts,conflict,{value}\n"
                        f"Speed Of Light,Duration,nsecond,{2 * value}\n")
                    handle.write(f"AP{ap}\t{variant}\t{arm_name(ap, variant)}\t"
                                 f"unused\t{path}\tNONE\n")
        analyze_acu(index, root / "acu.json", root / "acu.tsv")
        broken = index.read_text().replace("Bank Conflicts", "Other", 1)
        # The index points at files, so plant the actual first details file.
        first = details / "ap0-plain.csv"
        first.write_text(first.read_text().replace("Bank Conflicts", "Other"))
        try:
            analyze_acu(index, root / "red-acu.json", root / "red-acu.tsv")
        except AnalysisError:
            pass
        else:
            raise AssertionError("missing bank-conflict denominator stayed green")
        assert broken  # keep the text-level plant live under static analysis
    print("[fq-kpack4-scalezero-analysis:self-test] PASS exact 2x3x4 timing, "
          "D32/raw-bit/fused-read markers RED and six-arm bank-conflict ACU denominator")


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
            if args.iterations <= 0 or args.rounds != 4 or \
                    not 0 < args.threshold < 1:
                raise AnalysisError("scalezero A/B requires positive iterations and four rounds")
            analyze(args.master, args.runs_root, args.codegen_root,
                    args.iterations, args.rounds, args.threshold,
                    args.output_json, args.output_tsv)
        else:
            analyze_acu(args.index, args.output_json, args.output_tsv)
    except (AnalysisError, base.AnalysisError, delivery.AnalysisError, OSError,
            ValueError, KeyError, AssertionError, json.JSONDecodeError) as exc:
        print(f"[fq-kpack4-scalezero-analysis] FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

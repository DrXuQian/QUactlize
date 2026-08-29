#!/usr/bin/env python3
"""Adjudicate the 15-shape persistent ScaleFirst Xplane/K-pack4 A/B."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_scalefirst_q4k_kpack4_prefill_ab as base  # noqa: E402
import select_scalefirst_q4k_kpack4_prefill_real_shapes as selector  # noqa: E402


SCHEMA = "quactlize.scalefirst-q4k-kpack4-prefill-real-result.v2"
CONFIGS = tuple(
    f"{tm}x{tn}x{tk}_w{wm}x{wn}_s{stages}_bc0"
    for tm, tn, tk, wm, wn, stages in selector.CANDIDATES)


def bind_authority() -> None:
    # The causal two-shape analyzer owns the parser and paired-grid semantics.
    # This process replaces only its finite shape/config authorities; no
    # parsing, timing, fingerprint or verdict rule is forked.
    base.SHAPES = selector.SHAPES
    base.CONFIGS = CONFIGS


def enrich(result: dict, threshold_pct: float) -> dict:
    if len(result.get("shape_comparisons", [])) != 15:
        raise base.AnalysisError("real prefill lost one shape")
    grouped: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for row in result["shape_comparisons"]:
        m, n, k = map(int, row["shape"])
        grouped[(n, k)].append(row)
    if set(grouped) != set(selector.FAMILIES):
        raise base.AnalysisError("real prefill family denominator differs")
    families = []
    for (n, k), rows in sorted(grouped.items()):
        if {int(row["shape"][0]) for row in rows} != set(selector.PREFILL_M):
            raise base.AnalysisError(f"family {n}x{k} lost one M")
        deltas = [float(row["best_vs_best_delta_pct"]) for row in rows]
        families.append({
            "N": n, "K": k, "M_values": list(selector.PREFILL_M),
            "max_kpack4_regression_pct": max(deltas),
            "mean_kpack4_delta_pct": sum(deltas) / len(deltas),
            "within_threshold": max(deltas) <= threshold_pct,
        })
    worst = max(float(row["best_vs_best_delta_pct"])
                for row in result["shape_comparisons"])
    result.update({
        "schema": SCHEMA,
        "scope": "PERSISTENT_SCALEFIRST_FULL_OUTPUT_REAL_SHAPES",
        "shape_count": 15,
        "family_count": 5,
        "prefill_m": list(selector.PREFILL_M),
        "families": families,
        "worst_kpack4_regression_pct": worst,
        "shipping_prefill_verdict": (
            "KPACK4_PREFILL_WITHIN_THRESHOLD" if worst <= threshold_pct
            else "KPACK4_PREFILL_HOLD"),
    })
    return result


def analyze(runs: pathlib.Path, rounds: int, iterations: int,
            threshold_pct: float) -> dict:
    bind_authority()
    return enrich(base.analyze(runs, rounds, iterations, threshold_pct),
                  threshold_pct)


def emit(result: dict, output_json: pathlib.Path,
         output_tsv: pathlib.Path) -> None:
    base.emit(result, output_json, output_tsv)
    family_tsv = output_tsv.with_name("family-summary.tsv")
    lines = ["N\tK\tM_values\tmax_kpack4_regression_pct\t"
             "mean_kpack4_delta_pct\twithin_threshold"]
    for row in result["families"]:
        lines.append("\t".join(map(str, (
            row["N"], row["K"], ",".join(map(str, row["M_values"])),
            row["max_kpack4_regression_pct"],
            row["mean_kpack4_delta_pct"], row["within_threshold"]))))
        print("SF_KPACK4_PREFILL_FAMILY "
              f"N={row['N']} K={row['K']} "
              f"max_regression_pct={row['max_kpack4_regression_pct']:.6f} "
              f"mean_delta_pct={row['mean_kpack4_delta_pct']:.6f} "
              f"within_threshold={int(row['within_threshold'])}")
    family_tsv.write_text("\n".join(lines) + "\n")
    print("SF_KPACK4_PREFILL_VERDICT "
          f"verdict={result['shipping_prefill_verdict']} "
          f"shapes=15 families=5 "
          f"worst_regression_pct={result['worst_kpack4_regression_pct']:.6f} "
          f"threshold_pct={result['threshold_pct']:.6f}")


def fixture_values() -> dict:
    bind_authority()
    values = {}
    for shape_index, shape in enumerate(selector.SHAPES):
        for arm in base.ARMS:
            cells = {}
            for config_index, config in enumerate(CONFIGS):
                for grid in (72, 144):
                    us = 20.0 + shape_index + config_index + grid / 1000
                    if arm == "q4-kpack4":
                        us *= 1.01
                    cells[(config, grid)] = [us, us + .1]
            values[(arm, shape)] = cells
    return values


def self_test() -> None:
    values = fixture_values()
    result = enrich(base.summarize(values, 3.0), 3.0)
    if result["shipping_prefill_verdict"] != \
            "KPACK4_PREFILL_WITHIN_THRESHOLD" or \
            len(result["families"]) != 5:
        raise AssertionError("real prefill positive differs")
    broken = copy.deepcopy(values)
    del broken[("q4-kpack4", selector.SHAPES[-1])]
    try:
        base.summarize(broken, 3.0)
    except (base.AnalysisError, KeyError):
        pass
    else:
        raise AssertionError("missing-shape negative stayed green")
    with tempfile.TemporaryDirectory(prefix="qz-sf-kpack4-real-analysis-") as temp:
        root = pathlib.Path(temp)
        emit(result, root / "summary.json", root / "summary.tsv")
        if len((root / "summary.tsv").read_text().splitlines()) != 16 or \
                len((root / "family-summary.tsv").read_text().splitlines()) != 6:
            raise AssertionError("real prefill output denominator differs")
        shape = selector.SHAPES[0]
        shape_text = "x".join(map(str, shape))
        for arm, identity in base.ARMS.items():
            lines = [
                "SF_SHARD qtype=12 "
                f"artifact_tile_k={identity['artifact']} bchunk=0 "
                "typed_rows=7 selected_rows=7 "
                f"weight_layout={identity['weight_layout']} "
                f"weight_mapping_id={identity['mapping']} "
                "algorithm_mask=0x2 iterations=2"
            ]
            for config in CONFIGS:
                for sample, us in enumerate((70.0, 70.2)):
                    lines.append("SF_CELL " + json.dumps({
                        "shape": shape_text, "qtype": 12,
                        "artifact_tile_k": identity["artifact"], "bchunk": 0,
                        "config": config, "algorithm": "PERSISTENT",
                        "metric_scope": "FULL_OUTPUT", "split": 1,
                        "grid": 144, "status": "MEASURED", "sample": sample,
                        "sample_us": us, "raw_bad": 0,
                        "fingerprint": "0x1234",
                    }, separators=(",", ":")))
            lines.append(
                f"SF_COMPLETE status=COMPLETE shape={shape_text} "
                "typed_rows=7 iterations=2 roundtrip=PASS")
            path = root / f"{arm}.log"
            path.write_text("\n".join(lines) + "\n")
            parsed = base.parse_log(path, arm, shape, 2)
            if len(parsed["cells"]) != 7 or parsed["fingerprint"] != "0x1234":
                raise AssertionError("seven-row runtime parser differs")
    print("[sf-kpack4-prefill-real-analysis:self-test] PASS 15 shapes, "
          "five family minimax rows, ship/hold verdict and missing-shape RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("analyze")
    run.add_argument("--runs", type=pathlib.Path, required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--iterations", type=int, required=True)
    run.add_argument("--threshold-pct", type=float, default=3.0)
    run.add_argument("--output-json", type=pathlib.Path, required=True)
    run.add_argument("--output-tsv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            if args.rounds <= 0 or args.iterations <= 0 or \
                    not 0 < args.threshold_pct < 100:
                raise base.AnalysisError("rounds/iterations/threshold invalid")
            result = analyze(args.runs, args.rounds, args.iterations,
                             args.threshold_pct)
            emit(result, args.output_json, args.output_tsv)
        return 0
    except (AssertionError, base.AnalysisError, KeyError, OSError,
            TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"[sf-kpack4-prefill-real-analysis] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze native N64 versus four-way N16 K-pack4 AIU delivery."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_fq_q4k_kpack4_xplane_isomorphic_ab as base  # noqa: E402


SCHEMA = "quactlize.fq-q4k-n16-delivery-ab.v1"
PLAN_SCHEMA = "quactlize.fq-q4k-n16-delivery-ab-plan.v1"
CASES = (
    ("m1_n8192_k5120", (1, 8192, 5120), "N_WIDE_TARGET"),
    ("m1_n5120_k8192", (1, 5120, 8192), "BALANCED_CONTROL"),
    ("m1_n5120_k25600", (1, 5120, 25600), "K_HEAVY_CONTROL"),
)


class AnalysisError(ValueError):
    pass


def emit_plan(path: pathlib.Path) -> None:
    value = {
        "schema": PLAN_SCHEMA, "config": base.CONFIG, "split": base.SPLIT,
        "cases": [{"shape_key": key, "shape": list(shape), "role": role}
                  for key, shape, role in CASES],
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(f"[fq-n16-delivery-plan] PASS shapes=3 comparisons=6 output={path}")


def validate_plan(value: dict[str, Any]) -> None:
    expected = {key: (shape, role) for key, shape, role in CASES}
    actual = {row["shape_key"]: (tuple(row["shape"]), row["role"])
              for row in value.get("cases", [])}
    if value.get("schema") != PLAN_SCHEMA or \
            value.get("config") != base.CONFIG or \
            value.get("split") != base.SPLIT or actual != expected:
        raise AnalysisError("N16 delivery plan identity differs")


def load(path: pathlib.Path, arm: dict[str, Any],
         shape: tuple[int, int, int], iterations: int,
         delivery: str) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    for prefix in ("FQ_SHARD ", "FQ_SHAPE_DONE "):
        marker = base.exactly_one(text, prefix)
        if marker.get("weight_delivery") != delivery:
            raise AnalysisError(f"{path}: weight delivery identity differs")
    return base.load_run(path, arm, shape, iterations)


def classify(delta: float, stable: bool, threshold: float) -> str:
    if stable and delta <= -threshold:
        return "N16_DELIVERY_GAIN"
    if stable and delta >= threshold:
        return "N16_DELIVERY_REGRESSION"
    if abs(delta) < threshold:
        return "NO_MATERIAL_DELIVERY_EFFECT"
    return "UNRESOLVED_NOISY"


def analyze(master_path: pathlib.Path, plan_path: pathlib.Path,
            runs_root: pathlib.Path, iterations: int, rounds: int,
            threshold: float, output_json: pathlib.Path,
            output_tsv: pathlib.Path) -> None:
    arms = base.load_master(master_path)
    validate_plan(json.loads(plan_path.read_text()))
    comparisons = []
    for shape_key, shape, role in CASES:
        for ap in (0, 1):
            arm_x = arms[f"xplane-ap{ap}"]
            arm_k = arms[f"kpack4-ap{ap}"]
            all_rows: dict[str, dict[str, Any]] = {}
            for name, arm, delivery in (
                    ("xplane", arm_x, "native"),
                    ("kpack4-native", arm_k, "native"),
                    ("kpack4-n16", arm_k, "n16")):
                samples: list[float] = []
                medians: list[float] = []
                resources: set[tuple[int, int, int]] = set()
                for r in range(1, rounds + 1):
                    path = runs_root / shape_key / f"ap{ap}" / \
                        f"round-{r}-{name}.log"
                    row = load(path, arm, shape, iterations, delivery)
                    samples.extend(row["samples"])
                    medians.append(row["median_us"])
                    resources.add((row["shipping_smem"], row["split_smem"],
                                   row["partial_bytes"]))
                if len(resources) != 1:
                    raise AnalysisError(
                        f"resource identity drifted: {shape_key}/AP{ap}/{name}")
                all_rows[name] = {
                    "median_us": statistics.median(samples),
                    "min_us": min(samples), "max_us": max(samples),
                    "run_medians_us": medians, "samples": len(samples),
                    "resources": list(next(iter(resources))),
                }
            native, n16 = all_rows["kpack4-native"], all_rows["kpack4-n16"]
            if native["resources"] != n16["resources"]:
                raise AnalysisError(
                    f"K-pack4 shared/partial ABI changed: {shape_key}/AP{ap}")
            paired = [candidate / control - 1.0 for control, candidate in zip(
                native["run_medians_us"], n16["run_medians_us"])]
            stable = all(x > 0 for x in paired) or all(x < 0 for x in paired)
            delta = n16["median_us"] / native["median_us"] - 1.0
            gap = n16["median_us"] / all_rows["xplane"]["median_us"] - 1.0
            comparisons.append({
                "shape_key": shape_key, "shape": list(shape), "role": role,
                "a_provider": f"AP{ap}", "xplane": all_rows["xplane"],
                "kpack4_native": native, "kpack4_n16": n16,
                "n16_delta": delta, "n16_gap_vs_xplane": gap,
                "paired_run_deltas": paired, "same_sign": stable,
                "verdict": classify(delta, stable, threshold),
            })
    result = {
        "schema": SCHEMA, "config": base.CONFIG, "split": base.SPLIT,
        "iterations_per_run": iterations, "rounds": rounds,
        "material_threshold": threshold, "comparisons": comparisons,
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "shape\trole\tprovider\txplane_us\tkpack4_native_us\t"
        "kpack4_n16_us\tn16_delta_pct\tn16_gap_vs_xplane_pct\t"
        "paired_deltas_pct\tverdict"
    ]
    for row in comparisons:
        lines.append("\t".join((
            row["shape_key"], row["role"], row["a_provider"],
            f"{row['xplane']['median_us']:.9f}",
            f"{row['kpack4_native']['median_us']:.9f}",
            f"{row['kpack4_n16']['median_us']:.9f}",
            f"{100 * row['n16_delta']:.6f}",
            f"{100 * row['n16_gap_vs_xplane']:.6f}",
            ",".join(f"{100 * x:.6f}" for x in row["paired_run_deltas"]),
            row["verdict"],
        )))
        print("FQ_N16_DELIVERY_AB "
              f"shape={row['shape_key']} role={row['role']} "
              f"provider={row['a_provider']} "
              f"xplane_us={row['xplane']['median_us']:.9f} "
              f"native_us={row['kpack4_native']['median_us']:.9f} "
              f"n16_us={row['kpack4_n16']['median_us']:.9f} "
              f"delta_pct={100 * row['n16_delta']:.6f} "
              f"gap_vs_xplane_pct={100 * row['n16_gap_vs_xplane']:.6f} "
              f"same_sign={int(row['same_sign'])} verdict={row['verdict']}")
    output_tsv.write_text("\n".join(lines) + "\n")
    print(f"[fq-n16-delivery-ab] PASS comparisons={len(comparisons)} "
          f"output={output_json}")


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
    path.write_text(json.dumps({
        "schema": "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab.v1",
        "axes": {"qtype": 12, "tile_m": 8, "tile_n": 64,
                 "tactic_tile_k": 256, "warp_m": 8, "warp_n": 16,
                 "stages": 2, "bchunk": 0, "split": 4},
        "arms": arms,
    }) + "\n")
    return {arm["name"]: arm for arm in arms}


def self_test() -> None:
    assert classify(-0.03, True, 0.02) == "N16_DELIVERY_GAIN"
    assert classify(0.03, True, 0.02) == "N16_DELIVERY_REGRESSION"
    assert classify(0.01, False, 0.02) == "NO_MATERIAL_DELIVERY_EFFECT"
    assert classify(-0.03, False, 0.02) == "UNRESOLVED_NOISY"
    with tempfile.TemporaryDirectory(prefix="qz-n16-delivery-") as temp:
        root = pathlib.Path(temp)
        plan, master, runs = root / "plan.json", root / "master.json", root / "runs"
        emit_plan(plan)
        arms = synthetic_master(master)
        for shape_key, shape, _role in CASES:
            shape_text = "x".join(map(str, shape))
            for ap in (0, 1):
                directory = runs / shape_key / f"ap{ap}"
                directory.mkdir(parents=True)
                for name, arm_name, delivery, median in (
                        ("xplane", f"xplane-ap{ap}", "native", 10.0),
                        ("kpack4-native", f"kpack4-ap{ap}", "native", 10.5),
                        ("kpack4-n16", f"kpack4-ap{ap}", "n16", 10.0)):
                    arm = arms[arm_name]
                    mapping = base.MAPPING_ID if arm["weight_layout"] else \
                        "0x0000000000000000"
                    common = (f"q=12 A={arm['artifact_tile_k']} bchunk=0 "
                              f"shape={shape_text}")
                    samples = [median - .1, median, median + .1]
                    text = (
                        f"FQ_SHARD {common} weight_layout={arm['weight_layout']} "
                        f"weight_mapping_id={mapping} typed_rows=1 selected_rows=1 "
                        f"only_split=4 bc_mode=skip weight_delivery={delivery} "
                        "iterations=3 correctness_repeats=8\n"
                        f"FQ_TC_CELL {common} symbol={arm['row']['symbol']} "
                        "tm=8 tn=64 tk=256 wm=8 wn=16 stages=2 "
                        f"provider={arm['a_provider']} S=4 "
                        "scope=PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS "
                        f"provider_capacity_rows={ap} state=MEASURED "
                        f"us={median:.9f} raw_bad=0 reducer_untimed=1 "
                        "failure_step=NONE failure_repeat=-1 shipping_smem=1024 "
                        f"split_smem=2048 partial_bytes={shape[0]*shape[1]*16} "
                        f"samples=[{','.join(str(x) for x in samples)}]\n"
                        f"FQ_SHAPE_DONE {common} weight_layout={arm['weight_layout']} "
                        f"weight_mapping_id={mapping} typed_rows=1 selected_rows=1 "
                        f"only_split=4 bc_mode=skip weight_delivery={delivery} "
                        "iterations=3 status=PASS\n")
                    for r in (1, 2):
                        (directory / f"round-{r}-{name}.log").write_text(text)
        output_json, output_tsv = root / "summary.json", root / "summary.tsv"
        analyze(master, plan, runs, 3, 2, .02, output_json, output_tsv)
        result = json.loads(output_json.read_text())
        assert len(result["comparisons"]) == 6
        assert all(row["verdict"] == "N16_DELIVERY_GAIN"
                   for row in result["comparisons"])
        victim = runs / CASES[0][0] / "ap0" / "round-1-kpack4-n16.log"
        victim.write_text(victim.read_text().replace(
            "weight_delivery=n16", "weight_delivery=native"))
        try:
            analyze(master, plan, runs, 3, 2, .02,
                    root / "red.json", root / "red.tsv")
        except AnalysisError:
            pass
        else:
            raise AssertionError("delivery marker negative stayed green")
    print("[fq-n16-delivery-analysis:self-test] PASS exact 3x2x3 factorial, "
          "paired classification and runtime-marker negative RED")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    plan = sub.add_parser("plan")
    plan.add_argument("--output", type=pathlib.Path, required=True)
    run = sub.add_parser("analyze")
    run.add_argument("--master", type=pathlib.Path, required=True)
    run.add_argument("--plan", type=pathlib.Path, required=True)
    run.add_argument("--runs-root", type=pathlib.Path, required=True)
    run.add_argument("--iterations", type=int, required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--threshold", type=float, default=.02)
    run.add_argument("--output-json", type=pathlib.Path, required=True)
    run.add_argument("--output-tsv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "plan":
            emit_plan(args.output)
        else:
            if args.iterations <= 0 or args.rounds <= 0 or \
                    not 0 < args.threshold < 1:
                raise AnalysisError("invalid iterations/rounds/threshold")
            analyze(args.master, args.plan, args.runs_root,
                    args.iterations, args.rounds, args.threshold,
                    args.output_json, args.output_tsv)
    except (AnalysisError, base.AnalysisError, OSError, ValueError,
            AssertionError) as exc:
        print(f"[fq-n16-delivery-analysis] FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

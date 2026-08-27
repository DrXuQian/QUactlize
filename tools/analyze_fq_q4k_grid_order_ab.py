#!/usr/bin/env python3
"""Analyze the fixed Split-K native-grid vs N-on-x counterfactual."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_fq_q4k_kpack4_xplane_isomorphic_ab as base  # noqa: E402


SCHEMA = "quactlize.fq-q4k-grid-order-ab.v1"
PLAN_SCHEMA = "quactlize.fq-q4k-grid-order-ab-plan.v1"
SCHEDULES = ("native-grid", "n-on-x")
LAYOUTS = ("xplane", "kpack4")
CASES = (
    ("m1_n8192_k5120", (1, 8192, 5120), (0, 1), "N_WIDE_TARGET"),
    ("m1_n5120_k8192", (1, 5120, 8192), (0, 1), "BALANCED_CONTROL"),
    ("m1_n5120_k25600", (1, 5120, 25600), (0, 1), "K_HEAVY_CONTROL"),
)


class AnalysisError(ValueError):
    pass


def emit_plan(output: pathlib.Path) -> None:
    value = {
        "schema": PLAN_SCHEMA,
        "config": base.CONFIG,
        "split": base.SPLIT,
        "schedules": list(SCHEDULES),
        "cases": [
            {"shape_key": key, "shape": list(shape),
             "providers": list(providers), "role": role}
            for key, shape, providers, role in CASES
        ],
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(f"[fq-grid-order-ab-plan] PASS comparisons=12 output={output}")


def validate_plan(value: dict[str, Any]) -> None:
    expected = {
        key: (shape, providers, role)
        for key, shape, providers, role in CASES
    }
    actual = {
        row["shape_key"]: (
            tuple(row["shape"]), tuple(row["providers"]), row["role"])
        for row in value.get("cases", [])
    }
    if value.get("schema") != PLAN_SCHEMA or \
            value.get("config") != base.CONFIG or \
            value.get("split") != base.SPLIT or \
            tuple(value.get("schedules", [])) != SCHEDULES or \
            actual != expected:
        raise AnalysisError("grid-order run plan identity differs")


def classify(x_delta: float, k_delta: float,
             x_stable: bool, k_stable: bool,
             threshold: float) -> str:
    x_improves = x_delta <= -threshold and x_stable
    k_improves = k_delta <= -threshold and k_stable
    x_neutral = abs(x_delta) < threshold
    k_neutral = abs(k_delta) < threshold
    if k_improves and x_neutral:
        return "KPACK4_SPECIFIC_N_AXIS_GAIN"
    if k_improves and x_improves:
        return "GENERIC_N_AXIS_GAIN"
    if x_neutral and k_neutral:
        return "NO_MATERIAL_N_AXIS_EFFECT"
    if k_delta >= threshold and k_stable:
        return "N_ON_X_REGRESSES_KPACK4"
    return "UNRESOLVED_MIXED_OR_NOISY"


def analyze(master_path: pathlib.Path, plan_path: pathlib.Path,
            runs_root: pathlib.Path, iterations: int, rounds: int,
            threshold: float, output_json: pathlib.Path,
            output_tsv: pathlib.Path) -> None:
    arms = base.load_master(master_path)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    comparisons: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for shape_key, shape, providers, role in CASES:
        for ap in providers:
            for layout in LAYOUTS:
                base_name = f"{layout}-ap{ap}"
                schedule_rows: dict[str, dict[str, Any]] = {}
                for schedule in SCHEDULES:
                    samples: list[float] = []
                    medians: list[float] = []
                    resources: set[tuple[int, int, int]] = set()
                    for round_index in range(1, rounds + 1):
                        path = (runs_root / shape_key / f"ap{ap}" / layout /
                                f"round-{round_index}-{schedule}.log")
                        text = path.read_text(errors="replace")
                        shard = base.exactly_one(text, "FQ_SHARD ")
                        done = base.exactly_one(text, "FQ_SHAPE_DONE ")
                        if shard.get("split_grid_order") != schedule or \
                                done.get("split_grid_order") != schedule:
                            raise AnalysisError(
                                f"{path}: split grid-order identity differs")
                        run = base.load_run(
                            path, arms[base_name], shape, iterations)
                        samples.extend(run["samples"])
                        medians.append(run["median_us"])
                        resources.add((run["shipping_smem"], run["split_smem"],
                                       run["partial_bytes"]))
                    if len(resources) != 1:
                        raise AnalysisError(
                            f"runtime resource identity drifted for "
                            f"{shape_key}/{base_name}/{schedule}")
                    schedule_rows[schedule] = {
                        "median_us": statistics.median(samples),
                        "min_us": min(samples), "max_us": max(samples),
                        "run_medians_us": medians,
                        "samples": len(samples),
                    }
                native = schedule_rows["native-grid"]
                candidate = schedule_rows["n-on-x"]
                delta = candidate["median_us"] / native["median_us"] - 1.0
                paired = [
                    n / b - 1.0 for b, n in zip(
                        native["run_medians_us"],
                        candidate["run_medians_us"])
                ]
                stable = all(x > 0 for x in paired) or all(x < 0 for x in paired)
                row = {
                    "shape_key": shape_key, "shape": list(shape), "role": role,
                    "a_provider": f"AP{ap}", "layout": layout,
                    "native_grid": native, "n_on_x": candidate,
                    "delta": delta, "paired_run_deltas": paired,
                    "same_sign": stable,
                }
                comparisons.append(row)
                by_key[(shape_key, ap, layout)] = row

    verdicts = []
    for shape_key, _shape, providers, role in CASES:
        for ap in providers:
            x = by_key[(shape_key, ap, "xplane")]
            k = by_key[(shape_key, ap, "kpack4")]
            verdicts.append({
                "shape_key": shape_key, "role": role,
                "a_provider": f"AP{ap}",
                "xplane_delta": x["delta"],
                "kpack4_delta": k["delta"],
                "verdict": classify(
                    x["delta"], k["delta"], x["same_sign"],
                    k["same_sign"], threshold),
            })

    result = {
        "schema": SCHEMA, "config": base.CONFIG, "split": base.SPLIT,
        "iterations_per_run": iterations, "rounds": rounds,
        "material_threshold": threshold,
        "comparisons": comparisons, "paired_verdicts": verdicts,
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "shape\trole\tprovider\tlayout\tnative_us\tn_on_x_us\t"
        "n_on_x_delta_pct\tpaired_deltas_pct\tsame_sign"
    ]
    for row in comparisons:
        lines.append("\t".join((
            row["shape_key"], row["role"], row["a_provider"], row["layout"],
            f"{row['native_grid']['median_us']:.9f}",
            f"{row['n_on_x']['median_us']:.9f}",
            f"{100 * row['delta']:.6f}",
            ",".join(f"{100 * x:.6f}" for x in row["paired_run_deltas"]),
            str(int(row["same_sign"])),
        )))
        print("FQ_GRID_ORDER_AB "
              f"shape={row['shape_key']} provider={row['a_provider']} "
              f"layout={row['layout']} "
              f"native_us={row['native_grid']['median_us']:.9f} "
              f"n_on_x_us={row['n_on_x']['median_us']:.9f} "
              f"delta_pct={100 * row['delta']:.6f} "
              f"same_sign={int(row['same_sign'])}")
    output_tsv.write_text("\n".join(lines) + "\n")
    for row in verdicts:
        print("FQ_GRID_ORDER_VERDICT "
              f"shape={row['shape_key']} role={row['role']} "
              f"provider={row['a_provider']} verdict={row['verdict']} "
              f"xplane_delta_pct={100 * row['xplane_delta']:.6f} "
              f"kpack4_delta_pct={100 * row['kpack4_delta']:.6f}")
    print(f"[fq-grid-order-ab] PASS comparisons={len(comparisons)} "
          f"output={output_json}")


def self_test() -> None:
    assert classify(0.005, -0.04, True, True, 0.02) == \
        "KPACK4_SPECIFIC_N_AXIS_GAIN"
    assert classify(-0.03, -0.04, True, True, 0.02) == \
        "GENERIC_N_AXIS_GAIN"
    assert classify(0.005, -0.01, True, True, 0.02) == \
        "NO_MATERIAL_N_AXIS_EFFECT"
    assert classify(0.0, 0.03, True, True, 0.02) == \
        "N_ON_X_REGRESSES_KPACK4"
    assert classify(0.0, -0.04, True, False, 0.02) == \
        "UNRESOLVED_MIXED_OR_NOISY"
    import tempfile
    with tempfile.TemporaryDirectory(prefix="qz-grid-order-ab-") as temp:
        root = pathlib.Path(temp)
        path = root / "plan.json"
        emit_plan(path)
        value = json.loads(path.read_text())
        validate_plan(value)
        assert len(value["cases"]) == 3
        broken = json.loads(path.read_text())
        broken["schedules"].reverse()
        try:
            validate_plan(broken)
        except AnalysisError:
            pass
        else:
            raise AssertionError("schedule-order negative stayed green")

        arms = []
        for name in base.ARMS:
            kpack = name.startswith("kpack4")
            ap = int(name.endswith("ap1"))
            provider = "packed-row" if ap else "standard-aiu"
            artifact = 0 if kpack else 64
            arms.append({
                "schema": base.ARM_SCHEMA, "name": name,
                "layout": "q4-kpack4" if kpack else "xplane",
                "weight_layout": int(kpack), "artifact_tile_k": artifact,
                "a_provider": provider, "a_provider_id": ap,
                "selection_denominator": 1,
                "source_typed_denominator": 144,
                "source_global_typed_denominator": 918,
                "row": {
                    "qtype": 12, "artifact_tile_k": artifact,
                    "tile_m": 8, "tile_n": 64, "tactic_tile_k": 256,
                    "warp_m": 8, "warp_n": 16, "stages": 2,
                    "bchunk": 0, "a_provider": provider,
                    "symbol": f"symbol_{name}",
                },
            })
        master = root / "master.json"
        master.write_text(json.dumps({
            "schema": "quactlize.fq-q4k-kpack4-xplane-isomorphic-ab.v1",
            "axes": {"qtype": 12, "tile_m": 8, "tile_n": 64,
                     "tactic_tile_k": 256, "warp_m": 8, "warp_n": 16,
                     "stages": 2, "bchunk": 0, "split": 4},
            "arms": arms,
        }, sort_keys=True) + "\n")
        arm_by_name = {arm["name"]: arm for arm in arms}
        runs = root / "runs"
        for shape_key, shape, providers, _role in CASES:
            shape_text = "x".join(map(str, shape))
            for ap in providers:
                for layout in LAYOUTS:
                    name = f"{layout}-ap{ap}"
                    arm = arm_by_name[name]
                    directory = runs / shape_key / f"ap{ap}" / layout
                    directory.mkdir(parents=True)
                    mapping = base.MAPPING_ID if arm["weight_layout"] else \
                        "0x0000000000000000"
                    for schedule in SCHEDULES:
                        median = (10.5 if layout == "kpack4" else 10.0)
                        if schedule == "n-on-x":
                            median = 10.0 if layout == "kpack4" else 10.05
                        samples = [median - 0.1, median, median + 0.1]
                        for round_index in (1, 2):
                            common = (
                                f"q=12 A={arm['artifact_tile_k']} bchunk=0 "
                                f"shape={shape_text}")
                            text = (
                                f"FQ_SHARD {common} "
                                f"weight_layout={arm['weight_layout']} "
                                f"weight_mapping_id={mapping} typed_rows=1 "
                                "selected_rows=1 only_split=4 bc_mode=skip "
                                f"split_grid_order={schedule} iterations=3 "
                                "correctness_repeats=8\n"
                                f"FQ_TC_CELL {common} "
                                f"symbol={arm['row']['symbol']} "
                                "tm=8 tn=64 tk=256 wm=8 wn=16 stages=2 "
                                f"provider={arm['a_provider']} S=4 "
                                "scope=PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS "
                                f"provider_capacity_rows={ap} state=MEASURED "
                                f"us={median:.9f} raw_bad=0 reducer_untimed=1 "
                                "failure_step=NONE failure_repeat=-1 "
                                "shipping_smem=1024 split_smem=2048 "
                                f"partial_bytes={shape[0] * shape[1] * 16} "
                                f"samples=[{','.join(str(x) for x in samples)}]\n"
                                f"FQ_SHAPE_DONE {common} "
                                f"weight_layout={arm['weight_layout']} "
                                f"weight_mapping_id={mapping} typed_rows=1 "
                                "selected_rows=1 only_split=4 bc_mode=skip "
                                f"split_grid_order={schedule} iterations=3 "
                                "status=PASS\n")
                            (directory /
                             f"round-{round_index}-{schedule}.log").write_text(text)
        output_json, output_tsv = root / "summary.json", root / "summary.tsv"
        analyze(master, path, runs, 3, 2, 0.02, output_json, output_tsv)
        result = json.loads(output_json.read_text())
        if len(result["comparisons"]) != 12 or \
                len(result["paired_verdicts"]) != 6 or \
                any(row["verdict"] != "KPACK4_SPECIFIC_N_AXIS_GAIN"
                    for row in result["paired_verdicts"]):
            raise AssertionError("synthetic grid-order factorial did not close")
        victim = runs / CASES[0][0] / "ap0" / "kpack4" / \
            "round-1-n-on-x.log"
        victim.write_text(victim.read_text().replace(
            "split_grid_order=n-on-x", "split_grid_order=native-grid"))
        try:
            analyze(master, path, runs, 3, 2, 0.02,
                    root / "red.json", root / "red.tsv")
        except AnalysisError:
            pass
        else:
            raise AssertionError("runtime grid-order marker negative stayed green")
    print("[fq-grid-order-ab-analysis:self-test] PASS three exact shapes, "
          "12-cell schedule factorial, paired classification and marker/plan "
          "negatives RED")


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
    run.add_argument("--threshold", type=float, default=0.02)
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
                raise AnalysisError("iterations/rounds/threshold are invalid")
            analyze(args.master, args.plan, args.runs_root,
                    args.iterations, args.rounds, args.threshold,
                    args.output_json, args.output_tsv)
    except (AnalysisError, base.AnalysisError, OSError, ValueError,
            AssertionError) as exc:
        print(f"[fq-grid-order-ab-analysis] FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

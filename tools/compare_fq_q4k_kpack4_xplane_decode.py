#!/usr/bin/env python3
"""Compare authoritative K-pack4 and xplane Q4_K decode summaries."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import tempfile
from typing import Any


XPLANE_SCHEMA = "quactlize.fq_q4k_decode_real_shapes_result.v2"
KPACK4_SCHEMA = "quactlize.fq_q4k_kpack4_decode_real_shapes.v1"
OUTPUT_SCHEMA = "quactlize.fq_q4k_kpack4_vs_xplane_decode.v1"
DECODE_M = (1, 2, 4, 8)
KPACK4_CLASS = "q4-kpack4-transpose-v1"


class CompareError(ValueError):
    pass


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def exact_candidate(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CompareError(f"{label}: candidate is missing")
    required = ("algorithm", "config", "median_us", "min_us", "max_us")
    if any(key not in value for key in required):
        raise CompareError(f"{label}: candidate schema differs")
    median, low, high = (float(value[key]) for key in
                         ("median_us", "min_us", "max_us"))
    if low <= 0 or median <= 0 or high <= 0 or not low <= median <= high:
        raise CompareError(f"{label}: candidate envelope is malformed")
    scope = value.get("metric_scope")
    if scope not in {
            "FULL_OUTPUT",
            "PRODUCER_PLUS_MODELED_REDUCER",
            "PRODUCER_PLUS_MODELED_80PCT_HBM_REDUCER_ZERO_LAUNCH"}:
        raise CompareError(f"{label}: product metric scope differs: {scope}")
    return value


def layout_name(candidate: dict[str, Any], *, kpack4: bool) -> str:
    if kpack4:
        physical = candidate.get("physical_layout_class")
        if not isinstance(physical, dict) or physical.get("name") != KPACK4_CLASS:
            raise CompareError("K-pack4 candidate lost physical layout identity")
        return KPACK4_CLASS
    physical = candidate.get("physical_layout_class")
    if not isinstance(physical, dict) or not isinstance(physical.get("name"), str):
        raise CompareError("xplane candidate lost physical layout identity")
    return str(physical["name"])


def pair_verdict(xplane: dict[str, Any], kpack4: dict[str, Any]
                 ) -> tuple[str, str]:
    ordered = sorted((("xplane", xplane), ("kpack4", kpack4)),
                     key=lambda item: (float(item[1]["median_us"]), item[0]))
    winner_name, winner = ordered[0]
    _, runner = ordered[1]
    if float(winner["max_us"]) >= float(runner["min_us"]):
        return "UNRESOLVED_OVERLAPPING_ENVELOPES", winner_name
    return "RESOLVED", winner_name


def decision_by_family(value: dict[str, Any], label: str
                       ) -> dict[tuple[int, int], dict[str, Any]]:
    rows = value.get("layout_decisions")
    if not isinstance(rows, list):
        raise CompareError(f"{label}: layout decisions missing")
    result = {}
    for row in rows:
        key = (int(row["N"]), int(row["K"]))
        if key in result:
            raise CompareError(f"{label}: duplicate family {key}")
        result[key] = row
    return result


def selected_per_m(decision: dict[str, Any], *, kpack4: bool
                   ) -> tuple[str, dict[int, dict[str, Any]]]:
    selected = decision.get("selected")
    if not isinstance(selected, dict) or not selected.get("available"):
        raise CompareError("selected layout is unavailable")
    physical = selected.get("physical_layout_class")
    if kpack4:
        if not isinstance(physical, dict) or physical.get("name") != KPACK4_CLASS:
            raise CompareError("K-pack4 family class differs")
        name = KPACK4_CLASS
    else:
        if not isinstance(physical, str) or not physical:
            raise CompareError("xplane family class differs")
        name = physical
    rows = selected.get("per_m")
    if not isinstance(rows, list):
        raise CompareError("selected layout lacks per-M rows")
    per_m = {int(row["M"]): row for row in rows}
    if set(per_m) != set(DECODE_M) or len(rows) != len(DECODE_M):
        raise CompareError("selected layout lost one decode M")
    for m, row in per_m.items():
        if float(row["median_us"]) <= 0:
            raise CompareError(f"selected layout M={m} latency is invalid")
    return name, per_m


def compare(xplane_path: pathlib.Path, kpack4_path: pathlib.Path,
            output_dir: pathlib.Path) -> dict[str, Any]:
    xplane = json.loads(xplane_path.read_text())
    kpack4 = json.loads(kpack4_path.read_text())
    if xplane.get("schema") != XPLANE_SCHEMA or \
            kpack4.get("schema") != KPACK4_SCHEMA:
        raise CompareError("input summary schema differs")
    if xplane.get("policy_sha256") != kpack4.get("policy_sha256"):
        raise CompareError("xplane/K-pack4 reducer policy differs")
    if kpack4.get("layout", {}).get("name") != KPACK4_CLASS or \
            kpack4.get("shape_count") != 20 or \
            xplane.get("shape_count") != 20:
        raise CompareError("input layout/shape denominator differs")

    x_shapes = {row["shape_key"]: row for row in xplane.get("shape_winners", [])}
    k_shapes = {row["shape_key"]: row for row in kpack4.get("shape_winners", [])}
    if len(x_shapes) != 20 or len(k_shapes) != 20 or set(x_shapes) != set(k_shapes):
        raise CompareError("shape winner denominator differs")
    shape_rows = []
    census: collections.Counter[str] = collections.Counter()
    median_wins: collections.Counter[str] = collections.Counter()
    for key in sorted(x_shapes):
        xrow, krow = x_shapes[key], k_shapes[key]
        if xrow.get("shape") != krow.get("shape"):
            raise CompareError(f"{key}: logical shape differs")
        shape = tuple(map(int, xrow["shape"]))
        if shape[0] not in DECODE_M:
            raise CompareError(f"{key}: M is outside decode domain")
        xcandidate = exact_candidate(xrow.get("winner"), f"{key}/xplane")
        kcandidate = exact_candidate(krow.get("winner"), f"{key}/kpack4")
        xclass = layout_name(xcandidate, kpack4=False)
        kclass = layout_name(kcandidate, kpack4=True)
        verdict, median_winner = pair_verdict(xcandidate, kcandidate)
        census[verdict] += 1
        median_wins[median_winner] += 1
        xus, kus = float(xcandidate["median_us"]), float(kcandidate["median_us"])
        best = min(xus, kus)
        shape_rows.append({
            "shape_key": key, "shape": list(shape), "verdict": verdict,
            "median_winner": median_winner,
            "xplane": {"physical_layout_class": xclass,
                       "median_us": xus, "regret": xus / best - 1.0,
                       "winner": xcandidate},
            "kpack4": {"physical_layout_class": kclass,
                       "median_us": kus, "regret": kus / best - 1.0,
                       "delta_vs_xplane": kus / xus - 1.0,
                       "winner": kcandidate},
        })

    x_decisions = decision_by_family(xplane, "xplane")
    k_decisions = decision_by_family(kpack4, "kpack4")
    if len(x_decisions) != 5 or set(x_decisions) != set(k_decisions):
        raise CompareError("family decision denominator differs")
    family_rows = []
    for n, k in sorted(x_decisions):
        xname, xper = selected_per_m(x_decisions[(n, k)], kpack4=False)
        kname, kper = selected_per_m(k_decisions[(n, k)], kpack4=True)
        classes = []
        for name, family, per_m in ((xname, "xplane", xper),
                                    (kname, "kpack4", kper)):
            rows = []
            for m in DECODE_M:
                xus = float(xper[m]["median_us"])
                kus = float(kper[m]["median_us"])
                best = min(xus, kus)
                selected = per_m[m]
                rows.append({
                    "M": m, "median_us": float(selected["median_us"]),
                    "regret": float(selected["median_us"]) / best - 1.0,
                    "algorithm": selected["algorithm"],
                    "config": selected["config"],
                })
            classes.append({
                "family": family, "physical_layout_class": name,
                "max_regret": max(row["regret"] for row in rows),
                "mean_regret": statistics.mean(row["regret"] for row in rows),
                "within_5pct": max(row["regret"] for row in rows) <= 0.05,
                "per_m": rows,
            })
        classes.sort(key=lambda row: (row["max_regret"], row["mean_regret"],
                                      row["physical_layout_class"]))
        family_rows.append({
            "N": n, "K": k, "M_values": list(DECODE_M),
            "verdict": "DECODE_ONLY_MINIMAX_NOT_DEPLOYMENT",
            "selected": classes[0], "runner_up": classes[1],
            "all_scores": classes,
        })

    output = {
        "schema": OUTPUT_SCHEMA,
        "scope": "DECODE_M_1_2_4_8_ONLY_NOT_PREFILL_NOT_DEPLOYMENT",
        "xplane_summary": {"path": str(xplane_path.resolve()),
                           "sha256": sha256(xplane_path)},
        "kpack4_summary": {"path": str(kpack4_path.resolve()),
                           "sha256": sha256(kpack4_path)},
        "policy_sha256": xplane["policy_sha256"],
        "shape_count": len(shape_rows), "family_count": len(family_rows),
        "shape_census": dict(sorted(census.items())),
        "median_wins": dict(sorted(median_wins.items())),
        "shape_comparisons": shape_rows,
        "family_minimax": family_rows,
        "deployment_registry": "HELD_BACK_UNTIL_PREFILL_KPACK4_MEASURED",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "comparison.json", output)
    shape_lines = [
        "shape\tM\tN\tK\tverdict\tmedian_winner\txplane_class\t"
        "xplane_us\tkpack4_us\tkpack4_delta_vs_xplane\tkpack4_regret"
    ]
    for row in shape_rows:
        shape_lines.append("\t".join(map(str, (
            row["shape_key"], *row["shape"], row["verdict"],
            row["median_winner"], row["xplane"]["physical_layout_class"],
            row["xplane"]["median_us"], row["kpack4"]["median_us"],
            row["kpack4"]["delta_vs_xplane"], row["kpack4"]["regret"]))))
    atomic_text(output_dir / "shape-comparison.tsv", "\n".join(shape_lines) + "\n")
    family_lines = [
        "N\tK\tselected_family\tselected_class\tselected_max_regret\t"
        "runner_family\trunner_class\trunner_max_regret\t"
        "kpack4_max_regret\tkpack4_within_5pct"
    ]
    for row in family_rows:
        selected, runner = row["selected"], row["runner_up"]
        kscore = next(score for score in row["all_scores"]
                      if score["family"] == "kpack4")
        family_lines.append("\t".join(map(str, (
            row["N"], row["K"], selected["family"],
            selected["physical_layout_class"], selected["max_regret"],
            runner["family"], runner["physical_layout_class"],
            runner["max_regret"], kscore["max_regret"],
            kscore["within_5pct"]))))
    atomic_text(output_dir / "family-minimax.tsv", "\n".join(family_lines) + "\n")
    return output


def candidate(us: float, layout: str, *, kpack4: bool) -> dict[str, Any]:
    return {
        "family": "TENSOR_CORE", "algorithm": "TC_SPLITK_S4_MODELED_E2E",
        "metric_scope": "PRODUCER_PLUS_MODELED_REDUCER",
        "artifact_tile_k": 0 if kpack4 else 64,
        "physical_layout_class": {"name": layout},
        "config": "fixture", "split": 4, "producer_median_us": us - .1,
        "modeled_reducer_us": .1, "median_us": us,
        "min_us": us - .2, "max_us": us + .2,
        "samples_us": [us - .2, us, us + .2],
    }


def self_test() -> None:
    families = ((1024, 5120), (5120, 8192), (5120, 25600),
                (8192, 5120), (25600, 5120))
    x_shapes, k_shapes, x_decisions, k_decisions = [], [], [], []
    for family_index, (n, k) in enumerate(families):
        xper, kper = [], []
        for m in DECODE_M:
            xus = 10.0 + family_index + m * .01
            kus = xus * (0.98 if (family_index + m) % 2 else 1.04)
            key = f"m{m}_n{n}_k{k}_g32"
            xcandidate = candidate(xus, "xplane-q4k-tile-free-f1-le256",
                                   kpack4=False)
            kcandidate = candidate(kus, KPACK4_CLASS, kpack4=True)
            x_shapes.append({"shape_key": key, "shape": [m, n, k],
                             "winner": xcandidate})
            k_shapes.append({"shape_key": key, "shape": [m, n, k],
                             "winner": kcandidate})
            xper.append({"M": m, "reader_artifact_tile_k": 64,
                         "algorithm": xcandidate["algorithm"],
                         "config": "fixture", "median_us": xus, "regret": 0})
            kper.append({"M": m, "algorithm": kcandidate["algorithm"],
                         "split": 4, "config": "fixture", "median_us": kus})
        x_decisions.append({
            "N": n, "K": k, "selected": {
                "physical_layout_class": "xplane-q4k-tile-free-f1-le256",
                "available": True, "per_m": xper}})
        k_decisions.append({
            "N": n, "K": k, "selected": {
                "physical_layout_class": {"name": KPACK4_CLASS},
                "available": True, "per_m": kper}})
    policy_sha = "a" * 64
    xplane = {"schema": XPLANE_SCHEMA, "policy_sha256": policy_sha,
              "shape_count": 20, "shape_winners": x_shapes,
              "layout_decisions": x_decisions}
    kpack4 = {"schema": KPACK4_SCHEMA, "policy_sha256": policy_sha,
              "shape_count": 20, "layout": {"name": KPACK4_CLASS},
              "shape_winners": k_shapes, "layout_decisions": k_decisions}
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-xplane-compare-") as temp:
        root = pathlib.Path(temp)
        x_path, k_path = root / "x.json", root / "k.json"
        x_path.write_text(json.dumps(xplane)); k_path.write_text(json.dumps(kpack4))
        result = compare(x_path, k_path, root / "out")
        if result["shape_count"] != 20 or result["family_count"] != 5 or \
                sum(result["median_wins"].values()) != 20 or \
                result["deployment_registry"] != \
                "HELD_BACK_UNTIL_PREFILL_KPACK4_MEASURED":
            raise AssertionError("comparison denominator/scope differs")
        negatives = (
            ({**kpack4, "policy_sha256": "b" * 64}, "policy"),
            ({**kpack4, "shape_winners": k_shapes[:-1]}, "shape"),
        )
        for broken, label in negatives:
            k_path.write_text(json.dumps(broken))
            try:
                compare(x_path, k_path, root / f"red-{label}")
            except CompareError:
                pass
            else:
                raise AssertionError(f"{label} negative stayed green")
    print("[fq-kpack4-xplane-compare:self-test] PASS 20 shapes/5 families, "
          "product-scope envelopes and decode minimax; policy/missing-shape RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("run")
    run.add_argument("--xplane-summary", type=pathlib.Path, required=True)
    run.add_argument("--kpack4-summary", type=pathlib.Path, required=True)
    run.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            result = compare(args.xplane_summary, args.kpack4_summary,
                             args.output_dir)
            print("KPACK4_XPLANE_SHAPE_CENSUS "
                  f"verdicts={json.dumps(result['shape_census'], sort_keys=True)} "
                  f"median_wins={json.dumps(result['median_wins'], sort_keys=True)}")
            for row in result["family_minimax"]:
                kscore = next(score for score in row["all_scores"]
                              if score["family"] == "kpack4")
                print("KPACK4_XPLANE_FAMILY "
                      f"N={row['N']} K={row['K']} "
                      f"selected={row['selected']['family']} "
                      f"selected_class={row['selected']['physical_layout_class']} "
                      f"selected_max_regret={row['selected']['max_regret']:.9f} "
                      f"kpack4_max_regret={kscore['max_regret']:.9f} "
                      f"kpack4_within_5pct={int(kscore['within_5pct'])}")
            print(f"[fq-kpack4-xplane-compare] PASS output={args.output_dir}")
        return 0
    except (AssertionError, CompareError, json.JSONDecodeError, KeyError,
            OSError, TypeError, ValueError) as error:
        print(f"[fq-kpack4-xplane-compare] FAIL: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

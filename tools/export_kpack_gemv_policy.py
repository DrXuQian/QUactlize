#!/usr/bin/env python3
"""Replay a complete GEMV gate and export measured recipes, never defaults."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

CONFIGS = {f"{c}-{w}-{s}" for c in (16, 32) for w in (4, 8) for s in (1, 4)}


def export(summary):
    if (
        summary.get("status") != "PASS"
        or summary.get("failures")
        or not summary.get("results")
    ):
        raise ValueError("GEMV gate is incomplete or failed")
    expected = []
    for item in summary["plan"]:
        expected.extend(
            (
                item["q"],
                item["n"],
                item["k"],
                1 if c["mode"] == 0 else item["e"],
                c["mode"],
                c["rows"],
                c["channels"],
                c["topk"],
                1,
            )
            for c in item["cases"]
        )
    observed = {}
    receipts = []
    for weight in summary["results"]:
        for row in weight["records"]:
            case = row["case"]
            key = (
                row["q"],
                row["n"],
                row["k"],
                row["experts"],
                case["mode"],
                case["rows"],
                case["channels"],
                case["topk"],
                row.get("input_type"),
            )
            if (
                key in observed
                or key not in expected
                or row.get("correctness") != "PASS"
                or row.get("zero_low_negative") != "DETECTED_FINITE"
                or len(row["errors"]) != 8
                or any(
                    not math.isfinite(x) or x < 0 or x > 0.005 for x in row["errors"]
                )
            ):
                raise ValueError("GEMV coverage or correctness receipt differs")
            samples = row["gemv_samples_us"]
            if set(samples) != CONFIGS:
                raise ValueError("GEMV candidate denominator differs")
            medians = {}
            for name, rounds in samples.items():
                if (
                    len(rounds) != 3
                    or any(len(r) < 3 for r in rounds)
                    or any(not math.isfinite(x) or x <= 0 for r in rounds for x in r)
                ):
                    raise ValueError("invalid GEMV timing samples")
                medians[name] = statistics.median(x for r in rounds for x in r)
            winner = min(medians, key=lambda k: (medians[k], k))
            if row["winner"] != winner or not math.isclose(
                row["gemv_median_us"], medians[winner], rel_tol=1e-12
            ):
                raise ValueError("GEMV winner summary differs from samples")
            baseline = row["gemm_samples_us"]
            if (
                len(baseline) != 3
                or any(len(r) < 3 for r in baseline)
                or any(not math.isfinite(x) or x <= 0 for r in baseline for x in r)
            ):
                raise ValueError("invalid incumbent timing samples")
            incumbent = statistics.median(x for r in baseline for x in r)
            if not math.isclose(incumbent, row["gemm_median_us"], rel_tol=1e-12):
                raise ValueError("incumbent summary differs from samples")
            admitted = medians[winner] <= incumbent
            observed[key] = tuple(map(int, winner.split("-"))) if admitted else None
            round_medians = [statistics.median(r) for r in samples[winner]]
            receipts.append(
                dict(
                    key=key,
                    config=winner,
                    median_us=medians[winner],
                    round_spread_pct=100
                    * (max(round_medians) / min(round_medians) - 1),
                    incumbent_us=incumbent,
                    incumbent_selection=row.get("incumbent_selection"),
                    selected="GEMV" if admitted else ("RETAIN_TC" if row["q"]==8 else "RETAIN_FQ"),
                    scope=row.get("scope","MEASURED_GEMV_POOL_VS_FQ_CORE_NOT_GLOBAL_OPTIMUM"),
                )
            )
    if len(expected) != len(set(expected)) or set(observed) != set(expected):
        raise ValueError("missing/duplicate requested GEMV contexts")
    lines = ["KPACK_GEMV_POLICY_V1"] + [
        "\t".join(map(str, k + observed[k])) for k in sorted(observed) if observed[k]
    ]
    return "\n".join(lines) + "\n", receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execution-library", type=Path, required=True)
    p.add_argument("--native-bundle", type=Path)
    a = p.parse_args()
    summary = json.loads(a.results.read_text())
    with a.execution_library.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != summary.get("execution_sha256"):
        raise ValueError("GEMV execution image differs")
    if summary.get("native_manifest_sha256"):
        if (
            not a.native_bundle
            or hashlib.sha256(
                (a.native_bundle / "manifest.json").read_bytes()
            ).hexdigest()
            != summary["native_manifest_sha256"]
        ):
            raise ValueError("measured FQ selector package differs")
    text, receipts = export(summary)
    with a.output.open("x") as f:
        f.write(text)
    report = dict(
        source_sha256=hashlib.sha256(a.results.read_bytes()).hexdigest(),
        execution_sha256=actual,
        policy_sha256=hashlib.sha256(text.encode()).hexdigest(),
        native_manifest_sha256=summary.get("native_manifest_sha256"),
        device=summary["device"],
        entries=receipts,
    )
    with a.output.with_suffix(".json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"KPACK_GEMV_POLICY PASS entries={len(receipts)} output={a.output}")


if __name__ == "__main__":
    main()

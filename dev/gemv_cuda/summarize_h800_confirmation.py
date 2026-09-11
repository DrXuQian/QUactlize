#!/usr/bin/env python3
"""Adjudicate the frozen six-shape H800 experiment, not a production policy.

Both FP32-dot controls must be within the accepted five-percent regression
bound. Candidate arithmetic is explicit: FP32 group affine is not a pure
layout comparison with controls that reconstruct each weight in FP16.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

POLICY = {
    (1, 512, 2048): ("meta-static-global-rs-fast-bare-av", [1, 16, 1]),
    (1, 1024, 5120): ("affine8-early-fast-bare-a4", [2, 10, 1]),
    **{(1, n, k): ("affine4-early-fast-bare", [4, 8, 1]) for n, k in
       ((4096, 2048), (4096, 4096), (5120, 8192), (8192, 5120))},
}
MODES = ("warm", "rotating")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_records(path, rounds):
    data = json.loads(path.read_text())
    require(data["status"] == "MEASURED" and not data["failures"], "incomplete measurement")
    require(data["rounds"] == rounds, "wrong confirmation round count")
    output = []
    for case in data["cases"]:
        shape, mode = tuple(case["shape"]), case["mode"]
        require(shape in POLICY and mode in MODES, "unexpected experiment cell")
        candidate, recipe = POLICY[shape]
        require(set(data["arms"]) == {"xplane", "raw-reference", candidate}, "comparison arms differ")
        require(data["arms"][candidate]["recipes"] == [recipe], "candidate was not frozen")
        medians, errors = {}, {}
        for name, arm in data["arms"].items():
            rows = case["records"][name]
            expected = Counter((tuple(r), turn) for r in arm["recipes"] for turn in range(rounds))
            require(Counter((tuple(r["recipe"]), r["turn"]) for r in rows) == expected,
                    "missing or duplicate recipe/round")
            for row in rows:
                samples = row["samples_us"]
                require(len(samples) == 15 and all(math.isfinite(x) and x > 0 for x in samples),
                        "invalid event samples")
                require(row["status"] == "PASS" and row["mode"] == mode and
                        row["shape"] == "x".join(map(str, shape)), "record identity differs")
                require(abs(statistics.median(samples) - float(row["median_us"])) <= 2e-5,
                        "reported median differs")
                err = float(row["error"])
                require(math.isfinite(err) and 0 <= err < .005, "independent numeric oracle failed")
            medians[name] = min(statistics.median(float(r["median_us"]) for r in rows
                                                if r["recipe"] == cfg) for cfg in arm["recipes"])
            errors[name] = max(float(r["error"]) for r in rows)
        deltas = {name: 100 * (medians[candidate] / medians[name] - 1)
                  for name in ("xplane", "raw-reference")}
        output.append(dict(shape=list(shape), mode=mode, candidate=candidate, recipe=recipe,
                           weight_arithmetic=data["arms"][candidate]["weight_arithmetic"],
                           fixture_sha256=case["fixture_sha256"], median_us=medians,
                           delta_pct=deltas, max_normalized_error=errors,
                           verdict="PASS" if all(x < 5 for x in deltas.values()) else "OPEN"))
    return output, data["control_manifest_sha256"]


def summarize(confirmation, validation, fixtures):
    cells, controls = [], set()
    for path in confirmation:
        rows, control = read_records(path, 6)
        cells.extend(rows)
        controls.add(control)
    expected = Counter((shape, mode) for shape in POLICY for mode in MODES)
    require(Counter((tuple(r["shape"]), r["mode"]) for r in cells) == expected,
            "confirmation must cover exactly six shapes and two cache regimes")
    source_hashes = {tuple(r["shape"]): r["fixture_sha256"] for r in cells}
    require(all(source_hashes[tuple(r["shape"])] == r["fixture_sha256"] for r in cells),
            "weight fixture changed across cache regimes")
    registry = json.loads(fixtures.read_text())
    require(len(registry) == 12 and len({r["sha256"] for r in registry}) == 12,
            "random fixture registry must contain two seeds per shape")
    by_hash = {r["sha256"]: r for r in registry}
    checked = []
    for path in validation:
        rows, control = read_records(path, 1)
        controls.add(control)
        for row in rows:
            fixture = by_hash[row["fixture_sha256"]]
            require(fixture["shape"] == row["shape"] and fixture["weight_bytes_unchanged"] and
                    fixture["oracle"] == "OFFICIAL_GGUF_FP64_DOT" and
                    fixture["source_sha256"] == source_hashes[tuple(row["shape"])],
                    "random validation fixture differs")
            checked.append(dict(shape=row["shape"], mode=row["mode"], seed=fixture["seed"],
                                fixture_sha256=row["fixture_sha256"],
                                max_normalized_error=row["max_normalized_error"]))
    require(Counter((tuple(r["shape"]), r["mode"], r["seed"]) for r in checked) ==
            Counter((shape, mode, seed) for shape in POLICY for mode in MODES for seed in (93711, 93719)),
            "random validation coverage differs")
    require(len(controls) == 1, "control binary changed between experiments")
    return dict(verdict="PASS" if all(r["verdict"] == "PASS" for r in cells) else "OPEN",
                threshold_pct=5, comparison="STRICT_LESS_THAN_UNROUNDED_MEDIAN_REGRESSION",
                scope="H800_DENSE_M1_Q4_NOT_PPU_OR_MODEL_ADMISSION", production_changed=False,
                weight_bytes_changed=False, control_manifest_sha256=controls.pop(),
                confirmation_rounds=6, samples_per_round=15, cells=cells,
                random_validation=checked,
                authority=[dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in [*confirmation, *validation, fixtures]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.confirmation, args.validation, args.fixtures)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print("Q4_H800_FROZEN verdict=" + result["verdict"] + " performance_cells=12 random_cells=24")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Warmed, alternating CUDA GEMV comparisons without NumPy or PyTorch."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


CASES = {
    "q4-indexed": ("q12-n512-k2048-e256-c1.bin", (16, 4, 8), (16, 4, 8)),
    "q5-indexed": ("q13-n2048-k512-e256-c8.bin", (16, 8, 1), (16, 4, 1)),
    "q4-dense": ("q12-n4096-k2048-e1-c1.bin", (16, 4, 8), (16, 4, 8)),
}


def sha(path):
    with path.open("rb") as stream:
        # Python 3.10 on WSL has no hashlib.file_digest.
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
        return h.hexdigest()


def parse(text):
    rows = [line for line in text.splitlines() if line.startswith("GEMV_STANDALONE ")]
    if len(rows) != 1:
        raise ValueError("expected one standalone result")
    head, tail = rows[0].split(" samples=", 1)
    fields = dict(re.findall(r"(\w+)=([^ ]+)", head))
    samples = json.loads(tail)
    if fields.get("status") != "PASS" or len(samples) != 15 or not all(
        isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples
    ):
        raise ValueError("invalid standalone measurement")
    err = float(fields["error"])
    if not math.isfinite(err) or not 0 <= err < .005:
        raise ValueError("numeric gate did not pass")
    measured = statistics.median(samples)
    if abs(measured - float(fields["median_us"])) > 2e-5:
        raise ValueError("median differs from raw samples")
    return dict(**fields, samples_us=samples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = dict(runner=args.runner, baseline=args.baseline, candidate=args.candidate)
    if args.reference:
        files["reference"] = args.reference
    receipts = {name: dict(path=str(path.resolve()), sha256=sha(path)) for name, path in files.items()}
    result = dict(status="RUNNING", files=receipts, cases={},
                  script_sha256=sha(Path(__file__)), ppu_admission=False,
                  scope="WARM_GRAPH_COMPLETE_CALL_F32_ENDPOINTS_WITH_REDUCER_NO_GATHER_SCATTER",
                  recipe_scope="FIXED_RTX5090_RECIPES_NOT_A_PER_DEVICE_CONFIG_SEARCH",
                  graph_calls=32, excluded_graph_warmups=5, rounds=4)
    for case in args.cases:
        name, baseline_cfg, candidate_cfg = CASES[case]
        fixture = args.fixtures / name
        arms = dict(baseline=(args.baseline, "pair", baseline_cfg),
                    candidate=(args.candidate, "pair", candidate_cfg))
        if args.reference:
            arms.update(dmmv_fpA=(args.reference, "dmmv", (16, 4, 1)),
                        mmvq_q8A=(args.reference, "mmvq", (16, 4, 1)))
        records = {arm: [] for arm in arms}
        for turn in range(4):
            names = list(arms) if turn % 2 == 0 else list(reversed(arms))
            for arm in names:
                library, kind, cfg = arms[arm]
                command = [str(args.runner.resolve()), str(fixture.resolve()),
                           str(library.resolve()), kind, *map(str, cfg)]
                proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
                log = args.output / f"{case}.{turn}.{arm}.log"
                log.write_text(proc.stdout + proc.stderr)
                if proc.returncode:
                    raise RuntimeError(f"GEMV failed rc={proc.returncode}: {log}")
                row = parse(proc.stdout)
                if row["kind"] != kind or row["config"] != "-".join(map(str, cfg)):
                    raise ValueError("runner identity differs")
                records[arm].append(dict(command=command, result=row))
                print(f"GEMV_AB_PROGRESS case={case} round={turn+1}/4 arm={arm} us={row['median_us']}", flush=True)
        medians = {arm: statistics.median(float(r["result"]["median_us"]) for r in rows)
                   for arm, rows in records.items()}
        result["cases"][case] = dict(fixture_sha256=sha(fixture), records=records,
                                    median_us=medians,
                                    delta_pct=(medians["candidate"] / medians["baseline"] - 1) * 100)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, path in files.items():
        if sha(path) != receipts[name]["sha256"]:
            raise ValueError("binary changed during measurement")
    result["status"] = "PASS"
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("GEMV_AB_COMPLETE " + json.dumps({k: v["median_us"] for k, v in result["cases"].items()}))


if __name__ == "__main__":
    main()

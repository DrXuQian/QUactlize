#!/usr/bin/env python3
"""Host-only contract for the INBOX 122 Stream-K tail-shape plan."""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

import plan_streamk_tail_shapes as plan  # noqa: E402


def reject(label, rows) -> None:
    try:
        plan.validate_plan(rows)
    except ValueError:
        return
    raise AssertionError(f"planted regression {label!r} was accepted")


def main() -> int:
    rows = plan.build_plan(workers=288, tile_m=64, tile_n=128, k=4096, gs=128)

    # Independent fixed oracle: these are the seven rows printed in INBOX 122.
    # The implementation derives them from W ratios; this check stops a ratio
    # edit from silently changing the experiment while remaining self-consistent.
    expected_q = [288, 289, 320, 432, 576, 1024, 1152]
    expected_tail = [
        Fraction(0), Fraction(287, 289), Fraction(4, 5), Fraction(1, 3),
        Fraction(0), Fraction(1, 8), Fraction(0),
    ]
    if [r.q for r in rows] != expected_q:
        raise AssertionError(f"W=288 Q oracle changed: {[r.q for r in rows]}")
    if [r.tail for r in rows] != expected_tail:
        raise AssertionError(f"W=288 tail oracle changed: {[r.tail for r in rows]}")

    # A second worker count proves this is not a hidden W=288 lookup table and
    # still spans the requested causal bands.
    rows128 = plan.build_plan(workers=128, tile_m=64, tile_n=128, k=4096, gs=128)
    if [r.q for r in rows128] != [128, 129, 143, 192, 256, 456, 512]:
        raise AssertionError("W=128 dynamic plan did not follow W")
    plan.validate_plan(rows128)

    text = plan.render_tsv(rows)
    lines = text.splitlines()
    header = lines[0].split("\t")
    if tuple(header) != plan.TSV_FIELDS or "Q" not in header or "W" not in header or "tail_pct" not in header:
        raise AssertionError("TSV lacks the Q/W/tail attribution contract")
    if len(lines) != 8 or any(len(line.split("\t")) != len(header) for line in lines[1:]):
        raise AssertionError("TSV is not one complete record per planned shape")
    tail_col = header.index("tail_pct")
    if [line.split("\t")[tail_col] for line in lines[1:]] != [
            "0.000000", "99.307958", "80.000000", "33.333333",
            "0.000000", "12.500000", "0.000000"]:
        raise AssertionError("printed tail percentages changed or lost precision")

    # Exercise the CLI as the future runner will: no device, explicit kernel
    # geometry, and one machine-readable record per shape.
    proc = subprocess.run([
        sys.executable, str(ROOT / "benchmarks" / "plan_streamk_tail_shapes.py"),
        "--workers", "288", "--tile-m", "64", "--tile-n", "128",
        "--k", "4096", "--gs", "128", "--format", "jsonl",
    ], text=True, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout.splitlines()) != 7 or proc.stderr:
        raise AssertionError(
            f"planner CLI failed rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}")

    # Constructive negative controls.  Each is a historical silent-green
    # shape: scan only exact waves; print a stale tail; detach Q from M/N; or
    # quietly relabel an extreme row as medium.
    reject("only-divisible-Q", [r for r in rows if r.q % r.workers == 0])
    reject("stale-tail", [dataclasses.replace(rows[0], tail=Fraction(1, 99)), *rows[1:]])
    reject("shape-Q-disagree", [dataclasses.replace(rows[0], m=rows[0].m + rows[0].tile_m), *rows[1:]])
    reject("wrong-band", [rows[0], dataclasses.replace(rows[1], band="medium"), *rows[2:]])

    print("[streamk-tail-plan] PASS: W-derived 0/low/medium/extreme shapes; "
          "Q/W/waves/tail printed per row; four planted controls rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

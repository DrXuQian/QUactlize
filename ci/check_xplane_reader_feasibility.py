#!/usr/bin/env python3
"""Contract gate for INBOX 169's host-only xplane reader census."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "dev/fold_derivation/run_xplane_reader_feasibility.sh"
ANALYZER = ROOT / "dev/fold_derivation/xplane_reader_feasibility.py"
REPORT = ROOT / "dev/fold_derivation/XPLANE_READER_FEASIBILITY.md"


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise SystemExit(f"[xplane-reader-contract] FAIL: missing {needle!r}")


def main() -> int:
    runner = RUNNER.read_text()
    analyzer = ANALYZER.read_text()
    report = REPORT.read_text()
    for token in (
        "/workspace/quactlize-xplane-reader-feasibility",
        "run_l137_bc_arrangement_layout.sh",
        "wrong-permutation-bit",
        "missing-denominator",
        "PLANTED_RED",
    ):
        require(runner, token)
    for token in (
        "arrangement_supported_v",
        "ArrangementSlotPermutation",
        "q4_group",
        "oracle_coordinates",
        "classified_coordinates",
        "direct_cpw_same_word",
        "closure_run",
    ):
        require(analyzer, token)
    for token in (
        "11/17 arrangements",
        "20/26 planes",
        "0/26 planes",
        "partial generalisation, not a universal reader",
        "TODO #58 与出货路无关",
        "不动 π,不动离线摆放,不动 artifact 字节",
    ):
        require(report, token)
    if "/tmp" in runner:
        raise SystemExit("[xplane-reader-contract] FAIL: the new runner must not use /tmp")

    proc = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, text=True)
    if proc.returncode:
        return proc.returncode
    print("[xplane-reader-contract] PASS: INBOX 169 census is regenerated and both source plants are rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

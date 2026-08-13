#!/usr/bin/env python3
"""Regenerate L143 locally and pin the exact committed WK1 evidence.

The PPU box cannot compile this host-only CuTe oracle with its nvcc/GCC13
combination.  The box runner therefore copies the result-SHA's committed output
instead of pretending to rerun it.  This local-tier check is the executable
owner of that evidence and its red controls.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "dev/fold_derivation/run_l143_wk4_production_delivery.sh"
EXPECTED = ROOT / "dev/fold_derivation/l143_wk4_production_delivery.expected.txt"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qz-l143-committed-") as raw:
        out = Path(raw)
        env = dict(os.environ, QUACTLIZE_L143_OUT=str(out))
        proc = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        if proc.returncode:
            print(f"[l143-committed] FAIL: generator rc={proc.returncode}\n{proc.stdout}")
            return 1
        got = (out / "l143.out").read_bytes()
    wanted = EXPECTED.read_bytes()
    if got != wanted:
        print("[l143-committed] FAIL: generated output differs from result-SHA evidence")
        return 1

    # A self-comparison is not a negative control.  Each of these is a distinct
    # wrong delivery map whose exact red count/hash is part of the committed
    # output and must remain present exactly once.
    text = wanted.decode()
    reds = (
        "production-order hash=ea96e6b4155759c3 shipping-diff=12288/16384 EXPECTED-RED",
        "compact-order hash=17dfe6248fc38143 shipping-diff=15360/16384 EXPECTED-RED",
        "first32x2 shipping-diff=16384/16384 clean=0 EXPECTED-RED",
        "adjacent-nibble shipping-diff=8192/16384 pairs=8192 bad-pairs=14336 EXPECTED-RED",
        "swapped-sources shipping-diff=16384/16384 pairs=8192 bad-pairs=8192 EXPECTED-RED",
    )
    if any(text.count(red) != 1 for red in reds):
        print("[l143-committed] FAIL: one or more planted delivery maps lost its exact red")
        return 1
    print("[l143-committed] PASS: regenerated exact WK1 0/8192 map and five planted reds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

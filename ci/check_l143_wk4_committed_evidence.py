#!/usr/bin/env python3
"""Regenerate and pin the result-SHA's standalone Marlin local evidence.

The PPU box is not the authority for these host/compile-time oracles.  It
consumes their exact output from the commit it builds.  L167-L170 replace the
retired generic-WK4 xplane/two-source/cadence evidence: they bind the classic
format, pipeline cadence, generated standalone type and scheduler lifecycle.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "dev/fold_derivation/run_l143_dense_marlin_wk4_target.sh"
EXPECTED = ROOT / "dev/fold_derivation/l143_standalone_marlin.expected.txt"

REQUIRED_LINES = (
    "[L167] PASS: independent classic/direct and Awesome-CuTe/permutation anchors agree; "
    "asymmetric provider, byte, inverse, and negative controls proved",
    "[l168:runner] positive=PASS negative_controls=3/3_RED result=PASS",
    "[l169] PASS: generated wrapper reaches standalone Marlin kernel + collective device bodies; "
    "route-severed and collective-severed same-source controls suppress the exact marker",
    "[l170:runner] positive=PASS negative_controls=7/7_RED result=PASS",
    "[dense-marlin-wk4] PASS: standalone format/collective/scheduler/kernel wired; "
    "standalone tactic authority consumed; generic WK4 compatibility absent; "
    "thirteen structural plants rejected",
    "[classic-156] PASS: exact one-launch shape, source/tool/binary identity and full ACU capture "
    "are fail-closed",
    "[l143] PASS: standalone Marlin format + cadence + generated type + scheduler lifecycle; "
    "generic WK4 compatibility is absent; no device result claimed",
)

RETIRED_CLAIMS = (
    "shipping-xplane",
    "L154",
    "L155",
    "indexed converter equals four template arms",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--committed-only", action="store_true")
    parser.add_argument("--evidence", type=Path, default=EXPECTED)
    args = parser.parse_args()

    try:
        expected = args.evidence.read_text()
    except OSError as exc:
        print(f"[l143-committed] FAIL: cannot read evidence: {exc}")
        return 1

    if not args.committed_only:
        proc = subprocess.run(
            ["bash", str(RUNNER)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if proc.returncode:
            print(f"[l143-committed] FAIL: aggregate rc={proc.returncode}\n{proc.stdout}")
            return 1
        if proc.stdout != expected:
            print("[l143-committed] FAIL: regenerated standalone evidence differs from result SHA")
            print(proc.stdout, end="")
            return 1

    if tuple(expected.splitlines()) != REQUIRED_LINES:
        print("[l143-committed] FAIL: evidence is not the exact seven-line standalone contract")
        return 1
    if any(token in expected for token in RETIRED_CLAIMS):
        print("[l143-committed] FAIL: retired generic-WK4 claim still authorizes the box")
        return 1

    # The aggregate is only an evidence compositor.  Its component gates own
    # 3 + 7 + 13 independent red controls, printed in the committed lines
    # above.  Require those counts literally so a future rewrite cannot turn a
    # missing negative arm into an unchanged-looking PASS sentence.
    for token in ("negative_controls=3/3_RED", "negative_controls=7/7_RED",
                  "thirteen structural plants rejected"):
        if expected.count(token) != 1:
            print(f"[l143-committed] FAIL: negative-control closure drifted: {token}")
            return 1

    print(
        "[l143-committed] PASS: exact seven-line standalone evidence "
        f"{'validated' if args.committed_only else 'regenerated'}; "
        "format/cadence/type/scheduler/isolation negative controls remain closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

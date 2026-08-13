#!/usr/bin/env python3
"""Exhaust the proved standalone Marlin fixed launch.

The former program exhausted 656,230 tuples from the generic dense tactic
tables through ``PersistentTileSchedulerPPUMarlin``.  That remains useful
historical evidence about the vendor stripe core, but it is not evidence that
the new ``MarlinSchedulerPPU`` descriptor, reverse traversal and lock protocol
were used.

The first standalone production authority is intentionally one launch:
M=1,N=K=4096,L=1, tile=16x128x128, CU=72, B=1.  L168 independently derives
the classic/Awesome trace; L170 walks every production descriptor and every
(q,k-tile) cell, then runs seven causal plants.  This gate composes those two
proofs.  It does not claim that the broader future shape/sweep domain has been
exhausted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHECKS = (
    (
        "independent trace",
        ("bash", "dev/fold_derivation/run_l168_marlin_pipeline_trace.sh"),
        (
            "G=72 I=15 active=69 segments=98 copies=1024",
            "mma=65536 result=PASS",
            "positive=PASS negative_controls=3/3_RED result=PASS",
        ),
    ),
    (
        "production descriptor",
        ("bash", "dev/fold_derivation/run_l170_standalone_marlin_scheduler.sh"),
        (
            "segments=98 cells=1024 handoffs=66 locks-reset=32",
            "peers={3:30,4:2}",
            "positive=PASS negative_controls=7/7_RED result=PASS",
        ),
    ),
)


def main() -> int:
    summaries: list[str] = []
    for label, command, witnesses in CHECKS:
        run = subprocess.run(
            command, cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        missing = [token for token in witnesses if token not in run.stdout]
        if run.returncode != 0 or missing:
            print(
                f"[standalone-marlin-exhaustive] FAIL: {label} rc={run.returncode} "
                f"missing={missing}\n{run.stdout[-2600:]}",
                file=sys.stderr,
            )
            return 1
        summaries.append(
            next(line for line in run.stdout.splitlines()
                 if line.startswith("[l168:classic]") or line.startswith("[l170] PASS"))
        )

    print("[standalone-marlin-exhaustive] evidence: " + " | ".join(summaries))
    print(
        "[standalone-marlin-exhaustive] PASS: fixed launch 1024/1024 "
        "(q,k_tile) cells exact-once, reverse traversal and 32 global-q locks "
        "close; 10 causal plants red; generic-656230-domain="
        "PRE_STANDALONE_NOT_EVIDENCE; standalone-general-shape="
        "NOT_IMPLEMENTED_NOT_CLAIMED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

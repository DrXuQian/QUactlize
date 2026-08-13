#!/usr/bin/env python3
"""Fail-closed contract for the standalone Marlin PPU stack.

The pre-standalone version of this checker inspected
``ppu_aiu_gemm_mixed_input_marlin.hpp`` and the vendor Marlin tile scheduler.
Those types are deliberately no longer the shipping Marlin implementation.
Keeping their source tokens green would prove a retired generic compatibility
path while saying "dense Marlin" in CI.

This checker therefore composes the production-bound standalone proofs:

* L167: classic packed-int4/scale format with independent inverse anchors;
* L168: classic/Awesome cadence, fixed launch and 4->2->1 reduction trace;
* L174: dedicated load/dequant/MMA source split, exhaustive W4 arithmetic and
  Awesome-CuTe compute-cadence anchors;
* L175: native PPU ``FragmentC[4]`` type/ABI, no generic CuTe C fragment or
  whole-accumulator address escape, plus the independent L139 map/reduction;
* L170: production ``MarlinSchedulerPPU`` ABI, exact coverage and q-lock life;
* L173: production per-CTA state init and absolute-q/K segment rebasing;
* the structural stack gate, including absence of generic WarpK seams.

The wider tactic domain is owned separately by L172.  It is not silently
promoted to a production sweep by this contract.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHECKS = (
    (
        "format",
        ("bash", "dev/fold_derivation/run_l167_classic_marlin_format.sh"),
        "[L167] PASS: independent classic/direct and Awesome-CuTe/permutation anchors agree",
    ),
    (
        "cadence",
        ("bash", "dev/fold_derivation/run_l168_marlin_pipeline_trace.sh"),
        "[l168:runner] positive=PASS negative_controls=3/3_RED result=PASS",
    ),
    (
        "compute",
        ("bash", "dev/fold_derivation/run_l174_marlin_compute_contract.sh"),
        "[l174:runner] positive=PASS negative_controls=4/4_RED result=PASS",
    ),
    (
        "native-fragment",
        ("bash", "dev/fold_derivation/run_l175_native_fragment_contract.sh"),
        "[l175:runner] positive=source+compile+L139 negative_controls=3/3_RED wrong-layout-compile=RED result=PASS",
    ),
    (
        "scheduler",
        ("bash", "dev/fold_derivation/run_l170_standalone_marlin_scheduler.sh"),
        "[l170:runner] positive=PASS negative_controls=7/7_RED result=PASS",
    ),
    (
        "cta-state",
        (sys.executable, "ci/check_l173_marlin_cta_state.py"),
        "[l173-contract] PASS:",
    ),
    (
        "stack",
        (sys.executable, "ci/check_dense_marlin_wk4_target.py"),
        "generic WK4 compatibility absent;",
    ),
)


def main() -> int:
    evidence: list[str] = []
    for label, command, witness in CHECKS:
        run = subprocess.run(
            command, cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if run.returncode != 0 or witness not in run.stdout:
            print(
                f"[dense-marlin-contract] FAIL: standalone {label} proof "
                f"returned {run.returncode} or lacked {witness!r}\n"
                + run.stdout[-2400:],
                file=sys.stderr,
            )
            return 1
        lines = [line for line in run.stdout.splitlines() if line.strip()]
        evidence.append(lines[-1])

    print("[dense-marlin-contract] evidence: " + " | ".join(evidence))
    print(
        "[dense-marlin-contract] PASS: standalone format/collective/"
        "scheduler/kernel fixed target is bound; retired generic Marlin is not "
        "accepted as evidence; broader production sweep is outside this gate"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
* L177: one cached Split/First/Final decision and a zero-call unsplit handoff path;
* L178: final per-thread CTA invariants, rebased source pointers and one-CTA
  shared bases, exhaustively matched to independent classic/Awesome formulas;
* L179: checked host lowering owns all validity/overflow decisions, the
  assume-valid device path shares one output map with its exhaustive oracle,
  and raw Params/update seams are unavailable from the shipping handle;
* L180: the public 40-byte scheduler Params lower once to a pointer-free
  16-byte device traversal state with descriptor-identical behavior;
* L181: the dense-M1 PPU m8 extension packs one A row, assigns its sixteen
  16-byte vectors exactly once, uses the PPU-specific plain-x2 provider map,
  halves the output/accumulator extent, and leaves the B/scale artifact
  byte-identical to m16;
* L169-m8: that exact m8 production row reaches a generated device body;
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
        "[l174:runner] positive=PASS negative_controls=8/8_RED result=PASS",
    ),
    (
        "native-fragment",
        ("bash", "dev/fold_derivation/run_l175_native_fragment_contract.sh"),
        "[l175:runner] positive=source+compile+L139 negative_controls=4/4_RED wrong-layout-compile=RED result=PASS",
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
        "handoff-lifecycle",
        (sys.executable, "ci/check_l177_marlin_handoff_lifecycle.py"),
        "[l177-contract] PASS:",
    ),
    (
        "state-hoist",
        (sys.executable, "ci/check_l178_marlin_state_hoist.py"),
        "[l178-contract] PASS:",
    ),
    (
        "checked-lowering",
        ("bash", "dev/fold_derivation/run_l179_marlin_checked_lowering.sh"),
        "[l179:runner] positive=geometry+source+handle-lifecycle "
        "negative_controls=17/17_RED result=PASS",
    ),
    (
        "scheduler-hot-state",
        ("bash", "dev/fold_derivation/run_l180_marlin_scheduler_hot_state.sh"),
        "[l180:runner] positive=262144-schedule-equivalence "
        "negative_controls=7/7_RED result=PASS",
    ),
    (
        "m8-extension",
        ("bash", "dev/fold_derivation/run_l181_standalone_marlin_m8.sh"),
        "[l181:runner] positive=6-contracts negative=10/10_RED source=PASS result=PASS",
    ),
    (
        "m8-generated-body",
        (
            "bash", "-c",
            "QUACTLIZE_L169_VARIANT=m8 "
            "QUACTLIZE_L169_OUT=/workspace/quactlize-l169-m8-contract "
            "bash dev/fold_derivation/run_l169_standalone_marlin_unit.sh",
        ),
        "[l169] PASS: variant=m8 pipe_roll=0 generated wrapper reaches standalone Marlin kernel + collective device bodies",
    ),
    (
        "pipe-roll-experiment",
        (sys.executable, "ci/check_l182_marlin_pipe_roll.py"),
        "[l182:local] PASS: default/outer-only/inner-control routes reach the exact m8 device body; "
        "negative_controls=14/14_RED mixed-range=PASS; PPU static footprint/register/spill remains a mandatory box compile-only postcondition",
    ),
    (
        "ppu-admission-boundary",
        (sys.executable, "ci/check_l176_ppu_admission_boundary.py"),
        "[l176-boundary] PASS: local L169/L174/L175 execute only in local mode; "
        "PPU mode consumes result-SHA evidence before an override-free SDK-owned "
        "hgcc/hgobjdump build; negative_controls=7/7_RED",
    ),
    (
        "ppu-portability",
        (sys.executable, "dev/fold_derivation/ppu_portability_check.py"),
        "[ok]   ppu_portability:",
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
        "scheduler/kernel m16 baseline and m8 PPU extension are bound; retired generic Marlin is not "
        "accepted as evidence; broader production sweep is outside this gate"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

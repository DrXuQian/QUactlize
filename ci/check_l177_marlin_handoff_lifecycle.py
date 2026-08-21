#!/usr/bin/env python3
"""Bind the L177 lock oracle to the production kernel call structure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/marlin_kernel_ppu.hpp"
SCHEDULER = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/marlin_scheduler_ppu.hpp"
RUNNER = ROOT / "dev/fold_derivation/run_l177_marlin_handoff_lifecycle.sh"


def balanced_body(text: str, anchor: str) -> str:
    start = text.find(anchor)
    if start < 0:
        raise ValueError(f"missing anchor {anchor!r}")
    brace = text.find("{", start)
    if brace < 0:
        raise ValueError(f"missing body after {anchor!r}")
    depth = 0
    for end in range(brace, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start : end + 1]
    raise ValueError(f"unterminated body after {anchor!r}")


def audit(kernel: str, scheduler: str) -> list[str]:
    bad: list[str] = []
    try:
        op = balanced_body(
            kernel,
            "CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf)",
        )
        handoff = balanced_body(
            kernel, "CUTLASS_DEVICE static void global_handoff(",
        )
        acquire = balanced_body(
            scheduler,
            "CUTLASS_DEVICE static void acquire_peer_turn_assume_split(",
        )
        release = balanced_body(
            scheduler,
            "CUTLASS_DEVICE static void release_peer_turn_assume_split(",
        )
        lock_index = balanced_body(
            scheduler,
            "CUTLASS_HOST_DEVICE static constexpr int barrier_lock_index(",
        )
    except ValueError as exc:
        return [str(exc)]

    once = (
        "bool const split = TileScheduler::requires_handoff(work);",
        "bool const first = TileScheduler::is_first_peer(work);",
        "bool const final = TileScheduler::is_final_peer(work);",
    )
    for token in once:
        if op.count(token) != 1:
            bad.append(f"kernel must cache exactly once: {token}")

    split_block = """if (split) {
        TileScheduler::acquire_peer_turn_assume_split(
            params.scheduler, work, tid);
        global_handoff(accum, params, work, first, final,
                       problem_m, problem_n);
        TileScheduler::release_peer_turn_assume_split(
            params.scheduler, work, tid, final);
      }"""
    if op.count(split_block) != 1:
        bad.append("kernel split branch is not acquire->handoff->release exactly once")
    if op.count("if (final) {") != 1 or op.count("write_result(accum") != 1:
        bad.append("kernel final write does not consume the cached final bit")
    if "TileScheduler::is_final_peer(work)" in handoff or \
            "TileScheduler::is_first_peer(work)" in handoff:
        bad.append("global_handoff recomputes cached first/final")
    if "bool first_peer, bool final_peer," not in handoff:
        bad.append("global_handoff does not receive cached first/final")

    for label, body in (("acquire", acquire), ("release", release)):
        for forbidden in (
            "requires_handoff(", "is_first_peer(", "is_final_peer(",
            "peer_release_action(",
        ):
            if forbidden in body:
                bad.append(f"assume-split {label} repeats {forbidden}")
    if "BarrierType(work.peer_idx)" not in acquire:
        bad.append("assume-split acquire does not consume the cached peer ordinal")
    if "if (final_peer)" not in release or "if (!final_peer)" in release:
        bad.append("assume-split release does not reset only at cached final")
    if "int(work.N_idx)" not in lock_index or "work.N_idx &" in lock_index:
        bad.append("lock id is not the global output-tile q")
    return bad


def main() -> int:
    kernel = KERNEL.read_text()
    scheduler = SCHEDULER.read_text()
    bad = audit(kernel, scheduler)

    plants = (
        (
            "local-q",
            kernel,
            scheduler.replace("int(work.N_idx) : -1", "int(work.N_idx & 15) : -1", 1),
        ),
        (
            "early-reset",
            kernel,
            scheduler.replace("if (final_peer) {", "if (!final_peer) {", 1),
        ),
        (
            "skip-handoff",
            kernel.replace(
                "        global_handoff(accum, params, work, first, final,\n"
                "                       problem_m, problem_n);\n",
                "",
                1,
            ),
            scheduler,
        ),
    )
    for label, planted_kernel, planted_scheduler in plants:
        planted_bad = audit(planted_kernel, planted_scheduler)
        if not planted_bad:
            bad.append(f"source plant {label} was accepted")

    run = subprocess.run(
        ("bash", str(RUNNER)), cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if run.returncode != 0 or \
            "[l177:runner] positive=PASS negative_controls=3/3_RED result=PASS" not in run.stdout:
        bad.append(
            "direct lifecycle oracle failed or lacked its final witness:\n" +
            run.stdout[-2000:]
        )

    if bad:
        for item in bad:
            print(f"[l177-contract] FAIL: {item}", file=sys.stderr)
        return 1
    print(
        "[l177-contract] PASS: one cached Split/First/Final decision owns the "
        "split-only acquire->handoff->release branch; global-q, early-reset "
        "and skipped-handoff source plants RED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

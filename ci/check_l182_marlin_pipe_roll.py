#!/usr/bin/env python3
"""Local causal controls for the standalone-Marlin outer-pipe experiment."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
COLLECTIVE = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
UNIT = ROOT / "benchmarks/lowbit_dense_unit.inc"
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
L169 = ROOT / "dev/fold_derivation/run_l169_standalone_marlin_unit.sh"
RUNNER = ROOT / "tools/run_dense_marlin_m8_pipe_roll_acu_box.sh"
REPORT_PATH = ROOT / "dev/fold_derivation/l182_marlin_pipe_roll_report.py"
RUN_NONCE = time.time_ns()


def load_report():
    spec = importlib.util.spec_from_file_location("l182_report", REPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = load_report()


def audit(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    collective = files["collective"]
    unit = files["unit"]
    bench = files["bench"]
    runner = files["runner"]
    l169 = files["l169"]
    required_collective = (
        "#define PPU_MARLIN_PIPE_ROLL 0",
        "PPU_DEFS=PPU_MARLIN_PIPE_ROLL=1 TARGET=test_lowbit_dense_marlin_m8_ab ./build.sh",
        "static constexpr int PipeRollMode = PPU_MARLIN_PIPE_ROLL;",
        "static constexpr bool OuterPipeRolled = PipeRollMode != 0;",
        "static constexpr bool InnerLoopRolled = PipeRollMode == 2;",
        "#if PPU_MARLIN_PIPE_ROLL == 0\n      #pragma unroll\n#else\n      #pragma unroll 1\n#endif\n      for (int pipe = 0; pipe < Stages;)",
        "#if PPU_MARLIN_PIPE_ROLL == 2\n        #pragma unroll 1\n#else\n        #pragma unroll\n#endif\n        for (int inner = 0; inner < BInnerIters; ++inner)",
    )
    for token in required_collective:
        if token not in collective:
            bad.append(f"collective lost {token!r}")
    if "#pragma unroll PPU_MARLIN_PIPE_ROLL" in collective:
        bad.append("collective relies on implementation-defined macro expansion inside #pragma")
    for token in (
        "Kernel::CollectiveMainloop::PipeRollMode ==",
        "PPU_MARLIN_PIPE_ROLL",
    ):
        if token not in unit:
            bad.append(f"generated wrapper lost {token!r}")
    for token in (
        "pipe_roll=%d outer_pipe_rolled=%d inner_loop_rolled=%d",
        "Gemm::CollectiveMainloop::PipeRollMode",
        "Gemm::CollectiveMainloop::InnerLoopRolled",
        "(!defined(DENSE_MARLIN_WK4_AB) || !defined(DENSE_MARLIN_M8_AB))",
    ):
        if token not in bench:
            bad.append(f"runtime identity lost {token!r}")
    for token in (
        'pipe_roll="${QUACTLIZE_L169_PIPE_ROLL:-0}"',
        'defs+=" -DPPU_MARLIN_PIPE_ROLL=$pipe_roll"',
        'variant=m8 pipe_roll=$pipe_roll',
    ):
        if token not in l169:
            bad.append(f"L169 exact-unit route lost {token!r}")
    for token in (
        "MODES=(baseline outer-roll inner-roll-control)",
        "VALUES=(0 1 2)",
        "inner_control=compile-disassembly-only-never-executed",
        'for index in 0 1; do',
        'exec 9>"${OUT}.lock"',
        "flock -n 9",
        "verify_source_identity",
        "binaries=3/3",
        'for repeat in 1 2 3 4 5 6 7 8; do',
        'python3 "$REPORTER"',
        '--baseline-line "${LINES[0]}" --baseline-resource "${RESOURCES[0]}"',
        '--outer-roll-line "${LINES[1]}" --outer-roll-resource "${RESOURCES[1]}"',
        '--inner-roll-control-resource "${RESOURCES[2]}"',
        "primary_metric=Instruction Fetch share of all stall cycles",
        "fetch_share_supported=outer-roll drops >=10.0 absolute percentage-points",
        "fetch_share_unresolved=outer-roll drops >=5.0 and <10.0 absolute percentage-points",
        "fetch_share_falsified=outer-roll drops <5.0 absolute percentage-points",
        "hypothesis_falsifier=static mainloop shrinks >=3.5x but Instruction Fetch share falls <5.0 absolute percentage-points",
    ):
        if token not in runner:
            bad.append(f"box runner lost {token!r}")
    if "BPCS=" in runner or "for bpc in" in runner:
        bad.append("pipe-roll runner broadened beyond the preregistered BPC1 experiment")
    if runner.count('python3 "$REPORTER"') != 1:
        bad.append("box runner must invoke the exact reporter once")
    elif runner.index('python3 "$REPORTER"') > runner.index('ACU_CMD=('):
        bad.append("box runner can profile before the static codegen admission gate")
    return bad


def inst(address: int) -> str:
    return f" {address:04x}: 00 00 00 00 00 00 00 00       v.add.i32 vreg1, vreg2, vreg3"


def disassembly(
    source: Path,
    count: int,
    *,
    source_bound: bool = True,
    source_line: int | None = None,
    address_base: int = 0,
) -> str:
    line = source_line or REPORT.unique_line(source, "while (k_tiles_remaining > 0)")
    rows = []
    if source_bound:
        rows.append(f'File "{source}", line {line}')
    rows.extend(inst(address_base + i * 8) for i in range(count))
    return "\n".join(rows) + "\n"


def resource(registers: int, spill: int = 0) -> str:
    return (
        f"Registers: {registers}\n"
        f"Stack Frame: 0 bytes, Spill Stores: 0 bytes, Spill Loads: {spill} bytes\n"
    )


def parser_controls() -> None:
    out = Path("/workspace/quactlize-l182-controls")
    out.mkdir(parents=True, exist_ok=True)

    def mode(name: str, count: int, regs: int, spill: int = 0,
             source_bound: bool = True) -> dict[str, object]:
        line = out / f"{name}.line.txt"
        res = out / f"{name}.resource.txt"
        line.write_text(disassembly(COLLECTIVE, count, source_bound=source_bound))
        res.write_text(resource(regs, spill))
        return REPORT.classify_mode(line, res, COLLECTIVE)

    good = {
        "baseline": mode("baseline", 8, 124),
        "outer-roll": mode("outer", 2, 120),
        "inner-roll-control": mode("inner", 1, 130),
    }
    verdict = REPORT.compare(good)
    if verdict["baseline_to_outer_static_ratio"] != 4.0:
        raise RuntimeError(f"synthetic 4x footprint drifted: {verdict}")

    controls = []
    controls.append((
        "compiler-ignored-outer-pragma",
        {**good, "outer-roll": mode("outer-ignored", 8, 120)},
    ))
    controls.append((
        "inner-resource-check-never-red",
        {**good, "inner-roll-control": mode("inner-no-pressure", 1, 124)},
    ))
    for label, planted in controls:
        try:
            REPORT.compare(planted)
        except REPORT.ReportError:
            continue
        raise RuntimeError(f"parser control escaped: {label}")
    try:
        mode("missing-source-binding", 2, 120, source_bound=False)
    except REPORT.ReportError:
        pass
    else:
        raise RuntimeError("source-unbound whole-symbol instructions entered the mainloop numerator")
    no_local = out / "no-local.resource.txt"
    no_local.write_text("Registers: 120\n")
    try:
        REPORT.parse_resource(no_local.read_text())
    except REPORT.ReportError:
        pass
    else:
        raise RuntimeError("missing spill/stack evidence was treated as zero")
    duplicate_line = out / "duplicate-pc.line.txt"
    duplicate_line.write_text(
        disassembly(COLLECTIVE, 1) + disassembly(COLLECTIVE, 1)
    )
    duplicate_resource = out / "duplicate-pc.resource.txt"
    duplicate_resource.write_text(resource(120))
    try:
        REPORT.classify_mode(duplicate_line, duplicate_resource, COLLECTIVE)
    except REPORT.ReportError:
        pass
    else:
        raise RuntimeError("duplicate exact-symbol PCs inflated the static footprint")

    start = REPORT.unique_line(COLLECTIVE, "while (k_tiles_remaining > 0)")
    end = REPORT.unique_line(COLLECTIVE, "a_pointer += AGlobalOuter * Stages;")
    mixed_line = out / "mixed-range.line.txt"
    mixed_line.write_text(
        disassembly(COLLECTIVE, 2, source_line=start)
        + disassembly(COLLECTIVE, 5, source_line=end + 10, address_base=0x100)
    )
    mixed_resource = out / "mixed-range.resource.txt"
    mixed_resource.write_text(resource(120))
    mixed = REPORT.classify_mode(mixed_line, mixed_resource, COLLECTIVE)
    if mixed["mainloop_static_instructions"] != 2 or mixed["whole_symbol_static_instructions"] != 7:
        raise RuntimeError(f"out-of-range instructions entered the mainloop numerator: {mixed}")
    outside_line = out / "outside-only.line.txt"
    outside_line.write_text(disassembly(COLLECTIVE, 2, source_line=end + 10))
    try:
        REPORT.classify_mode(outside_line, mixed_resource, COLLECTIVE)
    except REPORT.ReportError:
        pass
    else:
        raise RuntimeError("out-of-range-only source binding was accepted as the mainloop")


def compile_mode(mode: int) -> str:
    env = {
        "QUACTLIZE_L169_VARIANT": "m8",
        "QUACTLIZE_L169_PIPE_ROLL": str(mode),
        "QUACTLIZE_L169_OUT": (
            f"/workspace/quactlize-l169-m8-pipe-{os.getpid()}-{RUN_NONCE}-{mode}"
        ),
    }
    merged = os.environ.copy()
    merged.update(env)
    run = subprocess.run(
        ["bash", str(L169)], cwd=ROOT, env=merged, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    witness = f"[l169] PASS: variant=m8 pipe_roll={mode}"
    if run.returncode != 0 or witness not in run.stdout:
        raise RuntimeError(
            f"exact generated-unit mode {mode} returned {run.returncode} or lacked {witness!r}\n"
            + run.stdout[-2400:]
        )
    return witness


def main() -> int:
    files = {
        "collective": COLLECTIVE.read_text(),
        "unit": UNIT.read_text(),
        "bench": BENCH.read_text(),
        "l169": L169.read_text(),
        "runner": RUNNER.read_text(),
    }
    bad = audit(files)
    plants = (
        ("collective", "default-flipped", "#define PPU_MARLIN_PIPE_ROLL 0", "#define PPU_MARLIN_PIPE_ROLL 1"),
        ("collective", "outer-roll-severed", "#pragma unroll 1\n#endif\n      for (int pipe", "#pragma unroll\n#endif\n      for (int pipe"),
        ("collective", "inner-also-rolled-in-mode1", "#if PPU_MARLIN_PIPE_ROLL == 2", "#if PPU_MARLIN_PIPE_ROLL == 1"),
        ("unit", "generated-type-witness-removed", "Kernel::CollectiveMainloop::PipeRollMode ==", "Kernel::CollectiveMainloop::PipeRollMode !="),
        ("bench", "runtime-identity-removed", "pipe_roll=%d outer_pipe_rolled=%d inner_loop_rolled=%d", "pipe_roll=unknown"),
        ("runner", "bpc-broadened", "VALUES=(0 1 2)", "VALUES=(0 1 2)\nBPCS=(1 2)"),
        ("runner", "reporter-call-removed", 'python3 "$REPORTER"', 'true # reporter removed'),
        ("runner", "inner-control-executed", "for index in 0 1; do", "for index in 0 1 2; do"),
    )
    for owner, label, old, new in plants:
        if files[owner].count(old) != 1:
            bad.append(f"plant {label} seam occurs {files[owner].count(old)} times")
            continue
        planted = dict(files)
        planted[owner] = planted[owner].replace(old, new, 1)
        if not audit(planted):
            bad.append(f"plant escaped: {label}")
    if bad:
        print("[l182:local] FAIL: " + "; ".join(bad), file=sys.stderr)
        return 1
    try:
        parser_controls()
        # nvcc's `-cuda` path may create basename-derived intermediates outside
        # QUACTLIZE_L169_OUT.  Keep the three same-source controls sequential:
        # parallel success would be weaker evidence if their compiler-driver
        # scratch files raced even though the explicit output roots differ.
        witnesses = [compile_mode(mode) for mode in (0, 1, 2)]
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if syntax.returncode != 0:
            raise RuntimeError("box runner syntax failed:\n" + syntax.stdout)
    except (OSError, RuntimeError, REPORT.ReportError) as exc:
        print(f"[l182:local] FAIL: {exc}", file=sys.stderr)
        return 1
    print("[l182:local] exact-unit: " + ", ".join(witnesses))
    print(
        "[l182:local] PASS: default/outer-only/inner-control routes reach the exact m8 device body; "
        "negative_controls=14/14_RED mixed-range=PASS; PPU static footprint/register/spill remains a mandatory box compile-only postcondition"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

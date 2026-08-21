#!/usr/bin/env python3
"""The R that PPU_A_PACK declares legal must actually compile.

WHY THIS EXISTS. quactlize_mma_mixed_input.hpp used to declare

    static_assert(kAPackRows >= 1 && kAPackRows <= 16,
                  "PPU_A_PACK=R requires 1 <= R <= the 16-row swzl instruction footprint")

but R=16 did not build: a later static_assert, aPackDisjoint(), fired with "packed first-R-row runs collide --
fix the derived pitch". The public ceiling was narrowed to the range proved across the compiled TileM geometries;
this gate keeps the declaration and the implementation from drifting apart again. The contradiction went unnoticed
because the only way to find it was to compile each R by hand, and #44's acceptance criteria omitted that step.

WHAT IT CHECKS, and it is DERIVED so it follows the source rather than restating it:

    * read the declared ceiling out of the static_assert itself
    * R=1 must compile              -- the floor the same assert declares
    * R=<declared> must compile     -- the claim under test
    * R=<declared>+1 must NOT       -- otherwise the ceiling is not a ceiling and the assert is decorative

Three compiles, ~22 s each. Not in the default local tier: it is the slowest check in ci/ and it guards a switch
that is off in every shipping build. Run it after touching the packed-A provider, its pitch derivation, or the
bound itself -- which is exactly when the two asserts can drift apart again.

A declared bound that does not build is worse than a lower bound honestly stated: it invites a caller to set a
value the compiler will reject, and the rejection names the PITCH rather than the bound, so the reader concludes
the pitch is broken when the actual defect may be that 16 was never reachable.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLLECTIVE = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
CONSUMER = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
STUBS = ROOT / "dev/fold_derivation/stub_inc"
ACTLIZE = ROOT / "third_party/actlize"

# The declared ceiling, read off the assert that declares it. `kAPackRows <= N` inside the same condition as
# `kAPackRows >= 1` -- anchored on both so a different assert about the same symbol cannot be mistaken for it.
BOUND_RE = re.compile(r"kAPackRows\s*>=\s*1\s*&&\s*kAPackRows\s*<=\s*(\d+)")

# Vendor asm in mma_ppu0015.hpp does not parse under nvcc; that floor is constant and unrelated to R, so the
# signal is the count of errors that are NOT it.
ASM_NOISE = "asm operand type size"


def compile_at(r: int) -> tuple[int, str]:
    """-> (non-asm error count, first such diagnostic)."""
    cmd = [
        "nvcc", "-std=c++17", "-arch=sm_80", "--expt-relaxed-constexpr",
        "-D__HGGCCC__", "-DPPU_FORCE_INSTANTIATE=1", f"-DPPU_A_PACK={r}",
        "-include", str(STUBS / "ppu_arch_shim.h"), "-Xcudafe", "--error_limit=100000",
        "-I", str(STUBS), "-I", str(ACTLIZE / "include"), "-I", str(ACTLIZE / "tools/util/include"),
        "-I", str(ROOT / "tests"), "-I", str(ROOT / "benchmarks"),
        "-I", str(ROOT / "quactlize/include"), "-I", str(ROOT / "dev"),
        "-cuda", "-o", "/dev/null", "-x", "cu", str(CONSUMER), "-Wno-deprecated-gpu-targets",
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    lines = [ln for ln in (p.stdout + p.stderr).splitlines()
             if ": error" in ln and ASM_NOISE not in ln]
    detail = ""
    for ln in (p.stdout + p.stderr).splitlines():
        m = re.search(r'static assertion failed with "([^"]+)"', ln)
        if m:
            detail = m.group(1)
            break
    if not detail and lines:
        detail = lines[0].strip()
    return len(lines), detail


def main() -> int:
    for path in (COLLECTIVE, CONSUMER):
        if not path.is_file():
            print(f"[a-pack-bound] ERROR: missing {path}")
            return 1
    m = BOUND_RE.search(COLLECTIVE.read_text())
    if m is None:
        # Not a skip. The bound moving out of this shape is exactly when the check stops checking, and a gate that
        # quietly passes when it can no longer find its subject is the failure this repository keeps paying for.
        print(f"[a-pack-bound] ERROR: no `kAPackRows >= 1 && kAPackRows <= N` assert in {COLLECTIVE.name}; "
              "the declared bound moved and this gate can no longer find it")
        return 1
    declared = int(m.group(1))
    print(f"[a-pack-bound] the source declares 1 <= R <= {declared}; compiling the floor, the ceiling and one past it")

    cases = [(1, True), (declared, True), (declared + 1, False)]
    bad = []
    for r, want_ok in cases:
        n, detail = compile_at(r)
        ok = (n == 0)
        verdict = "OK " if ok == want_ok else "BAD"
        print(f"  {verdict} R={r:<3} non-asm errors={n:<3} expected={'compiles' if want_ok else 'rejected'}"
              + (f"  [{detail}]" if detail else ""))
        if ok != want_ok:
            bad.append((r, want_ok, n, detail))

    if not bad:
        print(f"[a-pack-bound] PASS: R=1 and R={declared} build, R={declared + 1} is rejected")
        return 0
    for r, want_ok, n, detail in bad:
        if want_ok:
            print(f"[a-pack-bound] R={r} is DECLARED legal and does not compile ({n} errors): {detail}")
            print("               the two asserts disagree about the same number -- either fix what the second one "
                  "names, or make the declared bound the truth and stop naming a value that cannot be used")
        else:
            print(f"[a-pack-bound] R={r} is past the declared ceiling and compiled anyway -- the bound is decorative")
    return 1


if __name__ == "__main__":
    sys.exit(main())

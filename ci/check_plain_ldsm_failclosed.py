#!/usr/bin/env python3
"""Prove #114's twelve dormant ppu001 plain-LDSM entries fail in C++, not assembly."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ACT = ROOT / "third_party" / "actlize" / "include"
ACT_UTIL = ROOT / "third_party" / "actlize" / "tools" / "util" / "include"
STUB = ROOT / "dev" / "fold_derivation" / "stub_inc"
PROBE = ROOT / "dev" / "fold_derivation" / "plain_ldsm_failclose_probe.cu"
COPY = ACT / "cute" / "arch" / "copy_ppu.hpp"
MEMORY = ACT / "cutlass" / "arch" / "memory_ppu.h"

DIRECT_NAMES = (
    "PPU_U32x1_LDSM_N", "PPU_U32x2_LDSM_N", "PPU_U32x4_LDSM_N",
    "PPU_U16x2_LDSM_T", "PPU_U16x4_LDSM_T", "PPU_U16x8_LDSM_T",
)

TC02_FORMS = (
    "x1.m8n8.shared.b16", "x2.m8n8.shared.b16", "x4.m8n8.shared.b16",
    "x1.trans.m8n8.shared.b16", "x2.trans.m8n8.shared.b16", "x4.trans.m8n8.shared.b16",
)


def fail(message: str) -> int:
    print(f"[plain-ldsm-failclose] FAIL: {message}")
    return 1


def compile_probe(
    nvcc: str, out: Path, arch: int, calls: int, helpers: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            nvcc, "-std=c++17", "-arch=sm_80", "-w", "-D__HGGCCC__",
            "--expt-relaxed-constexpr", "--ptx",
            f"-DPPU_PLAIN_LDSM_PROBE_ARCH={arch}",
            f"-DPPU_PLAIN_LDSM_PROBE_CALLS={calls}",
            f"-DPPU_PLAIN_LDSM_PROBE_HELPERS={helpers}",
            f"-I{STUB}", f"-I{ACT}", f"-I{ACT_UTIL}",
            "-o", str(out), str(PROBE),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def main() -> int:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return fail("nvcc is required: this gate distinguishes a C++ diagnostic from an assembler diagnostic")
    if not PROBE.is_file():
        return fail(f"missing probe {PROBE.relative_to(ROOT)}")

    copy = COPY.read_text()
    memory = MEMORY.read_text()
    if "ppu.tc01.ex.ldmatrix" in copy or "ppu.tc01.ex.ldmatrix" in memory:
        return fail("assembler-rejected `.ex.` spelling remains in a vendor header")
    if copy.count("= delete;") != 6 or memory.count("= delete;") != 6:
        return fail("expected exactly six deleted direct atoms and six deleted CUTLASS specializations")
    for name in DIRECT_NAMES:
        if copy.count(name) < 2:
            return fail(f"direct atom {name} is not present in both ppu001 fail-close and retained API arms")
    reason = "ppu001 plain LDSM is disabled: its SDK grammar is unproved"
    if copy.count(reason) != 2:
        return fail("both legacy copy_ldsm helpers must carry the dependent static-assert reason")

    with tempfile.TemporaryDirectory(prefix="quactlize-plain-ldsm-") as td:
        tmp = Path(td)

        unused = compile_probe(nvcc, tmp / "ppu100-unused.ptx", 100, 0)
        if unused.returncode != 0:
            line = next((x for x in (unused.stdout + unused.stderr).splitlines() if "error:" in x), "no diagnostic")
            return fail(f"ppu001 headers fail even when no plain atom is called: {line}")

        rejected = compile_probe(nvcc, tmp / "ppu100-calls.ptx", 100, 1)
        rejected_log = rejected.stdout + rejected.stderr
        deleted = rejected_log.count("deleted function")
        rejected_errors = [line for line in rejected_log.splitlines() if "error:" in line]
        if (rejected.returncode == 0 or deleted != 12 or len(rejected_errors) != 12
                or any("deleted function" not in line for line in rejected_errors)):
            return fail("ppu001 call-all must fail with exactly 12 errors, all deleted-function diagnostics; "
                        f"rc={rejected.returncode} deleted={deleted} errors={len(rejected_errors)}")
        for forbidden in ("token recognition", "ptxas", "assembler", ".ex."):
            if forbidden in rejected_log:
                return fail(f"ppu001 negative reached or reported the assembler path: {forbidden!r}")
        for name in DIRECT_NAMES:
            if name not in rejected_log:
                return fail(f"ppu001 negative log never names {name}")
        for layout in ("RowMajor", "ColumnMajor"):
            for count in (1, 2, 4):
                if f"Layout=cutlass::layout::{layout}, MatrixCount={count}" not in rejected_log:
                    return fail(f"ppu001 negative log misses CUTLASS {layout} x{count}")

        helpers = compile_probe(nvcc, tmp / "ppu100-helpers.ptx", 100, 0, 1)
        helper_log = helpers.stdout + helpers.stderr
        helper_reasons = helper_log.count(reason)
        helper_errors = [line for line in helper_log.splitlines() if "error:" in line]
        if (helpers.returncode == 0 or helper_reasons != 2 or len(helper_errors) != 2
                or any(reason not in line for line in helper_errors)):
            return fail("legacy helpers must fail with exactly two errors, both the explicit static-assert reason; "
                        f"rc={helpers.returncode} reasons={helper_reasons} errors={len(helper_errors)}")
        if any(x in helper_log for x in ("token recognition", "ptxas", "assembler", ".ex.")):
            return fail("legacy helper negative reached or reported the assembler path")

        retained = compile_probe(nvcc, tmp / "ppu150-calls.ptx", 150, 1)
        if retained.returncode != 0:
            line = next((x for x in (retained.stdout + retained.stderr).splitlines() if "error:" in x), "no diagnostic")
            return fail(f"ppu0015 call-all no longer compiles: {line}")
        ptx = (tmp / "ppu150-calls.ptx").read_text()
        tc02 = ptx.count("ppu.tc02.ldmatrix")
        if tc02 != 12:
            return fail(f"ppu0015 PTX must retain all 12 tc02 calls, found {tc02}")
        for form in TC02_FORMS:
            count = ptx.count(f"ppu.tc02.ldmatrix.sync.aligned.{form}")
            if count != 2:
                return fail(f"ppu0015 PTX must retain CuTe+CUTLASS {form} exactly twice, found {count}")

    print("[plain-ldsm-failclose] PASS -- ppu001 unused header compiles; 12 calls stop at 12 C++ deleted-function "
          "diagnostics and 2 legacy helpers at 2 reasoned static_asserts, with no assembler path; ppu0015 retains "
          "all six tc02 forms twice and exact direct-function ABI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

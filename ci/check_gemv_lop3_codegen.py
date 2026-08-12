#!/usr/bin/env python3
"""Run L145 and prove its production-route binding rejects a planted drift.

The live disassembly is the evidence.  The plant is what prevents the explicit-instantiation file from becoming a
parallel model that remains green after ppu_backend.cu changes to a different shipping specialization.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "dev" / "fold_derivation" / "run_l145_gemv_lop3_codegen.sh"
BACKEND = ROOT / "quactlize" / "csrc" / "device" / "ppu_backend.cu"


def run(env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=merged,
                          text=True, capture_output=True)


def main() -> int:
    live = run()
    combined = live.stdout + live.stderr
    if live.returncode == 3:
        print(next((line for line in combined.splitlines() if "L145 SKIP:" in line),
                   "L145 SKIP: codegen runner returned 3 without a reason"))
        return 3
    required = (
        "mask_lop3=64 magic_lop3=64 shifts=48 [16,16,16] offset_hadd2=64",
        "extraction=176/64 = 2.75 integer instructions/pair",
        "#28 premise holds on RTX 5090 only",
    )
    if live.returncode or any(text not in combined for text in required):
        print(combined, end="")
        print("[gemv-lop3-codegen] FAIL: live production disassembly did not establish the pinned verdict")
        return 1

    original = BACKEND.read_text()
    anchor = "case 12: return RUN(Int4,  16, 128);"
    if original.count(anchor) != 1:
        print(f"[gemv-lop3-codegen] FAIL: cannot plant production-route drift; anchor count={original.count(anchor)}")
        return 1
    with tempfile.TemporaryDirectory(prefix="quactlize-l145-contract.") as td:
        planted = Path(td) / "ppu_backend.cu"
        planted.write_text(original.replace(anchor, "case 12: return RUN(Int4,  32, 128);"))
        red = run({"L145_BACKEND": str(planted)})
    red_log = red.stdout + red.stderr
    expected = "production qtype=12 device route is no longer exactly Int4/s16/t128"
    if red.returncode != 1 or expected not in red_log:
        print(red_log, end="")
        print("[gemv-lop3-codegen] FAIL: planted shipping-specialization drift was not rejected at the binding")
        return 1

    print(next(line for line in live.stdout.splitlines() if line.startswith("L145 SASS ")))
    print(next(line for line in live.stdout.splitlines() if line.startswith("L145 extraction=")))
    print("[gemv-lop3-codegen] PASS: real sm_120 instance unfused; planted production-route drift EXPECTED_RED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

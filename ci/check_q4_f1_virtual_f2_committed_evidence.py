#!/usr/bin/env python3
"""Regenerate locally or consume SHA-bound Q4 F1/virtual-F2 host evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "dev/fold_derivation/q4_f1_virtual_f2.expected.txt"
RUNNERS = (
    ROOT / "dev/fold_derivation/run_l224_q4_f1_virtual_f2.sh",
    ROOT / "dev/fold_derivation/run_l225_q4_f1_virtual_f2_type.sh",
    ROOT / "dev/fold_derivation/run_l226_q4_f1_virtual_f2_body.sh",
)
CAPABILITY = ROOT / "dev/fold_derivation/nvidia_nvcc_or_skip.sh"
BOX_RUNNER = ROOT / "tools/run_scalefirst_q4k_f1_virtual_f2_box.sh"
STUB_FP8 = ROOT / "dev/fold_derivation/stub_inc/hggc_fp8.h"

PREFIXES = (
    "L224_ROW ", "L224_NEGATIVE ", "L224_Q4_F1_VIRTUAL_F2 ",
    "[l224-runner] PASS", "L225_Q4_F1_VIRTUAL_F2_TYPE ",
    "[l225-runner] PASS:", "[l226] PASS ",
)


def canonical_lines(output: str) -> list[str]:
    lines: list[str] = []
    for raw in output.splitlines():
        if raw.startswith(PREFIXES):
            line = raw.split("; artifacts=", 1)[0]
            line = line.split(" artifacts=", 1)[0]
            lines.append(line)
    return lines


def validate(lines: list[str]) -> str | None:
    if len(lines) != 11:
        return f"expected 11 canonical lines, found {len(lines)}"
    rows = [line for line in lines if line.startswith("L224_ROW ")]
    if len(rows) != 4 or any("verdict=PASS" not in line for line in rows):
        return "L224 positive denominator is not exactly four PASS rows"
    required_rows = (
        "tag=t64-w32", "tag=t64-w64", "tag=t128", "tag=t256",
        "native_f2_scatter_diff=32", "native_f2_scatter_diff=96",
    )
    for token in required_rows:
        if sum(token in line for line in rows) != 1:
            return f"L224 row census drifted: {token}"
    if any("identity_scatter_diff=0" not in line or "physical_missing=0" not in line
           for line in rows):
        return "L224 identity/physical coverage is no longer exact"
    negatives = [line for line in lines if line.startswith("L224_NEGATIVE ")]
    if len(negatives) != 2 or any("fired=1" not in line for line in negatives):
        return "wrong-thread/missing-delivery negatives are not both red"
    required_once = (
        "weight_byte_multiplier=1 runtime_branches=0 t32=DEFERRED_MACROSTEP",
        "default=UNCHANGED physical=F1 compute=F2",
        "t64=TYPE_IDENTICAL t128_t256=MMA_ONLY smem_delta=0 runtime_branch_delta=0",
        "T64 identity, T128/T256 MMA-only, T32 RED",
        "body=REACHED vendor_asm_errors=84 nonvendor=0",
    )
    for token in required_once:
        if sum(token in line for line in lines) != 1:
            return f"proof invariant drifted: {token}"
    return None


def validate_runner_boundary(box: str, standalone: list[str]) -> str | None:
    for name in ("run_l224_q4_f1_virtual_f2.sh", "run_l225_q4_f1_virtual_f2_type.sh",
                 "run_l226_q4_f1_virtual_f2_body.sh"):
        if f'bash "$root/dev/fold_derivation/{name}"' in box:
            return f"box runner directly executes NVIDIA/stub oracle: {name}"
    required = {
        'git -C "$root" show "$sha:dev/fold_derivation/q4_f1_virtual_f2.expected.txt"': 1,
        "check_q4_f1_virtual_f2_committed_evidence.py": 2,
        "--committed-only --evidence": 1,
        "fresh-box-execution=0": 1,
    }
    for token, count in required.items():
        if box.count(token) != count:
            return f"box committed-evidence seam drifted: {token}"
    for source in standalone:
        if source.count("nvidia_nvcc_or_skip.sh") != 1:
            return "standalone host oracle lacks the real compiler-capability boundary"
    if STUB_FP8.exists():
        return "fake stub_inc/hggc_fp8.h would shadow the real PPU SDK header"
    return None


def nvidia_probe(nvcc: str) -> tuple[bool, str]:
    tmp = Path("/tmp") / f"quactlize-q4-vf2-nvcc-probe-{os.getpid()}"
    tmp.mkdir(exist_ok=False)
    src, binary = tmp / "probe.cu", tmp / "probe"
    src.write_text(
        "#include <cuda_fp16.h>\n"
        "__global__ void k(__half* p){ *p = __hadd(p[threadIdx.x], p[blockIdx.x]); }\n"
        "int main(){ return 0; }\n"
    )
    proc = subprocess.run([nvcc, "-std=c++17", "-arch=sm_80", str(src), "-o", str(binary)],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    shutil.rmtree(tmp)
    diagnostic = next((line.strip() for line in proc.stdout.splitlines()
                       if ": error:" in line or ": fatal error:" in line), "probe failed")
    return proc.returncode == 0, diagnostic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--committed-only", action="store_true")
    parser.add_argument("--evidence", type=Path, default=EXPECTED)
    args = parser.parse_args()
    try:
        expected = args.evidence.read_text().splitlines()
        box = BOX_RUNNER.read_text()
        standalone = [path.read_text() for path in RUNNERS]
    except OSError as exc:
        print(f"[q4-vf2-committed] FAIL: cannot read authority: {exc}")
        return 1
    why = validate(expected)
    if why:
        print(f"[q4-vf2-committed] FAIL: malformed evidence: {why}")
        return 1
    why = validate_runner_boundary(box, standalone)
    if why:
        print(f"[q4-vf2-committed] FAIL: {why}")
        return 1

    # Boundary/control plants: a direct box execution, one escaped stub, and
    # one lost denominator must each invalidate the evidence contract.
    planted_box = box + '\nbash "$root/dev/fold_derivation/run_l224_q4_f1_virtual_f2.sh"\n'
    if validate_runner_boundary(planted_box, standalone) is None:
        print("[q4-vf2-committed] FAIL: direct-execution plant escaped")
        return 1
    if validate(expected[:-1]) is None:
        print("[q4-vf2-committed] FAIL: missing-evidence plant escaped")
        return 1
    planted_value = [line.replace("weight_byte_multiplier=1", "weight_byte_multiplier=2")
                     for line in expected]
    if validate(planted_value) is None:
        print("[q4-vf2-committed] FAIL: weight-amplification plant escaped")
        return 1

    if not args.committed_only:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            print("[q4-vf2-committed] SKIP: NVIDIA nvcc unavailable; evidence validated only")
            return 2
        capable, diagnostic = nvidia_probe(nvcc)
        if not capable:
            print("[q4-vf2-committed] SKIP: nvcc is not a complete NVIDIA CUDA compiler; "
                  f"evidence validated only ({diagnostic[:180]})")
            return 2
        actual: list[str] = []
        for runner in RUNNERS:
            proc = subprocess.run(["bash", str(runner)], cwd=ROOT, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if proc.returncode:
                print(f"[q4-vf2-committed] FAIL: {runner.name} rc={proc.returncode}")
                print("\n".join(proc.stdout.splitlines()[-100:]))
                return 1
            actual += canonical_lines(proc.stdout)
        why = validate(actual)
        if why:
            print(f"[q4-vf2-committed] FAIL: regenerated evidence malformed: {why}")
            return 1
        if actual != expected:
            print("[q4-vf2-committed] FAIL: regenerated evidence differs from commit")
            print("\n".join(actual))
            return 1

    mode = "validated" if args.committed_only else "regenerated"
    print(f"[q4-vf2-committed] PASS: 11-line proof {mode}; four positives, two RED controls, "
          "default/type/body/performance invariants bound; box uses fresh hgcc with fresh_box_execution=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

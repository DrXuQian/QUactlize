#!/usr/bin/env python3
"""Regenerate or consume the SHA-bound Q8 resident-layout evidence.

The L208 oracle intentionally compiles with NVIDIA nvcc plus the repository's
small PPU API stubs.  A PPU box's `nvcc` delegates device preprocessing to
ppu_clang++, so neither that stub fixture nor the all-real SDK fixture is a
valid execution environment there.  The box therefore consumes these exact
local results from the commit it builds; shipping device code is still built
fresh through hgcc immediately afterwards.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "dev/fold_derivation/run_l208_q8_emit_layout.sh"
EXPECTED = ROOT / "dev/fold_derivation/l208_q8_emit_layout.expected.txt"
BOX_RUNNER = ROOT / "tools/run_prefill_sweep_box.sh"
ACTLIZE_CMAKE = ROOT / "third_party/actlize/CMakeLists.txt"
PPU_TOOLCHAIN = ROOT / "third_party/actlize/cmake/PPUToolchain.cmake"
Q8_CANDIDATES = ROOT / "benchmarks/prefill_q8_candidates.inc"

PREFIXES = (
    "[l208 emit] ",
    "[l208 placement] ",
    "[l208 q8-value] ",
    "[l208] ",
    "[l208-runner] PASS ",
)


def canonical_lines(output: str) -> list[str]:
    lines: list[str] = []
    for raw in output.splitlines():
        if raw.startswith(PREFIXES):
            # The artifact location is operational evidence, not a numerical
            # property, and contains the process id by design.
            lines.append(raw.split(" artifacts=", 1)[0])
    return lines


def candidate_denominator(source: str) -> int:
    declared = []
    active = 0
    for line in source.splitlines():
        if line.startswith("#define PREFILL_Q8_EXPECTED_ROWS "):
            declared.append(int(line.split()[-1]))
        if line.startswith("PREFILL_Q8_CANDIDATE("):
            active += 1
    if len(declared) != 1 or declared[0] != active or active <= 0:
        raise ValueError(f"Q8 denominator authority disagrees: declared={declared} active={active}")
    return active


def validate(lines: list[str], expected_rows: int) -> str | None:
    if len(lines) != 12:
        return f"expected 12 canonical lines, found {len(lines)}"
    required_once = (
        "mismatch=1 holes=1 duplicates=1",
        f"candidates={expected_rows - 1} canonical=A32/F1",
        "wrong_perm=EXPECTED_RED",
        "missing_candidate=EXPECTED_RED",
    )
    for token in required_once:
        if sum(token in line for line in lines) != 1:
            return f"negative-control closure drifted: {token}"
    if sum(f"candidates={expected_rows} canonical=A32/F1" in line for line in lines) != 2:
        return f"positive {expected_rows}-row denominator is not present in both relevant arms"
    if sum("checks=1024 bias_bad=0 value_bad=0" in line for line in lines) != 3:
        return "Q8 value oracle did not run in all three arms"
    return None


def validate_box_route(source: str) -> str | None:
    if 'bash "$root/dev/fold_derivation/run_l208_q8_emit_layout.sh"' in source:
        return "box runner executes the local nvcc/stub oracle"
    required_counts = {
        'git -C "$root" show "$sha:dev/fold_derivation/l208_q8_emit_layout.expected.txt"': 1,
        "check_l208_q8_committed_evidence.py": 2,  # tracked authority + invocation
        "--committed-only --evidence": 1,
        "fresh-box-execution=0": 1,
    }
    for token, count in required_counts.items():
        if source.count(token) != count:
            return f"box committed-evidence seam drifted: {token}"
    return None


def validate_shipping_include_graph(cmake: str, toolchain: str) -> str | None:
    for token in ('"-I${PPU_SDK_INCLUDE}"', '"-I${PPU_SDK_TARGETS_INCLUDE}"'):
        if cmake.count(token) != 1:
            return f"shipping device include authority drifted: {token}"
    command_seam = "${CUTLASS_PPU_DEV_INCLUDE_FLAGS}"
    if toolchain.count(command_seam) != 1:
        return "hgcc custom command no longer consumes the device include authority exactly once"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--committed-only", action="store_true")
    parser.add_argument("--evidence", type=Path, default=EXPECTED)
    args = parser.parse_args()

    try:
        expected_lines = args.evidence.read_text().splitlines()
        expected_rows = candidate_denominator(Q8_CANDIDATES.read_text())
    except OSError as exc:
        print(f"[l208-committed] FAIL: cannot read evidence: {exc}")
        return 1
    except ValueError as exc:
        print(f"[l208-committed] FAIL: {exc}")
        return 1
    why = validate(expected_lines, expected_rows)
    if why:
        print(f"[l208-committed] FAIL: committed evidence is malformed: {why}")
        return 1

    try:
        box_source = BOX_RUNNER.read_text()
        actlize_cmake = ACTLIZE_CMAKE.read_text()
        ppu_toolchain = PPU_TOOLCHAIN.read_text()
    except OSError as exc:
        print(f"[l208-committed] FAIL: cannot read build-route authority: {exc}")
        return 1
    why = validate_box_route(box_source)
    if why:
        print(f"[l208-committed] FAIL: {why}")
        return 1
    why = validate_shipping_include_graph(actlize_cmake, ppu_toolchain)
    if why:
        print(f"[l208-committed] FAIL: {why}")
        return 1

    # Causal controls for the boundary that failed on the box: executing L208
    # there, or dropping the SDK target include from shipping hgcc, must each
    # turn the source contract red for exactly that reason.
    planted_box = box_source.replace("--committed-only --evidence", "--evidence", 1)
    if validate_box_route(planted_box) is None:
        print("[l208-committed] FAIL: removing committed-only did not break the box boundary")
        return 1
    planted_cmake = actlize_cmake.replace('"-I${PPU_SDK_TARGETS_INCLUDE}"', "", 1)
    if validate_shipping_include_graph(planted_cmake, ppu_toolchain) is None:
        print("[l208-committed] FAIL: removing the SDK target include did not break shipping admission")
        return 1

    if not args.committed_only:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            print("[l208-committed] SKIP: NVIDIA nvcc is unavailable; committed evidence was validated only")
            return 2
        probe_dir = Path("/workspace") / f"quactlize-l208-nvcc-probe-{os.getpid()}"
        probe_dir.mkdir(parents=True, exist_ok=False)
        probe_src = probe_dir / "probe.cu"
        probe_bin = probe_dir / "probe"
        probe_src.write_text(
            "#include <cuda_fp16.h>\n"
            "__global__ void k(__half* p){ *p = __hadd(p[threadIdx.x], p[blockIdx.x]); }\n"
            "int main(){ return 0; }\n"
        )
        probe = subprocess.run(
            [nvcc, "-std=c++17", "-arch=sm_80", str(probe_src), "-o", str(probe_bin)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        shutil.rmtree(probe_dir)
        if probe.returncode:
            diagnostic = next(
                (line.strip() for line in probe.stdout.splitlines()
                 if ": error:" in line or ": fatal error:" in line),
                "device compile probe failed",
            )
            print(
                "[l208-committed] SKIP: this nvcc is not a complete NVIDIA device compiler; "
                f"committed evidence was validated only ({diagnostic[:180]})"
            )
            return 2
        env = os.environ.copy()
        env.setdefault("QUACTLIZE_L208_OUT", "/workspace/quactlize-l208-committed-check")
        proc = subprocess.run(
            ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if proc.returncode:
            print(f"[l208-committed] FAIL: local oracle rc={proc.returncode}")
            print("\n".join(proc.stdout.splitlines()[-80:]))
            return 1
        actual_lines = canonical_lines(proc.stdout)
        why = validate(actual_lines, expected_rows)
        if why:
            print(f"[l208-committed] FAIL: regenerated evidence is malformed: {why}")
            return 1
        if actual_lines != expected_lines:
            print("[l208-committed] FAIL: regenerated evidence differs from committed evidence")
            print("\n".join(actual_lines))
            return 1

    mode = "validated" if args.committed_only else "regenerated"
    print(
        f"[l208-committed] PASS: exact 12-line Q8 evidence {mode}; "
        f"{expected_rows}-row layout/value proof and four causal RED controls remain closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

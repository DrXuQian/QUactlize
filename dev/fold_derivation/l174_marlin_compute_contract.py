#!/usr/bin/env python3
"""Local source/semantic contract for the standalone Marlin compute row.

This gate intentionally separates two independent references:

* Awesome-CuTe is the strict dequant scheduling, packed-N traversal and
  logical-MMA source: D0,D1,S0,S1,M8,M8.
* PPU classic independently anchors the same constants and arithmetic, while
  its measured default body deliberately remains D0,S0,D1,S1,M16.  This gate
  prints that difference instead of falsely claiming both orders are equal.

Final PPU opcode selection remains a disassembly postcondition.  This local
gate proves the source shape handed to that compiler and all 65,536 relevant
four-nibble dequant inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MARLIN_ROOT = ROOT.parent
COLLECTIVE = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
LOAD = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_load_ppu.hpp"
DEQUANT = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_dequant_ppu.hpp"
MMA = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_mma_ppu.hpp"
CLASSIC = MARLIN_ROOT / "marlin_classic_ppu.cuh"
AWESOME = MARLIN_ROOT / "ref/awesome-cute/gemm/marlin_gemm/marlin_cute_trait.h"

EXPECTED_CLASSIC_SHA = "5bcc5647371237b5588bc835704c9a42c416464891394107feb5c6b31453577c"
EXPECTED_AWESOME_SHA = "a97c35c491ff94066f2ce61d91b1af165a99fd50848cf6732ff660fc663878c7"
EXPECTED_AWESOME_REV = "9f166294bd639cad712a531ac6a5e7aeb983ed37"
EXPECTED_CONSTANTS = (
    0x000F000F, 0x00F000F0, 0x64006400,
    0x64086408, 0x2C002C00, 0xD480D480,
)


def die(plant: str, reason: str) -> "NoReturn":
    if plant == "none":
        print(f"[l174] FAIL: {reason}", file=sys.stderr)
    else:
        print(
            f"[l174:red] plant={plant} caught=1 reason={reason} result=RED",
            file=sys.stderr,
        )
    raise SystemExit(1)


def read(path: Path, plant: str) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        die(plant, f"cannot read {path}: {exc}")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def body(text: str, needle: str, plant: str) -> str:
    start = text.find(needle)
    if start < 0:
        die(plant, f"function anchor missing: {needle}")
    brace = text.find("{", start)
    if brace < 0:
        die(plant, f"opening brace missing after: {needle}")
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    die(plant, f"unbalanced function body: {needle}")


def require_order(scope: str, tokens: list[str], plant: str, reason: str) -> None:
    cursor = 0
    for token in tokens:
        found = scope.find(token, cursor)
        if found < 0:
            die(plant, reason)
        cursor = found + len(token)


def parse_constants(scope: str, names: tuple[str, ...], plant: str) -> tuple[int, ...]:
    values: list[int] = []
    for name in names:
        match = re.search(rf"\b{re.escape(name)}\s*=\s*(0x[0-9a-fA-F]+)", scope)
        if match is None:
            die(plant, f"dequant constant {name} is missing")
        values.append(int(match.group(1), 16))
    return tuple(values)


def half(bits: int) -> float:
    return struct.unpack("<e", int(bits & 0xFFFF).to_bytes(2, "little"))[0]


def half_round(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def lanes(word: int) -> tuple[float, float]:
    return half(word), half(word >> 16)


def exhaustive_dequant(constants: tuple[int, ...], plant: str) -> None:
    lo_mask, hi_mask, exponent, subtract, multiply, add = constants
    sub = lanes(subtract)
    mul = lanes(multiply)
    bias = lanes(add)
    cases = 0
    for n0, n1, n2, n3 in itertools.product(range(16), repeat=4):
        q = n0 | (n1 << 4) | (n2 << 16) | (n3 << 20)
        lo = lanes((q & lo_mask) | exponent)
        hi = lanes((q & hi_mask) | exponent)
        got = (
            half_round(lo[0] - sub[0]),
            half_round(lo[1] - sub[1]),
            half_round(hi[0] * mul[0] + bias[0]),
            half_round(hi[1] * mul[1] + bias[1]),
        )
        want = (float(n0 - 8), float(n2 - 8), float(n1 - 8), float(n3 - 8))
        if got != want:
            die(plant, f"dequant semantic mismatch q=0x{q:08x} got={got} want={want}")
        cases += 1
    if cases != 65536:
        die(plant, f"dequant sweep incomplete: {cases}/65536")


def apply_plant(plant: str, collective: str, dequant: str) -> tuple[str, str]:
    if plant == "none":
        return collective, dequant
    if plant == "runtime-dispatch":
        marker = "    auto multiply = [&](int inner) {"
        collective = collective.replace(
            marker, marker + "\n      for (int n_block = 0; n_block < 4; ++n_block) {}",
            1,
        )
    elif plant == "wrong-dequant-constant":
        dequant = dequant.replace("0x64086408", "0x64086409", 1)
    elif plant == "wrong-nblock-order":
        collective = collective.replace("multiply_n_block<1>", "multiply_n_block<X>", 1)
        collective = collective.replace("multiply_n_block<2>", "multiply_n_block<1>", 1)
        collective = collective.replace("multiply_n_block<X>", "multiply_n_block<2>", 1)
    elif plant == "wrong-helper-order":
        old = """    marlin_ppu_detail::FragmentB b0 =
        marlin_ppu_detail::dequantize_biased_int4(q);
    marlin_ppu_detail::FragmentB b1 =
        marlin_ppu_detail::dequantize_biased_int4(q >> 8);
    marlin_ppu_detail::scale(b0, fragment_scale[NBlock], 0);"""
        new = """    marlin_ppu_detail::FragmentB b0 =
        marlin_ppu_detail::dequantize_biased_int4(q);
    marlin_ppu_detail::scale(b0, fragment_scale[NBlock], 0);
    marlin_ppu_detail::FragmentB b1 =
        marlin_ppu_detail::dequantize_biased_int4(q >> 8);"""
        if old not in collective:
            die(plant, "helper-order plant seam drifted")
        collective = collective.replace(old, new, 1)
    else:
        die(plant, "unknown plant")
    return collective, dequant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", default="none")
    args = parser.parse_args()
    plant = args.plant

    collective = read(COLLECTIVE, plant)
    load = read(LOAD, plant)
    dequant = read(DEQUANT, plant)
    mma = read(MMA, plant)
    classic = read(CLASSIC, plant)
    awesome = read(AWESOME, plant)
    collective, dequant = apply_plant(plant, collective, dequant)

    if sha(classic) != EXPECTED_CLASSIC_SHA or sha(awesome) != EXPECTED_AWESOME_SHA:
        die(plant, "classic or Awesome-CuTe source hash drifted; re-audit before accepting")
    try:
        awesome_rev = subprocess.check_output(
            ["git", "-C", str(AWESOME.parents[2]), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        die(plant, f"cannot read Awesome-CuTe revision: {exc}")
    if awesome_rev != EXPECTED_AWESOME_REV:
        die(plant, f"Awesome-CuTe revision drifted: {awesome_rev}")

    for include in (
        "marlin_load_ppu.hpp", "marlin_dequant_ppu.hpp", "marlin_mma_ppu.hpp",
    ):
        if collective.count(f'#include "quactlize_extensions/cutlass/gemm/collective/{include}"') != 1:
            die(plant, f"standalone helper include is not unique: {include}")
    for forbidden in ("struct FragmentB {", "lop3.b32", "cp.async.cg.shared.global", "void mma_n16("):
        if forbidden in collective:
            die(plant, f"primitive returned to monolithic collective: {forbidden}")
    for token in ("struct alignas(16) Vector128", "struct FragmentA", "cp.async.cg.shared.global", "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16"):
        if token not in load:
            die(plant, f"load helper drifted: {token}")
    if "void mma_n16(" not in mma or "FragmentC& accum" not in mma:
        die(plant, "MMA helper no longer accepts one native FragmentC")
    mma_operands = re.findall(r'"\+f"\s*\(accum\.value\[(\d)\]\)', mma)
    if mma_operands != [str(i) for i in range(8)]:
        die(plant, f"MMA helper no longer binds native value[0..7]: {mma_operands}")

    prod_dq = body(dequant, "FragmentB dequantize_biased_int4(int q)", plant)
    classic_dq = body(classic, "FragB dequant(int q)", plant)
    awesome_dq = body(awesome, "auto dequant(int q)", plant)
    prod_constants = parse_constants(
        prod_dq, ("kLo", "kHi", "kExponent", "kSubtract", "kMultiply", "kAdd"), plant
    )
    classic_constants = parse_constants(
        classic_dq, ("LO", "HI", "EX", "SUB", "MUL", "ADD"), plant
    )
    awesome_constants = parse_constants(
        awesome_dq, ("LO", "HI", "EX", "SUB", "MUL", "ADD"), plant
    )
    if classic_constants != EXPECTED_CONSTANTS or awesome_constants != EXPECTED_CONSTANTS:
        die(plant, "independent reference constants disagree")
    if prod_constants != classic_constants:
        die(plant, "production dequant constants differ from both references")
    for scope, label in ((prod_dq, "production"), (classic_dq, "classic"), (awesome_dq, "awesome")):
        if scope.count("lop3<(0xf0 & 0xcc) | 0xaa>") != 2:
            die(plant, f"{label} no longer has exactly two biased-int4 LOP3 constructions")
    exhaustive_dequant(prod_constants, plant)

    helper = body(collective, "static void multiply_n_block(", plant)
    require_order(
        helper,
        [
            "quant[NBlock]",
            "FragmentB b0",
            "dequantize_biased_int4(q)",
            "FragmentB b1",
            "dequantize_biased_int4(q >> 8)",
            "scale(b0, fragment_scale[NBlock], 0)",
            "scale(b1, fragment_scale[NBlock], 1)",
            "mma_n16<NBlock>",
        ],
        plant,
        "production helper is not Awesome-CuTe D0,D1,S0,S1,M order",
    )
    multiply = body(collective, "auto multiply =", plant)
    calls = [int(value) for value in re.findall(r"multiply_n_block<(\d)>", multiply)]
    if calls != [0, 1, 2, 3]:
        die(plant, f"nblock-order differs: {calls}")
    for forbidden in ("for (int n_block", "if (n_block", "switch ("):
        if forbidden in multiply:
            die(plant, f"runtime-dispatch present: {forbidden}")

    classic_matmul = body(classic, "auto matmul =", plant)
    require_order(
        classic_matmul,
        ["FragB frag_b0 = dequant(b_quant)", "scale(frag_b0", "FragB frag_b1 = dequant(b_quant_shift)", "scale(frag_b1", "mma_n16("],
        plant,
        "classic default D0,S0,D1,S1,M16 anchor drifted",
    )
    awesome_gemm = body(awesome, "auto launch_gemm =", plant)
    require_order(
        awesome_gemm,
        ["int quant_w =", "int quant_w_shift =", "dequant(quant_w)", "dequant(quant_w_shift)", "__hmul2", "gemm(mma", "gemm(mma"],
        plant,
        "Awesome-CuTe arithmetic/N traversal anchor drifted",
    )
    if "n_idx += 2" not in awesome_gemm:
        die(plant, "Awesome-CuTe logical n16 traversal drifted")

    print(
        "[l174:dequant] exhaustive=65536/65536 "
        "constants=000f000f,00f000f0,64006400,64086408,2c002c00,d480d480 "
        f"classic_sha={EXPECTED_CLASSIC_SHA[:16]} awesome_sha={EXPECTED_AWESOME_SHA[:16]} "
        f"awesome_rev={EXPECTED_AWESOME_REV[:12]} result=PASS"
    )
    print(
        "[l174:cadence] authority=awesome-cute "
        "awesome=D0,D1,S0,S1,M8x2 production=D0,D1,S0,S1,M16 "
        "classic-reference-difference=D0,S0,D1,S1,M16 "
        "nblocks=0,1,2,3 runtime-dispatch=ABSENT result=PASS"
    )
    print("[l174] PASS: standalone helpers are split, independently anchored, and source-specialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

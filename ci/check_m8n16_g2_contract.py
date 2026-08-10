#!/usr/bin/env python3
"""Source contract for #118's historical m8n16 address-fault replay.

The old PPU failure was not a lane-varying x4 cube coordinate.  It was
NVIDIA's plain-x2 per-lane provider formula indexing a PPU register
distribution.  G2 must therefore read one production 16x64 AIU/x4 payload at
uniform coordinates, prove that raw payload, and replay the old indexing on
that payload.  This device-free checker pins that one-variable experiment;
tools/run_m8n16_111_box.sh owns the real ppu001 numerical result.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests/test_ppu_m8n16_aiu.cu"

# Review seals for the comment/string-stripped executable bodies.  G2 is a
# box-only corruption detector, so accepting an unreviewed code change here is
# worse than making a harmless refactor update an explicit digest.  The named
# structural checks below explain the contract; these seals close aliasing and
# post-processing escape hatches that a token allow-list cannot prove absent.
DEVICE_CONTRACT_SHA256 = "8179fc3f50f48e01cd6797ac5f59be9372a4abc00c36aa731d5aad4a599796fc"
HOST_ORACLE_SHA256 = "49fb3ddd031bb72df2c82eeb3ec17a79d0aae7cc421bd49e13decc1092ec7940"


def strip_comments_and_literals(text: str) -> str:
    """Keep C++ tokens/newlines; blank comments, strings, and chars."""
    out: list[str] = []
    i = 0
    state = "code"
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                out.extend("  ")
                i += 2
                state = "line"
                continue
            if ch == "/" and nxt == "*":
                out.extend("  ")
                i += 2
                state = "block"
                continue
            if ch == '"':
                out.append(" ")
                i += 1
                state = "string"
                continue
            if ch == "'":
                out.append(" ")
                i += 1
                state = "char"
                continue
            out.append(ch)
            i += 1
            continue
        if state == "line":
            out.append("\n" if ch == "\n" else " ")
            i += 1
            if ch == "\n":
                state = "code"
            continue
        if state == "block":
            if ch == "*" and nxt == "/":
                out.extend("  ")
                i += 2
                state = "code"
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        quote = '"' if state == "string" else "'"
        if ch == "\\" and i + 1 < len(text):
            out.extend("  ")
            i += 2
        elif ch == quote:
            out.append(" ")
            i += 1
            state = "code"
        else:
            out.append("\n" if ch == "\n" else " ")
            i += 1
    return "".join(out)


def section(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) < 1:
        raise ValueError(f"cannot isolate {begin!r} .. {end!r}")
    return text.split(begin, 1)[1].split(end, 1)[0]


def frozen_digest(text: str) -> str:
    stripped = strip_comments_and_literals(text)
    stable = "\n".join(line.rstrip() for line in stripped.strip().splitlines())
    return hashlib.sha256(stable.encode()).hexdigest()


def historical_coincidences() -> tuple[list[tuple[int, int]], bool]:
    """Rebuild x2 routing from the SDK sample, then apply the old provider."""
    # This intentionally spells out getThreadAddr1D(size=128) instead of
    # repeating the simplified C++ inverse.  Each provider owns two 32-bit
    # words in its 64-bit window.  Map correct matrix coordinates back to that
    # (provider, in-window-word) pair first.
    size = 128
    num = size >> 6
    shift_size = 5 - (num >> 1)
    col_size_bytes = size >> 2
    providers: dict[tuple[int, int], tuple[int, int]] = {}
    for provider_lane in range(32):
        offset_index = provider_lane >> shift_size
        offset_bytes = offset_index << 4
        col_div = 1 << shift_size
        block_lane = provider_lane % col_div
        block_row_threads = 4 // num
        row = block_lane // block_row_threads
        start_bytes = row * col_size_bytes
        start_offset_bytes = (block_lane % block_row_threads) << (num + 1)
        base_word = (start_bytes + start_offset_bytes + offset_bytes) // 4
        row_base_word = row * (col_size_bytes // 4)
        for in_window_word in range(2):
            coord = (row, base_word - row_base_word + in_window_word)
            if coord in providers:
                raise AssertionError(f"PPU x2 provider map aliases {coord}")
            providers[coord] = (provider_lane, in_window_word)

    matches: list[tuple[int, int]] = []
    simplified_route_matches = True
    for lane in range(32):
        good_row = lane // 4
        lane_word = lane % 4
        for reg in range(2):
            good_word = lane_word + 4 * reg
            provider_lane, in_window_word = providers[(good_row, good_word)]
            simplified_route_matches &= (
                provider_lane == 2 * good_row + lane_word // 2 + 16 * reg and
                in_window_word == lane_word % 2
            )
            bad_row = provider_lane % 8
            bad_base_word = ((provider_lane // 8) * 4) % 8
            bad_word = bad_base_word + in_window_word
            if (bad_row, bad_word) == (good_row, good_word):
                matches.append((lane, reg))
    return matches, simplified_route_matches


def audit(source: str) -> list[str]:
    bad: list[str] = []
    code = strip_comments_and_literals(source)
    flat = re.sub(r"\s+", "", code)

    forbidden = (
        "PPU_U32x1_LDSM_N", "PPU_U32x2_LDSM_N", "PPU_U32x4_LDSM_N",
        "PPU_U16x2_LDSM_T", "PPU_U16x4_LDSM_T", "PPU_U16x8_LDSM_T",
        "copy_ldsm(", "copy_ldsm_trans(", "cutlass::arch::ldsm",
    )
    for token in forbidden:
        if token in code:
            bad.append(f"G2 instantiates forbidden legacy/plain LDSM token {token!r}")
    if '#include "cute/arch/copy_ppu.hpp"' in source:
        bad.append("G2 still includes the header whose dormant plain-LDSM atom caused #114")

    for token in (
        "KNOWN PRE-EXISTING ACTLIZE DEFECT", "cute/arch/copy_ppu.hpp",
        "cutlass/arch/memory_ppu.h", "six ppu001", "six counterparts",
    ):
        if token not in source:
            bad.append(f"the 12-site vendor defect/disposition no longer records {token!r}")

    # No second geometry, smem payload, or divergent-coordinate arm may return.
    for token in ("kGuardH", "GuardAiuWrite", "GuardSwzlRead", "guard_swzl",
                  "guard_input", "guard_good", "guard_bad"):
        if token in code:
            bad.append(f"G2 resurrected the rejected 32-row guard path via {token!r}")

    required_flat = (
        "constexprintkCubeH=16;",
        "constexprintkCubeW=64;",
        "std::uint16_tswzl[kCubeElements];",
        "usingSwzlRead=cute::PPU0010_TSM_LD_SWZL<half_t,kCubeH,kCubeW,true,false,1>;",
        "AiuWrite::copy(storage.swzl,src,desc,0,0,0);",
        "SwzlRead::copy(x4,storage.swzl,0,0,0,0);",
        "g2_device<<<kCases,kWarp>>>(d_input.get(),d_x4.get());",
        "intconstgood_row=lane/4;",
        "intconstlane_word=lane%4;",
        "intconstprovider_lane=2*good_row+lane_word/2+16*reg;",
        "intconstnvidia_row=provider_lane%kI;",
        "intconstnvidia_base_word=((provider_lane/kI)*(kJ/2))%kJ;",
        "intconstbad_word=nvidia_base_word+lane_word%2;",
        "intconstsrc_lane=4*nvidia_row+bad_word%4;",
        "intconstsrc_reg=bad_word/4;",
        "std::uint32_tconstgot=x4[src_lane*kX4Registers+src_reg];",
        "intconstgood_word=lane_word+4*reg;",
        "input[nvidia_row*kCubeW+2*bad_word+h];",
        "input[good_row*kCubeW+2*good_word+h];",
        "bad_map_bad+=halfword(got,h)!=bad_want;",
        "red_mismatches+=halfword(got,h)!=good_want;",
        "constexprintkExpectedCoincidentWords=2;",
        "(kProjectedWords-kExpectedCoincidentWords)*kHalfsPerRegister;",
        "red_mismatches==kExpectedRedMismatches&&bad_map_bad==0&&coincident_words==kExpectedCoincidentWords;",
        "x4_bad+projected_changed+(kLowerValues-lower_changed);",
    )
    for token in required_flat:
        if flat.count(token) != 1:
            bad.append(f"G2 must contain exactly one frozen contract token {token!r}")

    try:
        kernel = section(code, "__global__ void g2_device", "\nconstexpr std::uint16_t upper_tag")
        device_contract = "struct alignas(32) SharedStorage" + section(
            code, "struct alignas(32) SharedStorage", "\nconstexpr std::uint16_t upper_tag")
    except ValueError as e:
        return bad + [str(e)]
    host_oracle = "constexpr std::uint16_t upper_tag" + code.split(
        "constexpr std::uint16_t upper_tag", 1)[1]

    device_digest = frozen_digest(device_contract)
    host_digest = frozen_digest(host_oracle)
    if device_digest != DEVICE_CONTRACT_SHA256:
        bad.append(f"review-sealed G2 device body changed: {device_digest}")
    if host_digest != HOST_ORACLE_SHA256:
        bad.append(f"review-sealed G2 host oracle changed: {host_digest}")

    if code.count("SwzlRead::copy") != 1:
        bad.append("green/red must share the only lexical x4-swzl instruction seam")
    if code.count("AiuWrite::copy") != 1:
        bad.append("G2 must have exactly one production AIU write seam")
    if kernel.count("storage.swzl") != 2:
        bad.append("the one AIU writer and one x4 reader must share storage.swzl")
    if re.search(r"SwzlRead::copy\([^;]*(?:lane|nvidia|bad_word)", kernel):
        bad.append("x4 instruction coordinates became lane-dependent again")

    expected = [(0, 0), (1, 0)]
    coincidences, routing_ok = historical_coincidences()
    if not routing_ok:
        bad.append("source's simplified provider inverse disagrees with the SDK getThreadAddr1D reconstruction")
    if coincidences != expected:
        bad.append("checker's independent historical map no longer gives the two reviewed coincidences")

    for marker in (
        "[G2-control-path] same-payload=production-x4 cube=16x64 coords=(0,0)",
        "green=get_i/get_j red=historical-nvidia-x2-provider-map",
        "[G2-green-detail] x4_values=%d x4_bad=%d projected_changed=%d/%d",
        "[G2-negative-detail] same_payload=x4-swzl geometry=16x64",
        "bad_map_values=%d bad_map_bad=%d coincident_words=%d/%d",
    ):
        if marker not in source:
            bad.append(f"box-auditable output marker disappeared: {marker!r}")
    return bad


def plant(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(
            f"planted-control anchor occurs {source.count(old)} times, wanted 1: {old!r}")
    return source.replace(old, new, 1)


def main() -> int:
    source = SOURCE.read_text()
    bad = audit(source)
    if bad:
        print("[m8n16-g2-contract] FAIL: " + "; ".join(bad))
        return 1

    mutations = (
        ("different primitive", "SwzlRead::copy(x4, storage.swzl, 0, 0, 0, 0);",
         "cute::PPU_U32x4_LDSM_N::copy(/* planted different primitive */);"),
        ("different base", "x4, storage.swzl, 0, 0, 0, 0", "x4, storage.swzl + 16, 0, 0, 0, 0"),
        ("divergent row coordinate", "x4, storage.swzl, 0, 0, 0, 0", "x4, storage.swzl, 0, lane % 8, 0, 0"),
        ("physical height drift", "constexpr int kCubeH = 16;", "constexpr int kCubeH = 32;"),
        ("second x4 instruction", "SwzlRead::copy(x4, storage.swzl, 0, 0, 0, 0);",
         "SwzlRead::copy(x4, storage.swzl, 0, 0, 0, 0);\n"
         "  SwzlRead::copy(x4, storage.swzl, 0, 0, 0, 0);"),
        ("PPU x2 provider redistribution drift",
         "2 * good_row + lane_word / 2 + 16 * reg",
         "2 * good_row + lane_word / 2 + 8 * reg"),
        ("NVIDIA row drift", "int const nvidia_row = provider_lane % kI;",
         "int const nvidia_row = provider_lane % (kI - 1);"),
        ("NVIDIA provider base drift", "((provider_lane / kI) * (kJ / 2)) % kJ",
         "((provider_lane / kI) * (kJ / 4)) % kJ"),
        ("PPU x2 in-window word dropped",
         "int const bad_word = nvidia_base_word + lane_word % 2;",
         "int const bad_word = nvidia_base_word;"),
        ("x4 owner lane drift", "4 * nvidia_row + bad_word % 4",
         "4 * nvidia_row + (bad_word + 1) % 4"),
        ("x4 owner register drift", "int const src_reg = bad_word / 4;",
         "int const src_reg = bad_word / 8;"),
        ("bad source made direct green", "x4[src_lane * kX4Registers + src_reg]",
         "x4[lane * kX4Registers + reg]"),
        ("bad-map golden shifted", "input[nvidia_row * kCubeW + 2 * bad_word + h]",
         "input[nvidia_row * kCubeW + 2 * bad_word + h + 1]"),
        ("green golden shifted", "input[good_row * kCubeW + 2 * good_word + h]",
         "input[good_row * kCubeW + 2 * good_word + h + 1]"),
        ("poison witness removed", " + (kLowerValues - lower_changed)", ""),
        ("red made tautological", "red_mismatches == kExpectedRedMismatches &&",
         "red_mismatches >= 0 &&"),
        ("bad-map proof removed", "bad_map_bad == 0 &&", "bad_map_bad >= 0 &&"),
        ("coincidence count weakened", "coincident_words == kExpectedCoincidentWords",
         "coincident_words >= 0"),
        ("expected coincidences drift", "constexpr int kExpectedCoincidentWords = 2;",
         "constexpr int kExpectedCoincidentWords = 3;"),
        ("halfword oracle broken", "word >> (16 * h)", "word >> 0"),
    )
    for name, old, new in mutations:
        try:
            mutated = plant(source, old, new)
        except ValueError as e:
            print(f"[m8n16-g2-contract] FAIL: {e}")
            return 1
        if not audit(mutated):
            print(f"[m8n16-g2-contract] FAIL: checker accepted planted regression: {name}")
            return 1

    print("[m8n16-g2-contract] PASS -- one 16x64 AIU/x4 payload feeds both maps; "
          "historical NVIDIA provider indexing must name exact tags and differ in 124/128 values; "
          "20 regressions rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

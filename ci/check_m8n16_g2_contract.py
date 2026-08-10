#!/usr/bin/env python3
"""Source-level contract for #114's G2 negative control.

The failure being detected is silent address permutation.  A red arm that changes the load opcode, shared-memory
base, or legal address range proves nothing about that defect.  This device-free checker pins the one-variable
experiment; tools/run_m8n16_111_box.sh owns the real ppu001 numerical result.
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
DEVICE_CONTRACT_SHA256 = "6bcb9be4aea524de60b99845a2b4f5c2413992e5a05fa0fe9ab5c07c342a7176"
HOST_ORACLE_SHA256 = "cbbf6ad39328eb42d7f5d4bce67a30e04b3086043c53be4c612a09b436a1980c"


def strip_comments_and_literals(text: str) -> str:
    """Keep C++ tokens/newlines; blank comments, strings and chars so prose cannot satisfy a code audit."""
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
        # String/character literal. Preserve escaped bytes as blanks and the physical newline shape.
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

    required_flat = (
        "constexprintkCubeH=16;",
        "constexprintkGuardH=32;",
        "std::uint16_tguard_swzl[kGuardElements];",
        "usingGuardSwzlRead=cute::PPU0010_TSM_LD_SWZL<half_t,kGuardH,kCubeW,true,false,1>;",
        "GuardAiuWrite::copy(storage.guard_swzl,guard_input,guard_desc,0,0,0);",
        "std::uint32_tguard_good[kX4Registers]={};",
        "std::uint32_tguard_bad[kX4Registers]={};",
        "intconstnvidia_row=lane%kI;",
        "intconstnvidia_word=((lane/kI)*(kJ/2))%kJ;",
        "g2_guard_swzl_x4(guard_good,storage.guard_swzl,0,0);",
        "g2_guard_swzl_x4(guard_bad,storage.guard_swzl,2*nvidia_word,nvidia_row);",
        "guard_desc.dim_h=kGuardH;",
        "guard_desc.dim_w=kCubeW;",
        "guard_desc.cube_h=kGuardH;",
        "guard_desc.cube_w=kCubeW;",
        "intguard_bad_map_bad=0;",
        "intconstbad_row=(lane%8)+row;",
        "intconstbad_k=2*(((lane/8)*(8/2))%8)+2*word+h;",
        "guard_bad_map_bad+=halfword(guard_bad_got,h)!=guard_bad_want;",
        "intred_mismatches=0;",
        "std::uint16_tconstwant=guard_input[row*kCubeW+k];",
        "x4_bad+guard_x4_bad+projected_changed+(kLowerValues-lower_changed)",
        "red_mismatches==kExpectedRedMismatches&&guard_bad_map_bad==0&&zero_coord_lanes==2&&red_zero_coord_bad==0",
    )
    for token in required_flat:
        if flat.count(token) != 1:
            bad.append(f"G2 must contain exactly one frozen contract token {token!r}")

    try:
        helper = section(code, "CUTE_DEVICE void g2_guard_swzl_x4", "\n__global__ void g2_device")
        kernel = section(code, "__global__ void g2_device", "\nconstexpr std::uint16_t upper_tag")
    except ValueError as e:
        return bad + [str(e)]
    try:
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
    if helper.count("GuardSwzlRead::copy") != 1 or code.count("GuardSwzlRead::copy") != 1:
        bad.append("guard good/bad must share the only lexical GuardSwzlRead::copy instruction seam")
    if kernel.count("g2_guard_swzl_x4(") != 2:
        bad.append("kernel must invoke the one guard x4-swzl helper exactly twice (good and bad)")
    if kernel.count("storage.guard_swzl") != 3:
        # Exactly the AIU writer plus control-good and control-bad.  The descriptor
        # names the global source, not this shared-memory destination.
        bad.append("guard writer/good/bad no longer visibly share the one guard_swzl storage object")

    for marker in (
        "[G2-control-path] same-op=PPU0010_TSM_LD_SWZL<m8n8.x4.swzl>",
        "guard_x4_values=%d guard_x4_bad=%d",
        "[G2-negative-detail] same_op=x4-swzl bad_map_values=%d",
    ):
        if marker not in source:
            bad.append(f"box-auditable output marker disappeared: {marker!r}")
    return bad


def plant(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"planted-control anchor occurs {source.count(old)} times, wanted 1: {old!r}")
    return source.replace(old, new, 1)


def main() -> int:
    source = SOURCE.read_text()
    bad = audit(source)
    if bad:
        print("[m8n16-g2-contract] FAIL: " + "; ".join(bad))
        return 1

    mutations = (
        ("different primitive",
         "g2_guard_swzl_x4(\n      guard_bad, storage.guard_swzl, 2 * nvidia_word, nvidia_row);",
         "cute::PPU_U32x4_LDSM_N::copy(/* planted different primitive */);"),
        ("different base", "guard_bad, storage.guard_swzl, 2 * nvidia_word, nvidia_row",
         "guard_bad, storage.prod_swzl, 2 * nvidia_word, nvidia_row"),
        ("bad coords reset", "guard_bad, storage.guard_swzl, 2 * nvidia_word, nvidia_row",
         "guard_bad, storage.guard_swzl, 0, 0"),
        ("NVIDIA row drift", "int const nvidia_row = lane % kI;",
         "int const nvidia_row = lane % (kI - 1);"),
        ("poison witness removed", " + (kLowerValues - lower_changed)", ""),
        ("red made tautological", "red_mismatches == kExpectedRedMismatches &&",
         "red_mismatches >= 0 &&"),
        ("guard shrunk to unsafe height", "constexpr int kGuardH = 32;", "constexpr int kGuardH = 16;"),
        ("second instruction seam",
         "g2_guard_swzl_x4(\n      guard_bad, storage.guard_swzl, 2 * nvidia_word, nvidia_row);",
         "GuardSwzlRead::copy(guard_bad, storage.guard_swzl, 2 * nvidia_word, nvidia_row, 0, 0);"),
        ("helper changes the bad base",
         "GuardSwzlRead::copy(frag, smem_base, coord_w, coord_h, 0, 0);",
         "GuardSwzlRead::copy(frag, smem_base + (coord_w != 0), coord_w, coord_h, 0, 0);"),
        ("bad arm postprocess",
         "g2_guard_swzl_x4(\n      guard_bad, storage.guard_swzl, 2 * nvidia_word, nvidia_row);",
         "g2_guard_swzl_x4(\n      guard_bad, storage.guard_swzl, 2 * nvidia_word, nvidia_row);\n"
         "  guard_bad[0] ^= (nvidia_row != 0);"),
        ("red mismatch seeded", "int red_mismatches = 0;", "int red_mismatches = 1;"),
        ("guard valid height shrunk", "guard_desc.dim_h = kGuardH;", "guard_desc.dim_h = kCubeH;"),
        ("NVIDIA I constant drift", "constexpr int kI = 8;", "constexpr int kI = 7;"),
        ("origin golden shifted", "guard_input[row * kCubeW + k];", "guard_input[row * kCubeW + k + 1];"),
        ("bad-map golden shifted", "guard_input[bad_row * kCubeW + bad_k];",
         "guard_input[bad_row * kCubeW + bad_k + 1];"),
        ("guard allocation shrunk", "std::uint16_t guard_swzl[kGuardElements];",
         "std::uint16_t guard_swzl[kCubeElements];"),
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

    print("[m8n16-g2-contract] PASS -- production 16-row poison gate retained; guard good/bad share one "
          "32-row x4-swzl seam; bad arm must match all 512 shifted tags and exactly 120/128 origin values "
          "must differ; 17 regressions rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

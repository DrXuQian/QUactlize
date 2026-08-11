#!/usr/bin/env python3
"""Device-free source contract for G5's expert-boundary address census.

The numerical diagnosis still belongs to ppu001.  This checker proves that the
binary sent there preserves the one-variable experiment: one real grouped
scheduler, one zero plane, and four distinct observation layers.  In-memory
plants prove the audit can see the quiet regressions that would otherwise turn
the probe into another self-comparison.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "quactlize/include/ppu_aiu_gemm_mixed_input_group.hpp"
HARNESS = ROOT / "tests/test_ppu_grouped_metadata_address.cu"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
BOX = ROOT / "tools/run_grouped_metadata_address_probe_box.sh"


def audit(header: str, harness: str, cmake: str, box: str) -> list[str]:
    bad: list[str] = []

    def need(text: str, token: str, where: str) -> None:
        if token not in text:
            bad.append(f"{where}: missing {token!r}")

    # Boundary and scope: e128 is load-bearing; q==8 means this can prove only
    # metadata, never B.  Both source and box log must say so.
    need(header, "expert == 127 ? 0 : expert == 128 ? 1 : expert == 129 ? 2 : -1", "device trace")
    need(harness, "constexpr int kExperts[3] = {127, 128, 129};", "harness")
    need(harness, "0x88u); // q==8", "harness")
    need(harness, "scope=B_NOT_COVERED", "harness")
    need(harness, "[G5:ADDR][shape-detail]", "harness")
    need(harness, "gz_addr_delta_bytes", "harness")
    need(box, "scope=B_NOT_COVERED", "box gate")
    need(box, "gz_experts=", "box gate")
    need(box, "gz_addr_delta_bytes=", "box gate")
    need(box, "did not print an actual explicit-vs-gZ address/value witness", "box gate")

    # The explicit arm must remain genuinely 64-bit.  A 32-bit plant recreates
    # exactly the kind of boundary fold this probe was written to distinguish.
    if not re.search(r"explicit_plane\s*=\s*params\.mainloop\.ptr_Z\s*\+\s*\n?\s*int64_t\(expert\)", header):
        bad.append("device trace: explicit GEP is not visibly based on int64_t(expert)")
    need(header, "int64_t const plane_elements", "device trace")
    need(header, "explicit_addr", "device trace")
    need(header, "explicit_bits", "device trace")

    # gZ must be observed from CuTe, not aliased to the explicit arm.
    need(header, "&gZ(local_n, group_in_tile, metadata_tile)", "device trace")
    if "Zero const* gz_ptr = explicit_ptr" in header:
        bad.append("device trace: gZ observation aliases the explicit GEP")
    need(header, "gz_addr", "device trace")
    need(header, "gz_bits", "device trace")

    # partition_S must reproduce shipping's modulo seam. Raw get_slice(tid)
    # would be an invalid extra variable and can manufacture the very red.
    need(header, "copy_slot = thread_idx % thread_slots", "device trace")
    need(header, "get_slice(copy_slot)", "device trace")
    need(header, "partition_S(gZ)", "device trace")
    need(header, "partition_addr", "device trace")
    need(header, "partition_bits", "device trace")

    # The final arm must execute the production copy primitive and observe the
    # destination after its fence/wait, not copy src into the record twice.
    need(header, "copy(params.mainloop.gmem_tiled_copy_zero, src, dst);", "device trace")
    need(header, "cute::cp_async_fence();", "device trace")
    need(header, "cute::cp_async_wait<0>();", "device trace")
    need(header, "reinterpret_cast<uint16_t const*>(dst_ptr)", "device trace")
    need(header, "destination_addr", "device trace")

    # Workspace alias is avoided constructively: all 256 groups have one row,
    # so the launch's ragged-prefix write is unreachable.
    need(harness, "std::vector<int> group_m(kE, kM);", "harness")
    need(harness, "std::vector<GS> shapes(kE, cute::make_shape(kM, kN, kK));", "harness")
    need(header, "params.mtiles_uniform > 0", "device trace")
    need(header, "configuration_errors", "device trace")

    # Macro-on must reach both host and hgcc's custom device compile, and no
    # box recipe may call the local CUDA gate on the PPU SDK toolchain.
    target = re.search(
        r"quactlize_ppu_executable\(\s*test_ppu_grouped_metadata_address(.*?)\n\)",
        cmake, re.S)
    if not target or "DEV_COMPILE_FLAGS -DPPU_METADATA_ADDR_PROBE=1" not in target.group(1):
        bad.append("CMake: probe macro does not reach the hgcc device custom command")
    need(cmake, "target_compile_definitions(test_ppu_grouped_metadata_address PRIVATE", "CMake")
    if "local_gates.py" in "\n".join(
            line for line in box.splitlines() if not line.lstrip().startswith("#")):
        bad.append("box gate: invokes local_gates.py on a toolchain that cannot run it")

    return bad


def main() -> int:
    texts = {
        "header": HEADER.read_text(),
        "harness": HARNESS.read_text(),
        "cmake": CMAKE.read_text(),
        "box": BOX.read_text(),
    }
    bad = audit(**texts)
    if bad:
        print("[G5 address contract] FAIL")
        for item in bad:
            print(f"  {item}")
        return 1

    plants = [
        ("int32 GEP", {**texts, "header": texts["header"].replace(
            "int64_t(expert) * plane_elements", "int32_t(expert) * plane_elements", 1)}),
        ("gZ aliases explicit", {**texts, "header": texts["header"].replace(
            "Zero const* gz_ptr = cute::raw_pointer_cast(&gZ(local_n, group_in_tile, metadata_tile));",
            "Zero const* gz_ptr = explicit_ptr;", 1)}),
        ("cp.async bypass", {**texts, "header": texts["header"].replace(
            "copy(params.mainloop.gmem_tiled_copy_zero, src, dst);",
            "// planted direct scalar path", 1)}),
        ("e128 removed", {**texts, "harness": texts["harness"].replace(
            "constexpr int kExperts[3] = {127, 128, 129};",
            "constexpr int kExperts[3] = {126, 127, 129};", 1)}),
    ]
    escaped: list[str] = []
    for name, planted in plants:
        if not audit(**planted):
            escaped.append(name)
    if escaped:
        print("[G5 address contract] FAIL: planted controls escaped: " + ", ".join(escaped))
        return 1

    print("[G5 address contract] PASS: four distinct address/copy layers; 4/4 planted controls rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())

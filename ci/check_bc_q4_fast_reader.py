#!/usr/bin/env python3
"""Contract and executable gate for the shipping Q4 BC whole-word reader."""

from __future__ import annotations

import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dev/fold_derivation/l187_bc_q4_fast_reader.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l187_bc_q4_fast_reader.sh"
READER = ROOT / "quactlize/include/gguf_bc_q4_reader.hpp"
BC = ROOT / "quactlize/include/gguf_bc_vecdot.hpp"
KERNEL = ROOT / "quactlize/include/gguf_bc_q4_gemv.hpp"
DOC = ROOT / "dev/fold_derivation/BC_Q4_FAST_READER_VALIDATION.md"


def require(text: str, needle: str, owner: str) -> None:
    if needle not in text:
        raise SystemExit(f"[bc-q4-fast-reader] FAIL: {owner} lost {needle!r}")


def require_shipping_hot_path(text: str) -> None:
    """Reject the exact slow seams removed from the Q4 shipping branch."""
    try:
        body = text.split("if constexpr (T == KType::Q4_K)", 1)[1].split("} else {", 1)[0]
    except IndexError as error:
        raise SystemExit("[bc-q4-fast-reader] FAIL: Q4 shipping branch is not structurally visible") from error
    for forbidden in ("unit_group_sb<", "float(sz", "xplane_physical_code<"):
        if forbidden in body:
            raise SystemExit(
                f"[bc-q4-fast-reader] FAIL: Q4 shipping branch regressed to {forbidden!r}"
            )


def main() -> int:
    source = SOURCE.read_text()
    runner = RUNNER.read_text()
    reader = READER.read_text()
    bc = BC.read_text()
    kernel = KERNEL.read_text()
    doc = DOC.read_text()

    for token in (
        "place_derived<4", "recover_derived<4", "xplane_physical_code<KType::Q4_K",
        "q4_group_byte_offset<ArtifactTileK>", "physical_nibble_from_logical_k",
        "alignment_bad",
        "wrong-permutation", "missing-denominator", "checked == uint64_t(denominator) * kCodes",
        "device_binding_probe", "dequantize_word",
    ):
        require(source, token, "L187 oracle")
    for token in (
        "Q4WordPlan", "logical_k_from_physical_nibble", "physical_nibble_from_logical_k",
        "dequantize_word",
    ):
        require(reader, token, "shipping Q4 reader")
    for token in (
        "q4_group_byte_offset", "code_at", "q4_reader::", "ArtifactTileK == 64",
        "bc_q4_gemv::launch_default", "#if !defined(__HGGCCC__)",
    ):
        require(bc, token, "shipping BC consumer")
    require_shipping_hot_path(bc)
    # This plant proves the structural regression classifier itself is live.
    planted = bc.replace("auto const metadata = q4_reader::load_metadata(unit);",
                         "auto const metadata = packed_unit::unit_group_sb<T,0>(unit,0,0);", 1)
    try:
        require_shipping_hot_path(planted)
    except SystemExit:
        pass
    else:
        raise SystemExit("[bc-q4-fast-reader] FAIL: old metadata-reader plant was not rejected")
    for token in (
        "kernel<CTA_N, WARPS_N, WARPS_K>", "decode_scale_zero({m.x, m.y, m.z, m.w}",
        "default_admits", "launch<2, 4, 1>", "kLargestAdmittedK = 8192",
        "n % kOutputColumns", "!default_admits(1, 4096, 32768)",
    ):
        require(kernel, token, "shipping CUDA Q4 topology")
    body = bc.split("int64_t q4_group_byte_offset", 1)[1].split("template <KType T, int ArtifactTileK>", 1)[0]
    if "xplane_physical_code" in body:
        raise SystemExit("[bc-q4-fast-reader] FAIL: whole-word group base fell back to the scalar xplane inverse")
    for token in (
        "/workspace/quactlize-l187-bc-q4-fast-reader", "PLANTED_RED wrong-permutation DETECTED",
        "PLANTED_RED missing-denominator DETECTED", "nvdisasm", "ppu\\.", "LOP3.LUT", "HFMA2",
    ):
        require(runner, token, "L187 runner")
    for token in (
        "1,048,576", "versionless C++ reader", "Python descriptor-producing Q4 path",
        "ArtifactTileK=256", "shipping default is A64", "does not claim a latency improvement",
    ):
        require(doc, token, "L187 evidence document")
    if "/tmp" in runner or "mktemp" in runner:
        raise SystemExit("[bc-q4-fast-reader] FAIL: runner must keep artifacts under /workspace without mktemp")

    proc = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, text=True)
    if proc.returncode == 3:
        print("[bc-q4-fast-reader] SKIP: semantic contract passed; sm_120 toolchain unavailable")
        return 3
    if proc.returncode:
        return proc.returncode
    print("[bc-q4-fast-reader] PASS: shipping helper is writer/code_at-bound; both plants and target dispatch proved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

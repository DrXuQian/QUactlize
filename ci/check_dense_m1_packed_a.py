#!/usr/bin/env python3
"""Bind the ordinary dense M==1 route to its independent Rows=1 packed-A shipping type."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DISPATCH = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp"
POLICY = ROOT / "quactlize/include/ppu_mixed_policy.hpp"
BUILDER = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl"
COLLECTIVE = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
PACK_DETAIL = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/detail/ppu_a_pack.hpp"
LAUNCHER = ROOT / "quactlize/include/fpA_intB_ppu.cuh"
BACKEND = ROOT / "quactlize/csrc/device/ppu_dense_backend.cu"
TYPE_ORACLE = ROOT / "dev/fold_derivation/l186_dense_m1_packed_a.cu"
GEOMETRY_ORACLE = ROOT / "dev/fold_derivation/l186_dense_m1_packed_a_geometry.cu"


def source_errors(text: dict[str, str]) -> list[str]:
    errors: list[str] = []
    required = {
        "dispatch": (
            "struct KernelAiuPackedA", "struct a_provider_schedule_traits<KernelAiuPackedA<",
            "struct fold_schedule_traits<KernelAiuPackedA<",
            "MainloopQuactlizeMixedInput<Stages_, kContinous_, KernelAiuPackedA<",
        ),
        "policy": (
            "struct PackedAMainloopPolicy", "static_assert(APackRows == 1",
            "ordinary unfolded one-plane only", "using KernelSchedule = cutlass::gemm::KernelAiuPackedA<",
        ),
        "builder": (
            "struct MixGemm_AIU_OperandPackedA", "ScheduledAPackRows",
            "MmaInstM == 8 && physicalBlockM == 16", "ScheduledAPackRows == 0, OrdinaryOperandA",
        ),
        "collective": (
            "static constexpr int kACubeH      = PhysicalATileM",
            "static constexpr bool kPackedA = kAPackRows > 0;",
            "if constexpr (kPackedA)", "copy_A_packed_rows<kAPackRows>",
            "detail::aPackRunOffsetHalfs(kACubeH",
        ),
        "pack_detail": (
            "CUTE_HOST_DEVICE constexpr int aPackRunOffsetHalfs",
            "aPackRunOffsetHalfs(16, 0, 1) == 288",
            "aPackRunOffsetHalfs(16, 0, 2) == 528",
        ),
        "launcher": (
            "struct DensePackedAKernelTypes", "using KernelTypes = DenseKernelTypes<",
            "using SelectedKernelTypes = std::conditional_t<",
            "if constexpr (!std::is_void_v<KernelTypesOverride>)",
        ),
        "backend": (
            "bool UseM1PackedA = false", "if constexpr (UseM1PackedA && kOrdinaryOnePlane)",
            "if (m == 1)", "using PackedKernelTypes = fpa_intb_ppu::DensePackedAKernelTypes<1",
            "(DenseConfigId::ID == kDecodeDefaultDenseConfig), QueryOnly",
            "return launch_dense_config<Low, High, GroupSize, TacticTileK, ArtifactTileK, PackedScale, true>",
        ),
        "type_oracle": (
            "std::is_same_v<typename Ordinary::CollectiveMainloop, DirectMainloop>",
            "DensePackedAKernelTypes<", "the exact shipping packed-A type must use ppu001 m8n16k16",
            "struct ProductionPackedACell", "matrix=7(q4=4,q2=3)",
            "ProductionPackedACell<10, cutlass::uint2b_t",
            "ProductionPackedACell<12, cutlass::int4b_t",
        ),
        "geometry_oracle": (
            "constexpr int kLogicalM = 8;", "constexpr int kPhysicalM = 16;",
            "ppu0010_tsm_ld_swzl_m8_word", "make_ppu_read_inverse",
            "for (int visits : output)", "L186_BAD_DESTINATION_DELTA", "L186_BAD_SLICE_SWAP",
            "production writer -> independent hardware-calibrated PPU0010 reader",
        ),
    }
    for name, tokens in required.items():
        for token in tokens:
            if token not in text[name]:
                errors.append(f"{name} missing {token}")

    # The old authority must remain an independently named type. Replacing it globally with the packed type would
    # make M=2..7 silently pay for/consume the M==1 provider while all positive packed checks still passed.
    if text["launcher"].count("struct DenseKernelTypes {") != 1:
        errors.append("default DenseKernelTypes authority was rewritten or duplicated")
    if text["backend"].count("launch_dense_config<") < 3:
        errors.append("query and launch no longer converge on launch_dense_config")
    if text["backend"].count("launch_dense_tactic<") != 1:
        errors.append("dense config registry no longer has one compile-time tactic selection seam")
    if "if (m == 1)" not in text["backend"] or "M=2..7 falls through" not in text["backend"]:
        errors.append("runtime M==1 guard or explicit M>1 fallthrough disappeared")
    if text["collective"].count("detail::aPackRunOffsetHalfs(kACubeH") != 3 or \
            "aPackRunOff(" in text["collective"]:
        errors.append("collision proof and writer no longer share exactly one detail run-offset authority")
    reader = text["geometry_oracle"].split("// Independent PPU0010 reader:", 1)
    if len(reader) != 2 or "aPackRunOffsetHalfs" in reader[1]:
        errors.append("geometry expected/read side is no longer independent of the production writer authority")
    return errors


def plant(name: str, text: dict[str, str]) -> None:
    if name == "missing-m1-guard":
        text["backend"] = text["backend"].replace("if (m == 1)", "if (m >= 1)", 1)
    elif name == "default-type-wrapped":
        text["launcher"] = text["launcher"].replace(
            "struct DenseKernelTypes {", "struct DenseKernelTypes_Moved {", 1)
    elif name == "query-launch-diverged":
        text["backend"] = text["backend"].replace(
            "return launch_dense_config<Low, High, GroupSize, TacticTileK, ArtifactTileK, PackedScale, true>",
            "return launch_dense_tactic<Low, High, GroupSize, TacticTileK, ArtifactTileK, PackedScale, true>", 1)
    elif name == "coverage-denominator":
        text["geometry_oracle"] = text["geometry_oracle"].replace(
            "for (int visits : output)",
            "for (int visits : std::array<int, 1>{output[0]})", 1)
    elif name != "none":
        raise ValueError(f"unknown plant {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", default="none")
    args = parser.parse_args()
    paths = {
        "dispatch": DISPATCH, "policy": POLICY, "builder": BUILDER,
        "collective": COLLECTIVE, "pack_detail": PACK_DETAIL,
        "launcher": LAUNCHER, "backend": BACKEND,
        "type_oracle": TYPE_ORACLE, "geometry_oracle": GEOMETRY_ORACLE,
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        print("[dense-m1-packed-a] FAIL missing " + ", ".join(missing))
        return 1
    text = {name: path.read_text() for name, path in paths.items()}
    try:
        plant(args.plant, text)
    except ValueError as exc:
        print(f"[dense-m1-packed-a] FAIL {exc}")
        return 1
    errors = source_errors(text)
    if errors:
        label = "RED" if args.plant != "none" else "FAIL"
        print(f"[dense-m1-packed-a] {label}: " + "; ".join(errors))
        return 1
    if args.plant != "none":
        print(f"[dense-m1-packed-a] FAIL plant escaped: {args.plant}")
        return 0
    print("[dense-m1-packed-a] PASS: M1 query+launch share Rows1 packed type; M2..7/default type unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

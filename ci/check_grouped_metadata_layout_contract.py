#!/usr/bin/env python3
"""Source contract for L125's device-free exhaustive G5 metadata oracle."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATHS = (
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp",
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp",
    ROOT / "quactlize/include/ppu_aiu_gemm_mixed_input_group.hpp",
    ROOT / "quactlize/include/grouped_schedule_decode.hpp",
    ROOT / "tests/m8n16_g5_layout_spec.hpp",
    ROOT / "tests/m8n16_g5_contract.hpp",
    ROOT / "tests/test_ppu_m8n16_collective.cu",
    ROOT / "dev/fold_derivation/l125_grouped_metadata_layout.cu",
    ROOT / "dev/fold_derivation/run_l125_grouped_metadata_layout.sh",
)

RETIRED = (
    ROOT / "tests/test_ppu_grouped_metadata_address.cu",
    ROOT / "tools/run_grouped_metadata_address_probe_box.sh",
    ROOT / "ci/check_grouped_metadata_address_contract.py",
    ROOT / "dev/fold_derivation/syntax_baseline/test_ppu_grouped_metadata_address.cu.txt",
)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def section(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) < 1:
        raise ValueError(f"cannot isolate {begin!r} .. {end!r}")
    return text.split(begin, 1)[1].split(end, 1)[0]


def audit(texts: list[str]) -> list[str]:
    collective, helper, grouped, decoder, spec, contract, harness, oracle, runner = texts
    bad: list[str] = []
    try:
        load_init = section(collective, "auto\n  load_init(", "/// Perform a collective-scoped")
    except ValueError as exc:
        return [str(exc)]
    load = compact(load_init)
    helper_flat = compact(helper)
    grouped_flat = compact(grouped)
    decoder_flat = compact(decoder)
    contract_flat = compact(contract)
    harness_flat = compact(harness)
    oracle_flat = compact(oracle)
    runner_flat = compact(runner)

    helper_signature = (
        "template<classScaleTileShape,classElement,classStride>"
        "CUTE_HOST_DEVICEautomake_metadata_tile("
        "Elementconst*base,Strideconst&dS,intN,int64_tscale_k,intL,intl_coord,intn_coord)"
    )
    if helper_flat.count(helper_signature) != 1:
        bad.append("strided metadata helper is not one CUTE_HOST_DEVICE seam")
    for plane in ("S", "Z"):
        call = ("detail::make_metadata_tile<ScaleTileShape>("
                f"mainloop_params.ptr_{plane},mainloop_params.dS,"
                "N,scale_k,L,l_coord,n_coord)")
        if load.count(call) != 1:
            bad.append(f"production load_init does not route {plane} through the exact helper")
    for stale in ("mS_nkl", "mZ_nkl", "make_shape(N,scale_k,L)"):
        if stale in load:
            bad.append(f"production load_init rebuilt metadata locally via {stale}")

    if decoder_flat.count("CUTLASS_HOST_DEVICEconstexprExpertSlicedecode_uniform_z") != 1:
        bad.append("uniform scheduler decoder is not one host/device seam")
    if grouped_flat.count("quactlize::grouped_schedule::decode_uniform_z(int(blockIdx.z),S)") != 1:
        bad.append("shipping grouped uniform scheduler bypasses the shared decoder")

    for token in (
        "kN=m8n16_g5_layout_spec::kN",
        "kK=m8n16_g5_layout_spec::kK",
        "kGroupSize=m8n16_g5_layout_spec::kGroupSize",
        "kExperts=m8n16_g5_layout_spec::kExperts",
        "usingM8=Launch<8,8>",
        "QuantMode::FinegrainedScaleZero",
    ):
        if token not in contract_flat:
            bad.append(f"shipping G5 contract lost {token!r}")
    if harness_flat.count("usingContract=m8n16_g5_contract::Launch<TM,WM>") != 2:
        bad.append("G4/G5 do not both consume the one type-level launch contract")
    for token in (
        "f.zeros.assign(f.scales.size(),half_t(0.0f))",
        "f.zeros.begin()+std::size_t(e)*kScaleK*kN",
        "dZero.copy_from_host(f.zeros.data())",
        "rows_per_expert,kN,kK,kE,kGs",
    ):
        if token not in harness_flat:
            bad.append(f"G5 zero-plane base/extent contract lost {token!r}")

    for token in (
        "usingShipping=m8n16_g5_contract::M8",
        "std::is_same_v<typenameMainloop::MetadataPolicy,RuntimeMetadata>",
        "std::is_same_v<typenameMainloop::GmemTiledCopyZero,RuntimeCopy>",
        "class=CuTe-tiled-copy/Copy_Traits",
        "GmemTiledCopyScalePacked=NOT-SELECTEDscalar-global=NONEnaked-asm=NONE",
        "L125selectedpolicyisnottheshippingG5metadatatype",
        "md::make_metadata_tile<ScaleTile>",
        "tight_metadata_stride()",
        "decode_uniform_z(expert,1)",
        "for(inte=0;e<spec::kExperts;++e)",
        "thread%Plan::thread_slots",
        "partition_S(gZ)",
        "partition_D(sZ)",
        "std::memcpy(dp,sp,sizeof(Element))",
        "tag_roundtrip_bad==0",
        "scheduler_sweep_bad==0",
        "non_target_poison_bad==0",
        "folded_elements==128*kPlane",
        "wrong_l_stride==(spec::kExperts-1)*kPlane",
        "transpose_bad==spec::kExperts*(kPlane-2)",
        "raw_in==640&&raw_oob==384",
        "tile0_holes==192&&tile0_duplicate_coords==64",
        "duplicate_holes==spec::kExperts",
        "explicit-int64-tight-ABI+unique-raw16-tags",
        "independentofCopy_Traitsagreement",
        "zero-planeaddresschainisentirelymodelled",
        "cp.asyncistheterminalbyte-copy",
        "B-addressing=NOT-COVERED",
    ):
        if token not in oracle_flat:
            bad.append(f"L125 lost load-bearing oracle token {token!r}")
    for bypass in (
        "true||std::is_same_v<typenameSelectedPolicy::CollectiveOp,Mainloop>",
        "false&&std::is_same_v<typenameSelectedPolicy::CollectiveOp,Mainloop>",
        "if(false)",
    ):
        if bypass in oracle_flat:
            bad.append(f"L125 contains a constant-bypass escape {bypass!r}")
    if runner_flat.count("-DL125_SELECTED_WN=16") != 1 or \
            runner_flat.count("L125selectedpolicyisnottheshippingG5metadatatype") != 1:
        bad.append("L125 runner lost its compiled wrong-type negative")

    # The device census was retired because it asked hardware to answer pure
    # layout algebra.  Any surviving switch/trace resurrects that false split.
    return bad


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in PATHS if not p.is_file()]
    if missing:
        print("[l125-contract] FAIL: missing " + ", ".join(missing))
        return 1
    survivors = [str(p.relative_to(ROOT)) for p in RETIRED if p.exists()]
    if survivors:
        print("[l125-contract] FAIL: obsolete device probe survives: " + ", ".join(survivors))
        return 1
    texts = [p.read_text() for p in PATHS]
    bad = audit(texts)
    if bad:
        print("[l125-contract] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        (0, "mainloop_params.ptr_Z, mainloop_params.dS,\n            N, scale_k, L, l_coord, n_coord",
         "mainloop_params.ptr_Z, mainloop_params.dS,\n            N, scale_k, L, n_coord, n_coord",
         "production expert coordinate"),
        (0, "detail::make_metadata_tile<ScaleTileShape>(\n            mainloop_params.ptr_Z",
         "detail::make_metadata_tile_bypassed<ScaleTileShape>(\n            mainloop_params.ptr_Z",
         "production helper bypass"),
        (1, "CUTE_HOST_DEVICE auto make_metadata_tile",
         "auto make_metadata_tile", "host/device helper qualifier"),
        (2, "quactlize::grouped_schedule::decode_uniform_z(int(blockIdx.z), S)",
         "quactlize::grouped_schedule::ExpertSlice{int(blockIdx.z) / S, int(blockIdx.z) % S}",
         "shipping scheduler helper bypass"),
        (6, "using Contract = m8n16_g5_contract::Launch<TM, WM>;",
         "using Contract = m8n16_g5_contract::M8;", "G5 exact launch tuple"),
        (7, "thread % Plan::thread_slots", "thread", "physical copy-slot modulo"),
        (7, "std::is_same_v<typename SelectedPolicy::CollectiveOp, Mainloop>",
         "true || std::is_same_v<typename SelectedPolicy::CollectiveOp, Mainloop>",
         "exact type assertion"),
        (7, "duplicate_holes == spec::kExperts", "duplicate_holes > 0",
         "exact duplicate-owner negative"),
    )
    for index, old, new, label in plants:
        planted = list(texts)
        if old not in planted[index]:
            print(f"[l125-contract] FAIL: cannot plant {label}")
            return 1
        planted[index] = planted[index].replace(old, new, 1)
        if not audit(planted):
            print(f"[l125-contract] FAIL: checker accepted planted {label}")
            return 1
    print(f"[l125-contract] PASS: exact G5 type/helper/full sweep pinned; {len(plants)} plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

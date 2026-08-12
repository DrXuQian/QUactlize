#!/usr/bin/env python3
"""Contract for dA outer bases and logical-N metadata residues."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_argument_contract.hpp"
COLLECTIVES = (
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp",
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp",
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp",
)
ORACLE = ROOT / "dev/fold_derivation/l128_mixed_argument_contract.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l128_mixed_argument_contract.sh"
AUDIT = ROOT / "dev/fold_derivation/MIXED_ARGUMENT_ASSUMPTIONS.md"
ADMISSION_ORACLE = ROOT / "dev/fold_derivation/l129_mixed_argument_admission.cu"
ADMISSION_RUNNER = ROOT / "dev/fold_derivation/run_l129_mixed_argument_admission.sh"
KERNELS = (
    ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input.hpp",
    ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_serial.hpp",
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_persistent.hpp",
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_streamk.hpp",
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_marlin.hpp",
    ROOT / "quactlize/include/ppu_aiu_gemm_mixed_input_group.hpp",
    ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_streamk.hpp",
)
GROUP_SCHEDULE = ROOT / "quactlize/include/ppu_group_schedule.hpp"
DISPATCH_POLICY = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp"
MMA_BUILDER = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl"


def flat(text: str) -> str:
    return re.sub(r"\s+", "", text)


def audit(texts: list[str]) -> list[str]:
    helper, ordinary, folded, two_plane, oracle, runner, audit_doc = texts
    bad: list[str] = []
    h = flat(helper)
    for token in (
        "mixed_a_expert_base(",
        "int64_t(group_row_offsets[l_coord])*int64_t(cute::get<0>(dA))",
        "int64_t(l_coord)*int64_t(cute::get<2>(dA))",
        "mixed_logical_n_residue(",
        "N-int64_t(logical_tile_n)*int64_t(n_coord)",
        "mixed_subbyte_l_slice(",
        "make_gmem_ptr<Element>(static_cast<voidconst*>(base))",
        "raw_pointer_cast(logical_nk.data())",
    ):
        if token not in h:
            bad.append(f"shared argument helper lost {token!r}")

    for label, source, expected_b_slices in zip(
            ("ordinary", "fold", "two-plane"),
            (ordinary, folded, two_plane), (1, 1, 2)):
        s = flat(source)
        if s.count("detail::mixed_a_expert_base(") != 1:
            bad.append(f"{label} does not consume the shared dA outer-base seam once")
        if s.count("detail::mixed_logical_n_residue(") != 1:
            bad.append(f"{label} does not consume the logical-N residue seam once")
        if "a_row_off*K" in s:
            bad.append(f"{label} restored compact A outer-base arithmetic")
        if "scale_residue_n=N-size<0>(gB)*n_coord" in s:
            bad.append(f"{label} restored physical-B metadata residue arithmetic")
        if s.count("detail::mixed_subbyte_l_slice<") != expected_b_slices:
            bad.append(
                f"{label} must normalize exactly {expected_b_slices} "
                "noninterleaved subbyte B expert base(s)")

    o = flat(oracle)
    for token in (
        "kRowPitch=kK+16",
        "kExpertPitch=kM*kRowPitch+32",
        "explicit-int64-stride-anchor",
        "physical_formula_red+=legacy!=expected",
        "scope=dA-outer-base+logical-N-residue",
    ):
        if token not in o:
            bad.append(f"L128 lost load-bearing token {token!r}")
    if "nvcc-std=c++17-xcu-arch=sm_80" not in flat(runner):
        bad.append("L128 runner no longer compiles the host oracle")
    for token in (
        "| `dS` |",
        "| Zero-plane stride (`dZ`) |",
        "| Outer A base versus `dA` |",
        "| Runtime `group_size` versus static schedule group size |",
        "| Logical-N residue for metadata in fold / 2-plane |",
        "| Interleaved `dB` |",
        "| Plane-2 `dB2` / `dB2_valid` |",
        "| Packed `ptr_S`, `dS`, and `ptr_Z` semantics |",
        "| Divisibility of interleave/fold/packed extents |",
        "**L128 FIXED**",
        "FORMAT RESTRICTION / MISSING FAIL-CLOSE",
        "place_derived -> recover_derived == identity",
    ):
        if token not in audit_doc:
            bad.append(f"mixed-argument audit lost {token!r}")
    return bad


def admission_audit(helper: str, collectives: tuple[str, ...],
                    kernels: tuple[str, ...], oracle: str, runner: str,
                    audit_doc: str) -> list[str]:
    bad: list[str] = []
    h = flat(helper)
    for token in (
        "structMixedArgumentContract{",
        "mixed_argument_issues(",
        "mixed_arguments_supported(",
        "x.static_group_size==-1&&x.group_size!=x.k",
        "x.dB0!=low_k||x.dB1!=1||x.dBL!=canonical_l",
        "x.ptr_Z_nonnull",
        "x.k%x.group_size!=0",
        "(scale_k/x.scale_tile_k)%x.packed_tiles_per_unit!=0",
        "!mixed_bit_offset_byte_aligned(x.dBL,x.low_bits)",
        "!mixed_bit_offset_byte_aligned(high_l,x.high_bits)",
    ):
        if token not in h:
            bad.append(f"shared admission predicate lost {token!r}")

    for label, source in zip(("ordinary", "fold", "two-plane"), collectives):
        s = flat(source)
        if s.count("staticboolcan_implement(") != 1:
            bad.append(f"{label} collective does not expose exactly one admission method")
        if s.count("detail::mixed_arguments_supported(c)") != 1:
            bad.append(f"{label} collective bypasses the shared admission predicate")

    for path, source in zip(KERNELS, kernels):
        s = flat(source)
        if ("CollectiveMainloop::can_implement(" not in s and
                "mainloop_can_implement<CollectiveMainloop>(" not in s):
            bad.append(f"{path.relative_to(ROOT)} does not forward admission")

    o = flat(oracle)
    for token in (
        "for(intst:{-1,0,16,32,64,128})",
        "noninterleaved_dB2_consumed",
        "c.ptr_Z_nonnull=true",
        "MixedArgumentPackedGroupTail",
        "MixedArgumentPackedTileTail",
        "MixedArgumentPackedUnitTail",
        "MixedArgumentFractionalLowByte",
        "MixedArgumentFractionalHighByte",
        "scope=gs+interleaved-B+B2+packed+divisibility",
    ):
        if token not in o:
            bad.append(f"L129 lost load-bearing token {token!r}")
    if "nvcc-std=c++17-xcu-arch=sm_80" not in flat(runner):
        bad.append("L129 runner no longer compiles the host oracle")
    for token in (
        "L129 closes the remaining admission gaps",
        "`0` is runtime, `-1` is per-column",
        "**L129 FIXED**",
        "noninterleaved path verified consumed",
        "intentional ABI overload, now fail-closed",
    ):
        if token not in audit_doc:
            bad.append(f"audit lost L129 conclusion {token!r}")
    return bad


def gs16_audit(group_schedule: str, dispatch: str, builder: str) -> list[str]:
    bad: list[str] = []
    g, d, b = flat(group_schedule), flat(dispatch), flat(builder)
    if ("structSelector<16>{" not in g or
            "KernelAiuMultistageMixedInputFinegrainedGs16" not in g):
        bad.append("gs16 selector no longer names its own static schedule")
    if (d.count("StaticGroupSize=16") != 2 or
            "MainloopPPUAiuMixedInput2PlaneBase<Stages_,kContinous_,16," not in d):
        bad.append("gs16 must specialize ordinary, fold and two-plane policies")
    if "KernelAiuMultistageMixedInputFinegrainedGs16" not in b:
        bad.append("the collective builder no longer admits the owned gs16 tag")
    return bad


def main() -> int:
    paths = (HELPER, *COLLECTIVES, ORACLE, RUNNER, AUDIT,
             ADMISSION_ORACLE, ADMISSION_RUNNER, *KERNELS,
             GROUP_SCHEDULE, DISPATCH_POLICY, MMA_BUILDER)
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("[mixed-argument-contract] FAIL: missing " + ", ".join(missing))
        return 1
    texts = [p.read_text() for p in paths]
    legacy = texts[:7]
    admission_oracle = texts[7]
    admission_runner = texts[8]
    kernel_texts = tuple(texts[9:16])
    group_schedule, dispatch, builder = texts[16:19]
    bad = audit(legacy)
    bad += admission_audit(legacy[0], tuple(legacy[1:4]), kernel_texts,
                           admission_oracle, admission_runner, legacy[6])
    bad += gs16_audit(group_schedule, dispatch, builder)
    if bad:
        print("[mixed-argument-contract] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        (0, "int64_t(cute::get<0>(dA))", "int64_t(256)", "ragged A row pitch"),
        (0, "int64_t(cute::get<2>(dA))", "int64_t(l_coord) * 0 + int64_t(1792)",
         "uniform A expert pitch"),
        (0, "int64_t(logical_tile_n)", "int64_t(logical_tile_n / 2)",
         "logical N residue"),
        (0, "cute::make_gmem_ptr<Element>(static_cast<void const*>(base))",
         "cute::make_gmem_ptr(base)", "subbyte B pointer overload"),
        (1, "detail::mixed_a_expert_base(", "detail::planted_compact_a_base(",
         "ordinary helper bypass"),
        (2, "detail::mixed_subbyte_l_slice<RealInternalElementB>(",
         "detail::planted_raw_l_slice<RealInternalElementB>(",
         "fold subbyte B helper bypass"),
        (2, "detail::mixed_logical_n_residue(", "detail::planted_physical_n_residue(",
         "fold residue bypass"),
        (4, "physical_formula_red += legacy != expected;",
         "physical_formula_red += false;", "residue negative control"),
    )
    for index, old, new, label in plants:
        planted = list(legacy)
        if old not in planted[index]:
            print(f"[mixed-argument-contract] FAIL: cannot plant {label}")
            return 1
        planted[index] = planted[index].replace(old, new, 1)
        if not audit(planted):
            print(f"[mixed-argument-contract] FAIL: checker accepted planted {label}")
            return 1

    admission_texts = [legacy[0], *legacy[1:4], *kernel_texts,
                       admission_oracle, admission_runner, legacy[6]]
    admission_plants = (
        (0, "x.static_group_size == -1 && x.group_size != x.k", "false",
         "per-column group equality"),
        (0, "x.dB0 != low_k", "false", "interleaved dB pitch"),
        (0, "x.ptr_Z_nonnull", "false", "packed zero contradiction"),
        (0, "!mixed_bit_offset_byte_aligned(x.dBL, x.low_bits)", "false",
         "noninterleaved low-plane byte alignment"),
        (1, "detail::mixed_arguments_supported(c)", "true",
         "ordinary collective bypass"),
        (4, "mainloop_can_implement<CollectiveMainloop>(", "planted_admission_bypass(",
         "vendor kernel forwarding"),
        (11, "c.ptr_Z_nonnull = true;", "c.ptr_Z_nonnull = false;",
         "packed negative control"),
    )
    # Index map above: helper, 3 collectives, 7 kernels, oracle, runner, doc.
    for index, old, new, label in admission_plants:
        planted = list(admission_texts)
        if old not in planted[index]:
            print(f"[mixed-argument-contract] FAIL: cannot plant {label}")
            return 1
        planted[index] = planted[index].replace(old, new, 1)
        p_helper = planted[0]
        p_collectives = tuple(planted[1:4])
        p_kernels = tuple(planted[4:11])
        if not admission_audit(p_helper, p_collectives, p_kernels,
                               planted[11], planted[12], planted[13]):
            print(f"[mixed-argument-contract] FAIL: checker accepted planted {label}")
            return 1

    gs16_plants = (
        (group_schedule, "FinegrainedGs16", "FinegrainedGs32", "gs16 selector alias"),
        (dispatch, "StaticGroupSize = 16", "StaticGroupSize = 32", "gs16 static value"),
        (builder, "KernelAiuMultistageMixedInputFinegrainedGs16", "PlantedMissingGs16",
         "gs16 builder routing"),
    )
    for original, old, new, label in gs16_plants:
        if old not in original:
            print(f"[mixed-argument-contract] FAIL: cannot plant {label}")
            return 1
        planted = original.replace(old, new, 1)
        args = [group_schedule, dispatch, builder]
        args[args.index(original)] = planted
        if not gs16_audit(*args):
            print(f"[mixed-argument-contract] FAIL: checker accepted planted {label}")
            return 1

    run = subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    required = (
        "A uniform_bad=0 ragged_bad=0 explicit-int64-stride-anchor=PASS",
        "A old-row-times-K uniform_red=4/5 ragged_red=4/5 -> EXPECTED-RED",
        "N-residue cases=585 bad=0 physical-TileN-over-F-red=132 -> PASS/EXPECTED-RED",
        "result=PASS scope=dA-outer-base+logical-N-residue",
    )
    if run.returncode != 0:
        print(f"[mixed-argument-contract] FAIL: L128 rc={run.returncode}: {run.stdout[-1200:]}")
        return 1
    absent = [token for token in required if token not in run.stdout]
    if absent:
        print("[mixed-argument-contract] FAIL: output missing " + repr(absent) +
              "\n" + run.stdout[-1200:])
        return 1

    admission_run = subprocess.run(
        ["bash", str(ADMISSION_RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    admission_required = (
        "gs cases=30 accept=10 expected=10 negative_red=2",
        "interleaved canonical=3/3 perturb_red=7/7 noninterleaved_dB2_consumed=YES",
        "packed canonical=1/1 contradictions_red=7/7",
        "residues aligned=3/3 residue_red=5/5",
        "result=PASS scope=gs+interleaved-B+B2+packed+divisibility",
    )
    if admission_run.returncode != 0:
        print(f"[mixed-argument-contract] FAIL: L129 rc={admission_run.returncode}: "
              f"{admission_run.stdout[-1600:]}")
        return 1
    admission_absent = [t for t in admission_required if t not in admission_run.stdout]
    if admission_absent:
        print("[mixed-argument-contract] FAIL: L129 output missing " +
              repr(admission_absent) + "\n" + admission_run.stdout[-1600:])
        return 1
    print("[mixed-argument-contract] PASS: L128 caller strides/residues + L129 "
          "gs/interleaved/packed admission; 585 residue cases and 15 source plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

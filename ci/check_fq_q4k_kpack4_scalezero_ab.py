#!/usr/bin/env python3
"""Source contract for the D32 plain/store/load scale-zero A/B."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
COLLECTIVE = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
BENCH = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
DRIVER = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
L232 = ROOT / "dev/fold_derivation/l232_q4_kpack4_fused_metadata_read.cu"
ANALYZER = ROOT / "tools/analyze_fq_q4k_kpack4_scalezero_ab.py"
RUNNER = ROOT / "tools/run_fq_q4k_kpack4_delivery_ab_box.sh"
LOCAL_GATES = ROOT / "ci/local_gates.py"


class CheckError(ValueError):
    pass


def require(text: str, needles: tuple[str, ...], label: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} lost source seams: {missing}")


def check(texts: list[str]) -> None:
    collective, bench, driver, l232, analyzer, runner, local_gates = texts
    require(collective, (
        "static constexpr bool kFusedScaleZeroRead =",
        "defined(PPU_PACKED_SCALE_FUSED_READ)",
        "kFusedScaleZero && (kPackedFmt == cutlass::gguf_packed::Fmt::Q4K)",
        "static constexpr bool is_fused_scale_zero_read = kFusedScaleZeroRead;",
        "static void reload_scale_zero_metadata(",
        "MetadataAddress::template source<Scale_TileK>",
        "raw_pointer_cast(&scale_src(i))",
        "scale_dst(i) = cutlass::gguf_packed::lo_h2(bits);",
        "zero_dst(i)  = cutlass::gguf_packed::hi_h2(bits);",
        "MetadataPolicy::reload(info, views, group, stage);",
    ), "collective")
    if collective.count("reload_scale_zero_metadata(") != 4:
        raise CheckError("all three coarse/fine reload sites must use one fused-read seam")
    require(bench, (
        "bool scalezero_fused_read = false;",
        "struct MainloopScaleZeroFusedRead",
        "decltype(Mainloop::is_fused_scale_zero_read)",
        "result.scalezero_fused_read = MainloopScaleZeroFusedRead<",
    ), "benchmark")
    require(driver, (
        '"provider_capacity_rows=%d scalezero_fused=%d scalezero_fused_read=%d "',
        "int(cell.scalezero_fused_read)",
    ), "driver")
    require(l232, (
        "using AP0 = typename Types<0>::CollectiveMainloop;",
        "using AP1 = typename Types<1>::CollectiveMainloop;",
        "kQ4KPack4ResolvedDeliveryN == 32",
        "is_fused_scale_zero && AP1::is_fused_scale_zero",
        "int(stride<0>(Half{})) == 2",
        "int(stride<1>(Half{})) == 2 * int(stride<1>(Word{}))",
        "SharedStorage::scale_elements == 2 * cosize_v<Word>",
        "cutlass::gguf_packed::lo_h2(pair).raw() != lo",
        "cutlass::gguf_packed::hi_h2(pair).raw() != hi",
    ), "L232")
    require(analyzer, (
        'SHAPE = (1, 8192, 5120)',
        'VARIANTS = ("plain", "store", "load")',
        "DELIVERY_N = 32",
        'cell.get("scalezero_fused_read")',
        'value.get("scalezero_fused_read") is not fused_read',
        'if tuple(store["resources"]) != tuple(load["resources"]):',
        "Split-K partial workspace changed across variants",
        'row["resource_delta_vs_plain"]',
        'if actual != expected or len(rows) != 6:',
        "Shared Load bank-conflict denominator differs",
        "SHARED_LOAD_CONFLICT_REDUCED",
        "ACU exposed no common bank-conflict row",
        "load_vs_store_delta_pct",
        "args.rounds != 4",
    ), "analyzer")
    require(runner, (
        "scalezero) rounds_default=4",
        'resume="${RESUME:-0}"',
        "RESUME=1 requires explicit OUT",
        "resume source.patch is non-empty",
        "resume current tracked worktree differs from HEAD",
        "resume changed device/source authority outside analysis seam",
        "authority=ANALYSIS_ONLY",
        "resume build/codegen identity is incomplete",
        "resume build identity changed",
        "resume timing log is missing",
        "python3 -B \"$scalezero_analyzer\" self-test",
        "check_fq_q4k_kpack4_scalezero_ab.py",
        "l232_q4_kpack4_fused_metadata_read.cu",
        "specs='plain:32:0:0 store:32:1:0 load:32:1:1'",
        'defs="$defs PPU_PACKED_SCALE_FUSED_READ=1"',
        "fused-read compile ABI differs",
        "non-read control unexpectedly carries the fused-read define",
        "value['scalezero_fused_read']=fused_read",
        '1) order="plain store load"',
        '2) order="load store plain"',
        '3) order="store plain load"',
        '4) order="load plain store"',
        "specs='plain store load'",
        'python3 -B "$scalezero_analyzer" analyze',
        'python3 -B "$scalezero_analyzer" acu',
        "--shape=1x8192x5120",
        "--only-split=4 --tm8-max-m=8",
    ), "runner")
    if runner.count("--shape=1x8192x5120") != 2 or \
            runner.count("--only-split=4 --tm8-max-m=8") != 2:
        raise CheckError("timing and ACU must retain the same exact D32/S4 subject")
    require(local_gates, (
        '("l232_q4_kpack4_fused_metadata_read", [])',
        "-DPPU_PACKED_SCALE_FUSED_READ=1",
        "lint_fq_q4k_kpack4_scalezero_ab",
    ), "local tier")


def main() -> int:
    paths = (COLLECTIVE, BENCH, DRIVER, L232, ANALYZER, RUNNER, LOCAL_GATES)
    texts = [path.read_text() for path in paths]
    check(texts)
    plants = (
        (0, "raw_pointer_cast(&scale_src(i))", "raw_pointer_cast(&cute::get<2>(info)(i))"),
        (0, "zero_dst(i)  = cutlass::gguf_packed::hi_h2(bits);",
         "zero_dst(i)  = cutlass::gguf_packed::lo_h2(bits);"),
        (0, "reload_scale_zero_metadata(", "legacy_reload_scale_zero_metadata("),
        (1, "decltype(Mainloop::is_fused_scale_zero_read)",
         "decltype(Mainloop::is_fused_scale_zero)"),
        (2, "int(cell.scalezero_fused_read)", "int(cell.scalezero_fused)"),
        (3, "int(stride<0>(Half{})) == 2", "int(stride<0>(Half{})) == 1"),
        (4, 'VARIANTS = ("plain", "store", "load")',
         'VARIANTS = ("plain", "load")'),
        (4, 'if actual != expected or len(rows) != 6:',
         'if actual != expected:'),
        (4, 'if tuple(store["resources"]) != tuple(load["resources"]):',
         'if False:'),
        (5, "specs='plain:32:0:0 store:32:1:0 load:32:1:1'",
         "specs='plain:32:0:0 load:32:1:1'"),
        (5, '3) order="store plain load"', '3) order="plain store load"'),
        (5, "non-read control unexpectedly carries the fused-read define",
         "non-read control accepted"),
        (5, "resume changed device/source authority outside analysis seam",
         "resume accepts every changed source"),
        (5, "resume source.patch is non-empty",
         "resume accepts dirty original binaries"),
        (6, "-DPPU_PACKED_SCALE_FUSED_READ=1",
         "-DPPU_PACKED_SCALE_FUSED=1"),
    )
    for index, old, new in plants:
        broken = list(texts)
        if old not in broken[index]:
            raise CheckError(f"negative seam is absent: {old}")
        broken[index] = broken[index].replace(old, new, 1)
        try:
            check(broken)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative stayed green: {old}")
    print("[fq-kpack4-scalezero-ab:self-test] PASS exact D32 plain/store/load "
          "CuTe/raw-bit/codegen/timing/ACU chain; fifteen plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, CheckError, AssertionError) as exc:
        print(f"[fq-kpack4-scalezero-ab:self-test] FAIL: {exc}")
        raise SystemExit(2)

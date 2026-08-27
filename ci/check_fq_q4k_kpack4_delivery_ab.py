#!/usr/bin/env python3
"""Source contract for the matched K-pack4 auto64/D32/D16 experiment."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp"
BUILDER = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl"
COLLECTIVE = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
POLICY = ROOT / "quactlize/include/ppu_mixed_policy.hpp"
BENCH = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
DRIVER = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
L229 = ROOT / "dev/fold_derivation/l229_q4_kpack4_production_type.cu"
L231 = ROOT / "dev/fold_derivation/run_l231_q4_kpack4_production_fragment.sh"
ANALYZER = ROOT / "tools/analyze_fq_q4k_kpack4_delivery_ab.py"
RUNNER = ROOT / "tools/run_fq_q4k_kpack4_delivery_ab_box.sh"


class CheckError(ValueError):
    pass


def require(text: str, needles: tuple[str, ...], label: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} lost source seams: {missing}")


def check(texts: list[str]) -> None:
    dispatch, builder, collective, policy, bench, driver, l229, l231, analyzer, runner = texts
    require(dispatch, (
        "template<class WrappedSchedule_, int DeliveryN_ = 0>",
        "using WrappedSchedule = WrappedSchedule_;\n  static constexpr int DeliveryN = DeliveryN_;",
        "WrappedTraits::DeliveryN",
    ), "dispatch")
    require(builder, (
        "int ScheduledDeliveryN = 0",
        "static constexpr int AutoDeliveryN = Block_N{} < 64 ? Block_N{} : 64;",
        "Block_N{} < ScheduledDeliveryN ? Block_N{} : ScheduledDeliveryN",
        "static constexpr int InstNum = Block_N{} / DeliveryN;",
        "Shape<Int<DeliveryN>, PhysicalBlockK>",
        "Stride<_1, Int<DeliveryN>>",
        "DeliveryN, Swap, true, InstNum",
    ), "builder")
    require(collective, (
        "kQ4KPack4ScheduledDeliveryN",
        "kQ4KPack4ResolvedDeliveryN",
        "kQ4KPack4Transpose ? int(size<0>(InternalSmemLayoutAtomB{})) : 0",
    ), "collective")
    require(policy, (
        "int KPack4DeliveryN = 0",
        "static constexpr int ResolvedKPack4DeliveryN",
        "KPack4DeliveryN == 0",
        "Descriptor::kpack4_scheduled_delivery_n == KPack4DeliveryN",
        "Descriptor::kpack4_resolved_delivery_n ==",
    ), "policy")
    require(bench, (
        "#define FQ_TC_KPACK4_DELIVERY_N 0",
        "int KPack4DeliveryN = FQ_TC_KPACK4_DELIVERY_N",
        "use_kpack4 || KPack4DeliveryN == 0",
        "struct KPack4MainloopDelivery",
        "std::void_t<decltype(Mainloop::kQ4KPack4ScheduledDeliveryN)>",
        "KPack4MainloopDelivery<Mainloop>::value ==",
    ), "benchmark")
    require(driver, (
        "FQ_TC_KPACK4_DELIVERY_N == 16",
        "FQ_TC_KPACK4_DELIVERY_N == 32",
        '"weight_delivery_n=%d "',
    ), "driver")
    if driver.count('"weight_delivery_n=%d "') != 2:
        raise CheckError("shard and shape-done must both bind the delivery cap")
    require(l229, (
        "D32Mainloop::kQ4KPack4ResolvedDeliveryN == 32",
        "D16Mainloop::kQ4KPack4ResolvedDeliveryN == 16",
        "PackedD32Mainloop::kQ4KPack4ResolvedDeliveryN == 32",
        "PackedD16Mainloop::kQ4KPack4ResolvedDeliveryN == 16",
        "D32CapN16Mainloop::kQ4KPack4ResolvedDeliveryN == 16",
        "cosize_v<typename D32Mainloop::SmemLayoutB>",
        "cosize_v<typename D16Mainloop::SmemLayoutB>",
    ), "L229")
    require(l231, (
        "for delivery_n in 32 16",
        '-DL231_KPACK4_DELIVERY_N="$delivery_n"',
        "candidate identity denominator differs",
        "plant=rotated-destination",
        "plant=legacy-loader-stride",
    ), "L231 runner")
    require(analyzer, (
        "DELIVERIES = (0, 32, 16)",
        'SHAPE = (1, 8192, 5120)',
        'marker.get("weight_delivery_n")',
        'if len({tuple(row["resources"])',
        "Bank Conflicts",
        '"bank conflict", "shared load"',
        "ACU arm denominator differs",
    ), "analyzer")
    require(runner, (
        'iterations="${PERF_ITERATIONS:-201}"',
        'rounds="${PERF_ROUNDS:-3}"',
        'run_acu="${RUN_ACU:-1}"',
        'for delivery in 0 32 16',
        'PPU_DEFS="FQ_TC_KPACK4_DELIVERY_N=$delivery"',
        "generated-row/layout build ABI differs",
        'order="auto64 d32 d16"',
        'order="d16 auto64 d32"',
        'order="d32 d16 auto64"',
        "--shape=1x8192x5120",
        "--only-split=4 --tm8-max-m=8",
        "--profile-subject-only",
        '"$acu" --import "$report" --csv --page details',
        '"$acu" --import "$report" --csv --page raw',
        "acu-summary.tsv",
    ), "runner")
    if runner.count("--shape=1x8192x5120") != 2:
        raise CheckError("timing and ACU must use the same exact shape")
    if runner.count("--only-split=4 --tm8-max-m=8") != 2:
        raise CheckError("timing and ACU must use the same exact S4 subject")


def main() -> int:
    paths = (DISPATCH, BUILDER, COLLECTIVE, POLICY, BENCH, DRIVER,
             L229, L231, ANALYZER, RUNNER)
    texts = [path.read_text() for path in paths]
    check(texts)
    plants = (
        (0, "using WrappedSchedule = WrappedSchedule_;\n  static constexpr int DeliveryN = DeliveryN_;",
         "using WrappedSchedule = WrappedSchedule_;\n  static constexpr int DeliveryN = 64;"),
        (1, "static constexpr int InstNum = Block_N{} / DeliveryN;",
         "static constexpr int InstNum = 1;"),
        (1, "Stride<_1, Int<DeliveryN>>", "Stride<_1, Block_N>"),
        (2, "kQ4KPack4Transpose ? int(size<0>(InternalSmemLayoutAtomB{})) : 0",
         "kQ4KPack4Transpose ? 64 : 0"),
        (3, "Descriptor::kpack4_resolved_delivery_n ==",
         "Descriptor::kpack4_resolved_delivery_n !="),
        (4, "use_kpack4 || KPack4DeliveryN == 0", "true"),
        (5, '"weight_delivery_n=%d "', '"weight_delivery_n=0 "'),
        (6, "D32CapN16Mainloop::kQ4KPack4ResolvedDeliveryN == 16",
         "D32CapN16Mainloop::kQ4KPack4ResolvedDeliveryN == 32"),
        (7, "for delivery_n in 32 16", "for delivery_n in 32"),
        (8, 'marker.get("weight_delivery_n")', 'marker.get("ignored")'),
        (8, "ACU arm denominator differs", "ACU arms accepted"),
        (9, 'PPU_DEFS="FQ_TC_KPACK4_DELIVERY_N=$delivery"',
         "PPU_DEFS="),
        (9, 'order="d32 d16 auto64"', 'order="auto64 d32 d16"'),
        (9, "--profile-subject-only", "--profile-entire-run"),
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
    print("[fq-kpack4-delivery-ab:self-test] PASS matched auto64/D32/D16 "
          "type/layout/runtime/codegen/ACU chain; fourteen plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, CheckError, AssertionError) as exc:
        print(f"[fq-kpack4-delivery-ab:self-test] FAIL: {exc}")
        raise SystemExit(2)

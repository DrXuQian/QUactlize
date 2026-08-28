#!/usr/bin/env python3
"""Fail-closed source contract for the full K-pack4 fused-store A/B."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_fq_q4k_kpack4_fused_store_real_shapes_box.sh"
ANALYZER = ROOT / "tools/analyze_fq_q4k_kpack4_fused_store_real_shapes.py"
PILOT = ROOT / "tools/analyze_fq_q4k_kpack4_pilot.py"
COLLECTIVE = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
BENCH = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
DRIVER = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
L232 = ROOT / "dev/fold_derivation/l232_q4_kpack4_fused_metadata_store.cu"


class CheckError(ValueError):
    pass


def require(text: str, needles: tuple[str, ...], label: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} lost source seams: {missing}")


def check(texts: list[str]) -> None:
    runner, analyzer, pilot, collective, bench, driver, l232 = texts
    require(runner, (
        "INTERNAL_SWEEP_SPEC must name COMPLETE inventory-v2 JSON",
        'value["family_count"] == 5 and value["shape_count"] == 20',
        'assert value["decode_m"] == [1,2,4,8]',
        "typed_rows\"] == 144",
        "source_typed_rows\"] == 918",
        "build_arm \"$root\" \"$out\" \"$generated\" \"$jobs\" \"$units\" plain 0",
        "build_arm \"$root\" \"$out\" \"$generated\" \"$jobs\" \"$units\" store 1",
        "FQ_TC_KPACK4_DELIVERY_N=32",
        "PPU_PACKED_SCALE_FUSED=1",
        "plain binary carries fused-store define",
        "retained deleted fused-load define",
        "--iterations=2 --correctness-repeats=1 --only-split=1",
        "--iterations=1 --correctness-repeats=1",
        "--iterations=7 --correctness-repeats=2",
        "screen-union-symbols.txt",
        "confirm-union-symbols.txt",
        '--scalezero-fused "$fused" --delivery-n 32',
        "variants=plain+store D32",
        "source-state.sha256",
        "raw-authority.sha256",
    ), "runner")
    if runner.count("union-symbols --manifest") != 2:
        raise CheckError("screen/confirm must each form one matched arm union")
    if runner.count("analyze_fq_q4k_kpack4_pilot.py\" finalize") != 1:
        raise CheckError("runner must have one looped finalizer seam")
    require(analyzer, (
        'SCHEMA = "quactlize.fq_q4k_kpack4_fused_store_real_shapes.v1"',
        "DELIVERY_N = 32",
        'ARMS = {"plain": False, "store": True}',
        "symbol union requires exact plain/store inputs",
        'value.get("shape_count") != 20',
        'value.get("family_count") != 5',
        'plain_detail.get("symbols_sha256") != store_detail.get("symbols_sha256")',
        'plain_detail.get("confirmed_symbols") != store_detail.get("confirmed_symbols")',
        'if len(inputs) != 2:',
        'if delta <= -threshold:',
        "FQ_KPACK4_FUSED_STORE_CENSUS",
        "mismatched confirm union stayed green",
    ), "analyzer")
    require(pilot, (
        "scalezero_fused: int | None = None",
        "delivery_n: int | None = None",
        'cell.get("scalezero_fused") != str(scalezero_fused)',
        'marker.get("weight_delivery_n") != str(delivery_n)',
        'result["scalezero_fused"] = bool(scalezero_fused)',
        'output["weight_delivery_n"] = delivery_n',
        'command.add_argument("--scalezero-fused", type=int, choices=(0, 1))',
    ), "pilot analyzer")
    require(collective, (
        "static constexpr bool kFusedScaleZero =",
        "SmemLayoutScaleFusedWord",
        "sSZw(n, cute::Int<G>{}, stage) = cutlass::gguf_packed::pack_h2",
        "MetadataPolicy::reload(",
    ), "collective")
    if collective.count("MetadataPolicy::reload(") != 3:
        raise CheckError("collective did not retain all three typed metadata reload seams")
    require(bench, (
        "struct MainloopScaleZeroFused",
        "result.scalezero_fused = MainloopScaleZeroFused<",
    ), "bench")
    require(driver, (
        '"provider_capacity_rows=%d scalezero_fused=%d "',
        "int(cell.scalezero_fused)",
    ), "driver")
    require(l232, (
        "fused-metadata-store",
        "is_fused_scale_zero && AP1::is_fused_scale_zero",
        "SharedStorage::zero_elements == 0",
        "SharedStorage::scale_elements == 2 * cosize_v<Word>",
    ), "L232")
    # The runner deliberately names the retired define once in a negative
    # build-ABI guard.  Reachability is a production-source property, so do
    # not make that guard (or this checker's spelling of it) self-incriminating.
    forbidden = "PPU_PACKED_SCALE_" + "FUSED_READ"
    if any(forbidden in text for text in (pilot, collective, bench, driver)):
        raise CheckError("deleted fused-load experiment remains reachable")


def main() -> int:
    paths = (RUNNER, ANALYZER, PILOT, COLLECTIVE, BENCH, DRIVER, L232)
    texts = [path.read_text() for path in paths]
    check(texts)
    plants = (
        (0, 'value["family_count"] == 5 and value["shape_count"] == 20',
         'value["shape_count"] == 20'),
        (0, "build_arm \"$root\" \"$out\" \"$generated\" \"$jobs\" \"$units\" store 1",
         "build_arm \"$root\" \"$out\" \"$generated\" \"$jobs\" \"$units\" store 0"),
        (0, "--iterations=7 --correctness-repeats=2",
         "--iterations=1 --correctness-repeats=1"),
        (1, 'plain_detail.get("symbols_sha256") != store_detail.get("symbols_sha256")',
         "False"),
        (1, "DELIVERY_N = 32", "DELIVERY_N = 64"),
        (2, 'cell.get("scalezero_fused") != str(scalezero_fused)', "False"),
        (3, "MetadataPolicy::reload(", "legacy_reload("),
        (6, "SharedStorage::zero_elements == 0",
         "SharedStorage::zero_elements == 1"),
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
    print("[fq-kpack4-fused-store-real:self-test] PASS exact 2x144x20 "
          "D32 denominator, matched screen/confirm unions and eight plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, CheckError, AssertionError) as error:
        print(f"[fq-kpack4-fused-store-real:self-test] FAIL: {error}")
        raise SystemExit(2)

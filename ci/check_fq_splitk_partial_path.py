#!/usr/bin/env python3
"""Pin the repaired fixed Split-K partial path without retaining debug scaffolds."""

from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
KERNEL = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/kernel/"
    "ppu_aiu_gemm_mixed_input_splitk_parallel.hpp")
DIRECT = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/"
    "ppu_splitk_direct_accumulator_store.hpp")
PIPELINE = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/"
    "ppu_mixed_pipeline.hpp")
PACKED_OWNER = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/"
    "ppu_packed_metadata_ownership.hpp")
ONE_PLANE = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/"
    "quactlize_mma_mixed_input.hpp")
TWO_PLANE = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/"
    "ppu_mma_aiu_mixed_input_2plane.hpp")
TIMING = ROOT / "benchmarks/splitk_producer_timing.hpp"
SCALEFIRST = ROOT / "benchmarks/scalefirst_internal_sweep_bench.hpp"
FQ = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
PARALLEL_EPILOGUE = ROOT / (
    "third_party/actlize/include/cutlass/epilogue/collective/"
    "ppu_epilogue_vectorized_parallel.hpp")
LAUNCHER = ROOT / "quactlize/include/dense_splitk_parallel_ppu.cuh"
ACTLIZE_COPY = ROOT / "third_party/actlize/include/cute/algorithm/ppu_copy.hpp"
ACTLIZE_ASYNC = ROOT / "third_party/actlize/include/cute/arch/copy_ppu.hpp"
ACTLIZE_M8 = ROOT / "third_party/actlize/include/cute/arch/copy_ppu0010_aiu.hpp"

RETIRED_ONE_PLANE_MACROS = (
    "PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS",
    "PPU_Q4_KPACK4_LEGACY_LOADER_OUTPUT_LAYOUT",
    "PPU_PACKED_PAIR",
    "PPU_SCALE_PAD",
    "PPU_SCALE_SWIZZLE",
    "PPU_SCALE_PREFETCH",
)


def ordered(text: str, needles: tuple[str, ...], label: str) -> list[str]:
    errors: list[str] = []
    cursor = -1
    for needle in needles:
        position = text.find(needle, cursor + 1)
        if position < 0:
            errors.append(f"{label}: missing or reordered {needle!r}")
            break
        cursor = position
    return errors


def check(texts: dict[str, str]) -> list[str]:
    kernel = texts["kernel"]
    direct = texts["direct"]
    pipeline = texts["pipeline"]
    packed_owner = texts["packed_owner"]
    one_plane = texts["one_plane"]
    two_plane = texts["two_plane"]
    timing = texts["timing"]
    scalefirst = texts["scalefirst"]
    fq = texts["fq"]
    parallel_epilogue = texts["parallel_epilogue"]
    launcher = texts["launcher"]
    errors: list[str] = []

    if kernel.count("store_splitk_accumulators_direct(") != 1:
        errors.append("fixed Split-K must have one production direct store")
    if "PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE" in kernel:
        errors.append("retired shared-output negative remains in the product kernel")
    if "partial_epilogue(partial_shape" in kernel:
        errors.append("product Split-K still invokes the historical shared partial epilogue")

    banned = (
        "PPU_SPLITK_SHARED_PREFIX_POLICY",
        "PPU_SPLITK_SHARED_SYNC_POLICY",
        "PPU_PACKED_METADATA_OWNER_ONLY",
        "split_workspace_probe",
        "PPU_MIXED_A_PREPARE_AFTER_CONSUME",
        "PPU_MIXED_A_EXPLICIT_STAGE_VIEW",
        "PPU_PACKED_A_COMPILER_MEMORY_FENCE",
        "PPU_PACKED_A_SYNCHRONOUS_STORE",
        "PPU_PACKED_A_BEFORE_B",
        "PPU_PACKED_A_SEPARATE_ASYNC_GROUP",
        "PPU_M8_DIRECT_X4_PROJECTION",
        "PPU_PACKED_A_ASM_MEMORY_CONTRACT",
        "PPU_M8_LOGICAL_X2_SCALAR_LOAD",
        "PPU_MIXED_ASYNC_SHARED_FENCE",
        "PPU_AIU_SINGLE_LOGICAL_ISSUER",
        "PPU_SPLITK_STABLE_K_TILE_SHAPE",
        "PPU_PACKED_SPLIT_GROUPS",
    )
    combined = "\n".join(texts.values())
    for token in banned:
        if token in combined:
            errors.append(f"retired diagnostic seam returned: {token}")
    for token in RETIRED_ONE_PLANE_MACROS:
        if token in one_plane:
            errors.append(f"retired one-plane selector returned: {token}")

    owner_contract = (
        "static constexpr bool owns_physical_thread(int thread_idx)",
        "return thread_idx >= 0 && thread_idx < owner_threads;",
    )
    for token in owner_contract:
        if packed_owner.count(token) != 1:
            errors.append(f"packed metadata physical-owner contract differs: {token}")

    publication_order = (
        "ScaleCopyPlan::owns_physical_thread(thread_idx)",
        "PackedMetadataOwnership::owns_physical_thread(thread_idx)",
        "ScaleCopyPlan::logical_slot(thread_idx)",
        "PackedMetadataOwnership::copy_owner(thread_idx)",
        "// Start async loads for all pipes but the last",
    )
    for label, source in (("one-plane", one_plane), ("two-plane", two_plane)):
        errors += ordered(source, publication_order, f"{label} exact metadata publication")
        operator_marker = "CUTLASS_DEVICE void\n  operator() ("
        if operator_marker not in source:
            errors.append(f"{label} mixed collective operator marker differs")
        else:
            operator_body = source.split(operator_marker, 1)[1].split("\nprivate:", 1)[0]
            errors += ordered(
                operator_body,
                (
                    "auto extra_input_partitions = partition_extra_inputs(",
                    "// Start async loads for all pipes but the last",
                    "copy_async_extra_info(",
                ),
                f"{label} packed metadata transport order")
            if "if constexpr (kPackedScaleOn && Scale_NumThreads > 32 &&" in operator_body:
                errors.append(f"{label} restored redundant packed init barrier")
        clear_pattern = (
            r"if constexpr\s*\(!kPackedScaleOn\)\s*\{\s*"
            r"if\s*\(scale_copy_owner\)\s*clear\(tSsS\);")
        if source.count("clear(tSsS)") != 1 or not re.search(clear_pattern, source):
            errors.append(f"{label} fp16 initialization is not isolated from packed production")
        if source.count("PACKED METADATA TOTAL-OVERWRITE CONTRACT") != 1:
            errors.append(f"{label} packed total-overwrite contract differs")
        tail = source.split("PACKED METADATA TOTAL-OVERWRITE CONTRACT", 1)[-1]
        errors += ordered(
            tail,
            (
                "if (int64_t(n) >= residue_n) {",
                "sS(n, cute::Int<G>{}, stage) = NonVoidElementScale{};",
                "uint8_t const*",
            ),
            f"{label} decode-owner tail publication")
        if "sZ(n, cute::Int<G>{}, stage) = NonVoidElementZero{};" not in tail:
            errors.append(f"{label} decode owner does not zero the tail zero-plane")
        if label == "one-plane" and \
                "sSZw(n, cute::Int<G>{}, stage) = uint32_t(0);" not in tail:
            errors.append("one-plane fused scale/zero tail is not a total word overwrite")
        if "thread_idx % int(cute::size(GmemTiledCopyScalePacked{}))" in source:
            errors.append(f"{label} restored modulo-replayed packed publishers")
        if source.count("bool scale_copy_owner,\n        bool packed_copy_owner)") != 1:
            errors.append(f"{label} async-copy helper lost distinct scale/packed ownership")
        if source.count("int const scale_thread_idx,\n        int const packed_thread_idx,\n        bool scale_copy_owner)") != 1:
            errors.append(f"{label} partition helper lost distinct logical copy slots")
        if source.count("if (packed_copy_owner)") != 2:
            errors.append(f"{label} packed prologue/steady-state copies are not both owner-guarded")
        if "PPU_MIXED_LEGACY_MODULO_METADATA_PUBLISHERS" in source or \
                "kLegacyModuloMetadataPublishers" in source:
            errors.append(f"{label} retired modulo-publisher negative remains")

    required_direct = (
        "tiled_mma.get_thread_slice(thread_idx)",
        "thread_mma.partition_C(gD)",
        "thread_mma.partition_C(identity)",
        "size(decltype(tD){}) == size(AccumulatorTensor{})",
        "elem_less(coordinates(i)",
        "tD(i) = accumulators(i);",
    )
    for token in required_direct:
        if direct.count(token) != 1:
            errors.append(f"direct-store ownership seam differs: {token}")

    errors += ordered(
        pipeline,
        ("cp_async_wait<0>();", "__syncthreads();", "}  // namespace"),
        "mainloop terminal drain")

    errors += ordered(
        parallel_epilogue,
        (
            "copy(tiled_r2s,",
            "// Step 2. Wait for SMEM writes to complete",
            "synchronize();",
            "copy(tiled_s2r, tDsC, tDrC);",
            "// Step 4. Wait for SMEM reads to complete",
            "synchronize();",
        ),
        "legacy shared epilogue synchronization")

    errors += ordered(
        launcher,
        (
            "if (split_k_slices == 1)",
            "return fpa_intb_ppu::generic_launcher<",
            "WorkspacePlan workspace_plan;",
            "using SplitTypes = KernelTypes<",
        ),
        "shipping S1/custom Split-K type separation")

    errors += ordered(
        timing,
        (
            "hggcEventRecord(events.start, nullptr)",
            "producer()",
            "hggcEventRecord(events.stop, nullptr)",
            "consumer()",
            "hggcEventSynchronize(events.stop)",
            "hggcEventElapsedTime(&ms, events.start, events.stop)",
            "hggcDeviceSynchronize()",
        ),
        "producer-only ordered close")

    for label, source in (("ScaleFirst", scalefirst), ("FullyQuantized", fq)):
        if source.count("splitk_producer_timing::measure(") != 1:
            errors.append(f"{label} must use the common ordered-close timer once")
        if "[&] { return reducer.run(nullptr); }" not in source:
            errors.append(f"{label} timing must enqueue the real reducer")

    owned = ROOT / "quactlize/include"
    owned_headers = {
        path
        for pattern in ("*.hpp", "*.cuh", "*.h")
        for path in owned.rglob(pattern)
    }
    for path in sorted(owned_headers):
        if path == DIRECT:
            continue
        source = path.read_text(errors="replace")
        if "retile_S(accumulators)" in source:
            errors.append(
                f"owned completed-accumulator shared roundtrip remains: "
                f"{path.relative_to(ROOT)}")
    return errors


def self_test(texts: dict[str, str]) -> None:
    assert not check(texts), check(texts)
    plants = (
        ("kernel", "store_splitk_accumulators_direct(",
         "store_splitk_accumulators_retired("),
        ("pipeline", "cp_async_wait<0>();", "cp_async_wait<1>();"),
        ("timing", "auto const consumer_status = consumer();",
         "auto const consumer_status = retired_call();"),
        ("kernel", "namespace cutlass::gemm::kernel {",
         "PPU_SPLITK_SHARED_PREFIX_POLICY\nnamespace cutlass::gemm::kernel {"),
        ("parallel_epilogue", "copy(tiled_s2r, tDsC, tDrC);",
         "copy(retired_s2r, tDsC, tDrC);"),
        ("launcher", "return fpa_intb_ppu::generic_launcher<",
         "return retired_shipping_launcher<"),
        ("packed_owner", "return thread_idx >= 0 && thread_idx < owner_threads;",
         "return true;"),
        ("one_plane", "ScaleCopyPlan::owns_physical_thread(thread_idx)",
         "true"),
        ("two_plane", "PackedMetadataOwnership::owns_physical_thread(thread_idx)",
         "true"),
        ("one_plane", "// Start async loads for all pipes but the last",
         "if constexpr (kPackedScaleOn && Scale_NumThreads > 32 && true) {\n"
         "      __syncthreads();\n    }\n\n    // Start async loads for all pipes but the last"),
        ("two_plane", "if constexpr (!kPackedScaleOn) {",
         "if constexpr (true) {"),
        ("one_plane", "sSZw(n, cute::Int<G>{}, stage) = uint32_t(0);",
         "/* planted missing fused-tail store */"),
        ("two_plane", "sZ(n, cute::Int<G>{}, stage) = NonVoidElementZero{};",
         "/* planted missing zero-tail store */"),
    )
    for key, old, new in plants:
        planted = dict(texts)
        if old not in planted[key]:
            raise AssertionError(f"self-test plant source missing: {old}")
        planted[key] = planted[key].replace(old, new, 1)
        if not check(planted):
            raise AssertionError(f"negative plant stayed green: {key}/{old}")
    for token in RETIRED_ONE_PLANE_MACROS:
        planted = dict(texts)
        planted["one_plane"] = token + "\n" + planted["one_plane"]
        if not check(planted):
            raise AssertionError(f"retired one-plane macro plant stayed green: {token}")


def main() -> int:
    paths = {
        "kernel": KERNEL,
        "direct": DIRECT,
        "pipeline": PIPELINE,
        "packed_owner": PACKED_OWNER,
        "one_plane": ONE_PLANE,
        "two_plane": TWO_PLANE,
        "timing": TIMING,
        "scalefirst": SCALEFIRST,
        "fq": FQ,
        "parallel_epilogue": PARALLEL_EPILOGUE,
        "launcher": LAUNCHER,
        "actlize_copy": ACTLIZE_COPY,
        "actlize_async": ACTLIZE_ASYNC,
        "actlize_m8": ACTLIZE_M8,
    }
    try:
        texts = {name: path.read_text() for name, path in paths.items()}
        errors = check(texts)
        if errors:
            raise ValueError("; ".join(errors))
        self_test(texts)
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-splitk-partial-path] FAIL: {error}", file=sys.stderr)
        return 2
    print("[fq-splitk-partial-path] PASS direct ownership, fixed metadata ownership, "
          "mainloop/epilogue synchronization, distinct S1 type, "
          "ordered-close timing, packed decode-owner total overwrite, and nineteen negative plants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

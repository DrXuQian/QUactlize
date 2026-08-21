#!/usr/bin/env python3
"""Static contract for B2's Marlin-style valid-row FP32 handoff guard."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATHS = (
    ROOT / "third_party/actlize/include/cutlass/block_striped.h",
    ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp",
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_streamk.hpp",
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_streamk.hpp",
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_accumulator_residue_mask.hpp",
    ROOT / "dev/fold_derivation/l124_fp32_residue_mask.cu",
)


def section(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) < 1:
        raise ValueError(f"cannot isolate {begin!r} .. {end!r}")
    return text.split(begin, 1)[1].split(end, 1)[0]


def audit(block: str, scheduler: str, dense: str, grouped: str,
          mask: str, oracle: str) -> list[str]:
    bad: list[str] = []
    try:
        load = section(block, "/// Predicated load & add", "/// Store")
        store = section(block, "/// Predicated store", "// BlockStripedReduce")
        reduce = section(block, "/// Predicated scalar atomic reduction", "/// Utility for performing")
        pred = section(scheduler, "static void\n  fixup_helper_predicated(",
                       "template <class FrgTensorC, class BarrierManager>\n  CUTLASS_DEVICE\n  static void\n  separate_reduction(")
    except ValueError as exc:
        return [str(exc)]

    for name, body, access in (
        ("load_add", load, "access_data[i] = add(access_data[i], access_input[(BlockThreads * i) + thread_idx]);"),
        ("store", store, "access_output[(BlockThreads * i) + thread_idx] = access_data[i];"),
        ("reduce", reduce, "reduce(access_output + (BlockThreads * i) + thread_idx, access_data[i]);"),
    ):
        if body.count("if (predicate(i))") != 1 or body.count(access) != 1:
            bad.append(f"scalar-striped {name} is not guarded by its one predicate")

    for call in (
        "reduction_workspace_array, *accumulator_array, barrier_group_thread_idx, predicate);",
        "*accumulator_array, reduction_workspace_array, barrier_group_thread_idx, predicate);",
    ):
        expected = 2 if call.startswith("reduction_workspace_array") else 1
        if pred.count(call) != expected:
            bad.append(f"predicated fixup does not route all scalar operations through {call!r}")
    if pred.count("BarrierManager::arrive_inc(") != 1 or \
            pred.count("BarrierManager::wait_eq(") != 2 or \
            pred.count("BarrierManager::wait_lt(") != 1:
        bad.append("predicate changed the deterministic wait/arrival protocol")
    if "if (predicate" in pred:
        bad.append("scheduler predicates lock progress instead of only scalar workspace accesses")
    for token in (
        "CUTLASS_ASSERT(!params.requires_separate_reduction());",
        "CUTLASS_ASSERT(!work_tile_info.is_reduction_unit());",
        "static_assert(BlockStripedReduceT::kStripes == size(FrgTensorC{})",
    ):
        if pred.count(token) != 1:
            bad.append(f"predicated fixup lost fail-closed boundary {token!r}")

    for name, caller in (("dense", dense), ("grouped", grouped)):
        for token, count in (
            ("if (!requires_fixup || full_output_tile)", 1),
            ("detail::make_accumulator_residue_mask(", 1),
            ("TileScheduler::fixup(", 2),
        ):
            if caller.count(token) != count:
                bad.append(f"{name} caller requires {count} occurrence(s) of {token!r}")

    for token in (
        "auto coordinates = tiled_mma.get_thread_slice(thread_idx).partition_C(identity);",
        "auto physical_to_fragment = right_inverse(accumulators.layout());",
        "int(get<0>(mn)) < int(get<0>(residue_mn))",
        "int(get<1>(mn)) < int(get<1>(residue_mn))",
    ):
        if mask.count(token) != 1:
            bad.append(f"accumulator mask lost exact layout-derived coordinate {token!r}")
    for token in (
        "constexpr uint32_t kPoison = 0x7fc12345u;",
        "for (int peers = 1; peers <= 4; ++peers)",
        "bits(pred_out[slot]) == bits(full_out[slot])",
        "pred_touched[slot] == 0",
        "planted-coordinate=%s planted-address=%s ",
        "planted-fixed128-cohort=%s result=%s",
    ):
        if oracle.count(token) != 1:
            bad.append(f"l124 semantic oracle lost {token!r}")
    return bad


def main() -> int:
    if any(not path.is_file() for path in PATHS):
        missing = [str(path.relative_to(ROOT)) for path in PATHS if not path.is_file()]
        print("[fp32-residue-contract] FAIL: missing " + ", ".join(missing))
        return 1
    texts = [path.read_text() for path in PATHS]
    bad = audit(*texts)
    if bad:
        print("[fp32-residue-contract] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        (0, "if (predicate(i)) {", "if (true) {", "unguarded scalar workspace access"),
        (1, "BarrierManager::arrive_inc(\n", "if (predicate(0)) BarrierManager::arrive_inc(\n",
         "predicate-controlled lock progress"),
        (2, "if (!requires_fixup || full_output_tile)", "if (true || full_output_tile)",
         "dense residue path bypass"),
        (3, "detail::make_accumulator_residue_mask(", "detail::make_accumulator_full_mask(",
         "grouped mask bypass"),
        (4, "int(get<0>(mn)) < int(get<0>(residue_mn))",
         "int(get<0>(mn)) < int(get<1>(residue_mn))", "swapped mask axis"),
        (5, "pred_touched[slot] == 0", "pred_touched[slot] >= 0", "invalid poison ignored"),
    )
    for index, old, new, label in plants:
        planted = list(texts)
        if old not in planted[index]:
            print(f"[fp32-residue-contract] FAIL: cannot plant {label}")
            return 1
        planted[index] = planted[index].replace(old, new, 1)
        if not audit(*planted):
            print(f"[fp32-residue-contract] FAIL: checker accepted planted {label}")
            return 1
    print(f"[fp32-residue-contract] PASS: {len(plants)} source plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

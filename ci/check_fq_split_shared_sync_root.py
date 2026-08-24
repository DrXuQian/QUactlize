#!/usr/bin/env python3
"""Guard the single-variable shared-epilogue synchronization root experiment."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_shared_epilogue_sync_probe.hpp"
KERNEL = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp"
RUNNER = ROOT / "tools/run_fq_q4k_split_shared_sync_root_box.sh"
ORACLE = ROOT / "dev/fold_derivation/l223_fq_splitk_shared_epilogue_layout.cu"
EVIDENCE = ROOT / "dev/fold_derivation/l223_fq_splitk_shared_epilogue_layout.expected.txt"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_texts(helper: str, kernel: str, runner: str,
                oracle: str, evidence: str) -> None:
    require(helper.count(
        "splitk_shared_epilogue_sync<SyncPolicy, TiledCopyS2R>();") == 2,
        "shared clone must contain exactly two policy synchronization calls")
    for needle in (
        "copy(tiled_r2s, tCaC(_, mma_m, mma_n),",
        "copy(tiled_s2r, tDsC, tDrC);",
        "copy(CopyAtomR2G{}, tDrD(_, m, n), tDgDmn(_, m, n));",
        "ThreadEpilogueOp epilogue_op{};",
        "params.ptr_D += partial_plane * get<2>(params.dD);",
    ):
        require(helper.count(needle) == 1,
                f"shared clone operation changed: {needle}")
    require(helper.count("uint32_t(0)") == 1,
            "legacy integer-zero barrier arm changed")
    require(helper.count(
        "cutlass::arch::ReservedNamedBarriers::EpilogueBarrier") == 1,
        "reserved epilogue barrier arm changed")
    require(helper.count("__syncthreads();") == 1,
            "CTA barrier control changed")
    for forbidden in ("__threadfence", "atomicAdd", "fence_view_async_shared"):
        require(forbidden not in helper,
                f"probe changed more than synchronization selection: {forbidden}")

    require(kernel.count("#if defined(PPU_SPLITK_SHARED_SYNC_POLICY)") == 3,
            "kernel probe include/exclusion/call guards changed")
    require(kernel.count(
        "store_splitk_accumulators_shared_sync_probe<") == 1,
        "kernel shared synchronization probe call changed")
    require(kernel.count("store_splitk_accumulators_direct(") == 1,
            "production direct accumulator default changed")
    require(kernel.index("#if defined(PPU_SPLITK_SHARED_SYNC_POLICY)",
                         kernel.index("int const plane")) <
            kernel.index("#elif defined(PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE") <
            kernel.index("store_splitk_accumulators_direct(") ,
            "diagnostic/vendor/production branch order changed")

    expected_defs = {
        "vendor-user0": "PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1",
        "clone-user0": "PPU_SPLITK_SHARED_SYNC_POLICY=1",
        "reserved-id1": "PPU_SPLITK_SHARED_SYNC_POLICY=2",
        "cta": "PPU_SPLITK_SHARED_SYNC_POLICY=3",
    }
    require(runner.count(
        "for arm in vendor-user0 clone-user0 reserved-id1 cta") == 1,
        "runner arm denominator changed")
    for arm, define in expected_defs.items():
        require(runner.count(f"{arm}) defs='{define}'") == 1,
                f"runner define binding changed for {arm}")

    for macro in ("L223_BAD_R2S_ROTATE", "L223_BAD_S2R_THREAD_MODULO"):
        require(oracle.count(macro) >= 2,
                f"L223 negative plant missing: {macro}")
    require("verdict=PASS" in evidence and evidence.count("verdict=FAIL") == 2,
            "L223 committed positive/negative evidence changed")
    require("reader_holes=256 reader_duplicates=256" in evidence,
            "L223 ownership negative evidence changed")
    require("s2r_coord_bad=512 s2r_value_bad=512" in evidence,
            "L223 coordinate negative evidence changed")


def self_test(helper: str, kernel: str, runner: str,
              oracle: str, evidence: str) -> None:
    check_texts(helper, kernel, runner, oracle, evidence)
    plants = (
        (helper.replace(
            "splitk_shared_epilogue_sync<SyncPolicy, TiledCopyS2R>();",
            "", 1), kernel, runner, oracle, evidence),
        (helper.replace("uint32_t(0)",
                        "cutlass::arch::ReservedNamedBarriers::EpilogueBarrier"),
         kernel, runner, oracle, evidence),
        (helper.replace("__syncthreads();",
                        "__threadfence(); __syncthreads();"),
         kernel, runner, oracle, evidence),
        (helper, kernel.replace("store_splitk_accumulators_direct(",
                                "store_splitk_accumulators_removed("),
         runner, oracle, evidence),
        (helper, kernel, runner.replace(
            "reserved-id1) defs='PPU_SPLITK_SHARED_SYNC_POLICY=2'",
            "reserved-id1) defs='PPU_SPLITK_SHARED_SYNC_POLICY=1'"),
         oracle, evidence),
    )
    for plant in plants:
        try:
            check_texts(*plant)
        except ValueError:
            pass
        else:
            raise AssertionError("source-seam negative plant stayed green")


def main() -> int:
    try:
        texts = tuple(path.read_text() for path in
                      (HELPER, KERNEL, RUNNER, ORACLE, EVIDENCE))
        self_test(*texts)
        print("[fq-shared-sync-root-source:self-test] PASS exact two-sync "
              "factorial, production-default isolation, L223 oracle and "
              "five source-seam negatives")
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-shared-sync-root-source] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

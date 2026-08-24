#!/usr/bin/env python3
"""Source contract for the exact packed-metadata publisher A/B."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
COLLECTIVE = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
OWNERSHIP = ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/ppu_packed_metadata_ownership.hpp"
ORACLE = ROOT / "dev/fold_derivation/l221_packed_metadata_publishers.cu"
RUNNER = ROOT / "tools/run_fq_q4k_packed_owner_probe_box.sh"
CHECKER = ROOT / "tools/check_fq_packed_owner_candidate.py"


class CheckError(ValueError):
    pass


def require(label: str, text: str, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} contract missing: {missing}")


def check(collective: str, ownership: str, oracle: str,
          runner: str, checker: str) -> None:
    require("collective", collective, (
        "PPU_PACKED_METADATA_OWNER_ONLY",
        "static constexpr bool kPackedMetadataOwnerOnly = false;",
        "PackedMetadataOwnership::owns_physical_thread(thread_idx)",
        "if (packed_metadata_copy_owner)",
        "owner-only packed metadata copy cannot feed duplicate-owner split-group decode",
    ))
    if collective.count("if (packed_metadata_copy_owner)") != 2:
        raise CheckError("candidate must guard exactly the two packed copy sites")
    require("ownership", ownership, (
        "owns_physical_thread(int thread_idx)",
        "thread_idx < owner_threads",
    ))
    require("oracle", oracle, (
        "legacy-modulo-all",
        "owner-only",
        "read_duplicate_writer_overlap == 1024",
        "first_second_decoder_warp_column == 32",
    ))
    require("runner", runner, (
        "LEGACY_ARTIFACT",
        "PPU_PACKED_METADATA_OWNER_ONLY=1",
        "check_fq_packed_owner_candidate.py",
        "--split-workspace-probe",
    ))
    if runner.count("PPU_PACKED_METADATA_OWNER_ONLY=1") != 2:
        raise CheckError("runner must bind and verify the candidate define exactly twice")
    require("checker", checker, (
        'verdict = "OWNER_RACE_CLOSED_ALL_EXACT"',
        'verdict = "OWNER_RACE_CLOSED_DIRECT_GAP_REMAINS"',
        'verdict = "OWNER_ONLY_REFUTED"',
        'index % 64 != 32',
    ))


def main() -> int:
    paths = (COLLECTIVE, OWNERSHIP, ORACLE, RUNNER, CHECKER)
    texts = [path.read_text() for path in paths]
    check(*texts)
    plants = (
        (0, "if (packed_metadata_copy_owner)", "if (true)"),
        (1, "thread_idx < owner_threads", "thread_idx <= owner_threads"),
        (2, "read_duplicate_writer_overlap == 1024",
         "read_duplicate_writer_overlap == 0"),
        (3, "PPU_PACKED_METADATA_OWNER_ONLY=1",
         "PPU_PACKED_METADATA_OWNER_ONLY=0"),
        (4, "index % 64 != 32", "index % 64 != 0"),
    )
    for index, old, new in plants:
        changed = list(texts)
        changed[index] = changed[index].replace(old, new, 1)
        try:
            check(*changed)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative stayed green: {old}")
    print("[fq-packed-owner-source:self-test] PASS: exact guard count, real-layout "
          "duplicate negative, build define and local-N boundary; five plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError) as error:
        print(f"[fq-packed-owner-source:self-test] FAIL: {error}")
        raise SystemExit(2)

#!/usr/bin/env python3
"""Bind all mixed collectives to the independent A/B register schedule."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / (
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/"
    "detail/ppu_mixed_a_schedule.hpp"
)
COLLECTIVES = tuple(
    ROOT / "quactlize/include/actlize_extensions/cutlass/gemm/collective" / name
    for name in (
        "quactlize_mma_mixed_input.hpp",
        "ppu_mma_aiu_fold.hpp",
        "ppu_mma_aiu_mixed_input_2plane.hpp",
    )
)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def replace_all(source: str, old: str, new: str) -> str:
    """Plant every compile-time branch, not just the first duplicate arm."""
    if old not in source:
        raise ValueError(f"plant source lost {old}")
    return source.replace(old, new)


def audit(detail: str, collectives: tuple[str, ...]) -> list[str]:
    bad: list[str] = []
    d = compact(detail)
    for needle in (
        "struct MixedARegisterSchedule",
        "MmaKAtoms % ACopyBlocks == 0",
        "MmaKAtoms % BCopyBlocks == 0",
        "return (ABlock * AAtomsPerCopy) / BAtomsPerCopy",
        "ACopyBlocks == 1 && BCopyBlocks > 1 && BBlock == 0",
        "BBlock == BCopyBlocks - 1",
        "prepare_mixed_a_for_b",
        "finish_mixed_a_after_consume",
    ):
        if needle not in d:
            bad.append(f"shared schedule lost {needle}")

    for path, source in zip(COLLECTIVES, collectives):
        c = compact(source)
        label = path.name
        for needle in (
            "using ARegisterSchedule = detail::MixedARegisterSchedule<",
            "detail::prepare_mixed_a_for_b<ARegisterSchedule>",
            "detail::finish_mixed_a_after_consume<ARegisterSchedule>",
        ):
            if needle not in c:
                bad.append(f"{label} lost {needle}")
        # A B-derived k_block may still index B.  It must never directly index
        # an independently retiled A view outside the shared helper.
        if re.search(
            r"copy\s*\(\s*smem_tiled_copy_A\s*,[^;]*"
            r"tCrA_copy_view\s*\([^;]*k_block",
            without_comments(source),
            re.S,
        ):
            bad.append(f"{label} again indexes A with B's k_block")
    return bad


def main() -> int:
    detail = DETAIL.read_text(encoding="utf-8")
    collectives = tuple(path.read_text(encoding="utf-8") for path in COLLECTIVES)
    bad = audit(detail, collectives)
    if bad:
        print("[mixed-a-schedule] FAIL: " + "; ".join(bad), file=sys.stderr)
        return 1

    plants = (
        (
            detail.replace("ACopyBlocks == 1", "ACopyBlocks == 0", 1),
            collectives,
            "wrap condition",
        ),
        (
            detail,
            (replace_all(collectives[0],
                "detail::prepare_mixed_a_for_b<ARegisterSchedule>",
                "removed_prepare<ARegisterSchedule>"),) + collectives[1:],
            "ordinary bypass",
        ),
        (
            detail,
            collectives[:1] + (replace_all(collectives[1],
                "detail::finish_mixed_a_after_consume<ARegisterSchedule>",
                "removed_finish<ARegisterSchedule>"),) + collectives[2:],
            "fold finish bypass",
        ),
        (
            detail,
            collectives[:2] + (collectives[2] + (
                "\ncopy(smem_tiled_copy_A, tCsA_p(_,_,k_block), "
                "tCrA_copy_view(_,_,k_block));\n"),),
            "two-plane direct-coordinate relapse",
        ),
    )
    for planted_detail, planted_collectives, label in plants:
        if not audit(planted_detail, planted_collectives):
            print(f"[mixed-a-schedule] FAIL: {label} plant was false-green",
                  file=sys.stderr)
            return 1

    print(
        "[mixed-a-schedule] PASS: ordinary/fold/two-plane derive A ownership "
        "through MMA-K atoms; wrap/direct-coordinate/bypass plants red"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Static contract for the TM8/TN64/WN16 epilogue legacy-red closure."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests/test_ppu_m8n16_collective.cu"
RUNNER = ROOT / "tools/run_m8n16_epilogue_topology_box.sh"
BUILDER = (ROOT / "third_party/actlize/include/cutlass/epilogue/collective/"
           "builders/ppu_builder.inl")


def compact_code(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def audit(harness: str, runner: str, builder: str) -> list[str]:
    h = compact_code(harness)
    r = compact(runner)
    b = compact_code(builder)
    bad: list[str] = []

    harness_tokens = (
        "structG4KpackLegacyEpilogueTypes:G4KpackEpilogueTypes<TN,WN>",
        "usingG4KpackCandidateSubject=G4KpackEpilogueTypes<64,16>;",
        "usingG4KpackLegacySubject=G4KpackLegacyEpilogueTypes<64,16>;",
        "cute::size(typenameLegacyTiledCopyS2R::Tiler_MN{})==1024",
        "cute::size(typenameEpilogue::SmemLayout{})==512",
        "G4KpackCandidateSubject::Epilogue::TiledCopyS2R::Tiler_MN{})==512",
        'check_g4_first_tile_ownership<Legacy>("tn64-wn16-legacy")',
        "errors+=legacy_red==64?0:1;",
        'check_g4_first_tile_ownership<Candidate>("tn64-wn16-candidate")',
        'check_g4_epilogue_topology<Candidate>("tn64-wn16-candidate",9)',
        "run_g4_epilogue_topology_arm<Candidate,false>(9)",
        'FQ_M8_EPILOGUE_TOPOLOGY_ABM=%dN=%dcross_bad=%d/%zu',
        "usingTiledCopy=typenameTypes::Epilogue::TiledCopyS2R;",
        "CopyThreads=int(typenameTiledCopy::TiledNumThr{});",
        "OutputAlignment=int(typenameTiledCopy::TiledNumVal{});",
        "CopyTileValues=int(cute::size(typenameTiledCopy::Tiler_MN{}));",
        "static_assert(CopyTileValues==8*TN",
        "OutputAlignment==8&&EpiThreadM==8&&EpiThreadN==4",
        "OutputAlignment==4&&EpiThreadM==8&&EpiThreadN==16",
    )
    for token in harness_tokens:
        if h.count(token) != 1:
            bad.append(f"harness must contain exactly one {token!r}")
    if "constexprintOutputAlignment=8;" in h:
        bad.append("harness must derive output alignment from TiledCopyS2R")

    builder_tokens = (
        "staticconstexprintSharedFragmentSize=SmemM*BlockN/ThreadNum;",
        "staticconstexprintRequestedAlignment=platform::min(AlignmentC,AlignmentD);",
        "staticconstexprintEffectiveAlignment=platform::min(RequestedAlignment,"
        "platform::min(FragmentSize,SharedFragmentSize));",
        "AutoVectorizingCopyWithAssumedAlignment<sizeof(ElementD)*EffectiveAlignment*8>",
    )
    for token in builder_tokens:
        if b.count(token) != 1:
            bad.append(f"builder must contain exactly one {token!r}")

    runner_tokens = (
        ("check_m8n16_epilogue_topology_contract.py", 1),
        ("ownership_bad=64/576", 1),
        ("row8_written=64/64", 1),
        ("cohort_written=[16,16,16,16]ILLEGAL_ROW8_WRITE", 1),
        ("ownership_bad=0/576", 2),
        ("row8_written=0/64", 2),
        ("cohort_written=[0,0,0,0]EXACT_OWNER", 2),
        ("positive_bad=0/576", 2),
        ("negative_oracle_bad=0/576", 2),
        ("cta_threads=32fragment=8output_alignment=8epi_thread_map=8x4", 1),
        ("cta_threads=128fragment=4output_alignment=4epi_thread_map=8x16", 2),
        ("cross_bad=0/576verdict=CANDIDATE_MATCHES_CONTROL", 1),
        ("verdict=LEGACY_RED_CANDIDATE_GREEN", 1),
        ('["$verdict"=LEGACY_RED_CANDIDATE_GREEN]', 1),
    )
    for token, count in runner_tokens:
        if r.count(token) != count:
            bad.append(f"runner must contain exactly {count} {token!r}")
    return bad


def main() -> int:
    harness = HARNESS.read_text()
    runner = RUNNER.read_text()
    builder = BUILDER.read_text()
    bad = audit(harness, runner, builder)
    if bad:
        print("[m8-epilogue-topology] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        ("harness", "== 1024", "== 512", "legacy virtual tile"),
        ("harness", "== 512,\n              \"shipping TM8", "== 1024,\n              \"shipping TM8", "candidate exact tile"),
        ("harness", "legacy_red == 64", "legacy_red == 0", "legacy red denominator"),
        ("harness", "int(typename TiledCopy::TiledNumVal{})", "8",
         "type-derived output alignment"),
        ("builder", "platform::min(RequestedAlignment,\n"
                    "                    platform::min(FragmentSize, SharedFragmentSize))",
         "RequestedAlignment", "fragment cap"),
        ("builder", "sizeof(ElementD) * EffectiveAlignment * 8",
         "sizeof(ElementD) * AlignmentD * 8", "R2G cap"),
        ("runner", "row8_written=64/64", "row8_written=0/64", "legacy witness"),
        ("runner", '"$subject_owner" != *\' row8_written=0/64 \'*',
         '"$subject_owner" != *\' row8_written=64/64 \'*', "candidate witness"),
        ("runner",
         '"$subject" != *\' cta_threads=128 fragment=4 output_alignment=4 epi_thread_map=8x16 \'*',
         '"$subject" != *\' cta_threads=128 fragment=4 output_alignment=8 epi_thread_map=16x8 \'*',
         "candidate reported topology"),
        ("runner", "verdict=LEGACY_RED_CANDIDATE_GREEN",
         "verdict=EPILOGUE_EXCLUDED", "admission verdict"),
    )
    for target, old, new, label in plants:
        original = {"harness": harness, "runner": runner, "builder": builder}[target]
        if original.count(old) != 1:
            print(f"[m8-epilogue-topology] FAIL: cannot plant {label}")
            return 1
        ph, pr, pb = harness, runner, builder
        if target == "harness":
            ph = harness.replace(old, new, 1)
        elif target == "runner":
            pr = runner.replace(old, new, 1)
        else:
            pb = builder.replace(old, new, 1)
        if not audit(ph, pr, pb):
            print(f"[m8-epilogue-topology] FAIL: checker accepted planted {label}")
            return 1

    print("[m8-epilogue-topology] PASS legacy=virtual-16x64/RED "
          "candidate=exact-8x64/GREEN; type-derived alignment/map; "
          "fragment-capped S2R+R2G; ten plants RED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

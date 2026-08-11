#!/usr/bin/env python3
"""Keep #112/G5's caller-slot and real-expert index spaces distinct."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests/test_ppu_m8n16_collective.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l132_g5_harness_slot_map.sh"


def violations(text: str) -> list[str]:
    flat = "".join(text.split())
    required = {
        '#include"m8n16_g5_slot_map.hpp"': "shared slot-map helper is not included",
        "f.expert_for_slot=std::move(route.expert_for_slot)":
            "caller slot -> real expert map is not retained",
        "f.slot_for_expert=std::move(route.slot_for_expert)":
            "real expert -> caller slot map is not retained",
        "f.q_by_slot[slot]=std::move(q)":
            "Q oracle is no longer stored by caller slot",
        "f.s_by_slot[slot]=std::move(s)":
            "scale oracle is no longer stored by caller slot",
        "f.z_by_slot[slot]=std::move(z)":
            "zero oracle is no longer stored by caller slot",
        "f.a_by_slot[slot]=make_a_salted(rows_per_expert,salt)":
            "A oracle is no longer stored by caller slot",
        "f.A.begin()+std::size_t(f.row_offsets[expert])*kK":
            "A payload is not materialized at its real expert row",
        "std::size_t(f.row_offsets[expert])*kN":
            "ordinary G5 output is not read from ptr_D[expert]'s row",
        "std::size_t(f.row_offsets[next_expert])*kN":
            "row-shift negative does not name the next slot's real expert",
        "std::size_tconstrow=std::size_t(f.row_offsets[want])":
            "zero-plane ID probe still assumes slot is the output row",
    }
    bad = [why for token, why in required.items() if token not in flat]
    forbidden = {
        "f.q_e.push_back": "deleted expert-order Q push returned",
        "f.s_e.push_back": "deleted expert-order scale push returned",
        "f.z_e.push_back": "deleted expert-order zero push returned",
        "std::size_t(slot)*rows_per_expert*kN":
            "ordinary comparison again assumes slot == output row",
        "got[std::size_t(slot)*kN":
            "ID probe again assumes slot == output row",
    }
    bad.extend(why for token, why in forbidden.items() if token in flat)
    return bad


def main() -> int:
    source = HARNESS.read_text()
    bad = violations(source)
    if bad:
        for item in bad:
            print(f"[g5-slot-contract] FAIL: {item}", file=sys.stderr)
        return 1

    run = subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(run.stdout, end="")
    if run.returncode != 0 or not all(
        marker in run.stdout
        for marker in (
            "ordered-default row-diff=0/8 byte-diff=0/16 -> IDENTICAL",
            "shuffled rows=14,0,12,2,10,4,8,6 row-diff=0/8 recover-bad=0/16 -> PASS",
            "planted-old expert-push=8/8 slot-base=16/16 duplicate-valid=0 -> EXPECTED_RED",
            "G5 harness slot->expert->row_offsets PASS",
        )
    ):
        print("[g5-slot-contract] FAIL: L132 did not prove all four arms", file=sys.stderr)
        return 1

    # Each production seam must be load-bearing in the source checker itself.
    plants = [
        ('#include "m8n16_g5_slot_map.hpp"', '#include "m8n16_g5_contract.hpp"'),
        ("f.q_by_slot[slot] = std::move(q);", "f.q_e.push_back(std::move(q));"),
        ("std::size_t const base = std::size_t(f.row_offsets[expert]) * kN;",
         "std::size_t const base = std::size_t(slot) * rows_per_expert * kN;"),
        ("std::size_t const row = std::size_t(f.row_offsets[want]);",
         "std::size_t const row = std::size_t(slot);"),
    ]
    for old, new in plants:
        if source.count(old) != 1:
            print(f"[g5-slot-contract] FAIL: plant seam count for {old!r}", file=sys.stderr)
            return 1
        if not violations(source.replace(old, new, 1)):
            print(f"[g5-slot-contract] FAIL: planted regression escaped: {old!r}", file=sys.stderr)
            return 1

    print(f"[g5-slot-contract] PASS: L132 + {len(plants)} source plants; "
          "ordered bytes unchanged, shuffled ids explicit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Adjudicate the exact synchronization root of the legacy Split-K shared epilogue."""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import sys

from check_fq_packed_owner_candidate import (
    PROVIDER_NAMES,
    direct_summary,
    probe_groups,
    probe_summary,
)
from check_fq_split_workspace_probe import AP0, AP1, EXPECTED


ARM_NAMES = ("vendor-user0", "clone-user0", "reserved-id1", "cta")


def summarize(name: str, direct: str, probe: str) -> dict[str, object]:
    groups = probe_groups(probe)
    total, cells = probe_summary(groups, False)
    valid, direct_failures, failed = direct_summary(direct)
    if valid != 6:
        raise ValueError(f"{name}: direct valid-cell denominator differs")
    clean = not any(
        int(total[key])
        for key in ("partial_bad", "canary", "host_bad", "observed_bad")
    ) and direct_failures == 0
    print(
        "FQ_SHARED_SYNC_ARM "
        f"arm={name} samples={total['samples']} "
        f"bad_samples={total['bad_samples']} "
        f"partial_value_raw_bad={total['partial_bad']} "
        f"canary_words={total['canary']} "
        f"host_reduce_raw_bad={total['host_bad']} "
        f"sync_only_raw_bad={total['sync_bad']} "
        f"observed_reducer_raw_bad={total['observed_bad']} "
        f"direct_failures={direct_failures} clean={int(clean)}")
    for (symbol, split), cell in sorted(
            cells.items(), key=lambda item: (PROVIDER_NAMES[item[0][0]],
                                             int(item[0][1]))):
        origins = ",".join(str(value) for value in
                           sorted(cell["stripe_origins"])) or "NONE"
        print(
            "FQ_SHARED_SYNC_CELL "
            f"arm={name} provider={PROVIDER_NAMES[symbol]} S={split} "
            f"samples={cell['samples']} bad_samples={cell['bad_samples']} "
            f"partial_value_raw_bad={cell['partial_bad']} "
            f"bad_plane_mask=0x{int(cell['bad_plane_mask']):x} "
            f"stripe_origins={origins} "
            f"direct_failed={int((symbol, split) in failed)}")
    return {
        "clean": clean,
        "partial_bad": int(total["partial_bad"]),
        "bad_samples": int(total["bad_samples"]),
        "direct_failures": direct_failures,
    }


def adjudicate(logs: dict[str, tuple[str, str]]) -> str:
    arms = {name: summarize(name, *logs[name]) for name in ARM_NAMES}
    vendor = arms["vendor-user0"]
    clone = arms["clone-user0"]
    reserved = arms["reserved-id1"]
    cta = arms["cta"]
    if not vendor["partial_bad"]:
        verdict = "UNADJUDICATED_VENDOR_DID_NOT_REPRODUCE"
    elif not clone["partial_bad"]:
        verdict = "UNADJUDICATED_EXACT_CLONE_DID_NOT_REPRODUCE"
    elif reserved["clean"] and cta["clean"]:
        verdict = "INTEGER_USER_BARRIER_ID6_CAUSAL"
    elif not reserved["clean"] and cta["clean"]:
        verdict = "PPU_NAMED_BARRIER_SHARED_ORDERING_CAUSAL"
    elif reserved["clean"] and not cta["clean"]:
        verdict = "UNADJUDICATED_CTA_CONTROL_FAILED"
    else:
        verdict = "UNADJUDICATED_SHARED_HANDOFF_REMAINS"
    print(
        "FQ_SHARED_SYNC_ROOT "
        f"verdict={verdict} static_layout=L223-EXACT-BIJECTION "
        "changed_operation=TWO_SYNCHRONIZATION_CALLS_ONLY "
        "legacy_integer_0_effective_hardware_id=6 "
        "reserved_epilogue_effective_hardware_id=1")
    return verdict


def self_test() -> None:
    def logs(bad: bool) -> tuple[str, str]:
        direct: list[str] = []
        probe: list[str] = [
            "FQ_SHARD split_workspace_probe=1",
            "FQ_WORKSPACE_ORACLE exact=1 S1=0x1 S2=0x2 S4=0x4 S8=0x8",
        ]
        for symbol in sorted(EXPECTED):
            for split in (1, 2, 4, 8):
                is_bad = bad and symbol == AP0 and split == 4
                state = ("SPLIT_PARTITION" if split == 8 else
                         "RAW_FP16_MISMATCH" if is_bad else "MEASURED")
                direct.append(
                    f"FQ_TC_CELL symbol={symbol} S={split} state={state} "
                    f"raw_bad={32 if is_bad else 0} "
                    f"first_bad={32 if is_bad else 18446744073709551615}")
                probe_state = ("SPLIT_PARTITION" if split == 8 else
                               "WORKSPACE_PROBE_COMPLETE" if split in (2, 4)
                               else "MEASURED")
                probe.append(
                    f"FQ_TC_CELL symbol={symbol} S={split} "
                    f"state={probe_state} raw_bad=0 "
                    "first_bad=18446744073709551615")
            for split in (2, 4):
                for repeat in range(8):
                    sample_bad = (bad and symbol == AP0 and split == 4 and
                                  repeat == 0)
                    probe.append(
                        f"FQ_WORKSPACE_PROBE symbol={symbol} S={split} "
                        f"repeat={repeat} canary_words=0 "
                        f"partial_value_raw_bad={32 if sample_bad else 0} "
                        f"partial_bad_plane_mask=0x{1 if sample_bad else 0:x} "
                        f"partial_first_bad_plane={0 if sample_bad else 18446744073709551615} "
                        f"partial_first_bad_index={32 if sample_bad else 18446744073709551615} "
                        f"host_reduce_raw_bad={32 if sample_bad else 0} "
                        "sync_only_raw_bad=0 "
                        f"observed_reducer_raw_bad={32 if sample_bad else 0}")
        return "\n".join(direct), "\n".join(probe)

    bad = logs(True)
    clean = logs(False)
    with contextlib.redirect_stdout(io.StringIO()):
        assert adjudicate({
            "vendor-user0": bad,
            "clone-user0": bad,
            "reserved-id1": clean,
            "cta": clean,
        }) == "INTEGER_USER_BARRIER_ID6_CAUSAL"
        assert adjudicate({
            "vendor-user0": bad,
            "clone-user0": bad,
            "reserved-id1": bad,
            "cta": clean,
        }) == "PPU_NAMED_BARRIER_SHARED_ORDERING_CAUSAL"
        assert adjudicate({
            "vendor-user0": bad,
            "clone-user0": clean,
            "reserved-id1": clean,
            "cta": clean,
        }) == "UNADJUDICATED_EXACT_CLONE_DID_NOT_REPRODUCE"
    print("[fq-shared-sync-root:self-test] PASS ID6, named-barrier and "
          "clone-nonreproduction verdicts; exact denominator retained")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    for name in ARM_NAMES:
        key = name.replace("-", "_")
        parser.add_argument(f"--{name}-direct", dest=f"{key}_direct",
                            type=pathlib.Path)
        parser.add_argument(f"--{name}-probe", dest=f"{key}_probe",
                            type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        logs: dict[str, tuple[str, str]] = {}
        for name in ARM_NAMES:
            key = name.replace("-", "_")
            direct = getattr(args, f"{key}_direct")
            probe = getattr(args, f"{key}_probe")
            if direct is None or probe is None:
                parser.error(f"{name} direct/probe logs are required")
            logs[name] = (direct.read_text(), probe.read_text())
        verdict = adjudicate(logs)
        return 0 if verdict in {
            "INTEGER_USER_BARRIER_ID6_CAUSAL",
            "PPU_NAMED_BARRIER_SHARED_ORDERING_CAUSAL",
        } else 2
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-shared-sync-root] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

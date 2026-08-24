#!/usr/bin/env python3
"""Adjudicate the data seam inside the legacy Split-K shared round trip."""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import sys

from check_fq_split_shared_sync_root import summarize
from check_fq_split_workspace_probe import AP0, EXPECTED


ARM_NAMES = (
    "cta-baseline",
    "discard-roundtrip",
    "pre-r2s-cta",
    "disjoint-storage",
    "identity-convert",
    "scalar-r2s",
    "scalar-s2r",
    "scalar-both",
)


def adjudicate(logs: dict[str, tuple[str, str]]) -> str:
    arms = {name: summarize(name, *logs[name]) for name in ARM_NAMES}
    baseline = arms["cta-baseline"]
    discard = arms["discard-roundtrip"]
    pre = bool(arms["pre-r2s-cta"]["clean"])
    disjoint = bool(arms["disjoint-storage"]["clean"])
    identity = bool(arms["identity-convert"]["clean"])
    r2s = bool(arms["scalar-r2s"]["clean"])
    s2r = bool(arms["scalar-s2r"]["clean"])
    both = bool(arms["scalar-both"]["clean"])

    if not baseline["partial_bad"]:
        verdict = "UNADJUDICATED_CTA_BASELINE_DID_NOT_REPRODUCE"
    elif not discard["clean"]:
        verdict = "ACCUMULATOR_OR_COMPILER_FOOTPRINT_REMAINS"
    elif pre and disjoint and not any((identity, r2s, s2r, both)):
        verdict = "SHARED_STORAGE_REUSE_BEFORE_QUIESCENCE_CAUSAL"
    elif disjoint and not any((pre, identity, r2s, s2r, both)):
        verdict = "SHARED_STORAGE_ALIAS_LIFETIME_CAUSAL"
    elif identity and not any((pre, disjoint, r2s, s2r, both)):
        verdict = "ACCONVERT_FLOAT1_CAUSAL"
    elif r2s and both and not any((pre, disjoint, identity, s2r)):
        verdict = "AUTOVECTORIZED_R2S_CAUSAL"
    elif s2r and both and not any((pre, disjoint, identity, r2s)):
        verdict = "AUTOVECTORIZED_S2R_CAUSAL"
    elif both and not any((pre, disjoint, identity, r2s, s2r)):
        verdict = "AUTOVECTORIZED_R2S_S2R_INTERACTION_CAUSAL"
    elif not any((pre, disjoint, identity, r2s, s2r, both)):
        verdict = "UNADJUDICATED_SHARED_DATA_PATH_REMAINS"
    else:
        verdict = "UNADJUDICATED_MULTIPLE_CADENCE_SENSITIVE_ARMS"

    print(
        "FQ_SHARED_HANDOFF_ROOT "
        f"verdict={verdict} baseline_bad={int(bool(baseline['partial_bad']))} "
        f"discard_roundtrip_clean={int(bool(discard['clean']))} "
        f"pre_r2s_clean={int(pre)} disjoint_storage_clean={int(disjoint)} "
        f"identity_convert_clean={int(identity)} scalar_r2s_clean={int(r2s)} "
        f"scalar_s2r_clean={int(s2r)} scalar_both_clean={int(both)} "
        "static_layout=L223-EXACT-BIJECTION sync_policy=CTA-ALL-ARMS")
    return verdict


def synthetic_logs(bad: bool) -> tuple[str, str]:
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


def self_test() -> None:
    bad = synthetic_logs(True)
    clean = synthetic_logs(False)

    def case(*clean_names: str) -> dict[str, tuple[str, str]]:
        names = set(clean_names)
        return {name: clean if name in names else bad for name in ARM_NAMES}

    expected = (
        (("discard-roundtrip", "pre-r2s-cta", "disjoint-storage"),
         "SHARED_STORAGE_REUSE_BEFORE_QUIESCENCE_CAUSAL"),
        (("discard-roundtrip", "identity-convert"),
         "ACCONVERT_FLOAT1_CAUSAL"),
        (("discard-roundtrip", "scalar-r2s", "scalar-both"),
         "AUTOVECTORIZED_R2S_CAUSAL"),
        (("discard-roundtrip", "scalar-s2r", "scalar-both"),
         "AUTOVECTORIZED_S2R_CAUSAL"),
        (("discard-roundtrip", "scalar-both"),
         "AUTOVECTORIZED_R2S_S2R_INTERACTION_CAUSAL"),
        (("discard-roundtrip",),
         "UNADJUDICATED_SHARED_DATA_PATH_REMAINS"),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        for clean_names, verdict in expected:
            assert adjudicate(case(*clean_names)) == verdict
        assert adjudicate(case()) == "ACCUMULATOR_OR_COMPILER_FOOTPRINT_REMAINS"
    print("[fq-shared-handoff-root:self-test] PASS lifetime, converter, R2S, "
          "S2R, interaction, residual and accumulator-footprint verdicts")


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
        return 0 if verdict.endswith("_CAUSAL") else 2
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-shared-handoff-root] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

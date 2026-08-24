#!/usr/bin/env python3
"""Adjudicate the first corrupting prefix of the legacy shared epilogue."""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import sys

from check_fq_split_shared_handoff_root import synthetic_logs
from check_fq_split_shared_sync_root import summarize


ARM_NAMES = (
    "production-direct",
    "accumulator-opaque",
    "clone-opaque",
    "cta-only",
    "flat-constant-disjoint",
    "flat-accumulator-disjoint",
    "r2s-vector-disjoint",
    "r2s-scalar-disjoint",
    "r2s-snapshot-disjoint",
    "r2s-s2r-vector-disjoint",
    "r2s-s2r-scalar-disjoint",
    "full-discard",
)

CAUSAL = {
    "ACCUMULATOR_OPAQUE_LIVENESS_CAUSAL",
    "ADDITIONAL_REGISTER_FRAGMENT_FOOTPRINT_CAUSAL",
    "POST_MAINLOOP_CTA_SYNC_CAUSAL",
    "SHARED_STORE_BACKEND_OR_FOOTPRINT_CAUSAL",
    "LIVE_ACCUMULATOR_SHARED_STORE_DEPENDENCY_CAUSAL",
    "AUTOVECTORIZED_R2S_CAUSAL",
    "CUTE_R2S_LIVE_SOURCE_VIEW_CAUSAL",
    "R2S_SNAPSHOT_INTERACTION_CAUSAL",
    "AUTOVECTORIZED_S2R_CAUSAL",
}


def adjudicate(logs: dict[str, tuple[str, str]]) -> str:
    arms = {name: summarize(name, *logs[name]) for name in ARM_NAMES}
    clean = {name: bool(value["clean"]) for name, value in arms.items()}

    if not clean["production-direct"]:
        verdict = "UNADJUDICATED_PRODUCTION_CONTROL_FAILED"
    elif clean["full-discard"]:
        verdict = "UNADJUDICATED_FULL_SHARED_NEGATIVE_DID_NOT_REPRODUCE"
    elif not clean["accumulator-opaque"]:
        verdict = "ACCUMULATOR_OPAQUE_LIVENESS_CAUSAL"
    elif not clean["clone-opaque"]:
        verdict = "ADDITIONAL_REGISTER_FRAGMENT_FOOTPRINT_CAUSAL"
    elif not clean["cta-only"]:
        verdict = "POST_MAINLOOP_CTA_SYNC_CAUSAL"
    elif not clean["flat-constant-disjoint"]:
        verdict = "SHARED_STORE_BACKEND_OR_FOOTPRINT_CAUSAL"
    elif not clean["flat-accumulator-disjoint"]:
        verdict = "LIVE_ACCUMULATOR_SHARED_STORE_DEPENDENCY_CAUSAL"
    elif (not clean["r2s-vector-disjoint"] and
          clean["r2s-scalar-disjoint"]):
        verdict = "AUTOVECTORIZED_R2S_CAUSAL"
    elif (not clean["r2s-scalar-disjoint"] and
          clean["r2s-snapshot-disjoint"]):
        verdict = "CUTE_R2S_LIVE_SOURCE_VIEW_CAUSAL"
    elif (not clean["r2s-scalar-disjoint"] and
          not clean["r2s-snapshot-disjoint"]):
        verdict = "R2S_SCATTER_OR_REGISTER_FOOTPRINT_REMAINS"
    elif not clean["r2s-snapshot-disjoint"]:
        verdict = "R2S_SNAPSHOT_INTERACTION_CAUSAL"
    elif (not clean["r2s-s2r-vector-disjoint"] and
          clean["r2s-s2r-scalar-disjoint"]):
        verdict = "AUTOVECTORIZED_S2R_CAUSAL"
    elif not clean["r2s-s2r-scalar-disjoint"]:
        verdict = "S2R_READBACK_OR_REGISTER_FOOTPRINT_REMAINS"
    elif not clean["r2s-s2r-vector-disjoint"]:
        verdict = "UNADJUDICATED_VECTOR_S2R_WITHOUT_SCALAR_CONTROL"
    else:
        verdict = "FULL_EPILOGUE_COMPOSITION_OR_COMPILER_FOOTPRINT_REMAINS"

    flags = " ".join(
        f"{name.replace('-', '_')}_clean={int(clean[name])}"
        for name in ARM_NAMES
    )
    print(
        "FQ_SHARED_PREFIX_ROOT "
        f"verdict={verdict} {flags} "
        "static_layout=L223-EXACT-BIJECTION "
        "storage=DISJOINT-EXCEPT-FULL-NEGATIVE "
        "publication=PRODUCTION-DIRECT-AFTER-PREFIX")
    return verdict


def self_test() -> None:
    clean_log = synthetic_logs(False)
    bad_log = synthetic_logs(True)

    def case(*bad_names: str) -> dict[str, tuple[str, str]]:
        bad = {"full-discard", *bad_names}
        return {name: bad_log if name in bad else clean_log
                for name in ARM_NAMES}

    expected = (
        (("accumulator-opaque",),
         "ACCUMULATOR_OPAQUE_LIVENESS_CAUSAL"),
        (("clone-opaque",),
         "ADDITIONAL_REGISTER_FRAGMENT_FOOTPRINT_CAUSAL"),
        (("cta-only",), "POST_MAINLOOP_CTA_SYNC_CAUSAL"),
        (("flat-constant-disjoint",),
         "SHARED_STORE_BACKEND_OR_FOOTPRINT_CAUSAL"),
        (("flat-accumulator-disjoint",),
         "LIVE_ACCUMULATOR_SHARED_STORE_DEPENDENCY_CAUSAL"),
        (("r2s-vector-disjoint",), "AUTOVECTORIZED_R2S_CAUSAL"),
        (("r2s-scalar-disjoint",),
         "CUTE_R2S_LIVE_SOURCE_VIEW_CAUSAL"),
        (("r2s-scalar-disjoint", "r2s-snapshot-disjoint"),
         "R2S_SCATTER_OR_REGISTER_FOOTPRINT_REMAINS"),
        (("r2s-snapshot-disjoint",),
         "R2S_SNAPSHOT_INTERACTION_CAUSAL"),
        (("r2s-s2r-vector-disjoint",),
         "AUTOVECTORIZED_S2R_CAUSAL"),
        (("r2s-s2r-vector-disjoint", "r2s-s2r-scalar-disjoint"),
         "S2R_READBACK_OR_REGISTER_FOOTPRINT_REMAINS"),
        ((), "FULL_EPILOGUE_COMPOSITION_OR_COMPILER_FOOTPRINT_REMAINS"),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        for bad_names, verdict in expected:
            assert adjudicate(case(*bad_names)) == verdict
        control_bad = case("production-direct")
        assert (adjudicate(control_bad) ==
                "UNADJUDICATED_PRODUCTION_CONTROL_FAILED")
        no_negative = {name: clean_log for name in ARM_NAMES}
        assert (adjudicate(no_negative) ==
                "UNADJUDICATED_FULL_SHARED_NEGATIVE_DID_NOT_REPRODUCE")
    print("[fq-shared-prefix-root:self-test] PASS ten causal/residual "
          "prefix verdicts plus production/full-negative controls")


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
        return 0 if verdict in CAUSAL else 2
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-shared-prefix-root] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed source contract for the AP0/AP1 producer-workspace oracle."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
MAIN = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
RUNNER = ROOT / "tools/run_fq_q4k_split_workspace_probe_box.sh"
CHECKER = ROOT / "tools/check_fq_split_workspace_probe.py"
SELECTOR = ROOT / "tools/select_fq_split_timing_closure.py"


class CheckError(ValueError):
    pass


def require(label: str, text: str, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} contract missing: {missing}")


def check(bench: str, main: str, runner: str, checker: str,
          selector: str) -> None:
    require("bench", bench, (
        "bool split_workspace_probe = false;",
        "hggcMemset(in.workspace, 0xa5, plan.partial_bytes)",
        "producer_launch()",
        "hggcDeviceSynchronize()",
        "hggcMemcpy(host_partials.data(), partials, plan.partial_bytes,",
        "inspect_host_partials(in, splits, host_partials, sample)",
        "expected_partials[offset]",
        "sample.partial_value_raw_bad",
        "sample.partial_bad_plane_mask",
        'result.failure_step = "WORKSPACE_PROBE_PARTIAL_ORACLE";',
        "sample.sync_only_raw_bad = result.raw_bad;",
        "sample.observed_reducer_raw_bad = result.raw_bad;",
        'result.failure_step = "WORKSPACE_PROBE_COMPLETE";',
    ))
    first = bench.find("// Arm 1 inserts only a device-wide completion boundary")
    second = bench.find("// Arm 2 copies the exact producer bytes to the host")
    if first < 0 or second <= first:
        raise CheckError("sync-only and host-observed arm order changed")
    require("main", main, (
        '"--split-workspace-probe"',
        '"FQ_WORKSPACE_PROBE q=%d A=%d shape=%dx%dx%d symbol=%s "',
        '"FQ_WORKSPACE_ORACLE exact=1 S1=0x%016llx S2=0x%016llx "',
        "make_fixture(shape, cli.split_workspace_probe)",
        "if (build_partial_golden)",
        "f.partial_golden[slot][offset] += float(contribution);",
        '"partial_value_raw_bad=%llu partial_bad_plane_mask=0x%x "',
        '"bc_batch=native-grid-y-m-lt8 split_workspace_probe=%d "',
        '"iterations=%d correctness_repeats=%d\\n"',
    ))
    require("runner", runner, (
        "--split-workspace-probe",
        "direct.log",
        "workspace-probe.log",
        "check_fq_split_workspace_probe.py",
    ))
    require("checker", checker, (
        'verdict = "PRODUCER_PARTIAL_VALUE_BAD"',
        'verdict = "SAME_STREAM_PUBLICATION_GAP"',
        'verdict = "D2H_VISIBILITY_BRIDGE_REQUIRED"',
        'verdict = "REDUCER_LOAD_OR_INDEX_BAD"',
        '"FQ_SPLIT_WORKSPACE_SLICE "',
        '"FQ_SPLIT_WORKSPACE_PROVIDER "',
        '"MIXED_PROVIDER_VERDICTS"',
        '"exact partial-workspace oracle marker differs"',
        "partial mismatch lost plane identity",
        "direct AP0 S2/S4 failure denominator was not reproduced",
    ))
    require("selector", selector, (
        "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
        "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
    ))


def main() -> int:
    paths = (BENCH, MAIN, RUNNER, CHECKER, SELECTOR)
    texts = [path.read_text() for path in paths]
    check(*texts)
    plants = (
        (0, "sample.sync_only_raw_bad = result.raw_bad;",
         "sample.sync_only_raw_bad = 0;"),
        (0, "inspect_host_partials(in, splits, host_partials, sample)",
         "/* lost host oracle */"),
        (0, "expected_partials[offset]", "expected_partials[0]"),
        (1, "f.partial_golden[slot][offset] += float(contribution);",
         "/* lost independent slice golden */"),
        (1, "make_fixture(shape, cli.split_workspace_probe)",
         "make_fixture(shape, true)"),
        (1, '"--split-workspace-probe"', '"--lost-workspace-probe"'),
        (2, "workspace-probe.log", "workspace-lost.log"),
        (3, 'verdict = "SAME_STREAM_PUBLICATION_GAP"',
         'verdict = "UNKNOWN"'),
        (4, "_ap1\"", "_ap2\""),
    )
    for index, old, new in plants:
        changed = list(texts)
        if old not in changed[index]:
            raise CheckError(f"negative seam missing: {old}")
        changed[index] = changed[index].replace(old, new, 1)
        try:
            check(*changed)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative stayed green: {old}")
    print("[fq-split-workspace-probe:self-test] PASS: independent exact "
          "per-split golden, producer bytes, sync-only/host-observed reducer "
          "arms, diagnostic-only allocation, exact AP0/AP1 denominator, "
          "and nine negatives")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, OSError) as error:
        print(f"[fq-split-workspace-probe:self-test] FAIL: {error}")
        raise SystemExit(2)

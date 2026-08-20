#!/usr/bin/env python3
"""Local contract for the conservative Q4_K pruning pilot."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNNER = TOOLS / "run_scalefirst_q4k_pruned_box.sh"
REAL_RUNNER = TOOLS / "run_scalefirst_q4k_real_shapes_pruned_box.sh"
PRUNER = TOOLS / "prune_scalefirst_q4k_pilot.py"
PLANNER = TOOLS / "plan_scalefirst_q4k_real_shapes.py"
POLICY = ROOT / "benchmarks/scalefirst_q4k_pruned_policy.json"
REAL_POLICY = ROOT / "benchmarks/scalefirst_q4k_real_shapes_pruned_policy.json"
BENCH = ROOT / "benchmarks/scalefirst_internal_sweep_bench.hpp"
DRIVER = ROOT / "benchmarks/test_scalefirst_internal_sweep.cu"
EXHAUSTIVE = TOOLS / "run_scalefirst_internal_sweep_box.sh"
L210 = ROOT / "dev/fold_derivation/run_l210_q4_a32_consumer_layout.sh"
L210_EXPECTED = ROOT / "dev/fold_derivation/l210_q4_a32_consumer_layout.expected.txt"


L210_ROW = re.compile(
    r"^\[l210 row\] name=(\S+) tm=(\d+) tn=(\d+) tk=(\d+) wm=(\d+) wn=(\d+) "
    r"rule=([01]) compatible=([01]) byte_bad=(\d+)/(\d+) code_bad=(\d+)/(\d+)$")
L210_SUMMARY = (
    "[l210] PASS rows=10 positives=4 negatives=6 "
    "exact_device_byte_bad=32768/32768 "
    "exact_device_code_bad=65536/65536 one_bit_corruption_bad=1"
)


def l210_canonical_lines(output: str) -> list[str]:
    return [line for line in output.splitlines()
            if line.startswith("[l210 row] ") or line.startswith("[l210] ")]


def validate_l210(lines: list[str]) -> str | None:
    if len(lines) != 11 or lines[-1] != L210_SUMMARY:
        return "expected exact ten-row census plus PASS summary"
    expected_names = {
        "canonical-a", "canonical-large-t", "scaled-a", "scaled-large-t",
        "exact-device-failure", "physical-n-too-small", "too-many-n-warps",
        "too-few-n-warps", "second-reader-instance", "larger-tile-same-wn",
    }
    seen = set()
    positives = negatives = 0
    for line in lines[:-1]:
        match = L210_ROW.fullmatch(line)
        if match is None:
            return f"malformed row: {line}"
        name = match.group(1)
        if name in seen or name not in expected_names:
            return f"unexpected or duplicate row {name}"
        seen.add(name)
        rule, compatible = map(int, match.group(7, 8))
        byte_bad, byte_total, code_bad, code_total = map(
            int, match.group(9, 10, 11, 12))
        if byte_total != 32768 or code_total != 65536 or rule != compatible:
            return f"denominator/rule closure differs for {name}"
        if compatible:
            positives += 1
            if byte_bad or code_bad:
                return f"positive reader {name} is not byte/code exact"
        else:
            negatives += 1
            if byte_bad == 0 or code_bad == 0:
                return f"negative reader {name} did not turn red"
        if name == "exact-device-failure" and \
                (byte_bad != byte_total or code_bad != code_total):
            return "exact device failure is no longer all-byte/all-code red"
    if seen != expected_names or positives != 4 or negatives != 6:
        return "reader-class denominator differs"
    return None


def require(text: str, tokens: tuple[str, ...], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{label} lost contract tokens: {missing}")


def check_q8_artifact_contract(driver: str) -> None:
    match = re.search(
        r'static_assert\((SCALEFIRST_SWEEP_QTYPE\s*!=\s*8\s*\|\|\s*'
        r'SCALEFIRST_SWEEP_ARTIFACT_TK\s*==\s*32)\s*,\s*'
        r'"Q8 has one canonical A32 artifact"\s*\);', driver)
    if match is None:
        raise AssertionError("driver lost the exact Q8 artifact implication")
    expression = match.group(1)

    def compile_case(qtype: int, artifact: int) -> subprocess.CompletedProcess[str]:
        source = (f"#define SCALEFIRST_SWEEP_QTYPE {qtype}\n"
                  f"#define SCALEFIRST_SWEEP_ARTIFACT_TK {artifact}\n"
                  f"static_assert({expression}, "
                  '"Q8 has one canonical A32 artifact");\n')
        return subprocess.run(
            ["c++", "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
            input=source, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, cwd=ROOT)

    for qtype, artifact in ((12, 64), (8, 32)):
        result = compile_case(qtype, artifact)
        if result.returncode != 0:
            raise AssertionError(
                f"valid qtype/artifact {qtype}/A{artifact} failed:\n"
                + result.stdout)
    planted = compile_case(8, 64)
    if planted.returncode == 0 or "Q8 has one canonical A32 artifact" not in \
            planted.stdout:
        raise AssertionError("Q8/A64 negative did not fail with the bound reason")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--committed-only", action="store_true")
    parser.add_argument("--evidence", type=pathlib.Path, default=L210_EXPECTED)
    args = parser.parse_args()
    subprocess.run(["bash", "-n", str(RUNNER)], cwd=ROOT, check=True)
    subprocess.run(["bash", "-n", str(REAL_RUNNER)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-B", str(PRUNER), "self-test"],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-B", str(PLANNER), "self-test"],
                   cwd=ROOT, check=True)
    expected_lines = args.evidence.read_text().splitlines()
    why = validate_l210(expected_lines)
    if why:
        raise AssertionError(f"committed L210 evidence is malformed: {why}")
    planted_l210 = list(expected_lines)
    planted_l210[4] = planted_l210[4].replace(
        "code_bad=65536/65536", "code_bad=65535/65536")
    if validate_l210(planted_l210) is None:
        raise AssertionError("L210 one-code denominator defect stayed green")
    if not args.committed_only:
        proc = subprocess.run(
            ["bash", str(L210)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout, end="")
        if proc.returncode:
            raise AssertionError(f"L210 local oracle returned rc={proc.returncode}")
        actual_lines = l210_canonical_lines(proc.stdout)
        why = validate_l210(actual_lines)
        if why:
            raise AssertionError(f"fresh L210 evidence is malformed: {why}")
        if actual_lines != expected_lines:
            raise AssertionError("fresh L210 evidence differs from result SHA")
    sys.path.insert(0, str(TOOLS))
    import prune_scalefirst_q4k_pilot as pruner
    import plan_scalefirst_q4k_real_shapes as planner
    import scalefirst_internal_matrix as matrix

    policy = pruner.load_policy(POLICY)
    fmt = matrix.format_for(12)
    typed = [row for row in matrix.emitted_tactics(12, 64)
             if row.bchunk == 0 and
             matrix.classify(fmt, 64, row)[0] == "TYPE_ADMISSION_REQUIRED"]
    if len(typed) != 1824:
        raise AssertionError(f"Q4_K/A64/bc0 denominator drifted {len(typed)}/1824")
    anchor_axes = (64, 64, 64, 64, 32, 3, 0)
    anchors = [row for row in typed if
               (row.tile_m, row.tile_n, row.tactic_tile_k, row.warp_m,
                row.warp_n, row.stages, row.bchunk) == anchor_axes]
    if len(anchors) != 1:
        raise AssertionError("historical Q4_K anchor is not typed exactly once")

    runner, real_runner, bench, driver, exhaustive = (
        path.read_text() for path in
        (RUNNER, REAL_RUNNER, BENCH, DRIVER, EXHAUSTIVE))
    check_q8_artifact_contract(driver)
    if any(token in runner for token in ("/tmp", "mktemp", "rm -")):
        raise AssertionError("pilot runner reintroduced temporary/destructive paths")
    require(runner, (
        'OUT must be a strict /workspace child',
        '--qtype 12 --artifact-tk 64 --bchunk 0',
        'typed_rows"] != 1824',
        '--algorithm=nonpersistent',
        '--symbol-file="$out/results/screen-shortlist.txt"',
        '--symbol-file="$out/results/confirm-shortlist.txt"',
        'screen-shortlist.txt', 'confirm-shortlist.txt',
        'source-authority.json', 'bundle.json',
        'phase=screen', 'phase=scheduler', 'phase=confirm',
    ), "runner")
    require(bench, (
        'kAllAlgorithms = kNonPersistent | kPersistent | kSplitK',
        'unsigned algorithm_mask = kAllAlgorithms',
        'options.includes(Options::kNonPersistent)',
        'options.includes(Options::kPersistent)',
        'options.includes(Options::kSplitK)',
    ), "benchmark")
    require(driver, (
        '--algorithm=', '--symbol-file=',
        'symbol file contains a duplicate',
        'symbol file names an unknown generated symbol',
        'selected_rows=%zu algorithm_mask=0x%x',
        'SCALEFIRST_SWEEP_QTYPE != 8 ||',
        'SCALEFIRST_SWEEP_ARTIFACT_TK == 32',
        '#if SCALEFIRST_SWEEP_QTYPE == 8',
    ), "driver")
    master = planner.load_master(REAL_POLICY)
    if master["artifact_tile_k"] != [32, 64, 128, 256] or \
            master["prefill_m"] != [64, 2048, 4096]:
        raise AssertionError("real-shape format/layout/workload denominator changed")
    dynamic = planner.cell_policy(master, {
        "artifact_tile_k": 32, "shape": [64, 4096, 4096],
        "cell_key": "a32/m64_n4096_k4096_g32"})
    pruner.validate_policy(dynamic)
    planted_layout = json.loads(json.dumps(dynamic))
    planted_layout["layout"]["fold_n"]["low"] = 1
    try:
        pruner.validate_policy(planted_layout)
    except pruner.ContractError:
        pass
    else:
        raise AssertionError("wrong FoldN in shape policy stayed green")
    planted_reader = json.loads(json.dumps(dynamic))
    planted_reader["layout"]["reader_contract"] = "FOLD_ONLY"
    try:
        pruner.validate_policy(planted_reader)
    except pruner.ContractError:
        pass
    else:
        raise AssertionError("wrong A32 reader contract stayed green")
    expected_by_artifact = {32: 490, 64: 1824, 128: 1036, 256: 401}
    for artifact, expected in expected_by_artifact.items():
        rows = [row for row in matrix.emitted_tactics(12, artifact)
                if row.bchunk == 0 and
                matrix.classify(fmt, artifact, row)[0] ==
                "TYPE_ADMISSION_REQUIRED"]
        if len(rows) != expected:
            raise AssertionError(
                f"Q4_K/A{artifact} typed denominator drifted "
                f"{len(rows)}/{expected}")
    if any(token in real_runner for token in
           ("/tmp", "mktemp", "probe_box_identity", "rm -")):
        raise AssertionError(
            "real-shape runner reintroduced temp/probe/destructive seams")
    require(real_runner, (
        'INTERNAL_SWEEP_SPEC must name the COMPLETE inventory-v2 JSON',
        'for artifact in 32 64 128 256',
        '32) expected=490', '64) expected=1824',
        '128) expected=1036', '256) expected=401',
        '--algorithm=nonpersistent', '--algorithm=all',
        '--symbol-file="$result_dir/screen-shortlist.txt"',
        '--symbol-file="$result_dir/confirm-shortlist.txt"',
        'phase=screen full-typed-graph', 'phase=scheduler shortlist=',
        'phase=confirm shortlist=', '--models-root "$out/models"',
        'l210_q4_a32_consumer_layout.expected.txt',
        '--committed-only --evidence "$l210_evidence"',
        'Never paper over this with a fake fp8 SDK header',
        'source-authority.json', 'binary-hashes.json', 'commit.json',
        'bundle.json', 'resume bundle lost plan.json',
        'incomplete uncommitted evidence',
    ), "real-shape runner")
    require(bench, (
        'KTileDoesNotDivide', 'INADMISSIBLE_K_TILE_DOES_NOT_DIVIDE',
        'if (in.k % TK)', 'add_shape_terminals(State::KTileDoesNotDivide)',
    ), "shape-specific TileK terminal")
    require(PLANNER.read_text(), (
        'expected = {(artifact, key) for artifact in ARTIFACTS for key in keys}',
        'drop-one shape/layout negative stayed green',
        'models_root / model_id / key / "summary.json"',
        '"reader_contract": winner["layout"]["reader_contract"]',
        'DECODE_NOT_SCALEFIRST_PREFILL', 'OUTSIDE_REGISTERED_PREFILL_M',
    ), "real-shape planner")
    # The production exhaustive runner must not opt into either pilot filter.
    if '--algorithm=' in exhaustive or '--symbol-file=' in exhaustive:
        raise AssertionError("exhaustive runner was silently converted to pruning")

    planted = json.loads(POLICY.read_text())
    planted["anchor_symbol"] += "_missing"
    scratch = dict(policy)
    scratch["anchor_symbol"] = planted["anchor_symbol"]
    try:
        # Exercise the same invariant without writing a second policy source.
        if scratch["anchor_symbol"] != \
                "sf_q12_a64_tm64_tn64_tk64_wm64_wn32_s3_bc0":
            raise pruner.ContractError("historical anchor symbol changed")
    except pruner.ContractError:
        pass
    else:
        raise AssertionError("mutated historical anchor stayed green")
    evidence_mode = "committed" if args.committed_only else "fresh-local"
    print("[q4k-prune-runner] PASS pilot anchor exact; real Q4_K "
          "A32/A64/A128/A256 denominators bound; shape-specific TileK "
          "terminal, model folders, and three-phase authority present; "
          f"L210={evidence_mode}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"[q4k-prune-runner] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)

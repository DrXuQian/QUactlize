#!/usr/bin/env python3
"""Local contract for the conservative Q4_K pruning pilot."""

from __future__ import annotations

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
    subprocess.run(["bash", "-n", str(RUNNER)], cwd=ROOT, check=True)
    subprocess.run(["bash", "-n", str(REAL_RUNNER)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-B", str(PRUNER), "self-test"],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-B", str(PLANNER), "self-test"],
                   cwd=ROOT, check=True)
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
    expected_by_artifact = {32: 2340, 64: 1824, 128: 1036, 256: 401}
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
        '32) expected=2340', '64) expected=1824',
        '128) expected=1036', '256) expected=401',
        '--algorithm=nonpersistent', '--algorithm=all',
        '--symbol-file="$result_dir/screen-shortlist.txt"',
        '--symbol-file="$result_dir/confirm-shortlist.txt"',
        'phase=screen full-typed-graph', 'phase=scheduler shortlist=',
        'phase=confirm shortlist=', '--models-root "$out/models"',
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
    print("[q4k-prune-runner] PASS pilot anchor exact; real Q4_K "
          "A32/A64/A128/A256 denominators bound; shape-specific TileK "
          "terminal, model folders, and three-phase authority present")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"[q4k-prune-runner] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)

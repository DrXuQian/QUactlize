#!/usr/bin/env python3
"""Fail-closed source contract for the Q4_K K-pack4/xplane isomorphic A/B."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "tools/select_fq_q4k_kpack4_xplane_isomorphic_ab.py"
ANALYZER = ROOT / "tools/analyze_fq_q4k_kpack4_xplane_isomorphic_ab.py"
RUNNER = ROOT / "tools/run_fq_q4k_kpack4_xplane_isomorphic_ab_box.sh"
BENCH = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
DRIVER = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
POLICY = ROOT / "quactlize/include/ppu_mixed_policy.hpp"


class CheckError(ValueError):
    pass


def require(text: str, needles: tuple[str, ...], label: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise CheckError(f"{label} lost source seams: {missing}")


def check(selector: str, analyzer: str, runner: str, bench: str,
          driver: str, policy: str) -> None:
    require(selector, (
        '("xplane-ap0", "xplane", 64, 0)',
        '("kpack4-ap0", "q4-kpack4", 0, 0)',
        '("xplane-ap1", "xplane", 64, 1)',
        '("kpack4-ap1", "q4-kpack4", 0, 1)',
        '"tile_m": 8', '"tile_n": 64', '"tactic_tile_k": 256',
        '"warp_m": 8', '"warp_n": 16', '"stages": 2',
        '"typed_rows": 144', '"source_typed_rows": 918',
        '"same_tactic": True', '"same_split": True',
        '"ap0_isolates_weight_layout": True',
        '"ap1_isolates_weight_layout": True',
    ), "selector")
    require(analyzer, (
        'CONFIG = "8x64x256_w8x16_s2"', 'SPLIT = 4',
        '"m1_n5120_k8192"', '"m1_n5120_k25600"',
        '"m1_n8192_k5120"', '"m2_n5120_k25600"',
        '"m4_n5120_k8192"',
        '"GemmUniversalMixedInputSplitKParallel"',
        '"LastArriverM1Fp16Completion" not in pair[1]',
        '"m8n8.x4.swzl.shared.b16"',
        '"m16n16.x1.swzl.trans.shared.b16"',
        '"KernelAiuQ4KPack4Transpose" in demangled',
        'has_ap1 = "KernelAiuPackedA<" in demangled',
        'if has_kpack != expects_kpack',
        '"a_provider_identity": "GENERATED_UNIT_BUILD_ABI_AND_RUNTIME_CELL"',
        '"provider": str(arm["a_provider"])',
        'if focus["tsm_load"] <= 0',
        'reader_lowering = "HGOBJDUMP_TSM_LOWERED"',
        'if len(mma_counts) != 1',
        'same_sign = all(value > 0 for value in paired)',
        'requires_acu = abs(delta) >= gap_threshold and same_sign',
        '"instruction_delta"', 'xplane_registers\\t',
        'xplane_spill\\t', 'xplane_ldmatrix\\t',
    ), "analyzer")
    require(runner, (
        'iterations="${PERF_ITERATIONS:-101}"',
        'rounds="${PERF_ROUNDS:-2}"',
        'gap_threshold="${ACU_GAP_THRESHOLD:-0.03}"',
        'run_acu="${RUN_ACU:-auto}"',
        'for arm in xplane-ap0 kpack4-ap0 xplane-ap1 kpack4-ap1',
        'FQ_SWEEP_WEIGHT_LAYOUT="$layout"',
        '"$hgobjdump" -lelf "$binary"',
        '"$hgobjdump" -line "-func=$(cat "$symbol")"',
        '"$hgobjdump" "-res-usage=$(cat "$symbol")"',
        'order="xplane-ap$ap kpack4-ap$ap"',
        'order="kpack4-ap$ap xplane-ap$ap"',
        '--only-split=4 --tm8-max-m=8',
        '--profile-subject-only',
        "launches=1 reducer_launches=0",
        '"$acu" --import "$report" --csv --page details',
        'acu-targets.tsv',
        'RESUME must be 0 or 1',
        'resume changed a measurement source',
        'sha256sum -c "$out/results/binary-${arm}.sha256"',
        'validate-inputs',
        "assert value['a_provider_id']==ap and value['a_provider']==provider",
        'assert unit.read_text().count(macro)==1',
        'assert registry.read_text().count(macro)==1',
        '! grep -F "$(basename "$unit")" "$target_make"',
        'build-identity-${arm}.sha256',
    ), "runner")
    if runner.count('--only-split=4 --tm8-max-m=8') != 2:
        raise CheckError("timing and ACU must both bind the exact S4 subject")
    require(bench, (
        'bool profile_subject_only = false;',
        'if (options.profile_subject_only)',
        'if (producer_launch() != cutlass::Status::kSuccess ||',
        'result.state = State::ProfileSubject;',
    ), "benchmark")
    if bench.count("if (options.profile_subject_only)") != 2:
        raise CheckError("profile-only seam must cover shipping and Split-K exactly once")
    require(driver, (
        '"--profile-subject-only"',
        'rows.size() != 1 || cli.only_split == 0',
        'cli.bc_mode != Cli::BcMode::Skip',
        'FQ_PROFILE_SUBJECT symbol=%s shape=%dx%dx%d',
        'launches=1 reducer_launches=0',
    ), "driver")
    require(policy, (
        'int APackRows = 0',
        'cutlass::gemm::KernelAiuPackedA<APackRows, KPack4Schedule>',
        'using BProvider = KPack4TransposedBProvider;',
    ), "K-pack4 policy")


def main() -> int:
    paths = (SELECTOR, ANALYZER, RUNNER, BENCH, DRIVER, POLICY)
    texts = [path.read_text() for path in paths]
    check(*texts)
    plants = (
        (0, '("kpack4-ap1", "q4-kpack4", 0, 1)',
         '("kpack4-ap1", "q4-kpack4", 0, 0)'),
        (0, '"same_tactic": True', '"same_tactic": False'),
        (1, 'SPLIT = 4', 'SPLIT = 1'),
        (1, 'if len(mma_counts) != 1', 'if False'),
        (1, 'if has_kpack != expects_kpack',
         'if False'),
        (1, 'if focus["tsm_load"] <= 0', 'if False'),
        (1, '"provider": str(arm["a_provider"])',
         '"provider": "standard-aiu"'),
        (1, 'requires_acu = abs(delta) >= gap_threshold and same_sign',
         'requires_acu = False'),
        (2, '--only-split=4 --tm8-max-m=8',
         '--only-split=1 --tm8-max-m=8'),
        (2, 'launches=1 reducer_launches=0',
         'launches=2 reducer_launches=1'),
        (2, 'resume changed a measurement source',
         'resume accepted a measurement source'),
        (2, "assert value['a_provider_id']==ap and value['a_provider']==provider",
         "assert value['a_provider_id']==ap"),
        (2, '! grep -F "$(basename "$unit")" "$target_make"',
         ': # generated unit not bound to target'),
        (3, 'if (producer_launch() != cutlass::Status::kSuccess ||',
         'if (full_launch() != cutlass::Status::kSuccess ||'),
        (4, 'rows.size() != 1 || cli.only_split == 0',
         'rows.empty() || cli.only_split == 0'),
        (5, 'using BProvider = KPack4TransposedBProvider;',
         'using BProvider = AiuBProvider;'),
    )
    for index, old, new in plants:
        broken = list(texts)
        if old not in broken[index]:
            raise CheckError(f"negative seam is absent: {old}")
        broken[index] = broken[index].replace(old, new, 1)
        try:
            check(*broken)
        except CheckError:
            pass
        else:
            raise CheckError(f"negative stayed green: {old}")
    print("[fq-kpack4-xplane-ab:self-test] PASS exact AP0/AP1 factorial, "
          "same-config/S4 AB-BA timing, codegen resources and one-producer "
          "conditional ACU plus analysis-only binary reuse; sixteen plants RED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, CheckError, AssertionError) as exc:
        print(f"[fq-kpack4-xplane-ab:self-test] FAIL: {exc}")
        raise SystemExit(2)

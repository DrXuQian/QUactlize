#!/usr/bin/env python3
"""Fail-closed contract for the complete PPU dense fixed-Split-K sweep.

The performance winner is not a hand-written row.  The committed int4 dense
table is filtered by the exact production M==1 packed-A type domain and then
crossed with runtime S={1,2,4,8}.  This gate proves the denominator and the
source/build/runner seams without pretending to execute PPU device code.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "benchmarks/lowbit_dense_configs.inc"
BENCH = ROOT / "benchmarks/dense_splitk_parallel_bench.hpp"
MAIN = ROOT / "benchmarks/test_lowbit_dense_splitk_sweep.cu"
UNIT = ROOT / "benchmarks/dense_splitk_parallel_unit.inc"
HANDLE = ROOT / "quactlize/include/dense_splitk_parallel_ppu.cuh"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
RUNNER = ROOT / "tools/run_dense_splitk_sweep_box.sh"
EXACT_TN64 = ROOT / "benchmarks/dense_splitk_exact_warm_ab_tn64.cu"
EXACT_TN128 = ROOT / "benchmarks/dense_splitk_exact_warm_ab_tn128.cu"

ROW_RE = re.compile(
    r"^\s*X\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),B\)\s*\\?\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Counts:
    source: int
    packed: int
    cells: int
    measured: int
    inadmissible: int
    per_split: tuple[int, int, int, int]


def parse_rows(text: str) -> list[tuple[int, ...]]:
    return [tuple(map(int, match.groups())) for match in ROW_RE.finditer(text)]


def calculate(rows: list[tuple[int, ...]], splits: tuple[int, ...]) -> Counts:
    packed = [row for row in rows if row[0] == 8 and row[3] == 8 and row[6] == 0]
    per_split: list[int] = []
    for split in splits:
        runnable = 0
        for _, _, tk, _, _, stages, _ in packed:
            kt = 4096 // tk
            runnable += (
                4096 % tk == 0
                and kt % split == 0
                and (split == 1 or kt // split >= stages - 1)
            )
        per_split.append(runnable)
    measured = sum(per_split)
    cells = len(packed) * len(splits)
    return Counts(
        source=len(rows), packed=len(packed), cells=cells,
        measured=measured, inadmissible=cells - measured,
        per_split=tuple(per_split),
    )


def require(text: str, token: str, owner: str, bad: list[str]) -> None:
    if token not in text:
        bad.append(f"{owner}: missing {token!r}")


def forbid(text: str, token: str, owner: str, bad: list[str]) -> None:
    if token in text:
        bad.append(f"{owner}: forbidden {token!r}")


def audit(
    rows: list[tuple[int, ...]], splits: tuple[int, ...], files: dict[str, str]
) -> list[str]:
    bad: list[str] = []
    counts = calculate(rows, splits)
    expected = Counts(1772, 201, 804, 684, 120, (201, 195, 168, 120))
    if counts != expected:
        bad.append(f"denominator: got {counts}, expected {expected}")
    if len(set(rows)) != len(rows):
        bad.append("authority: duplicate committed tactic row")
    packed = [row for row in rows if row[0] == 8 and row[3] == 8 and row[6] == 0]
    if len(set(packed)) != len(packed):
        bad.append("authority: duplicate generated packed-A symbol")

    bench = files["bench"]
    for token in (
        "DensePackedAKernelTypes<",
        "QuantMode::FinegrainedScaleOnly",
        "kArtifactTileK = 64",
        "MainloopPolicy::PackedARows == 1",
        "AtomShape_MNK{}) == 8",
        "std::is_same_v<typename Split::CollectiveMainloop",
        "std::vector<double> e2e_samples_us",
        "result.e2e_samples_us = samples",
        "PreparedOnePlaneLauncher::run()",
        "a skipped launch",
        "result.fingerprint != stable_fingerprint",
        "run_exact_warm_ab(",
        "measure_warm_aggregate_for_diagnostics(",
        "packed_internal.run_producer_only_for_diagnostics(nullptr)",
        "cutlass::epilogue::EpilogueSimtVectorized>::CollectiveOp",
        "historical Cfg default scheduler must remain void",
        "result.shipping_ordinary_reshape_us",
    ):
        require(bench, token, "per-row shipping type", bad)
    ranking = bench[bench.find("bool run_row("):bench.find(
        "bool measure_warm_aggregate_for_diagnostics(")]
    forbid(ranking, "run_producer_only_for_diagnostics",
           "production cold ranking", bad)

    handle = files["handle"]
    for token in (
        "class PreparedOnePlaneLauncher",
        "bool initialize(",
        "cutlass::Status run(",
        "run_with_events(",
        "run_producer_only_for_diagnostics(",
        "S==1 still instantiates and runs ShippingTypes::Gemm verbatim",
        "prepared S==1 and S>1 handles must share the exact mainloop",
        "prepared fixed Split-K must retain the shipping dense tactic guard",
    ):
        require(handle, token, "prepared launch seam", bad)

    main = files["main"]
    for token in (
        "shipping_byte_diff=%zu/%zu roundtrip_bad=%zu/%zu",
        "[splitk device] ordinal=%d measured_cu=%d",
        "ORDER-INDEPENDENT+FP16-EXACT",
        "std::ceil(2.16 * double(cli.l2_bytes))",
        "std::size_t(128) << 20",
        "std::shuffle(tasks.begin(), tasks.end(), rng)",
        "schedule_seed=0x6a09e667f3bcc909",
        "CELL_SAMPLE",
        "expected_measured != 684 || expected_inadmissible != 120",
        "historical_17us_admission",
        "winner_binding=SWEEPED",
        "midpoint_events=DIAGNOSTIC_NOT_RANKING",
        "kConfirmationCandidates = 8",
        "SWEEPED+TOP8-CONFIRMED",
        "clock drift masquerade as Split-K",
        "confirmation_envelopes=%s",
        "OVERLAP/UNRESOLVED",
        "empty-launch attribution failed",
        "namespace dense_splitk_sweep_generated {",
        "&dense_splitk_sweep_generated::FN",
        "--exact-warm-ab",
        "EXACT_WARM_AB cfg=8x%dx128_w8x16_s2_bc0",
        "artifact=shape-specific-xplane-repack",
        "dense_splitk_sweep_exact::tn64",
        "dense_splitk_sweep_exact::tn128",
        "constexpr double kHistoricalRelativeTolerance = 0.03",
        "constexpr ExactRequest requests[]{{64, 7.854}, {128, 7.696}}",
        "historical_admission=%s range=[%.6f,%.6f]_us",
        "result.packed_reshape_us - result.shipping_ordinary_reshape_us",
        "result.packed_internal_producer_us - result.packed_reshape_us",
        "all_ok = all_ok && ok && historical_admitted && delta_conserved",
        "conservation_error=%+.9f_us/%s",
    ):
        require(main, token, "sweep driver", bad)
    forbid(main, "marlin-wk4-aligned-single-row", "sweep driver", bad)

    unit = files["unit"]
    for token in (
        "DENSE_SPLITK_UNIT_CONFIGS(DENSE_SPLITK_DEFINE_WRAPPER)",
        "run_row<TM,TN,TK,WM,WN,ST,BC>",
        "namespace dense_splitk_sweep_generated {",
    ):
        require(unit, token, "generated wrapper", bad)

    cmake = files["cmake"]
    marker = "# Final dense fixed Split-K performance search."
    block = cmake[cmake.find(marker):] if marker in cmake else ""
    for token in (
        "if(_tm EQUAL 8 AND _wm EQUAL 8 AND _bc EQUAL 0)",
        "NOT _DENSE_SPLITK_SOURCE_ROW_COUNT EQUAL 1772",
        "NOT _DENSE_SPLITK_ROW_COUNT EQUAL 201",
        "DENSE_SPLITK_CONFIGS_PER_UNIT \"4\"",
        "test_lowbit_dense_splitk_sweep",
        "dense_splitk_exact_warm_ab_tn64.cu",
        "dense_splitk_exact_warm_ab_tn128.cu",
        "runtime_cells=804",
    ):
        require(block, token, "CMake generator", bad)

    runner = files["runner"]
    for token in (
        "OUT:-/workspace/quactlize-dense-splitk-sweep-",
        "refusing to overwrite existing bundle",
        "binary_sha256=",
        "table_sha256=",
        "source_patch_sha256=",
        "source_state_sha256=",
        "PPU_ARCHS=ppu0010",
        "third_party/actlize is dirty",
        "full_B_plus_scale_rotation_over_max_2.16xL2_128MiB",
        "tee \"$run_log\"",
        "EXACT_WARM_AB",
        "--exact-warm-ab",
        "exact_same_address_warm_aggregate_historical_vs_shipping_ordinary_vs_packedA_reshape_vs_internal_S8_producer",
        'timed_iterations="${ITERATIONS:-100}"',
    ):
        require(runner, token, "box runner", bad)
    for token in ("mktemp", "/tmp/"):
        forbid(runner, token, "box runner", bad)

    for owner in ("exact_tn64", "exact_tn128"):
        exact = files[owner]
        for token in (
            "#define PPU_B_CHUNK 0",
            "run_exact_warm_ab<8,",
            "dense_splitk_sweep_exact",
        ):
            require(exact, token, owner, bad)
    return bad


def main() -> int:
    table_text = TABLE.read_text()
    rows = parse_rows(table_text)
    files = {
        "bench": BENCH.read_text(), "main": MAIN.read_text(),
        "unit": UNIT.read_text(), "handle": HANDLE.read_text(),
        "cmake": CMAKE.read_text(), "runner": RUNNER.read_text(),
        "exact_tn64": EXACT_TN64.read_text(),
        "exact_tn128": EXACT_TN128.read_text(),
    }
    splits = (1, 2, 4, 8)
    bad = audit(rows, splits, files)
    if bad:
        for item in bad:
            print(f"[dense-splitk-sweep] FAIL: {item}")
        return 1

    controls: list[tuple[str, list[tuple[int, ...]], tuple[int, ...], dict[str, str]]] = []
    controls.append(("dropped-source-row", rows[:-1], splits, files))
    controls.append(("duplicate-source-row", rows + [rows[0]], splits, files))
    controls.append(("missing-S8", rows, (1, 2, 4), files))
    changed_stage = list(rows)
    target = next(i for i, row in enumerate(changed_stage)
                  if row[0] == 8 and row[3] == 8 and row[6] == 0)
    changed_stage[target] = (*changed_stage[target][:5], 99, changed_stage[target][6])
    controls.append(("pipeline-depth-drift", changed_stage, splits, files))
    for name, token, owner in (
        ("lost-shipping-mode", "QuantMode::FinegrainedScaleOnly", "bench"),
        ("lost-cold-threshold", "std::ceil(2.16 * double(cli.l2_bytes))", "main"),
        ("lost-generator-filter", "if(_tm EQUAL 8 AND _wm EQUAL 8 AND _bc EQUAL 0)", "cmake"),
        ("lost-raw-samples", "CELL_SAMPLE", "main"),
        ("lost-multitu-namespace", "&dense_splitk_sweep_generated::FN", "main"),
        ("lost-envelope-verdict", "confirmation_envelopes=%s", "main"),
        ("lost-worktree-binding", "source_state_sha256=", "runner"),
        ("lost-timed-output-gate", "a skipped launch", "bench"),
        ("lost-interleaved-confirmation", "clock drift masquerade as Split-K", "main"),
        ("lost-arch-binding", "PPU_ARCHS=ppu0010", "runner"),
    ):
        planted = dict(files)
        planted[owner] = planted[owner].replace(token, "PLANTED_ABSENT")
        controls.append((name, rows, splits, planted))
    planted = dict(files)
    planted["bench"] = planted["bench"].replace(
        "handle->run(nullptr)",
        "handle->run_producer_only_for_diagnostics(nullptr)",
        1,
    )
    controls.append(("producer-only-leaked-into-ranking", rows, splits, planted))
    for name, token, owner in (
        ("lost-exact-warm-cli", "--exact-warm-ab", "main"),
        ("lost-producer-only-seam", "run_producer_only_for_diagnostics(", "handle"),
        ("lost-exact-tn64-tu", "run_exact_warm_ab<8,64,", "exact_tn64"),
        ("lost-exact-tn128-tu", "run_exact_warm_ab<8,128,", "exact_tn128"),
        ("lost-historical-anchor", "constexpr ExactRequest requests[]{{64, 7.854}, {128, 7.696}}", "main"),
        ("lost-historical-admission", "all_ok = all_ok && ok && historical_admitted && delta_conserved", "main"),
    ):
        planted = dict(files)
        planted[owner] = planted[owner].replace(token, "PLANTED_ABSENT")
        controls.append((name, rows, splits, planted))
    for name, old, new in (
        ("swapped-provider-delta",
         "result.packed_reshape_us - result.shipping_ordinary_reshape_us",
         "result.shipping_ordinary_reshape_us - result.packed_reshape_us"),
        ("swapped-internal-delta",
         "result.packed_internal_producer_us - result.packed_reshape_us",
         "result.packed_reshape_us - result.packed_internal_producer_us"),
    ):
        planted = dict(files)
        planted["main"] = planted["main"].replace(old, new, 1)
        controls.append((name, rows, splits, planted))
    escaped = 0
    for name, planted_rows, planted_splits, planted_files in controls:
        if not audit(planted_rows, planted_splits, planted_files):
            print(f"[dense-splitk-sweep] FAIL: negative control escaped: {name}")
            escaped += 1
    if escaped:
        return 1
    counts = calculate(rows, splits)
    print(
        "[dense-splitk-sweep] PASS: "
        f"source={counts.source} packedA={counts.packed} cells={counts.cells} "
        f"runnable={counts.measured} inadmissible={counts.inadmissible} "
        f"per-S={counts.per_split}; {len(controls)} semantic plants rejected"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

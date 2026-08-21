#!/usr/bin/env python3
"""Fail-closed source contract for the absolute persistent-DP grid seam.

L202 proves the integer ownership rule independently.  This check binds the
planned public CLI to a payload field named ``grid_ctas_override`` and then to
the persistent kernel's physical-grid calculation.  It intentionally does not
claim device performance or runtime occupancy.

The production seam may be absent while the feature is being implemented; in
that state this script fails rather than classifying the missing feature as a
pass.  ``--self-test`` exercises the checker itself with synthetic source and
does not inspect production files.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATHS = {
    "bench": ROOT / "benchmarks" / "test_lowbit_dense_bench.cu",
    "kernel": ROOT / "quactlize" / "include" /
        "actlize_extensions" / "cutlass" / "gemm" / "kernel" /
        "ppu_aiu_gemm_mixed_input_persistent.hpp",
    "oracle": ROOT / "dev" / "fold_derivation" /
        "l202_persistent_absolute_grid_oracle.py",
    "runner": ROOT / "tools" / "run_dense_persistent_grid_q4k65_box.sh",
    "scheduler": ROOT / "third_party" / "actlize" / "include" / "cutlass" /
        "gemm" / "kernel" / "static_tile_scheduler.hpp",
}

ORACLE_WITNESS = (
    "[l202] PASS: Q=2048 capacity=576 grids=9/9; "
    "q=worker+tG exact-once; default0=capacity; "
    "worker/grid/baseline/effective capacity separated; "
    "negative-controls=5/5_RED"
)


def require(text: str, token: str, owner: str, bad: list[str],
            count: int = 1) -> None:
    actual = text.count(token)
    if actual != count:
        bad.append(
            f"{owner}: expected {count} occurrence(s) of {token!r}, found {actual}"
        )


def require_at_least(text: str, token: str, owner: str,
                     bad: list[str], minimum: int = 1) -> None:
    actual = text.count(token)
    if actual < minimum:
        bad.append(
            f"{owner}: expected at least {minimum} occurrence(s) of "
            f"{token!r}, found {actual}"
        )


def audit(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    bench = files["bench"]
    kernel = files["kernel"]
    oracle = files["oracle"]
    runner = files["runner"]
    scheduler = files["scheduler"]

    # Public CLI: zero is the exact historical capacity-selected path.  The
    # spelling is intentionally pinned; an alias not connected to the payload
    # would recreate the dead-switch failure mode.
    for token in (
        "int persistent_grid_ctas = 0;",
        'cmd.get_cmd_line_argument("persistent-grid-ctas", persistent_grid_ctas);',
        "--persistent-grid-ctas=<n>",
        "arguments.grid_ctas_override",
        "#if defined(DENSE_PERSISTENT_AB) || defined(DENSE_STREAMK_AB) || "
        "defined(DENSE_MARLIN_AB)\n#define DENSE_SCHEDULER_AB 1",
    ):
        require(bench, token, "dense benchmark CLI/payload", bad)
    require_at_least(
        bench, "options.persistent_grid_ctas", "dense benchmark CLI/payload", bad)

    # The field must cross Arguments -> Params and be consumed by grid shape.
    # Two identical declarations are expected: one in each ABI aggregate.
    require(
        kernel,
        "uint32_t grid_ctas_override = 0;",
        "persistent Arguments/Params ABI",
        bad,
        count=2,
    )
    for token in (
        "args.grid_ctas_override",
        "params.grid_ctas_override",
        "resident_workers",
        "logical_tiles",
        "uint64_t(logical_grid.x)",
        "uint64_t(args.grid_ctas_override) <= logical_tiles",
    ):
        require_at_least(kernel, token, "persistent grid lowering", bad)
    default_forms = (
        "params.grid_ctas_override == 0",
        "params.grid_ctas_override != 0",
    )
    present_default_forms = [form for form in default_forms if form in kernel]
    if len(present_default_forms) != 1:
        bad.append(
            "persistent grid lowering: expected exactly one explicit "
            f"0=resident-capacity selection, found {present_default_forms}"
        )

    # L202 remains an independent authority.  These anchors pin both the
    # actual stride expression and the deliberately non-CU-multiple G=512.
    for token in (
        "DEFAULT_Q = 2048",
        "DEFAULT_CAPACITY = 576",
        "DEFAULT_GRIDS = (72, 144, 216, 288, 360, 432, 504, 512, 576)",
        "tile = worker + iteration * stride",
        '"stride-uses-capacity"',
        '"missing-one-unit"',
        '"grid-exceeds-capacity"',
        '"default-zero-is-not-capacity"',
        '"expected-grid-drift"',
        '"[l202] PASS: Q=2048 capacity=576 grids=9/9; "',
        '"worker/grid/baseline/effective capacity separated; "',
        '"negative-controls=5/5_RED"',
    ):
        require(oracle, token, "L202 independent oracle", bad)

    # Bind L202 to the actual device ownership seam.  The physical block ID is
    # linearized from gridDim and every subsequent tile advances by that same
    # total grid size.  Without these anchors L202 would only prove a parallel
    # host model, not the production device scheduler.
    for token in (
        "current_work_linear_idx_ = uint64_t(blockIdx.x)",
        "total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);",
        "current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count);",
    ):
        require_at_least(scheduler, token, "production persistent ownership stride", bad)

    # The device runner must keep the ordinary control and pure-DP subject in
    # one binary while binding every requested absolute grid to the production
    # marker. It must not quietly turn this experiment back into Stream-K.
    for token in (
        "test_lowbit_dense_streamk_q4k65_ab",
        'PERSISTENT_GRID_CTAS_LIST:-72 144 216 288 360 432 504 512 576',
        "== exact non-persistent control ==",
        "== exact persistent default grid (request=0) ==",
        "persistent_grid_authority=persistent-default-arm",
        "persistent_default_occupancy_api",
        "== exact persistent pure-DP grid=%s ==",
        '--persistent "--persistent-grid-ctas=$grid"',
        '"$persistent_cu" "$persistent_occupancy"',
        "persistent requested/resolved markers=",
        "marker_effective_overhead",
        "effective_capacity_overhead_pct",
        "all independent cells were attempted",
    ):
        require_at_least(runner, token, "persistent-grid box runner", bad)
    if "--streamk " in runner or "--streamk-tail" in runner:
        bad.append("persistent-grid box runner must not launch a Stream-K subject")
    return bad


def synthetic_files() -> dict[str, str]:
    return {
        "bench": """
struct Options { int persistent_grid_ctas = 0; };
cmd.get_cmd_line_argument("persistent-grid-ctas", persistent_grid_ctas);
help("--persistent-grid-ctas=<n>");
use(options.persistent_grid_ctas);
arguments.grid_ctas_override = options.persistent_grid_ctas;
#if defined(DENSE_PERSISTENT_AB) || defined(DENSE_STREAMK_AB) || defined(DENSE_MARLIN_AB)
#define DENSE_SCHEDULER_AB 1
#endif
""",
        "kernel": """
struct Arguments { uint32_t grid_ctas_override = 0; };
struct Params { uint32_t grid_ctas_override = 0; };
pass(args.grid_ctas_override);
auto resident_workers = capacity();
auto logical_tiles = params.scheduler.blocks_per_problem_;
auto logical_grid = get_grid();
auto exact_logical_tiles = uint64_t(logical_grid.x);
auto exact_requested_grid = uint64_t(args.grid_ctas_override) <= logical_tiles;
auto requested = params.grid_ctas_override == 0 ? resident_workers
                                                 : params.grid_ctas_override;
""",
        "oracle": """
DEFAULT_Q = 2048
DEFAULT_CAPACITY = 576
DEFAULT_GRIDS = (72, 144, 216, 288, 360, 432, 504, 512, 576)
tile = worker + iteration * stride
"stride-uses-capacity"
"missing-one-unit"
"grid-exceeds-capacity"
"default-zero-is-not-capacity"
"expected-grid-drift"
"[l202] PASS: Q=2048 capacity=576 grids=9/9; "
"worker/grid/baseline/effective capacity separated; "
"negative-controls=5/5_RED"
""",
        "runner": """
target=test_lowbit_dense_streamk_q4k65_ab
grids="${PERSISTENT_GRID_CTAS_LIST:-72 144 216 288 360 432 504 512 576}"
echo "== exact non-persistent control =="
echo "== exact persistent default grid (request=0) =="
echo "persistent_grid_authority=persistent-default-arm"
echo "persistent_default_occupancy_api"
printf "== exact persistent pure-DP grid=%s =="
run --persistent "--persistent-grid-ctas=$grid"
bind "$persistent_cu" "$persistent_occupancy"
echo "persistent requested/resolved markers="
marker_effective_overhead=0
effective_capacity_overhead_pct=0
echo "all independent cells were attempted"
""",
        "scheduler": """
current_work_linear_idx_ = uint64_t(blockIdx.x);
total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);
current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count);
""",
    }


def mutate_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(
            f"negative {label}: expected one anchor {old!r}, found {count}"
        )
    return text.replace(old, new, 1)


def negative_controls() -> list[str]:
    failures: list[str] = []
    cases = (
        (
            "CLI default no longer means capacity",
            "bench",
            "int persistent_grid_ctas = 0;",
            "int persistent_grid_ctas = 1;",
        ),
        (
            "CLI spelling disconnected",
            "bench",
            '"persistent-grid-ctas"',
            '"persistent-grid-size"',
        ),
        (
            "payload assignment removed",
            "bench",
            "arguments.grid_ctas_override = options.persistent_grid_ctas;",
            "consume(options.persistent_grid_ctas);",
        ),
        (
            "Params field removed",
            "kernel",
            "struct Params { uint32_t grid_ctas_override = 0; };",
            "struct Params {};",
        ),
        (
            "grid calculation ignores payload",
            "kernel",
            "params.grid_ctas_override == 0",
            "true",
        ),
        (
            "direct API accepts a grid larger than the logical tile count",
            "kernel",
            "uint64_t(args.grid_ctas_override) <= logical_tiles",
            "true",
        ),
        (
            "runner drops absolute grid flag",
            "runner",
            '--persistent "--persistent-grid-ctas=$grid"',
            "--persistent",
        ),
        (
            "runner drops persistent-default occupancy authority",
            "runner",
            '"$persistent_cu" "$persistent_occupancy"',
            '"$control_cu" "$control_occupancy"',
        ),
        (
            "device scheduler strides by capacity instead of physical grid",
            "scheduler",
            "current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count);",
            "current_work_linear_idx_ += resident_capacity * uint64_t(advance_count);",
        ),
        (
            "Q4K65 Stream-K target no longer implies its persistent arm",
            "bench",
            "#if defined(DENSE_PERSISTENT_AB) || defined(DENSE_STREAMK_AB) || defined(DENSE_MARLIN_AB)",
            "#if defined(DENSE_PERSISTENT_AB) || defined(DENSE_STREAMK_ONLY) || defined(DENSE_MARLIN_AB)",
        ),
    )
    for label, key, old, new in cases:
        files = synthetic_files()
        files[key] = mutate_once(files[key], old, new, label)
        if not audit(files):
            failures.append(f"{label}: planted defect stayed green")
    return failures


def run_oracle() -> list[str]:
    result = subprocess.run(
        [sys.executable, str(PATHS["oracle"])],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-8:])
        return [f"L202 exited {result.returncode}:\n{tail}"]
    if result.stdout.count(ORACLE_WITNESS) != 1:
        return ["L202 exited zero without its unique PASS witness"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise the checker with synthetic source; do not inspect production",
    )
    args = parser.parse_args()

    failures = negative_controls()
    if args.self_test:
        failures.extend(audit(synthetic_files()))
    else:
        missing = [str(path) for path in PATHS.values() if not path.is_file()]
        if missing:
            failures.append("missing contract inputs: " + ", ".join(missing))
        else:
            files = {key: path.read_text() for key, path in PATHS.items()}
            failures.extend(audit(files))
            failures.extend(run_oracle())

    if failures:
        for failure in failures:
            print(f"[persistent-grid-contract] FAIL: {failure}", file=sys.stderr)
        return 1
    scope = "checker-self-test" if args.self_test else "production+oracle"
    print(
        "[persistent-grid-contract] PASS: "
        f"scope={scope}; grid_ctas_override/--persistent-grid-ctas bound; "
        "L202 exact ownership bound to production gridDim stride; "
        "source-negative-controls=10/10_RED; "
        "oracle-negative-controls=5/5_RED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

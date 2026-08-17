#!/usr/bin/env python3
"""L202: exhaustive host oracle for an absolute persistent-DP grid.

The production static scheduler starts a worker at its linear block index and
advances by the *physical grid size*.  For a selected absolute grid ``G`` its
entire ownership rule is therefore::

    q(worker, iteration) = worker + iteration * G

This file proves that rule cell-by-cell for the preregistered Q4_K65 geometry.
It deliberately keeps four different quantities separate:

* ``worker_balance`` is the padded logical-worker rounds relative to Q;
* ``resident_grid_fraction`` is the selected G divided by resident capacity W;
* ``baseline_wave_overhead`` is the ordinary Q-at-capacity wave quantization;
* ``effective_capacity_overhead`` charges every logical worker round at the
  full resident capacity W.  It therefore includes the slots deliberately not
  launched when G < W.

A grid of 512 is the useful discriminator: it has perfectly balanced
four-tile workers while launching only 88.89% of the 576-CTA resident grid.

No PPU behavior is modelled here.  The remaining device questions are whether
the runtime occupancy really grants the preregistered capacity and what each
grid costs in time.
"""

from __future__ import annotations

import dataclasses
import math


DEFAULT_Q = 2048
DEFAULT_CAPACITY = 576
DEFAULT_GRIDS = (72, 144, 216, 288, 360, 432, 504, 512, 576)


class OracleError(RuntimeError):
    """An ownership or fail-closed contract did not hold."""


@dataclasses.dataclass(frozen=True)
class GridEvidence:
    requested_grid: int
    selected_grid: int
    long_workers: int
    short_workers: int
    long_iterations: int
    short_iterations: int
    max_iterations: int
    iteration_spread: int
    logical_rounds: int
    logical_empty_slots: int
    worker_balance: float
    resident_grid_fraction: float
    baseline_wave_overhead: float
    effective_capacity_overhead: float
    missing: int
    duplicate: int
    out_of_range: int


# This is an independent, hand-pinned census.  In particular, G=512 and
# G=576 point in opposite directions for worker balance versus resident-slot
# fill, so a model that silently substitutes capacity for G cannot pass.
EXPECTED = {
    # G: long_workers, short_workers, long_iter, short_iter,
    #    logical_empty, worker_balance, grid_fraction, baseline_wave,
    #    effective_capacity
    72:  (32, 40, 29, 28, 40, 0.01953125, 0.125, 0.125, 7.15625),
    144: (32, 112, 15, 14, 112, 0.0546875, 0.25, 0.125, 3.21875),
    216: (104, 112, 10, 9, 112, 0.0546875, 0.375, 0.125, 1.8125),
    288: (32, 256, 8, 7, 256, 0.125, 0.5, 0.125, 1.25),
    360: (248, 112, 6, 5, 112, 0.0546875, 0.625, 0.125, 0.6875),
    432: (320, 112, 5, 4, 112, 0.0546875, 0.75, 0.125, 0.40625),
    504: (32, 472, 5, 4, 472, 0.23046875, 0.875, 0.125, 0.40625),
    512: (0, 512, 4, 4, 0, 0.0, 8.0 / 9.0, 0.125, 0.125),
    576: (320, 256, 4, 3, 256, 0.125, 1.0, 0.125, 0.125),
}


def resolve_grid(requested: int, capacity: int) -> int:
    """Resolve the public 0=capacity default, rejecting impossible grids."""
    if capacity <= 0:
        raise OracleError(f"capacity must be positive, got {capacity}")
    selected = capacity if requested == 0 else requested
    if selected <= 0:
        raise OracleError(f"absolute grid must be positive, got {selected}")
    if selected > capacity:
        raise OracleError(
            f"absolute grid G={selected} exceeds exact capacity={capacity}"
        )
    return selected


def prove_grid(
    q_tiles: int,
    capacity: int,
    requested_grid: int,
    *,
    stride_override: int | None = None,
    omit: tuple[int, int] | None = None,
    expected_grid: int | None = None,
) -> GridEvidence:
    """Expand every worker/iteration pair and require exact-once ownership."""
    if q_tiles <= 0:
        raise OracleError(f"Q must be positive, got {q_tiles}")
    grid = resolve_grid(requested_grid, capacity)
    if expected_grid is not None and grid != expected_grid:
        raise OracleError(
            f"selected grid G={grid} differs from expected G={expected_grid}"
        )

    stride = grid if stride_override is None else stride_override
    if stride <= 0:
        raise OracleError(f"worker stride must be positive, got {stride}")

    visits = bytearray(q_tiles)
    iterations = [0] * grid
    out_of_range = 0
    for worker in range(grid):
        iteration = 0
        while True:
            tile = worker + iteration * stride
            if tile >= q_tiles:
                break
            if omit != (worker, iteration):
                visits[tile] += 1
                iterations[worker] += 1
            iteration += 1

    missing = sum(value == 0 for value in visits)
    duplicate = sum(value > 1 for value in visits)
    if missing or duplicate or out_of_range:
        raise OracleError(
            "q=worker+tG is not exact-once: "
            f"Q={q_tiles} G={grid} stride={stride} missing={missing} "
            f"duplicate={duplicate} out_of_range={out_of_range}"
        )

    minimum = min(iterations)
    maximum = max(iterations)
    iteration_spread = maximum - minimum
    if iteration_spread > 1:
        raise OracleError(
            "persistent workers differ by "
            f"{iteration_spread} iterations, expected <=1"
        )

    remainder = q_tiles % grid
    long_workers = remainder
    short_workers = grid - remainder
    short_iterations = q_tiles // grid
    long_iterations = short_iterations + (1 if remainder else 0)
    if remainder == 0:
        # No worker is strictly longer.  Keep both iteration columns equal so
        # the printed row remains directly comparable with non-divisible G.
        long_iterations = short_iterations
    if (iterations.count(maximum), iterations.count(minimum)) != (
        (long_workers if remainder else grid),
        (short_workers if remainder else grid),
    ):
        # In the divisible case min==max and both count() calls return G.  The
        # explicit branch above documents that intentional overlap.
        raise OracleError(
            f"long/short worker census drifted for Q={q_tiles} G={grid}"
        )

    logical_rounds = math.ceil(q_tiles / grid)
    logical_empty = logical_rounds * grid - q_tiles
    worker_balance = logical_empty / q_tiles
    resident_grid_fraction = grid / capacity
    baseline_rounds = math.ceil(q_tiles / capacity)
    baseline_wave_overhead = (
        baseline_rounds * capacity - q_tiles
    ) / q_tiles
    effective_capacity_overhead = (
        logical_rounds * capacity - q_tiles
    ) / q_tiles

    return GridEvidence(
        requested_grid=requested_grid,
        selected_grid=grid,
        long_workers=long_workers,
        short_workers=short_workers,
        long_iterations=long_iterations,
        short_iterations=short_iterations,
        max_iterations=maximum,
        iteration_spread=iteration_spread,
        logical_rounds=logical_rounds,
        logical_empty_slots=logical_empty,
        worker_balance=worker_balance,
        resident_grid_fraction=resident_grid_fraction,
        baseline_wave_overhead=baseline_wave_overhead,
        effective_capacity_overhead=effective_capacity_overhead,
        missing=missing,
        duplicate=duplicate,
        out_of_range=out_of_range,
    )


def require_expected(evidence: GridEvidence) -> None:
    expected = EXPECTED.get(evidence.selected_grid)
    if expected is None:
        raise OracleError(f"G={evidence.selected_grid} missing from denominator")
    got = (
        evidence.long_workers,
        evidence.short_workers,
        evidence.long_iterations,
        evidence.short_iterations,
        evidence.logical_empty_slots,
        evidence.worker_balance,
        evidence.resident_grid_fraction,
        evidence.baseline_wave_overhead,
        evidence.effective_capacity_overhead,
    )
    if len(got) != len(expected) or any(
        not math.isclose(float(actual), float(want), rel_tol=0.0, abs_tol=1e-12)
        for actual, want in zip(got, expected)
    ):
        raise OracleError(
            f"G={evidence.selected_grid} census drifted: got={got} expected={expected}"
        )


def expect_red(label: str, callback, fragment: str) -> None:
    try:
        callback()
    except OracleError as error:
        if fragment not in str(error):
            raise AssertionError(
                f"negative {label} failed for the wrong reason: {error}"
            ) from error
        print(f"[l202 negative] {label}=RED reason={str(error)!r}")
        return
    raise AssertionError(f"negative {label} unexpectedly stayed green")


def main() -> int:
    rows: list[GridEvidence] = []
    for grid in DEFAULT_GRIDS:
        evidence = prove_grid(
            DEFAULT_Q, DEFAULT_CAPACITY, grid, expected_grid=grid
        )
        require_expected(evidence)
        rows.append(evidence)
        print(
            "[l202 grid] "
            f"Q={DEFAULT_Q} capacity={DEFAULT_CAPACITY} G={grid} "
            f"long={evidence.long_workers}x{evidence.long_iterations} "
            f"short={evidence.short_workers}x{evidence.short_iterations} "
            f"max_iterations={evidence.max_iterations} "
            f"iteration_spread={evidence.iteration_spread} "
            f"logical_rounds={evidence.logical_rounds} "
            f"logical_empty={evidence.logical_empty_slots} "
            f"worker_balance={100.0 * evidence.worker_balance:.6f}% "
            f"resident_grid_fraction={100.0 * evidence.resident_grid_fraction:.6f}% "
            f"baseline_wave_overhead={100.0 * evidence.baseline_wave_overhead:.6f}% "
            f"effective_capacity_overhead={100.0 * evidence.effective_capacity_overhead:.6f}% "
            "coverage=exact-once"
        )

    default = prove_grid(DEFAULT_Q, DEFAULT_CAPACITY, 0)
    if default.selected_grid != DEFAULT_CAPACITY or default != rows[-1]:
        # requested_grid intentionally differs between the two dataclasses;
        # compare the behavior-bearing fields explicitly below instead.
        if dataclasses.replace(default, requested_grid=DEFAULT_CAPACITY) != rows[-1]:
            raise OracleError("default G=0 is not bit-identical to capacity G=576")
    print(
        "[l202 default] requested=0 selected=576 "
        "identity=explicit-capacity/PASS"
    )

    expect_red(
        "stride-uses-capacity",
        lambda: prove_grid(DEFAULT_Q, DEFAULT_CAPACITY, 512,
                           stride_override=DEFAULT_CAPACITY),
        "not exact-once",
    )
    expect_red(
        "missing-one-unit",
        lambda: prove_grid(DEFAULT_Q, DEFAULT_CAPACITY, 576, omit=(0, 0)),
        "missing=1",
    )
    expect_red(
        "grid-exceeds-capacity",
        lambda: prove_grid(DEFAULT_Q, DEFAULT_CAPACITY, 577),
        "exceeds exact capacity",
    )
    expect_red(
        "default-zero-is-not-capacity",
        lambda: prove_grid(DEFAULT_Q, DEFAULT_CAPACITY, 0,
                           expected_grid=DEFAULT_CAPACITY - 1),
        "differs from expected",
    )
    expect_red(
        "expected-grid-drift",
        lambda: prove_grid(DEFAULT_Q, DEFAULT_CAPACITY, 504,
                           expected_grid=512),
        "differs from expected",
    )

    if len(rows) != len(DEFAULT_GRIDS) or set(EXPECTED) != set(DEFAULT_GRIDS):
        raise OracleError(
            "grid denominator did not close: "
            f"rows={len(rows)} expected={len(DEFAULT_GRIDS)}"
        )
    print(
        "[l202] PASS: Q=2048 capacity=576 grids=9/9; "
        "q=worker+tG exact-once; default0=capacity; "
        "worker/grid/baseline/effective capacity separated; "
        "negative-controls=5/5_RED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

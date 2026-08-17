#!/usr/bin/env python3
"""L201: exhaustive host oracle for dense DP + tail-only Stream-K.

This is deliberately independent of ``PersistentTileSchedulerPPUStreamKParams``.
It reads the committed dense X-macro authority, derives Q/Kt/W itself, and
models seam repair as a transformation of balanced interval *boundaries*.
That representation is different from the production start/count code and
makes the proof small:

* raw balanced boundaries partition every group's flattened (tile, K) line;
* seam repair snaps both users of a shared boundary to the same position;
* strictly increasing repaired boundaries are necessary and sufficient for
  nonempty work units and an exact partition of that line;
* groups own disjoint q residue classes, so exact group partitions imply
  exact-once global (q, k_tile) coverage.

The production tail policy first tries the legacy <=8 locality group.  There
are 404 committed row/BPC tuples for which that group collapses at least one
work unit.  They are an expected fail-closed baseline, never valid evidence.
The reviewed fallback keeps U unchanged and searches divisors of U in
ascending order; all 404 find an exact group.  Every admitted tuple is then
expanded cell-by-cell below, rather than accepted from the boundary theorem
alone.
"""

from __future__ import annotations

import collections
import dataclasses
import pathlib
import re
import sys
from collections.abc import Callable


ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "benchmarks" / "lowbit_dense_configs.inc"

PROBLEM_M = 2048
PROBLEM_N = 4096
PROBLEM_K = 4096
REAL_CUS = 72
MIN_SK_ITERS = 8
MAX_LEGACY_GROUPS = 8
L2_ALIGNMENT = 128
FP32_BYTES = 4
LOCK_BYTES = 4

EXPECTED_SOURCE_ROWS = 1772
EXPECTED_ELIGIBLE_ROWS = 577
EXPECTED_FILTERED_ROWS = 1195
EXPECTED_TUPLES = EXPECTED_ELIGIBLE_ROWS * 8

EXPECTED_THREAD_CENSUS = {64: 248, 128: 329}
EXPECTED_STAGE_CENSUS = {2: 132, 3: 131, 4: 119, 6: 108, 8: 87}
EXPECTED_TILE_M_CENSUS = {16: 88, 32: 175, 64: 199, 128: 97, 256: 18}

# The preferred group is the unchanged legacy locality choice.  These 404
# tuples contain at least one duplicated repaired boundary and therefore must
# not launch with G=8.
EXPECTED_PREFERRED_REPRESENTABLE = 4212
EXPECTED_PREFERRED_COLLAPSED = 404
EXPECTED_PREFERRED_EMPTY_UNITS = 15952
EXPECTED_EMPTY_UNIT_HISTOGRAM = {
    8: 206,
    16: 39,
    24: 48,
    32: 18,
    48: 3,
    88: 36,
    96: 18,
    192: 36,
}

# Production preserves G=8 for the 4212 already-exact tuples.  Only the 404
# collapsed tuples enter the ascending exact-divisor search.
EXPECTED_FINAL_GROUP_CENSUS = {
    8: 4212,
    9: 33,
    18: 20,
    27: 10,
    36: 6,
    40: 67,
    42: 35,
    45: 51,
    54: 18,
    72: 36,
    90: 46,
    108: 36,
    126: 10,
    144: 36,
}

# (Q, Kt, BPC, S, U, selected fallback G) -> number of committed configs.
EXPECTED_FALLBACK_SIGNATURES = {
    (512, 16, 2, 80, 144, 36): 3,
    (512, 16, 4, 224, 288, 36): 3,
    (512, 32, 5, 152, 360, 40): 9,
    (1024, 16, 3, 160, 216, 54): 18,
    (1024, 16, 4, 160, 288, 72): 18,
    (1024, 16, 5, 304, 360, 45): 18,
    (1024, 16, 8, 448, 576, 72): 18,
    (2048, 16, 5, 248, 360, 90): 36,
    (2048, 16, 6, 320, 432, 108): 36,
    (2048, 16, 8, 320, 576, 144): 36,
    (4096, 32, 5, 136, 360, 40): 58,
    (8192, 16, 1, 56, 72, 9): 33,
    (8192, 16, 5, 272, 360, 45): 33,
    (8192, 32, 7, 128, 504, 42): 35,
    (16384, 16, 1, 40, 72, 18): 10,
    (16384, 16, 2, 112, 144, 18): 10,
    (16384, 16, 3, 184, 216, 27): 10,
    (16384, 16, 5, 184, 360, 90): 10,
    (16384, 16, 7, 256, 504, 126): 10,
}

EXPECTED_AGGREGATE = {
    "dp_tiles": 18_973_008,
    "sk_tiles": 671_408,
    "sk_units": 1_240_056,
    "qk_cells": 26_928_768,
    "whole_tiles": 127_816,
    "split_tiles": 543_592,
    "peer_excess": 793_536,
    "peer_ranges": 1_464_944,
    "valid_fixup_elements": 3_129_352_192,
    "reduction_bytes": 10_601_857_024,
    "barrier_bytes": 2_806_144,
    "workspace_bytes": 10_604_663_168,
    "logical_fixup_rw_bytes": 25_034_817_536,
    "all_sk_tuples": 26,
}


ROW_RE = re.compile(
    r"^\s*X\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),B\)",
    re.MULTILINE,
)


def ceil_div(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        raise ValueError(f"ceil_div requires a>=0,b>0; got {a},{b}")
    return (a + b - 1) // b


def round_up(value: int, alignment: int = L2_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("round_up requires nonnegative value and positive alignment")
    return ceil_div(value, alignment) * alignment


@dataclasses.dataclass(frozen=True)
class Config:
    tm: int
    tn: int
    tk: int
    wm: int
    wn: int
    stages: int
    bchunk: int

    @property
    def threads(self) -> int:
        if self.tm % self.wm or self.tn % self.wn:
            raise ValueError(f"non-integral warp topology: {self}")
        return 32 * (self.tm // self.wm) * (self.tn // self.wn)

    @property
    def m_tiles(self) -> int:
        return ceil_div(PROBLEM_M, self.tm)

    @property
    def n_tiles(self) -> int:
        return ceil_div(PROBLEM_N, self.tn)

    @property
    def q(self) -> int:
        return self.m_tiles * self.n_tiles

    @property
    def kt(self) -> int:
        return ceil_div(PROBLEM_K, self.tk)


@dataclasses.dataclass(frozen=True)
class BoundaryPlan:
    groups: int
    boundaries: tuple[tuple[int, ...], ...]
    empty_units: int
    reversed_units: int
    endpoint_errors: int

    @property
    def exact_nonempty(self) -> bool:
        return (
            self.groups > 0
            and self.empty_units == 0
            and self.reversed_units == 0
            and self.endpoint_errors == 0
        )


@dataclasses.dataclass(frozen=True)
class PartitionEvidence:
    qk_cells: int
    missing_cells: int
    duplicate_cells: int
    q_oob: int
    peer_holes: int
    empty_units: int
    min_interval: int
    max_interval: int
    whole_tiles: int
    split_tiles: int
    peer_excess: int
    peer_ranges: int

    @property
    def exact_nonempty(self) -> bool:
        return (
            self.qk_cells > 0
            and self.missing_cells == 0
            and self.duplicate_cells == 0
            and self.q_oob == 0
            and self.peer_holes == 0
            and self.empty_units == 0
            and self.min_interval >= MIN_SK_ITERS
        )


@dataclasses.dataclass(frozen=True)
class AuditPolicy:
    name: str
    tile_selector: Callable[[int, int], int]
    use_safe_group_fallback: bool = True
    admit_collapsed_preferred: bool = False


@dataclasses.dataclass
class AuditResult:
    errors: list[str] = dataclasses.field(default_factory=list)
    error_count: int = 0
    attempts: int = 0
    per_bpc: collections.Counter[int] = dataclasses.field(
        default_factory=collections.Counter
    )
    preferred_representable: int = 0
    preferred_collapsed: int = 0
    preferred_empty_units: int = 0
    preferred_empty_histogram: collections.Counter[int] = dataclasses.field(
        default_factory=collections.Counter
    )
    fallback: int = 0
    rejected: int = 0
    final_groups: collections.Counter[int] = dataclasses.field(
        default_factory=collections.Counter
    )
    fallback_signatures: collections.Counter[tuple[int, ...]] = dataclasses.field(
        default_factory=collections.Counter
    )
    aggregate: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    min_interval: int = 1 << 30
    max_interval: int = 0

    def add_error(self, message: str) -> None:
        self.error_count += 1
        if len(self.errors) < 20:
            self.errors.append(message)

    @property
    def ok(self) -> bool:
        return self.error_count == 0


def load_authority() -> tuple[list[Config], list[Config]]:
    text = TABLE.read_text()
    source = [Config(*map(int, match.groups())) for match in ROW_RE.finditer(text)]
    if len(source) != EXPECTED_SOURCE_ROWS:
        raise AssertionError(
            f"source authority has {len(source)} rows, expected {EXPECTED_SOURCE_ROWS}"
        )
    if len(set(source)) != len(source):
        raise AssertionError("source authority contains duplicate config rows")

    eligible = [
        row
        for row in source
        if row.tm >= 16
        and row.threads in (64, 128)
        and row.stages - 1 <= 8
    ]
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
        raise AssertionError(
            f"eligible authority has {len(eligible)} rows, expected {EXPECTED_ELIGIBLE_ROWS}"
        )
    if len(source) - len(eligible) != EXPECTED_FILTERED_ROWS:
        raise AssertionError("source/eligible/filtered denominator does not close")
    if collections.Counter(row.threads for row in eligible) != EXPECTED_THREAD_CENSUS:
        raise AssertionError("eligible CTA-thread census changed")
    if collections.Counter(row.stages for row in eligible) != EXPECTED_STAGE_CENSUS:
        raise AssertionError("eligible stage census changed")
    if collections.Counter(row.tm for row in eligible) != EXPECTED_TILE_M_CENSUS:
        raise AssertionError("eligible TileM census changed")
    if any(row.bchunk != 0 for row in eligible):
        raise AssertionError("int4 Stream-K authority unexpectedly grants B-chunk")
    return source, eligible


def tail_tiles(q: int, workers: int) -> int:
    """Unique x in [0,W) for which Q-x is a complete DP-wave multiple."""
    return q % workers


def legacy_two_wave_tiles(q: int, workers: int) -> int:
    """The unchanged forced-StreamK two-wave region, used only as a control."""
    full_waves = q // workers
    dp_waves = full_waves - 1 if full_waves > 1 else 0
    return q - dp_waves * workers


def sk_units(sk_tiles: int, kt: int, workers: int) -> int:
    if sk_tiles == 0:
        return 0
    return min(workers, (sk_tiles * kt) // MIN_SK_ITERS)


def preferred_groups(config: Config, sk_tiles: int, units: int, kt: int) -> int:
    """Independent spelling of the unchanged <=8 locality-group heuristic."""
    # RasterOrder::Heuristic rasterizes along the longer dimension, so the
    # number of independent locality groups is capped by the shorter one.
    groups = min(config.m_tiles, config.n_tiles, MAX_LEGACY_GROUPS)
    fallback = 0

    def splits_too_small(candidate: int) -> bool:
        units_per_group = units // candidate
        if units_per_group == 0:
            return True
        group_tiles = sk_tiles // candidate
        return (group_tiles * kt) // units_per_group < MIN_SK_ITERS

    def ideal(candidate: int) -> bool:
        return (
            units % candidate == 0
            and sk_tiles % candidate == 0
            and not splits_too_small(candidate)
        )

    def valid(candidate: int) -> bool:
        return units % candidate == 0 and not splits_too_small(candidate)

    while groups > 1 and not ideal(groups):
        if fallback == 0 and valid(groups):
            fallback = groups
        groups -= 1
    return fallback if groups == 1 and fallback else groups


def snap_boundary(boundary: int, kt: int) -> int:
    """Snap exactly as the device's start/end seam repair, but once per edge."""
    residue = boundary % kt
    if residue < MIN_SK_ITERS:
        return boundary - residue
    if residue > kt - MIN_SK_ITERS:
        return boundary + kt - residue
    return boundary


def make_boundary_plan(sk_tiles: int, units: int, kt: int, groups: int) -> BoundaryPlan:
    if (
        groups <= 0
        or sk_tiles <= 0
        or units <= 0
        or groups > sk_tiles
        or units % groups != 0
        or kt < MIN_SK_ITERS
    ):
        return BoundaryPlan(groups, (), 0, 0, 1)

    units_per_group = units // groups
    all_boundaries: list[tuple[int, ...]] = []
    empty = 0
    reversed_units = 0
    endpoint_errors = 0
    for group in range(groups):
        tiles_in_group = sk_tiles // groups + (group < sk_tiles % groups)
        total = tiles_in_group * kt
        base, big = divmod(total, units_per_group)
        raw = [
            base * unit + min(unit, big)
            for unit in range(units_per_group + 1)
        ]
        repaired = tuple(snap_boundary(boundary, kt) for boundary in raw)
        endpoint_errors += repaired[0] != 0 or repaired[-1] != total
        for begin, end in zip(repaired, repaired[1:]):
            empty += begin == end
            reversed_units += begin > end
        all_boundaries.append(repaired)
    return BoundaryPlan(
        groups,
        tuple(all_boundaries),
        empty,
        reversed_units,
        endpoint_errors,
    )


def select_safe_groups(
    preferred: int, sk_tiles: int, units: int, kt: int
) -> tuple[int, bool]:
    if make_boundary_plan(sk_tiles, units, kt, preferred).exact_nonempty:
        return preferred, False
    for candidate in range(1, min(sk_tiles, units) + 1):
        if units % candidate:
            continue
        if make_boundary_plan(sk_tiles, units, kt, candidate).exact_nonempty:
            return candidate, True
    return 0, True


def expand_partition(
    sk_tiles: int, units: int, kt: int, plan: BoundaryPlan
) -> PartitionEvidence:
    visits = bytearray(sk_tiles * kt)
    peers = [0] * sk_tiles
    q_oob = 0
    min_interval = 1 << 30
    max_interval = 0
    peer_ranges = 0

    for group, boundaries in enumerate(plan.boundaries):
        for begin, end in zip(boundaries, boundaries[1:]):
            count = end - begin
            if count <= 0:
                continue
            min_interval = min(min_interval, count)
            max_interval = max(max_interval, count)
            first_local = begin // kt
            last_local = (end - 1) // kt
            for local_tile in range(first_local, last_local + 1):
                q = local_tile * plan.groups + group
                if q >= sk_tiles:
                    q_oob += 1
                    continue
                tile_begin = local_tile * kt
                cell_begin = max(begin, tile_begin) - tile_begin
                cell_end = min(end, tile_begin + kt) - tile_begin
                if cell_begin >= cell_end:
                    q_oob += 1
                    continue
                peers[q] += 1
                peer_ranges += 1
                offset = q * kt
                for k_tile in range(cell_begin, cell_end):
                    visits[offset + k_tile] += 1

    missing = sum(value == 0 for value in visits)
    duplicate = sum(value > 1 for value in visits)
    peer_holes = sum(peer == 0 for peer in peers)
    whole = sum(peer == 1 for peer in peers)
    split = sum(peer > 1 for peer in peers)
    excess = sum(max(peer - 1, 0) for peer in peers)
    return PartitionEvidence(
        qk_cells=len(visits),
        missing_cells=missing,
        duplicate_cells=duplicate,
        q_oob=q_oob,
        peer_holes=peer_holes,
        empty_units=plan.empty_units,
        min_interval=0 if min_interval == 1 << 30 else min_interval,
        max_interval=max_interval,
        whole_tiles=whole,
        split_tiles=split,
        peer_excess=excess,
        peer_ranges=peer_ranges,
    )


def workspace_components(sk_tiles: int, tm: int, tn: int) -> tuple[int, int, int]:
    if sk_tiles == 0:
        return 0, 0, 0
    reduction = round_up(sk_tiles * tm * tn * FP32_BYTES)
    barriers = round_up(sk_tiles * LOCK_BYTES)
    return reduction, barriers, reduction + barriers


def _minimality_error(q: int, workers: int, selected: int) -> str | None:
    expected = q % workers
    solutions = [
        candidate
        for candidate in range(workers)
        if (q - candidate) % workers == 0
    ]
    if solutions != [expected]:
        return f"Euclidean tail oracle is not unique: Q={q} W={workers} {solutions}"
    if selected != expected:
        return (
            "tail tile minimality violated: "
            f"Q={q} W={workers} selected={selected} unique_remainder={expected}"
        )
    return None


def audit_domain(
    configs: list[Config],
    policy: AuditPolicy,
    *,
    fixed_expectations: bool,
    fail_fast: bool = False,
) -> AuditResult:
    result = AuditResult()

    def reject(message: str) -> bool:
        result.add_error(message)
        return fail_fast

    for config_index, config in enumerate(configs):
        q = config.q
        kt = config.kt
        for bpc in range(1, 9):
            workers = REAL_CUS * bpc
            result.attempts += 1
            result.per_bpc[bpc] += 1
            selected_tiles = policy.tile_selector(q, workers)
            error = _minimality_error(q, workers, selected_tiles)
            if error is not None:
                if reject(f"config={config_index} bpc={bpc}: {error}"):
                    return result
                continue
            if not (0 <= selected_tiles < workers and selected_tiles <= q):
                if reject(
                    f"config={config_index} bpc={bpc}: tail extent out of bounds"
                ):
                    return result
                continue

            dp_tiles = q - selected_tiles
            if dp_tiles % workers:
                if reject(
                    f"config={config_index} bpc={bpc}: DP prefix is not complete waves"
                ):
                    return result
                continue
            units = sk_units(selected_tiles, kt, workers)
            if selected_tiles == 0:
                if units != 0 or workspace_components(0, config.tm, config.tn) != (0, 0, 0):
                    if reject("zero-tail case retained units or workspace"):
                        return result
                result.aggregate["dp_tiles"] += dp_tiles
                continue
            if units <= 0 or units > workers or units * MIN_SK_ITERS > selected_tiles * kt:
                if reject(
                    f"config={config_index} bpc={bpc}: invalid U={units} for S={selected_tiles},Kt={kt}"
                ):
                    return result
                continue

            preferred = preferred_groups(config, selected_tiles, units, kt)
            preferred_plan = make_boundary_plan(selected_tiles, units, kt, preferred)
            if preferred_plan.exact_nonempty:
                result.preferred_representable += 1
            else:
                result.preferred_collapsed += 1
                result.preferred_empty_units += preferred_plan.empty_units
                result.preferred_empty_histogram[preferred_plan.empty_units] += 1

            if policy.admit_collapsed_preferred:
                selected_groups = preferred
                used_fallback = False
            elif policy.use_safe_group_fallback:
                selected_groups, used_fallback = select_safe_groups(
                    preferred, selected_tiles, units, kt
                )
            else:
                selected_groups = preferred if preferred_plan.exact_nonempty else 0
                used_fallback = False

            if selected_groups == 0:
                result.rejected += 1
                if reject(
                    f"config={config_index} bpc={bpc}: no exact nonempty tail group"
                ):
                    return result
                continue
            if preferred_plan.exact_nonempty and (
                selected_groups != preferred or used_fallback
            ):
                if reject(
                    f"config={config_index} bpc={bpc}: exact legacy G={preferred} was not preserved"
                ):
                    return result

            final_plan = make_boundary_plan(
                selected_tiles, units, kt, selected_groups
            )
            evidence = expand_partition(selected_tiles, units, kt, final_plan)
            if final_plan.empty_units or evidence.empty_units:
                if reject(
                    "empty repaired unit admitted: "
                    f"config={config_index} bpc={bpc} S={selected_tiles} U={units} "
                    f"G={selected_groups} empty={final_plan.empty_units}"
                ):
                    return result
                continue
            if not evidence.exact_nonempty:
                if reject(
                    "(q,k) exact-once failed: "
                    f"config={config_index} bpc={bpc} S={selected_tiles} U={units} "
                    f"G={selected_groups} evidence={evidence}"
                ):
                    return result
                continue
            if evidence.whole_tiles + evidence.split_tiles != selected_tiles:
                if reject("DP/SK tile census did not close"):
                    return result
            if evidence.peer_ranges != selected_tiles + evidence.peer_excess:
                if reject("peer-range/excess identity did not close"):
                    return result

            if used_fallback:
                result.fallback += 1
                result.fallback_signatures[
                    (q, kt, bpc, selected_tiles, units, selected_groups)
                ] += 1
            result.final_groups[selected_groups] += 1
            result.min_interval = min(result.min_interval, evidence.min_interval)
            result.max_interval = max(result.max_interval, evidence.max_interval)

            reduction, barriers, workspace = workspace_components(
                selected_tiles, config.tm, config.tn
            )
            old_tiles = legacy_two_wave_tiles(q, workers)
            old_workspace = workspace_components(
                old_tiles, config.tm, config.tn
            )[2]
            if workspace > old_workspace:
                if reject("tail-only workspace exceeds unchanged two-wave workspace"):
                    return result
            if not (
                reduction >= selected_tiles * config.tm * config.tn * FP32_BYTES
                and reduction - selected_tiles * config.tm * config.tn * FP32_BYTES < L2_ALIGNMENT
                and barriers >= selected_tiles * LOCK_BYTES
                and barriers - selected_tiles * LOCK_BYTES < L2_ALIGNMENT
            ):
                if reject("workspace/lock alignment bounds failed"):
                    return result

            valid_fixup_elements = (
                evidence.peer_excess * config.tm * config.tn
            )
            result.aggregate.update(
                {
                    "dp_tiles": dp_tiles,
                    "sk_tiles": selected_tiles,
                    "sk_units": units,
                    "qk_cells": evidence.qk_cells,
                    "whole_tiles": evidence.whole_tiles,
                    "split_tiles": evidence.split_tiles,
                    "peer_excess": evidence.peer_excess,
                    "peer_ranges": evidence.peer_ranges,
                    "valid_fixup_elements": valid_fixup_elements,
                    "reduction_bytes": reduction,
                    "barrier_bytes": barriers,
                    "workspace_bytes": workspace,
                    "logical_fixup_rw_bytes": 2 * FP32_BYTES * valid_fixup_elements,
                    "all_sk_tuples": int(dp_tiles == 0),
                }
            )

    if result.attempts != EXPECTED_TUPLES:
        result.add_error(
            f"attempt denominator {result.attempts}, expected {EXPECTED_TUPLES}"
        )
    if result.per_bpc != collections.Counter({bpc: 577 for bpc in range(1, 9)}):
        result.add_error(f"BPC axes do not each contain 577 rows: {result.per_bpc}")

    if fixed_expectations:
        checks = (
            (
                result.preferred_representable == EXPECTED_PREFERRED_REPRESENTABLE,
                "preferred representable census",
            ),
            (
                result.preferred_collapsed == EXPECTED_PREFERRED_COLLAPSED,
                "preferred collapsed census",
            ),
            (
                result.preferred_empty_units == EXPECTED_PREFERRED_EMPTY_UNITS,
                "preferred empty-unit census",
            ),
            (
                dict(result.preferred_empty_histogram)
                == EXPECTED_EMPTY_UNIT_HISTOGRAM,
                "preferred empty-unit histogram",
            ),
            (result.fallback == EXPECTED_PREFERRED_COLLAPSED, "fallback census"),
            (result.rejected == 0, "final rejected census"),
            (
                dict(result.final_groups) == EXPECTED_FINAL_GROUP_CENSUS,
                "final group census",
            ),
            (
                dict(result.fallback_signatures) == EXPECTED_FALLBACK_SIGNATURES,
                "fallback signature census",
            ),
            (
                {key: result.aggregate[key] for key in EXPECTED_AGGREGATE}
                == EXPECTED_AGGREGATE,
                "aggregate DP/SK/workspace census",
            ),
            (
                result.min_interval == MIN_SK_ITERS and result.max_interval == 64,
                "final interval bounds",
            ),
            (
                result.preferred_representable + result.preferred_collapsed
                == EXPECTED_TUPLES,
                "preferred denominator closure",
            ),
            (
                sum(result.final_groups.values()) + result.rejected
                == EXPECTED_TUPLES,
                "final denominator closure",
            ),
        )
        for passed, label in checks:
            if not passed:
                result.add_error(f"{label} changed")
    return result


def anchor_partition(sk_tiles: int, units: int, kt: int, groups: int) -> PartitionEvidence:
    plan = make_boundary_plan(sk_tiles, units, kt, groups)
    if not plan.exact_nonempty:
        raise AssertionError(f"anchor boundary plan is not exact: {plan}")
    evidence = expand_partition(sk_tiles, units, kt, plan)
    if not evidence.exact_nonempty:
        raise AssertionError(f"anchor cell expansion is not exact: {evidence}")
    return evidence


def prove_fixed_anchors() -> None:
    q, workers, kt = 2048, 576, 64
    old_sk = legacy_two_wave_tiles(q, workers)
    tail_sk = tail_tiles(q, workers)
    old_units = sk_units(old_sk, kt, workers)
    tail_units = sk_units(tail_sk, kt, workers)
    old = anchor_partition(old_sk, old_units, kt, 8)
    tail = anchor_partition(tail_sk, tail_units, kt, 8)
    old_expected = (896, 576, 1152, 440, 456, 456, 57_344)
    tail_expected = (320, 576, 1728, 0, 320, 456, 20_480)
    old_got = (
        old_sk,
        old_units,
        q - old_sk,
        old.whole_tiles,
        old.split_tiles,
        old.peer_excess,
        old.qk_cells,
    )
    tail_got = (
        tail_sk,
        tail_units,
        q - tail_sk,
        tail.whole_tiles,
        tail.split_tiles,
        tail.peer_excess,
        tail.qk_cells,
    )
    if old_got != old_expected:
        raise AssertionError(
            f"unchanged two-wave anchor drifted: got={old_got} expected={old_expected}"
        )
    if tail_got != tail_expected:
        raise AssertionError(
            f"tail-only anchor drifted: got={tail_got} expected={tail_expected}"
        )
    if workspace_components(old_sk, 64, 64) != (
        14_680_064,
        3_584,
        14_683_648,
    ):
        raise AssertionError("two-wave anchor workspace drifted")
    if workspace_components(tail_sk, 64, 64) != (
        5_242_880,
        1_280,
        5_244_160,
    ):
        raise AssertionError("tail-only anchor workspace drifted")
    print(
        "[l201 anchor] legacy-two-wave "
        "Q=2048 W=576 Kt=64 G=8 DP=1152 SK=896 U=576 "
        "whole=440 split=456 peer_excess=456 qk=57344 workspace=14683648"
    )
    print(
        "[l201 anchor] tail-only "
        "Q=2048 W=576 Kt=64 G=8 DP=1728 SK=320 U=576 "
        "whole=0 split=320 peer_excess=456 qk=20480 workspace=5244160"
    )


def prove_no_seam_controls() -> None:
    # Exact waves lower to pure DP: no SK units, FP32 scratch, or q locks.
    workers = 576
    for q in (workers, 2 * workers, 4 * workers):
        sk = tail_tiles(q, workers)
        if sk != 0 or sk_units(sk, 64, workers) != 0:
            raise AssertionError(f"exact-wave Q={q} did not lower to pure DP")
        if workspace_components(sk, 64, 64) != (0, 0, 0):
            raise AssertionError(f"exact-wave Q={q} retained workspace or locks")

    # A nonzero tail can also have no peer seam.  Kt==Min gives one whole-K
    # unit for the one residual tile; this is exact but is not split evidence.
    sk = tail_tiles(577, workers)
    units = sk_units(sk, 8, workers)
    plan = make_boundary_plan(sk, units, 8, 1)
    evidence = expand_partition(sk, units, 8, plan)
    if not (
        sk == units == 1
        and evidence.exact_nonempty
        and evidence.whole_tiles == 1
        and evidence.split_tiles == 0
        and evidence.peer_excess == 0
    ):
        raise AssertionError(f"whole-K no-seam control failed: {evidence}")
    print(
        "[l201 no-seam] exact-waves=3 pure-DP/workspace0; "
        "Q=577 W=576 Kt=8 SK-whole=1 split=0 peer_excess=0 exact-once"
    )


def require_negative_control(
    configs: list[Config], policy: AuditPolicy, expected_fragment: str
) -> None:
    planted = audit_domain(
        configs, policy, fixed_expectations=False, fail_fast=True
    )
    if planted.ok:
        raise AssertionError(f"negative control {policy.name!r} stayed green")
    if not any(expected_fragment in error for error in planted.errors):
        raise AssertionError(
            f"negative control {policy.name!r} failed for the wrong reason: "
            f"{planted.errors}"
        )
    print(
        f"[l201 negative] {policy.name}=RED reason={expected_fragment!r}"
    )


def main() -> int:
    source, configs = load_authority()
    prove_fixed_anchors()
    prove_no_seam_controls()

    canonical = AuditPolicy("canonical-tail", tail_tiles)
    result = audit_domain(
        configs, canonical, fixed_expectations=True, fail_fast=False
    )
    if not result.ok:
        detail = "; ".join(result.errors)
        raise AssertionError(
            f"canonical audit has {result.error_count} error(s): {detail}"
        )

    print(
        "[l201 denominator] "
        f"source={len(source)} eligible={len(configs)} filtered={len(source)-len(configs)} "
        f"BPC=1..8 tuples={result.attempts} closure=577*8"
    )
    print(
        "[l201 preferred-G] G=8 "
        f"representable={result.preferred_representable} "
        f"collapsed={result.preferred_collapsed} "
        f"empty_units={result.preferred_empty_units} "
        "outcome=FAIL-CLOSED/not-admitted-without-fallback"
    )
    group_text = ",".join(
        f"{group}:{count}" for group, count in sorted(result.final_groups.items())
    )
    print(
        "[l201 final-groups] "
        f"preferred={result.preferred_representable} fallback={result.fallback} "
        f"rejected={result.rejected} groups={{{group_text}}}"
    )
    print(
        "[l201 exact-once] "
        f"qk_cells={result.aggregate['qk_cells']} missing=0 duplicate=0 "
        f"empty_units=0 interval=[{result.min_interval},{result.max_interval}] "
        f"whole={result.aggregate['whole_tiles']} split={result.aggregate['split_tiles']} "
        f"peer_excess={result.aggregate['peer_excess']}"
    )
    print(
        "[l201 workspace] "
        f"reduction={result.aggregate['reduction_bytes']} "
        f"locks={result.aggregate['barrier_bytes']} "
        f"total={result.aggregate['workspace_bytes']} "
        f"valid_fixup_elements={result.aggregate['valid_fixup_elements']} "
        "q-lock-range=[0,SK) DP-lock-touches=0 bounds=PASS"
    )

    require_negative_control(
        configs,
        AuditPolicy("two-wave-masquerades-as-tail", legacy_two_wave_tiles),
        "tail tile minimality violated",
    )
    require_negative_control(
        configs,
        AuditPolicy(
            "off-by-one-remainder",
            lambda q, workers: q % workers + 1,
        ),
        "tail tile minimality violated",
    )
    require_negative_control(
        configs,
        AuditPolicy(
            "empty-unit-not-rejected",
            tail_tiles,
            use_safe_group_fallback=False,
            admit_collapsed_preferred=True,
        ),
        "empty repaired unit admitted",
    )

    print(
        "[l201] PASS: 1772=577+1195 authority; 4616=4212 preferred+404 "
        "exact-divisor fallback; every admitted (q,k) cell exact-once and "
        "nonempty; legacy two-wave anchor unchanged; negative-controls=3/3_RED"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError) as error:
        print(f"[l201] FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)

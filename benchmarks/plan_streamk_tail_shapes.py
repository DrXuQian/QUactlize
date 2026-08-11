#!/usr/bin/env python3
"""Construct a controlled dense shape plan for measuring Stream-K tail removal.

This is a PLAN, not a benchmark runner.  It deliberately constructs synthetic
output-tile grids whose last-wave waste is known before a device is touched.
Real-model coverage remains the job of ``fixtures.py`` / ``workloads.py``; the
rows here isolate one causal variable that those shapes do not control.

The worker count is an input because the same kernel can report a different
``maximum_active_blocks()`` on a different machine or build.  Every emitted row
therefore carries both Q and W and prints

    tail = ceil(Q / W) * W / Q - 1

next to the shape.  A result without those fields cannot be attributed to a
last wave and is outside this plan's contract.

Example (host-only; does not run a kernel):

    python3 benchmarks/plan_streamk_tail_shapes.py \
        --workers 288 --tile-m 64 --tile-n 128 --k 4096 --gs 128

For W=288 this reconstructs the seven points recorded in INBOX 122:
Q={288,289,320,432,576,1024,1152}, spanning exact waves, a low 12.5% tail,
a medium 33.3% tail, and two extreme tails.  The Q values are derived from W,
not copied into the implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from fractions import Fraction
from typing import Iterable, Sequence


@dataclasses.dataclass(frozen=True)
class TailCase:
    name: str
    target: str
    band: str
    m: int
    n: int
    k: int
    l: int
    gs: int
    tile_m: int
    tile_n: int
    m_tiles: int
    n_tiles: int
    q: int
    workers: int
    waves: int
    last_wave_tiles: int
    tail: Fraction

    @property
    def last_wave_util(self) -> Fraction:
        return Fraction(self.last_wave_tiles, self.workers)


def ceil_div(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        raise ValueError(f"ceil_div requires a>=0,b>0; got {a},{b}")
    return (a + b - 1) // b


def ceil_fraction(value: Fraction) -> int:
    return ceil_div(value.numerator, value.denominator)


def _target_qs(workers: int) -> list[tuple[str, str, int]]:
    """Ratios, not a W=288 lookup table.

    The 10/9 and 32/9 points are the INBOX 122 Q=320 and Q=1024 rows when
    W=288.  Ceil keeps them on the intended side of their wave boundary when
    W is not divisible by nine.
    """
    return [
        ("zero-1wave", "1*W", workers),
        ("extreme-one-past", "W+1", workers + 1),
        ("extreme-sparse-wave", "ceil(10/9*W)", ceil_fraction(Fraction(10 * workers, 9))),
        ("medium-half-wave", "ceil(3/2*W)", ceil_fraction(Fraction(3 * workers, 2))),
        ("zero-2wave", "2*W", 2 * workers),
        ("low-a0-ratio", "ceil(32/9*W)", ceil_fraction(Fraction(32 * workers, 9))),
        ("zero-4wave", "4*W", 4 * workers),
    ]


def tail_for(q: int, workers: int) -> tuple[int, int, Fraction]:
    if q <= 0 or workers <= 0:
        raise ValueError(f"Q and W must be positive; got Q={q}, W={workers}")
    waves = ceil_div(q, workers)
    last_wave_tiles = q - (waves - 1) * workers
    tail = Fraction(waves * workers - q, q)
    return waves, last_wave_tiles, tail


def tail_band(tail: Fraction) -> str:
    if tail == 0:
        return "zero"
    if tail <= Fraction(1, 5):
        return "low"
    if tail <= Fraction(1, 2):
        return "medium"
    return "extreme"


def choose_tile_grid(q: int, tile_m: int, tile_n: int) -> tuple[int, int]:
    """Factor Q into a reasonably square *physical* MxN problem.

    Q itself is the controlled variable.  Factor choice must not change it, so
    every divisor pair is legal.  The score minimises the exact aspect ratio
    max(M,N)/min(M,N), then maximum dimension, then M.  No floating-point
    logarithm is involved in deciding a shape.
    """
    if q <= 0 or tile_m <= 0 or tile_n <= 0:
        raise ValueError("Q and tile dimensions must be positive")
    candidates: list[tuple[Fraction, int, int, int, int]] = []
    for m_tiles in range(1, math.isqrt(q) + 1):
        if q % m_tiles:
            continue
        n_tiles = q // m_tiles
        for mt, nt in ((m_tiles, n_tiles), (n_tiles, m_tiles)):
            m, n = mt * tile_m, nt * tile_n
            aspect = Fraction(max(m, n), min(m, n))
            candidates.append((aspect, max(m, n), m, mt, nt))
    if not candidates:
        raise AssertionError(f"positive Q={q} had no factor pair")
    _, _, _, m_tiles, n_tiles = min(candidates)
    return m_tiles, n_tiles


def build_plan(*, workers: int, tile_m: int, tile_n: int, k: int, gs: int,
               l: int = 1) -> list[TailCase]:
    if workers < 8:
        # Below this, W+1 is no longer an "extreme" tail and the requested
        # three-band experiment is mathematically impossible with this plan.
        raise ValueError(f"workers must be >=8 for the causal bands; got {workers}")
    if min(tile_m, tile_n, k, gs, l) <= 0:
        raise ValueError("tile dimensions, K, group size and L must be positive")
    if l != 1:
        raise ValueError(
            "the controlled dense tail plan currently requires L=1; "
            "do not fold a batch axis into Q without adding it as a separately controlled variable")

    out: list[TailCase] = []
    for name, target, q in _target_qs(workers):
        m_tiles, n_tiles = choose_tile_grid(q, tile_m, tile_n)
        m, n = m_tiles * tile_m, n_tiles * tile_n
        waves, last_wave_tiles, tail = tail_for(q, workers)
        out.append(TailCase(
            name=name, target=target, band=tail_band(tail), m=m, n=n,
            k=k, l=l, gs=gs, tile_m=tile_m, tile_n=tile_n,
            m_tiles=m_tiles, n_tiles=n_tiles, q=q, workers=workers,
            waves=waves, last_wave_tiles=last_wave_tiles, tail=tail))
    validate_plan(out)
    return out


def validate_plan(rows: Sequence[TailCase]) -> None:
    """Fail closed if the plan can no longer answer the tail question."""
    if len(rows) != 7:
        raise ValueError(f"tail plan must have seven causal points, got {len(rows)}")
    names = [r.name for r in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"tail-plan names are not unique: {names}")
    workers = {r.workers for r in rows}
    tiles = {(r.tile_m, r.tile_n) for r in rows}
    if len(workers) != 1 or len(tiles) != 1:
        raise ValueError("one comparison plan must hold workers and tile geometry fixed")

    for r in rows:
        q_from_shape = ceil_div(r.m, r.tile_m) * ceil_div(r.n, r.tile_n) * r.l
        if q_from_shape != r.q or r.m_tiles * r.n_tiles * r.l != r.q:
            raise ValueError(
                f"{r.name}: emitted shape gives Q={q_from_shape}, recorded Q={r.q}")
        waves, last, tail = tail_for(r.q, r.workers)
        if (r.waves, r.last_wave_tiles, r.tail) != (waves, last, tail):
            raise ValueError(f"{r.name}: stale or fabricated wave/tail fields")
        if r.band != tail_band(r.tail):
            raise ValueError(f"{r.name}: band={r.band} disagrees with tail={r.tail}")

    bands = {r.band for r in rows}
    required = {"zero", "low", "medium", "extreme"}
    if not required.issubset(bands):
        raise ValueError(f"tail plan misses causal bands {sorted(required - bands)}")
    if sum(r.tail == 0 for r in rows) < 3:
        raise ValueError("tail plan needs exact-wave controls at 1, 2 and 4 waves")
    if sum(r.tail > Fraction(1, 2) for r in rows) < 2:
        raise ValueError("tail plan needs both one-past and sparse-wave extreme controls")
    if all(r.q % r.workers == 0 for r in rows):
        raise ValueError("all planned Q divide W; this is the silent zero-tail sweep")


TSV_FIELDS = (
    "case", "target", "band", "M", "N", "K", "L", "gs", "TM", "TN",
    "m_tiles", "n_tiles", "Q", "W", "waves", "last_wave_tiles",
    "last_wave_util_pct", "tail_pct",
)


def row_dict(r: TailCase) -> dict[str, object]:
    return {
        "case": r.name, "target": r.target, "band": r.band,
        "M": r.m, "N": r.n, "K": r.k, "L": r.l, "gs": r.gs,
        "TM": r.tile_m, "TN": r.tile_n,
        "m_tiles": r.m_tiles, "n_tiles": r.n_tiles,
        "Q": r.q, "W": r.workers, "waves": r.waves,
        "last_wave_tiles": r.last_wave_tiles,
        "last_wave_util_pct": f"{100.0 * float(r.last_wave_util):.6f}",
        "tail_pct": f"{100.0 * float(r.tail):.6f}",
    }


def render_tsv(rows: Iterable[TailCase]) -> str:
    lines = ["\t".join(TSV_FIELDS)]
    for r in rows:
        d = row_dict(r)
        lines.append("\t".join(str(d[k]) for k in TSV_FIELDS))
    return "\n".join(lines) + "\n"


def render_jsonl(rows: Iterable[TailCase]) -> str:
    return "".join(json.dumps(row_dict(r), sort_keys=True) + "\n" for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, required=True,
                    help="runtime W = CU count * maximum_active_blocks() for this exact kernel")
    ap.add_argument("--tile-m", type=int, required=True)
    ap.add_argument("--tile-n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True,
                    help="K for the controlled scan; explicit so this tool cannot invent a model shape")
    ap.add_argument("--gs", type=int, required=True,
                    help="group size for the future run; explicit, not inferred from a format name")
    ap.add_argument("--l", type=int, default=1)
    ap.add_argument("--format", choices=("tsv", "jsonl"), default="tsv")
    a = ap.parse_args()
    try:
        rows = build_plan(workers=a.workers, tile_m=a.tile_m, tile_n=a.tile_n,
                          k=a.k, gs=a.gs, l=a.l)
    except ValueError as e:
        ap.error(str(e))
    print(render_tsv(rows) if a.format == "tsv" else render_jsonl(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Partitioning K: what Marlin does, what actlize does, and what the measurements actually settle

Written 2026-08-11, at the point where the question "should we port Marlin's scheduler into actlize" was answered
**no**. The evidence for that answer lived in a conversation and in a comment in a file outside this repo
(`marlin_classic_ppu.cuh`), so it is written down here instead. The numbers matter more than the verdict: a later
change of tactic, shape or hardware can flip the verdict, and then these numbers are what makes the flip legible
rather than a reversal of opinion.

## Four ways to cut the same work

The anchor throughout is the **S068 decode shape**: `Q = 128` output tiles, `Kt = 8` k-tiles per output tile,
`G = 432` workers. Total unit work `T = Q·Kt = 1024`.

*Classic Marlin ceil stripe*: flatten `(output tile, k-tile)` and cut into equal stripes.
`I = ceil(T/G) = 3` units per CTA, so `A = ceil(T/I) = 342` CTAs are active and 90 are idle. The 342 CTAs create
341 internal stripe boundaries; `floor(341/8) = 42` of them happen to land exactly on an output-tile boundary and
need no handoff, giving **299 handoffs**.

| strategy | active units | machine capacity used | peer excess (handoffs) |
|---|---:|---:|---:|
| uniform split-K, S=3 | 384 | 88.89% | **256** |
| classic Marlin ceil stripe | 342 | 79.17% | 299 |
| **actlize Min2 StreamK** | **432** | **100%** | 304 |
| uniform split-K, S=4 | 512 (two waves) | — | 384 |

Two things to read off this table, both of which contradict something that was believed before it was computed:

* **The ordering "Marlin's stripe is finer, so it needs fewer reduces" is false.** A classic stripe can straddle an
  output-tile boundary, which makes that CTA a peer of **two** tiles. Counting `units − Q` assumes one tile per CTA
  and undercounts (it gives 214; the truth is 299). Finer steps do not imply fewer handoffs, and `Q | W` is not the
  condition for zero handoffs either — the real condition also involves `I` and the `Kt` alignment.
* **"Split-K always has more global reduces than Marlin" is false at this anchor.** Uniform `S=3` beats classic on
  *both* axes at once: more active workers (384 vs 342) and fewer handoffs (256 vs 299).

actlize's Min2 StreamK uses the whole machine and pays 5 more handoffs than classic. Trading 90 idle CTAs for 5
handoffs is not a trade worth reversing, which is the whole case against building the exact Marlin stripe.

## The 33× curve, and what it does *not* prove

Measured on this box with Marlin classic, decode `M=1, N=4096`, holding the shape fixed and overriding only the grid
(`MARLIN_BLOCKS`). Recorded at `marlin_classic_ppu.cuh:859`:

```
  blocks      72     144     288     576    1024    3584
  slice_cnt    3       4       8      16      32     112    <- barrier_acquire chain length per output tile
  K=4096    17.8    18.4    47.5   176.2   588.1     --  us
  K=14336   27.1    28.7    87.7   182.2   474.8  7960.6 us
```

Monotone, and `t ~ slice_count^1.5`. This is why Marlin's launcher is `blocks = (tiles >= sms) ? tiles : sms`
(`:887`) — a line that reads like a heuristic and is really a swerve around this curve.

**The mechanism is specific to Marlin's reduce, and that limits what the curve settles.** Marlin's `global_reduce`
is a serial chain: `barrier_acquire(&locks[slice_col], slice_idx)` admits peers strictly in order `0,1,…,S−1`, so a
tile's reduce costs `S` serialized steps, and only `n_tiles` locks exist so more CTAs also means more spinning on
the same lock. The linear term is the chain, the superlinear term is the contention.

So the curve **does** bound the cost of porting Marlin's stripe decomposition together with Marlin's reduce, and it
does show that split-K's useful parallelism here is exhausted by `slice_count` 3–4. It does **not** establish that
K-splitting is catastrophic on this hardware in general — actlize's own uniform split-K is the control and shows
nothing like it, because its fixup is `store → atomic_add → load_add`, not an ordered lock chain.

It also does not settle the reduce-*traffic* question either way, and it is worth being explicit about why, because
the comment's own words invite the mistake: it says *"NOT a traffic problem (M=1 writes 128 halves per slice)"*.
That is true **of Marlin at M=1**, where a partial is 256 B. actlize's full-FP32 fixup moves 2048 B per peer, 8×
more, and at a different `peer_excess`. Different regime; the curve is silent on it.

## Reduce traffic: three tiers, and the 32× correction

At `peer_excess = 304`, with the deterministic seam `C = 2W(S−1) + D` (each handoff is a write *and* a read):

| fixup storage | bytes per peer | extra traffic | modelled increment |
|---|---:|---:|---:|
| full FP32 tile (16×32×4) | 2048 | 1,245,184 B | **+26.162%** |
| valid elements, FP32 (1×32×4) | 128 | 77,824 B | **+1.635%** |
| valid elements, fp16 (1×32×2) | 64 | 38,912 B | +0.818% |

The three rows are in exact 16 : 1 : 0.5 ratio, which is the check that they are the same quantity measured three
ways rather than three separate estimates. The percentages are against the S068 traffic model's baseline, which
these numbers **imply** to be ~4.76 MB rather than state — if that model changes, recompute the percentages from
the byte column, which is the part that stands on its own.

The middle row is the whole finding. An earlier claim that Marlin's fp16 reduce buys **+14%** was wrong by a factor
of 32, because Marlin guards its store with `if (r < prob_m)` — at `Mvalid = 1` its partial is a *row*, not a padded
tile. Once the valid-element guard is applied, **FP32 already captures 94% of the available saving** (26.162 → 1.635)
and the remaining 0.8 points is what fp16 would buy at the cost of precision and a C/D aliasing constraint. That is
why the recommended single optimisation is FP32-lite and not the fp16 C-chain: the guard is the mechanism, the
narrower type was never the point.

Caveat on the middle row: it counts *logical* accesses. If the valid fragment is scattered across cache lines the
DRAM saving will be smaller than 16×, and only a device counter can say by how much.

## What is actually upstream of all of this

For decode, the scheduler is not the binding constraint and neither is the reduce. `M <= 16` selects
`thread_n = 128`, so only `N/128` output tiles exist at all — **32 tiles for N=4096 against 72 CUs (0.44×)**. The
same kernel at N=14336 has 112 tiles and reaches 65.7% of HBM; at N=4096 it reaches 17.5%. Marlin's tile is 16×128
and decode uses one of those 16 rows.

This is the strongest argument in the document against porting the scheduler: a better partition of work cannot
create tiles that the tile shape does not admit. It also names the axis that can — the one mechanism Marlin has that
we do not is **warp-K**, splitting K *inside* a CTA. That path never increments `slice_count`, creates no peer,
takes no lock, and therefore does not enter the curve above; it raises residency (`Q=128` at 2 warps/CTA is ~3.56
work-warps per CU of 64 available) without paying a single handoff. Whether the register and shared-memory cost
lets that residency materialise is a device question, and `Gemm::maximum_active_blocks()` is the arbiter.

## What would overturn each conclusion

* **"Don't build the exact Marlin stripe."** Overturned if actlize's StreamK stops reaching 100% capacity at the
  shapes we care about, or if handoff cost per peer rises enough that 304 vs 299 stops being noise. Recompute the
  table; it is four lines of arithmetic.
* **"FP32-lite over fp16."** Overturned if a device counter shows the valid-element fragment is scattered badly
  enough that the logical 16× collapses — then the type change is back on the table because it also compacts.
* **"Warp-K is the axis Marlin has and we don't."** Overturned if the offline artifact turns out to depend on WK.
  Then WK is not a tactic axis but a *format* axis, and its cost is a different order entirely — the same shape of
  mistake as assuming one weight file serves every TileK, which held only for single-plane `F=1, T<=256`.

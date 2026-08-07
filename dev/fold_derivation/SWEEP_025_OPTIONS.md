# Sweep 025: dense and grouped arrangement/tactic space

## 029 answer

The answer is **(a)**. The two figures labelled

    dense int1 TK64/F4  215.23 us
    dense int2 TK64/F2  233.76 us

are recorded in `HANDOFF_TASK12.md` under `test_q3_bconcat_bench`. That source includes
`moe_grouped_ppu.cuh`, and both its `I1F` and `I2F` macros call `moe_grouped_ppu::filter_and_run`; it never calls
`fpA_intB_ppu`. `L=1 dense` described the problem shape, not the implementation. Before 029, those arrangements
were grouped-reachable and fpA-dense-unreachable.

Dense now copies grouped's mechanism: the low-plane fold is derived for single- and two-plane formats, the physical
stride is `(N/F, F*K)`, and a sub-32-byte run wraps the group-size schedule in `KernelAiuFold<F, BaseSchedule>`.
Representative folded rows for all five k-quant schemas pass the local full-instantiation nvcc gate. This is compile
evidence; it is not a device result.

## Finite domain

`emit_tactic_space.cpp` emits the Cartesian product below once from `TacticSpace`. `DenseSpace` and `GroupedSpace`
are compatibility aliases of that type, so a second emitted copy would compare the generator with itself. Every cell
is printed as `reachable` or `excluded: <one clause>`.

| axis | complete values |
|---|---|
| bits/schema | `i1`, `i2`, `i4`, `Q3_K=2+1`, `Q5_K=4+1`, `Q6_K=4+2` |
| TileK | `32, 64, 128, 256` |
| TileM | `16, 32, 64, 128, 256` |
| TileN | `16, 32, 64, 128` |
| WarpM | `16, 32, 64` |
| WarpN | `16, 32, 64, 128` |

That is 5,760 cells per operator. `WN=128` is present deliberately: it is required by the delivery predicate for an
int1 plane at TK32. Today those otherwise-legal rows say why they are unreachable instead of disappearing.

F is not an axis. It is always the minimum delivery fold

    RUN = TK * bits / 8 bytes
    F   = RUN >= 32 ? 1 : 32 / RUN

and each plane of a two-plane format derives its own value:

| bits | TK32 | TK64 | TK128 | TK256 |
|---:|---:|---:|---:|---:|
| 1 | 8 | 4 | 2 | 1 |
| 2 | 4 | 2 | 1 | 1 |
| 4 | 2 | 1 | 1 | 1 |

Over-folds are not emitted. They can be constructed offline, but neither launcher selects them because both derive the
minimum F from `(bits,TK)`.

Stage is not a stored-arrangement field. The user-set timing axis is `{2,3,4}`; split-K stays out. Stage 2 remains the
ordinary-path existence test in the arrangement emitter: if the conservative footprint cannot fit at the shallowest
supported pipeline, no deeper stage can make that ordinary topology legal. A timing command must enumerate all three
stage values explicitly.

Small-M compact A makes reachability depend on A-row capacity. The complete n-token axis is
`{1,2,4,64,2048,4096}`. Ordinary builds allocate `TileM*TileK*2` A bytes per stage. Compact capacities 1, 2, and 4
allocate `capacity*TileK*2` and are three different collective types, not three timings of one binary. For dense,
capacity equals M at 1/2/4. For grouped it is per-expert Mmax, not global tokens: both target MoE models have 256
experts/top-8. The pinned `token-topk-hot16x4-wor-sm64-s44-v1` router produces Mmax 1/2/3 at global tokens 1/2/4,
so those points require compact capacities 1/2/4. It produces Mmax 12/239/447 at tokens 64/2048/4096, so all three
prefill points require the ordinary path. The benchmark prints this decision and refuses a compact build whose
capacity is below fixture Mmax; it never widens or falls back silently.

At this checkpoint compact A is reachable only for unfolded one-plane formats:

| schema | compact-reachable TileK |
|---|---|
| i1 | `256/F1` |
| i2 | `128/F1`, `256/F1` |
| i4 | `64/F1`, `128/F1`, `256/F1` |
| Q3_K, Q5_K, Q6_K | none: the two-plane collective has no compact-A reader |

Folded i1/i2/i4 cells likewise remain ordinary-only because the fold collective has no compact-A reader. Defining
`PPU_A_CPASYNC` is not reachability: every collective exposes a type-level row-capacity witness, and the launcher
rejects folded/two-plane builds whose witness is zero. This is a current implementation exclusion, not a claim that
the port is impossible.

## One source of legality

The host-readable rules and axes live in `quactlize/include/ppu_tactic_space.hpp`.

- `fpA_intB_ppu.cuh` names the public `DenseSpace` alias and static-asserts its kernel constraints.
- `moe_grouped_ppu.cuh` names the public `GroupedSpace` alias and static-asserts its kernel constraints.
- `lowbit_moe_bench.hpp::moe_ok` asks that same generator instead of carrying another predicate.
- `DenseSpace` and `GroupedSpace` are type-identical by static assertion. The local gate rejects a planted independent
  grouped type and compares both emitter routes over every argument set declared by the 13 shipping tables.

The ordered exclusion predicate is:

1. MMA-atom alignment;
2. warp divides tile;
3. at most 32 warps/block;
4. both derived folds divide TileN;
5. per-plane delivery fits the warp fragment (`WN*TK*bits >= 4096`);
6. fp32 accumulator at most the sweep's 192-register ceiling;
7. conservative gs16 scale+zero shared memory fits at stage 2;
8. offline producer uses a consumer-validated `WN <= 64`;
9. Q6/TK256 is excluded because its two-plane inverse is incomplete.

For the compact inventory, item 7 is recomputed at each stage using the compact row capacity where the selected
collective supports it. Otherwise the ordinary TileM footprint remains in force, with the one-clause reason
“selected folded or two-plane collective has no compact-A reader.”

“Reachable” means the consumer can select the derived schedule, an exact-geometry offline placement is available in the
current tree, and the cell clears those checks. It does **not** mean every cell was device-verified, nor that an arbitrary
shape can be substituted for the canonical `tools/pack_gguf.py` artifact without checking byte equivalence.

## Emitted result at this checkpoint

Build and emit with no device SDK:

    c++ -std=c++17 -Iquactlize/include \
      dev/fold_derivation/emit_tactic_space.cpp -o /tmp/quactlize_emit_tactics
    /tmp/quactlize_emit_tactics > /tmp/quactlize_tactic_space.txt

The shared generator currently emits:

| schema | reachable TK/F cells (number of tile/warp shapes) |
|---|---|
| i1 | `64/F4: 24`, `128/F2: 53`, `256/F1: 59`; `32/F8: 0` (needs WN128, producer cap is 64) |
| i2 | `32/F4: 24`, `64/F2: 59`, `128/F1: 97`, `256/F1: 59` |
| i4 | `32/F2: 59`, `64/F1: 103`, `128/F1: 97`, `256/F1: 59` |
| Q3_K | `64/F2/F4: 24`, `128/F1/F2: 53`, `256/F1/F1: 59`; TK32 has no reachable int1-plane row |
| Q5_K | `64/F1/F4: 24`, `128/F1/F2: 53`, `256/F1/F1: 59`; TK32 has no reachable int1-plane row |
| Q6_K | `32/F2/F4: 24`, `64/F1/F2: 59`, `128/F1/F1: 97`; TK256 is the incomplete inverse |

Totals: **5,760 emitted, 1,145 reachable, 4,615 excluded with a clause**. Both table routes consume this one result;
only the durable dense/grouped table label, macro prefix, and output filename differ.

That emitter describes the ordinary arrangement/tactic domain. `benchmarks/size_sweep.cpp` adds the stage and compact
build axis from the same predicate. Its current `{2,3,4}` scope reports **3,190 ordinary builds** (1,145 / 1,131 /
914 by stage). Compact row-capacity specialisations are not enumerated by that host tool, so they must not be added
to this number by multiplying an obsolete per-stage count. This is a sizing result, not a box request; the guarded
rules in `SWEEP_032_PRUNING_CODEX.md` must reduce it first.

No ppu001 sweep should use the old `test_fpA_intB_ppu` command as a completeness claim: that binary remains an int4,
hand-curated tactic list. The emitted inventory is the coverage contract the eventual box command must name and compare
before timing any subset.

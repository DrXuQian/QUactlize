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

`emit_tactic_space.cpp` emits the Cartesian product below twice, once by asking `DenseSpace` and once by asking
`GroupedSpace`. Every cell is printed as `reachable` or `excluded: <one clause>`.

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

Stage and split-K are not stored-arrangement fields and therefore are not in the requested tuple. Stage 2 is the
existence test: if the conservative gs16 scale+zero footprint cannot fit at the shallowest supported pipeline, no deeper
stage can make the topology legal. A timing command must still enumerate its stage and dense split-K axes explicitly.

## One source of legality

The host-readable rules and axes live in `quactlize/include/ppu_tactic_space.hpp`.

- `fpA_intB_ppu.cuh` names `DenseSpace` and static-asserts its kernel constraints.
- `moe_grouped_ppu.cuh` names `GroupedSpace` and static-asserts its kernel constraints.
- `lowbit_moe_bench.hpp::moe_ok` asks `GroupedSpace` instead of carrying another copy of the predicate.
- `emit_tactic_space.cpp` asks the two wrappers independently. A future difference therefore appears in emitted output
  and can be justified or rejected; it cannot silently shrink a sweep.

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

“Reachable” means the consumer can select the derived schedule, an exact-geometry offline placement is available in the
current tree, and the cell clears those checks. It does **not** mean every cell was device-verified, nor that an arbitrary
shape can be substituted for the canonical `tools/pack_gguf.py` artifact without checking byte equivalence.

## Emitted result at this checkpoint

Build and emit with no device SDK:

    c++ -std=c++17 -Iquactlize/include \
      dev/fold_derivation/emit_tactic_space.cpp -o /tmp/quactlize_emit_tactics
    /tmp/quactlize_emit_tactics > /tmp/quactlize_tactic_space.txt
    python3 ci/tactic_space.py /tmp/quactlize_tactic_space.txt

Both sides currently emit:

| schema | reachable TK/F cells (number of tile/warp shapes) |
|---|---|
| i1 | `64/F4: 24`, `128/F2: 59`, `256/F1: 80`; `32/F8: 0` (needs WN128, producer cap is 64) |
| i2 | `32/F4: 24`, `64/F2: 59`, `128/F1: 103`, `256/F1: 80` |
| i4 | `32/F2: 59`, `64/F1: 103`, `128/F1: 103`, `256/F1: 80` |
| Q3_K | `64/F2/F4: 24`, `128/F1/F2: 59`, `256/F1/F1: 80`; TK32 has no reachable int1-plane row |
| Q5_K | `64/F1/F4: 24`, `128/F1/F2: 59`, `256/F1/F1: 80`; TK32 has no reachable int1-plane row |
| Q6_K | `32/F2/F4: 24`, `64/F1/F2: 59`, `128/F1/F1: 103`; TK256 is the incomplete inverse |

Totals per operator: **5,760 emitted, 1,286 reachable, 4,474 excluded with a clause**. After replacing only the
operator prefix, dense and grouped output is byte-identical at this checkpoint; `ci/tactic_space.py` reports
`dense 5760 configs, grouped 5760 -> COINCIDE`.

No ppu001 sweep should use the old `test_fpA_intB_ppu` command as a completeness claim: that binary remains an int4,
hand-curated tactic list. The emitted inventory is the coverage contract the eventual box command must name and compare
before timing any subset.

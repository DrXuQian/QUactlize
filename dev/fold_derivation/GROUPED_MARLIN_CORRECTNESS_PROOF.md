# Grouped Marlin correctness boundary

The grouped kernel does not reinterpret ragged MoE as one uniform `M`.  The
host builds

```text
P[0] = 0
P[e+1] = P[e] + ceil(M[e]/TM) * ceil(N/TN)
Q = P[E]
```

and lowers the already-proved Marlin scheduler through the synthetic shape
`(Q*TM, TN, K, 1)`.  Therefore scheduler `M_idx` is the globally unique output
tile `q`; only the mainloop/epilogue decode `q` through `P` to an expert-local
`(m,n,e)`.  The cooperative still receives the original `sched_work`, so its
workspace and lock IDs remain global `q` and cannot alias across experts.

Zero-row experts make `P[e+1] == P[e]`.  Production and the oracle call the
same `GroupedRaggedOutputTiles` helper: `upper_bound(P,q)-1` skips every
repeated prefix entry, so an empty expert creates neither work nor a hole in
the global numbering.

## Proposition A: exhaustive composition

`ci/check_grouped_marlin_exhaustive.py` derives its authority rather than
transcribing rows:

- all 9,138 rows in the ten committed grouped tables (prefill/decode for i4,
  i2, Q3_K, Q5_K and Q6_K), collapsed only after production lowering to 85
  distinct `(TM,TN,TK)` geometries;
- all 36 declared MoE projection/token rows from `workloads.py` and
  `fixtures.py`, collapsed only after the pinned router to 24 shapes;
- both observed PPU CU counts, 32 and 72.

This is 657,936 raw table/shape/CU tuples and 1,790 distinct production
parameter sets.  L136 exhausts every resulting work segment and all
22,257,024 `(q,k_tile)` cells.  The measured host proof census is:

```text
raw=657936/657936 remaining=0 unique=1790/1790 remaining=0
groups=458240 zero-groups=91542 outputs=1549448
segments=1559783 logical-(q,k)=22257024 handoffs=10335
cross-group=1490
Q>=CU classes=1448 handoffs=0
```

Every cell is visited exactly once, every output has one globally unique lock,
zero experts receive no `q`, and the peer protocol is checked in reverse
Marlin order.  For every `Q>=CU` class, `G=Q`, `I=Kt`, and handoffs are exactly
zero.  The fixture contribution is an integer in `{-1,0,1}` with at most the
declared K-tile count, so reassociation cannot change it in FP32 or FP16; the
criterion is raw equality fixed before the run.  Six independently causal
plants (zero expert consumes a tile, wrong prefix comparison, wrong expert
decode, occupancy-multiplied grid, floor stripe, local lock) all fail.

## Collective coverage and sweep tables

Scheduling is format-neutral, but legality and resident artifacts are not.
The grouped sweep must therefore remain split by format and band: the existing
ten i4/i2/Q3_K/Q5_K/Q6_K prefill/decode tables remain the authority.  Marlin
does not create a second copy of those rows.

L135 instantiates the same grouped Marlin kernel with:

- ordinary one-plane int4 (`F=1`),
- folded one-plane int2 (`F=2`), and
- two-plane Q3 (`int2 + int1`, high-plane `F=4`).

All three use the shared `MainloopPolicy`; converter, scale/zero, fold,
B-chunk and two-plane logic are not present in the scheduler/kernel wrapper.
The type compile and its contract plants are local evidence only, not a PPU
execution claim.

The only correctness property still requiring a device is the named-barrier
and memory-order protocol under concurrent peers.  No device result is claimed
by this proof.

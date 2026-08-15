# Dense fixed Split-K parallel

## Objective

For a dense `M=1` problem, internal Split-K must expose the same independent
work geometry as the externally reshaped `(N*S, K/S)` experiment without
changing the weight artifact.  Version 1 therefore keeps the shipping
mixed-input collective byte-for-byte and changes only scheduling and output:

1. `Q*S` CTAs consume disjoint, contiguous absolute K-tile ranges.
2. Every CTA writes one compact FP32 partial plane.
3. A second kernel adds planes in fixed `peer_idx` order and converts to FP16
   exactly once.

The measured success criterion is end-to-end GEMM plus reduction time.  A GEMM
time that omits the reduction is not a result.

## Version 1 contract

- `S` is one of `1,2,4,8`.
- `Kt % S == 0`; ragged K is rejected until its pipeline tail is proved.
- Work order is q-major and peer-fast.  The descriptor units are K tiles and
  FP32 elements, never bytes or sub-byte logical codes.
- Partial slot `(q, peer)` is unique.  The physical compact output is
  `[S][M][N]`, so the reducer reads `partial[s*M*N + m*N + n]`.
- `S==1` delegates to the historical shipping launcher and kernel type.  It is
  not reconstructed through the new path.
- The main kernel has no semaphore, atomic, D read, or final epilogue.  Stream
  ordering is the only producer/consumer edge between the two kernels.

The deliberately simple two-kernel implementation is both the first usable
path and the permanent arithmetic oracle for a later fused path.

## Why this is a separate thin kernel

CUTLASS 2.x `GemmSplitKParallel` supplies the right protocol shape (contiguous
K slices, split-major FP32 workspace, then a reducer), but its producer is tied
to legacy `Mma::IteratorA/B`.  It cannot consume the shipping C3 mixed
`load_init()` tuple carrying B/S/Z/B2, fold/xplane, and packed-A semantics.
Likewise, the public `kGemmSplitKParallel` enum does not make a C3
`GemmUniversalAdapter` launch a reducer.

The existing Stream-K kernel does reuse the right collective, but its dynamic
work units, q locks, `BlockStripedReduce`, and final-peer epilogue are exactly
the semantics fixed Split-K is replacing.  Turning it into independent planes
would change more code than a thin producer shell.  Version 1 therefore reuses
only the shipping collective and the useful CUTLASS 2.x two-launch protocol;
it does not copy or modify the converter, loader, or mainloop.

Encoding the slices as a uniform grouped/MoE GEMM was also rejected for v1.
It can manufacture the same arithmetic partition, but introduces pointer
arrays, group-prefix decoding, zero/ragged-group rules, and a different public
ABI merely to describe one dense matrix.  It also cannot preserve the required
S==1 type identity.  That route is useful as an independent geometry model,
not as the minimum production change.

## Future fused fixup

The fused protocol must not appoint a particular peer as reducer.  In
particular, giving the last logical peer less GEMM work does not make it the
last physical arrival; appointing it would require polling and can deadlock or
waste a resident CTA.

Instead, every output tile owns:

- `peer_count` unique FP32 partial slots;
- one completion counter initially zero;
- descriptors carrying `q`, absolute `k_begin`, `k_count`, `peer_idx`, and
  `peer_count`.

Each CTA follows this protocol:

1. Compute its possibly non-uniform absolute K interval.
2. Store its complete FP32 partial into slot `(q, peer_idx)`.
3. Synchronize the CTA so the elected arrival thread follows all partial
   stores, then publish them with release ordering.
4. Atomically increment `done[q]` once.
5. If the returned old value is not `peer_count-1`, retire immediately.
6. The actual last arriver acquires the published partials, adds slots in
   increasing `peer_idx` order with the same reduction primitive as version 1,
   and runs the final epilogue.

There is no inter-CTA spin loop.  Completion order chooses *who* reduces, while
`peer_idx` chooses the numerical order, so scheduling cannot change the bits.
A short final interval may be launched late as a performance bias, but it is
never a correctness owner.

Non-uniform scheduling is a separate policy over the same `Work` fields.  It
may distribute `Kt % S` one-tile remainders or use a Stream-K-style linear
interval, provided the host oracle proves exact-once `(q,k_tile)` coverage and
the device consumes the same descriptor.  It must not duplicate the K
arithmetic in the epilogue.

Counter reuse requires an explicit lifecycle.  The first implementation should
zero counters before launch; generation-tagged counters are a later launch-cost
optimization, not part of the correctness protocol.

## Required proofs

Local proofs:

- exhaustive exact-once `(q,k_tile)` coverage and unique partial slots;
- `S==1` shipping type identity;
- exact caller-supplied production mainloop-type reuse for `S>1`;
- workspace size/stride overflow rejection;
- fixed-order reduction, FP16 tail predication, and same-stream launch order;
- planted errors in split count, K start/count, slot identity, order, and
  workspace size must all fail.

Device postconditions:

- exact/ordered fixture correctness for every admitted `S`;
- end-to-end latency, plus main and reducer launch times separately;
- repeated output fingerprints;
- for the fused version, completion-counter reset/reuse and release/acquire
  visibility under repeated launches.

## First device canary (registered before measurement)

`test_lowbit_dense_splitk_parallel` fixes one explicit M==1 packed-A proof row
and one exact, order-independent `M=1,N=K=4096,gs=128` fixture.  It runs
`S=1,2,4,8`; all four arms must match the same host golden in raw FP16 bits on
eight separately poisoned launches and retain the workspace and output
redzones.  Its end-to-end event time for `S>1` contains both launches.

This row is **not** registered as the winner behind the historical ~17 us
measurement.  That measurement did not preserve the complete config string or
kernel identity in the repository, and current M==1 production dispatch also
selects a packed-A provider.  Treating an arbitrary TK128 proof row as that
winner would make a failed admission look like a performance result.  The
canary therefore prints `PERF-UNADJUDICATED`; its warm, single-artifact timings
are observations only.

The target performance experiment becomes admissible only after it binds all
of the following in one artifact bundle:

- the exact independently swept winner's kernel and full config string;
- the same M==1 provider used by that production route;
- a cold-cache rotation matching the ~17 us baseline conditions;
- S==1 within 3% of that same-binary baseline;
- end-to-end, producer, and reducer launch times.

Once those are bound, the externally reshaped `(32768,512)` result (~7.4 us)
remains a geometry proxy rather than a promised Split-K result: the internal
path also pays one reduction launch and FP32 partial traffic.  The hard success
condition is that a correct `S>1` end-to-end row beats its admitted S==1
control.  A plausible target remains 8.5--12 us (about 1.4--2.0x), but no split
count becomes a default until that PPU curve is measured.

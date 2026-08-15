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
- exact shipping mainloop reuse for `S>1`;
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

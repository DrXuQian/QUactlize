# Q4_K A32 transposed WN16 reader: static feasibility analysis

## Decision

A transposed physical B format **can** remove the coupling between
`ArtifactTileK=32` and the PPU001 non-transposed AIU minimum K run.  The
constraint is not eliminated; it moves to N, where a WN16 warp already owns a
32-byte run when the transport element is an opaque b16 word.

This is not an alternate reader for the existing `xplane-q4k-fold2-a32`
bytes.  It is a new physical byte class with K-packed b16 words and N as the
contiguous AIU dimension.  It is statically feasible and has no inherent
extra byte or instruction-count term relative to the successful A64/F1 WN16
reader, but it needs a new project-owned operand/format contract and one PPU
correctness closure before it can be admitted.

## Why transpose changes the useful constraint

For the current non-transposed Q4 operand, the AIU contiguous span is K:

```text
bytes_per_run = ArtifactTileK * 4 / 8
A32            = 16 B                         (too small)
A64            = 32 B                         (legal)
```

The A32 format therefore folds two N columns into one 32-byte run.  Its
logical `N64 x K32` tile is physically `N32 x K64`; this is the contract
recorded by `l227_q4_a32_packed_decode_type.cu`.  One four-register swizzle
delivery then represents `N32 x K32` Q4 codes, which is why the proven
consumer family requires WN at least 32.  `l210_q4_a32_consumer_layout.cu`
records the exact compatibility rule and retains WN16 as a negative.

For a transposed transport, pack four consecutive K codes belonging to one N
column into an opaque b16 word:

```text
word[kg,n] = q[4*kg+0,n]
           | q[4*kg+1,n] << 4
           | q[4*kg+2,n] << 8
           | q[4*kg+3,n] << 12
```

The b16 tensor has logical shape `(N, K/4)` with N contiguous.  A WN16 warp
therefore supplies exactly the minimum AIU run:

```text
16 N columns * 2 B = 32 B
```

The K extent is no longer required to make a 32-byte contiguous run.  A32
remains a serialization/metadata boundary, while one hardware delivery is
allowed to span two adjacent A32 chunks.

## Exact payload ledger

PPU001's existing transposed b16 reader is:

```text
ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16
```

It produces four b32 registers per lane:

```text
32 lanes * 4 registers * 4 B = 512 B
```

With four Q4 codes in each opaque b16 word, one `16 x 16` b16 cube represents:

```text
16 N * 16 K-groups * 4 Q4/K-group = N16 x K64 Q4 codes
16 * 16 * 2 B                     = 512 B
```

This is exactly one WN16 warp's B payload for K64.  It spans two A32 artifact
chunks but does not read padding or duplicate weight bytes.  For a TK256
tactic, four such K64 deliveries cover the tactic K span.

The current Q4 converter also consumes four b32 source registers per lane and
emits 32 fp16 values per lane.  Across a warp this is 1,024 Q4 codes, again
exactly `N16 x K64`.  The payload sizes therefore compose without an
additional shared load, register payload, or conversion pass.

## What actlize already provides

The relevant implementation facts are:

1. `copy_ppu0010_aiu.hpp` implements `PPU0010_AIU_LOAD<..., Element, true>`
   for b16.  The dense fp16 unit suite enables B RowMajor/`TransB=true`, so
   this producer path is not merely dead syntax.
2. The same file implements the matched b16
   `m16n16.x1.swzl.trans.shared.b16` reader.
3. `DefaultGemm_AIU_Operand<..., true>` in `gemm_operands.hpp` makes N the
   contiguous dimension and pairs the transposed AIU write with the
   transposed swizzle read.
4. The Q4 mixed-input builder currently has only `Trans=false`
   specializations.  It uses a b8 shadow and the non-transposed
   `m8n8.x4.swzl.shared.b16` reader.
5. Although the PPU001 AIU producer has a b8 `Trans=true` spelling, the
   generic b8 transposed TSM reader deliberately ends in
   `CUTE_INVALID_CONTROL_PATH`.  The s8 B-transposed AIU unit test is disabled.

Consequently the safe reuse is a **matched b16 write/read pair used as opaque
bits**.  Pairing a b8 swizzle writer with the b16 transposed reader is not
admissible: the write/read cube geometry is part of the swizzle cancellation
contract.

## What cannot be reused unchanged

The four-register size match is necessary but not sufficient.

### Register emission map

`MixGemmEmit<4>` describes the current non-transposed x4 delivery.  The
transposed b16 instruction has a different `(lane, vreg) -> (n, k-group)`
association.  Either:

- the new offline writer must place words using the inverse of the composed
  transposed delivery plus the existing converter; or
- a transposed Q4 emission layout must convert the delivered registers into
  the ordinary MMA B fragment.

This must be expressed as a CuTe layout and proved bijective.  Reusing the
existing descriptive `Copy_Traits<PPU0010_TSM_LD_SWZL>::LogicalTV` is invalid:
its own comment derives it specifically from the non-transposed simulator.

### Scale and zero ownership

One transposed delivery covers K64, hence two Q4_K gs32 metadata groups.  The
composed layout must prove, for every emitted fp16 value, that:

```text
metadata_group = logical_k / 32
metadata_column = logical_n
```

The current folded path expands the two nibbles primarily through its
N-folded ownership map; that result cannot be assumed for a K-packed word.
Scale and zero need the same proof, including the packed-metadata path.

### Collective and descriptor identity

The existing folded collective assumes a physical `(N/F, F*K)` shared tile.
The transposed format instead has an `(N, K/4)` b16 transport view and an
ordinary logical `(N,K)` MMA view.  It therefore needs a distinct provider and
descriptor identity, for example fields equivalent to:

```text
artifact_tile_k = 32
transport_k     = 64
k_codes_per_b16 = 4
transposed      = true
fold_n          = 1
```

It must not be labelled `xplane-q4k-fold2-a32`, and it cannot consume an
existing Fold2 artifact.

## Performance expectation from static accounting

Compared with the successful A64/F1 WN16 path, the proposed A32-transposed
path can retain:

- identical Q4 weight bytes;
- one four-register swizzle load per warp per K64;
- the same four source registers and 32 fp16 results per lane;
- the same WN16 warp topology and MMA count;
- two gs32 scale/zero groups per K64.

There is therefore no structural reason for the earlier A32/WN32 slowdown to
remain.  Unknowns that require measurement are the relative throughput/bank
behavior of the transposed swizzle opcode and any extra permutation left by
an imperfect converter implementation.  A correct offline inverse should
avoid a runtime shuffle.

Because this is a new byte class, its decode result cannot by itself select a
deployment format.  The same resident bytes must also be measured at the
already-covered prefill M values before choosing one format for all batches of
the same layer.

## Refuted alternative: lane-local x4 bases

The earlier attempt tried to retain the existing Fold2 bytes and make four
WN16 warps gather different sub-cubes by supplying a different
`stage_base` from each lane to the shipping x4 swizzle instruction.

PPU box artifact:

```text
/workspace/quactlize-swzl-x4-ba14fbd
```

Observed verdict:

```text
lane-local did not consume all cube bases: [0]
```

Every lane supplied a distinct aligned, uniquely tagged 512-byte cube, yet
all returned values came from cube base 0.  PPU001 tc01 x4.swzl therefore does
not have NVIDIA-style lane-local address-gather semantics.  That route is
closed and its temporary source/runner were removed after recording this
result.

## Implementation sequence if this format is pursued

1. Define the K-packed b16 offline map and exact inverse.  Exhaust asymmetric
   `(n,k)` tags and retain wrong-K-pack and old-Fold2 bytes as RED controls.
2. Add a project-owned transposed-Q4 operand using the existing matched b16
   AIU writer and b16 swizzle-trans reader.  Do not modify the vendor b8 path.
3. Derive a transposed `(lane,vreg) -> (n,k)` layout from the real fp16
   transposed identity chain, then compose it with the Q4 converter and MMA B
   fragment.  Require a whole-tile bijection.
4. Prove both gs32 metadata groups and every N column against an independent
   `(n,k/32)` oracle.
5. Close one exact device row first:
   `M1/N1024/K5120, TM8/TN64/TK256/WM8/WN16`, S1 before split-K.
6. Compare raw-bit correctness and codegen/performance against A64/F1 WN16
   and A32/F2 WN32.  Only then add the format to decode and prefill sweeps.

## Static verdict

`FEASIBLE_NEW_FORMAT`, not `DROP_IN_READER`.

Transpose genuinely solves the original K-contiguity coupling for this
workload because N16 b16 already supplies 32 contiguous bytes.  The remaining
work is a layout/converter/metadata proof, not an unresolved AIU minimum-grain
problem.

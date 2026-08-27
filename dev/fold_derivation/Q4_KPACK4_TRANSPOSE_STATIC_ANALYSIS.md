# Q4_K canonical K-pack4 transpose: static composition

## Verdict

The long-term Q4_K offline format should be a transposed, converter-native
K-pack4 byte class.  It removes `FoldN` from the weight format and gives all
decode/prefill tactics the same physical bytes.  `ArtifactTileK` no longer
selects a weight permutation.

The host/CuTe composition is exact:

```text
transposed b16 transport payload       N16 x K64 Q4 = 512 B
PPU0010 delivered registers            32 lanes x 4 b32 = 512 B
existing int4 converter composition    1024 / 1024 logical codes exact
source duplicates                      0
destination duplicates                 0
b16 words retaining one logical N      256 / 256
b16 words retaining one gs32 group     256 / 256
```

This is a static format/layout/converter result, not yet a device-performance
claim.  The next authority is one raw-bit S1 box closure.

The executable proof is
`l228_q4_kpack4_transpose.cu`, run by:

```bash
bash dev/fold_derivation/run_l228_q4_kpack4_transpose.sh
```

The runner binds the proof to the real PPU0010 transposed b16 AIU writer,
`m16n16.x1.swzl.trans.shared.b16` reader, and the existing RowMajor-B fp16
device authority.  It builds host layout code with `stub_inc` first, avoiding
the known and unrelated `hggc_fp8.h` host-toolchain failure.

## Canonical format

The tempting spelling packs four *consecutive* K codes into one b16.  That has
the correct byte count but is not the order consumed by the shipping int4
converter: only 64 of 1,024 positions compose correctly.

The converter-native spelling packs four codes from one N column and one
Q4_K gs32 group:

```text
group       = k / 32
residue     = k % 8
nibble      = (k % 32) / 8
physical_kg = 8 * group + residue

word[physical_kg][n].nibble[nibble] = q[n][k]
```

Therefore one physical b16 contains:

```text
{q[n][32*g+r], q[n][32*g+r+8],
 q[n][32*g+r+16], q[n][32*g+r+24]}
```

for one fixed `(n,g,r)`.  N is the contiguous physical dimension:

```text
word address = physical_kg * N + n
```

This is still K-pack4: four K codes occupy one b16.  The quartet is the
converter's native group-local interleave instead of `{k,k+1,k+2,k+3}`.
That distinction is what removes a runtime shuffle.

The mapping is a whole-matrix bijection, and its storage is exactly the raw Q4
payload size:

```text
(K/4) * N * sizeof(b16) = N * K / 2 bytes
```

Neither batch M, `ArtifactTileK`, tactic `TK`, `TN`, nor `WN` appears in the
address.  Q4_K already requires K in complete gs32 groups, so every b16 is
owned by exactly one metadata group.

## Why FoldN disappears

The current non-transposed A32 format makes K the AIU-contiguous dimension:

```text
A32 * 4 bit = 16 B
```

It folds two N columns merely to construct a 32-byte run.  That creates the
distinct `xplane-q4k-fold2-a32` byte class and couples legal readers to its
physical N tiling.

K-pack4 transpose makes N contiguous and transports opaque b16 words:

```text
N16 * sizeof(b16) = 32 B
```

The minimum run is now satisfied independently of the artifact K boundary.
The format has no FoldN field and no A32/A64/A128/A256 variants.

## Real PPU transport and compute composition

Actlize already contains the matched PPU0010 b16 pair:

```text
ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16
ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16
```

The fp16 RowMajor-B GEMM suite is the device authority that this pair delivers
an ordinary `(N,K)` B fragment.  K-pack4 treats each b16 as opaque bits rather
than a numeric fp16 value.

One transposed instruction delivers a `16 x 16` b16 microtile:

```text
16 N * 16 physical K-groups * 2 B = 512 B
16 physical K-groups * 4 codes    = K64
```

The four b32 registers per lane contain eight b16 words.  Expanding them with
the existing `MixGemmNumericArrayConverter<half_t,int4b_t,32>` produces 32
fp16 values per lane, exactly the four K16 MMA-B atoms for `N16 x K64`.

L228 composes the real PPU0010 M8 B-fragment layout, the delivered b16 halves,
`MixGemmEmit<4>`, and the four-atom K64 destination.  With the canonical
group-local K-pack4 mapping the result is 1,024/1,024 exact.  Thus the format
needs no runtime register permutation and can retain the shipping int4
conversion instructions.

## Scale and zero

One K64 transport contains two gs32 groups:

```text
physical_kg 0..7   -> group 0
physical_kg 8..15  -> group 1
```

All four nibbles in one b16 stay inside that group.  The existing shared
metadata policy for one K64 copy and four K16 MMA atoms is:

```text
FineScalePolicy<ScaleGroups=2, CopySteps=1, MmaAtoms=4>
atoms_per_group = 2
atom groups     = {0,0,1,1}
```

L228 binds and checks that production policy directly.  Scale/zero storage can
retain its existing `(N,K/32)` semantics; only the low-code weight bytes need
the new offline placement.

## Tactic and tail consequences

### WN

WN16 is the native microtile.  WN32 and WN64 compose two or four adjacent N16
reads over the same canonical bytes.  There is no reader-specific weight
format.

### TK64, TK128 and TK256

These compose one, two or four K64 transports.  L228 exhausts representative
WN/TK groupings and observes every logical code exactly once.

### TK32

TK32 does **not** require FoldN or another byte format.  Its two adjacent K32
compute slices are the two halves of one K64 transport.  However, the current
mainloop stages one tactic K tile at a time.  Reusing one K64 load across two
K32 compute slices needs either:

1. a K64 physical stage with two K32 consume steps; or
2. a separately proved half-width transposed reader.

This is a mainloop/resource/performance question, not a format-compatibility
question.  Until measured, TK32 must not be advertised as a drop-in existing
tactic.

### Matrix tails

The AIU producer is `padz`.  A final N or K/4 microtile can therefore be padded
in shared delivery without storing padding in the offline file.  L228 checks
an `N=18,K=96` compact buffer: all 1,728 valid codes recover exactly and all
2,368 transport-tail positions are zero.  Q4_K's gs32 requirement guarantees
that no stored b16 crosses a partial metadata group.

## Performance ledger

Relative to the successful A64/F1 WN16 reader, K-pack4 transpose retains:

- identical Q4 weight bytes;
- one 512-byte shared-to-register instruction per `N16 x K64`;
- four source b32 registers per lane;
- the existing int4 conversion instruction sequence;
- four K16 MMA atoms and the same scale/zero arithmetic;
- no runtime format shuffle.

Relative to a Fold2+x2 reader, K-pack4 uses one 512-byte transposed instruction
instead of two 256-byte x2 instructions for the same `N16 x K64` payload.  The
remaining performance unknowns are transposed-AIU/TSM throughput, bank
behavior, generated address arithmetic, and the K64 physical-stage impact on
TK32-prefill tactics.  Those require a box; the static model should not assign
them a speedup.

## Format-unification consequence

One layer can now keep one Q4_K weight buffer across all batches:

```text
canonical (K/4,N) K-pack4 bytes
├── decode M=1/2/4/8: WN16 reader
└── prefill M=2048/4096: WN16/32/64 composition
```

Different layers may still select different kernels and tactics.  They no
longer need different weight byte classes solely because their winning reader
uses A32 versus A64.  `ArtifactTileK` may remain an inventory or metadata
boundary during migration, but it must not enter the K-pack4 mapping hash.

## RED controls

The proof must fail for all three independent mistakes:

1. naive consecutive-K b16 packing;
2. rotating the converter destination by one fp16 slot;
3. shifting the metadata atom-to-group assignment.

It also verifies that N-major storage and the existing Fold2 address map are
not byte-identical aliases of this format.

## Implementation order

1. Add a named K-pack4 offline descriptor and exact pack/unpack map from L228.
2. Add a project-owned Q4 b16-transposed operand.  Reuse the vendor matched
   b16 writer/reader; do not reinterpret the unsupported b8-transposed path.
3. Retile its four raw b32 registers into the existing int4 converter and an
   ordinary `N x K64` MMA-B destination.
4. Bind two gs32 packed-metadata groups through the existing fine policy.
5. Close one exact S1 row first:
   `M1/N1024/K5120, TM8/TN64/TK256/WM8/WN16`.
6. Compare codegen and performance with A64/F1 WN16 and A32/F2 WN32.
7. Only after S1 closes, enable S2/S4/S8 and add the format to decode/prefill
   selection.

## Backup

`xplane-q4k-fold2-a32` plus a project-owned plain `ldmatrix.x2` WN16 reader
remains the compatibility backup.  It reuses deployed Fold2 bytes, but retains
FoldN as a physical-format axis and needs two x2 deliveries per `N16 x K64`.
It should not displace the canonical K-pack4 route unless the transposed device
closure exposes a concrete correctness or throughput blocker.

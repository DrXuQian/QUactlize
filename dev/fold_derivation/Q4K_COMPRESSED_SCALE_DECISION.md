# Q4_K-equivalent compressed scale channel: design decision

## Decision

Use the existing **Q4_K-equivalent, 16-byte reordered device unit** as the
selected representation.  Keep the 4-bit code plane unchanged.  Raw GGUF
Q4_K metadata is also 16 bytes, but its 12 code bytes are not separable at the
half-superblock boundary; the selected unit preserves the byte count while
making both four-group runs self-contained.  Do not ship an in-GEMM decode as
a general decode optimization:

- `K=8192` is the only justified prototype target.  Merely improving current
  Native has an absolute warm `f=1` budget of **0.7403 us**, but matching the
  sampled PDF scalar/pair arms after the ideal byte saving leaves only
  **0.1953/0.5078 us** for decode and publication.  These are preregistered
  thresholds, not a claim that byte service is already proved critical;
- `K=1024` is **not an implementation target**.  Even a free decoder plus the
  full ideal warm byte saving gives 2.6240 us, still slower than both sampled
  PDF arms (2.4600/2.4890 us).  It remains a required negative control;
- TODO #18 must not be enabled on this fp16 path.  It is algebraically
  composable with metadata decode and could remove the consumer's magic-offset
  subtract, but that composition has already failed the numerical gate.

This is a design decision, not a shipping-code change or a device result.

## Representation and exact byte accounting

For one output column and one 256-weight Q4_K superblock:

| representation | code bytes | scale metadata | total | bytes/weight |
|---|---:|---:|---:|---:|
| current Native affine int4 | 128 | 8 fp16 scales + 8 fp16 zeros = 32 B | 160 B | 0.625 |
| **Q4_K-equivalent selected form** | 128 | `d,dmin` fp16 + eight `(sc6,mn6)` pairs = 16 B | **144 B** | **0.5625** |
| byte-aligned fallback | 128 | `d,dmin` + eight int8 scale/min pairs = 20 B | 148 B | 0.578125 |

Thus Native is 11.111% larger than the Q4_K form, while replacing Native saves
**16/160 = 10.0% of the current bytes**.  Those denominators are not
interchangeable.  The byte-aligned fallback saves 7.5% of current bytes but
does not close the Q4_K target, so it is not the selected representation.

The selected 16-byte unit is already defined and independently anchored by the
raw Q4_K fixture.  Its first four bytes are `d,dmin`; its remaining 12 bytes
reorder the same eight 6-bit scale/min pairs into two self-contained 6-byte
runs.  Raw-GGUF decode -> selected-unit pack -> selected-unit decode reproduces
all fields, and each run is checked not to borrow bits from the other.  It is a
new *device layout* but not a new quantization: stored bytes and value semantics
are unchanged, as is the int4 code-plane placement.

## Converter cost on the existing path

The selected implementation shape is the existing amortised loader: one owner
thread per N column reads one 16-byte unit, decodes all eight groups, and
publishes the same fp16 `(scale,zero)` shared-memory view consumed today.  It
does not decode independently in every MMA lane.

For one 32-weight group, generated code for the existing packed-pair fast path
reduced the decoder from about 15 to about **11 opcodes**: constant-position
field extraction, packed bias/subtract, and a packed fp16 FMA producing
`(d*sc,-dmin*mn)`.  The 32-bit shared store that publishes the pair is
additional; it is not hidden inside that count.

For the measured implementation shape, a useful issue-count proxy is therefore
`(8 groups * 11 decode + 8 stores + 1 vector 16-byte load + 1 hoisted setup) /
256 = 0.3828125` issued operations per weight.  This is **not an ISA lower
bound**: field positions, vectorization, and compiler scheduling can change the
generated count, so the target instance must be disassembled and recounted.
Pipeline fill/residue may also multiply it: the pinned PPU path decoded nine
passes for eight K tiles, i.e. 1.125 times the logical work.  The count describes
the loader-thread chain, not whole-kernel dynamic instructions or elapsed time.

Source-level live state per loader thread is budgeted as:

- four 32-bit words for the native unit;
- one 32-bit packed `half2(d,-dmin)`;
- approximately three reusable extraction/arithmetic temporaries.

That is eight 32-bit source objects, **not eight additional physical
registers**.  The pinned codegen reported 98 registers for the packed arm
versus 102 for the fp16-plane arm, so that exact instance did not pay an
allocation/occupancy penalty; its physical delta was in fact negative.  Other
tiles must re-establish this from their own generated code.

The raw staging cost is `TN * 16 B * Stages`.  It and the explicit shared
stores remain part of the cost even if the arithmetic is hidden.  Do **not**
use the historical +12.9% or +2.4% packed-vs-base timings here: the repository's
current authority records that those arms computed different numbers, so they
are not a like-for-like latency measurement.  Their generated-code counts and
register allocations remain structural evidence; their time ratios do not.

## When ten percent fewer bytes can become time

Let `T` be the current time, `f` the fraction of that time whose critical path
is service of the 160-byte representation, and `delta` the added decode,
staging, and publication time.  The change can improve latency only if

```text
delta < 0.10 * f * T .
```

The right-hand side is an upper bound: it assumes the removed bytes scale the
critical service linearly.  Warm `GB/s` is cache-equivalent and may exceed DRAM
nameplate; it is not evidence that `f=1`.

| 132B shape/state | current Native | absolute `f=1` budget | decision |
|---|---:|---:|---|
| K=8192 warm | 7.4030 us, 3544.6 GB/s | **0.7403 us** | Prototype target. Ideal byte-only time is 6.6627 us; matching sampled scalar/pair 6.8580/7.1705 leaves **0.1953/0.5078 us** for all added work. |
| K=8192 cold | 17.7152 us | **1.7715 us** | Ideal byte-only time is 15.9437 us; matching sampled pair/scalar 16.7216/16.7424 leaves **0.7779/0.7987 us**. |
| K=1024 warm | 2.9155 us, 1128.1 GB/s | **0.29155 us** | Negative target. Ideal byte-only time is 2.6240 us, still slower than sampled scalar/pair 2.4600/2.4890; representation size alone cannot close the gap. |
| K=1024 cold | 3.9835 us | **0.39835 us** | Ideal byte-only time is 3.5852 us; matching sampled scalar/pair 3.6495/3.6790 leaves only **0.06435/0.09385 us**.  This is not a credible implementation budget without contrary codegen evidence. |

The prediction is therefore **not the same across the two K values**.  The
bytes move in the same mathematical direction, but K=8192 alone leaves a
meaningful measured-arm budget after paying for the decoder.  K=1024 is a
falsifying control for a global “smaller representation is faster” claim.

The source A/B marked every formal timing verdict `UNRESOLVED` because its
admissible timer quantum was unknown, and the scalar/pair arms are an ambiguity
in the reconstructed PDF path.  The numbers above are therefore thresholds
for a future experiment, not upgraded verdicts.  Warm `GB/s` is
cache-equivalent and cannot establish `f=1` or DRAM saturation.

## Relationship to TODO #18

TODO #18 proposed leaving the converter's magic offset in the fp16 value and
folding it into the affine constant:

```text
raw * s' + (z - 1024*s').
```

That is algebraically composable with compressed-scale unpack: the metadata
decoder could publish the modified affine bias and remove the consumer's
magic-offset subtract.  It is nevertheless not a valid fp16 continuation.  It
creates two magnitude-approximately-10 fp16 terms whose cancellation must
recover a small dequantized value.  The checked implementation measured maximum
relative errors from 1.9e-2 to 4.6e-2; ScaleOnly is worse because the entire
result is the cancellation.  The shipping converter instead subtracts the
integer magic offset exactly before multiplication.

The proposed #20 decoder first recovers small exact `sc,mn` values, then forms
normal-magnitude fp16 scale/zero.  Reusing #18 would require it to undo that
safety and manufacture the large folded bias.  The operations also occur at
different frequencies, and the active GEMM loader and GEMV converter are not
one shared call chain: metadata unpack is once per column/superblock/group,
while weight conversion is per packed weight group.

**Conclusion:** #20 and #18 conflict in general fp16.  Do not merge them.  #18
can be reopened only as a different experiment using fp32 intermediates (with
its added conversions/registers) or a proved restricted scale domain.  Neither
is part of this representation.

## Evidence authority

- `quactlize/include/quactlize_extensions/cutlass/gguf_packed_scale.h` owns the
  selected 16-byte layout, the raw-layout round-trip anchor, and the measured
  15-to-11 opcode reduction for the packed-pair decoder.
- `dev/fold_derivation/Q4K_PDF_5090_AB.md` owns the 5090 representation sizes,
  raw medians, protocol caveats, and `UNRESOLVED` formal verdicts used above.
- `benchmarks/run_batch.sh` is the current authority that invalidates the old
  +12.9%/+2.4% time ratios as non-like-for-like comparisons.
- `quactlize/include/gemv_lowbit/gemv_converter.hpp` owns the TODO #18
  cancellation counterexample and measured fp16 error range.

## Implementation gate if K=8192 is pursued

The targeted implementation is accepted only if all of these hold in one A/B:

1. the code plane is byte-identical and the 16-byte metadata round-trips every
   `d,dmin,sc6,mn6` field;
2. decoded fp16 scale/zero is bit-identical to the current planes on the same
   fixture, including subnormal `d` coverage;
3. incremental registers do not lower the chosen tactic's active-block limit;
4. raw staging plus fused stores are included in `delta`;
5. warm and cold K=8192 each beat the current Native arm by more than the
   preregistered timer resolution.  K=1024 is reported as a negative control,
   not used to retune the design after the fact.

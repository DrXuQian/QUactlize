# Q4 reader reuse: a bounded PPU experiment

This follows the [393-config cold sweep](Q4_CONFIG_SWEEP_RESULTS_20260912.md).
The remaining N5120/K8192 gap has nearly equal grid/block, achieved occupancy
and DRAM bytes versus raw reference, but 3.70x L1/L2 traffic and 1.78x vector
loads. This experiment changes reader internals without changing canonical
K-pack4 bytes, dot accumulation precision, reduction topology or production
selection. [PPU results are now reviewed](Q4_READER_REUSE_RESULTS_20260912.md):
all declared numerical cells pass; one of two shapes meets the strict
reference performance target. Shipping admission remains out of scope.

## Exact experiment matrix

Dense M=1, Q4_K, rotating weights >=2.25 x verified L2, two shapes only:

| N | K | Previous confirmed winner | Alternate geometry |
|---:|---:|---|---|
| 5120 | 8192 | C4/W8/P4 | C8/W16/P4 |
| 8192 | 5120 | C8/W10/P4 | C4/W10/P8 |

The alternate N5120 geometry improves B coalescing but lost the old sweep;
the N8192 alternate was a close contender. Reuse changes may change that
tradeoff, so both geometries are explicit, not substituted globally.

Each geometry gets the full three-switch factorial:

| Variant key | Cooperative A | Cooperative units | 32-bit metadata decode |
|---|---|---|---|
| v0 | off | off | off |
| v1 | on | off | off |
| v2 | off | on | off |
| v3 | on | on | off |
| v4 | off | off | on |
| v5 | on | off | on |
| v6 | off | on | on |
| v7 | on | on | on |

That is 32 reader/config contexts. All are confirmed, without another
393-config screen: six alternating rounds x 15 samples, plus the previous
winner, raw reference and Xplane each round, **228 timing cells** total.
Profile all eight variants at the previous winner's geometry, the fastest
alternate-geometry variant and the three controls: **24 ACU reports**.

## What changes

### A: contiguous cooperative load, register redistribution

The C lanes sharing one K group each load a disjoint slice of its 64 B of A:
C4 uses 16 B/lane and C8 uses 8 B/lane. Each lane then receives the same 32
FP16 values as before through indexed warp shuffles, in the same dot order.
This is not a new A layout or a shared-memory prepass.

### Metadata: one contiguous warp load of the output tile's units

The first TileN lanes each load one 16 B unit for adjacent N columns. Every
consumer obtains its original P units by register shuffle. Declared
geometries guarantee that a warp's K groups belong to the same superblock,
and that shuffle operations execute on whole active warps. C4 can share one
header across all eight groups of a superblock; C8 still reads it once for
each four-group warp. Reuse across separate warps/CTAs is not claimed.

### Decode: two 24-bit streams using only 32-bit shifts

The same six-bit scale/min codes are obtained from `uint4` metadata with
constant 32-bit assembly shifts followed by a variable shift of 0/6/12/18.
No variable 64-bit shift is needed. FP16 header conversion and FP32 affine
arithmetic stay unchanged. This factor changes no addresses.

All variants retain the existing `lop3 + half2` code unpack and FP32 dot.
The original packed-B load block and final reduction body are unchanged.
No inter-CTA Split-K, external reducer, expanded scale plane or AIU is added.

## Address and native-code evidence

The generated plan includes each C/W/P and switch combination. Per-variant
`access-patterns.json` records B/A/unit lane addresses, logical source-load
bytes, duplicate/sector footprint models and source shuffle steps. Timed
rows repeat the model with actually observed pointer-base alignment.
`implemented_streams` identifies the active paths; other streams are
baseline/alternative models, not extra traffic to add together.

At N5120/K8192, C4/W8/P4, logical bytes per call are:

| Operand | Original | Cooperative | Change |
|---|---:|---:|---:|
| Packed B | 20,971,520 | 20,971,520 | unchanged |
| A | 20,971,520 | 5,242,880 | 4x less |
| Units | 20,971,520 | 2,621,440 | 8x less |

These are source logical lane bytes, **not measured DRAM or L1/L2 bytes**.
Under aligned 64 B footprint models, cooperative A/unit source loads cover
complete sectors. Register redistribution costs extra instructions and
dependencies, so lower source bytes do not establish a net speedup.

The PPU native build confirms `v.shuffle.idx.b32`, rather than added shared
staging. The same final CTA shared reduction/barrier remains. Header32
variants have no `v.shrl.b64`; all 32 specializations retain native lop3/half2
code unpack and FP32 FMA. `isa-stats.json` includes loads, shuffles, shared
operations and control branches. **The compiler changes loop unrolling in
some variants**: do not compare raw static instruction counts as if they
were dynamic counts. ACU decides traffic, stalls and occupancy on device.

## Correctness and acceptance

Every reader/config first executes the immutable old config-sweep library
at **the identical C/W/P**, then the new reader, and compares FP32 output
bits. This also checks the unchanged v0 clone. Only then does the inherited
independent-GGUF, zero-code, zero-A, output/workspace guard and graph-replay
gate permit timing. A reader must not pass merely by agreeing with itself.

Local tests cover every declared warp/pass/lane A and metadata owner,
wrong-owner negatives, all eight metadata bit fields on 131,072 random units,
source preservation, native exports/ISA and full orchestration with injected
failure/retry and profile receipt checks. Native builds completed locally
in 16.7 seconds; local tests do not certify PPU numerical behavior.

Compare each variant to v0 at its own geometry, and the overall winner to the
remeasured previous winner and controls. The bottom line remains
**K-pack <= raw-reference median, with no 5% regression allowance**. A
complete experiment may still report `REF_REGRESSION`. Do not promote an
unproved reader into shipping dispatch or extrapolate two-shape M1 results
to small-N, M2–7, indexed/grouped GEMV or other qtypes.

## Box command

No compile or JIT is needed. Keep the selected card otherwise idle.

```bash
(
  cd /sim/eec/shared/junfu.qx/quactlize &&
  GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop &&
  git lfs pull --include='prebuilt/ppu0010/q4-reader-reuse-v1/*.so,prebuilt/ppu0010/q4-config-sweep-v1/*.so,prebuilt/ppu0010/q4-cold-shapes-v1/*.so,prebuilt/ppu0010/q4-h800-port-v1/*.so,prebuilt/ppu0010/q4-simt-ab-v1/*.so' &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
  bash tools/run_q4_reader_reuse_ppu_box.sh
)
```

Optional `FIXTURES=/workspace/q4-config-sweep-ppu.BdEY54/fixtures` reuses the
previous run's inputs when that exact directory still exists. Without it,
the script creates only the two required fixtures. Do not delete old runs.

Plan around **8–15 minutes plus fixture preparation**, not a guaranteed
deadline: the preceding measured campaign took 603 s with more child
contexts and the same 24 profiles. Setup, first launch and graph upload are
excluded from kernel timing. Progress is printed per batch and within it.

On failure, completed independent cells stay valid. Retry with
`RESUME_RUN=/exact/q4-reader-reuse-ppu.RUN` using the same source/device/SDK;
only missing/failed cells are executed. `ACU=0` skips profiles if explicitly
desired, but cannot provide the counter evidence this experiment seeks.

Return the printed `results=...results.tgz`: summary.tsv, summary.json with
same-geometry deltas and round samples, address/ISA/build evidence, raw logs
and ACU reports are included. The caller's Docker shell remains open.

# Q4 SIMT B transport and FP32 raw-GGUF reference

This is an isolated development experiment. It does not change the canonical
K-pack4 bytes, production dispatch, or the running shared-A package.

## Compared implementations

| Arm | Weight bytes | A | Dot/reduction/output | Purpose |
| --- | --- | --- | --- | --- |
| `xplane` | Historical Xplane | FP16, CTA shared staging | FP32 | Frozen PPU-selected comparator |
| `baseline` | Canonical K-pack4 | FP16, direct global reads | FP32 | Frozen current N4 reader |
| `raw-reference` | Original GGUF Q4_K blocks | FP16, CTA shared staging | FP32 | Supplied `gemv_ref.cu`, with FP32 inner accumulators |
| `fragment-global` | Canonical K-pack4 | FP16, direct global reads | FP32 | Direct B loads with the new fragment/compute topology |
| `fragment-aiu` | Canonical K-pack4 | Same as `fragment-global` | Same FP32 order as `fragment-global` | AIU -> swizzled shared -> transposed ldmatrix |

The three raw-reference recipes are `(columns-per-warp, N-warps, split) =
(1,8,1), (2,8,1), (4,8,1)`. They are a bounded comparison, not a claim of
global optimality. The Xplane/current K-pack recipes are unchanged from the
previous PPU receipt.

The B experiment uses only `(stage-K, K-warps, split) = (256,4,1), (512,8,1),
(1024,8,1)`. All CTAs compute N16; increasing stage-K amortizes AIU completion
and shared-buffer lifetime barriers. There is one B buffer, not an unproved
double-buffer pipeline. No separate reduction kernel, MMA, or full-weight
FP16 expansion is used. The transport's shared footprint and barriers are part
of its latency cost, not excluded setup.

## What can be attributed to B transport

The original N4 mapping and native ldmatrix mapping differ. Therefore comparing
`fragment-aiu` only with the original `baseline` cannot isolate transport.
`fragment-global` and `fragment-aiu` share A accesses, metadata formulas,
FP32 dot order, output ownership, block/grid geometry and reduction. The
runner requires exact FP32 output equality between these two. Their generated
register/shared resources can differ; report those as costs of the transport.

The existing b16 pair is called directly through Actlize's operations:

```text
PPU0010_AIU_LOAD<..., half_t, Trans=true, Swzl=true>
  -> PPU0010_TSM_LD_SWZL<half_t, H, 16, Swap=true, Trans=true>
  -> 4 b32 registers/lane -> SIMT FP32
```

One read delivers N16 x K64 Q4. Register `v` contains two opaque b16 words:

```text
n  = lane/4 + 8*(v/2)
kg = 2*(lane%4) + 8*(v%2) + half
k  = 32*(kg/8) + kg%8 + 8*nibble
```

The adjacent b16 halves are K residues of the same N column, not adjacent
columns. `bload_layout.cu` composes this with real CuTe
`partition_fragment_B`/`right_inverse`, checking 256 words and 1024 codes;
wrong register halves and wrong nibbles are explicit negatives.

On PPU, a separate raw-b16 gate uses the actual AIU writer and ldmatrix reader
over four N tiles and repeated K stages. It compares all raw registers with
independent coordinate tags and rejects stale-stage/wrong-N interpretations.
Only after this passes are numeric results timed. AIU has exactly one DMA
issuer per CTA, a wait and publication barrier, and a consumer-completion
barrier before buffer reuse.

## Supplied reference adaptation

`dev/gemv_ppu/reference/gemv_ref.cuh` is the exact supplied file, SHA256
`6d575665df5f9fe16d9259178d28dc625867bdcba9967e49ad8985b0af528e65`.
`bload_source.reference_fp32` generates a separate adaptation. It changes all
four half2 dot accumulators to float2, uses scalar FP32 FMA on both components,
folds/reduces in FP32 and writes FP32. It preserves raw block addressing,
vector header/code loads, A staging and the reference's FP16 dequantization
formulas. The original file is not silently edited.

Raw-reference and K-pack use different affine rounding formulas. All arms are
checked against the SAME independent GGUF dot oracle, but raw-reference vs
K-pack is not a bit-identical dequantization or pure-layout comparison.

For NVIDIA compilation only, the packed-half inline-assembly constants use
register constraints instead of immediate constraints, as required by ptxas.
Constant bits, dequantization and arithmetic precision are unchanged. There
is no AIU emulation on NVIDIA.

## PPU execution

Default: N5120/K8192 only, warm and >L2 rotating, four alternating rounds of
15 graph samples. First graph upload/replays are excluded. Three recipes are
batched per process where possible. This is not a Cartesian sweep. Failed
children retain already-validated cells; other arms continue in fresh
processes. No production library, JIT or box compilation is involved.

```bash
(
  GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop &&
  git lfs pull --include="prebuilt/ppu0010/q4-simt-ab-v1/*.so,prebuilt/ppu0010/q4-bload-v1/*.so" --exclude="" &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
  bash tools/run_q4_bload_ppu_box.sh
)
```

Set `ALL_SHAPES=1` to include all six existing M1 shapes; default anchors the
new reader before expanding its admission. `ACU=0` skips profiling; default
profiles all five arms at N5120/K8192. The direct fragment ACU recipe is the
same as the selected AIU recipe. Counters use forced-cold replay, and those
profiled durations do not replace warm/rotating event timings.

`FIXTURES=/workspace/q4-simt-ppu.0uCEmO/fixtures` may reuse the previous exact
inputs. `RESUME_RUN=/workspace/q4-bload-ppu.<suffix>` reuses successful timing
cells only when runtime/source/device/fixture hashes match. ACU reports are
retained, but an explicit resume currently recollects them.

Upload the printed `q4-bload-ppu.<suffix>.results.tgz`. `status=PASS` means
valid measurements, not a performance win. Compare each recipe's
`matched_transport` before attributing improvement to AIU. No device result or
latency improvement is claimed from compilation or the host layout proof.

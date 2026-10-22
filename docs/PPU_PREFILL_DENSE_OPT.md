# Dense prefill helpers, Q6 expansion and Full-FP16 candidates

Items 1/2/3/5 from the Qwen3-32B PP2048 analysis, based on 179ef70. Item4
(TC/SF configuration selection) remains the separate frozen Asys sweep.
No existing selector or default BF16/MoE route is changed by this experiment.

## Library changes

- `quactlize_dense_io_v1`: explicit F16/BF16 contiguous 8-element vector
  conversion; safe unaligned/tail/strided fallback, unchanged RNE/overflow.
  Output reads stable per-call error status once per vector and propagates
  NaNs when nonzero. No clipping; F16 range is not BF16 range.
- `quactlize_kpack_dequant_dense_v1`: direct raw-GGUF FP32 arithmetic with
  one final F16/BF16 rounding. Packed-code exchange candidates for Q4/Q5/Q6,
  TileK128/256 and N/K CTA ordering controls. The direct reference covers
  Q2..Q6. No offline layout changes or expanded-weight reuse.
- `quactlize_kpack_dense_full_*_v1`: an address-free, dense-only full route
  with per-call A/B/C/workspace/stream, F32 endpoints, F16 or BF16 intermediate
  storage and FP32 cuBLAS accumulation. Existing v1 BF16/MoE APIs remain.
  A per-handle mutex protects stream/workspace/enqueue, not GPU completion.
  Query uses existing `query_dynamic_v1`; no allocation/JIT/sync in run.
- The small prefill library build includes these additive entries. The caller
  and production policy do not automatically select them.

The old BF16 input/output kernels and old full expansion are retained as the
control. Conversion optimization, expansion optimization, storage precision,
and provider algorithm are separate arms, not one opaque before/after change.
The F16 control keeps math-mode16 (no reduced-precision reduction), compute68
(32F); algorithm -1 and99 are compared. This is not a claim that all other
native llama provider settings are identical. Actual device performance is
pending, and old BF16 timing is never relabelled F16 evidence.

## Address-pattern and resource hypotheses

Vector conversion: each thread moves 8 elements, two F32 float4 transactions
and one 16-bit uint4 transaction; the warp's combined footprint is 1024B F32
and512B F16/BF16. Pointer/row alignment is checked before selecting vector
access. All full prefixes and tails are covered; there is no per-row division
in the flat body. Paired conversion intrinsics still require native ISA checks.

Q6: producers read N8 uint4s across contiguous N, preserving both packed
planes. Four selected K+8 codes travel in one shared32-bit word. Consumers
read paired shared uint4s and write contiguous K8 uint4s. Affine metadata is
computed once per(N,K16), not broadcast repeatedly for each expanded value.
An xor layout distributes affine producers across shared banks while keeping
consumer groups local. This is a logical bank model, not measured conflicts.

TileK128 holds4096B codes+2048B affine=6144B, versus the old `full_wide`
expanded tile's16512B. TileK256 uses12288B for twice the K work. The older
Q6-V config2 has a different smaller tile; do not generalize the K128 shared
reduction to that control or claim a latency win from storage size alone.

## Box

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop
unset RUN
GPU=2 IDLE_CONFIRMED=1 PPU_SDK=/usr/local/PPU_SDK JOBS=4 \
  bash tools/run_prefill_dense_opt_box.sh
```

Choose an idle card; do not overlap the Asys sweep on that card. No llama
build/model loading or TC compilation is needed. Local git objects freeze
Quactlize and the actlize pin; no HTTPS nested submodule fetch.1700MHz is
requested; actual clocks are recorded without failing on TDP throttling.

The gate covers7 PP2048 families and11 arms with three alternating-order
rounds. Input/output conversions, weight expansion, GEMM, reset and complete
calls have separate ordinary-event measurements; these are not raw kernel
cycles and their sum is not asserted equal to the full call. All cuBLAS
warmups occur before timed events. Native llama is the separate Asys sweep
control, not an implied arm of this helper experiment.

After numerical/timing success,15 bounded ACU captures profile representative
gate/up, Q6-down and Q6-V stages. Each capture uses one weight copy and one
explicitly ranged component; no cold rings, CUDA Graph, or timed loops inside
ACU. GEMM captures retain every provider subkernel, not just the first launch.
Q6 captures include old BF16, the best measured BF16 packed-dequant candidate,
and the best full-FP16 arm, so precision changes cannot hide exchange costs.
ACU cache-control=all results are forced-cold profiler measurements, not the
ordinary resident-helper/event protocol.

Upload only `results/review-*.tgz` (JSON, bounded logs, raw counter CSV).
Reports remain at `results/profile-*.acurep`; compiled libraries/weights and
reports are excluded from the compact archive. No performance/production
admission is asserted until returned target-device evidence is reviewed.

## Round 2: address chain, per-instruction coalescing, CTA geometry

Round1 returned `review-1792630720834391642.tgz` (SHA256
`958025e30093278cb01bbd5c34ad72390938550bdf9059b2c73c7773cd042594`).
This round retains its exact old kernels; it does not promote its screen
minima to production. No llama rebuild or GEMM config sweep is required.

- IO option2 uses two independent vec4 chunks/thread. At one FP32 vector
  instruction, old vec8 lanes request `[32*lane,32*lane+16)`: 512 useful
  bytes touch 16 64B sectors. New lanes request `[16*lane,16*lane+16)`:
  512 useful bytes touch8 sectors. Both halves remain explicit independent
  loads; the 16-bit side uses contiguous8B/lane vectors. No shared staging.
  L2 merging made the old combined footprint look coalesced; inspect LSU
  transactions as well as L2/DRAM, not only combined bytes. Safe scalar tails,
  misalignment/stride fallback and IEEE RNE conversion are preserved.
- Dequant options4/5 keep N32/T128/K128 or K256 but use unsigned validated
  coordinates, slot arithmetic and64-bit plane/metadata products. Q6 scale
  values remain signed; affine math and final F16/BF16 rounding are unchanged.
  Canonical Q5 high-plane N/K permutation is retained.
- Options6/7 independently try N16/T128/K256 and N32/T256/K256. For the small
  Q6 N1024 K5120 point, the old640 CTAs x128 threads cannot fill the theoretical
  72x16 CTA slots. N16 gives1280x128; T256 gives640x256. Neither is declared
  faster merely because it exposes more warps. K128 remains a control.

Local evidence: real PPU compile produces72 instances, all zero stack.
188 related host tests pass (numeric/layout, dynamic ABI, timers and harness).
The old40 instances have byte-identical encoded instructions. Q6 K128 static
`v.shra.i32` goes27→2; `v.shrl.b32`148→90, same38 vector registers. These are
static codegen counts, not dynamic instruction counts or target speedups.
Independent non-dyadic GGUF tests cover Q4/Q5/Q6 xF16/BF16 xall four new
geometries; wrong-slot and wrong-thread-stride mutations are detected.
Tests also prove lane ownership, tails, vector alignment and64-bit addressing.
Target GPU numerics/performance still require the command below.

```bash
cd /sim/eec/shared/junfu.qx/quactlize
git pull --ff-only origin develop
unset RUN
DENSE_ROUND=2 GPU=2 IDLE_CONFIRMED=1 PPU_SDK=/usr/local/PPU_SDK JOBS=4 \
  bash tools/run_prefill_dense_opt_box.sh
```

Seven PP2048 families,14 arms/point, four alternating-order component rounds.
Only dequant and IO are screened; the two dtype-specific combinations then
get separately measured complete calls against their controls (four rounds).
All provider warmups precede ordinary event timing. The same algorithm-1,
FP32 accumulator and4MiB explicit provider workspace remain fixed. The sum
of isolated components is NOT labelled measured full-call latency.

Then24 ACU captures: six component pairs in AB/BA order. Each capture is
one kernel/one weight copy, with no cold-ring or timed loop in the profiler
range. Pairs cover BF16 input/output, Q4 expansion, Q6 expansion in both
dtypes, and the small-Q6 geometry control. Every other case still has
numerical and event-timing coverage. `cycle-pairs.json` reports CE cycles
and the actual frequency separately; a higher frequency is not a cycle gain.
The requested1700MHz clock is not a guarantee against TDP throttling.
There is no production update or model-E2E/no-regression admission.

Upload the printed `results/review-*.tgz` only; raw counters are included.
The24 `.acurep` files and binaries stay on Box. For a stopped measurement,
reuse the frozen source/output (not a newer checkout), same device and SDK:

```bash
python3 /workspace/prefill-dense-r2.<RUN>/source/tools/run_prefill_dense_opt.py \
  --sdk /usr/local/PPU_SDK --output /workspace/prefill-dense-r2.<RUN>/results \
  --device 2 --l2-bytes 67108864 --confirm-idle --round-two --resume
```

Resume verifies source, provider, SDK, device and compiled binary hashes.
Completed points/profiles are reused unchanged; failed/interrupted work gets
new attempt logs. A partial build is not reused. Use the same SDK environment
as the initial wrapper. No raw evidence is silently re-labelled or deleted.

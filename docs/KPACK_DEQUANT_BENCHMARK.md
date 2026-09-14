# Independent weight-expansion measurements

## Full-weight reader follow-up: v3

Reviewed return: `kpack-prefill-cost.9jSpJD.results.tgz`, SHA256
`bac13ed261e20d1c8cfbec292642aeb015fd939d73622ac7fd77d08189a03c87`.
Source `210d6c9`, same PPU PCI `0000:08:00.0`, 72 compute units, 64 MiB L2.
All 78 dequant cases / 408 timed configs validate; all nine ACU reports
were imported locally. BF16 dense has 20 valid cells. The 12 grouped
families stop at the zero-A/NaN-output test before timing: no BF16 MoE cost
is admitted. This failure remains separate work; v3 does not bypass it.

Compared with contemporaneous old per-shape bests, SF improves in all 34
cases (median time reduction 45.42%); full BF16 expansion improves in all
34 (median 26.45%). Full c4/c5 win 13/21 cases respectively. Keep both as
controls. This is **full weight expansion**, not fully-quantized GEMM.

| Evidence | Old best | v2 best |
|---|---:|---:|
| Q4 N5120/K8192/E1 full event time | 120.180 us | 88.571 us |
| Same full kernel, ACU DRAM read bytes | 106,516,224 | 23,609,600 |
| Same full kernel, ACU L1/L2 traffic bytes | 228,751,616 | 130,819,968 |
| Same full kernel, ACU vector-load instructions | 1,474,560 | 368,640 |
| Same full kernel, shared bank-conflict latency sum | 2,580,480 | 2,580,480 |
| Q5 N2048/K512/E256 full event time | 755.340 us | 543.230 us |
| Same Q5 full kernel, ACU DRAM read bytes | 714,804,480 | 184,567,168 |

The read amplification is largely removed in these anchors. Remaining
shared exchange, repeated metadata requests and integer/decode work are
candidate targets, not proved wall-time attribution. SF expands only two
scale/zero planes; its speedup percentage is not the full-weight ceiling.

The v3 package retains all earlier kernel bodies and adds four Q4/Q5-only
reader variants. Global B vector ownership remains N32/K32 per warp:
`col0=8*(lane%4)`, `residue=lane/4`. Each B request comprises eight aligned
64-byte segments, no extra A input, unchanged canonical Q5 high-plane fold.
The emitted output stores remain uint4. FP32 multiply/subtract and final
BF16 RNE are unchanged; no FP16 fast-dequant substitution.

| Config | K per CTA / shared stage | Shared exchange | Metadata lifetime | CTA barriers |
|---|---|---|---|---:|
| 4/5 | 128/128 | Previous measured controls, unchanged | Previous readers | 1 |
| 6 | 128/128 | CuTe bank swizzle, two aligned K4 shared vectors per output K8 | Same per-group global unit reads as c5 | 1 |
| 7 | 128/128 | Same as c6 | One unit per N per CTA, published before use | 2 |
| 8 | 256/256 | Full superblock shared stage | One unit per N for all eight groups | 2 |
| 9 | 256/128 | Reuse a K128 shared stage twice | Same full-superblock unit cache | 4 |

The shared address function is
`Swizzle<3,2,3>(N*StageK+K) xor (N&24)` in uint32 cells. Host tests execute
the actual CuTe function: complete bijection, unique producer, matching
consumer, aligned K4 vectors, and wrong-view negatives. The 32-bank/4-byte
model has uniform vector reads and unique scalar producer banks; PPU ACU
must verify actual latency. This is not a claim of hardware bank parity.

The **old** c5 shared loads were already vectorized by the compiler; c6
tests their address layout, not a fictional new load-width improvement.
For c5/c6 the requested metadata is 128 bytes per N/superblock; c7 reduces
it to 32, c8/c9 to 16. These are request bytes, not distinct DRAM bytes.
Cached metadata costs 512 shared bytes and a publishing barrier. Shared
weight storage is 16 KiB for c6/7/9 and 32 KiB for c8. The staged c9 also
uses more registers (Q4:66, Q5:72 in the initial local compile), so its gain
is not assumed. All new kernels have zero reported stack size. Native
receipts retain opcodes and inspector register/shared-allocation fields.

Box command (prebuilt library; no GEMM, JIT, or SF retest):

```bash
git pull --ff-only origin develop &&
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_dequant_ppu_box.sh full-reader
```

The `full-reader` inventory contains 2 untimed Q4/Q5 smoke cases and the
same 34 timed weight domains, configs 4--9: **204 timings**. Earlier c0--c3
remain in the library but are not re-swept: the reviewed v2 result selects
c4 or c5 for every domain. New variants are not instantiated for Q2/Q3/Q6;
their precision/storage and admission are not inferred from Q4/Q5. The
input/output rings and 3x5 alternating-order protocol remain unchanged.
At most four ACU profiles compare each anchor's current old best to the
new winner. Successful independent cases remain resumable on failure.
The prior full-only portion consumed 2307.6 seconds of measured wall time;
kernel-only timings are not a campaign ETA. New performance is PPU pending.

## Current handoff: vector dequant and separate BF16 providers

The v1 result `kpack-dequant-ppu.0046wb.results.tgz` was reviewed on
2026-09-14: 78/78 stage cases, 238 timed configs, 8 locally imported ACU
reports, all numerical/guard/negative checks passed. Archive SHA256:
`344f3267a56a4ee9fc1cd5abc29ab1ccdbe5ef07516bebb5ad7094cb9cc774bd`.
SF means **only metadata expansion**, even when E256 metadata is expanded;
it never materializes the low/high weight codes. Q4 dense5120x8192 improved
12.3983 to 11.7072 us, and Q5 grouped2048x512/E256 improved 61.8511 to 54.0911 us.
The respective useful-byte bandwidths are 671.7 and 930.5 GB/s. They are not
ACU DRAM bandwidths. In these SF profiles outputs largely remain in cache;
single-kernel DRAM writes are near zero. Use the complete-ring event timing
for the declared standalone cost, not that incomplete writeback interval.

Full BF16 c2 won all 34 full cases against c0/c1, but its ACU profiles still
show shared bank-conflict latency and amplified DRAM reads. A higher read
count is evidence to investigate output write allocation/tile order, not
proof of that unique cause. The experiment is not yet tuned to a bandwidth
ceiling and no production choice is changed.

`kpack-dequant-v2` adds candidates while retaining every old config:

| Stage/config | Change | Unchanged numerical/storage contract |
|---|---|---|
| SF4/5, Q4/Q5 only | One aligned uint4 per N/superblock, decode all8 groups from registers, block128/256 | Same packed16 metadata and exact FP16 scale/zero rounding |
| Full3 | N32/K128 tile, block128, paired BF16 output | Same scalar B reader and FP32 raw-GGUF arithmetic |
| Full4 | Same tile, uint4 output stores | Same B load and decode as Full3 |
| Full5 | uint4 low/high B loads, metadata broadcast across8 K residues, uint4 output | Same canonical plane maps; Q5 high-plane fold preserved |

SF uint4 requests 512 contiguous bytes per warp, not 32 isolated byte loads;
each output group still has 64 contiguous bytes of FP16 stores. Full5 gives
lane `l` N base `8*(l%4)` and K residue `l/4`: four adjacent16-byte N vectors
per residue, eight residues per warp. The four residue-zero lanes read
metadata and broadcast decoded affine values. A warp owns N32/K32; four
warps own N32/K128. A uint4 output warp writes two256-byte contiguous rows.
All plane bases and vector addresses are16-byte aligned. Shared[32][129]
remains an experimental choice: source-level padding does not prove a
conflict-free PPU instruction. Preserve ACU bank counters.

Local compilation explicitly rejects scalarized uint4 metadata loads or
vector output stores. The local host, dispatch and receipt regression set
passes 210 tests; the PPU compile takes about 13 seconds and emits 49 kernel
specializations. Numerical execution/performance for the new variants
is **PPU pending**. Default v2 dequant plan:10 untimed smoke cases,
68 timed stage cases, **408 config timings**; timing protocol unchanged.

To run the two independent phases together:

```bash
git pull --ff-only origin develop &&
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_prefill_cost_ppu_box.sh
```

The second phase uses installed libraries, not a replacement GEMM:

- Dense: the SDK `libcublas.so` cuBLAS compatibility API, BF16 A/B/output,
  FP32 compute with reduced-precision reductions disabled. Actual loaded
  cuBLAS/PPU BLAS backend images are recorded. This is not NVIDIA hardware.
- Grouped: installed DeepGEMM's public
  `m_grouped_gemm_bf16_bf16_bf16_nt_nopad`, with its own default selection.
  Resident sorted row IDs and row counts are supplied. Any internal block
  directory kernel remains timed. Missing entry/import is a failure, never
  silently replaced by torch.matmul or padded batched GEMM.

Default providers: Q4/Q5, the same5 dense/6 grouped families, M/tokens2048
and4096: **44 GEMM cells in22 weight-family processes**. Grouped uses E256,
top8 and the previous weighted-without-replacement real router; its total
rows are tokens*8, not tokens. `MS=512,2048,4096` extends the M ladder.
The actual dequant output is verified and consumed as BF16[E,N,K].
Provider setup/JIT/first invocation, graph upload and two initial replays are
excluded and the first-use wall time is reported. Torch graph capture owns
provider temporary allocations. The complete BF16 weight ring exceeds2.25L2;
changed-A replay, zero-A, full output and guards are checked outside timing.
Independent BF16 dot references use official-GGUF-rounded weights and
factorized, BF16-representable activations; no compared GEMM computes its
own oracle.

The runner returns `results/bf16/summary.tsv` with independent full-dequant
and BF16-provider times plus their **sum estimate**, not measured E2E.
External routing/gather, A dtype conversion and output adapters are excluded
and must be charged before model admission. The GEMM cache state is not
claimed to reproduce the immediate dequant consumer. Failures continue to
other families; `RESUME_RUN=/workspace/kpack-prefill-cost.XXXXXX` preserves
completed matching cases and retries missing M values. `ACU=0` explicitly
omits counters. Box compilation of our kernels is unnecessary; installed
DeepGEMM may JIT during untimed setup. No whole-run deadline is inferred
from kernel duration; progress estimates observed wall time.

Keep three prefill candidates: FQ GEMM alone, SF expansion+SF GEMM, and full
expansion+BF16 provider. `prefill_candidates()` marks missing components
UNMEASURED. This handoff measures the new dequant/BF16 pieces; a matched
FQ/SF GEMM remeasurement and model-adapter cost remain required before
selecting the winning production route. FQ must not be excluded from
prefill just because M is large.

## Original v1 measurement contract

SF metadata expansion must be remeasured before a measured large-M cost
policy can use it. Do not mix its historical bandwidth estimate with a
measured full-weight dequant time. The current experiment **never launches
GEMM**. The dense cuBLAS and grouped DeepGEMM measurements are separate next
steps, with their initialization/JIT/first launch excluded.

| Stage | Input read | Output written | Numerical contract |
|---|---|---|---|
| SF expansion | Canonical packed metadata units only | Two FP16 planes, each `[E,K/group,N]` | Existing production scale/zero rounding, bit exact |
| Full expansion | Canonical low/high code planes and packed units | BF16 `[E,N,K]`, contiguous K | Original GGUF FP32 multiply/subtract, final BF16 round-to-nearest-even |

The full path does **not** first round weights/scales to FP16. It uses the
existing canonical word/bit registry and adds a device-only BF16 consumer.
This changes neither the offline format nor existing GEMV/GEMM arithmetic.
It is an experimental library, not selected by the production heuristic.

M is not a dequant dimension. Reuse the independently measured `(q,N,K,E)`
cost for an M ladder only when the consumed expert set and expansion policy
are identical. `E=8` here is a supplied contiguous eight-expert slice, not
a timed GPU gather out of an E256 tensor. `E=256` expands every expert. Any
future sparse selection/remapping cost must be accounted for explicitly.

## Bounded candidates

SF config0 invokes the **unchanged production header kernel**, block256.
Config1 assigns 16 consecutive N columns per warp; config2 assigns 32,
both block256. Config3 uses the same N32 mapping with block128. The candidate
grid directly supplies the N tile, superblock and expert; it avoids dynamic
64-bit quotient/remainder decomposition. Shape limits: N divisible by256,
K divisible by the format quantum, `E<=65535`, `K/32<=65535`.

Full config0 is a K-major direct scalar reference. Config1/2 use an
N32xK32 tile, respectively block256/128:

- Each warp loads 32 consecutive b16 code words (64 bytes). A thread reuses
  its low/high words for the K+8 slots; scale/min products are reused across
  the group's values. Q5's distinct high-plane map is preserved.
- Shared `[32][33]` 32-bit cells transpose the expanded values. One CTA
  barrier precedes coalesced paired BF16 output stores. Local ownership and
  bank-address tests cover every tile cell, including both block sizes.
- Native ISA must contain BF16 conversion. Candidate coordinates must not
  lower to FP64 division. `native.json` retains all30 kernel symbol/opcode
  records, not a claim about dynamic counts or speed.

Every result includes an explicit first-warp/pass address model: unique and
requested bytes, 32/64/128-byte sector footprints and addresses. It includes
metadata and output, not only B. Actual allocations/guard offsets retain
128-byte alignment. The model does not replace ACU transaction counters.
Full BF16 uses FP32 arithmetic, **not** the FP16 magic-number fast-dequant
path: changing that precision would require a new numerical contract.

## Timing and bandwidth

Inputs and outputs rotate through separate rings. Input bytes alone exceed
2.25 times the runtime-query-verified L2 size; the output ring is also checked
against that threshold. Every timing graph traverses the complete ring twice.
Two initial graph replays are excluded. Config order alternates over three
rounds of five samples. No GEMM, H2D, D2H, allocation or standalone flush
kernel is inside the timed graph.

Report `effective_GBps = (useful_read_bytes + useful_written_bytes)/time`.
The percent denominator defaults to the operator's stated **2700GB/s** and
is recorded as such, not inferred from a device name. Effective bandwidth
is **not actual DRAM utilization**: dirty writes can complete after a kernel
event, and cache sectors can amplify or reduce DRAM traffic. Steady repeated
complete-ring traversals reduce this artifact; they do not prove every byte
was DRAM traffic during the event interval. ACU reports provide the separate
DRAM/L2/memory-issue evidence. ACU replay explicitly flushes caches and is
labelled differently from rotating-event timing.

No bandwidth percentage is an admission gate yet. A numerically valid but
low-bandwidth candidate remains diagnostic evidence for further tuning. Do
not substitute a target utilization for its measured time. A sum of isolated
dequant and GEMM times is a **cost estimate**, not measured end-to-end latency
or proof that the consumer has the same cache state.

## Run on box

```bash
git pull --ff-only origin develop &&
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_dequant_ppu_box.sh
```

No compilation or JIT on box. Default plan:

- 10 untimed numerical smoke cases: Q2..Q6, both expansion stages, E3.
- Q4/Q5 timings: five previously measured dense N/K families; all six
  previously measured grouped families (512x2048,512x3072,2048x512,
  3072x512,1024x2048,1024x3072), each E8/E256. Both stages:
  **68 stage cases,238 configuration timings**, each15 samples.
- ACU: original and winning config for Q4 dense5120x8192 and Q5 grouped
  2048x512/E256, both expansion stages, at most8 reports.

`QTYPES=10,11,12,13,14` extends timings to all five formats, not a larger
configuration product. `ACU=0` skips counters explicitly. `RESUME_RUN` points
to a previous printed run directory; device/runtime/harness/fixture-package
identity must match. Each stage/shape is a fresh process. Successful cases
are validated and reused; failed cases do not invalidate independent passes.
Result writes are atomic. The script preserves the calling Docker shell and
packages the JSON/raw logs/ACU reports into the printed `.results.tgz`.

Progress includes observed elapsed/remaining time for the timing phase.
Initial numerical-smoke rates are not a prediction of the much larger
fixtures, and ACU is a separate phase. No unmeasured whole-run deadline is
claimed.

## Local validation and next admission

The actual host C++ BF16 reader, including the word/metadata-reuse variant,
is checked against independent official-GGUF BF16 bytes for all five formats.
Negative zero-code tests must fail. The box gate separately requires complete
output, guards, finite values and independent metadata/weight oracles before
any timing. Local host correctness and PPU compilation do not certify PPU
execution or high bandwidth.

After reviewing these results, optimize the losing expansion stages using
ACU, then measure the installed cuBLAS dense and DeepGEMM grouped providers
separately on the same weight/row domains. Keep current SF estimates out of
that measured comparison until this new measurement is admitted.

# Q4 cold dense M2..8: SIMT and Tensor Core comparison

The returned run is numerically complete for **42/42** dense cases. Within
this bounded candidate pool, SIMT wins 15 cases and Tensor Core wins 27.
SIMT beats the raw-GGUF reference in 35 cases; the remaining seven are all
N512/K2048 and exceed the reference +5% margin. Runtime PASS is not a
performance pass or model admission.

Source `896b2b8`; archive `q4-smallm-ppu.3vCskO.results.tgz`, SHA256
`e850d8ee757ae33834d2a18d5c89632ea59164cce284e26df253d10525eb5057`.
The [review receipt](measurements/q4_smallm_20260913/review.json) and
[full 42-row table](measurements/q4_smallm_20260913/summary.tsv) preserve the
selected recipes, contemporaneous timing, round medians and source/report
identity. Production selection and kernels are unchanged.

## Observed implementation boundary

TC below means the best measured member of the five-parent TC union with
S1/2/4/8, not the old policy alone or a global-optimality claim. All six
paired rounds agree on the SIMT/TC winner for every case.

| N x K | SIMT faster, M | TC faster, M | M8 SIMT, us | M8 TC best, us |
|---|---|---|---:|---:|
| 512 x 2048 | 2..8 | none | 6.448984 | 11.596406 |
| 1024 x 5120 | 2..6 | 7..8 | 22.462692 | 16.060577 |
| 4096 x 2048 | 2..3 | 4..8 | 27.813125 | 13.499687 |
| 4096 x 4096 | 2 | 3..8 | 48.832500 | 18.515000 |
| 5120 x 8192 | none | 2..8 | 124.113137 | 30.332570 |
| 8192 x 5120 | none | 2..8 | 116.719430 | 28.734285 |

The current policy misses an important TC choice at N4096/K4096. It uses
TM64/TN128/TK256, WM64/WN16, stages2, DN64, S1, taking about35.4us.
TM8/TN64/TK256, WM8/WN16, stages2, DN64, S4 takes18.2..18.5us **including
the real reducer**. At M8 that is35.466876 ->18.515000us (-47.80%).
Its profiled producer has256 CTAs x128 threads,86 registers/thread and
38,928 shared bytes/CTA; the old choice has32 CTAs x256 threads,
238 registers/thread and110,736 shared bytes/CTA. This is evidence for
remeasuring the policy's tile/Split-K choice, not a new offline layout.

## Remaining small-family regression

N512/K2048 selected SIMT versus raw reference:

| M | SIMT, us | Raw ref, us | Difference |
|---|---:|---:|---:|
| 2 | 3.239141 | 2.930977 | +10.51% |
| 3 | 3.848438 | 3.484297 | +10.45% |
| 4 | 4.305625 | 3.830000 | +12.42% |
| 5 | 4.933594 | 4.679297 | +5.43% |
| 6 | 5.656719 | 4.873984 | +16.06% |
| 7 | 6.421680 | 5.808242 | +10.56% |
| 8 | 6.448984 | 5.861680 | +10.02% |

At M8, forced-cold ACU reports about0.649MB DRAM read for K-pack and
0.648MB for raw reference, but L1/L2 traffic is36.327MB versus5.068MB,
and vector loads49,152 versus18,432. The current small reader has16-byte
B lane requests separated by1,024 bytes:50%/25%/12.5% unique footprint
utilization at32/64/128-byte granularity. Metadata lane requests have4x
duplication. This motivates a transaction/decode organization experiment;
the counters alone do not assign all latency to one cause. It is not an
eightfold increase in DRAM weight bytes.

## Why multirow SIMT falls behind TC

The lifted SIMT kernel still assigns one row per CTA. It makes one launch,
but does not explicitly share decoded B across rows. At M8/N5120/K8192,
SIMT and TC producer each read about23.903MB from DRAM. L1/L2 traffic is
705.895MB versus43.033MB, with819,200 versus61,440 vector loads.
TC additionally has a reducer; that cost is included in event timing.
These observations are consistent with repeated per-row on-chip reads and
decode work in SIMT, whereas TC reuses B within its M tile. They do not
imply SIMT fetched eight complete weight matrices from DRAM.

ACU uses forced-cold per-kernel replay, so its durations are not the
resident full-call event timings in the tables. Its reducer DRAM reads
also do not model producer-to-reducer cache residency in the full call.

## Verification

- 847 source hashes, compiled package, all module identities and runtime
  libraries match the submitted source/build.169 child logs are hash-bound.
- 1,393 screen records include84 exact structural TC exclusions;
  1,309 screen records and1,620 confirmation records pass numerical checks.
  All30,845 event samples are finite and medians were recomputed.
- Independent GGUF errors are at most5.49122e-5 for raw reference,
  5.60404e-5 for SIMT and8.04269e-5 for TC, below0.005. Zero-code/zero-A,
  guards, unchanged-row-body/actual M1 controls and replay checks agree.
  The row-zero-alias negative is explicitly a host-oracle plant.
- All43 ACU reports imported successfully, with60 kernels. Symbols,
  launch geometry and72-CU device attributes match. No warning is present;
  exclusive device use is not independently established by these logs.
- Weight rotation exceeds2.25x64MiB L2;64MiB is corroborated by a runtime
  attribute. The separate properties field `sm=1` is not the physical CU count.
- Six alternating rounds x15 samples confirm each shortlisted candidate;
  maximum selected round-median span is1.47%. Setup/JIT/first graph use is
  excluded. Campaign time is802.42 seconds (13.37 minutes), excluding fixture setup.
- All arms accumulate FP32 with FP16 A. SIMT/reference output FP32;
  existing TC outputs FP16. Dequant rounding also differs for group-affine
  readers. This is an implementation comparison, not identical arithmetic.

## Next order agreed on 2026-09-13

1. Port the tuned reader variants into a real indexed MoE entry and test
   single-token and multi-token semantics before expanding the sweep.
2. Add GEMV and TC as competing implementations for the prior decode shape
   inventory. Retain historical TC winners, runtime Split-K and complete
   reducer/adapter cost. Raw reference is a comparator, not a shipping layout.
3. Update the heuristic from those results, not by extrapolating dense
   M2..8 boundaries to MoE or other qtypes.
4. Preserve MoE chain fusion and verify actual selected symbols in llama.cpp,
   then test steady-state numerical/model performance with first JIT excluded.

One token with top8 means eight different experts each receiving one row,
not dense M8 sharing a single B. The current `QKG_INDEXED` API already carries
GPU IDs, activation strides/channels and output slot order without a separate
gather/scatter. The optimized experimental readers are not yet represented by
the production v1 columns/warps/split recipe. Also, the present llama.cpp
`prepare_moe` rejects direct GEMV plans, so individually faster GEMVs can
disable the existing chain fusion; that compatibility must be tested, not
assumed when fitting the heuristic.

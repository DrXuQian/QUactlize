# Full BF16 expansion: packed-code shared exchange

## PPU result and composition

`kpack-full-packed-ppu.NU9nBv.results.tgz` (SHA256
`9904a22f22a194d12fef65021fb9cdcabf6f0d87fb9bd1981d7afdc27cd81648`)
passes all 170 timing cells and ten untimed config smokes. All 34 per-shape
best medians improve over contemporaneous c4/c5 controls. Six ACU reports
import successfully: shared bank-conflict totals drop to zero in all three
new-kernel anchors, with essentially unchanged DRAM read bytes.

All twelve Q4/Q5 E256 domains select c11. Q4 time decreases 21.15--22.95%
and reaches 67.52--69.58% useful MBU; Q5 decreases 13.44--16.32% and reaches
56.92--59.19%. Dense/E8 still need per-shape c10/c11/c12 selection. No other
qtype or unmeasured sparse expert domain is admitted by this experiment.
The [44-point composed prefill board](KPACK_PREFILL_COMPONENT_COSTS.md)
now uses these measurements; the production route policy is unchanged.

## Evidence and scope, 2026-09-14

The DeepGEMM Python return `kpack-deepgemm-python.XOCxcL.results.tgz`
(SHA256 `c31bc734a57459c06a03777d1da409cb5128a2481a937a85349606a8fc1da646`)
has 24/24 valid numerical/timing cells and 75 matching result-file hashes.
All ten harness source hashes match the delivered `ca3b96d` sources. The
12 full-dequant domains match the previous v2 receipts, including golden
weights and the PPU identity: PCI0000:08:00.0,72CUs,64MiB L2.

Both tokens2048 and4096 use all256 experts in the pinned top8 router.
Each provider call consumes16384 or32768 sorted rows, respectively. Thus
the E256 full expansion matches these particular active-weight sets. This
does not license charging E256 expansion to sparse routes: only the union
of consumed experts should be expanded. The current contiguous-slice
dequant ABI does not implement sparse E256 expert selection; that remains
open, and the E8 standalone cases are supplied slices, not a timed gather.

All output/zero-A/guard/changed-A graph checks pass. Worst condition-normalized
error is0.003886891, below0.005. Round-median spread is at most0.845%.
Provider JIT/first-use and graph upload are excluded. Equal-shape BF16 GEMM
times from Q4/Q5 weights differ by at most0.458%.

Representative times in microseconds, E256/top8, tokens2048:

| Source weights, N/K | Full dequant v2 | DeepGEMM BF16 | Isolated-cost sum |
|---|---:|---:|---:|
| Q4,512/2048 | 480.150 | 379.560 | 859.710 |
| Q4,2048/512 | 478.210 | 335.220 | 813.430 |
| Q5,2048/512 | 543.230 | 334.230 | 877.460 |
| Q5,1024/3072 | 1611.720 | 1111.590 | 2723.310 |

Across the24 cells, full expansion accounts for52.3--61.9% of that sum.
The sum is not measured E2E and excludes external routing/gather, activation
conversion and output adapters. FQ/SF GEMM components are still missing
from a matched three-way prefill comparison; no production choice changes.

The overall return is INCOMPLETE only because the final ACU capture aborts
with `std::out_of_range: map::at` in
`hgbit::InstructionLifter::CollectBlockEnterAndExitALL`. No DeepGEMM ACU
report was produced. Keep all24 valid timings; repeat only the profile
when its instrumentation path is resolved. The failure does not establish
a numerical kernel defect or a precise SDK incompatibility cause.

## Why another dequant reader

Current c5 already issues coalesced uint4 B loads and BF16 stores. A warp
reads eight fully occupied64-byte N segments; the output warp writes four
fully occupied128-byte sectors. It is not a scalar uncoalesced baseline.

The v2 Q5 N2048/K512/E256 ACU report shows184,567,168 DRAM read bytes and
482,013,696 write bytes, with8,388,608 shared-load and8,126,464 shared-store
bank-conflict counts. Occupancy is89.94% of the reported sustained-active
maximum. These counters identify work to reduce, not additive wall-time
fractions. The profiled writeback interval is not the rotating-event ring.

The v3 metadata-cache/swizzle trials reduced global load instructions but
did not broadly improve time. Retain admitted v2 c4/c5 as controls; do not
select v3 c6--c9. This experiment instead moves the transpose **before**
FP32 dequantization:

1. Load the same canonical packed low/high words using aligned uint4.
2. Publish four codes per32-bit shared cell. Q5's four selected high bits
   occupy bits19:16 of that cell; its offline high-plane fold is unchanged.
3. Read codes in K-fast output order, apply the original FP32 multiply and
   subtract, then BF16 RNE and uint4 output stores. No expanded values cross
   shared memory, and no scale-broadcast warp shuffles are needed.

The shared address function is CuTe-owned and used directly by host tests.
Tests prove complete unique ownership, aligned four-word vectors, correct
Q5 high bits, and wrong-slot/wrong-view negatives. The32-bank model counts
distinct addresses per bank (including multicast); hardware bank latency
still requires PPU ACU. This does not promise a particular MBU or speedup.

| Config | Tile N/K | Shared codes + affine | CTA order |
|---|---|---:|---|
| 4/5 controls | 32/128 | 16,512 B expanded cells | N-fast |
| 10 | 32/128 | 4,096 +1,024 B | N-fast |
| 11 | 32/256 | 8,192 +2,048 B | N-fast |
| 12 | 32/256 | 8,192 +2,048 B | K-fast |

All candidates have128 threads and one CTA barrier. K128 uses38/40 vector
registers for Q4/Q5; K256 uses56/58, with zero stack allocation. The compiler
pairs final BF16 conversions into `v.pcnvt.bf16x2`; this is not FP16 affine
arithmetic. Native static instruction counts for c4/c5 match the v2 package.
The independent library contains63 specializations and is approximately1MB.

## Box comparison

```bash
git pull --ff-only origin develop &&
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_dequant_ppu_box.sh full-packed
```

This fetches `prebuilt/ppu0010/kpack-dequant-v4` through Git LFS. No box
compile, GEMM, DeepGEMM JIT or SF retest. It compares c4/c5/10/11/12 across
the same34 Q4/Q5 weight domains:170 timing cells plus two untimed smokes.
The3x5 alternating-order, independent input/output-ring protocol is unchanged.
Three anchors collect old-best/new-best ACU: Q4 N5120/K8192/E1,
Q4 N512/K2048/E256 and Q5 N2048/K512/E256. Independent successful cases are
resumable; failures do not erase other results. The surrounding Docker shell
is preserved. Return the printed `results=...results.tgz` archive.

Local host and Python regression tests pass 64 cases. The final CUDA
development carrier directly compiles the candidate/control headers:
40/40 checks pass on RTX 5070, spanning eight original-GGUF fixtures and
five configs, including multi-expert and K768 domains, low/high plane
negatives and output guards. All 78,643,200 output elements match the
independent BF16 oracle. The [source-bound receipt](measurements/dequant_packed_5070_20260914.json)
records the final header hashes and separate CUDA/PPU binary hashes.
NVIDIA checks are numerical portability evidence only. The separate PPU
return summarized above supplies the measured device evidence.

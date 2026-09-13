# Independent weight-expansion measurements

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

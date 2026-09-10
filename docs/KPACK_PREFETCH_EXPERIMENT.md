# Cross-call weight prefetch experiment

Status: local host contracts and PPU SDK compilation only; device performance
and correctness are pending. No production kernel, selection, offline format,
or JIT cache key is changed. The helper lives under `dev/l2_prefetch`.

## Questions, kept separate

1. **Cache benefit:** initialize cache pressure, prefetch target weights, then
   time the target FQ call. Report prefetch time separately; also report total
   prefetch-plus-target time. A faster target alone does not prove net gain.
2. **Interference:** after the same pressure initialization, run the current
   FQ call and prefetch another weight allocation on separate streams. Record
   current completion **before** joining the prefetch stream. Then run the
   target call and report its time and the complete pair time.

Controls include no prefetch and a repeated-target warm-weight control. The
warm control also warms non-weight data and is not a weight-only proof.
Both real `ppu.prefetch.global.L2` hints and blocking `ld.global.cg.u32` reads
are tested. Local ISA inspection confirms `vmem.prefetch` and cached loads.
A completed hint kernel proves hint issuance, not completed cache fill or
persistence. The load control completes real reads and checks their checksum.
Neither control assumes that PPU cache hierarchy/placement equals NVIDIA's.

The measured subjects are the existing, PPU-admitted **Q4 compact TM8/S4**
(N512/K2048) and **Q5 compact TM8/S1** (N2048/K512), each E256/top8/M8,
max expert rows 1. Their immutable module hashes are copied into the helper
manifest, while the binaries remain in the postops package. No GEMM rebuild.
The whole native call includes GPU metadata, compact directory, GEMM and
the selected reducer. It excludes llama.cpp IDs/gather/scatter and CPU setup;
it is not a producer-only time or a full-model speedup.

The latest Q4 S4 ACU producer (`kpack-grouped-postops.YH4lgS`) used 256 CTAs
of 128 threads, 38,912 shared bytes and 90 registers/thread. Shared memory
limits residency to 6 CTAs/CU (37.5% theoretical occupancy); achieved
occupancy is 21.889%, waves/CU 0.592593 and DRAM throughput 13.526% of peak.
The older 128-CTA launch was S2. These are replay-profiled counters, not
evidence that a half-CU partition retains its performance.

## Experimental boundaries

- Current and target have separate full device allocations. Target contents
  are an expert permutation, not aliases of already-read current weights.
- Only the eight selected experts' low/high/unit planes are prefetched.
  Next experts are assumed known, not predicted. This models an upper-bound
  cache opportunity, not a deployable next-layer MoE router.
- Default prefetch grids are 4, 16 and 36 CTAs, 128 threads each, no shared
  memory. These are **not pinned CU counts**. Current GEMM scheduling is
  unchanged. A half-CU scheduling experiment remains a separate next step.
- Every timing replay first reads a separate nonconstant pressure buffer,
  at least four times the SDK-reported L2 size and at least 256 MiB by
  default. This is matched cache pressure, **not a proven architectural
  cache flush**. The buffer size is recorded and can be increased up to
  1 GiB with `--pressure-mib` for a sensitivity check.
- Each complete experiment is one captured graph, replayed once per timing
  sample. There is no Python launch gap inside the measured sequence.
  Output poison precedes every replay on the consumer stream. Independent
  GGUF output checks, unchanged output bits, prefetch coverage/checksum and
  guards are checked. We do not alter numerical tolerances.
- Events measure whole-call time windows. Their overlap does not prove
  that the GEMM **producer** and prefetch kernels ran concurrently: queued
  work and auxiliary kernels are included. Use Asys to inspect the actual
  kernel timeline before making a concurrency claim.

## Profiling and cache flushing

The default runner deliberately does **not** launch ACU. Existing operator
capture helpers use `--replay-mode kernel --cache-control all`, which erases
the intended prefetch state before consumer replay and can change concurrent
execution. Capturing each kernel separately does not fix this. Disabling
cache clearing alone also does not restore the original cache history when
only the consumer is replayed repeatedly.

Primary performance evidence is unprofiled GPU event timing. For supporting
counters, each profiler pass must re-execute the entire setup/prefetch/target
sequence without an intervening consumer cache clear. Validate the installed
ACU's application/range/graph replay support first; do not assume another
profiler's command line or cache semantics. Concurrent interference requires
a mode that preserves concurrency; otherwise use Asys for the timeline and
the unprofiled run for timing.

## Run on PPU

The only new binary is `prebuilt/ppu0010/kpack-prefetch-v1/libkpack_prefetch.so`.
The Q4/Q5 parent images are reused from `kpack-grouped-postops-v1/modules`.
The box needs the same SDK 2.1.1, Python >=3.11, NumPy and gguf. It does not
compile anything or require the six-library llama intake bundle.

From the Quactlize `develop` checkout, after pulling sources and LFS payloads:

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 \
bash tools/run_kpack_prefetch_box.sh
```

Use one idle device. The script is a child shell; a failure does not exit the
caller's Docker shell. Both independent cases are attempted, and failures
retain their logs. There are 15 configurations per case, 3 alternating rounds
and 11 samples: 990 measured graph replays total, plus setup/warmups. This is
a bounded experiment, not a shape/config sweep. Wall time has not yet been
measured on PPU; fixture construction and cache-pressure reads are additional.

Upload the printed `/workspace/kpack-prefetch.*.results.tgz`. It contains
raw event samples, separate `target_delta_pct`, `current_delta_pct`,
`total_delta_pct`, whole-call overlap windows, selected parent identities,
shared/residency resources and correctness receipts. A negative delta is
faster. No row is automatically promoted to production.

Local compile-only reproduction:

```bash
python3 tools/build_kpack_prefetch.py --sdk "$PPU_SDK" --output /data/fresh-prefetch-build
python3 -m pytest -q tests/test_kpack_prefetch.py
```

The output directory must be new. The builder creates one small auxiliary
object/library and inspects its PPU ISA; it never recompiles the GEMM parents.

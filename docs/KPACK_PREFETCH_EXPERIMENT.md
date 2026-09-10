# Cross-call weight prefetch experiment

Status: **PARKED** by user decision on 2026-09-10; resume llama integration.
The same-shape Q4/Q5 controls pass on PPU in `qmk0by` at `89a9e3e`.
The reported Q4-to-Q5 run `fBihrU` also passes; its raw archive has not yet
been reviewed locally. No production kernel, selection, offline format, or
JIT cache key is changed. The helper lives under `dev/l2_prefetch`.

## Reported Q4-to-Q5 result and deferred work

The supplied `fBihrU` summary reports 33 samples per arm:

| Arm | Current, us | Target, us | Combined, us | Combined delta |
|---|---:|---:|---:|---:|
| No prefetch | 20.80 | 20.28 | 41.24 | baseline |
| Concurrent hint, 16 CTAs | 22.00 | 18.52 | 40.88 | -0.87% |
| Concurrent hint, 36 CTAs | 22.12 | 18.52 | 40.92 | -0.78% |
| Concurrent load, 36 CTAs | 22.40 | 18.48 | 48.04 | +16.49% |
| Completed load before timing, 36 CTAs | 20.68 | 18.52 | 39.40 | -4.46%, preload excluded |

The hint makes the target about 8.7% faster but delays the current call,
leaving less than 1% combined improvement in these instrumented summaries.
The last row excludes preload cost and interference; it is not a deployable
net gain or a proof of complete L2 residency. The repeated-target warm
control (14.44 us) also warms non-weight state and cannot justify that gain
from weight prefetch alone.

Keep production prefetch disabled. If reopened: audit the raw archive,
validate net latency with fewer timing nodes, and inspect producer overlap
and cache behavior before changing scheduling. No additional prefetch box
run or CU-partition work is required for the current integration milestone.

## Q4 gate/up window, Q5 down prefetch

Run the two already-built projection operators with the same eight known
expert IDs: current Q4 N512/K2048/S4, target Q5 N2048/K512/S1. This is one
gate/up-shaped Q4 call followed by one down-shaped Q5 call, **not** a complete
gate-plus-up MLP. Down still consumes an independently checked synthetic input;
no SiLU, product, activation handoff, router, gather/scatter or llama integration
is added. The purpose is to measure a compute/prefetch window in isolation.

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 \
bash tools/run_kpack_prefetch_box.sh --case q4-to-q5 --blocks 16 36
```

This runs 12 configurations, 3 alternating rounds and 11 samples: 396 measured
replays, plus the small timer gate and setup. There is no rebuild or new binary.
The ordinary `pair` arms include the current call, any prefetch tail, and down;
their total delta is the actual instrumented sequence comparison.

An additional `primed-pair/load` arm finishes the same blocking weight reader
**before** the timing origin, then runs current and down. It excludes preload
cost and resource overlap intentionally. Its receipt says
`OPTIMISTIC_CURRENT_PLUS_TARGET_NOT_NET_LATENCY`; never promote its delta as a
net speedup. It primes weights without running a full target GEMM first.

All selected experts' low/high/unit ranges are traversed, at one request every
32 bytes. The load control reads one 4-byte word at each address and validates
its checksum; it does not read every byte individually. Range coverage and
completed loads are **not proof of complete L2 residency**, even for the primed
control. The 16/36 CTA choices change parallelism, not weight coverage or a
fixed CU partition.

## First box results: same-shape controls

`kpack-prefetch.qmk0by.results.tgz` contains 990 finite samples and passing
timer/numerical checks for Q4 and Q5. Runtime libraries match the build;
only compiler/inspector hashes differ. Recomputed summaries agree with the
archive. For concurrent hints with 16 CTAs:

| Case | Current, us | Target, us | Combined, us |
|---|---:|---:|---:|
| Q4 to Q4 | 20.60 → 21.96 | 16.56 → 15.36 | 37.48 → 37.80 |
| Q5 to Q5 | 20.16 → 21.40 | 16.52 → 15.20 | 36.80 → 37.00 |

Target intervals improve, but the current-call interference offsets them;
no net gain is established. These are not Q4-to-Q5 results. A repeated-target
warm control reaches 14.64/14.44 us, but also warms non-weight state and is not
a weight-only prefetch promise.

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

- Current and target have separate full device allocations. Same-format
  controls use an expert permutation; cross-projection controls use their
  respective Q4/Q5 fixtures. Neither aliases already-read current weights.
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

## Captured timestamp repair

The first box replay failed at `hggcEventElapsedTime` with status 1. The old
runner captured ordinary `hggcEventRecord` calls: capture dependencies are
not sufficient to supply replay-readable timestamps. Timing events now use
`hggcEventRecordWithFlags(..., hggcEventRecordExternal)` inside capture;
each must appear exactly once as an event-record node in the resulting graph.
The local PPU SDK declares that API and flag in `hggc_runtime_api.h` and
`driver_types.h`. The corresponding capture meaning is documented in the
[CUDA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).
The repaired timer passes on PPU in `qmk0by`; the supplied `fBihrU` log also
reports the added primed timeline passing, pending raw-archive review.

Fork and join use **different** ordinary capture events, with wait flags 0.
No external wait or dependency on an uncaptured setup event is introduced.
Before constructing the full weight fixture, the runner exercises these same
six timeline shapes using small device memsets, validates graph record nodes,
replays each three times and checks the writes. Its `*.timing.json` receipt is
saved on pass or failure. A failed interval names the arm, event endpoints,
query status and expected timing-node count; it is never treated as zero time.

Timestamp nodes have overhead and may perturb scheduling. All reported times
are instrumented full-call intervals, not overhead-free producer latency;
there is no subtraction of an assumed event cost. In particular, extra
prefetch timing nodes differ from the no-prefetch control. Confirm small gains
with a minimally instrumented end-to-end comparison before production use.
This repair changes only the runner: helper and GEMM binaries are unchanged.

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
The box needs PPU runtime libraries, Python >=3.11, NumPy and gguf. It does not
compile anything or require the six-library llama intake bundle. The published
images were built with SDK 2.1.1. Their original SDK digest is retained, with
per-file hashes for the compiler, inspector and four runtime libraries.

From the Quactlize `develop` checkout, after pulling sources and LFS payloads:

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 \
bash tools/run_kpack_prefetch_box.sh
```

Different or absent `hgcc`/`hgobjdump` tools do not block prebuilt execution
when all four runtime-library hashes match. Different runtime libraries still
stop by default, listing each difference. To explicitly try another installed
runtime, append `--allow-unverified-sdk` to the command above. This records
`UNVERIFIED_RUNTIME`, not SDK compatibility; missing runtime libraries, modified
device payloads and numerical failures are never bypassed. The `.sdk.json`
receipt is written before GPU initialization, even when admission rejects the
SDK. Successful case results also retain that receipt. Do not silently pool
timings collected under different runtime identities.

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

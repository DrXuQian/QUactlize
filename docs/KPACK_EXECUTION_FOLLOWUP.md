# K-pack execution follow-up

Updated: 2026-09-08. Development work; no new device admission or runtime
bundle is implied by this plan. Keep the canonical offline planes unchanged.

## Tracked delivery

| Item | State | Completion condition |
| --- | --- | --- |
| 3. Native selected-module binding | Pending | Full parent/build identity, cached handles, no Python or online timing in inference; explicit K-pack-capable miss path |
| 4. Device-only grouped metadata | Locally compiled / box pending | Additive `qk_device_call_v2` and query/prepare exports; GPU bounds -> shape/directory on each run, no rows_host or per-token D2H; ten existing FQ/SF parents compiled. Box checks empty experts and mutable graph replay against same-parent v1 |
| 5. ScaleFirst prefill | Prepass compiled / loader lifetime pending | Five production prepass specializations compiled; paired prepass+SF and resident measurements added. Weight-owned reuse, memory-budget admission and final model wiring remain pending |
| Decode GEMV | Locally compiled / box pending | Five canonical formats, dense/compact-grouped/indexed-grouped, eight explicit column/warp/split recipes. Direct indexed path has no gather/scatter. Existing GEMM comparison is explicitly pre-gathered core-only; model adapter/selector admission remains pending |
| Model decode regression | Open performance debt | Isolate the measured per-token gap with matched work; retain the old GEMM incumbent and do not attribute the whole gap to one missing algorithm |

## Reviewed GSM8K pilot

Archive `llama-kpack-gsm8k.EEZKMu.results.tgz`, SHA-256
`cb8b7906bd9af587e1c7d5b3dd98a50d6d76e16eced7823c37fd2619e923a624`,
records llama.cpp `a39917a66acd742ae02b2034549fded8ce9506bb`.
All 128 paired raw responses, route receipts and prompt/decode timers were
replayed against the summaries. Both processes exited zero. Each arm scored
122/128 (95.3125%), with no paired correctness flips. Each has one truncated
and unparseable answer, but those are different questions between arms.
Only 43 complete outputs are identical. This is scoped task-accuracy evidence,
not bit equality, full GSM8K, or a new device trace.

| Aggregate | Ordinary GPU reference | K-pack |
| --- | ---: | ---: |
| Input tokens | 14,012 | 14,012 |
| Generated tokens | 42,433 | 42,822 |
| Prompt time, s | 22.492235 | 8.467084 |
| Decode time, s | 343.420132 | 448.025652 |
| Token-weighted decode, ms/token | 8.093232 | 10.462511 |
| Request wall time, s | 366.056342 | 456.649109 |

Decode latency per token is 29.27% higher. The 43 identical-output pairs
still have a median paired increase of 29.17%. All cache accesses hit, no
GPU repack occurs, and both arms reuse graphs. The ordinary reference is
llama.cpp's GPU path, not the historical standalone Xplane board. Runs are
reference-first, not interleaved ABBA; no clock/co-tenant receipt or kernel
trace was collected. These timings establish an observed regression, not
its isolated kernel cause.

The existing sweep explicitly excluded canonical K-pack BC/GEMV
(`NO_CANONICAL_KPACK_BC_READER`). The current llama.cpp K-pack branch calls
FQ grouped GEMM even for one token, while its ordinary quantized branch can
use MMVQ. Add GEMV as a separate algorithm candidate, not as another GEMM
tile or an unmeasured default. Keep prior measurements and coverage labels.

## ScaleFirst preparation cost and lifetime

For E experts and logical weights N x K, let W = E*N*K, G be the scale
group size, and U the packed metadata bytes per 256-weight superblock.
The current ScaleFirst contract materializes **both** FP16 scale and zero
planes, including the canonical affine correction for Q3/Q6:

    packed metadata bytes      = W * U / 256
    expanded scale+zero bytes  = W * 4 / G
    ideal prepass traffic      = W * (U/256 + 4/G)

This is a minimum logical read/write volume, not measured DRAM transactions
or time. Actual duplicate loads, decode arithmetic, launch overhead and
allocation can increase cost. Weight-code planes are not expanded or copied.

| Format | G | U | Expanded metadata / packed code bytes |
| --- | ---: | ---: | ---: |
| Q2_K | 16 | 20 | 100% |
| Q3_K | 16 | 14 | 66.67% |
| Q4_K | 32 | 16 | 25% |
| Q5_K | 32 | 16 | 20% |
| Q6_K | 16 | 18 | 33.33% |

The reviewed model's cache manifest contains 80 Q4_K tensors at
N512/K2048/E256 and 40 Q5_K tensors at N2048/K512/E256. Every tensor has
16 MiB of packed metadata and requires 32 MiB of expanded metadata.
Across 120 tensors: 1.875 GiB read, 3.75 GiB written, at least 5.625 GiB
logical traffic. Keeping packed units for GEMV/FQ and the FP16 planes for
SF adds 3.75 GiB of resident device memory; it is not byte-neutral.
At a *hypothetical effective* 500 GB/s, traffic alone is about 12.08 ms for
all tensors. This is a scale estimate, not a PPU timing prediction or bound.

Long prefill amortizes metadata preparation over many tokens/requests. A
short MoE chunk may touch few experts while a full prepass expands all 256;
using active-expert bytes to price that full prepass is incorrect. Current
metadata preparation has no active-expert selection. A future partial cache
would also need initialization state and stream-safe publication; it is not
part of the first implementation.

The reviewed model has 256 experts and top-8 routing. At the measured
128-token submission size there are 1,024 routed rows, **four rows per
expert on average**, not M=128 for every expert. A longer prompt submitted
as 128-token chunks does not change that per-call mean. SF/FQ selection must
consider token chunking and expert work, not only the total sequence length.
The device-only gate therefore uses the caller-available max-rows bound 128,
not the tighter maximum visible only to its host oracle.

Implementation and measurement requirements:

1. Own metadata with the immutable weight artifact. Prepare once, outside
   graph capture, then record readiness and reuse across prefill requests.
   Decode GEMV continues to read packed units without requiring SF allocation.
2. Keep preparation and weight backcopy independent. Compute waits only for
   the device data it consumes, never for D2H/disk publication.
3. Record memory capacity before enabling SF. Do not eagerly allocate every
   expert plane without accounting for KV/workspaces and the extra 3.75 GiB.
4. Time prepass, resident SF GEMM, first-use prepass+SF, and FQ on identical
   expert routing. Include lazy allocation in model first-use wall timing,
   not in the kernel-only prepass timer. Cold-D2H publication is separate.
5. First-use SF wins only if `prepass + SF < FQ`. Over R reused calls, compare
   `prepass + sum(SF)` with `sum(FQ)`. Do not train the first-use selector on
   resident-only timing or charge preparation again on every request.

## Locally compiled box package

`prebuilt/ppu0010/kpack-execution-v1` contains the 560,232-byte execution DSO
and ten separate grouped parents, 2,176,200 bytes of DSOs total. The execution
DSO SHA-256 is `53165c9ec37a7f73057b31289434c90a2a4da0065f37c93ca78208cecd4ba1de`.
Builds use the local PPU SDK 2.1.1; no PPU execution is claimed. All manifest
flags remain `device_validated=false`, `heuristic_admitted=false`.
Local validation: 75 reader/metadata/fixture/selected-policy tests pass;
the historical fixture and its calibration hashes remain unchanged. The
execution ELF contains all 20 format-specific GEMV entry kernels, five
reducers and five prepasses. All ten grouped DSOs expose the old identity
and new query/prepare exports. No old six-library bundle is replaced.

The ordinary v2 grouped route uses a conservative per-expert 3-D grid bound,
whereas v1 can use an exact host-derived flat grid. Persistent v2 rebuilds
the compact device directory. Correctness of the same parent does not make
these scheduling costs equal: the gate records v1/v2 resident timings, and
v2 ordinary does not inherit old flat-grid performance admission.

Run `bash tools/run_kpack_execution_box.sh` after pulling the LFS package,
with the existing `PPU_SDK` and `QUACTLIZE_PPU_BUNDLE`. This performs **no
compilation** and does not run or modify llama.cpp. It preserves the Docker
shell and archives even incomplete results. Both independent gates run:

- GEMV: 32 weight geometries / 81 work points / eight recipes, three rounds
  of 11 samples. Five formats, real dense families and the Q4/Q5 E256 model
  anchors; signed activations, all-output GGUF oracle, output guards and
  zero-low negatives. Compare with the current legacy GEMM incumbent; this
  is not proof against every optimal GEMM from the historical search.
- Grouped: ten parents / fourteen parent-shape cases, three routing profiles
  each. Same-parent v1 raw equality, independent GGUF tolerance, output guards,
  and captured replay with changed GPU bounds. SF reports first prepass,
  resident GEMM and repeated prepass+GEMM. The latter is not cold model
  allocation/JIT/TTFT and must not be labeled as such.

Model anchors use the real N/K/E/top-k/chunk sizes with reproducible synthetic
weights/routing, not IDs captured from GSM8K. Before promoting a GEMV rule,
also compare it with the already-measured optimal GEMM candidate for the same
work (including supported adapters). Beating the current legacy default alone
does not establish that broader algorithm choice.

Return the printed `kpack-execution.*.results.tgz`. Results do not automatically
change the heuristic. C++ selected-module binding and final llama.cpp route/
metadata ownership are still open; the current model route is unchanged.

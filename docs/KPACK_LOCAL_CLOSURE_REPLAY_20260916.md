# PPU closure replay: Q8 results and the remaining test failures

Archive: `/root/kpack-local-closure.wqNPz5.results.tgz`, SHA256
`74bfe0d324f550a2d9b7e83a6bc27798fc9f4ece4853a87c4b301c698363f06f`.
[Machine-readable review](measurements/local_closure_replay_ppu_20260916.json)
contains all 50 Q8 comparisons, case-file hashes and the deterministic Q4
arithmetic replay. Production execution remains
`1122afabaa9c8b749bb88a12281887c2635155468d3048956986b3befeb9be43`.

| Gate | Returned result |
| --- | --- |
| Q8 baseline/candidate numeric | 12,480/12,480 PASS, maximum normalized error 2.66e-8 |
| Q8 rotating performance | 50/50 complete; candidate median lower in 48 cases |
| Independent TC/SIMT/range and selected-Q4 | All pass again; selected Q4 covers 258 BF16 cells |
| Complete BF16 gate | 733/746 pass; one Q4 case fails and 12 later Q4 cases are unexecuted |
| Q6 complete chains | 18/18 PASS; scalar-oracle repair confirmed |
| MoE prepare experiment | Stops at an unobserved row-map negative; no new helper timings |

## Q4: the same-input hypothesis is excluded; rounding order explains the result

The downloaded SwiGLU input hash equals the independently reconstructed host
input hash. `swiglu_max_bf16_ulp=0`. The previous change to use the actual
input therefore does **not** explain these eight discrepancies.

The bounded CPU diagnostic
`python -m dev.bf16_compute.rounding_replay` reconstructs the original fixture
from the official GGUF decoder and emulates V3/C4/W4/P4/S1's FP32 FMA and
K-reduction order. After the specified BF16 projection boundary, its entire
32,768-element output has exactly the returned PPU hash:
`1d95c4dfa905fe0845673a519e4cfd3109238d9edec7d484262d4f4434f0434e`.

For row32, column370:

| Quantity | Value |
| --- | ---: |
| Independent unrounded F64 dot | 0.07104491675272584 |
| Native-order FP32 dot | 0.071044921875 |
| BF16 rounding midpoint | 0.071044921875 |
| Separately rounded host golden | 0.07080078125 |
| Native-order BF16 result / PPU result | 0.0712890625 |

The FP32 dot differs by only 5.12e-9 but reaches the BF16 midpoint and rounds
to the even upper value. The independent F64 dot is below the midpoint and
rounds down. The fixture reuses one expert pattern for token4's eight slots,
so the same column fails in rows32--39. The replay reproduces exactly those
eight indices and the old 0.0066948020 normalized error.

Across all outputs, native-order FP32 error is 1.23e-7; the BF16 output's
error against the unrounded independent dot is 0.0037376954. The original
0.005 threshold admits that result. Comparing two separately rounded values
can approach a full BF16 ULP instead of the rounding error relative to the
underlying dot.

The gate now compares grouped and MoE projection error to the independent
unrounded dot. Composition still rounds the independent gate/up projections
to BF16 before SwiGLU, and rounds down before weighted finishing. Actual
down input, ordered finish, changed graphs, no-clipping range checks and
the whole-chain 0.02 bound remain. No tolerance or production kernel changed.
Wrong-column and zero-output negatives still fail. Generic SIMT and the
outlier gate already used the unrounded oracle and need no change.

This diagnoses the returned stage failure, not the unexecuted remainder of
the complete chain. Its other repeat and later cases still need device replay.
The CPU replay is a diagnostic with a returned-output hash, not a replacement
golden and not a raw-F32 device capture.

## Prepare: order the negative fixture before its consumer

The failure is `tokens=6 k=3072 mask=7 merged=0 router=2 arm=1 repeat=0`:
`wrong row-map negative accepted`. It occurs in BF16 based on the loop order;
the old log did not distinguish its two alignment cases. Normal row maps,
router output and composed SwiGLU were checked before the negative.

The negative used pageable `cudaMemcpy` on the default stream, then launched
SwiGLU on a nonblocking stream. There is no ordering edge between them.
The installed SDK's `CUDA_SDK/include/cuda_runtime_api.h` synchronization
documentation explicitly permits pageable H2D to return before final DMA
completion (lines75--78); nonblocking streams do not synchronize implicitly
with stream0 (lines1881--1883). Prior passes are not proof of a valid negative.

The test now uploads and restores the altered map on the consumer stream,
waits for that untimed fixture copy, and verifies its device readback. It
also requires that the negative golden differs, and that the actual output
is exactly the corresponding row swap. The context now prints compute type
and alignment. A host delayed-DMA model reproduces the missing-edge negative
and passes using the actual corrected upload method.

This is a proven test-ordering defect consistent with the intermittent
failure; confirmation still requires the PPU replay. No production prepare,
SwiGLU, GEMV, GEMM or benchmark timing body changed. The wrapper now prints
the actual binary failure tail instead of only a generic admission error.

## Q8 performance: compare complete calls, not just producers

Same canonical K-pack2, F32 storage, FP32 accumulation, explicit FP16/BF16
compute. Each arm includes its real Split-K reducer when selected. Case hashes,
DSO/runtime/PCI identity, 6x15 finite samples and recomputed medians agree.
Every active-weight ring exceeds 2.25x the operator-verified 64MiB L2; the SDK
query returns zero. The reported SM count is not usable as occupancy evidence.
External GPU-idle admission was not recorded. These are old/new K-pack
readers, not llama-native or tensor-core comparisons.

Dense M1 medians, microseconds:

| N x K | FP16 old/new | BF16 old/new |
| --- | --- | --- |
| 512 x 2048 | 5.971 / 5.732 | 5.738 / 5.673 |
| 1024 x 2048 | 6.592 / 6.539 | 6.469 / 6.481 |
| 1024 x 5120 | 10.524 / 9.735 | 10.470 / 9.683 |
| 2048 x 512 | 5.010 / 4.834 | 5.056 / 5.149 |
| 2048 x 4096 | 12.091 / 10.864 | 12.025 / 10.845 |
| 4096 x 2048 | 10.526 / 9.249 | 10.580 / 9.285 |
| 5120 x 8192 | 34.360 / 30.523 | 34.311 / 30.551 |
| 8192 x 2048 | 16.369 / 13.747 | 16.271 / 13.991 |

All 17 FP16 dense contexts improve; 15/17 BF16 dense improve and two small
M1 contexts regress by 0.2%/1.84%. All 16 indexed contexts improve by
11.4--32.2%. Preserve per-case baseline choices for regressions; tiny changes
are not statistically established wins. Use the matched medians as reader
evidence, not a model TPOT claim. No production selector was changed here.

## Bounded retry

Retain the Q8 results and selected-Q4 evidence. The local closure supports
`LOCAL_PHASES=moe-prepare,bf16`, records that scope and does not call omitted
phases passed. The repaired prepare test binary is small; all production
DSOs, TC modules, Q8 binary, packer and caller remain unchanged.

Model BF16 admission and production promotion of the new Q8/prepare readers
remain separate follow-up work.

Repair source `47e5c89762d6f6781c7707335827cdd88b95cbba`, artifact
`b9154a339524752b725a86d727809b24df52bc6b`. Related local regression:89 tests
pass; the later ID-negative ordering scope check also passes. The repaired
PPU helper compiles, and package verification passes with the unchanged
production execution hash. Only one new 1.3MiB test executable is transferred.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    test "$(git branch --show-current)" = develop
    GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop
    git submodule update --init third_party/actlize
    unset BUNDLE
    LOCAL_PHASES=moe-prepare,bf16 \
    PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    CUDA_VISIBLE_DEVICES=0 \
    bash tools/run_kpack_local_closure_box.sh
)
```

Upload the printed `.results.tgz`. The selected phases run all3,840 prepare
contexts and all746 capability cases, without repeating Q8's50 performance
contexts or compiling on box. The preceding replay spent197 seconds in
the BF16 gate; a roughly4--6 minute window is an estimate, not a deadline.

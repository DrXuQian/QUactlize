# Local decode work and PPU handoff

This is development work, not admission to `main`. Canonical weight bytes
and production Q8/prepare implementations are unchanged. Q8_0 already uses
K-pack2: two 8-bit codes per word, one FP16 scale per 32 codes, no high plane.
FP16 and BF16 consumers share those bytes; computation type is explicit.

## Completed locally

| Work | Evidence | Remaining PPU check |
| --- | --- | --- |
| Matched small-M selector and llama caller | 1,842 exact rows, 634 bounded buckets; independent FP16/BF16 keys; all generated TC tickets/geometry/Split-K checked in the real C++ dispatcher | Selected mixed chains, then actual model PP/TG and trace |
| BF16 Q4 specialized SIMT and Q2/Q4 packed-A TC | Typed exports; FP32 accumulation; large finite input preserved; 40 TC modules and the full execution DSO compile with the PPU SDK | 746 capability cases and the selected Q4 gate below |
| Q8 vector reader experiment | RTX5070: 12,480 numerical cells, 50 matched cold performance contexts; full NCU reports imported | Same old/new readers on PPU, including actual S>1 reducer |
| MoE prepare experiment | RTX5070: 3,840 candidate contexts including changed graphs, BF16 range, weak alignment and actual SwiGLU composition | 48 matched helper comparisons; model composition before any default change |
| Delivery | Small dispatcher/execution rebuilt locally; prefill DSO and weight packer reused; no llama binaries included | Caller uses its `.aoneci` build on box |

The table import audits all 169,896 sealed raw receipts from
`smallm-closure.e1WvIL.results.tgz`, not just the summary. See
[selection contract](KPACK_SMALLM_TABLE.md). Small-M excludes full dequant.
TC comparisons already include the measured reducer; it is not added twice.

## Q8 measurements: NVIDIA guidance only

Same K-pack2 bytes and FP32 dot order. The candidate vectorizes adjacent-N
scale loads, halves live packed-B words and avoids Q8's unused affine-zero
work. Two-byte scale alignment still has a scalar fallback. This changes
several implementation details, so the timing is not proof that any single
one is the sole cause.

Dense M1, F32 storage / FP16 compute, rotating active weight addresses:

| N × K | Old best, µs | Candidate best, µs |
| --- | ---: | ---: |
| 512 × 2048 | 4.084 | 3.827 |
| 1024 × 2048 | 5.677 | 5.244 |
| 1024 × 5120 | 11.446 | 11.021 |
| 2048 × 512 | 3.871 | 3.379 |
| 2048 × 4096 | 15.886 | 15.740 |
| 4096 × 2048 | 15.657 | 15.644 |
| 5120 × 8192 | 72.603 | 72.143 |
| 8192 × 2048 | 29.951 | 29.683 |

The 50 contexts also cover BF16 compute, M2/4/8, indexed shared-A and
per-slot-A. Each arm screens 120 recipes; its two finalists get six
alternating rounds of fifteen samples. A complete Split-K call includes
its reducer. Cache ring capacity is at least 2.25 times physical L2.
Small changes are not statistically established wins. These are not TC,
llama-native or PPU comparisons; GPU idle admission was not established.

NCU, identical M1/N2048/K512/C8/W4/P4/S1 recipe:

| Metric | Old | Candidate |
| --- | ---: | ---: |
| Registers / thread | 72 | 56 |
| Executed global load instructions | 5,376 | 4,608 |
| L1 global-load sectors | 45,056 | 38,912 |
| DRAM read, decimal MB | 1.164544 | 1.142016 |
| Forced-cold profiler duration, µs | 4.928 | 4.288 |

The address model reports low/A/metadata footprints at 32/64/128 bytes;
source footprints, L1 sectors and physical DRAM traffic are different
quantities. NCU durations are not substituted for rotating event samples.

## MoE prepare: avoid work that its consumers do not need

Prepare does not move weights. The mixed-TC candidate shares token routing
and source conversion and vectorizes aligned activation copies. The all-SIMT
candidate keeps token/slot order and emits identity row maps: it does not
sort rows, build TC descriptors, or replicate A for a TC consumer that is
absent. The actual SwiGLU helper is included in the numerical composition
check; identity row maps are a private chain protocol, not an offline format.

RTX5070, BF16, K2048, merged gate/up; warm helper event medians:

| Tokens | Old all-SIMT, µs | New all-SIMT, µs | Old mixed SIMT/TC, µs | New mixed SIMT/TC, µs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.453 | 2.136 | 2.763 | 3.263 |
| 2 | 4.815 | 2.133 | 4.826 | 4.446 |
| 4 | 5.070 | 2.160 | 5.181 | 5.439 |
| 8 | 16.473 | 2.318 | 16.473 | 8.530 |

Do not enable this globally: mixed M1/M4 regress on this GPU. Full NCU
all-SIMT M8 shows executed global loads 4,096 → 64, but registers increase
52 → 117. Measured DRAM reads increase 60.160 → 80.896 KB under replay;
the lower instruction count is not a claim of proportionally lower DRAM
traffic. PPU occupancy/register effects still need ACU.

The matched numerical/timing denominator is **46/48**, not 48/48. The old
F16 M1 all-TC gather control faults on NVIDIA for K512/K2048. BF16 controls
and the candidate pass. Minimal inline/helper carriers narrow the difference
but do not prove a compiler bug or a PPU fault. `dev/moe_prepare/gather_probe.cu`
retains the diagnostic; no shipping helper was rewritten to hide the fault.
The candidate's independent 3,840-case gate passes. A test-fixture default-
stream poison race was fixed in untimed setup; no inference CPU wait was added.

Compact raw samples, source/image hashes, numerical receipts and imported
NCU metrics are in `measurements/local_optimizations_20260915.json.gz`.
Full `.ncu-rep` files are retained locally; they are not needed by production.

## One prebuilt box entry

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git pull --ff-only origin develop
    git submodule update --init third_party/actlize
    PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
    bash tools/run_kpack_local_closure_box.sh
)
```

Use an idle PPU-ZW810. This entry runs **no compilation, model or full
config campaign**. It fetches the pinned LFS package, then Q8 numerical /
50-context performance, MoE numerical / 48 comparisons, the 746-case BF16
capability gate, and selected Q4 BF16 coverage. Independent failed processes
do not invalidate completed peers. Return the printed `.results.tgz`.

The Q4 gate covers 192 actual selector requests: 129 select SIMT and 63
correctly decline to TC. It executes 258 BF16 storage/compute cells,
129 real F16 v1/v2 controls and 129 out-of-range F16 negatives. Those 63
declines are not counted as device-compute passes. It reuses the exact model
execution DSO, including dense and indexed/shared/per-slot readers and merged
gate/up shapes. All40 compiled selected reader recipes are exercised.

Remaining after the device return: admit/reject experimental helper/reader
domains, then model numerical checks and warmed PP2048/TG timing with the
matched selector. No TPOT saving is claimed by adding isolated kernel times.
Qwen3-32B range correctness, output-head N128 tail admission, routing-sensitive
M2--8 choices, residual fusion and prefetch remain explicit debts.

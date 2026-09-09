# Q4 grouped decode: coverage and isolated replay

Reviewed 2026-09-09. This is a development diagnostic, not a new production
algorithm, format, heuristic, or admission of global optimality.

## The shape was measured

The Q4 gate/up operation is N512/K2048, E256, tokens1/top8: eight active
experts with one row each and 248 empty experts. Total rows8 does not mean
one expert with M8. The original layout plan contains
`grouped_t1_n512_k2048_e256_top8`.

The overnight archive `kpack-overnight-c0c1361-results.tgz` has SHA256
`03e81a5e53e445d5ba4f2813f087775f6b3eab17673c0f801673d51fd8e938f1`.
Request `c517a2fcaf2f0080c3efdc67bc003d9308acb82b2111410df0318fa2808cd0d4`
has 42 measured screen runtime cells, 19 measured neighbor cells (overlap is
possible), and eight configurations in each of three confirmation rounds.
Both TM8/TN32 and TM8/TN64 were confirmed; TM8 was not omitted.

| Confirmed nonpersistent parent (WM/WN, stages, DN) | Pooled median us |
| --- | ---: |
| TM16/TN64/TK256 (16/16, 2, 64), winner | 18.440000713 |
| TM8/TN64/TK256 (8/16, 2, 64) | 19.200000912 |
| TM8/TN32/TK256 (8/16, 2, 32) | 19.479999319 |

The current selector returns the historical winner, not an arbitrary default:
`fqg_q12_l1_tm16_tn64_tk256_wm16_wn16_s2_ap0_dn64_nonpersistent`, build
`7a68e2b3e1ef3e4a8d2f861748931f97fb927bc9b7b4186af9fb5bb229ae84d6`.
The native package retains five Q4 FQ grouped parents, none TM8. That alone
does not prove a sweep omission: the package is a selected closure, while
the historical evidence includes TM8. It also does not prove that the old
winner remains best under the new device-only scheduling contract.

The bounded GEMM search is not every possible implementation. In particular,
the tuner skips dense BC; grouped Split-K/BC are structurally unavailable in
that harness. The later indexed GEMV is a distinct implementation, not the
older retuned `gemv_lowbit` kernel. Its eight recipes do not cover every
vectorization, dequantization, scheduling, or fusion option of that family.

## Recent measurements and their scopes

`kpack-native-model.O0ki3q.results.tgz`, SHA256
`8afcd682676625990105770495a99f486ea52dc5f05ea798e1936a647681a766`, contains
both this exact geometry and the same selected parent:

| Source | us | Scope |
| --- | ---: | --- |
| Native GEMV gate, FQ incumbent | 23.800000548 | Pre-gathered FP16 A/output; includes GPU metadata, excludes llama adapters |
| Native GEMV gate, best indexed GEMV | 28.440000489 | F32 indexed input/output, including reduction; excludes the FQ adapters by design |
| User-pasted model trace, main GEMM | 21.960 | Main kernel only, profiled |
| Same trace, metadata | 1.560 | Separate descriptor kernel |
| Same trace, IDs + gather + scatter | 7.200 | Additional adapter kernels |

These are not equal-scope GEMV/FQ comparisons. The all-14-context GEMV gate
retained FQ, but that decision must be revisited with full adapters included.
Do not add metadata twice: it is already included in the 23.8-us incumbent.
The reference Q4 MMVQ symbol enables fusion and uses Q8_1 activations;
equal-work standalone and fused-path comparisons remain separate tasks.

Necessary active weights are 4 MiB codes plus 0.5 MiB packed metadata.
Dividing by 21.96 us gives 214.9 GB/s, or 7.77% of the nominal 2766 GB/s.
Even the historical 18.44 us is only 255.9 GB/s / 9.25% on this byte model.
Neither number is a measured DRAM counter. They establish a low effective
byte rate, not its cause. Recovering 18.44 us is not the performance goal.

Historical 40%+ figures exist, but must retain their geometry and byte scope:
BACKTEST D6 (42.2%) is dense M1/N8192/K5120; the older GEMV decode band uses
eight active experts at N2048/K2048. Neither is this N512/K2048 case.

The older **grouped tensor-core** decode record is 20.74 us / 37.5% modeled
HBM-equivalent throughput at eight active one-row experts, N2048/K2048,
int4 gs32; its separate ACU capture is 19.55 us / 38.94% DRAM throughput
(`dev/fold_derivation/TODO.md`, decode winner table). These are explicit
historical records, not a certified all-time maximum across every archive.
The 47.4% entry is the retuned SIMT GEMV, not this grouped tensor-core result.

An earlier same-geometry **ScaleFirst** anchor also exists: S068,
N512/K2048/E256/top8, 11.020 us kernel-only / 17.3% modeled MBU, using
`16x32:256 w16x16 s3` (`.coord/INBOX.md`, item 115). BOX item 113 separately
records 11.122 us kernel-only versus 20.62 us host wall for this geometry and
5,283,840 distinct ScaleZero operand bytes. Thus 18.44 us is the confirmed
K-pack FQ winner, not the fastest historical result for every route at this
geometry. Old timing mode, metadata representation, tiles and source epoch
differ; this must not be called a measured same-kernel regression or used
to subtract an invented fixed overhead.

### Fully-quantized-only bandwidth cross-check

Recomputing necessary code/unit/A/output bytes over the confirmed overnight
time (nominal 2766 GB/s; not actual DRAM counters) gives:

| Q4 FQ route | N/K | Rows | us | Effective MBU |
| --- | --- | --- | ---: | ---: |
| Dense S1 | 25600/5120 | M1 | 49.159999937 | 54.27% |
| Dense S4 | 8192/5120 | M1 | 20.400000736 | 41.86% |
| Grouped S1 | 3072/512 | E256, 8 active x M1 | 13.679999858 | 18.86% |
| Grouped S1 | 512/2048 | E256, 8 active x M1 | 18.440000713 | 9.33% |

The S4 numerator counts necessary operands, not extra partial/reduction
traffic. The grouped denominator counts only active experts' weights, not
all 256. The 9.33% includes A/output; weight-only is 9.25%. High FQ bandwidth
is therefore present in the archive, but these high rows are dense, not the
current grouped geometry. In the earlier `fq-kquant-heuristic-handoff.tgz`
layout comparison, the same grouped geometry was FQ K-pack 23.96 us versus
FQ Xplane 24.36 us. It was not previously a demonstrated 40%-MBU FQ case.

The same O0ki3q native gate has pooled medians FQ 24.48 us and SF 19.48 us
for N512/K2048/E256/rows8, each including its GPU metadata/directory. SF is
20.42% faster in that paired gate, not the cross-epoch ratio of old 11.02 us
against FQ 18.44 us. Different selected configurations still prevent
attributing this entire difference to metadata decoding alone.

## Isolated same-parent test (no compilation)

`tools/run_kpack_grouped_decode_probe.py` consumes the existing native bundle.
It compares the selected production v2 call against v1 on the same module,
same resident weights, broadcast activation, expert IDs and output contract:

- `device-only`: GPU metadata plus the rectangular grouped GEMM. For the
  current Q4 parent, the source-derived grid is 1x8x256; only 64 of 2048 CTAs
  have output tiles. Empty CTAs return before the collective weight loads.
- `host-compact`: host metadata prepared once, outside timing, with the
  source-derived compact grid 8x8x1. This is a diagnostic comparator, not a
  production solution for dynamic GPU routing. It removes both descriptor
  launch and empty-grid work, so the difference is not a grid-only effect.

Profiler launch dimensions/counters are still needed for hardware confirmation
and main-kernel-only timing. The probe reports graph-batched event time per
call: 16 calls/graph, three alternating rounds of 11 samples by default.
The fixture is synthetic, numerically checked against official GGUF, and
reuses fixed resident IDs/weights without a cache flush. It is not a model
tensor dump or a cold-HBM performance measurement. Both arms must also agree
bitwise, preserve guards, and agree between eager and graph replay.

```bash
python3 tools/run_kpack_grouped_decode_probe.py \
  --sdk /workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  --case q4-up --output /workspace/kpack-q4-single-new/results
```

Use a fresh output directory. `--case q5-down` covers N2048/K512;
`--arm device-only --graph-repeats 1 --samples 3 --rounds 1 --warmups 1`
is a short isolated target for Asys/ACU. Keep profiled time separate from
unprofiled event measurements. Return `summary.json` and the console log.

Local validation covers orchestration, matching pointers/parent, distinct
v1/v2 row contracts, graph timing normalization and rejected invalid samples.
The full Q4 fixture was generated locally and the zero-output negative was
detected. No new PPU execution has been performed locally.

Next: measure the two paths on the same box; then compare suitable K-pack
GEMV and reference MMVQ with equal work, including any gate/up fusion. Re-rank
a small decode-specific candidate set under the actual runtime contract.
Neither the empty grid nor a missing algorithm is yet proven to explain the
whole model regression. Do not declare this closed on reproducing the old
low-efficiency GEMM winner.

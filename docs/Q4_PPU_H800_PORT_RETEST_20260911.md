# Q4 H800-reader port: PPU retest

The uploaded `q4-h800-port-ppu.4sc46w.results.tgz` completes numerical
validation and timing, but **does not close PPU performance parity**.
K-pack is within 5% of Xplane in 9/12 cells and of the supplied FP32
raw-GGUF reference in 3/12 cells. The joint gate passes 3/12.

## Measured times

Dense M=1, Q4_K, FP16 A, FP32 accumulation/output, one complete kernel,
no separate reducer. Times are microseconds, median of six round medians,
15 event samples per round. First launches and graph upload are excluded.
Positive deltas mean K-pack is slower.

| N | K | Cache | Xplane | Raw reference | K-pack | vs Xplane | vs reference |
|---:|---:|---|---:|---:|---:|---:|---:|
| 512 | 2048 | warm | 2.901 | 2.429 | 2.739 | -5.58% | +12.79% |
| 512 | 2048 | rotating | 3.452 | 2.487 | 2.937 | -14.92% | +18.08% |
| 1024 | 5120 | warm | 5.535 | 4.865 | 5.788 | +4.57% | +18.97% |
| 1024 | 5120 | rotating | 6.864 | 5.306 | 6.910 | +0.68% | +30.23% |
| 4096 | 2048 | warm | 5.958 | 5.756 | 5.818 | -2.34% | +1.09% |
| 4096 | 2048 | rotating | 6.984 | 6.509 | 7.098 | +1.62% | +9.05% |
| 4096 | 4096 | warm | 9.747 | 9.436 | 9.520 | -2.33% | +0.89% |
| 4096 | 4096 | rotating | 11.241 | 10.554 | 11.765 | +4.66% | +11.48% |
| 5120 | 8192 | warm | 20.621 | 20.269 | 19.797 | -4.00% | -2.33% |
| 5120 | 8192 | rotating | 22.922 | 22.542 | 24.927 | +8.75% | +10.58% |
| 8192 | 5120 | warm | 19.752 | 19.851 | 23.329 | +18.11% | +17.52% |
| 8192 | 5120 | rotating | 22.475 | 22.373 | 27.817 | +23.77% | +24.34% |

These are frozen **H800-selected K-pack recipes**, not PPU-tuned K-pack
winners. Xplane screens 12 recipes and the raw reference screens 60, then
confirms the best two on PPU. The arithmetic boundary is also explicit:
medium/large K-pack use FP32 group-affine evaluation, while the controls
reconstruct each weight in FP16 before FP32 accumulation. This is an
implementation comparison, not a pure layout-only comparison.

The maximum span among any winning recipe's six round medians is 1.51%.
The larger regressions are well outside this observed variation; this is
not a statistical confidence interval or proof of system-wide exclusivity.

## ACU findings

Nine reports imported locally with SDK 2.1.1. They use forced-cold kernel
replay, so their durations are **not** the benchmark's warm/rotating times.
At N=5120, K=8192:

| Counter | Xplane | Raw reference | K-pack |
|---|---:|---:|---:|
| Grid × threads | 320 × 256 | 320 × 256 | 320 × 256 |
| Registers/thread | 80 | 100 | 58 |
| Shared bytes/block | 16,384 | 16,512 | 512 |
| Achieved active-warp occupancy | 53.24% | 54.36% | 54.24% |
| DRAM bytes read | 23,635,328 | 23,638,912 | 23,656,960 |
| DRAM read throughput, reported peak fraction | 35.74% | 37.04% | 34.35% |
| L1↔L2 transaction bytes | 24,838,016 | 25,039,616 | 92,297,728 |
| Vector-memory instruction count, loads | 92,160 | 92,160 | 163,840 |
| Vector-memory pipe-busy/issue ratio | 0.000894 | 0.428263 | 0.645225 |
| Vector-memory dependency/issue ratio | 1.084649 | 0.410596 | 1.190483 |

The K-pack implementation has approximately 3.7 times the L1↔L2 traffic
and 1.78 times the vector load instruction count, **not** 3.7 times the
DRAM weight bytes. Occupancy and grid size are similar; shared memory is
much smaller. These observations prioritize repeated A/metadata requests,
load coalescing and instruction scheduling. They do not isolate which
individual load causes the extra traffic; per-instruction/source analysis
or an isomorphic A/B is still needed.

For N=1024, K=5120, the frozen K-pack recipe launches only 64 CTAs and has
13.73% achieved active-warp occupancy. The reference launches 128 CTAs and
has 43.07%. Recipe/CTA work distribution is therefore a separate, concrete
optimization axis for the medium family. No inter-CTA Split-K is required
by the current experiment; the reference's third recipe field is K warps
inside a CTA, not a second reduction kernel.

There is no N=8192/K=5120 ACU report in this archive. Its 18–24% regression
must not be assigned the same causal explanation without measuring it.

## Evidence and limits

- Archive SHA256:
  `c0bbdf93c1cee065c3ba4d82e53425f7406152b86a534bb8cce1866e6c76757e`.
- 1,236 complete records revalidated against raw log hashes, exact device,
  expected recipe sets, sample medians, zero-code/zero-A checks and six
  confirmation rounds. Maximum conditioned dot error: `5.953598e-5`,
  below the unchanged `0.005` bound. All nine report hashes match.
- Runtime: 768.17 seconds including screening, confirmation and ACU.
- ACU reports 72 CUs and 67,108,864 LLC bytes. The runner's properties-ABI
  `sm=1` is not a physical SM count and does not select these SIMT grids.
- The previous `FwqtFC` run had a confirmed concurrent inference request;
  do not use it as the final performance comparison. This retest's ACU
  warnings list only PID 3304848, `python3`, and no `llama-server`.
  A waiting benchmark parent also owns a context, so that warning alone
  proves neither concurrent work nor complete exclusivity.
- [Validated timing/counter summary](measurements/q4_h800_port_ppu_20260911/summary.json)
  and [all numeric/timing records](measurements/q4_h800_port_ppu_20260911/samples.json.gz).
- CPU-only reproduction: `python dev/gemv_ppu/review_h800_port.py --help`.
  It checks the frozen source/package hashes and imports the reports; it
  does not launch GPU work. A private newer-glibc loader is optional.

## Next optimization boundary

Keep canonical offline bytes unchanged. First tune the current K-pack
reader's CTA geometry on PPU, preserving all historical winning recipes.
Then reduce redundant A/metadata traffic with measured load-schedule or
cooperative-exchange A/Bs. Target the raw reference as well as Xplane.
Close the N8192/K5120 counter gap before attributing its slowdown.
No production selector, shipping library or offline-format contract changes
are admitted by this report.

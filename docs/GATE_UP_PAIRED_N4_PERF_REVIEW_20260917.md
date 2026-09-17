# Paired gate/up full-call performance review, 2026-09-17

Verdict: **PASS for the declared 16-point component cohort**. Every point has
a confirmed fused implementation faster than its current per-token incumbent.
This is not a native-llama comparison, model-accuracy admission, or TPOT result.
No production selector, caller, offline cache identity or binary was changed
while reviewing this return.

## Evidence and validation

- Archive: `gate-up-perf.qLUCBs.results.tgz`, SHA256
  `91ee820416b7f2bbae44ded371030cdb995432c4a7664403747ba4ff80001d56`.
- Source: `babc95ca8e1d5c5cc168f423ffebb7013aef1312`.
- Artifact: `e366351c629a9bd314e169f2d91a28b5d45edc3b`,
  `prebuilt/ppu0010/gate-up-perf-v1`.
- Manifest SHA256:
  `27b1c0cffd3410b7a48346402d8b4ec597a06f7380cb28d7b5f4340398657952`.
- Summary SHA256:
  `6a83aca409ee169bac6c7155d4f5f143a4b591b82a6ad412665d10718fc74535`.
- PCI `0000:08:00.0`, one visible device. L2 attribute and explicit setting both
  report 67,108,864 bytes. Environment idleness is **NOT_PROVEN**; no ACU report
  accompanies this return.

All 13 saved Python helper hashes match the frozen source. All 16 point-file
hashes, exact workload/arm inventories, inherited policy choices, screen winner
selection, confirmation samples, medians, deltas and MBU calculations were
rechecked. The unchanged execution and fusion image identities match the pin.
The review independently regenerated all 50 distinct expert fixtures, their
aggregate raw-GGUF hashes, routing-ID hashes and active packed-byte counts.

Coverage: 272 configurations, 544 changed-input endpoint checks, 272 detected
zero-A negatives, graph correctness checks, 816 screening samples and 2,112
fresh confirmation samples. Each finalist has four rounds of 11 samples.
Maximum reported independent GGUF-dot-plus-SwiGLU error is
`1.1084730683720599e-7`; no numerical failure or rounding exception was needed.
Every measured winner beats its incumbent in each of the four round medians.
Total campaign wall time was 92.6114 seconds, including fixture/child overhead
but excluding the wrapper's artifact fetch. No box compilation or JIT occurred.

## Complete-call results

All times are microseconds per **whole token batch**, not microseconds per
token. Input/output storage is F32; shared compute is F16 with unrounded F32
G/U projections, routed compute is BF16 with BF16 G/U projection rounding.
One projection is N512/K2048; paired physical N and routed concatenated N are
1024. The routed profile is E256/top8, one deterministic spread profile.

| Role | Tokens | Incumbent | Fused SIMT finalist | Fused TC finalist | Lowest confirmed configuration | Latency reduction |
|---|---:|---:|---:|---:|---|---:|
| Q8 shared | 1 | 12.8221 | 7.2032 | 10.7897 | SIMT S1 W8 | 43.82% |
| Q8 shared | 2 | 13.1226 | 7.0091 | 10.8526 | SIMT S1 W8 | 46.59% |
| Q8 shared | 3 | 14.4015 | 7.0826 | 10.9924 | SIMT S1 W8 | 50.82% |
| Q8 shared | 4 | 13.4897 | 7.2547 | 11.0991 | SIMT S1 W8 | 46.22% |
| Q8 shared | 5 | 16.5582 | 7.5594 | 11.0441 | SIMT S1 W8 | 54.35% |
| Q8 shared | 6 | 16.8868 | 7.8065 | 11.0850 | SIMT S1 W8 | 53.77% |
| Q8 shared | 7 | 17.8679 | 8.8029 | 11.0379 | SIMT S1 W8 | 50.73% |
| Q8 shared | 8 | 18.0979 | 9.0271 | 11.0497 | SIMT S1 W8 | 50.12% |
| Q4 routed | 1 | 18.8425 | 17.1537 | 20.3913 | SIMT S1 W8 | 8.96% |
| Q4 routed | 2 | 37.0650 | 30.2325 | 31.0100 | SIMT S1 W4 | 18.43% |
| Q4 routed | 3 | 46.7433 | 47.9733 | 42.3400 | TC S2 TM16 | 9.42% |
| Q4 routed | 4 | 61.3480 | 52.6920 | 52.2280 | TC S2 TM8 | 14.87% |
| Q4 routed | 5 | 75.4500 | 64.1200 | 61.7850 | TC S1 TM16 | 18.11% |
| Q4 routed | 6 | 84.4500 | 80.5100 | 71.2150 | TC S1 TM16 | 15.67% |
| Q4 routed | 7 | 97.9600 | 85.3200 | 82.7533 | TC S1 TM8 | 15.52% |
| Q4 routed | 8 | 96.9267 | 93.9667 | 92.0400 | TC S1 TM8 | 5.04% |

The Q8 TC finalist is S8/TM8 for all eight points, but loses to fused S1 SIMT
throughout. The Q4 SIMT finalist uses W4 for tokens2/5/8, W8 otherwise, always
S1. Do not replace these with one generic TC or SIMT recipe.

Q4 tokens4 is nearly tied between backends: TC is only0.88% faster in the
pooled medians, with round differences0.24..1.20%. This is weak evidence for
a dedicated backend switch, despite the clear gain against the incumbent.
Conversely Q4 tokens3 fused SIMT is **2.63% slower** than the incumbent: the
confirmed TC result is needed there. Q4 tokens1 TC is **8.22% slower** than
the incumbent. These losing arms remain in the evidence, not discarded.

## What the gains do and do not mean

The Q8 incumbent executes two projections plus standalone SwiGLU; the winning
S1 candidate computes paired projections and activation without materializing
the two full projection outputs. Its M1 saving is5.6188us per component call.
The Q4 incumbent already has one concatenated projection, so fusion removes
less work. Its M1 saving is1.6888us. The difference is consistent with the
declared call boundaries; this run does not isolate the contribution of each
load, launch, store or instruction. Use ACU for that attribution.

Both arms traverse different physical weight addresses covering at least2.25x
L2 in active weight bytes; complete traversals and first-use exclusion are
checked in the frozen harness. Plane bases are256-byte aligned. Active routed
experts for tokens1..8 are8/16/24/29/34/39/44/49, not all256 and not always8*M.

At the assumed2700GB/s roof, modeled weight MBU is:

- Q8 M1:6.44% ->11.46%; all winning Q8 points9.14..11.77%.
- Q4 M1:18.55% ->20.38%; all winning Q4 points20.38..24.77%.

These are complete-call effective weight bandwidth ratios, not ACU DRAM
counters or a claim that the small40%/large60% goals have been met. Those
targets remain open. A fusion win alone does not prove high bandwidth use.

The Q4 M8 incumbent includes standalone indexed prepare/GEMM/finish plus
minimal SwiGLU. It is not the already-prepared inner GEMM in a shared MoE
chain. Baseline postprocessing is deliberately minimal, not all llama helper
overhead. No routing/top-k, down projection, weighted finish or full-model
effects are measured. The fixture uses dyadic weights; real-weight model
accuracy is a separate gate. Do not multiply these savings by layer count
and report that as measured TPOT.

## Next admission step

The measurements justify scoped caller integration testing, not automatic
global selector replacement:

1. Keep Q8 shared S1/W8 and per-token Q4 candidates as explicit paired-layout
   choices; keep the old canonical path for untested shapes/types/profiles.
2. Integrate the paired packer and consumer together, with a distinct cache
   identity. Reuse existing graph fusion checks and preserve other consumers
   of G/U; do not reinterpret old gate-then-up bytes as paired-N4.
3. Preserve F16 shared and BF16 routed contracts. Test actual shared/routed
   model weights and downstream precision before performance admission.
4. Run the warmed Qwen3.5-35B Q4 model comparison, excluding first-use/JIT;
   inspect Asys for removal of the intended calls and measure actual TPOT.
   Profile the scoped finalists with ACU if further reader tuning is needed.

No repeat of this same 16-point component campaign is needed unless relevant
source, precision, routing profile or candidate selection changes.

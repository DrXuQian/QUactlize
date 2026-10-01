# First-weight-tile PDL experiment

Status, 2026-10-01: **correct on physical H800; not promoted to a default**.
Native PPU1.7 compilation, numerics and performance are still separate,
unrun gates. PPU1.0/1.5 and production selection are unchanged.

The public PDL v1 interface still waits before all operand reads. Only the
internal `prepare_tma<..., Pdl=true, PrefetchB=true>` experiment promises that
low/high weight planes and their descriptors are immutable and already ready
before the predecessor starts. This is one initial raw B tile per CTA, not
prefetching the next layer's complete weights or changing the offline format.

## Implementation and lifetime

1. Initialize stage0's transaction barrier with one outstanding arrival.
2. Expect B bytes **without arriving**, and issue only the low/high TMA copies.
3. Wait for predecessor completion before reading A, grouped offsets, packed
   units or freshly expanded SF metadata.
4. An active CTA adds A bytes and arrives once. The original stage0 wait drains
   A+B. An empty CTA instead arrives without A, drains B and only then exits.
5. Later stage byte counts, parity, GMMA completion, arithmetic and output
   stores are unchanged.

Keeping the arrival outstanding prevents B from completing the phase before
A's bytes are accounted for. The first candidate used an independent B barrier;
the retained candidate reuses stage0 and removes that additional wait/barrier.
Speculative loads for empty experts are still real traffic: numerical safety
does not imply a benefit for very sparse grouped work.

## Evidence and result

Physical device: H800 PCIe 80GB, driver595.71.05; CUDA12.8 SM90a build and pinned
CUTLASS4.3.0. Every measurement used an idle-device precheck and sampled foreign
process monitoring. No clocks, power limits or GPU partitions were changed.

| Check | Result |
|---|---|
| Ordinary library specializations | 216/216 encoded bodies unchanged |
| Full rebuilt library gate | 432 ordinary/PDL bodies, host and ABI PASS |
| Host tests including old/optional binary ABI | 29 PASS, no skips |
| Early-B native CUDA codegen | 24 bodies; B-only descriptors before wait; 5 regression plants rejected |
| Physical correctness, each candidate | 96/96 cases; 36 comparisons per case |
| Final timing, default SF edge | 24/24 numerical checks PASS; SF performance regresses |
| Final timing, ordinary SF helper | 24 forward + 24 reverse cases PASS; no default promotion |

Correctness covers Q8/Q2/Q3/Q4/Q5/Q6, FP16/BF16 compute, FQ/SF, dense,
empty/ragged grouped experts, N tails, stage wrap, changing inputs/routes,
direct launches and graph replay. It requires a CPU dot oracle, guards and
raw equality among ordinary, protocol-only and early-B arms. The original
plan's eight illegal Q3/Q6 K256 requests were preserved as host rejects;
canonical K512 replacements passed, followed by fresh full gates for C1/P2.

Timing is the complete **GEMM -> optional SF expansion -> TMA GEMM** chain:
Q4, N512/K2048, FP16/BF16, dense M1/M128 and grouped rows `[0,1,8,65]`.
Producer config2 and target config5 are fixed controls, not selected best
kernels. SF expansion is included on every chain. Warm and rotating weights
(working set greater than twice queried L2) are separate. Preparation,
capture and three warmup graphs are excluded; each point has 11 samples with
alternating arm order. These results are not llama.cpp TPOT or native PPU MBU.

Final candidate, ordinary SF helper, both forward/reverse runs:

| Cohort | Early-B chain delta vs ordinary | Extra delta vs protocol-only |
|---|---:|---:|
| FQ dense | -0.47% to +0.19% | -0.22% to +0.10% |
| FQ grouped | -3.74% to -0.98% | -0.43% to +0.30% |
| SF dense | -1.61% to +1.10% | -0.92% to +0.80% |
| SF grouped | -2.55% to -0.88% | -0.32% to +0.06% |

Negative delta is faster. Early-B and protocol-only sample ranges overlap in
46/48 points. Most grouped improvement is already present in protocol-only;
there is no established broad incremental prefetch benefit. The earlier
default-edge run also includes a +2.00% FQ dense point: later repeats do not
erase it. Preserve all observations, not just the fastest run.

### SF edge/resource regression

The SF PDL graph regressed even in the **protocol-only** control. The final
default-edge cohort retained SF regressions of roughly +42% to +150%.
For the C1 FP16 dense M1 warm chain (microseconds):

| Diagnostic | Ordinary | Protocol-only | Early-B |
|---|---:|---:|---:|
| Default resource policy, PDL SF | 246.32 | 601.37 | 601.85 |
| Target preferred shared carveout25, PDL SF | 252.05 | 281.00 | 280.61 |
| Only SF helper uses ordinary launch | 250.51 | 249.91 | 250.25 |

All C1 target arms are in the same translation unit; their eight ordinary and
protocol-only encoded bodies equal the baseline's. The resource control is a
host function attribute, not a kernel rewrite. Fresh child processes isolate
the attribute; it is never set by existing callers.

Nsight Systems sees the SF target grow from about209us in direct execution
to about599us in the default PDL graph. Carveout25 changes its reported shared
configuration from233472 to65536 bytes and reduces the graph target to236us;
making SF ordinary restores207us without changing that target resource
configuration. This identifies an edge/resource interaction, **not proof that
the entire excess is a dependency wait or L1 misses**. Further counters or
phase instrumentation are needed to distinguish those mechanisms. Profiled
timings are explanatory only; the table above uses unprofiled event runs.

Do not blanket-enable PDL on helpers or adopt carveout25 as a production preset.
Keep SF ordinary unless a matched complete-chain comparison admits otherwise.

## Reproduction

Use a PDL-enabled baseline built by `quactlize/backends/ppu17/build.py`, the
same compiler and pinned CUTLASS dependency. Build outputs must be outside the
checkout. These commands are for physical Hopper, **not the one-launch native
PPU simulator workflow**. Each output directory must be fresh.

```bash
BASELINE=/workspace/q17-pdl-baseline/BUNDLE_KEY
EXPERIMENT=/workspace/q17-pdl-prefetch
CUTLASS=/workspace/quactlize-ppu17-cutlass430-20260928/dependency

python3 tests/ppu17/pdl_prefetch/build.py \
  --baseline "$BASELINE" --cutlass "$CUTLASS" \
  --compiler /usr/local/cuda-12.8/bin/nvcc --output "$EXPERIMENT/build"

GPU_UUID=$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)
python3 tests/ppu17/pdl_prefetch/run.py \
  --baseline "$BASELINE" --build "$EXPERIMENT/build" \
  --device 0 --expected-uuid "$GPU_UUID" --phase correctness \
  --output "$EXPERIMENT/correctness"

python3 tests/ppu17/pdl_prefetch/run.py \
  --baseline "$BASELINE" --build "$EXPERIMENT/build" \
  --device 0 --expected-uuid "$GPU_UUID" --phase timing \
  --correctness "$EXPERIMENT/correctness/summary.json" \
  --serial-sf --output "$EXPERIMENT/timing"
```

Repeat timing with `--reverse` and a fresh output. Omit `--serial-sf` to retain
the default-edge counterexample; use `--only 6 7 --carveout 25` for the bounded
resource diagnostic. No global environment knob changes shipping routes.

Evidence: `/workspace/quactlize-ppu17-pdl-prefetch-20261001/` contains the plan,
immutable builds P1/C1/P2, ordinary code comparison, full library validation,
hash-checked raw logs, `benchmark.csv`, `summarize.py` and Nsight reports.
Source checkpoints: P1 `53234b1`, C1 `584480c`, P2 `02cecb2` (detached task
history, not additional maintained branches). Binary artifacts are not pushed.
The historical benchmark's `launches_per_graph` field counts logical chains;
`chains_per_graph` is explicit, and actual kernel count is two per FQ chain or
three per SF chain. This naming does not change event intervals or divisors.

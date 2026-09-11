# Q4 SIMT PPU: A reuse and subsequent optimization plan

## Goal and comparison boundary

Keep the canonical K-pack4 bytes and FP16 activations with FP32 dot and
reduction. The immediate goal is no more than 5% regression against the
FP32 Xplane comparator on **each** of six dense M1 shapes, in both warm and
greater-than-L2 rotating regimes. Averages do not close a failing shape.
Xplane remains a development comparator, not a shipping format or fallback.

Shapes `(N,K)`: `(512,2048)`, `(1024,5120)`, `(4096,2048)`, `(4096,4096)`,
`(5120,8192)`, `(8192,5120)`. Start causal work at `(5120,8192)`, retaining
the other families as holdouts. Do not reopen the Cartesian config sweep.

## Evidence behind the ordering

The [PPU receipt](Q4_SIMT_PPU_ACU_20260911.md) gives 22.9280 us for FP32
Xplane and 28.8857 us for K-pack at M1/N5120/K8192, rotating. Both use S1,
320 CTAs and 256 threads. K-pack does not have a separate reducer here.
DRAM read bytes are almost equal, but vector-load instruction count grows
from 92,160 to 286,720; transactions per global-load instruction are eight
in both captures. This is not proof of worse coalescing per instruction.

Source addresses plus native PPU ISA explain those instruction totals:

| Global-load source | Xplane | Current N4 K-pack |
| --- | ---: | ---: |
| A | 10,240 | 163,840 |
| B codes | 40,960 | 81,920 |
| Packed metadata | 40,960 | 40,960 |
| Total | 92,160 | 286,720 |

A accounts for 78.95% of the additional global-load instructions. Xplane
stages A once per CTA; N4 repeatedly fetches half2 pairs from global memory.
Cache hits do not eliminate those vector-load instructions.

## Ordered work

| Order | Experiment / deliverable | Acceptance and stopping rule | State |
| --- | --- | --- | --- |
| 0 | Repair and remeasure the current FQ TC comparison arm | Same shape/input/cache protocol; report complete call including reducer. Historical TC timing is context, not a substitute for the missing new run. | Small dispatcher repaired in `ff86e1b`; PPU remeasurement pending |
| 1 | A-only cooperative shared staging | Keep B bytes, metadata formulas, grid, S1, FP32 dot and reduction order. Independent GGUF oracle plus exact raw FP32 comparison with original K-pack before timing. Check actual codegen as well as latency. | Compile-only add-on ready; device pending |
| 2 | Recover efficient load issue / metadata codegen | Separate variant for back-to-back vector metadata/B loads before decode; remove unnecessary signed index correction where positive ranges are proved. Compare vector widths, requests, dependency stalls and registers. Do not combine with arithmetic changes. | Next; A staging already exposes metadata scalarization |
| 3 | Decouple B load ownership from SIMT compute ownership | Map cooperative loads onto contiguous physical N words, exchange via a small shared tile when useful. Preserve `[K/4,N]` b16 bytes and nibble meaning. Compare net latency including exchange/barriers, not only transaction counts. | Planned |
| 4 | AIU/swizzled-shared/ldmatrix-trans B reader | Reuse the existing b16 transport concept, not an MMA compute kernel. Prove `(lane,register,slot)->(n,k)` against the SIMT consumer; do not expand all weights to FP16 or create a new offline format. Start at one stage, then consider two only if latency warrants it. | Planned, after simpler load controls |
| 5 | Bounded topology/config selection | Challenge each retained winner with at most a few local `Columns/Warps` alternatives. S1 remains a first-class candidate; Split-K must win including its actual reducer. Check all twelve shape/cache rows after each retained change. | After a reader wins |
| 6 | Extend and integrate | Validate M2/M4/M8, grouped/indexed expert access, empty experts and then Q2/Q3/Q5/Q6 readers. Select between admitted SIMT and FQ TC in the production policy; recheck full adapter latency and real-model correctness. | After Q4 dense closure |

Using a faster TC route may improve shipping performance, but does not itself
close the SIMT-versus-Xplane parity goal. Prefetch remains deferred until
intra-kernel request pressure is addressed; measure complete pair latency if
it is revisited.

## Uploaded reference: what transfers

Reviewed local inputs (not redistributed in this repository):

- `gemv q4k优化流程.docx`, SHA256
  `dce5eac2203f46305e0e46bce6d11444431cc2eeaedd0d0c8034dc8a375aed11`.
- `gemv_ref.cu`, SHA256
  `6d575665df5f9fe16d9259178d28dc625867bdcba9967e49ad8985b0af528e65`.

Useful mechanisms: cooperative A staging; issue independent header/code
loads before decoding; consume register pair order without unpack/repack
round trips; independent accumulation chains; unsigned index arithmetic;
balance warps across N and K instead of leaving K workers idle.

The current N4 reader already keeps four independent FP32 chains per output
and consumes packed words directly. Its extra request pressure must not be
mistaken for a missing bit-trick converter. The raw GGUF reference's contiguous
per-lane `qs` addresses and its scale-byte extraction do **not** transfer
unchanged to K-pack.

`gemv_ref.cu::q4k_dot_word` uses an outer `__hfma2` to accumulate products in
FP16 before a later FP32 fold; output is also FP16. Its timings are not a
precision-matched FP32-dot baseline. Its changed scale/zero rounding and the
suggested algebraic extraction of scale/zero are independent numerical
experiments, not part of this A-only comparison. DP4A with quantized A is
outside the retained FP16-A contract.

## First add-on and local checks

`dev/gemv_ppu/astage_source.py` derives the small/large static N4 bodies from
the same source generator as the previous package. It changes only A staging
and the A pointer inside the device body, plus the necessary dynamic shared
allocation in its launcher. Removing those source changes recovers the
original body exactly. The seven distinct shape/recipe pairs are frozen
from the uploaded winning rows; invalid shape/recipe, M>1, non-F16 input,
misalignment or S>1 is rejected rather than falling through to a generic
kernel. The production APIs and original control package are untouched.

Compiler, flags and original source closure match the controls. The added
library is approximately 1 MiB; its local build took 21.3 seconds. This is
compile-only evidence, **not** device numerical or performance admission.

Native ISA at C4/W8/N5120/K8192 exposes an important compiler side effect:

| Static instructions in the selected kernel body | Original | Shared A |
| --- | ---: | ---: |
| `vmem.ld.b32` | 64 (A) | 64 (metadata) |
| `vmem.ld.b32x2` | 32 (B) | 32 (B) |
| `vmem.ld.b32x4[.sign]` | 16 (metadata) | 4 (A staging) |
| `tsm.ld.b32x4` | 8 | 24 |
| FP32 FMA | 528 | 528 |
| FP16x2 FMA, dequantization | 256 | 256 |
| Butterfly shuffle | 24 | 24 |
| CTA barrier | 1 | 2 |

The A loads do move to shared, but metadata's 16-byte loads are split into
scalar loads by this compiler. Thus the optimistic source-only prediction
of a 54% global-load reduction is **not achieved by this binary**. Static
global-load sites decrease from 112 to 100, and branch/mask behavior still
requires ACU. Do not promise a latency gain from that count. This add-on
measures the literal A-only source change; metadata load scheduling is the
next separate control if it offsets the benefit.

Host checks cover exact source seams, A vector ownership and bit transport,
missing/duplicate/shifted owners, compiled C query admission, strict result
parsing, and shell safety. On device the add-on additionally requires exact
FP32 equality with original K-pack, finite independent GGUF error below
0.005, zero-A and zero-code checks, and untouched output/workspace guards.

## Box command

Run from the quactlize checkout. No .so compilation, JIT, or config sweep is
performed by this command. The original Xplane and K-pack controls are
remeasured in alternating order with the candidate. Each row has six rounds
of fifteen graph samples, with first graph launch/upload excluded.

```bash
(
  GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop &&
  git lfs pull --include="prebuilt/ppu0010/q4-simt-ab-v1/*.so,prebuilt/ppu0010/q4-astage-v1/*.so" --exclude="" &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
  bash tools/run_q4_astage_ppu_box.sh
)
```

By default this covers all six shapes in warm and rotating mode, plus three
forced-cold ACU reports for the N5120/K8192 anchor. Set `ANCHOR_ONLY=1` for
only that shape or `ACU=0` to omit profiling. Profiling durations are not
used as warm/rotating timing results. This is 216 fixed-arm timing batches,
not the previous exhaustive search; no box wall-time claim is made before
this runner has been measured.

Optional `FIXTURES=/workspace/q4-simt-ppu.0uCEmO/fixtures` reuses previous
inputs without regeneration. To resume, use the same options and
`RESUME_RUN=/workspace/q4-astage-ppu.<suffix>` (and the same `FIXTURES` if it
was external). Successful hash-bound batches remain valid; only failed or
missing batches rerun. A failing arm does not discard other completed cases.

Upload the printed `q4-astage-ppu.<suffix>.results.tgz`. It includes
`summary.tsv`, full per-round receipts, failures and anchor ACU reports.
`status=PASS` means all requested measurements are valid; only each row's
`WITHIN_5_PERCENT` verdict indicates that row reached the performance goal.

## Historical Tensor Core context, not this cache experiment

The `c0c1361` overnight receipt selected FQ
`TM8/TN64/TK256/WM8/WN16/stages2/AP1/DN64`, Split-K=8, at **21.4800 us**
for M1/N5120/K8192. Its raw log says `scope=FULL_OUTPUT`,
`reducer_untimed=0`, `raw_bad=0`; the reducer is included.
The same historic table reports SF at 28.2000 us.

At the user-supplied 2700 GB/s peak, the 23,592,960 logical Q4+metadata bytes
give effective bandwidth utilization of **40.68%** for the FQ time. At
500 TFLOP/s, `2*M*N*K` gives useful-work MFU of **0.781%**. These are
logical-work/time models, not ACU DRAM/TC-pipeline utilization. Do not compare
that older cache protocol directly with the new warm/rotating SIMT receipt,
or call it the current dispatcher result.

# Two-shape Q4 cold latency experiment

Status: **locally compiled and host/ISA checked; PPU numeric/performance
admission pending**. The [latest results](Q4_READER_FOLLOWUP_RESULTS_20260912.md)
leave dense M=1 Q4_K N512/K2048 and N1024/K5120 above the user's ref+5% line.
The four closed shapes are not rerun. Canonical offline bytes, production
dispatch, shipping libraries and llama.cpp are unchanged.

## Bounded inventory

| N x K | Immutable measured anchor | Meta Width8 | Residue Width4 | Affine | Total |
|---|---|---:|---:|---:|---:|
| 512 x 2048 | meta-h0-n8-w16 | 24 | 8 | 0 | 32 |
| 1024 x 5120 | v4-c4-w20-p4 | 24 | 8 | 16 | 48 |

`dev/gemv_ppu/small_latency.py::plan()` and the prebuilt manifest enumerate
every key, recipe, grid/block, K pass and shared-memory size. This is a
bounded 80-context search, not the full product of all possible configs or
a claim of global optimality. All calls remain one-kernel S1, FP16 input,
FP32 accumulation/output; no inter-CTA split, extra reducer or weight repack.

Factors:

- **H0/H1:** conditional 32-bit packed-field extraction versus
  `lop3` bit-select. The fields, scale/min equations and reconstruction
  precision remain unchanged. These are new experiment labels: the preceding
  small-reader `meta-h0` anchor uses 64-bit extraction and is still measured
  from its immutable binary, not silently replaced by this new H0.
- **L0/L1:** ordinary compiler scheduling versus preloading A/B/metadata
  before their decode/use dependency. Empty inline-assembly dependencies
  constrain compiler scheduling; they are not GPU barriers. Whether they
  change native code is checked in the emitted ISA.
- **A0/A1/A2, Width8:** contiguous vector load plus existing register
  transpose; direct strided FP16 loads; or cooperative CTA shared-A staging
  followed by direct reads. A2 adds a barrier and 2*K shared bytes. This
  reader already avoids duplicate logical A reads within a CTA, so staging
  is not advertised as reducing its logical global-A byte count.
- **Width4 residue reader:** each lane owns two K residues and four output
  columns, reducing per-thread column work and changing CTA count/reduction
  ownership. Only direct A and shared A are tested. Its aligned 64-byte B
  footprint utilization is 12.5%, worse than Width8's 25%; that tradeoff is
  explicit, not a claim that all source changes improve coalescing.
- **Affine, N1024 only:** C4-W20-P4 and C4-W10-P2 retain FP32 group-affine
  arithmetic. L1 loads all eight packed rows and A values before the dot;
  A1 requests 16-byte A vectors and A2 stages A. In one compiled geometry
  A0 and A1 produce the same native load/instruction structure: a wider
  source type is not evidence of a new native load width or speedup.

Meta warps are 8/16 on the smaller shape, 10/20 on the medium shape.
Residue warps are 4/8 and 10/20, respectively. These bound one/two K passes
near current useful geometries. No closed shape, qtype extension, AIU path
or broad Split-K sweep is added.

## Native evidence and correctness contract

For N512/K2048, meta H0/A0/W16 L0 versus L1:

| Static native property | L0 | L1 |
|---|---:|---:|
| Global loads before the first vector-load wait | 1 | 3 |
| CTA synchronization instructions | 1 | 1 |
| Global 16-byte load instructions | 2 | 2 |
| Global 8-byte load instructions | 1 | 1 |

The full counted instruction mix is unchanged for this pair; load/wait
ordering changes. The package stores per-context native instruction/order
receipts, and all 80 contexts contain fast `lop3`/half2 code dequantization
and FP32 FMA. This proves that the intended code-generation change exists,
not that it improves latency. Metadata decoding and FP32 dot work remain;
fast half2 dequantization is not FP16 accumulation.

Before each usable timing the candidate runs against an untouched body at
the same geometry, with exact FP32 output-bit comparison. Existing meta
N512 and affine geometries additionally compare that rebuilt control with
the immutable preceding binary. Meta N1024 and residue geometries are new
instantiations of unchanged bodies: they are not claimed to have an already
measured immutable binary. All also require independent original-GGUF dot,
zero-code/zero-A, guards and replay checks. Source-level coordinate/decoder
proofs alone do not admit a PPU numeric result.

All candidates record actual A/B/unit plane alignment and per-warp addresses,
vector widths, unique/duplicate bytes and 32/64/128-byte footprint models.
Shared-A offsets are relative to the shared allocation, not the global-A
base. The package is five isolated experiment DSOs, about 1.6 MB including
receipts, built locally in 11.3 seconds. No production DSO was rebuilt.

## Box work and entry point

1. Five-sample screen: 80 candidates plus six contemporaneous anchor cells.
2. Confirm top three candidates and old winner/ref/Xplane per shape:
   six alternating rounds x15 samples = 72 cells.
3. Profile the two best confirmed candidates and three anchors per shape:
   ten ACU reports. Profiler durations are not event medians.

Total **158 timing cells +10 profiles** in 66 child invocations if none
fails. The preceding 132-child experiment took 462.6 seconds; allow roughly
**4–8 minutes plus fixture generation** on an idle device. This estimate
includes orchestration/profiling, not just kernel time, and is not a deadline.
Progress prints shape/phase/cell count and elapsed seconds.

The weight ring is at least 2.25 times verified L2 and full rings are timed.
First-use, graph upload and setup are excluded. No JIT/compilation runs on
box. Do not run inference concurrently on the selected device.

After pulling the code and LFS payloads, execute with `bash`, never source:

```bash
CUDA_VISIBLE_DEVICES=0 \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
L2_BYTES=67108864 \
bash tools/run_q4_small_latency_ppu_box.sh
```

`FIXTURES` can reuse the previous experiment's fixture directory. `ACU=0`
explicitly omits profiling. `RESUME_RUN` selects the exact printed run
directory: valid logs/reports are hash-checked and reused, failed/missing
cells rerun in fresh processes. Changed source/device/runtime/fixture
identity rejects resume. A failed candidate does not discard other valid
cells. The caller's Docker shell stays open on failure.

Return the printed `results=...results.tgz`. It includes summary TSV/JSON,
raw logs, plan, authority, native ISA receipts and ACU reports. Measurement
`status=PASS` means complete valid data; performance remains separately
`WITHIN_5_PERCENT` or `PARITY_OPEN` relative to this cohort's raw reference.

Package: `prebuilt/ppu0010/q4-small-latency-v1`; transitive controls remain
the frozen reader-followup, reader-reuse, config-sweep, cold-shapes,
h800-port and simt-ab packages. Do not edit their source under a bound run.

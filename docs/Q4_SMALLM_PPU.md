# Q4 dense M2..8: SIMT versus Tensor Core

This is a new device-admission experiment, not an extension of the M1 pass
by assumption. It preserves canonical Q4 K-pack4 and FP16 activations.
No production policy, shipping library or llama.cpp route is changed.

## Exact scope

- M = **2, 3, 4, 5, 6, 7, 8**, all six N/K shapes: **42 cases**.
- N/K: 512/2048, 1024/5120, 4096/2048, 4096/4096, 5120/8192, 8192/5120.
- Cold/rotating **weights** only: at least 2.25x verified L2, complete ring
  traversals. A is resident; different rows of one call naturally share B.
- Three arms: selected-family K-pack SIMT, supplied raw-GGUF FP32 reference,
  and our actual fully-quantized Tensor Core modules.
- FP32 accumulation throughout. SIMT/reference output FP32; existing TC
  outputs FP16. Reference reconstructs weights in FP16; affine SIMT uses
  its recorded FP32 group-affine expression. This is an implementation
  comparison, not identical cross-arm arithmetic or a model benchmark.

## What is scanned

SIMT retains every [actual M1 winner](Q4_MEDIUM_REFINE_RESULTS_20260912.md)
and its exact source/package identity. There are 52 reader/geometry contexts
across six shapes, tested at every requested M. The small shape has four
issue/header/warp choices; the medium shape has eight proven fold/warp
choices; larger shapes retain H32 plus unit/A reuse variants, P4/P8 and a
bounded fewer-K-worker neighborhood. `plan.json` records all keys.

The multirow lift changes only A/output row bases and the launch's row
axis. It runs **one kernel launch with one row per CTA**, not M host launches
and not a new explicit cross-row B-reuse algorithm. Each candidate is
checked against the original body invoked per row outside timing; the M1
anchor additionally matches its actual immutable prebuilt image. No new
SIMT Split-K reducer is introduced.

The raw reference keeps its existing multirow body and screens its actual
M1 winner plus a small intra-CTA N/K-warp neighborhood. The third reference
parameter counts K warps inside the CTA, not a separate reduction kernel.

Tensor Core has two separately reported results:

1. **Current policy**: the exact result of the production C++ policy query
   for each M/N/K, including its policy kind and Split-K. Some requests use
   general fallback policy; this is not called an exact measured optimum.
2. **Best scanned TC**: the union of the five selected FQ parents, each
   challenged with S=1/2/4/8. These are ordinary dense parents with FP16 A;
   no activation-quantized alternative is substituted. This bounded union
   is not a new exhaustive TC tile/provider search or a global optimum.

Whole-K-tile alignment and pipeline-fill exclusions come directly from
`quactlize/runtime/module.cuh`. They are recorded as `STRUCTURAL`, without
fake timing. Unexpected query, initialization or numerical failures remain
failures. The current policy is always retained for six-round confirmation
even if it is not among the two fastest screen choices.

TC timing calls the existing prepared `quactlize_kpack_run_v1`: **producer
and actual Split-K reducer are both included**. Pointer setup, allocations,
query/prepare and first-use graph upload are outside resident event timing.
There is no JIT or compilation on box.

## Protocol and output

Screen uses five event samples. The best two configurations per arm and
M/N/K are confirmed in six alternating rounds of fifteen samples; TC also
retains current policy if different. All arms use the same original GGUF
bytes and the same eight distinct FP16-exact activation rows. The receipt
binds both source fixtures and the actual M-row activation digest.

Independent GGUF FP64-dot, nonfinite checks, device zero-code/zero-A negatives,
host-oracle repeated-row-zero negative, unused-row/output guards, TC workspace guards,
same-body FP32 row equivalence and post-replay determinism precede accepted
timing. Incorrect candidates never enter selection. A failed child ends its
device context; remaining candidates restart separately. Valid hashed cells
survive and resume retries missing/failed cells only.

Default ACU captures the best SIMT/reference/TC at M2 and M8 for each shape,
plus current TC when different: 36 to 48 reports. TC reports can contain
producer and reducer kernels. ACU uses forced-cold per-kernel replay; do not
substitute its durations for complete-call rotating event measurements.

`results/summary.tsv` contains M/N/K, selected SIMT/ref/TC keys, current-policy
TC time, best-scanned TC time, SIMT gaps to both TC values and raw reference,
and which implementation is faster. A complete scan is not automatically a
performance pass. Reference +5% remains the current SIMT tuning criterion.

The payload is twelve small DSOs totaling about **3.65 MB**: six lifted
readers, one reference, five TC modules. Fresh local compile took 111.8 s;
the final same-body rebuild with TC cache reuse took 29.1 s. These are build times,
not a proof of box campaign duration. Execution is larger than the previous
one-shape refinement; progress prints the current shape/arm/phase and wall
time. No hour/day-scale Cartesian campaign is launched.

Local validation: 74 host tests passed across the new scan, frozen M1
refinement and preceding small-M contracts; native compilation checks all
52 SIMT specializations and five TC modules. No local PPU device result is
claimed. The host end-to-end runner test includes a planted child failure,
preserved successful cells, changed-shortlist resume and all 42 summaries.

## Box command

From the quactlize checkout, with one otherwise idle PPU:

```bash
(
  GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin develop &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
  bash tools/run_q4_smallm_ppu_box.sh
)
```

The script fetches only the required LFS payloads. `FETCH_PAYLOADS=0` skips
that fetch after files are installed. `FIXTURES=/workspace/earlier-run/fixtures`
reuses a complete six-shape fixture set. `ACU=0` skips profiling; do not change
that setting on resume. For retry, use the same command with
`RESUME_RUN=/workspace/q4-smallm-ppu.<suffix>` and the same external fixture
path if one was used. It validates source/runtime/device/fixture identity.

Upload the printed `q4-smallm-ppu.<suffix>.results.tgz`. The caller's Docker
shell remains open on errors. M1 timings are not rerun or relabelled as this
new M2..8 result. Indexed/MoE remains a separate next experiment.

## Relation to Tensor Core tuning

This SIMT reader work does not replace TC mainloop tuning: AIU/shared-memory
delivery, MMA fragments and the staged pipeline already organize A/B reuse
differently. The transferable checks are native address/dequant codegen,
actual transactions, resource/parallelism balance and complete reducer cost.
TileM/N/K, MMA warp geometry, stages and Split-K must be evaluated through
the TC implementation rather than copying the SIMT C/W/P switches.

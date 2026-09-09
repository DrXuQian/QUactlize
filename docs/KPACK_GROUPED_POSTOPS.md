# Grouped Split-K post-operations

Status: implemented on develop; nine exact PPU parents compile locally.
PPU correctness/performance admission and production bundle selection remain
pending. This is not a new offline format, routing rule or large sweep.

## Two changes

1. The runtime's S>1 FP32 pointer-array epilogue now calls the same
   `store_splitk_accumulators_direct` primitive as dense. The descriptor index
   is `expert + slice*E`; its pointer already contains the compact expert-row
   prefix and split-plane offset. The local view uses L=1, l=0, plane=0 so none
   of those offsets is applied twice. There is no scale/bias/fusion parameter
   on this internal partial-publication edge.
2. `PpuMixedInputSplitKParallelCompactReduction<2>` reuses the dense fixed-S
   vector body for contiguous `[S][M][N]`, viewed as `[S][M*N]`. It accepts the
   actual vector alignment, including the grouped workspace's 16-byte offset.
   The original dense dispatcher keeps its 128-byte alignment requirement.
   Output tails, padded strides, weak output alignment and HostAdapter calls
   retain the checked generic fallback.

For M=8/N=512, the reducer changes from four 128-thread CTAs to 64 32-thread
CTAs. Every output still accumulates S=0,1,... in FP32 and converts to FP16 once.
No atomics, CPU router readback, allocation or synchronization is added to run().

The S1 epilogue remains its exact original type. The mainloop, A/B provider,
metadata publication, K partition schedule, producer grid and workspace layout
are unchanged. The partial epilogue retains its existing shared-storage
envelope to isolate the code-path change: it removes shared R2S/S2R and their
two barriers per epilogue subtile, but does not shrink the resource query.
Persistent next-task synchronization is unchanged.

## Evidence boundaries

- `tests/kpack_grouped_postops_types.cu` uses real hgcc and runtime types to
  prove S1 identity, mainloop identity, resource envelope, pointer strides and
  exact C-coordinate type equivalence to the CUDA test projection.
- `dev/gemv_cuda/direct.cu` executes the production direct-store epilogue on
  real CUDA with that projection. It does not execute or emulate a PPU MMA.
  It checks empty experts, later M tiles, N tails, all split planes, poison,
  guards and a rotated-slice negative: 216 positive/negative cells.
- The CUDA reducer experiment compares the unchanged generic body, vector
  bodies and actual compact dispatcher, including +16/+128-byte workspace
  offsets, tails, padded-stride admission, weak-alignment fallback and
  cancellation-sensitive fixed-order raw-FP16 outputs.
- No NVIDIA timing establishes a PPU speedup. The PPU gate checks every
  output and partial against independent GGUF arithmetic, then checks the
  reducer against the downloaded FP32 planes in fixed order.

The final RTX 5090 run passed 256 SIMT GEMV configurations and the 216
direct-store cells. At M=8/N=512/S4 the actual compact reducer measured
0.874 µs versus 1.681 µs for the generic body. Full scope, reproduction steps
and evidence hashes are in [the CUDA experiment](../dev/gemv_cuda/README.md).
PPU numerical/performance admission is still pending.

## Box: no compilation

```bash
(
  git pull --ff-only &&
  git lfs pull --include='prebuilt/ppu0010/kpack-grouped-postops-v1/**' &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
    bash tools/run_kpack_grouped_postops_box.sh
)
```

The 6.4 MiB package contains nine new and nine immutable same-parent controls.
It covers Q2/Q3/Q4/Q5/Q6 FQ, TM8 Q4/Q5 compact and persistent, S1 controls and
valid S2/S4/S8. ScaleFirst is excluded. Default four alternating-order rounds
yield 384 measured cells, with 11 samples of 16 complete calls each. The full
GPU metadata + directory + producer + reducer path is timed; input transfer
and correctness checks are outside timing. Same-parent output bits and
shared/workspace/producer-CTA resource envelopes must agree.

Each job runs in a fresh process. One failed job does not discard another's
results. After all numerical checks pass, Q4 compact S4 baseline and candidate
get separate ACU reports. Those replay times are not mixed into warm timing.
Return the printed `kpack-grouped-postops.*.results.tgz`.

ACU is resolved before measurements: `SDK/asight/bin/acu` first, then
`SDK/bin/acu`. An explicit `--acu /absolute/path/to/acu` overrides discovery
and must be executable. With `--skip-acu`, no profiler installation is needed.

To retry only missing/failed jobs with unchanged source, device, SDK, payloads
and timing options, invoke `run_kpack_grouped_postops.py --resume` with the same
`--output` results directory and SDK. Successful complete jobs are reused;
failed JSON/logs are preserved under timestamped names. Changed authority is
rejected. Any ACU captures are fresh.

## First PPU result and fixture ordering

Reviewed `kpack-grouped-postops.NNT8ZP.results.tgz` (SHA256
`26435d94ebfe991dd49372763a078225e90a9fb75feebfc4e4c989693c478171`).
Its manifest and all measurement-source hashes match the delivered 1b4323c
revision. Seven complete jobs / 288 cells pass; two jobs are incomplete:

| Job | Completed cells | First failing call | Failure |
| --- | ---: | --- | --- |
| Q4 TM8 ordinary | 15/64 | ragged S2, round 4, baseline | four directory-header words equal `0xa5a5a5a5` |
| Q6 TM16 ordinary | 26/32 | ragged S8, round 2, candidate | same poison header |

These are not reported dot-product mismatches. The directory header builder
writes all four words before its error/empty-expert return. The test fills
that header with poison on the default stream, but computes on a nonblocking
stream without an explicit edge from that fill. CUDA documents both the
[device-memset host-asynchronous behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
and [nonblocking-stream exclusion from legacy synchronization](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html).
That is a concrete test-ordering gap, not yet proof of the PPU failure's cause.

The real CUDA `dev/gemv_cuda/stream_poison.cu` experiment reproduces exactly
four poison header words under a deliberately delayed default stream, with
both eager and graph consumers. Same-stream poison and an explicit
default-stream drain both turn green (six cases total). This proves the
ordering mechanism on CUDA, not that the PPU kernels are already admitted.

The fixture now enqueues all poison fills on its consumer stream. Pageable
H2D setup is completed before use on that nonblocking stream. Neither fill nor
host setup waits enter the timed graph, module run(), or llama.cpp production
path. Kernels and every packaged DSO remain unchanged. Failure logs now name
the arm, profile, repeat and eager/graph path.

The complete-job timing evidence is encouraging: ragged S2 improves 8.36–10.07%,
S4 16.06–19.99%, S8 23.81–30.65%; ragged S1 stays within 0.22%. On the model
shapes, Q5 compact S2 improves 22.58→20.80 µs, but S1 remains faster at
14.34 µs. Q4 persistent S2/4/8 improve, yet S1 remains its winner. Q4 compact's
model case was not reached. This is combined direct-store + reducer evidence,
not isolated reducer timing or a new default-selection verdict.

Keep that archive unchanged. To diagnose only the two incomplete jobs under
the corrected driver, without rebuilding or rerunning the seven others:

```bash
(
  git pull --ff-only &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
    bash tools/run_kpack_grouped_postops_box.sh \
      --only-jobs fq-q14-tm16-ordinary fq-q12-tm8-ordinary
)
```

This writes a fresh result directory: 96 cells plus two Q4 ACU captures if
numerical checks pass. Completion says `jobs=2/9 scope=SELECTED_JOBS`, not a
fresh all-nine run. Earlier successful receipts are retained as earlier
evidence, not rewritten to claim they ran the new source. Cross-source
`--resume` remains rejected. A fresh full nine-job run remains available by
omitting `--only-jobs`.

## Local rebuild

```bash
python3 tools/build_kpack_grouped_postops.py \
  --sdk /root/ppu-sdk/2.1.1 --jobs 8 \
  --cache /root/autodl-tmp/kpack-grouped-postops-build-v1 \
  --output /root/autodl-tmp/kpack-grouped-postops-new
```

This compiles nine parents, not the full configuration Cartesian product.
The old compact bundle and deployed native/model bundle are not overwritten.

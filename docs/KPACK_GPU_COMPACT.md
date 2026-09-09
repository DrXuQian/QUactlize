# GPU compact grouped execution

Status: implementation and local admission complete; PPU device admission is
pending. Sixteen new modules compile with SDK 2.1.1 and 109 local tests pass.
Production policy and the deployed native bundle are unchanged pending device
results. This package is not yet a model-performance claim.

The `kpack-decode.XZM60u` run passed 260/260 cells in 178.8 seconds. The same
Q4 TM16/TN64/TK256 parent measured 19.410 us on device-only S1 versus 16.035 us
on host-compact S1 and 14.2425 us on host-compact S2. Q5 measured 26.7925 us,
11.240 us and 19.365 us respectively. Host preparation was excluded; these
numbers do not promise deployable latency or isolate the metadata kernel alone.

## Implementation contract

- Keep grouped v2 ABI, device offsets, concatenated activations and canonical
  weight bytes. No per-run allocation, D2H, host wait or CPU routing.
- Reuse the existing device M-tile directory. Ordinary GPU execution launches
  a bounded one-task-per-CTA grid; persistent execution walks the same tasks.
- Extend the shared task identity with a K slice. Reads still select the real
  expert; FP32 outputs select `expert + slice * experts`. Reduction stays
  ordered and runs after every producer on the same stream.
- Preserve the existing host-prepared ordinary path and large-expert fallback.
  Persistent and compact directory builders currently support at most 1024
  experts. This is an explicit implementation bound, not a new offline format.

## Counts and performance guard

Before: ordinary device-only runs one metadata kernel, a rectangular producer
grid and, for S>1, one reducer. Persistent S1 additionally builds a directory.
The pair SIMT reader is unchanged by this work.

After: compact execution adds the existing directory build, removes expert-
padded producer work, and keeps the metadata and reducer launches. The data
mainloop, copy/converter types, intra-task barriers and arithmetic order are
unchanged for S1. At S>1 each task walks K tiles `s,s+S,...`; total logical MMA
work is unchanged, but prologues/epilogues increase and one reducer is required.
Persistent work transitions keep the existing shared-lifetime CTA barrier.

Tests must cover missing/duplicate/slice-rotated task negatives, safe row-count
bounds, changed device offsets in graph replay, every FP32 partial, reducer
identity and output/workspace guards. Compare both new schedules against the
previous immutable module in the same run. No timing is admitted from a wrong
result. GPU compact is not assumed faster until its directory cost is included.

## Run the prebuilt experiment

The package has 16 new modules, four immutable ordinary baseline modules and
the unchanged five-format SIMT library. Eleven isolated jobs check 204 cells:
five-format FQ controls; Q4/Q5 TM16 and TM8 model-shaped comparisons; Q4/Q5
resident-SF controls; and the previous best scalar/pair SIMT controls. Each
cell checks output/partial guards and every FP32 K slice before timing. Device
offset profiles change during graph replay, including the actual M-tile count.
Unused directory capacity must remain poisoned. SF controls exclude preparation
and are not billed as first-use SF timing.

Run on one idle PPU from the development checkout; no box compilation:

```bash
git pull --ff-only
git lfs pull --include='prebuilt/ppu0010/kpack-gpu-compact-v1/**' --exclude=''
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_kpack_gpu_compact_box.sh
```

Return the printed `/workspace/kpack-compact.*.results.tgz`. A failed job does
not discard other jobs. On the same source/SDK/device and parameters, set
`RESUME_RUN` to that run directory to reuse validated jobs and retry failures.
The expected final marker is `GPU_COMPACT_DONE status=PASS cells=204/204`.
Numerical admission is separate from reviewing the timing deltas.

## Remaining GEMV question

The previous pair candidate improved Q4 25.0325 to 18.600 us and Q5 best
37.350 to 22.755 us. These are not DRAM saturation measurements. C16/W8 static
inspection reports zero stack bytes, but still 1950/2339 instructions for
Q4/Q5, with substantial integer address/bit operations and FP16/FP32 conversion.
Static counts include branches and are not dynamic instruction counts. Repeated
warm weights, metadata decoding and activation broadcasts need source/ISA and
device-counter separation before assigning a bandwidth root cause.

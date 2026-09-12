# Q4 cold GEMV config sweep

This experiment tunes the existing canonical K-pack4 reader on PPU. It does
not change offline bytes, accumulation precision or production dispatch.
The selected reader is not yet a llama.cpp plugin update.

## Scope and acceptance

Q4_K, dense M=1, six existing shapes, rotating weights at least 2.25 times the
verified L2 capacity. No warm result is used for acceptance. Xplane and the
supplied raw-GGUF reference both accumulate/output FP32; their individual
weights are reconstructed in FP16. The affine K-pack candidates use FP16 A,
exact integer codes, FP32 group-affine arithmetic, FP32 accumulation/output.
These are different legal rounding orders, checked independently against GGUF.

The latest user bottom line is **K-pack median <= raw-reference median**:
zero allowed regression against reference. Xplane within 5% is a secondary
comparison. All six confirmation rounds and their raw samples remain in the
result; a marginal median difference is not a statistical proof of a win.
Incomplete or incorrect candidates cannot claim performance acceptance.

| N | K | Compiled configs | Pinned previous winner |
|---:|---:|---:|---|
| 512 | 2048 | 52 | meta-static, C1/W16 |
| 1024 | 5120 | 71 | affine8, C2/W10 |
| 4096 | 2048 | 52 | affine4, C4/W8/P4 |
| 4096 | 4096 | 68 | affine4, C4/W8/P4 |
| 5120 | 8192 | 79 | affine4, C4/W8/P4 |
| 8192 | 5120 | 71 | affine4, C8/W8/P4 |

The previous winners are bound to
[`selection.json`](measurements/q4_cold_shapes_20260912/selection.json).
Only their identity is reused: all timings are measured again. C8 is not
substituted globally. In that uploaded cohort, best K-pack versus reference
was +20.19%, +30.45%, +8.56%, +11.08%, +10.70%, +2.41%; **0/6** meets the new
zero-regression requirement. The sixth row met the older 5% requirement.

## Config space and address pattern

The kernel remains `q4_group_affine<C,W,P,N,K,true>`:

- C: lanes collaborating on adjacent N columns, `{1,2,4,8,16}`.
- W: warps per CTA, `{1,2,4,5,8,10,16,20,32}`.
- P: output columns per lane, `{2,4,8}`; **not Split-K**.
- TileN = C*P, threads = 32*W, K workers = 32*W/C, grid = N/TileN.

`plan.json` contains every included/excluded key and reason. TileN >32 is
outside this reduce-scatter implementation. Idle K workers and >16 serial
passes are explicit search-budget pruning, **not proofs those configurations
cannot win**. The result is the best confirmed config in a declared bounded
space, not a claim of global optimality. All candidates are S1, one kernel,
with an internal warp/shared reduction and no external reducer.

For `t = warp*32+lane`, `g = pass*(32*W/C)+t/C`,
`n0 = block*C*P+(t%C)*P`, byte addresses relative to each plane are:

| Input | Representative lane address | Source width |
|---|---|---:|
| B, one of eight packed-word slots | `2*((8*g+r)*N+n0)`, r=0..7 | 2*P B |
| Metadata for output p | `16*((g/8)*N+n0+p)`, p=0..P-1 | 16 B |
| A, first half2 of a 4-value group | `2*(32*g+8*slot+4*half)` | 4 B |
| A, second half2 | previous address +4 | 4 B |

`access-patterns.json` lists first warp, next CTA, last active warp, duplicate
lane bytes and 32/64/128-byte footprints for **each config**. Timed rows also
record first-warp models using the actually observed ring-pointer alignments.
For P4, C4 covers 32 contiguous B bytes per K group, C8 covers 64: under an
aligned 64-byte sector model B utilization rises from 50% to 100%. This does
not account for cache reuse, A/metadata traffic or reduced CTA concurrency.

Logical source-load volumes per call are B=NK/2, metadata=NK/2 and A=2NK/P
bytes. Distinct inputs are NK/2, NK/16, 2K bytes respectively. Duplicate
metadata requests may broadcast/cache; A is reread across N tiles. These
are **not** measured DRAM bytes or dynamic instruction counts.

## Fast dequant, not FP16 dot

The generated unsigned code decoder constructs two half values using
`lop3.b32` and half2 multiply/add. All four nibble slots recover exact 0..15
integers. Native ISA receipts confirm `v.lop3.b32`, `v.add/fma.f16x2` and FP32
FMA in all 393 compiled specializations. `isa-stats.json` retains static
counts and actual native load widths. The compiler can merge adjacent A
loads: source-level widths must not be confused with native instructions.

Scale/min decoding is separate: packed-unit metadata still uses bit extraction
(including 64-bit shifts), conversions and FP32 affine work. The inner dot
accumulates FP32, never half2. Calling the code decoder “fast dequant” does
not establish that total dequant/address overhead is optimal.

## Run on PPU

Six small prebuilt libraries contain the 393 configs; no box compile or JIT.
The older comparison libraries are immutable experiment controls, not new
production dependencies. Keep the selected PPU idle during measurement.

```bash
(
  cd /sim/eec/shared/junfu.qx/quactlize &&
  git pull --ff-only &&
  git lfs pull --include='prebuilt/ppu0010/q4-config-sweep-v1/*.so,prebuilt/ppu0010/q4-cold-shapes-v1/*.so,prebuilt/ppu0010/q4-h800-port-v1/*.so,prebuilt/ppu0010/q4-simt-ab-v1/*.so' &&
  PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  CUDA_VISIBLE_DEVICES=0 L2_BYTES=67108864 \
  bash tools/run_q4_config_sweep_ppu_box.sh
)
```

Each shape screens all declared configs in **one process sharing its fixture
and device allocation**, five graph-event samples each. Confirm the best
three plus the previous winner and both controls: six alternating rounds,
15 samples each. Setup, first launch and graph upload are excluded from
timing. ACU then profiles the controls, baseline and best new config with
forced-cold replay; ACU duration does not replace event timing.

The previous fixed comparison took 457.6 seconds for 144 timing cells and
24 profiles (fixture creation excluded). This sweep retains the same 24
profiles, adds a batched 393-config screen and confirms 216 timing cells in
144 child processes. Allow roughly **15–25 minutes plus fixture creation**
initially; this is a planning estimate, not a measured duration guarantee.
Progress is printed per phase and every eight configs within each batch.

Optional `FIXTURES=/exact/old/run/fixtures` reuses verified input files.
`RESUME_RUN=/exact/run` validates source/device/runtime identity and reruns
only missing/failed cells; correct completed cells survive a later failure.
`ACU=0` skips profiling, but does not provide counter evidence for diagnosis.
`TOP_K=5` enlarges confirmation at additional runtime cost.

Return the printed `results=...results.tgz`. It includes summary.tsv,
config-winners.json, per-config samples/address patterns, ISA/build receipts,
raw child logs and ACU reports. Success means complete numeric/timing
measurement; the separate reference verdict decides performance acceptance.

Local build, if the config body is subsequently changed:

```bash
python dev/gemv_ppu/build_config_sweep.py --sdk /root/ppu-sdk/2.1.1 \
  --output /root/autodl-tmp/q4-config-sweep-new --jobs 6
python -m pytest -q tests/test_q4_config_sweep_ppu.py
```

The initial six-shape native build completed locally in 61.8 seconds; PPU
correctness/performance for the expanded config space is still pending.

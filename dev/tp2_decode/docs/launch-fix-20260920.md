# Decode entry and chain admission repair

Parent Quactlize `69af891`, caller `1f7da3b`. Preserve the published runtime
manifest `5bda5fabf565c54494a358479a3ce9923bcde8d204338aede0742517491a4aab`
as the baseline. Work branches: `dev/dispatch-launch-fix` and
`dev/quactlize-chain-admission`. Evidence: `asys-result-20260920.md`.

## Contract and order

1. Route public F16 v1 and typed F16 v2 through one generated device resolver.
   Preserve both validation contracts, BF16-only v2 storage, strides, ordered
   reducers and the measured exact/bucket catalog. No new precision or layout.
2. Reproduce the missing multi-layer MoE admission locally. Preserve graph
   consumer, device, concurrent-stream and memory lifetime constraints. A new
   admission rule needs a rejected-old/accepted-new witness and negatives.
3. Compile only changed execution/host modules and caller translation units;
   reuse unchanged TC, packer and prefill images. Publish no llama binaries.
4. The box validates actual v1 and v2 calls at M1..8, then warmed TP2 model
   numerics/ABBA and private Asys. Capture exact producer/reducer identity,
   not only a plan string or a geometry tuple. Exclude first complete requests.

## Low-bandwidth follow-up

Priority: dense F32/F16 Q8 K/V N256 K3072 and shared down N3072 K1024;
indexed F32/BF16 Q5 down N3072 K512, E256/top8/channels8. Retain their current
measured minima. Verify corrected-body latency before any selector update.
Further explicit configurations must stay in a small declared inventory with
complete-call timing and independent correctness, not a full Cartesian sweep.
Q8 small K/V has 16 CTAs at its current S1 tile; that is a hypothesis about
parallelism, not proof that adding CTAs or Split-K will be faster. Q5 down
already has many CTAs; do not apply the same diagnosis blindly.

No GPU is available locally for PPU admission. Local host/compile evidence
does not close numerical, latency, ACU or model accuracy checks. Next handoff
must name the pinned source/caller/artifact and preserve passed independent
cells. There is no promised TPOT delta before device measurement.

## Implemented and locally checked

The public v1 entry now constructs an F16 v2 call and uses exactly the same
generated device resolver. Public validation is retained. A compiled host
boundary test invokes both actual public entries with every measured recipe,
all six formats and generic token/split scopes, plus invalid pointers/storage.
All eleven promoted native functions now have byte-identical instructions and
identical resources to the sealed measured component images, including Q5 down.
The 1852-row selection audit and all 937 public SIMT queries pass unchanged.

An allocated three-layer GGML graph reproduces final output overlapping router
weights in earlier layers and IDs in the final layer. The original admission
rejects all three; the new staged admission and unchanged caller memory check
accept all three. The new optional execution entry snapshots 32--256 bytes of
finish weights inside the existing all-SIMT prepare kernel, into the unused
private `down.outputs` descriptor array. Finish reads that copy, after all ID
consumers have completed. No extra launch, copy, allocation or host wait occurs.
The rule excludes TC/mixed chains, separate gate/up and private/live-buffer
overlap. Old execution libraries lack the optional entry and still decline.
This establishes a concrete local cause, not a claim that every box layer had
the same rejection; the next Asys verifies actual coverage.

The device gate tests F16 v1 and v2 explicitly: 17 points / 28 entry checks,
tokens 1--8, changing inputs and invalid IDs. An additional 128-case snapshot
gate covers F16/BF16, tokens 1--8, router on/off, output/weight/ID overlap and
unaligned-within-F32 output aliases. Its CPU fake checks orchestration only;
the production GPU helper and the model still require PPU numerical admission.
Existing BF16 projection/SwiGLU rounding is retained, not silently changed to
match the formerly unfused graph. Full model accuracy remains pending review.

The focus experiment keeps the published v2 measured body as its immutable
baseline (not the accidentally generic v1). It compares Q8 K/V N256 K3072,
Q8 shared down N3072 K1024 and indexed BF16 Q5 down N3072 K512. It tests 22
configurations total, including three exact clones. No TC compile and no
Cartesian product. Numeric controls span tokens 1--8; timing is rotating M1
complete-call only. Each candidate reports source-load 32/64/128-byte footprints
and native ISA/resources. S2 includes its ordered reducer. No winner is promoted
until PPU timing/ACU and numeric results are reviewed.

Devices: physical 6 and 7 for TP2, logical ranks 0 and 1 inside that process.
Entry/focus tests use physical 6 only, sequentially before the model run.
Asys uses private mount/PID namespaces, without touching shared profiler
services. First requests/JIT are excluded. Candidate failures remain independent
of production model results and support point-level resume.

Local verification: 406 related host tests pass, including both actual public
launch boundaries, unchanged caller graph checks and all 128 CPU-backed gate
orchestration cases. All 22 experimental kernels retain fast mantissa code
decode, FP32 FMA and vector-load instructions in the native inspection. Narrow
Q8 C4/P2 reduces each K worker group's contiguous B span from 32 to 16 bytes:
it increases parallelism but may waste a 32-byte transaction sector. Its source
footprint is reported explicitly; more CTAs is not assumed to be a win.

## Handoff

Implementation source: `cb3a8ba24356dc8c1b8546220dcc6c77dc3b057a`.
Artifact: `artifacts/kpack-launch-fix-v1`,
`46727e96de92c6a7da15aa12f066a31f1bed8673`. The model package remains at
`prebuilt/ppu0010/kpack-model-runtime-v1`; experiments use the separate
`prebuilt/ppu0010/tp2-decode-focus-v1` closure. Both are narrowly fetched by
their manifest pins. All 68 ELF paths use LFS (63 model, five experiment),
plus three ISA reports. No caller binary is included. Caller remains
`dev/quactlize-selected-decode`, `1f7da3bd94b1fc6b9520e156456c89221fddd07f`.

Run from source branch `dev/dispatch-launch-fix`:

```bash
CUDA_VISIBLE_DEVICES=6,7 JOBS=192 \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
python3 -u tools/run_selected_decode_box.py \
  --previous-run /sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9/kpack-tp2.8VYueY
```

The previous run supplies exact SDK, caller build, NCP build and compatibility
overlay paths. Overrides remain supported. No model weights/cache are removed.
The runner reports its new run directory and results archive. Full Asys reports
stay at `results/trace/qwen35-122b-q4km/{reference,native}/proof.asysrep`; the
small archive contains receipts/statistics, not large Asys payloads. Focus ACU
reports and raw CSVs are under `results/decode-focus` and included in the archive.

Resume only failed experimental points with the same source/runtime/device:

```bash
CUDA_VISIBLE_DEVICES=6 python3 -u tools/run_decode_focus_box.py \
  --sdk "$PPU_SDK" --result-root /sim/eec/shared/junfu.qx/kpack-newbox.QYl8F9 \
  --output /absolute/new-run/results/decode-focus --resume
```

Do not reuse model ABBA timings when other GPU workloads overlap. Captures and
component sums are diagnostic, not replacements for warmed unprofiled TPOT.

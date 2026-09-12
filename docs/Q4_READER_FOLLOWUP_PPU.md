# Q4 cold reader followup: remaining four shapes

The latest user decision permits **up to 5% slower than raw-GGUF reference**.
This supersedes the earlier zero-regression tuning gate, but does not rewrite
historical experiment protocols or their verdicts. The accepted regime remains
cold/rotating weights, dense M=1, FP16 A, FP32 accumulation and output, Q4_K.
No production selector, offline format, or shipping library changes here.

The [last reader comparison](Q4_READER_REUSE_RESULTS_20260912.md) closes
N5120/K8192 (+1.58% versus ref) and N8192/K5120 (-7.46%) under the updated
criterion. They are **not remeasured in this followup**. M2–7, indexed/grouped
GEMV and other qtypes still need separate admission.

## Inventory and controls

| N | K | Historical K-pack winner retained | Reader candidates |
|---:|---:|---|---:|
| 512 | 2048 | cooperative metadata, Width8/Warps16 | 44 |
| 1024 | 5120 | affine C4/W10/P2 | 34 |
| 4096 | 2048 | affine C8/W16/P4 | 40 |
| 4096 | 4096 | affine C8/W16/P4 | 40 |

The immutable prior config sweep supplies the starting geometry shortlist.
The exact bounded inventory and pruning rules are in
`dev/gemv_ppu/reader_followup.py::plan` and the package manifest. There are
158 candidate contexts, not a full Cartesian sweep or proof of global optimality.

The affine readers reuse the previously tested A reuse / metadata reuse /
32-bit header extraction switches. Only C4/C8 can use cooperative A or units;
C2 spans superblocks within a warp, so C2 is restricted to original or H32-only
decoding. Whole inactive tail warps are permitted; partially active shuffle
groups are not. P8 tests 16-byte per-lane B requests at selected geometries.

The smallest shape keeps its distinct cooperative-metadata reader. It already
uses vector A loads and a register transpose. Width8/16/32 and Warps4/8/16
are tested with the original header extraction and a 32-bit extraction clone.
The clone preserves half reconstruction, code conversion, dot and reduction
order. **Width8/Warps16 is the actual historical winner**; its old API recipe
`[1,16,1]` is not an affine Columns/Warps/P tuple.

Numeric checks precede each usable timing:

- Independent original-GGUF dot oracle, nonzero input, zero-code and zero-A
  negatives, output/workspace guards, deterministic post-replay output.
- Affine: exact FP32 bits against the immutable config DSO at the same C/W/P.
- Small: exact FP32 bits against the unchanged small body at the same
  Width/Warps. The historical Width8/Warps16 also checks against its immutable
  old binary. Newly instantiated small geometries are not falsely described
  as previously measured or as an independent decoder oracle.

Raw reference and Xplane reconstruct individual weights in FP16; affine
K-pack performs FP32 group-affine arithmetic. All accumulate/output FP32,
but rounding is not identical across these families. Cross-family admission
uses the independent oracle, not raw-bit identity between different formulas.

## Work and timing scope

1. Screen all 158 candidates and 12 control cells, five event samples each.
2. Confirm the top three candidates per shape plus old winner/ref/Xplane:
   six alternating rounds of fifteen samples, 144 timing cells.
3. Capture ACU for the top two confirmed candidates and the three controls
   per shape: 20 reports. ACU durations do not replace event timing.

Total: **314 timing cells + 20 profiles** on a complete first run. Warmup,
graph upload and first-use are excluded; no JIT or compilation runs on box.
Each call is one complete S1 kernel, without inter-CTA Split-K or a reducer.
The ring covers at least 2.25 times verified L2 and graphs traverse complete
rings. Source footprints for A/B/metadata, 32/64/128-byte granules and observed
base alignment are recorded per candidate. Native ISA checks establish fast
code conversion and FP32 FMA, not dynamic bandwidth or PPU correctness.

Local compilation produced five isolated DSOs, 158 native specializations,
in 29.6 seconds (about 3 MB including receipts). Box runtime is unmeasured for
this expanded job. The previous two-shape job took 338.6 seconds, including
24 profiles; this job has more child processes and smaller weights. Allow
roughly **10–20 minutes plus fixture generation** on an idle device, an
estimate rather than a deadline. Progress includes phase, shape, cell count
and elapsed time. Do not run inference on the measured device concurrently.

## Entry points

`tools/run_q4_reader_followup_ppu_box.sh` executes the local prebuilt package
and packages `results/` even after a failure. Run it with `bash`, never source
it. Its subshell preserves the caller's Docker shell.

```bash
CUDA_VISIBLE_DEVICES=0 \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
L2_BYTES=67108864 \
bash tools/run_q4_reader_followup_ppu_box.sh
```

ACU is on by default; `ACU=0` explicitly omits counters. `FIXTURES` may point
to the prior six-shape fixture directory. To resume, set `RESUME_RUN` to the
exact printed run directory and use the same command. Successful cells and
profiles are hash-checked and reused; failed/missing cells run in fresh child
contexts. If recovered screen cells change the top three, new shortlist cells
are measured and their already valid intersections are reused. Source, device,
runtime and fixture identity must stay unchanged for resume.

Return the printed `results=...results.tgz`, containing `summary.tsv`,
`summary.json`, plan, source/image/runtime/fixture authority, per-cell logs,
ISA receipts and ACU reports. Process `status=PASS` means complete valid
measurement; **performance** is separately `WITHIN_5_PERCENT` or `PARITY_OPEN`.
Missing/invalid data yields `INCOMPLETE`, never a performance pass.

Package: `prebuilt/ppu0010/q4-reader-followup-v1`; transitive controls are the
frozen reader-reuse, config-sweep, cold-shapes, h800-port and simt-ab packages.
These are development experiment artifacts, not the final product bundle.

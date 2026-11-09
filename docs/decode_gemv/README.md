# Decode GEMV / small-M study on ZW610 (M=1..8)

Baseline: `origin/develop` 5b5b9a45 (deployed revision), production `prepare_v1` selection,
public dense (F16) / MoE (BF16) ABI, Asys GPU kernel activity, cold weights.

| document | content |
|---|---|
| `plan.md` | identity, environment, measurement protocol, iteration plan |
| `workload_manifest.json` | 5 models, 55 unique Quactlize shapes with calls per token, non-Quactlize matrices listed separately |
| `draft.md` | baseline evidence (M=1), C1 register-direct MMA reader results, step-1 summary |
| `step1-access-pattern.md` | step 1: prototype vs production TC vs streaming probes (ACU + Asys) |
| `actlize-extension.md` | optional exact-code collective, delivery policies and bounded compile gate |

Data (`data/`, small files only; traces and ACU reports stay in the investigation root):
- `baseline-m1.parsed.jsonl`: production M=1, one Asys round, per-call kernel components.
- `screen-mma-reader-v3-m1-best.csv`: best register-direct MMA-reader config per shape vs the
  in-run production arm (single round; `base_capture`/`best_capture` = `foreign-light` marks
  the captures during which a light foreign pytest ran on the card).
- `correct-mma-reader-v3.jsonl` (46 cases), `correct-mma-staged-v1.jsonl` (31 cases, 2 rejected
  configurations by design): `qgemv_bench.py --check` against gguf-py.
- `pattern-sweep{1,2}.{configs.txt,kernels.jsonl}`: access-pattern probe sweeps (Asys).
- `acu-step1-*.md`: ACU comparison tables (`tools/decode_gemv/acu_summary.py`).

Paths in the documents (`tools/...`, `runs/...`, `profile/...`, `inventory/...`) are relative
to the investigation root; in this tree the tools are `tools/decode_gemv/`.

Status: research prototypes only. `quactlize/execution/mma_reader.cuh` (register-direct,
correctness-checked, screened) and `quactlize/execution/mma_staged.cuh` (cp.async-staged,
correctness-checked, not timed) are not included by any product source, build or selection.

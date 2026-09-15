# Small-M decode selection tables

The automatic caller now asks one library selector for both SIMT and TC.
This changes decode decisions for Q2_K, Q3_K, Q5_K, Q6_K and Q8_0, including
requests absent from the exact table. Q4 keeps its existing joint SIMT/TC
table. Prefill selection is unchanged.

## Lookup contract

1. Look up `(qtype, dense/indexed, N, K, tokens, A channels)` in the exact
   table. `QKS_SMALLM_EXACT` means an exact key, not a new model admission.
2. Otherwise use the nearest occupied logarithmic bucket of the same
   format, operator and channel mode. Tokens have priority over N/K
   distance. This is logged as `BUCKET_PREDICTED` (`QKS_SMALLM_BUCKET`).
3. Validate the chosen recipe and artifact contract. Unsupported requests
   return MISS to the existing selector; a missing TC module requires its
   normal compile-only JIT, not an unmeasured SIMT substitution.

Scope is F32 caller input/output, tokens 1--8. Indexed requests use E=256,
top-8 and either shared A or one A vector per slot. No host ID readback or
online timing is introduced. SIMT accumulation/output remain F32; the
current activation arithmetic still rounds to F16. This does not fix the
separate wide-range activation overflow problem.

There are 305 exact entries, 213 bucket representatives and 30 exact SIMT
choices. These are generated data, not 305 hand-written conditions.
`quactlize/dispatch/smallm.hpp` is the common lookup; the caller does not
contain a second per-format policy. Regenerate JSON and C++ together with:

```bash
python3 tools/fit_kpack_smallm.py
python3 -m pytest -q tests/test_kpack_smallm.py
```

The compact evidence is `docs/measurements/smallm_20260915.json.gz`.
It retains the selected SIMT recipe's four rounds of eleven samples,
numerical replay proof and original result hashes. It also retains the
historical TC full-call observations and model Q8 producer-plus-reducer
observations. Source hashes are in the evidence and generated JSON.

| Request | New automatic choice |
| --- | --- |
| Q8 dense M1, N512 K2048 | SIMT v1/C4/W8/P2/S1 |
| Q2 dense M1, N512 K2048 | SIMT v2/C4/W8/P2/S1 |
| Q5 indexed down, one token, N2048 K512 | SIMT v3/C4/W2/P8/S1 |
| Q3 dense M4, N1024 K5120 | TC 16x16x256/W16x16/S2, Split-K4 |
| Q6 dense M1, N5120 K8192 | TC 8x128x128/W8x32/S2, Split-K8 |

## Evidence boundary

The 200-context PPU SIMT sweep passes correctness, but its old scalar SIMT
control is not the TC incumbent. TC historical full-call costs and the Q8
model trace are different measurement cohorts. A SIMT proposal requires
10% headroom even against the fastest available old TC observation; all
required Split-K reduction is included. No contemporaneous speedup,
5% parity guarantee or model accuracy admission follows from that rule.
Requests with no TC comparison remain review candidates, not fabricated
exact SIMT wins. Bucket predictions are explicitly labeled.

The caller logs the table source and full selected recipe. The trace reader
checks the SIMT recipe against the execution-library inventory and matches
the actual register-reuse compute symbol, not only routing/reduction kernels.
Mixed MoE chains use v3 endpoints and private per-projection Split-K
workspace. Q4's old endpoint and older callers remain supported.

## Next PPU model check

Published source `e82a8b7`, private caller `00d24db55`, runtime artifact
`01d8537`. The 12 ELF payloads total 23,366,792 bytes (about 22.3 MiB) and
use Git LFS. They contain no llama binaries. The old artifact commit remains
available; no old box results or caches need to be deleted. Local lookup,
ABI, composition and package tests pass, and the caller adapters compile
with the PPU SDK. This is not a new PPU model performance result.

The regular model runner has a bounded performance mode:

```bash
MODEL_PHASES=perf MODEL_NAMES=qwen35-35b-q4km \
LLAMA_CI_DIR=/sim/eec/shared/junfu.qx/llama.cpp \
NCP_LIB_DIR=/sim/eec/shared/junfu.qx/ncp_flash_lib \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 JOBS=192 \
bash tools/run_kpack_q4_model_box.sh
```

Update both development branches first. The runner fetches only its pinned
Quactlize package, builds llama through `.aoneci`, checks six selected mixed
chains (tokens 1/2/8, supplied/fused router), then runs warmed ABBA PP2048/TG128
and a separate warmed Asys capture. No full sweep is required. Compilation,
JIT and the first model request are excluded from performance samples.
Existing caller/NCP builds may be reused through `LLAMA_CI_BUILD_DIR` and
`NCP_CI_DIR`; their source/compiler checks still apply.

Return the printed results archive. Full `.asysrep` files remain under
`$RUN/results/trace/*/{reference,native}/proof.asysrep` and are not embedded
in the small archive. This mode intentionally does not rerun numerical PPL;
it reports `accuracy=NOT_RETESTED`. Do not interpret a fast run as a numeric
fix for Qwen3-32B. Genuine BF16 compute is developed separately and is not
part of this performance package.

# Selection, inventory and prewarm

Small-M Auto selection uses one host implementation,
`quactlize/dispatch/selection.hpp`, for the public v3 query and offline planning.
It does not time kernels, initialize a device or run an online search.

| Responsibility | Owner |
|---|---|
| Exact/bucket eligibility, exclusions and donor order | `smallm_matched.hpp` |
| Fold measured Q8 replacements into the existing donors | `effective.hpp`, evaluated at compile time |
| Final kind, recipe, compute precision and provenance | `selection.hpp` |
| Generic/fixed/hoisted producer and conditional reducer | `execution/simt_strategy.hpp`, also used by launchers |
| Load/JIT the selected TC, query resources, cache handles | `dispatch/binding.cpp` |
| Check exact compiled inventory and plan typed prewarm | `dispatch/planning.py` |

An exact winner remains measured; a bounded bucket donor remains predicted.
Changing a TC donor to a measured SIMT implementation does not broaden the
original donor's eligibility. Unresolved exclusions remain misses. No blanket
all-SIMT rule is applied. Paired GateUp+SwiGLU retains its separate operator and
layout contract; an ordinary projection result cannot select a fused chain.

## Explain the final Auto decode choice

From the repository root, create a fresh output directory:

```bash
python3 tools/kpack_jit.py plan-smallm \
  --request 8 0 2048 4096 1 1 1 1 0 \
  --request 13 2 2048 512 256 8 8 1 1 \
  --output /path/new-smallm-plan
```

Each request is `Q MODE N K EXPERTS TOPK CHANNELS TOKENS COMPUTE`:

- `MODE`: 0 dense, 2 indexed/MoE; `TOKENS`: 1–8.
- `COMPUTE`: 0 F16, 1 BF16; external input/output storage is F32.
- Q8_0 has qtype 8; Q2_K through Q6_K have qtypes 10 through 14.
- Indexed requests currently require 256 experts, top-k 8 and channels 1 or 8.
- Use the **local shard** N/K, not an unsplit TP model's dimensions.

`plan.json` records the status, exact/bucket policy, donor, kind, recipe and
implementation strategy. A policy miss produces a nonzero exit code with the
incomplete plan retained. It does not invent a tactic or emulate the caller's
legacy fallback. Planning checks structure, not device correctness/performance.
For TC, final occupancy, grid and workspace still require a real module query.
For reductions, vectorization also depends on runtime pointer/stride alignment.

## Prewarm only what was selected

```bash
python3 tools/kpack_jit.py prewarm --sdk /path/ppu-sdk \
  --plan /path/new-smallm-plan/plan.json \
  --cache /path/kpack-module-cache --jobs 8 \
  --receipt /path/new-smallm-plan/prewarm.json
```

Typed plans own compute and endpoint identity; do not add `--compute-type` or
`--dense-io`. Only distinct selected TC parents compile. SIMT and Q4 readers are
already in the execution library. An all-SIMT plan compiles zero kernels. This
is CPU compilation, not tuning or device admission. It belongs before capture.

The legacy `kpack_jit.py plan --request Q ROUTE M N K E MAX_ROWS` explicitly plans
**fixed FQ/SF TC routes**, useful for component experiments and prefill. It is
not an Auto decode plan. Its JSON scope is labelled accordingly.

## Package and cache boundary

`build_kpack_dispatch.py` additionally writes `final-selection.json` for the
complete exact catalog (bucket donors share this catalog). It validates each
selected SIMT tuple `(qtype, compute, variant, columns, warps, values, split)`,
the Q4 execution-policy identity and typed TC module availability. Missing TC
modules require explicitly enabled JIT; a manifest reports that requirement.
Missing SIMT recipes are packaging errors, not reasons to silently use TC.

Internal cache keys separate selection profile, channels, compute and endpoint
type. A host policy change alone does not invalidate unchanged TC compilation
keys. Public C ABI and offline arrangement identities are unchanged.

For model performance, exclude cold JIT/first-use requests and record their cost
separately. Component gates and prewarm receipts do not prove end-to-end TPOT.

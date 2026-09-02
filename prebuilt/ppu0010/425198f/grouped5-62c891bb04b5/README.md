# Five-format grouped K-pack router pilot

This is an execute-only PPU0010 measurement artifact. It was built from clean
source `425198f5d52377faf85eae4160cd44826e7f4388` with actlize
`8d46b758c8931807df840a6ed87d272d74a8fdf4`, CUTLASS
`f94ec46f4f63f96003d6cfdf2014731e7672c281`, and PPU SDK
`2.1.1-a5c56e`.

The bundle manifest SHA-256 is
`62c891bb04b5211601897290c8d5c4259ca8c4d87a07ad2854b5648843f760dc`.
It binds one executable and one format-selected runtime library for each of
Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K. The `evidence/` directory records the exact
build input authority and per-format build logs.

The pilot measures six router profiles for each format. A successful run proves
the 30-cell raw-bit and timing denominator only; it is not by itself a shipping
selector decision. Run the independent 3% minimax/regret adjudicator from
develop commit `b87c4d7` or later on the returned raw logs.

## Verify and execute

Use a clean source worktree at the exact source commit. Hydrate this artifact's
LFS files, source the target box SDK, and run:

```bash
SOURCE=/workspace/quactlize-source-425198f
ARTIFACT=/workspace/quactlize-artifact-a05-425198f
REL=prebuilt/ppu0010/425198f/grouped5-62c891bb04b5
BUNDLE="$ARTIFACT/$REL/bundle"
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK/CUDA_SDK
OUT=/workspace/quactlize-a05-grouped-router-425198f

test "$(git -C "$SOURCE" rev-parse HEAD)" = \
  425198f5d52377faf85eae4160cd44826e7f4388
test "$(sha256sum "$BUNDLE/manifest.json" | awk '{print $1}')" = \
  62c891bb04b5211601897290c8d5c4259ca8c4d87a07ad2854b5648843f760dc
test ! -e "$OUT"

source "$PPU_SDK/envsetup.sh"
CUDA_VISIBLE_DEVICES=0 \
PPU_SDK="$PPU_SDK" \
FQ_GROUPED_MULTI_ROUTER_BUNDLE="$BUNDLE" \
PERF_ITERATIONS=11 \
PERF_WARMUPS=3 \
PERF_ROUNDS=3 \
OUT="$OUT" \
bash "$SOURCE/tools/run_fq_grouped_multi_router_prebuilt_box.sh"
```

The measurement closure requires 15 qtype/round completion rows, 30 aggregate
profile rows, and both:

```text
FQ_GROUPED_MULTI_ROUTER verdict=PILOT_COMPLETE qtypes=5 profiles=6 cells=30
[fq-grouped-multi-router] DIAGNOSTIC_COMPLETE ...
```

Every measured candidate must report `raw_bad=0`. Preserve `OUT/runs` in full;
winner-only output cannot adjudicate the grouped selector.

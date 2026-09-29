# Branch and build policy

As of 2026-09-29, the maintained branches are:

| Branch | Purpose |
|---|---|
| `main` | Product source; this consolidation does not change its contents. |
| `develop` | Integrated development source, including TP2 and PPU1.7. |

`develop` includes the local and published develop histories, the TP2 chain
through `a8f396a`, the isolated PPU1.7 CUTLASS 4.3 integration at `5991449`,
and the short-K Q4 admission fix at `025c7e4`. The intermediate TP2, dispatch,
GEMV and fusion branches are no longer needed as development entry points.
The BF16 branches have patch-equivalent implementations already in develop.
Old pre-rebase history is archived, not replayed over the newer implementation.

Published source branch tips are retained as immutable archive tags under
`archive/source-20260929/`. For example, the legacy short-K rebuild helper
fetches `refs/tags/archive/source-20260929/fix/q4-smallk-admission` and still
verifies its exact source commit. Archive tags are historical inputs, not
additional maintained branches or performance recommendations.

PPU1.0/1.5 remain on `third_party/actlize`. PPU1.7 is separately built from
`quactlize/backends/ppu17` with the pinned PPU CUTLASS 4.3 tree; its build and
device-validation boundaries are documented in
[the backend README](../quactlize/backends/ppu17/README.md).
Merging the sources does not constitute a new PPU device or model-performance
validation, and does not change the main branch's admission policy.

## Build on the box

Prebuilt binary delivery through `artifacts/*` branches is retired. New
deliveries use source plus a reproducible box-side build command. Compiled
outputs, JIT caches and model binaries are local outputs, not source branches.
Offline K-pack **weight formats** are unchanged; they are not the retired
binary-delivery artifacts.

Historical reports and result pins remain as evidence. Their old commands
that fetch `artifacts/*` are not valid fresh-install entry points. In
particular the model, TP2, prefill and legacy overlay runners still containing
such downloads need source-build orchestration before a new cold run; this
branch consolidation does not claim that migration is complete. The legacy
short-K helper additionally requires an already available base bundle.

Uncommitted work and experimental output files are preserved in their original
worktrees. Retiring a branch does not discard those files or imply that their
unreviewed changes were merged.

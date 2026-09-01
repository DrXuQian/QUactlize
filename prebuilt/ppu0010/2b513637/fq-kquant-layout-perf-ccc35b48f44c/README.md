# PPU0010 prebuilt FQ K-quant layout performance bundle

This artifact executes the production-C-ABI Xplane/K-pack benchmark for
Q2_K, Q3_K, Q4_K, Q5_K and Q6_K without compiling on the PPU box.  It is
bound to source commit `2b513637fc3d315077b14ab81784ff1fb21e1bb7` and the
pinned actlize/cutlass submodule commits recorded in `manifest.json`.

The ten large payloads are Git LFS objects: one
`test_fq_kquant_layout_perf` executable and its matching
`libquactlize_ppu.so` for each qtype.  Never mix a binary and library from
different qtype directories.  `verify-bundle.py` checks all payload sizes,
modes and SHA-256 digests, the qtype/packed-format mapping, both immutable
plans and the exact source-side planner/analyzer/fitter authorities before
execution.

## Fetch and verify

```bash
git fetch origin artifacts/ppu0010/2b513637-fq-kquant-ccc35b48f44c
git switch --detach origin/artifacts/ppu0010/2b513637-fq-kquant-ccc35b48f44c
git lfs pull

BUNDLE=prebuilt/ppu0010/2b513637/fq-kquant-layout-perf-ccc35b48f44c
"$BUNDLE/run-prebuilt.sh" --verify-only
```

The build used installed PPU SDK release `2.1.1-a5c56e`; the compiler binary
SHA-256 is `fa62c590c67411c23fa4028f15fa562b39ce0cf830830d038a1ec04c59d8c76e`.
The official 1,875,743,407-byte archive at `/root/ppu-sdk-cache/` was
independently re-hashed during this build as
`63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd`;
the manifest records the exact path, size and `verified_this_build: true`.

The SDK targets Ubuntu 24.04.  Its wrapper requires `GLIBC_2.38` and
`GLIBCXX_3.4.32`; this bundle deliberately refuses any other distribution or
an older runtime.  Device compilation completed on the Ubuntu 22.04 build
host.  Source `2b513637` did not yet put the compatibility option on this
target: after its device objects and libraries completed, each benchmark
object was therefore re-linked in a controlled step by adding only
`-Wl,--allow-shlib-undefined` to its recorded command.  Commit
`44ca067ca75941ea0c0b861ead324b6bb1cb6881` later codifies that policy for new
builds; it is provenance, not the source of these binaries.  The original
commands and expected initial ABI-floor failures are retained under
`evidence/`.

## Execute only

Provide a previously captured `quactlize-box-identity-probe-v1` JSON, the exact
SDK root, and a fresh strict child of `/workspace`:

```bash
BOX_IDENTITY_JSON=/workspace/box-identity.json \
PPU_SDK=/opt/ppu-sdk-2.1.1 \
OUT=/workspace/fq-kquant-heuristic-run \
  "$BUNDLE/run-prebuilt.sh"
```

`PPU_SDK="$PPU_SDK" "$BUNDLE/run-prebuilt.sh" --preflight-only` performs the
same fail-closed host-floor, SDK hash, ELF NEEDED/RPATH and five loader-closure
checks without launching a benchmark.  The normal execution performs these
checks first and records them in `inputs/runtime-preflight.json`.  It hashes
but never invokes the SDK compiler or inspector, and it has no compilation
fallback.

The default SDK policy remains exact: release receipt, compiler, inspector,
runtime libraries and runtime alias must match the build manifest.  To run a
deliberate compatibility measurement with a different installed SDK, set
`ALLOW_UNVERIFIED_SDK=1`.  This opt-in relaxes identity equality only; the SDK
root, tools, runtime files, runtime alias, host floor and all five ELF loader
closures must still be usable.  The runner records both expected and actual
size/SHA-256 identities, per-field match booleans and mismatches.  A run with
an actual identity mismatch is labelled `evidence_grade=unverified-sdk` in both
`inputs/runtime-preflight.json` and `results/result-authority.json`; it is
never reported as an exact SDK match.

`RESUME=1` binds the already committed runtime preflight and its digest
sidecar before reusing a benchmark log.  Changing the SDK root, actual SDK
identity, strict/opt-in policy, or host preflight requires a fresh `OUT` and
cannot relabel measurements from an earlier run.

If a JSON identity receipt is unavailable, the runner accepts all four
one-line operator assertions `QUACTLIZE_BOX_DEVICE_MODEL`,
`QUACTLIZE_BOX_PCI_IDENTITY` (full PCI BDF),
`QUACTLIZE_BOX_DRIVER_VERSION`, and `QUACTLIZE_BOX_SDK_COMPILER_IDENTITY`.  They
are labelled as weaker, operator-supplied evidence in the result.  An external
JSON is accepted only after `tools/box_identity_schema.py::validate`; an empty
or merely JSON-shaped object is rejected.

The default is the heuristic-training denominator: 143 dense shapes, 52
grouped shapes, every compiled tactic, 11 timed iterations after 3 warmups,
and 3 rounds with alternating A/B order.  Q4_K is grouped-only by contract.
The runner emits `results/summary.json`, `results/summary.tsv`,
`results/config-heuristic.json`, and a hash-bound
`results/result-authority.json`.  `RESUME=1 OUT=...` reuses only committed
logs whose sidecar hashes still agree.

The optional environment controls are `PERF_ITERATIONS`, `PERF_WARMUPS`,
`PERF_ROUNDS`, `REGRESSION_THRESHOLD_PCT`, `HEURISTIC_MAX_LEAVES`,
`HEURISTIC_MIN_LEAF_ROWS`, and `HEURISTIC_MIN_LEAF_FAMILIES`.
`SWEEP_PROFILE=layout-ab` selects the smaller 77+24 plan, but the runner
still requires `SWEEP_CONFIGS=1` so every successful execution produces both
a summary and a measured heuristic.  The runner has no build fallback.

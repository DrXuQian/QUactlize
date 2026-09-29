# Resume

Latest review: [returned paired-chain results](docs/result-20260920.md).
`paired-chain.HLHoKz.results.tgz` passes source/runtime/fixture/240 numeric /
720 timing sample / four re-imported ACU checks. TP2 `paired-1` wins the
complete chain 43.366286 -> 39.847999 us (-8.112954%, six of six rounds).
35B keeps existing `paired-shipping` at 30.855499 us; the new candidate is
9.672511% slower than that incumbent. The scoped development integration is now
implemented; its production-entry/model device gates are pending. No global
paired default. Earlier pending entries below describe historical build-time state.

## Paired recipe integration, 2026-09-20

- Generated semantic request -> complete recipe -> explicit cached launch.
  Ordinary 1,852 exact / 637 bucket selections are byte-identical to the
  baseline. Old ABI selection and its actual hidden reader choices are frozen.
- New TP2 promotion is only Q4 logical N512/K3072, E256/top8, M1, original
  slots, BF16 compute/F32 endpoints, Q5 SIMT down N3072. Explicit M2..8
  capability tests do not promote those requests. N512/K2048 retains its winner.
- Host regression: 165 passed; caller trace/selection tests: 42 passed. Both
  modified caller translation units compile with the PPU SDK. Full caller CI
  is deliberately left to the box through `.aoneci/scripts/build.sh`.
- Fusion build: 16 translation units in 120.3 seconds. New TP2 paired16,
  retained Q4 paired32, and Q8 tile16 all have identical native instruction
  bytes and resource records to their corresponding measured bodies (3/3).
- Package reuses ordinary execution, TC modules, packer and prefill images
  byte-for-byte. Only the small host dispatcher and fusion library are rebuilt.
  No caller binaries are published in Quactlize.
- New entry: `tools/run_paired_selection_box.sh`, devices 6,7. See
  `docs/production-handoff.md` for component/model coverage, resume and return.
  The original experiment entry remains available separately.
- Implementation `3424f553253c09ada4320646780a8e3fba36c601`; caller
  `905dbdd486325189f88e519470a37bbd8badb685` on
  `DrXuQian/llama.cpp:dev/quactlize-paired-recipes`.
- Runtime `f8af8e7e550c9fd4ab7ab216281fbae1bfed8ec6` on
  `artifacts/kpack-paired-recipes-v2`, manifest
  `3e325fcdbd170ee76a0f8aa19056d4cf0b4d077d31a5e7b290bc8e4c958056bd`.
  All 79 published files are tracked; ELF payloads use LFS. Own-fork push,
  fresh scoped LFS checkout, full source/SDK verification and the actual shell
  entry's `VERIFY_ONLY=1 PAIRED_PHASES=components` passed (`runner_rc=0`).
  These checks did not create a device context or measure performance.
- Final fusion SHA-256
  `cd80283d0b77c4a6c782b8353cd2eb4c0f29765fff843a16316a84943804af75`;
  repeated native ISA/resource comparison remains 3/3 byte-identical.

2026-09-20: implementing the matched paired-chain experiment after reviewing
`dev/tp2_decode/docs/launch-trace-20260920.md`. Production remains unchanged.
Priority TP2 N512/K3072; 35B N512/K2048 is the retained regression control.
Read `docs/plan.md` and `plan.py` for scope and precision. Compilation,
numerical device admission and full-call timing are separate gates.

## Local handoff, 2026-09-20

- Two candidate TUs compiled in 8.5 seconds; four kernels passed native ISA /
  resource inventory inspection. No shipping runtime or caller was rebuilt.
- The complete chain runner retains the frozen execution and fusion libraries;
  BF16 range, phase 2 activation, internal pointer ownership, aliased router
  weights, and four/five-stage profile coverage have host regression tests.
- Related host regression suite: 164 passed with the existing caller host
  libraries. The caller and production selections remain unchanged.
- Box execution is pending on devices 6,7. No production selector changes and
  no claims about numerical or performance admission from host checks.
- Entry: `tools/run_paired_chain_box.sh`; scope, resume and return protocol are
  in this directory's README. Review complete-chain samples before promotion.
- Source implementation: `70b42cbeb9063b03383aae3eb142094aed684875`.
  Package: `54514275e65ebdbee12f6c153e962ca7b8040732`, manifest SHA-256
  `577c6d368175c39f21ae43f8d7675cd85267f83d968f65f186fbe4d1eccb0c0c`.
  Payload index verified: eleven payloads including all four libraries; six
  binary/ISA payloads use LFS. Approximate package size is 42.9 MB, of which
  almost all is the unchanged frozen execution/fusion control pair.
- Own-fork artifact push and a fresh `fetch -> scoped LFS pull -> manifest /
  source / SDK preflight` completed successfully. Box run still pending; the
  local verification did not create a GPU context or report performance.

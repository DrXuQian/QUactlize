# Resume

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

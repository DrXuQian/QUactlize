# Resume

2026-09-19: baseline TP2 model/ABBA results reviewed; no candidate admitted.
Read `docs/plan.md` for the 10 local decode shapes and eight restriction classes.

Current worktree is `dev/kpack-tp2`; only profiler orchestration and this audit
change here. Production runtime/caller binaries remain those in
`kpack-tp2.om9JQj`. Asys SDK path normalization and a trace-only TP2 resume entry
are host-tested. Private profiler namespace fallback is device-environment
pending; it does not stop host profiler services. No new decode performance
claim has been made.

Local implementation is complete on `dev/tp2-fastpaths`, based on 4326381,
with caller branch `dev/quactlize-tp2-fastpaths`, based on 1bac078de. The original
TP2 branches and runtime pin remain unchanged for trace recovery.

Structural capability and measured default selection are separate. Read
`README.md` for build/replay entrypoints. No new default selector was admitted.

Local evidence (2026-09-19):

- Quactlize host suite: 434 passed, 15 subtests passed (34.67 seconds).
- Caller native/trace suite: 41 passed; graph matching covers189 graphs.
- PPU candidate build: 17 points, 117 SIMT variants and four TC controls,
  95.25 seconds with eight compile jobs. ISA inspection and package verification
  pass; no device was used.
- Prepare gate binary, Q4/Q8 production fusion TUs, BF16 Q8/fixed Q5 smoke
  instantiations and the changed caller TU compile. Dispatcher syntax check passes.
- Package: `/tmp/tp2-fastpaths-package-20260919`, manifest SHA256
  `4addcd47773099db4d636f1c0e0d51b8ac1982624edf109733e17ccca7546460`.
- Prepare: `/tmp/tp2-prepare-build-20260919-r2/bench`, SHA256
  `6383bc26e6b4b6e2e2622ead04154202f994aa680edfd30002d2b59eb5fa8af2`.
- Caller object: `/tmp/tp2-fastpaths-caller-final-20260919.o`, SHA256
  `4df96eb6f82a361ec180f60c22e167c2cdc7462213d0ba764945608de3362ed4`.
- Additional compile receipt: `/tmp/tp2-fastpaths-extra-xu0b2n4n/receipt.json`.

The initial experiment wrapper used the non-Q4 arrangement constructor for
an unpaired Q4 point. The new host query check caught it before a device run;
the final package uses the exact Q4 registry descriptor. Discard the earlier
`tp2-fastpaths-build-20260919` and `-r2` candidates as handoff authorities.
No user files or old artifacts were deleted.

Next requires PPU: run independent numeric controls, cold full-call comparison,
prepare shape/alias gates and ACU. Then promote only measured scopes, compose
the runtime with matching caller, and repeat model ABBA/Asys. The original TP2
branches can still replay the unchanged baseline trace independently.

Device-result update,2026-09-19: the component run has now returned in
`/root/tp2-fastpaths.IwJJxB.results.tgz` (SHA256
`5944aa7547e4dd5c9b87dae0f906102d7df5d4e39f42d7007c25731408fee075`).
17 points/134 total arms/2576 numerical records and34 ACU exports pass local
receipt review. All4590 confirmation samples recompute correctly. Prepare
shape64/router-edge56/alias360 gates pass; K3072 M1 improves37.17%, M8 improves
87.38%. Read `docs/results-20260919.md` and `benchmark.csv` for the complete
table, retained incumbents, ACU evidence and limitations. No production
selection was changed. The next action is selective M1/prepare integration
and matched TP2 model replay, not another unrestricted sweep. Keep Q6 TC and
the old Q4/Q5/Q8-SSM winners; Q4 projection and paired-chain gains differ.

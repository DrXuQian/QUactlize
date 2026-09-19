# Resume

2026-09-19: baseline TP2 model/ABBA results reviewed; no candidate admitted.
Read `docs/plan.md` for the 10 local decode shapes and eight restriction classes.

Current worktree is `dev/kpack-tp2`; only profiler orchestration and this audit
change here. Production runtime/caller binaries remain those in
`kpack-tp2.om9JQj`. Asys SDK path normalization and a trace-only TP2 resume entry
are host-tested. Private profiler namespace fallback is device-environment
pending; it does not stop host profiler services. No new decode performance
claim has been made.

Next: get the unchanged-runtime trace, then implement generalization candidates
in another worktree. Retain the old shape regressions in every relevant gate.
Production selection must not change simply because a template can compile.

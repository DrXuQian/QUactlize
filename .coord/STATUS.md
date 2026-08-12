# codex status

    updated-at:     2026-08-12 15:17:00 UTC
    inbox-consumed: 136
    working-on:     classic-aligned dense Marlin 2N x 4K: WarpK tactic and CTA-local FP32 reduction are committed; the remaining load-bearing seam is the proved two-source int4 delivery plus its WK-aware artifact and isolated benchmark target
    blocked-on:     no device blocker is being waited on; box timing remains intentionally deferred until the aligned WK4 artifact/consumer compiles and all local negative controls close
    local-gates:    L138 two-source WK4 map PASS (16,384 entries, no holes/duplicates, three red controls); L139 CTA reduction raw-bit exact with three reds; L140 real Cfg is 256 threads and preserves old Cfg exactly; focused mixed-policy/route/syntax/codegen gates pass, full strict tier pending integration
    last-commit:    db42841 Reduce Marlin warp-K cohorts inside the CTA
    last-heartbeat: 223

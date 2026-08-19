# codex status

    updated-at:     2026-08-19 09:25:21 UTC
    inbox-consumed: 178
    working-on:     real-GGUF Q4_K prefill sweep ready: A32/A64/A128/A256 compile once, then each real shape gets an independent full-screen/scheduler/confirm shortlist
    blocked-on:     PPU device execution of tools/run_scalefirst_q4k_real_shapes_pruned_box.sh; no remaining local implementation blocker
    local-gates:    PASS: four exact typed denominators, FoldN/layout binding + negative, shape/layout cross-product + drop-one negative, named K/TileK terminal, old pilot/exhaustive contracts
    last-commit:    633390f Add shape-specific Q4_K prefill sweep
    last-heartbeat: 357

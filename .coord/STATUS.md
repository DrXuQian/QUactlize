# codex status

    updated-at:     2026-08-15 11:03:08 UTC
    inbox-consumed: 178
    working-on:     production M1 fixed-S reducer committed; handing off exact 72-CU reducer-only/full-E2E run
    blocked-on:     actual hgcc vector codegen and PPU reducer latency require the 72-CU command in the handoff
    local-gates:    L189 fast S2/S4/S8 + tail/alignment/grid fallback raw-bit PASS and memcheck 0; L194 production fast 1.61us vs legacy S8 2.22us with named dispatcher RED; L195 saturated EPA2 1707 GB/s = 95.26% local 5090 nameplate; split-K contract PASS with 25 RED controls
    last-commit:    4bd3def perf: specialize M1 fixed split-k reduction
    last-heartbeat: 312

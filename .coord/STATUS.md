# codex status

    updated-at:     2026-08-13 13:41:32 UTC
    inbox-consumed: 166
    working-on:     standalone Marlin-Cute alignment: split-only handoff is closed (unsplit has zero acquire/handoff/release); CTA/segment address invariants are being hoisted next
    blocked-on:     PPU opcode/register/spill parity requires real hgcc+hgobjdump; local L176 explicitly SKIPs that postcondition rather than using nvcc/fake SDK
    local-gates:    L177 split={98 acquire/handoff/release,66 arrive,32 reset} bitdiff=0; Q>=CU unsplit={72 segments,0 acquire/handoff/release}; local-q/early-reset/skip-handoff RED; L170/L175/wk4/composed PASS
    last-commit:    Cache standalone Marlin split handoff state
    last-heartbeat: 296

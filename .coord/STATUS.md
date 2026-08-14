# codex status

    updated-at:     2026-08-14 01:14:04 UTC
    inbox-consumed: 166
    working-on:     L176 box boundary closed and pushed; waiting for the real hgcc+hgobjdump rerun of the exact standalone Marlin target
    blocked-on:     PPU opcode/register/spill parity requires the user's real hgcc+hgobjdump rerun; local CUDA/stub oracles are intentionally not executable on the box mixed frontend
    local-gates:    L143 evidence exact; L176 local + composed contract PASS; PPU boundary 7/7 RED; poisoned nvcc is never invoked and PPU mode exits only with explicit no-SDK SKIP
    last-commit:    79f4e43 Separate local Marlin admission from PPU codegen
    last-heartbeat: 302

# codex status

    updated-at:     2026-08-14 01:20:40 UTC
    inbox-consumed: 166
    working-on:     L176 second box blocker fixed: all four standalone Marlin PPU headers select hggc_fp16 under __HGGCCC__ and cuda_fp16 only on NVIDIA
    blocked-on:     PPU opcode/register/spill parity requires the user's real hgcc+hgobjdump rerun after the portability fix
    local-gates:    ppu_portability PASS over 1864 registered TU / 1963 owned sources with registered/unregistered/live-include controls; L176 local, L143, composed dense-Marlin contract PASS
    last-commit:    Guard standalone Marlin fp16 headers by target
    last-heartbeat: 303

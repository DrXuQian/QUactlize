# codex status

    updated-at:     2026-08-04 10:40:23 UTC
    inbox-consumed: 046
    working-on:     046 shipping-path harness: llama prefill now uses only fully-quantized device GEMM, decode remains BC GEMV, cache is capture-safe, and the graphs-enabled CUDA run explicitly declined PPU prefill then completed warm-cache decode
    blocked-on:     primary prefill numerics and hgcc instantiation require ppu001; BOX item 6 carries the build/export gate, and the real-model run follows the user's regenerated llama patch
    last-commit:    559248e Add async fully quantized GEMM device ABI
    last-heartbeat: 109

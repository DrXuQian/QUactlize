#pragma once

namespace quactlize::prefill {
struct DenseHelpers {
    int dequant = -1;
    int io = -1;
};

// Qwen3-32B PP2048 / ZW810 R4: independent numerical, component and full-call
// evidence. This selects helpers, NOT the GEMM route or its measured cost.
// Unknown devices, M, formats and N/K keep their existing implementation.
// Evidence: docs/PPU_PREFILL_DENSE_PRODUCTION.md. Do not extrapolate IO3 to
// small outputs (measured regression), or DQ21 to Q4 down (within noise).
inline DenseHelpers dense_helpers(int q, int m, int n, int k, int storage) {
    if (m != 2048 || (storage != 0 && storage != 1)) return {};
    if (q == 12) {
        if ((n == 25600 || n == 8192) && k == 5120) return {21, 3};
        if (n == 5120 && k == 8192) return {20, 1};
        if (n == 5120 && k == 25600) return {8, 1};
        if (n == 1024 && k == 5120) return {20, 2};
    }
    if (q == 14) {
        if (n == 5120 && k == 25600) return {21, storage ? 2 : 1};
        if (n == 1024 && k == 5120) return {20, 2};
    }
    return {};
}
} // namespace quactlize::prefill

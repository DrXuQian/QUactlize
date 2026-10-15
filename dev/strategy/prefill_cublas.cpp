#include <hggc_runtime_api.h>
#include <cstdint>
#include <exception>
#include "dev/strategy/prefill_events.hpp"

extern "C" int strategy_prefill_io(int, int, const void*, void*, int, int, int64_t,
                                  const int*, int, int*, void*);

extern "C" int strategy_prefill_cublas_launch(void*, void*, const void*, const void*, void*, int, int, int);

extern "C" int strategy_prefill_cublas_events(void* entry, void* handle, void* stream,
        const void* src, void* a, const void* const* weights, void* out, void* dst, int* error,
        int m, int n, int k, int copies, int warmups, int rounds, int count,
        float* event_us, double* submit_us) {
    if (!entry || !handle || !src || !a || !weights || !out || !dst || !error ||
        m <= 0 || n <= 0 || k <= 0 || copies < 2 || copies > 4096) return 1;
    for (int i = 0; i < copies; ++i) if (!weights[i]) return 1;
    auto call = [&](int phase, int copy) {
        if (phase == 0) return strategy_prefill_io(1, 1, src, a, m, k, k, nullptr, m, error, stream);
        if (phase == 2) return strategy_prefill_io(1, 0, out, dst, m, n, n, nullptr, m, error, stream);
        return strategy_prefill_cublas_launch(entry, handle, a, weights[copy], out, m, n, k);
    };
    try {
        return strategy_prefill::ordinary_events(static_cast<hggcStream_t>(stream), copies, warmups,
            rounds, count, call, event_us, submit_us);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "STRATEGY_CUBLAS_TIMER exception=%s\n", e.what());
        return int(hggcErrorMemoryAllocation);
    }
}

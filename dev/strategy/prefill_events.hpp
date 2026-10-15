// Experiment-only ordinary-call timer. SDK runtime declarations come from the carrier.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace strategy_prefill {

// Every phase has its own event pair. No graph, Python callbacks, per-sample
// synchronization, or numerical/readback work between timed rounds.
template<class Call>
int ordinary_events(hggcStream_t stream, int copies, int warmups, int rounds, int count,
                    Call call, float* event_us, double* submit_us) {
    if (!stream || copies < 2 || copies > 4096 || rounds != 3 ||
        count != std::max(3, copies) || warmups < std::max(11, 2*copies) ||
        warmups > 100000 || warmups % copies || !event_us || !submit_us) {
        return int(hggcErrorInvalidValue);
    }
    const int slots = rounds*count*3;
    // One extra pair warms the event path, with the last ring address.
    std::vector<hggcEvent_t> events(2*(slots+1), nullptr);
    int created = 0;
    auto report = [](const char* stage, int rc, int slot) {
        if (rc) std::fprintf(stderr, "STRATEGY_CUBLAS_TIMER stage=%s rc=%d slot=%d\n", stage, rc, slot);
        return rc;
    };
    auto finish = [&](int rc) {
        // Also drain pending calls on a failed submission before releasing events.
        if (rc) report("error-drain", int(hggcStreamSynchronize(stream)), -1);
        for (int i = created-1; i >= 0; --i) {
            const int cleanup = report("event-destroy", int(hggcEventDestroy(events[i])), i);
            if (!rc) rc = cleanup;
        }
        return rc;
    };
    for (auto& event : events) {
        int rc = report("event-create", int(hggcEventCreate(&event)), created);
        if (rc) return finish(rc);
        ++created;
    }
    for (int i = 0; i < warmups; ++i) {
        for (int phase = 0; phase < 3; ++phase) {
            int rc = report("warmup-call", call(phase, i % copies), i*3+phase);
            if (rc) return finish(rc);
        }
    }
    int rc = report("warmup-start", int(hggcEventRecord(events[2*slots], stream)), -1);
    if (!rc) rc = report("warmup-provider", call(1, copies-1), -1);
    if (!rc) rc = report("warmup-stop", int(hggcEventRecord(events[2*slots+1], stream)), -1);
    if (rc) return finish(rc);
    using Clock = std::chrono::steady_clock;
    for (int i = 0; i < slots; ++i) {
        rc = report("record-start", int(hggcEventRecord(events[2*i], stream)), i);
        if (rc) return finish(rc);
        const auto before = Clock::now();
        rc = call(i % 3, (i/3) % copies);
        const auto after = Clock::now();
        if (report("measured-call", rc, i)) return finish(rc);
        rc = report("record-stop", int(hggcEventRecord(events[2*i+1], stream)), i);
        if (rc) return finish(rc);
        // Diagnostic only: never subtract host submission from GPU event time.
        submit_us[i] = std::chrono::duration<double, std::micro>(after-before).count();
    }
    rc = report("completion", int(hggcEventSynchronize(events[2*slots-1])), -1);
    if (rc) return finish(rc);
    for (int i = 0; i < slots; ++i) {
        float ms = 0;
        rc = report("elapsed", int(hggcEventElapsedTime(&ms, events[2*i], events[2*i+1])), i);
        if (rc) return finish(rc);
        if (!std::isfinite(ms) || ms <= 0) return finish(report("invalid-time", int(hggcErrorInvalidValue), i));
        event_us[i] = ms*1000;
    }
    return finish(0);
}

} // namespace strategy_prefill

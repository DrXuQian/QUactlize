// Diagnostic-only ordinary submission shared by native and Quactlize.
// The caller supplies POINT_RT(cuda/hggc names) before including this file.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace prefill_point {
using Stream = POINT_RT(Stream_t);
using Event = POINT_RT(Event_t);
inline void check(int rc, const char * what) {
    if (rc) throw std::runtime_error(std::string(what) + " rc=" + std::to_string(rc));
}

// mode0: isolated stage, mode1: per-stage intervals inside the sequence,
// mode2: one interval around the complete sequence (no inner events).
// No event allocation, Python, readback or synchronization between samples.
template<class Invoke>
void sample(Stream stream, int stages, int copies, int mode, int stage,
            int samples, Invoke invoke, double * values) {
    if (stages < 1 || copies < 1 || samples < 3 || mode < 0 || mode > 2 ||
        stage < 0 || stage >= stages || !values) throw std::runtime_error("bad point timer contract");
    const int width = mode == 1 ? stages : 1;
    const int calls = std::max(samples, copies);
    std::vector<Event> events(2*calls*width, nullptr);
    auto cleanup = [&] {
        for (auto e : events) if (e) POINT_RT(EventDestroy)(e);
    };
    auto one = [&](int phase, int copy) { check(invoke(phase, copy), "point stage"); };
    try {
        for (auto & e : events) check(POINT_RT(EventCreate)(&e), "point event create");
        // Seed every intermediate, then warm this exact measurement schedule.
        for (int i = 0; i < 2*copies; ++i)
            for (int j = 0; j < stages; ++j) one(j, i % copies);
        for (int i = 0; i < std::max(16, 2*copies); ++i) {
            if (mode == 0) one(stage, i % copies);
            else for (int j = 0; j < stages; ++j) one(j, i % copies);
        }
        for (int i = 0; i < calls; ++i) {
            for (int j = 0; j < width; ++j) {
                const int pos = i*width+j;
                check(POINT_RT(EventRecord)(events[2*pos], stream), "point start");
                if (mode == 2) for (int k = 0; k < stages; ++k) one(k, i % copies);
                else one(mode == 0 ? stage : j, i % copies);
                check(POINT_RT(EventRecord)(events[2*pos+1], stream), "point stop");
            }
        }
        check(POINT_RT(EventSynchronize)(events.back()), "point completion");
        for (int i = 0; i < calls*width; ++i) {
            float ms = 0;
            check(POINT_RT(EventElapsedTime)(&ms, events[2*i], events[2*i+1]), "point interval");
            if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("invalid point interval");
            values[i] = double(ms)*1000;
        }
    } catch (...) {
        POINT_RT(StreamSynchronize)(stream); cleanup(); throw;
    }
    cleanup();
}
}

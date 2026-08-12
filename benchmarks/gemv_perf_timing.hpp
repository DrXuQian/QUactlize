// Raw, per-launch device-event timing for the GEMV tactic sweep.
//
// This is intentionally benchmark-only.  Production GEMV launch semantics do
// not depend on a timer, and an averaged event interval is not an acceptable
// input to sweep_gemv_perf.py: the analyser needs the timer's raw float lattice.
//
// Protocol (kept explicit because every item has failed in an older bench):
//   * create every event before any launch;
//   * give the warmup its own pair and wait only for that stop event;
//   * give every measured launch a distinct pair;
//   * enqueue the measured batch without a per-launch synchronization;
//   * perform exactly one final device synchronization;
//   * preserve the raw float-millisecond word as well as printable microseconds.
//
// An incomplete or non-positive observation invalidates the WHOLE batch.  A
// caller must emit one `excluded` record, never rank the surviving prefix.
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <hggc_runtime.h>

namespace gemv_perf_timing {

struct RawEventSample {
  float event_ms = 0.0f;
  std::uint32_t event_ms_bits = 0;
  double event_us = 0.0;
};

struct RawEventBatch {
  bool ok = false;
  std::string error;
  std::vector<RawEventSample> samples;
  // Audit only.  This is the host interval around the measured enqueue/drain
  // batch divided by launches; it is never a candidate-selection measurement.
  double wall_us_per_launch = 0.0;
  std::size_t expected_samples = 0;

  bool complete() const {
    return ok && expected_samples > 0 && samples.size() == expected_samples;
  }
};

// The indirection is deliberate: a pure-host fake can prove the synchronization
// protocol and failure paths without pretending that a stub event is a timer.
struct HggcEventApi {
  using Event = hggcEvent_t;
  using Stream = hggcStream_t;
  using Status = hggcError_t;

  static Status create(Event* event) { return hggcEventCreate(event); }
  static Status destroy(Event event) { return hggcEventDestroy(event); }
  static Status record(Event event, Stream stream) { return hggcEventRecord(event, stream); }
  static Status synchronize_event(Event event) { return hggcEventSynchronize(event); }
  static Status synchronize_device() { return hggcDeviceSynchronize(); }
  static Status elapsed(float* ms, Event begin, Event end) {
    return hggcEventElapsedTime(ms, begin, end);
  }
  static Status launch_status() { return hggcPeekAtLastError(); }
  static bool success(Status status) { return status == hggcSuccess; }
  static std::string describe(Status status) {
    char const* name = hggcGetErrorName(status);
    char const* text = hggcGetErrorString(status);
    std::string out = name ? name : "unknown";
    if (text && *text) {
      out += ": ";
      out += text;
    }
    return out;
  }
};

namespace detail {

template <class Api>
inline bool check(typename Api::Status status, char const* operation, std::string& error) {
  if (Api::success(status)) return true;
  error = operation;
  error += " failed: ";
  error += Api::describe(status);
  return false;
}

inline std::uint32_t float_bits(float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t), "raw timer word requires binary32");
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

}  // namespace detail

// Api is a small static interface matching HggcEventApi.  It is public solely
// so the local protocol oracle can inject a deterministic fake runtime.
template <class Api = HggcEventApi, class Launch>
RawEventBatch measure_raw_launches(
    Launch&& launch,
    std::size_t measured_launches,
    typename Api::Stream stream = typename Api::Stream{}) {
  RawEventBatch out;
  out.expected_samples = measured_launches;
  if (measured_launches == 0) {
    out.error = "measured_launches must be positive";
    return out;
  }

  struct Pair {
    typename Api::Event begin{};
    typename Api::Event end{};
    bool have_begin = false;
    bool have_end = false;
  };

  // Slot zero belongs only to warmup.  Slots 1..N are measured launches.
  std::vector<Pair> pairs(measured_launches + 1);
  std::string error;
  for (Pair& pair : pairs) {
    if (!detail::check<Api>(Api::create(&pair.begin), "event-create(begin)", error)) break;
    pair.have_begin = true;
    if (!detail::check<Api>(Api::create(&pair.end), "event-create(end)", error)) break;
    pair.have_end = true;
  }

  auto destroy_all = [&]() {
    // Destruction failures invalidate a nominally successful capture too: a
    // runtime whose event lifecycle failed cannot establish a trustworthy run.
    for (Pair& pair : pairs) {
      if (pair.have_end) {
        std::string cleanup;
        if (!detail::check<Api>(Api::destroy(pair.end), "event-destroy(end)", cleanup) &&
            error.empty()) error = cleanup;
        pair.have_end = false;
      }
      if (pair.have_begin) {
        std::string cleanup;
        if (!detail::check<Api>(Api::destroy(pair.begin), "event-destroy(begin)", cleanup) &&
            error.empty()) error = cleanup;
        pair.have_begin = false;
      }
    }
  };

  if (!error.empty()) {
    destroy_all();
    out.error = error;
    return out;
  }

  auto record_launch = [&](Pair& pair) -> bool {
    if (!detail::check<Api>(Api::record(pair.begin, stream), "event-record(begin)", error))
      return false;
    launch();
    if (!detail::check<Api>(Api::record(pair.end, stream), "event-record(end)", error))
      return false;
    return detail::check<Api>(Api::launch_status(), "kernel launch", error);
  };

  // Warmup completes before host-wall auditing starts.  Waiting for one warmup
  // stop is not a measured-launch synchronization; the measured batch below
  // still has one and only one final device sync.
  bool enqueued = record_launch(pairs[0]);
  if (enqueued) {
    enqueued = detail::check<Api>(
        Api::synchronize_event(pairs[0].end), "warmup event synchronize", error);
  }

  auto wall_begin = std::chrono::steady_clock::time_point{};
  auto wall_end = wall_begin;
  if (enqueued) {
    wall_begin = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < measured_launches; ++i) {
      if (!record_launch(pairs[i + 1])) {
        enqueued = false;
        break;
      }
    }
    // Even a rejected enqueue prefix gets exactly one final drain attempt.  It
    // remains excluded regardless of whether this cleanup sync succeeds.
    std::string sync_error;
    bool const synced = detail::check<Api>(
        Api::synchronize_device(), "final device synchronize", sync_error);
    wall_end = std::chrono::steady_clock::now();
    if (!synced && error.empty()) error = sync_error;
    enqueued = enqueued && synced;
  }

  if (enqueued) {
    out.samples.reserve(measured_launches);
    for (std::size_t i = 0; i < measured_launches; ++i) {
      float event_ms = 0.0f;
      if (!detail::check<Api>(
              Api::elapsed(&event_ms, pairs[i + 1].begin, pairs[i + 1].end),
              "event-elapsed-time", error)) {
        break;
      }
      double const event_us = static_cast<double>(event_ms) * 1000.0;
      if (!std::isfinite(event_ms) || event_ms <= 0.0f ||
          !std::isfinite(event_us) || event_us <= 0.0) {
        error = "event-elapsed-time returned a non-finite or non-positive raw sample";
        break;
      }
      out.samples.push_back({event_ms, detail::float_bits(event_ms), event_us});
    }
  }

  if (wall_end != std::chrono::steady_clock::time_point{} &&
      wall_begin != std::chrono::steady_clock::time_point{}) {
    out.wall_us_per_launch =
        std::chrono::duration<double, std::micro>(wall_end - wall_begin).count() /
        static_cast<double>(measured_launches);
  }

  destroy_all();
  if (!error.empty() || out.samples.size() != measured_launches) {
    if (error.empty()) error = "raw event batch is incomplete";
    out.samples.clear();  // fail closed: never expose a rankable prefix
    out.error = error;
    return out;
  }
  out.ok = true;
  return out;
}

}  // namespace gemv_perf_timing

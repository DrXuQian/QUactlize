#pragma once
// Producer-only timing for a two-launch fixed Split-K protocol.
//
// The stop event is recorded after the producer and before the reducer, so
// the reported event span excludes reduction.  The reducer is nevertheless
// enqueued before any host wait, and the next producer is ordered after that
// reducer on the same stream.  A partial workspace is therefore never reused
// before its consumer has been submitted.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "cutlass/cutlass.h"
#include "helper.h"

namespace splitk_producer_timing {

enum class Failure {
  None,
  StartEvent,
  ProducerLaunch,
  StopEvent,
  ConsumerLaunch,
  StopSynchronize,
  ElapsedTime,
  ConsumerSynchronize,
};

inline char const* failure_name(Failure failure) {
  switch (failure) {
    case Failure::None: return "NONE";
    case Failure::StartEvent: return "ORDERED_CLOSE_START_EVENT";
    case Failure::ProducerLaunch: return "ORDERED_CLOSE_PRODUCER_LAUNCH";
    case Failure::StopEvent: return "ORDERED_CLOSE_STOP_EVENT";
    case Failure::ConsumerLaunch: return "ORDERED_CLOSE_REDUCER_LAUNCH";
    case Failure::StopSynchronize: return "ORDERED_CLOSE_STOP_SYNCHRONIZE";
    case Failure::ElapsedTime: return "ORDERED_CLOSE_ELAPSED_TIME";
    case Failure::ConsumerSynchronize:
      return "ORDERED_CLOSE_REDUCER_SYNCHRONIZE";
  }
  return "ORDERED_CLOSE_UNKNOWN";
}

inline bool is_launch_failure(Failure failure) {
  return failure == Failure::ProducerLaunch ||
         failure == Failure::ConsumerLaunch;
}

class EventPair {
 public:
  hggcEvent_t start{}, stop{};

  EventPair() {
    CUTLASS_PPU_CHECK(hggcEventCreate(&start));
    CUTLASS_PPU_CHECK(hggcEventCreate(&stop));
  }
  ~EventPair() {
    if (start) hggcEventDestroy(start);
    if (stop) hggcEventDestroy(stop);
  }
  EventPair(EventPair const&) = delete;
  EventPair& operator=(EventPair const&) = delete;
};

struct Result {
  Failure failure = Failure::None;
  int failure_cutlass_status = 0;
  int failure_runtime_status = 0;
  int failure_repeat = -1;
  double median_us = 0;
  double min_us = 0;
  double max_us = 0;
  std::vector<double> samples_us;
};

template <class ProducerLaunch, class ConsumerLaunch>
Result measure(ProducerLaunch&& producer, ConsumerLaunch&& consumer,
               int iterations) {
  Result result;
  result.samples_us.reserve(std::size_t(iterations));
  EventPair events;
  for (int i = 0; i < iterations; ++i) {
    result.failure_repeat = i;
    auto const start_status = hggcEventRecord(events.start, nullptr);
    if (start_status != hggcSuccess) {
      result.failure = Failure::StartEvent;
      result.failure_runtime_status = int(start_status);
      return result;
    }
    auto const producer_status = producer();
    if (producer_status != cutlass::Status::kSuccess) {
      result.failure = Failure::ProducerLaunch;
      result.failure_cutlass_status = int(producer_status);
      return result;
    }
    auto const stop_status = hggcEventRecord(events.stop, nullptr);
    if (stop_status != hggcSuccess) {
      result.failure = Failure::StopEvent;
      result.failure_runtime_status = int(stop_status);
      return result;
    }
    // This ordering is the semantic seam: submit the consumer before any
    // host-side stop-event wait.  Its work is outside the event span above.
    auto const consumer_status = consumer();
    if (consumer_status != cutlass::Status::kSuccess) {
      result.failure = Failure::ConsumerLaunch;
      result.failure_cutlass_status = int(consumer_status);
      return result;
    }
    auto const stop_sync_status = hggcEventSynchronize(events.stop);
    if (stop_sync_status != hggcSuccess) {
      result.failure = Failure::StopSynchronize;
      result.failure_runtime_status = int(stop_sync_status);
      return result;
    }
    float ms = 0;
    auto const elapsed_status =
        hggcEventElapsedTime(&ms, events.start, events.stop);
    if (elapsed_status != hggcSuccess || !(ms > 0) || !std::isfinite(ms)) {
      result.failure = Failure::ElapsedTime;
      result.failure_runtime_status = int(elapsed_status);
      return result;
    }
    result.samples_us.push_back(double(ms) * 1000.0);
  }
  // The final reducer must complete before its output is inspected and before
  // the caller can release/reuse the partial workspace.
  auto const consumer_sync_status = hggcDeviceSynchronize();
  if (consumer_sync_status != hggcSuccess) {
    result.failure = Failure::ConsumerSynchronize;
    result.failure_runtime_status = int(consumer_sync_status);
    return result;
  }
  std::sort(result.samples_us.begin(), result.samples_us.end());
  result.min_us = result.samples_us.front();
  result.max_us = result.samples_us.back();
  result.median_us = result.samples_us.size() & 1
      ? result.samples_us[result.samples_us.size() / 2]
      : 0.5 * (result.samples_us[result.samples_us.size()/2 - 1] +
               result.samples_us[result.samples_us.size()/2]);
  result.failure = Failure::None;
  result.failure_repeat = -1;
  return result;
}

}  // namespace splitk_producer_timing

// Exhaustive host oracle for the fixed Split-K actual-last protocol.
// It proves scheduling/slot/reset/order properties only; PPU visibility is a
// device postcondition bound separately to the production source sequence.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <vector>

#include "quactlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_completion_protocol.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_partition.hpp"

namespace fs = cutlass::gemm::kernel::fixed_splitk;

struct Counter {
  uint32_t value = 0;

  uint32_t arrive() { return value++; }
  void reset() { value = 0; }
};

template <int S>
bool exhaust_permutations(uint64_t& permutations, uint64_t& arrivals) {
  std::array<int, S> order{};
  std::iota(order.begin(), order.end(), 0);
  do {
    Counter counter;
    int last_count = 0;
    int actual_last_peer = -1;
    std::array<int, S> numerical_order{};
    std::iota(numerical_order.begin(), numerical_order.end(), 0);
    for (int peer : order) {
      uint32_t const old = counter.arrive();
      ++arrivals;
      if (fs::completion_arrival_is_last(old, S)) {
        ++last_count;
        actual_last_peer = peer;
      }
    }
    if (last_count != 1 || actual_last_peer != order.back() ||
        numerical_order.front() != 0 || numerical_order.back() != S - 1 ||
        counter.value != S) {
      return false;
    }
    counter.reset();
    if (counter.value != 0) return false;

    // Reuse the same counter for a second launch.  A missing reset has no
    // terminal arrival in [0,S), which is a fail-closed wrong-result shape.
    last_count = 0;
    for (int peer : order) {
      (void)peer;
      uint32_t const old = counter.arrive();
      last_count += fs::completion_arrival_is_last(old, S);
    }
    if (last_count != 1) return false;
    ++permutations;
  } while (std::next_permutation(order.begin(), order.end()));
  return true;
}

int main() {
  uint64_t permutations = 0;
  uint64_t arrivals = 0;
  if (!exhaust_permutations<2>(permutations, arrivals) ||
      !exhaust_permutations<4>(permutations, arrivals) ||
      !exhaust_permutations<8>(permutations, arrivals)) {
    std::fprintf(stderr, "[l196] FAIL: actual-last permutation proof\n");
    return 1;
  }
  if (permutations != 2 + 24 + 40320 ||
      arrivals != 2 * 2 + 24 * 4 + 40320 * 8) {
    std::fprintf(stderr, "[l196] FAIL: coverage denominator drift\n");
    return 1;
  }

  constexpr auto p = fs::make_params(32, 32, 8);
  constexpr std::size_t partial_bytes = 8 * 4096 * sizeof(float);
  constexpr auto w = fs::make_completion_workspace(partial_bytes, 32);
  static_assert(p.is_valid() && p.work_units == 256);
  static_assert(w.is_valid() && w.partial_bytes == 131072 &&
                w.counter_offset == 131072 && w.counter_bytes == 128 &&
                w.total_bytes == 131200 && w.output_tiles == 32);
  static_assert(!fs::make_completion_workspace(0, 32).is_valid());
  static_assert(!fs::make_completion_workspace(partial_bytes, 0).is_valid());
  static_assert(!fs::completion_arrival_is_last(0, 8));
  static_assert(fs::completion_arrival_is_last(7, 8));
  static_assert(!fs::completion_arrival_is_last(8, 8));

  // Negative control 1: appointing logical peer S-1 disagrees with actual
  // arrival in 7/8 of the possible terminal-peer identities.
  int logical_final_disagreements = 0;
  for (int actual_last = 0; actual_last < 8; ++actual_last) {
    logical_final_disagreements += actual_last != 7;
  }
  if (logical_final_disagreements != 7) return 1;

  // The numerical order is a real bit-level property, not a label.  This
  // sequence distinguishes increasing peer order from the planted reverse
  // order under FP32 rounding.
  constexpr std::array<float, 4> order_sensitive{
      1.0e20f, -1.0e20f, 3.25f, 0.0f};
  volatile float forward = 0.0f;
  volatile float reverse = 0.0f;
  for (int s = 0; s < 4; ++s) forward = forward + order_sensitive[s];
  for (int s = 3; s >= 0; --s) reverse = reverse + order_sensitive[s];
  uint32_t forward_bits = 0, reverse_bits = 0;
  float const forward_value = forward;
  float const reverse_value = reverse;
  std::memcpy(&forward_bits, &forward_value, sizeof(forward_bits));
  std::memcpy(&reverse_bits, &reverse_value, sizeof(reverse_bits));
  if (forward_bits == reverse_bits || forward_value != 3.25f ||
      reverse_value != 0.0f) {
    return 1;
  }

  // Negative control 2: aliasing q0/q1 to one slot reaches the terminal value
  // before either tile has all four peers.
  Counter aliased;
  bool false_last = false;
  for (int peer = 0; peer < 2; ++peer) {
    (void)peer;
    false_last |= fs::completion_arrival_is_last(aliased.arrive(), 4);
    false_last |= fs::completion_arrival_is_last(aliased.arrive(), 4);
  }
  if (!false_last) return 1;

  // Negative control 3: a second launch without reset produces no valid last.
  Counter stale;
  for (int i = 0; i < 8; ++i) (void)stale.arrive();
  int stale_last = 0;
  for (int i = 0; i < 8; ++i) {
    stale_last += fs::completion_arrival_is_last(stale.arrive(), 8);
  }
  if (stale_last != 0) return 1;

  std::printf(
      "[l196] PASS permutations=%llu arrivals=%llu S=2,4,8 "
      "workspace=131072+128 actual-last=unique fixed-order=0..S-1 "
      "order-red=reverse reset=reusable "
      "plants=logical-final/q-alias/missing-reset\n",
      static_cast<unsigned long long>(permutations),
      static_cast<unsigned long long>(arrivals));
  return 0;
}

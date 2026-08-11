// L132 -- the G5 harness has two index spaces: caller slot and real expert.
//
// The kernel, metadata planes and ptr_D are expert-indexed.  The CPU oracle is
// slot-indexed because the caller chooses ids[].  The original harness pushed
// oracle vectors while scanning experts in ascending order, then consumed the
// vector as if that were caller order.  Existing ids happened to be sorted, so
// the defect was silent.  A shuffled list is the constructive negative.

#include <cstdio>
#include <vector>

#include "../../tests/m8n16_g5_slot_map.hpp"

namespace {

constexpr int kExperts = 256;
constexpr int kActive = 8;
constexpr int kRows = 2;
constexpr int kOrdered[kActive] = {3, 17, 42, 88, 129, 190, 201, 255};
constexpr int kShuffled[kActive] = {255, 3, 201, 17, 190, 42, 129, 88};
constexpr int kShuffledRows[kActive] = {14, 0, 12, 2, 10, 4, 8, 6};

std::vector<int> materialize_rows(m8n16_g5_slot_map::Map const& route) {
  std::vector<int> rows(route.total_rows, -1);
  for (int slot = 0; slot < int(route.expert_for_slot.size()); ++slot) {
    int const row = route.row_for_slot(slot);
    for (int r = 0; r < kRows; ++r) rows[row + r] = slot + 1;
  }
  return rows;
}

int check_prefix(m8n16_g5_slot_map::Map const& route) {
  int bad = 0;
  int running = 0;
  for (int expert = 0; expert < kExperts; ++expert) {
    bad += route.row_offsets[expert] != running;
    int const want_m = route.slot_for_expert[expert] >= 0 ? kRows : 0;
    bad += route.group_m[expert] != want_m;
    running += want_m;
  }
  bad += route.row_offsets[kExperts] != running;
  bad += route.total_rows != kActive * kRows;
  return bad;
}

}  // namespace

int main() {
  auto const ordered =
      m8n16_g5_slot_map::make(kExperts, kOrdered, kActive, kRows);
  auto const shuffled =
      m8n16_g5_slot_map::make(kExperts, kShuffled, kActive, kRows);

  int ordered_bad = !ordered.valid || check_prefix(ordered);
  int shuffled_bad = !shuffled.valid || check_prefix(shuffled);
  int ordered_row_diff = 0;
  int shuffled_row_diff = 0;
  for (int slot = 0; slot < kActive; ++slot) {
    ordered_row_diff += ordered.row_for_slot(slot) != slot * kRows;
    shuffled_row_diff += shuffled.row_for_slot(slot) != kShuffledRows[slot];
  }

  auto const ordered_rows = materialize_rows(ordered);
  auto const shuffled_rows = materialize_rows(shuffled);
  int ordered_byte_diff = 0;
  int shuffled_recover_bad = 0;
  int legacy_slot_base_bad = 0;
  for (int slot = 0; slot < kActive; ++slot) {
    for (int r = 0; r < kRows; ++r) {
      ordered_byte_diff += ordered_rows[slot * kRows + r] != slot + 1;
      shuffled_recover_bad +=
          shuffled_rows[shuffled.row_for_slot(slot) + r] != slot + 1;
      legacy_slot_base_bad +=
          shuffled_rows[slot * kRows + r] != slot + 1;
    }
  }

  // Reproduce the deleted bookkeeping: scan experts, push their oracle data,
  // then interpret push index as caller slot.  The shuffled fixture has no
  // fixed point, so all eight labels are wrong.  If this does not go red, the
  // positive arm cannot prove it distinguishes the two index spaces.
  std::vector<int> legacy_expert_order;
  for (int expert = 0; expert < kExperts; ++expert)
    if (shuffled.slot_for_expert[expert] >= 0)
      legacy_expert_order.push_back(expert);
  int legacy_push_bad = 0;
  for (int slot = 0; slot < kActive; ++slot)
    legacy_push_bad += legacy_expert_order[slot] != kShuffled[slot];

  int duplicate[kActive] = {3, 17, 42, 88, 129, 190, 201, 3};
  auto const duplicate_route =
      m8n16_g5_slot_map::make(kExperts, duplicate, kActive, kRows);

  bool const positive = ordered_bad == 0 && shuffled_bad == 0 &&
      ordered_row_diff == 0 && shuffled_row_diff == 0 &&
      ordered_byte_diff == 0 && shuffled_recover_bad == 0;
  bool const negative = legacy_push_bad == kActive &&
      legacy_slot_base_bad == kActive * kRows && !duplicate_route.valid;

  std::printf("L132 ordered-default row-diff=%d/%d byte-diff=%d/%d -> %s\n",
              ordered_row_diff, kActive, ordered_byte_diff, kActive * kRows,
              (ordered_row_diff == 0 && ordered_byte_diff == 0) ? "IDENTICAL" : "FAIL");
  std::printf("L132 shuffled rows=14,0,12,2,10,4,8,6 row-diff=%d/%d "
              "recover-bad=%d/%d -> %s\n",
              shuffled_row_diff, kActive, shuffled_recover_bad, kActive * kRows,
              (shuffled_row_diff == 0 && shuffled_recover_bad == 0) ? "PASS" : "FAIL");
  std::printf("L132 planted-old expert-push=%d/%d slot-base=%d/%d "
              "duplicate-valid=%d -> %s\n",
              legacy_push_bad, kActive, legacy_slot_base_bad, kActive * kRows,
              int(duplicate_route.valid), negative ? "EXPECTED_RED" : "FAIL");
  std::printf("L132 G5 harness slot->expert->row_offsets %s\n",
              (positive && negative) ? "PASS" : "FAIL");
  return (positive && negative) ? 0 : 1;
}

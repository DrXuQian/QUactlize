#pragma once

// Host-side ownership map for #112/G5.
//
// The grouped kernel names data by real expert id, while the test driver names
// expected results by caller slot.  Those are different index spaces: ids[] is
// not required to be sorted.  Keep the conversion explicit so a shuffled
// active list cannot silently make the CPU oracle describe a different expert
// from the row written by ptr_D[expert].

#include <vector>

namespace m8n16_g5_slot_map {

struct Map {
  bool valid = false;
  std::vector<int> expert_for_slot;
  std::vector<int> slot_for_expert;
  std::vector<int> group_m;
  std::vector<int> row_offsets;
  int total_rows = 0;

  int row_for_slot(int slot) const {
    return row_offsets[expert_for_slot[slot]];
  }
};

inline Map make(int experts, int const* ids, int active, int rows_per_expert) {
  Map out;
  if (experts <= 0 || active < 0 || active > experts ||
      rows_per_expert <= 0 || (active > 0 && ids == nullptr)) {
    return out;
  }

  if (active > 0) out.expert_for_slot.assign(ids, ids + active);
  out.slot_for_expert.assign(experts, -1);
  out.group_m.assign(experts, 0);
  out.row_offsets.assign(experts + 1, 0);

  for (int slot = 0; slot < active; ++slot) {
    int const expert = out.expert_for_slot[slot];
    if (expert < 0 || expert >= experts || out.slot_for_expert[expert] != -1) {
      return out;  // out-of-range or duplicate ids make ownership ambiguous
    }
    out.slot_for_expert[expert] = slot;
    out.group_m[expert] = rows_per_expert;
  }
  for (int expert = 0; expert < experts; ++expert) {
    out.row_offsets[expert + 1] =
        out.row_offsets[expert] + out.group_m[expert];
  }
  out.total_rows = out.row_offsets[experts];
  out.valid = true;
  return out;
}

}  // namespace m8n16_g5_slot_map

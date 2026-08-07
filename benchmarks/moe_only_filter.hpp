#pragma once

// The two MOE_ONLY gates share these strings. Keep them dependency-free so the complete shape->row selection
// path can be compiled and exercised on a host with no PPU SDK.
#include <cstddef>
#include <cstdio>
#include <cstring>
#include "bench_select.hpp"

namespace moe_only_filter {

inline void format_shape(char* out, std::size_t cap, const char* name,
                         int tm, int tn, int tk, int wm, int wn) {
  // A shape is deliberately only the fields shared by all rows packed together. Stage and PPU_B_CHUNK are row/unit
  // fields. Appending bc here broke the bidirectional substring contract: a stage-bearing filter had `sN` where the
  // shape had `bcN->N`, so neither string contained the other and the row gate was never reached.
  std::snprintf(out, cap, "%s %dx%d:%d w%dx%d", name, tm, tn, tk, wm, wn);
}

inline void format_tag(char* out, std::size_t cap, const char* name,
                       int tm, int tn, int tk, int wm, int wn, int stages,
                       int bc_requested, int bc_effective, bool abcast) {
  bench_measure::format_tag(out, cap,
      bench_measure::Tactic{name, tm, tn, tk, wm, wn, stages,
                            bc_requested, bc_effective, abcast});
}

inline bool row_selected(const char* tag, const char* filter) {
  return !filter || std::strstr(tag, filter) != nullptr;
}

inline bool shape_selected(const char* shape, const char* filter) {
  return !filter || std::strstr(shape, filter) != nullptr || std::strstr(filter, shape) != nullptr;
}

inline bool candidate_selected(const char* shape, const char* tag, const char* filter) {
  return shape_selected(shape, filter) && row_selected(tag, filter);
}

}  // namespace moe_only_filter

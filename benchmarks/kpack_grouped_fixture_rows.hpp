#pragma once

// Host-only loader for exact grouped discovery row histograms.  Production
// kernels still receive the ordinary rows/offsets arrays; this helper merely
// prevents two router controls with the same aggregate shape from collapsing
// into one synthetic token/top-k fixture.

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace kpack_grouped_fixture_rows {

struct Rows {
  std::vector<int> per_expert;
  int total = 0;
  int max = 0;
  int active = 0;
  int zero = 0;
  std::uint64_t fnv64 = UINT64_C(14695981039346656037);
};

inline void set_why(char* why, std::size_t capacity, char const* text) {
  if (!why || capacity == 0) return;
  std::size_t const count = std::min(capacity - 1, std::char_traits<char>::length(text));
  std::copy_n(text, count, why);
  why[count] = '\0';
}

inline std::uint64_t rows_fnv64(std::vector<int> const& rows) {
  std::uint64_t value = UINT64_C(14695981039346656037);
  for (int row : rows) {
    auto const bits = static_cast<std::uint32_t>(row);
    for (int shift = 0; shift != 32; shift += 8) {
      value ^= std::uint8_t(bits >> shift);
      value *= UINT64_C(1099511628211);
    }
  }
  return value;
}

inline bool summarize(std::vector<int> rows, int experts, Rows& out,
                      char* why = nullptr, std::size_t why_capacity = 0) {
  if (experts <= 0 || rows.size() != std::size_t(experts)) {
    set_why(why, why_capacity, "row count differs from experts");
    return false;
  }
  std::int64_t total = 0;
  int maximum = 0, active = 0;
  for (int value : rows) {
    if (value < 0) {
      set_why(why, why_capacity, "row count is negative");
      return false;
    }
    total += value;
    if (total > std::numeric_limits<int>::max()) {
      set_why(why, why_capacity, "row total exceeds int");
      return false;
    }
    maximum = std::max(maximum, value);
    active += value != 0;
  }
  if (total <= 0 || active <= 0) {
    set_why(why, why_capacity, "row histogram has no work");
    return false;
  }
  out.per_expert = std::move(rows);
  out.total = int(total);
  out.max = maximum;
  out.active = active;
  out.zero = experts - active;
  out.fnv64 = rows_fnv64(out.per_expert);
  return true;
}

inline bool load(char const* path, int experts, Rows& out,
                 char* why = nullptr, std::size_t why_capacity = 0) {
  if (!path || !path[0]) {
    set_why(why, why_capacity, "row file path is empty");
    return false;
  }
  std::ifstream stream(path);
  if (!stream) {
    set_why(why, why_capacity, "cannot open row file");
    return false;
  }
  std::vector<int> rows;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) {
      set_why(why, why_capacity, "row file contains an empty line");
      return false;
    }
    errno = 0;
    char* end = nullptr;
    long const value = std::strtol(line.c_str(), &end, 10);
    if (errno || !end || *end || value < 0 ||
        value > std::numeric_limits<int>::max()) {
      set_why(why, why_capacity, "row file contains an invalid integer");
      return false;
    }
    rows.push_back(int(value));
    if (rows.size() > std::size_t(experts)) {
      set_why(why, why_capacity, "row file has too many entries");
      return false;
    }
  }
  if (!stream.eof()) {
    set_why(why, why_capacity, "row file read failed");
    return false;
  }
  return summarize(std::move(rows), experts, out, why, why_capacity);
}

}  // namespace kpack_grouped_fixture_rows

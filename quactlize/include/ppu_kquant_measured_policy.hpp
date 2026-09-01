// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Host-readable selector for K-quant tactics observed by the committed measurement
// campaign. The generated table compresses equal adjacent dense winners, but
// this layer admits only exact measured M/N/K values. Grouped routing remains
// on its compiled default until the public API carries the full expert-row
// distribution measured by its sweep.
#pragma once

#include <cstddef>

#include "ppu_dense_shipping_policy.hpp"
#include "ppu_kquant_measured_policy_data.inc"

namespace ppu_kquant_measured_policy {

using DenseConfigId = ppu_dense_shipping::ConfigId;

struct DenseInterval {
  int dynamic_min;
  int dynamic_max;
  DenseConfigId config;
};

struct DenseFamily {
  int qtype;
  int n;
  int k;
  int first_interval;
  int interval_count;
};

inline constexpr int kMeasuredDynamicValues[] = {
#define QUACTLIZE_PPU_KQUANT_MEASURED_DYNAMIC_ROW(VALUE) VALUE,
  QUACTLIZE_PPU_KQUANT_MEASURED_DYNAMIC_VALUES(
      QUACTLIZE_PPU_KQUANT_MEASURED_DYNAMIC_ROW)
#undef QUACTLIZE_PPU_KQUANT_MEASURED_DYNAMIC_ROW
};

inline constexpr DenseFamily kDenseFamilies[] = {
#define QUACTLIZE_PPU_KQUANT_DENSE_FAMILY_ROW(QTYPE, N, K, FIRST, COUNT) \
  {QTYPE, N, K, FIRST, COUNT},
  QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_FAMILIES(
      QUACTLIZE_PPU_KQUANT_DENSE_FAMILY_ROW)
#undef QUACTLIZE_PPU_KQUANT_DENSE_FAMILY_ROW
};

inline constexpr DenseInterval kDenseIntervals[] = {
#define QUACTLIZE_PPU_KQUANT_DENSE_INTERVAL_ROW(MIN, MAX, ID) \
  {MIN, MAX, DenseConfigId::ID},
  QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_INTERVALS(
      QUACTLIZE_PPU_KQUANT_DENSE_INTERVAL_ROW)
#undef QUACTLIZE_PPU_KQUANT_DENSE_INTERVAL_ROW
};

static_assert(
    sizeof(kMeasuredDynamicValues) / sizeof(kMeasuredDynamicValues[0]) ==
        QUACTLIZE_PPU_KQUANT_MEASURED_DYNAMIC_VALUE_COUNT);
static_assert(sizeof(kDenseFamilies) / sizeof(kDenseFamilies[0]) ==
              QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_FAMILY_COUNT);
static_assert(sizeof(kDenseIntervals) / sizeof(kDenseIntervals[0]) ==
              QUACTLIZE_PPU_KQUANT_DENSE_MEASURED_INTERVAL_COUNT);
constexpr bool measured_dynamic_value(int value) {
  for (int measured : kMeasuredDynamicValues) {
    if (value == measured) return true;
  }
  return false;
}

constexpr bool select_dense(
    int qtype, int m, int n, int k, DenseConfigId& result) {
  if (!measured_dynamic_value(m)) return false;
  for (auto const& family : kDenseFamilies) {
    if (family.qtype != qtype || family.n != n || family.k != k) continue;
    int const end = family.first_interval + family.interval_count;
    for (int i = family.first_interval; i < end; ++i) {
      auto const& interval = kDenseIntervals[static_cast<std::size_t>(i)];
      if (m >= interval.dynamic_min && m <= interval.dynamic_max) {
        result = interval.config;
        return true;
      }
    }
    return false;
  }
  return false;
}

// Deployment precedence is explicit compiled name, exact measured point, then
// the existing shipping default.  An unknown non-empty explicit name remains a
// miss and is never reinterpreted as either a measured or default selection.
constexpr bool find_dense_config(
    char const* name, int qtype, int m, int n, int k,
    DenseConfigId& result) {
  if (name && name[0]) {
    return ppu_dense_shipping::find_config(name, m, result);
  }
  if (select_dense(qtype, m, n, k, result)) return true;
  return ppu_dense_shipping::find_config(nullptr, m, result);
}

constexpr bool interval_tables_are_valid() {
  constexpr std::size_t dynamic_count =
      sizeof(kMeasuredDynamicValues) / sizeof(kMeasuredDynamicValues[0]);
  constexpr std::size_t family_count =
      sizeof(kDenseFamilies) / sizeof(kDenseFamilies[0]);
  constexpr std::size_t interval_count =
      sizeof(kDenseIntervals) / sizeof(kDenseIntervals[0]);
  for (std::size_t i = 0; i < dynamic_count; ++i) {
    if (kMeasuredDynamicValues[i] <= 0 ||
        (i != 0 &&
         kMeasuredDynamicValues[i - 1] >= kMeasuredDynamicValues[i])) {
      return false;
    }
  }
  std::size_t next_interval = 0;
  for (std::size_t family_index = 0; family_index < family_count;
       ++family_index) {
    auto const& family = kDenseFamilies[family_index];
    if (family.first_interval < 0 || family.interval_count <= 0 ||
        static_cast<std::size_t>(family.first_interval) != next_interval ||
        static_cast<std::size_t>(family.first_interval) > interval_count ||
        static_cast<std::size_t>(family.interval_count) >
            interval_count -
                static_cast<std::size_t>(family.first_interval)) {
      return false;
    }
    if (family_index != 0) {
      auto const& previous = kDenseFamilies[family_index - 1];
      if (previous.qtype > family.qtype ||
          (previous.qtype == family.qtype && previous.n > family.n) ||
          (previous.qtype == family.qtype && previous.n == family.n &&
           previous.k >= family.k)) {
        return false;
      }
    }
    int const end = family.first_interval + family.interval_count;
    for (int i = family.first_interval; i < end; ++i) {
      auto const& interval = kDenseIntervals[static_cast<std::size_t>(i)];
      if (interval.dynamic_min > interval.dynamic_max ||
          !measured_dynamic_value(interval.dynamic_min) ||
          !measured_dynamic_value(interval.dynamic_max) ||
          static_cast<int>(interval.config) < 0 ||
          static_cast<int>(interval.config) >=
              static_cast<int>(DenseConfigId::Count) ||
          (i != family.first_interval &&
           kDenseIntervals[static_cast<std::size_t>(i - 1)].dynamic_max >=
               interval.dynamic_min)) {
        return false;
      }
    }
    for (int measured : kMeasuredDynamicValues) {
      int matches = 0;
      for (int i = family.first_interval; i < end; ++i) {
        auto const& interval = kDenseIntervals[static_cast<std::size_t>(i)];
        matches += measured >= interval.dynamic_min &&
                   measured <= interval.dynamic_max;
      }
      if (matches != 1) return false;
    }
    next_interval += static_cast<std::size_t>(family.interval_count);
  }
  return next_interval == interval_count;
}

static_assert(interval_tables_are_valid(),
              "generated K-quant measured policy tables must be ordered, "
              "bounded, exhaustive over exact measured points and reference "
              "only compiled dense configs");

}  // namespace ppu_kquant_measured_policy

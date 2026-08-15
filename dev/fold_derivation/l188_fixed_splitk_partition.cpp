#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "quactlize_extensions/cutlass/gemm/kernel/ppu_fixed_splitk_partition.hpp"

namespace fs = cutlass::gemm::kernel::fixed_splitk;

namespace {

enum class Plant {
  None,
  KOverlap,
  LogicalWorkIdAlias,
  DropFinalWork,
  PeerCountOffByOne,
};

struct Census {
  uint64_t descriptors = 0;
  uint64_t qk_cells = 0;
  uint64_t logical_work_ids = 0;
  uint64_t errors = 0;
};

fs::FixedSplitKWork apply_plant(fs::FixedSplitKWork work, Plant plant) {
  switch (plant) {
    case Plant::None:
      break;
    case Plant::KOverlap:
      if (work.peer_idx != 0) {
        --work.k_begin;
      }
      break;
    case Plant::LogicalWorkIdAlias:
      work.logical_work_id = work.q;
      break;
    case Plant::DropFinalWork:
      break;
    case Plant::PeerCountOffByOne:
      if (work.peer_count > 1) {
        --work.peer_count;
      }
      break;
  }
  return work;
}

Census validate(fs::Params const& params, Plant plant) {
  Census out;
  if (!params.is_valid()) {
    ++out.errors;
    return out;
  }
  if (params.output_tiles >
      std::numeric_limits<size_t>::max() / params.k_tiles_per_output) {
    ++out.errors;
    return out;
  }

  size_t const qk_size =
      size_t(params.output_tiles) * size_t(params.k_tiles_per_output);
  std::vector<uint8_t> qk_visits(qk_size, uint8_t(0));
  std::vector<uint8_t> slot_visits(size_t(params.work_units), uint8_t(0));
  std::vector<uint8_t> peer_visits(size_t(params.work_units), uint8_t(0));
  std::vector<uint32_t> arrival_count(size_t(params.output_tiles), 0u);

  uint64_t work_limit = params.work_units;
  if (plant == Plant::DropFinalWork && work_limit != 0) {
    --work_limit;
  }

  for (uint64_t linear = 0; linear < work_limit; ++linear) {
    fs::FixedSplitKWork work =
        apply_plant(fs::work_for_linear(params, linear), plant);
    ++out.descriptors;
    if (!fs::work_matches_params(params, work)) {
      ++out.errors;
    }
    if (!work.is_valid() || work.q >= params.output_tiles) {
      continue;
    }

    uint64_t const peer_slot =
        work.q * uint64_t(params.splits) + work.peer_idx;
    if (peer_slot >= peer_visits.size() || peer_visits[size_t(peer_slot)]++) {
      ++out.errors;
    }

    if (work.logical_work_id >= slot_visits.size() ||
        slot_visits[size_t(work.logical_work_id)]++) {
      ++out.errors;
    }

    if (work.completion_slot() >= arrival_count.size()) {
      ++out.errors;
    }
    else {
      uint32_t const arrivals = ++arrival_count[size_t(work.completion_slot())];
      if (arrivals > work.peer_count) {
        ++out.errors;
      }
    }

    uint64_t const k_end = uint64_t(work.k_begin) + work.k_count;
    if (k_end > params.k_tiles_per_output) {
      ++out.errors;
      continue;
    }
    for (uint32_t k = work.k_begin; k < k_end; ++k) {
      size_t const idx = size_t(work.q) * params.k_tiles_per_output + k;
      if (qk_visits[idx] == std::numeric_limits<uint8_t>::max()) {
        ++out.errors;
      }
      else {
        ++qk_visits[idx];
      }
    }
  }

  out.qk_cells = qk_visits.size();
  out.logical_work_ids = slot_visits.size();
  out.errors += uint64_t(std::count_if(
      qk_visits.begin(), qk_visits.end(), [](uint8_t n) { return n != 1; }));
  out.errors += uint64_t(std::count_if(
      slot_visits.begin(), slot_visits.end(), [](uint8_t n) { return n != 1; }));
  out.errors += uint64_t(std::count_if(
      peer_visits.begin(), peer_visits.end(), [](uint8_t n) { return n != 1; }));
  for (uint32_t arrivals : arrival_count) {
    if (arrivals != params.splits) {
      ++out.errors;
    }
  }
  return out;
}

bool fail_close_controls() {
  bool ok = true;
  auto reject = [&](uint64_t q, uint32_t kt, uint32_t splits) {
    bool const red = !fs::make_params(q, kt, splits).is_valid();
    if (!red) {
      std::printf("[l188][fail-close] MISSED Q=%llu Kt=%u S=%u\n",
                  static_cast<unsigned long long>(q), kt, splits);
      ok = false;
    }
  };

  reject(0, 32, 8);
  reject(32, 0, 8);
  reject(32, 32, 0);
  reject(32, 32, 3);
  reject(32, 32, 16);
  reject(32, 30, 8);
  reject(std::numeric_limits<uint64_t>::max() / 8 + 1, 32, 8);

  fs::Params const valid = fs::make_params(32, 32, 8);
  fs::Params bad_units = valid;
  --bad_units.work_units;
  fs::Params bad_k_quantum = valid;
  ++bad_k_quantum.k_tiles_per_split;
  ok = ok && valid.is_valid() &&
       !fs::work_for_linear(valid, valid.work_units).is_valid() &&
       !bad_units.is_valid() && !bad_k_quantum.is_valid();
  std::printf("[l188][fail-close] %s controls=10\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool planted_controls() {
  fs::Params const params = fs::make_params(7, 32, 8);
  struct Control {
    Plant plant;
    char const* name;
  };
  std::array<Control, 4> const controls{{
      {Plant::KOverlap, "K overlap plus final hole"},
      {Plant::LogicalWorkIdAlias, "logical-work-id alias"},
      {Plant::DropFinalWork, "missing final work unit"},
      {Plant::PeerCountOffByOne, "peer-count ABI drift"},
  }};

  bool ok = true;
  for (Control const& control : controls) {
    Census const planted = validate(params, control.plant);
    bool const red = planted.errors != 0;
    std::printf("[l188][plant] %-24s %s errors=%llu\n",
                control.name, red ? "EXPECTED_RED" : "FALSE_GREEN",
                static_cast<unsigned long long>(planted.errors));
    ok = ok && red;
  }
  return ok;
}

}  // namespace

int main() {
  constexpr std::array<uint32_t, 4> splits{{1, 2, 4, 8}};
  uint64_t cases = 0;
  uint64_t descriptors = 0;
  uint64_t qk_cells = 0;
  uint64_t logical_work_ids = 0;
  uint64_t errors = 0;

  // Exhaust every Q/Kt/S point in this finite domain.  Invalid indivisible
  // points must fail closed; every admitted point receives a full cell-level
  // visit census rather than representative sampling.
  for (uint64_t q = 1; q <= 37; ++q) {
    for (uint32_t kt = 1; kt <= 128; ++kt) {
      for (uint32_t s : splits) {
        fs::Params const params = fs::make_params(q, kt, s);
        if (kt % s != 0) {
          if (params.is_valid()) {
            ++errors;
          }
          continue;
        }
        ++cases;
        Census const census = validate(params, Plant::None);
        descriptors += census.descriptors;
        qk_cells += census.qk_cells;
        logical_work_ids += census.logical_work_ids;
        errors += census.errors;
      }
    }
  }

  // Bind the motivating decode decomposition explicitly: 32 N tiles, 32 K
  // tiles and S=8 must become 256 unique partials of four contiguous K tiles.
  fs::Params const target = fs::make_params(32, 32, 8);
  Census const target_census = validate(target, Plant::None);
  bool const target_ok = target.is_valid() && target.work_units == 256 &&
                         target.k_tiles_per_split == 4 &&
                         target_census.errors == 0;
  std::printf(
      "[l188][target] Q=32 Kt=32 S=8 units=%llu k/peer=%u %s\n",
      static_cast<unsigned long long>(target.work_units),
      target.k_tiles_per_split, target_ok ? "PASS" : "FAIL");

  bool const fail_close_ok = fail_close_controls();
  bool const plants_ok = planted_controls();
  bool const pass = errors == 0 && target_ok && fail_close_ok && plants_ok;
  std::printf(
      "[l188] %s exhaustive_cases=%llu descriptors=%llu qk_cells=%llu "
      "logical_work_ids=%llu split_set=1,2,4,8 plants=4\n",
      pass ? "PASS" : "FAIL",
      static_cast<unsigned long long>(cases),
      static_cast<unsigned long long>(descriptors),
      static_cast<unsigned long long>(qk_cells),
      static_cast<unsigned long long>(logical_work_ids));
  return pass ? 0 : 1;
}

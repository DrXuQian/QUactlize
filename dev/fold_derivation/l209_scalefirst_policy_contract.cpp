#include "scalefirst_persistent_policy.hpp"

#include <cstdint>
#include <cstdio>

int main() {
  auto const grids = quactlize::scalefirst_policy::grid_space(2048, 72, 8);
  bool capacity_576_b8 = false;
  bool balanced_512_b8 = false;
  for (auto const& grid : grids) {
    capacity_576_b8 |= grid.grid == 576 && (grid.capacity_b_mask & (std::uint64_t(1) << 8));
    balanced_512_b8 |= grid.grid == 512 && (grid.balanced_b_mask & (std::uint64_t(1) << 8));
  }
  if (!capacity_576_b8 || !balanced_512_b8) return 2;

  char record[1024];
  int const size = quactlize::scalefirst_policy::format_q8_policy_cell(
      record, sizeof record, 64, 8192, 2048, "persistent", "balanced", 512,
      "32x64x64_w32x32_s3_bc0", 0, 17.25, 23.5, 38.25, 8192, 72, 8,
      0, std::uint64_t(1) << 8, 2501);
  if (size <= 0 || std::size_t(size) >= sizeof record) return 3;
  std::fputs(record, stdout);
  std::printf("Q8_GRID_WITNESS capacity=576 balanced=512 Q=2048 CU=72 b=8\n");
  return 0;
}

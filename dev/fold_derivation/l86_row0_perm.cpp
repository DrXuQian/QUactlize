// L86 -- THE FIRST-R-ROW PERMUTATION, exported for the packed-cube writer. Each logical row owns four physical
// 8-word runs. Within each 4-word half-run logical k stays ordered, while odd cache lines swap the two half-runs.
// The writer must account for that swap when it expands beyond row 0.
#include <cstdio>
#include <map>

static int phys_word(int lane, int v, int slice_idx, int CUBE_H, int* out_row) {
  int slice_word_base   = CUBE_H * 8 * slice_idx;
  int slice_start_vec   = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;
  int lane_row_idx      = lane / 4;  // coord_h = 0
  int lane_col_idx      = lane % 4;
  int vreg_row_idx      = (v / 2) * 8 + lane_row_idx;
  int vreg_line_idx     = vreg_row_idx / 4;
  int vreg_vec_idx      = (vreg_row_idx % 4) * 2 + (v % 2);
  int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
  int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;
  *out_row = vreg_row_idx;
  return slice_word_base + vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
}

int main() {
  constexpr int CUBE_H = 16;
  constexpr int SLICES = 4;
  bool ok = true;

  for (int row_wanted = 0; row_wanted < CUBE_H; ++row_wanted) {
    std::map<int, int> logical_to_phys;
    for (int lane = 0; lane < 32; ++lane)
      for (int v = 0; v < 4; ++v)
        for (int slice = 0; slice < SLICES; ++slice) {
          int row;
          int const physical = phys_word(lane, v, slice, CUBE_H, &row);
          if (row != row_wanted) continue;
          int const logical = 8 * slice + 4 * (v % 2) + lane % 4;
          if (logical_to_phys.count(logical) && logical_to_phys[logical] != physical) ok = false;
          logical_to_phys[logical] = physical;
        }

    bool row_ok = logical_to_phys.size() == 32;
    for (int logical = 0; logical < 32; ++logical) {
      auto const it = logical_to_phys.find(logical);
      if (it == logical_to_phys.end()) {
        row_ok = false;
        continue;
      }
      int const slice = logical / 8;
      int const logical_half = (logical % 8) / 4;
      int const lane_col = logical % 4;
      int const line = row_wanted / 4;
      int const slice_start_vec = (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2;
      int const run_start = CUBE_H * 8 * slice + line * 32
                          + ((2 * (row_wanted % 4) + slice_start_vec) % 8) * 4;
      int const expected = run_start + 4 * (logical_half ^ (line & 1)) + lane_col;
      row_ok &= it->second == expected;
    }
    std::printf("row %2d: 32 words, four runs, half-run permutation %s -> %s\n",
                row_wanted, ((row_wanted / 4) & 1) ? "swap" : "identity", row_ok ? "PASS" : "FAIL");
    ok &= row_ok;
  }

  std::printf("first-R writer permutation: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

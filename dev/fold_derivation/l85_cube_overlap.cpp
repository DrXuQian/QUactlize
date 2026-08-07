// L85 -- CAN EIGHT CUBES OVERLAP WHEN EACH CARRIES ITS FIRST R ROWS? The old proof checked row 0 only. This one
// compares every row of cube i with every row of cube j, for R=1..16, using the exact word placement replayed from
// ppu_tsm_ld_swzl_sim. Eight sub-tiles is the largest current packed-A consumer: int2 has two 64-half cubes per
// stage and at most four stages. The in-kernel check repeats the comparison for the instantiated cube/stage count.
//
// Pitches are searched, not extrapolated from R=1. The selected pitch is the smallest collision-free candidate whose
// base is a 32-word/64-half/128-byte multiple, preserving both every cube base's alignment and smem_b's alignment.
#include <cstdio>
#include <set>

static int word_of(int lane, int v, int coord_h, int slice_idx, int CUBE_H, int* out_row) {
  int slice_word_base   = CUBE_H * 8 * slice_idx;
  int slice_start_vec   = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;
  int lane_row_idx      = lane / 4 + coord_h;
  int lane_col_idx      = lane % 4;
  int vreg_row_idx      = (v / 2) * 8 + lane_row_idx;
  int vreg_line_idx     = vreg_row_idx / 4;
  int vreg_vec_idx      = (vreg_row_idx % 4) * 2 + (v % 2);
  int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
  int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;
  *out_row = vreg_row_idx;
  return slice_word_base + vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
}

static std::set<int> row_words(int row_wanted, int CUBE_H, int SLICES) {
  std::set<int> words;
  for (int lane = 0; lane < 32; ++lane)
    for (int v = 0; v < 4; ++v)
      for (int s = 0; s < SLICES; ++s) {
        int row;
        int const word = word_of(lane, v, 0, s, CUBE_H, &row);
        if (row == row_wanted) words.insert(word);
      }
  return words;
}

static std::set<int> first_rows_words(int rows, int CUBE_H, int SLICES) {
  std::set<int> words;
  for (int row = 0; row < rows; ++row) {
    auto const one_row = row_words(row, CUBE_H, SLICES);
    words.insert(one_row.begin(), one_row.end());
  }
  return words;
}

// The union contains every one of the R rows exactly once (checked in main), so comparing the two unions is the
// complete R x R comparison without rebuilding the same per-row sets in the pitch-search inner loop.
static bool disjoint(int rows, int subtiles, int pitch_words, int CUBE_H, int SLICES) {
  auto const words = first_rows_words(rows, CUBE_H, SLICES);
  for (int i = 0; i < subtiles; ++i)
    for (int j = i + 1; j < subtiles; ++j)
      for (int word_i : words) {
        int const word_j = i * pitch_words + word_i - j * pitch_words;
        if (words.count(word_j)) return false;
      }
  return true;
}

static int first_clean_pitch(int rows, int subtiles, int step_words, int CUBE_H, int SLICES) {
  for (int pitch = step_words; pitch <= CUBE_H * 32; pitch += step_words)
    if (disjoint(rows, subtiles, pitch, CUBE_H, SLICES)) return pitch;
  return 0;
}

int main() {
  constexpr int CUBE_H = 16;
  constexpr int CUBE_W_HALFS = 64;
  constexpr int SLICES = CUBE_W_HALFS / 16;
  constexpr int SUBTILES = 8;
  constexpr int CUBE_WORDS = CUBE_H * CUBE_W_HALFS / 2;
  constexpr int ALIGN_WORDS = 32;  // 64 halfs = 128 B
  bool ok = true;

  std::printf("R  owned words  minimum 128-B-aligned clean pitch  total read span (%d sub-tiles)\n", SUBTILES);
  for (int rows = 1; rows <= CUBE_H; ++rows) {
    int const aligned = first_clean_pitch(rows, SUBTILES, ALIGN_WORDS, CUBE_H, SLICES);
    int const span = (SUBTILES - 1) * aligned + CUBE_WORDS;
    auto const owned = first_rows_words(rows, CUBE_H, SLICES);
    std::printf("%2d %5zu words             %4d words/%4d halfs  %5d words/%6d halfs\n",
                rows, owned.size(), aligned, aligned * 2, span, span * 2);
    ok &= owned.size() == size_t(rows * 32) && aligned > 0;
    ok &= disjoint(rows, SUBTILES, aligned, CUBE_H, SLICES);
    ok &= (aligned % ALIGN_WORDS) == 0;
  }

  std::printf("R x R collision model and 128-B alignment: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

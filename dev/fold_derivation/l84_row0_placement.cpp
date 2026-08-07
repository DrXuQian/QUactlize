// L84 -- WHICH WORDS BELONG TO THE FIRST R ROWS? Host replay of ppu_tsm_ld_swzl_sim's arithmetic, verbatim.
// The replay was validated 0-mismatch against hardware in l2l3/l17/l7/l10/l12/l13/l16. This promotes the old
// row-0 placement proof: every row must own four 8-word (16-half) runs, and the first R rows must own exactly
// R*32 distinct words before they can be backed by overlapping cube allocations.
#include <cstdio>
#include <set>

// Verbatim from cute/arch/copy_ppu0010_aiu.hpp: ppu_tsm_ld_swzl_sim, SWAP=true branch (fp16 A).
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

static int print_runs(std::set<int> const& words) {
  int previous = -99;
  int runs = 0;
  int start = -1;
  for (int word : words) {
    if (word != previous + 1) {
      if (start >= 0) std::printf("[%d,%d] ", start, previous);
      start = word;
      ++runs;
    }
    previous = word;
  }
  if (start >= 0) std::printf("[%d,%d]", start, previous);
  return runs;
}

int main() {
  constexpr int CUBE_H = 16;
  constexpr int SLICES = 4;  // CUBE_W = 64 halfs = 32 words/row = four 8-word slices
  bool ok = true;
  std::set<int> first_r_words;

  for (int row = 0; row < CUBE_H; ++row) {
    auto const words = row_words(row, CUBE_H, SLICES);
    std::printf("row %2d owns %2zu words in ", row, words.size());
    int const runs = print_runs(words);
    std::printf("  -> %d runs\n", runs);
    ok &= words.size() == 32 && runs == 4;
    first_r_words.insert(words.begin(), words.end());
    std::printf("  first R=%-2d rows own %3zu distinct words (expected %3d)\n",
                row + 1, first_r_words.size(), (row + 1) * 32);
    ok &= first_r_words.size() == size_t((row + 1) * 32);
  }

  std::printf("first-R placement: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

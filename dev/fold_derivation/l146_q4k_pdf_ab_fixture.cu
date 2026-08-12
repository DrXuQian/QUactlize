// Host-only contract for the two representations used by INBOX 132B.
#include "q4k_pdf_ab_fixture.hpp"

#include <cstdio>
#include <string>

int main() {
  q4k_pdf_ab::Shape const shape{"L146", 2, 16, 1024, 2, 8, 1, false};
  auto problem = q4k_pdf_ab::make_problem(shape);
  std::string why;
  if (!q4k_pdf_ab::verify_representation(problem, why)) {
    std::fprintf(stderr, "positive representation gate failed: %s\n", why.c_str());
    return 1;
  }
  std::size_t const raw_want = std::size_t(2) * 16 * (1024 / 256) * 144;
  std::size_t const ours_want = std::size_t(2) * 16 * 1024 / 2
                              + std::size_t(2) * 2 * (1024 / 32) * 16 * sizeof(half);
  std::size_t const ours_got = problem.low.size()
                             + (problem.scales.size() + problem.zeros.size()) * sizeof(half);
  if (problem.raw.size() * sizeof(q4k_pdf_reconstruction::block_q4_K) != raw_want ||
      ours_got != ours_want) {
    std::fprintf(stderr, "representation byte model drift raw=%zu/%zu ours=%zu/%zu\n",
                 problem.raw.size() * sizeof(q4k_pdf_reconstruction::block_q4_K), raw_want,
                 ours_got, ours_want);
    return 2;
  }

  int reds = 0;
  problem.raw[0].qs[0] ^= 1;
  if (!q4k_pdf_ab::verify_representation(problem, why)) ++reds;
  problem.raw[0].qs[0] ^= 1;
  if (!q4k_pdf_ab::verify_representation(problem, why)) return 3;

  problem.low[0] ^= 1;
  if (!q4k_pdf_ab::verify_representation(problem, why)) ++reds;
  problem.low[0] ^= 1;
  if (!q4k_pdf_ab::verify_representation(problem, why)) return 4;

  std::uint16_t bits = q4k_pdf_ab::half_bits(problem.scales[0]);
  std::uint16_t planted = std::uint16_t(bits + 1);
  std::memcpy(&problem.scales[0], &planted, sizeof(planted));
  if (!q4k_pdf_ab::verify_representation(problem, why)) ++reds;
  std::memcpy(&problem.scales[0], &bits, sizeof(bits));
  if (!q4k_pdf_ab::verify_representation(problem, why)) return 5;

  if (reds != 3) {
    std::fprintf(stderr, "negative controls did not all red: %d/3\n", reds);
    return 6;
  }
  std::printf("L146 PASS: raw/native Q4_K agree over %d codes; 3/3 planted pack faults red; bytes=%zu/%zu\n",
              shape.l * shape.n * shape.k, raw_want, ours_want);
  return 0;
}

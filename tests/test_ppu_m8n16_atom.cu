// G1 for the ppu001 m8n16k16 atom.  This deliberately stops below every collective layer:
// no ldmatrix, shared memory, dequantization, scheduler, or epilogue is involved.  One warp
// constructs the documented A2/B4/C4 register ABI from ordinary global loads, invokes the
// atom once, and writes all 8x16 results through the documented accumulator map.
//
// Why both a basis sweep and an asymmetric case:
//   * the 16 one-hot K cases make every half-word in A2 and B4 individually load-bearing;
//   * the dense asymmetric case makes an M/N/K permutation visible, and starts from nonzero C.
// All numbers are small powers-of-two fractions, so the FP32 golden is exact rather than a
// tolerance-dependent proxy for a different accumulation order.
//
// This file exports run_ppu_m8n16_g1() and deliberately has no main: G1 and G2 are linked into
// one gate target so one ppu001 build/run produces the complete red/green evidence.  G0 must
// separately audit that the generated hgcc command contains only -arch=ppu_10.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "cutlass/half.h"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cutlass/util/device_memory.h"
#include "cute/arch/mma_ppu0010.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "helper.h"

namespace {

using Atom   = cute::PPU0010_8x16x16_F32F16F16F32_TN;
using Traits = cute::MMA_Traits<Atom>;
using half_t = cutlass::half_t;

constexpr int M = 8;
constexpr int N = 16;
constexpr int K = 16;

// Freeze the exact contract that G1 is meant to verify.  A later accidental edit to a trait
// cannot turn the test into a self-consistent test of the wrong layout.
using ExpectedShape = cute::Shape<cute::_8, cute::_16, cute::_16>;
using ExpectedA = cute::Layout<
    cute::Shape<cute::Shape<cute::_4, cute::_8>, cute::Shape<cute::_2, cute::_2>>,
    cute::Stride<cute::Stride<cute::_16, cute::_1>, cute::Stride<cute::_8, cute::_64>>>;
using ExpectedB = cute::Layout<
    cute::Shape<cute::Shape<cute::_4, cute::_8>, cute::Shape<cute::_2, cute::_2, cute::_2>>,
    cute::Stride<cute::Stride<cute::_32, cute::_1>,
                 cute::Stride<cute::_16, cute::_128, cute::_8>>>;
using ExpectedC = cute::Layout<
    cute::Shape<cute::Shape<cute::_4, cute::_8>, cute::_4>,
    cute::Stride<cute::Stride<cute::_8, cute::_1>, cute::_32>>;

static_assert(std::is_same_v<typename Traits::Shape_MNK, ExpectedShape>);
static_assert(std::is_same_v<typename Traits::ALayout, ExpectedA>);
static_assert(std::is_same_v<typename Traits::BLayout, ExpectedB>);
static_assert(std::is_same_v<typename Traits::CLayout, ExpectedC>);
static_assert(std::is_same_v<typename Traits::ValTypeA, half_t>);
static_assert(std::is_same_v<typename Traits::ValTypeB, half_t>);
static_assert(std::is_same_v<typename Traits::ValTypeC, float>);
static_assert(std::is_same_v<typename Traits::ValTypeD, float>);
static_assert(std::extent_v<Atom::ARegisters> == 2);
static_assert(std::extent_v<Atom::BRegisters> == 4);
static_assert(std::extent_v<Atom::CRegisters> == 4);
static_assert(std::extent_v<Atom::DRegisters> == 4);

template <class Arch, int TileM, int WarpM>
using SelectedAtom = typename cutlass::gemm::config::GetMmaInstForShape<
    Arch, half_t, half_t, float, TileM, WarpM>::type;

// The selector is deliberately narrower than the raw instruction's type signature.  These four
// assertions prevent a future "convenient" type-only substitution from changing established m16
// kernels or making ppu0015 appear to support an instruction that fails during intrinsic lowering.
static_assert(std::is_same_v<SelectedAtom<cutlass::arch::PPU0010, 8, 8>, Atom>);
static_assert(std::is_same_v<SelectedAtom<cutlass::arch::PPU0010, 16, 8>,
                             cute::PPU0010_16x16x16_F32F16F16F32_TN>);
static_assert(std::is_same_v<SelectedAtom<cutlass::arch::PPU0010, 8, 16>,
                             cute::PPU0010_16x16x16_F32F16F16F32_TN>);
static_assert(std::is_same_v<SelectedAtom<cutlass::arch::PPU0015, 8, 8>,
                             cute::PPU0015_16x16x16_F32F16F16F32_TN>);

// These formulas are the independently recorded ppu001 register ABI.  Do not derive them from
// Traits inside this gate: the instruction and a bad trait must not be able to agree by accident.
// A scalar value v is one 16-bit half-word; two consecutive values occupy one uint32 register.
__global__ void g1_ppu_m8n16k16_atom(half_t const* A, half_t const* B, float const* C, float* D) {
  int const lane = int(threadIdx.x) & 31;
  if (blockIdx.x != 0 || threadIdx.x >= 32) return;

  uint32_t ar[2] = {0, 0};
  uint32_t br[4] = {0, 0, 0, 0};

  // A: m = lane/4, k = 2*(lane%4) + (v%2) + 8*(v/2), v in [0,4).
  int const m = lane / 4;
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    int const k = 2 * (lane % 4) + (v % 2) + 8 * (v / 2);
    ar[v / 2] |= uint32_t(A[m * K + k].raw()) << (16 * (v % 2));
  }

  // The CuTe B atom coordinate is (N,K), although the ordinary GEMM input below is stored KxN.
  // Hence n = lane/4 + 8*(v/4), k = 2*(lane%4) + (v%2) + 8*((v/2)%2),
  // followed by the transpose load B[k,n].  This is the validated ppu001 row.col B ABI.
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    int const n = lane / 4 + 8 * (v / 4);
    int const k = 2 * (lane % 4) + (v % 2) + 8 * ((v / 2) % 2);
    br[v / 2] |= uint32_t(B[k * N + n].raw()) << (16 * (v % 2));
  }

  float cr[4];
  float dr[4];
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    int const n = lane % 4 + 4 * v;
    cr[v] = C[m * N + n];
  }

  Atom::fma(dr[0], dr[1], dr[2], dr[3],
            ar[0], ar[1], br[0], br[1], br[2], br[3],
            cr[0], cr[1], cr[2], cr[3]);

#pragma unroll
  for (int v = 0; v < 4; ++v) {
    int const n = lane % 4 + 4 * v;
    D[m * N + n] = dr[v];
  }
}

struct TestCase {
  std::vector<half_t> a = std::vector<half_t>(M * K);
  std::vector<half_t> b = std::vector<half_t>(K * N);
  std::vector<float> c = std::vector<float>(M * N);
};

TestCase make_basis(int basis_k) {
  TestCase t;
  for (int m = 0; m < M; ++m) {
    // Nonuniform row factors keep an M permutation visible in a one-hot case.
    t.a[m * K + basis_k] = half_t(float(m + 1) * 0.25f);
  }
  for (int n = 0; n < N; ++n) {
    // Exact, distinct, nonzero dyadics: every B half-word remains load-bearing in every K basis.
    t.b[basis_k * N + n] = half_t(float(2 * n - 15) * 0.0625f);
  }
  return t;
}

TestCase make_asymmetric_nonzero_c() {
  TestCase t;
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      t.a[m * K + k] = half_t(float((m * 13 + k * 7 + 3) % 15 - 7) * 0.125f);
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      t.b[k * N + n] = half_t(float((k * 11 + n * 5 + 1) % 17 - 8) * 0.0625f);
    }
  }
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      // All 128 accumulator inputs are unique and nonzero, so every C4 register mapping matters.
      t.c[m * N + n] = float(m * N + n + 1) * (1.0f / 256.0f);
    }
  }
  return t;
}

std::vector<float> cpu_golden(TestCase const& t) {
  std::vector<float> out(M * N);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = t.c[m * N + n];
      for (int k = 0; k < K; ++k) {
        acc += float(t.a[m * K + k]) * float(t.b[k * N + n]);
      }
      out[m * N + n] = acc;
    }
  }
  return out;
}

std::vector<float> device_run(TestCase const& t) {
  cutlass::DeviceAllocation<half_t> da(M * K), db(K * N);
  cutlass::DeviceAllocation<float> dc(M * N), dd(M * N);
  da.copy_from_host(t.a.data());
  db.copy_from_host(t.b.data());
  dc.copy_from_host(t.c.data());

  // An unwritten output must fail rather than look like a plausible zero.
  std::vector<float> poison(M * N, std::numeric_limits<float>::quiet_NaN());
  dd.copy_from_host(poison.data());
  g1_ppu_m8n16k16_atom<<<1, 32>>>(da.get(), db.get(), dc.get(), dd.get());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<float> got(M * N);
  dd.copy_to_host(got.data());
  return got;
}

int compare(char const* name, std::vector<float> const& got,
            std::vector<float> const& want, bool print_all) {
  int bad = 0;
  float max_abs = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    float const err = std::fabs(got[i] - want[i]);
    if (!std::isfinite(got[i]) || got[i] != want[i]) ++bad;
    if (std::isfinite(err) && err > max_abs) max_abs = err;
  }

  if (print_all) {
    std::printf("\n[G1][%s] all 128 outputs as got/want%s\n", name,
                bad ? "  (* marks mismatch)" : "");
    for (int m = 0; m < M; ++m) {
      std::printf("  m=%d", m);
      for (int n = 0; n < N; ++n) {
        int const i = m * N + n;
        bool const mismatch = !std::isfinite(got[i]) || got[i] != want[i];
        std::printf("  %+.6g/%+.6g%s", got[i], want[i], mismatch ? "*" : "");
      }
      std::printf("\n");
    }
  }
  std::printf("[G1][%s] outputs=%d bad=%d max_abs=%g\n", name, M * N, bad, max_abs);
  return bad;
}

}  // namespace

int run_ppu_m8n16_g1() {
  std::printf("== [G1] ppu001 m8n16k16 pure atom: no ldmatrix/dequant/collective ==\n");

  int total_bad = 0;

  // Sweep all 16 K bases.  Every case reports its failure count; k=13 also displays
  // the complete 128-output matrix so the run log contains the requested per-output evidence.
  int basis_bad = 0;
  for (int k = 0; k < K; ++k) {
    TestCase const t = make_basis(k);
    char name[32];
    std::snprintf(name, sizeof(name), "one-hot k=%d", k);
    basis_bad += compare(name, device_run(t), cpu_golden(t), k == 13);
  }
  std::printf("[G1][one-hot sweep] cases=16 outputs=%d bad=%d\n", K * M * N, basis_bad);
  total_bad += basis_bad;

  // This case exercises every A/B half-word at once and verifies that C is accumulated, not ignored.
  {
    TestCase const t = make_asymmetric_nonzero_c();
    total_bad += compare("asymmetric + nonzero C", device_run(t), cpu_golden(t), true);
  }

  std::printf("[G1] %s: total_bad=%d\n", total_bad == 0 ? "PASS" : "FAIL", total_bad);
  return total_bad == 0 ? 0 : 1;
}

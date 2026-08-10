// G2 for the ppu001 m8n16k16 A fragment.  This is deliberately below the
// collective: it exercises the real AIU .padz.swzl write and the real
// m8n8.x4.swzl shared-to-register instruction, then projects only v0/v1 into
// the two registers required by the m8 atom.
//
// The physical AIU cube stays 16 rows high.  Cases 0 and 1 carry identical,
// uniquely tagged rows 0..7 and different poison in rows 8..15.  A valid m8
// projection must therefore satisfy all three conditions:
//   * the complete x4 delivery agrees bit-for-bit with the documented map;
//   * projected v0/v1 do not change when the lower-eight poison changes;
//   * v2/v3 do change, proving that the poison really traversed the same AIU
//     write/read pair instead of being optimised away or left unwritten.
//
// The red control is intentionally a separate, linear shared tile.  It copies
// the exact NVIDIA address expression formerly used by llama.cpp for
// tile<8,8> and feeds that address to PPU's real plain m8n8.x2 instruction.
// PPU x2 has a different register distribution, so comparing those two output
// registers with the m8 ALayout MUST find mismatches.  A zero-mismatch red
// control means this gate cannot see the historical silent-corruption shape;
// in that case G2 fails even if the green path happens to pass.
//
// This source has no main.  The aggregate G0/G1/G2 target calls:
//
//     int run_ppu_m8n16_g2();

// G2 is box-only and must be built for ppu001 (-arch=ppu_10).


#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "cutlass/half.h"
#include "cutlass/util/device_memory.h"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/arch/mma_ppu0010.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "helper.h"

namespace {

using half_t = cutlass::half_t;
using Atom = cute::PPU0010_8x16x16_F32F16F16F32_TN;
using Traits = cute::MMA_Traits<Atom>;

constexpr int kCases = 2;
constexpr int kCubeH = 16;
constexpr int kCubeW = 64;
constexpr int kCubeElements = kCubeH * kCubeW;
constexpr int kWarp = 32;
constexpr int kX4Registers = 4;
constexpr int kM8Registers = 2;
constexpr int kHalfsPerRegister = 2;

// Freeze the contract independently of the implementation used below.  A bad
// edit to the atom trait must not make the read gate self-consistently green.
using ExpectedA = cute::Layout<
    cute::Shape<cute::Shape<cute::_4, cute::_8>,
                cute::Shape<cute::_2, cute::_2>>,
    cute::Stride<cute::Stride<cute::_16, cute::_1>,
                 cute::Stride<cute::_8, cute::_64>>>;
static_assert(std::is_same_v<typename Traits::ALayout, ExpectedA>);
static_assert(std::extent_v<Atom::ARegisters> == kM8Registers);

// One b16 AIU operation writes the whole 16x64 physical cube (16384 bits).
using AiuWrite = cute::PPU0010_AIU_LOAD<
    cute::C<kCubeElements * 16>, half_t, false, true>;
static_assert(std::is_same_v<typename cute::Copy_Traits<AiuWrite>::ThrID,
                             cute::Layout<cute::_1>>);
using SwzlRead = cute::PPU0010_TSM_LD_SWZL<
    half_t, kCubeH, kCubeW, true, false, 1>;

struct alignas(32) SharedStorage {
  // AIU requires a 32-byte-aligned TSM destination on ppu001.
  std::uint16_t swzl[kCubeElements];
  // A distinct linear tile for the planted x2-address defect.  Keeping it
  // separate prevents the red control from merely detecting "plain ldmatrix
  // cannot decode AIU swizzle", which is not the bug this control promises.
  std::uint16_t linear[8 * 16];
};

__global__ void g2_device(std::uint16_t const* input,
                          std::uint32_t* x4_output,
                          std::uint32_t* bad_x2_output) {
  int const lane = int(threadIdx.x);
  int const test_case = int(blockIdx.x);
  if (lane >= kWarp || test_case >= kCases) return;

  __shared__ SharedStorage storage;
  std::uint16_t const* src = input + test_case * kCubeElements;

  // Populate the independent linear tile used only by the red control.
  for (int i = lane; i < 8 * 16; i += kWarp) {
    int const row = i / 16;
    int const k = i % 16;
    storage.linear[i] = src[row * kCubeW + k];
  }
  __syncthreads();

  // Production-equivalent physical write: AIU is a single-thread bulk DMA,
  // matching Copy_Traits<PPU0010_AIU_LOAD>::ThrID == Layout<_1>.  Exactly lane
  // zero issues the transfer; all lanes still execute commit/wait and the CTA
  // barrier below.  Calling the raw operation from every lane would launch 32
  // duplicate DMAs and would not test the collective's ownership contract.
  // dim_h==cube_h is deliberate: rows 8..15 contain poison rather than pad
  // zeroes, while the opcode itself remains the real .padz.swzl form used by
  // the collective.
  cute::AiuDesc desc{};
  desc.gmem_ptr = reinterpret_cast<std::uint8_t const*>(src);
  desc.dim_h = kCubeH;
  desc.dim_w = kCubeW;
  desc.cube_h = kCubeH;
  desc.cube_w = kCubeW;
  desc.offset_w = 0;
  if (lane == 0) {
    AiuWrite::copy(storage.swzl, src, desc, 0, 0, 0);
  }
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // The safe m8 form: retain the physical x4 read and project v0/v1.  The
  // host checks all four registers first, so this cannot pass by accidentally
  // manufacturing the projected values after a bad read.
  std::uint32_t x4[kX4Registers] = {};
  SwzlRead::copy(x4, storage.swzl, 0, 0, 0, 0);
#pragma unroll
  for (int v = 0; v < kX4Registers; ++v) {
    x4_output[(test_case * kWarp + lane) * kX4Registers + v] = x4[v];
  }

  // Planted defect, copied algebraically from NVIDIA's tile<8,8> loader:
  //
  //   row      = lane % I
  //   word_col = ((lane / I) * (J/2)) % J       I=J=8
  //
  // The addresses are valid and 16-byte aligned; only the assumed register
  // distribution is wrong for PPU.  This is the important failure shape:
  // finite values at valid addresses, silently permuted.
  constexpr int kI = 8;
  constexpr int kJ = 8;       // uint32 columns: 16 scalar fp16 columns
  constexpr int kStride = 8;  // uint32 words per row
  int const row = lane % kI;
  int const word_col = ((lane / kI) * (kJ / 2)) % kJ;
  auto const* linear_words = reinterpret_cast<std::uint32_t const*>(storage.linear);
  auto const& nvidia_address = *reinterpret_cast<cute::uint128_t const*>(
      linear_words + row * kStride + word_col);

  std::uint32_t wrong[2] = {};
  cute::PPU_U32x2_LDSM_N::copy(nvidia_address, wrong[0], wrong[1]);
  bad_x2_output[(test_case * kWarp + lane) * 2 + 0] = wrong[0];
  bad_x2_output[(test_case * kWarp + lane) * 2 + 1] = wrong[1];
}

constexpr std::uint16_t upper_tag(int row, int k) {
  // 0x0400..0x05ff: 512 unique values, disjoint from either poison set.
  return std::uint16_t(0x0400u + unsigned(row * kCubeW + k));
}

constexpr std::uint16_t lower_tag(int test_case, int row, int k) {
  // The two cases differ at every lower-half coordinate.
  unsigned const base = test_case == 0 ? 0x6000u : 0xa000u;
  return std::uint16_t(base + unsigned((row - 8) * kCubeW + k));
}

std::uint16_t halfword(std::uint32_t word, int h) {
  return std::uint16_t((word >> (16 * h)) & 0xffffu);
}

}  // namespace

int run_ppu_m8n16_g2() {
  std::printf("== ppu001 m8n16 G2: AIU physical-cube delivery and planted x2 defect ==\n");
  std::printf("[G2-path] write=ppu.cp.async.aiu...padz.swzl.2d.b16 cube=16x64 "
              "read=ppu.tc01.ldmatrix...m8n8.x4.swzl.shared.b16 project=v0,v1\n");

  std::vector<std::uint16_t> input(kCases * kCubeElements);
  for (int c = 0; c < kCases; ++c) {
    for (int row = 0; row < kCubeH; ++row) {
      for (int k = 0; k < kCubeW; ++k) {
        input[(c * kCubeH + row) * kCubeW + k] =
            row < 8 ? upper_tag(row, k) : lower_tag(c, row, k);
      }
    }
  }

  cutlass::DeviceAllocation<std::uint16_t> d_input(input.size());
  cutlass::DeviceAllocation<std::uint32_t> d_x4(
      kCases * kWarp * kX4Registers);
  cutlass::DeviceAllocation<std::uint32_t> d_bad_x2(
      kCases * kWarp * kM8Registers);
  d_input.copy_from_host(input.data());

  g2_device<<<kCases, kWarp>>>(d_input.get(), d_x4.get(), d_bad_x2.get());
  CUTLASS_PPU_CHECK(hggcGetLastError());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<std::uint32_t> x4(kCases * kWarp * kX4Registers);
  std::vector<std::uint32_t> bad_x2(kCases * kWarp * kM8Registers);
  d_x4.copy_to_host(x4.data());
  d_bad_x2.copy_to_host(bad_x2.data());

  int x4_bad = 0;
  int projected_changed = 0;
  int lower_changed = 0;
  constexpr int kProjectedValues = kWarp * kM8Registers * kHalfsPerRegister;
  constexpr int kLowerValues = kProjectedValues;

  // Independent word map for the real x4 delivery:
  //   row  = lane/4 + 8*(v/2)
  //   word = lane%4 + 4*(v%2)
  // This is also why x4[v0,v1] is exactly the m8 ALayout.
  for (int c = 0; c < kCases; ++c) {
    for (int lane = 0; lane < kWarp; ++lane) {
      for (int v = 0; v < kX4Registers; ++v) {
        std::uint32_t const got = x4[(c * kWarp + lane) * kX4Registers + v];
        int const row = lane / 4 + 8 * (v / 2);
        int const word = lane % 4 + 4 * (v % 2);
        for (int h = 0; h < kHalfsPerRegister; ++h) {
          std::uint16_t const want = input[(c * kCubeH + row) * kCubeW + 2 * word + h];
          x4_bad += halfword(got, h) != want;
        }
      }
    }
  }

  for (int lane = 0; lane < kWarp; ++lane) {
    for (int v = 0; v < kX4Registers; ++v) {
      std::uint32_t const a = x4[(0 * kWarp + lane) * kX4Registers + v];
      std::uint32_t const b = x4[(1 * kWarp + lane) * kX4Registers + v];
      for (int h = 0; h < kHalfsPerRegister; ++h) {
        if (v < kM8Registers) {
          projected_changed += halfword(a, h) != halfword(b, h);
        } else {
          lower_changed += halfword(a, h) != halfword(b, h);
        }
      }
    }
  }

  // Compare the deliberately wrong x2 result with the independently frozen
  // m8 ALayout.  Only case 0 is needed: this is a planted-fault sensitivity
  // check, not another implementation candidate.
  int red_mismatches = 0;
  for (int lane = 0; lane < kWarp; ++lane) {
    for (int reg = 0; reg < kM8Registers; ++reg) {
      std::uint32_t const got = bad_x2[lane * kM8Registers + reg];
      int const row = lane / 4;
      for (int h = 0; h < kHalfsPerRegister; ++h) {
        int const k = 2 * (lane % 4) + h + 8 * reg;
        std::uint16_t const want = input[row * kCubeW + k];
        red_mismatches += halfword(got, h) != want;
      }
    }
  }

  int const green_mismatches =
      x4_bad + projected_changed + (kLowerValues - lower_changed);
  bool const green_pass = green_mismatches == 0;
  bool const red_pass = red_mismatches > 0;

  std::printf("[G2-green-detail] x4_values=%d x4_bad=%d projected_changed=%d/%d "
              "lower_poison_changed=%d/%d\n",
              kCases * kWarp * kX4Registers * kHalfsPerRegister,
              x4_bad, projected_changed, kProjectedValues,
              lower_changed, kLowerValues);
  std::printf("[G2-green] mismatches=%d %s\n",
              green_mismatches, green_pass ? "PASS" : "FAIL");
  std::printf("[G2-negative] mismatches=%d %s\n",
              red_mismatches, red_pass ? "EXPECTED_RED/PASS" : "UNEXPECTED_GREEN/FAIL");

  bool const pass = green_pass && red_pass;
  std::printf("G2 %s: green=%s negative_control=%s\n",
              pass ? "PASS" : "FAIL",
              green_pass ? "PASS" : "FAIL",
              red_pass ? "EXPECTED_RED" : "UNEXPECTED_GREEN");
  return pass ? 0 : 1;
}

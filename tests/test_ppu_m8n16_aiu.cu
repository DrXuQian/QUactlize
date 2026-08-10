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
// The red control replays the historical fault on the SAME raw production
// payload; it does not issue a second instruction.  The old llama.cpp PPU arm
// reused NVIDIA's plain-x2 address-provider formula
//
//   row = lane % 8;  base_word = ((lane / 8) * 4) % 8;
//
// even though PPU x2 has a different provider-to-result redistribution.  For
// output (lane=4*row+a, reg), the PPU provider is
// `p=2*row+a/2+16*reg`; that provider supplies word offset `a%2` from its
// contiguous 64-bit window.  The correct m8 operand instead requires
// element-wise get_i/get_j.  To replay that exact indexing mistake without
// reviving the unproved plain-x2 opcode, the host applies the old NVIDIA
// formula to provider p on the real 16x64 x4 delivery, then inverts the frozen
// PPU x4 owner map.  Green and red therefore share the one AIU write, the one
// uniform-(0,0) x4 read, the same smem base, and the same physical 16-row
// geometry; only the logical indexing differs.  Unique tags make the result
// exact: only (lane,reg)=(0,0),(1,0) coincide, so the planted historical map
// must mismatch 124/128 halfwords.  A zero-mismatch red still fails G2.
//
// KNOWN PRE-EXISTING ACTLIZE DEFECT, FAIL-CLOSED BY #114.  The six ppu001
// plain-LDSM atoms in cute/arch/copy_ppu.hpp and the six counterparts in
// cutlass/arch/memory_ppu.h formerly contained assembler-rejected
// `ppu.tc01.ex.ldmatrix` spellings.  Their correct non-swzl SDK grammar is
// still unproved: the direct ppu001 entries are now deleted C++ functions and
// the two legacy helpers use dependent static_asserts; ppu0015's tc02 entries
// remain intact.  G2 deliberately instantiates none of them.  A future plain
// x1/x2/x4 N/T implementation still requires SDK compile plus numerical gates.
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
};

__global__ void g2_device(std::uint16_t const* input,
                          std::uint32_t* x4_output) {
  int const lane = int(threadIdx.x);
  int const test_case = int(blockIdx.x);
  if (lane >= kWarp || test_case >= kCases) return;

  __shared__ SharedStorage storage;
  std::uint16_t const* src = input + test_case * kCubeElements;

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
  std::printf("== ppu001 m8n16 G2: AIU delivery and historical x2-index replay ==\n");
  std::printf("[G2-path] production write=ppu.cp.async.aiu...padz.swzl.2d.b16 cube=16x64 "
              "read=ppu.tc01.ldmatrix...m8n8.x4.swzl.shared.b16 project=v0,v1\n");
  std::printf("[G2-control-path] same-payload=production-x4 cube=16x64 coords=(0,0) "
              "green=get_i/get_j red=historical-nvidia-x2-provider-map\n");

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
  d_input.copy_from_host(input.data());

  g2_device<<<kCases, kWarp>>>(d_input.get(), d_x4.get());
  CUTLASS_PPU_CHECK(hggcGetLastError());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<std::uint32_t> x4(kCases * kWarp * kX4Registers);
  d_x4.copy_to_host(x4.data());

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
          std::uint16_t const want =
              input[(c * kCubeH + row) * kCubeW + 2 * word + h];
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

  // Replay the exact historical pre-fix index on the raw case-0 production
  // payload.  Reconstruct PPU x2's provider-to-result redistribution from the
  // SDK's getThreadAddr1D(128) example.  For output lane `4*row+a`, register
  // `reg` receives word `a%2` from provider `2*row+a/2+16*reg`.  Apply the
  // historical NVIDIA pointer formula to that provider, then invert the
  // already-validated x4 delivery map to find the resulting word:
  //
  //   provider = 2*row + a/2 + 16*reg
  //   bad_row  = provider % 8
  //   bad_base = ((provider / 8) * 4) % 8
  //   bad_word = bad_base + a%2
  //   owner    = x4[4*bad_row + bad_word%4][bad_word/4]
  //
  // Green uses the frozen m8 get_i/get_j map:
  // row=lane/4, word=lane%4+4*reg.
  int red_mismatches = 0;
  int bad_map_bad = 0;
  int coincident_words = 0;
  constexpr int kI = 8;
  constexpr int kJ = 8;
  for (int lane = 0; lane < kWarp; ++lane) {
    int const good_row = lane / 4;
    int const lane_word = lane % 4;
    for (int reg = 0; reg < kM8Registers; ++reg) {
      int const provider_lane =
          2 * good_row + lane_word / 2 + 16 * reg;
      int const nvidia_row = provider_lane % kI;
      int const nvidia_base_word =
          ((provider_lane / kI) * (kJ / 2)) % kJ;
      int const bad_word = nvidia_base_word + lane_word % 2;
      int const src_lane = 4 * nvidia_row + bad_word % 4;
      int const src_reg = bad_word / 4;
      std::uint32_t const got = x4[src_lane * kX4Registers + src_reg];
      int const good_word = lane_word + 4 * reg;
      coincident_words +=
          nvidia_row == good_row && bad_word == good_word;
      for (int h = 0; h < kHalfsPerRegister; ++h) {
        std::uint16_t const bad_want =
            input[nvidia_row * kCubeW + 2 * bad_word + h];
        bad_map_bad += halfword(got, h) != bad_want;
        std::uint16_t const good_want =
            input[good_row * kCubeW + 2 * good_word + h];
        red_mismatches += halfword(got, h) != good_want;
      }
    }
  }

  int const green_mismatches =
      x4_bad + projected_changed + (kLowerValues - lower_changed);
  constexpr int kProjectedWords = kWarp * kM8Registers;
  constexpr int kExpectedCoincidentWords = 2;
  constexpr int kExpectedRedMismatches =
      (kProjectedWords - kExpectedCoincidentWords) * kHalfsPerRegister;
  bool const green_pass = green_mismatches == 0;
  bool const red_pass =
      red_mismatches == kExpectedRedMismatches &&
      bad_map_bad == 0 && coincident_words == kExpectedCoincidentWords;

  std::printf("[G2-green-detail] x4_values=%d x4_bad=%d projected_changed=%d/%d "
              "lower_poison_changed=%d/%d\n",
              kCases * kWarp * kX4Registers * kHalfsPerRegister,
              x4_bad, projected_changed, kProjectedValues,
              lower_changed, kLowerValues);
  std::printf("[G2-green] mismatches=%d %s\n",
              green_mismatches, green_pass ? "PASS" : "FAIL");
  std::printf("[G2-negative-detail] same_payload=x4-swzl geometry=16x64 "
              "bad_map_values=%d bad_map_bad=%d coincident_words=%d/%d "
              "red_expected=%d/%d\n",
              kProjectedValues, bad_map_bad, coincident_words,
              kProjectedWords,
              kExpectedRedMismatches, kProjectedValues);
  std::printf("[G2-negative] mismatches=%d %s\n",
              red_mismatches, red_pass ? "EXPECTED_RED/PASS" : "UNEXPECTED_GREEN/FAIL");

  bool const pass = green_pass && red_pass;
  std::printf("G2 %s: green=%s negative_control=%s\n",
              pass ? "PASS" : "FAIL",
              green_pass ? "PASS" : "FAIL",
              red_pass ? "EXPECTED_RED" : "UNEXPECTED_GREEN");
  return pass ? 0 : 1;
}

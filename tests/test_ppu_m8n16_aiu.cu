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
// The red control must not change the load instruction while claiming to test
// an address bug.  A separate 32x64 guard cube therefore runs BOTH control
// arms through one PPU0010_TSM_LD_SWZL helper: good passes (0,0), while bad
// passes the exact NVIDIA tile<8,8> row/word formula as PPU coordinates.  The
// 32-row height is load-bearing: x4 adds lane/4 + 8*(v/2), so the bad coord can
// reach row 22; doing this on the 16-row production cube makes an out-of-range
// read, and a red result would no longer isolate address arithmetic.  The host
// checks every one of the bad arm's 512 halfwords against the shifted valid
// tags before accepting its exact 120/128 mismatch against the origin.
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
constexpr int kGuardH = 32;
constexpr int kCubeW = 64;
constexpr int kCubeElements = kCubeH * kCubeW;
constexpr int kGuardElements = kGuardH * kCubeW;
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
using ProdAiuWrite = cute::PPU0010_AIU_LOAD<
    cute::C<kCubeElements * 16>, half_t, false, true>;
using GuardAiuWrite = cute::PPU0010_AIU_LOAD<
    cute::C<kGuardElements * 16>, half_t, false, true>;
static_assert(std::is_same_v<typename cute::Copy_Traits<ProdAiuWrite>::ThrID,
                             cute::Layout<cute::_1>>);
static_assert(std::is_same_v<typename cute::Copy_Traits<GuardAiuWrite>::ThrID,
                             cute::Layout<cute::_1>>);
using ProdSwzlRead = cute::PPU0010_TSM_LD_SWZL<
    half_t, kCubeH, kCubeW, true, false, 1>;
using GuardSwzlRead = cute::PPU0010_TSM_LD_SWZL<
    half_t, kGuardH, kCubeW, true, false, 1>;

struct alignas(32) SharedStorage {
  // AIU requires a 32-byte-aligned TSM destination on ppu001.
  std::uint16_t prod_swzl[kCubeElements];
  std::uint16_t guard_swzl[kGuardElements];
};

// ONE lexical instruction seam for both guard arms.  The static #114 checker
// rejects a second primitive or a second smem base; only coord_w/coord_h may
// differ between good and bad.
CUTE_DEVICE void g2_guard_swzl_x4(std::uint32_t* frag,
                                  std::uint16_t* smem_base,
                                  int coord_w, int coord_h) {
  GuardSwzlRead::copy(frag, smem_base, coord_w, coord_h, 0, 0);
}

__global__ void g2_device(std::uint16_t const* prod_input,
                          std::uint16_t const* guard_input,
                          std::uint32_t* prod_x4_output,
                          std::uint32_t* guard_good_output,
                          std::uint32_t* guard_bad_output) {
  int const lane = int(threadIdx.x);
  int const test_case = int(blockIdx.x);
  if (lane >= kWarp || test_case >= kCases) return;

  __shared__ SharedStorage storage;
  std::uint16_t const* prod_src = prod_input + test_case * kCubeElements;

  // Production-equivalent physical write: AIU is a single-thread bulk DMA,
  // matching Copy_Traits<PPU0010_AIU_LOAD>::ThrID == Layout<_1>.  Exactly lane
  // zero issues the transfer; all lanes still execute commit/wait and the CTA
  // barrier below.  Calling the raw operation from every lane would launch 32
  // duplicate DMAs and would not test the collective's ownership contract.
  // dim_h==cube_h is deliberate: rows 8..15 contain poison rather than pad
  // zeroes, while the opcode itself remains the real .padz.swzl form used by
  // the collective.
  cute::AiuDesc desc{};
  desc.gmem_ptr = reinterpret_cast<std::uint8_t const*>(prod_src);
  desc.dim_h = kCubeH;
  desc.dim_w = kCubeW;
  desc.cube_h = kCubeH;
  desc.cube_w = kCubeW;
  desc.offset_w = 0;

  cute::AiuDesc guard_desc{};
  guard_desc.gmem_ptr = reinterpret_cast<std::uint8_t const*>(guard_input);
  guard_desc.dim_h = kGuardH;
  guard_desc.dim_w = kCubeW;
  guard_desc.cube_h = kGuardH;
  guard_desc.cube_w = kCubeW;
  guard_desc.offset_w = 0;
  if (lane == 0) {
    ProdAiuWrite::copy(storage.prod_swzl, prod_src, desc, 0, 0, 0);
    GuardAiuWrite::copy(storage.guard_swzl, guard_input, guard_desc, 0, 0, 0);
  }
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // The safe m8 form: retain the physical x4 read and project v0/v1.  The
  // host checks all four registers first, so this cannot pass by accidentally
  // manufacturing the projected values after a bad read.
  std::uint32_t x4[kX4Registers] = {};
  ProdSwzlRead::copy(x4, storage.prod_swzl, 0, 0, 0, 0);
#pragma unroll
  for (int v = 0; v < kX4Registers; ++v) {
    prod_x4_output[(test_case * kWarp + lane) * kX4Registers + v] = x4[v];
  }

  // SAME guard cube, SAME x4-swzl helper, SAME four-register delivery.  The
  // only changed inputs are the address coordinates.  The planted formula is
  // copied algebraically from NVIDIA's tile<8,8> loader:
  //
  //   row      = lane % I
  //   word_col = ((lane / I) * (J/2)) % J       I=J=8
  //
  // `nvidia_word` is in uint32 words and GuardSwzlRead's coord_w is in fp16
  // elements, hence the explicit factor two.  The 32-row guard makes every x4
  // access valid even after adding this per-lane row coordinate.
  constexpr int kI = 8;
  constexpr int kJ = 8;       // uint32 columns: 16 scalar fp16 columns
  int const nvidia_row = lane % kI;
  int const nvidia_word = ((lane / kI) * (kJ / 2)) % kJ;

  std::uint32_t guard_good[kX4Registers] = {};
  std::uint32_t guard_bad[kX4Registers] = {};
  g2_guard_swzl_x4(guard_good, storage.guard_swzl, 0, 0);
  g2_guard_swzl_x4(
      guard_bad, storage.guard_swzl, 2 * nvidia_word, nvidia_row);
#pragma unroll
  for (int v = 0; v < kX4Registers; ++v) {
    int const out = (test_case * kWarp + lane) * kX4Registers + v;
    guard_good_output[out] = guard_good[v];
    guard_bad_output[out] = guard_bad[v];
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

constexpr std::uint16_t guard_tag(int row, int k) {
  // 0x2000..0x27ff: all 2048 guard values are unique and disjoint from
  // production's upper tags and both poison sets.
  return std::uint16_t(0x2000u + unsigned(row * kCubeW + k));
}

std::uint16_t halfword(std::uint32_t word, int h) {
  return std::uint16_t((word >> (16 * h)) & 0xffffu);
}

}  // namespace

int run_ppu_m8n16_g2() {
  std::printf("== ppu001 m8n16 G2: AIU physical-cube delivery and planted address defect ==\n");
  std::printf("[G2-path] production write=ppu.cp.async.aiu...padz.swzl.2d.b16 cube=16x64 "
              "read=ppu.tc01.ldmatrix...m8n8.x4.swzl.shared.b16 project=v0,v1\n");
  std::printf("[G2-control-path] same-op=PPU0010_TSM_LD_SWZL<m8n8.x4.swzl> "
              "cube=32x64 same-base=guard_swzl only-delta=coordinates\n");

  std::vector<std::uint16_t> prod_input(kCases * kCubeElements);
  for (int c = 0; c < kCases; ++c) {
    for (int row = 0; row < kCubeH; ++row) {
      for (int k = 0; k < kCubeW; ++k) {
        prod_input[(c * kCubeH + row) * kCubeW + k] =
            row < 8 ? upper_tag(row, k) : lower_tag(c, row, k);
      }
    }
  }
  std::vector<std::uint16_t> guard_input(kGuardElements);
  for (int row = 0; row < kGuardH; ++row) {
    for (int k = 0; k < kCubeW; ++k) {
      guard_input[row * kCubeW + k] = guard_tag(row, k);
    }
  }

  cutlass::DeviceAllocation<std::uint16_t> d_prod_input(prod_input.size());
  cutlass::DeviceAllocation<std::uint16_t> d_guard_input(guard_input.size());
  cutlass::DeviceAllocation<std::uint32_t> d_prod_x4(
      kCases * kWarp * kX4Registers);
  cutlass::DeviceAllocation<std::uint32_t> d_guard_good(
      kCases * kWarp * kX4Registers);
  cutlass::DeviceAllocation<std::uint32_t> d_guard_bad(
      kCases * kWarp * kX4Registers);
  d_prod_input.copy_from_host(prod_input.data());
  d_guard_input.copy_from_host(guard_input.data());

  g2_device<<<kCases, kWarp>>>(d_prod_input.get(), d_guard_input.get(),
      d_prod_x4.get(), d_guard_good.get(), d_guard_bad.get());
  CUTLASS_PPU_CHECK(hggcGetLastError());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<std::uint32_t> prod_x4(kCases * kWarp * kX4Registers);
  std::vector<std::uint32_t> guard_good(kCases * kWarp * kX4Registers);
  std::vector<std::uint32_t> guard_bad(kCases * kWarp * kX4Registers);
  d_prod_x4.copy_to_host(prod_x4.data());
  d_guard_good.copy_to_host(guard_good.data());
  d_guard_bad.copy_to_host(guard_bad.data());

  int x4_bad = 0;
  int guard_x4_bad = 0;
  int guard_bad_map_bad = 0;
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
        std::uint32_t const got = prod_x4[(c * kWarp + lane) * kX4Registers + v];
        int const row = lane / 4 + 8 * (v / 2);
        int const word = lane % 4 + 4 * (v % 2);
        for (int h = 0; h < kHalfsPerRegister; ++h) {
          std::uint16_t const want =
              prod_input[(c * kCubeH + row) * kCubeW + 2 * word + h];
          x4_bad += halfword(got, h) != want;

          std::uint32_t const guard_got =
              guard_good[(c * kWarp + lane) * kX4Registers + v];
          std::uint16_t const guard_want =
              guard_input[row * kCubeW + 2 * word + h];
          guard_x4_bad += halfword(guard_got, h) != guard_want;

          // Independent golden for the planted arm.  Unlike the origin
          // comparison below, this proves that every nonzero coordinate
          // returns the exact valid tag it names -- not clamp data, poison,
          // or arbitrary bits that merely differ from the origin.  The
          // direct logical map is the NVIDIA row/word offset applied to the
          // already-validated x4 delivery map.
          int const bad_row = (lane % 8) + row;
          int const bad_k = 2 * (((lane / 8) * (8 / 2)) % 8) +
              2 * word + h;
          std::uint32_t const guard_bad_got =
              guard_bad[(c * kWarp + lane) * kX4Registers + v];
          std::uint16_t const guard_bad_want =
              guard_input[bad_row * kCubeW + bad_k];
          guard_bad_map_bad +=
              halfword(guard_bad_got, h) != guard_bad_want;
        }
      }
    }
  }

  for (int lane = 0; lane < kWarp; ++lane) {
    for (int v = 0; v < kX4Registers; ++v) {
      std::uint32_t const a = prod_x4[(0 * kWarp + lane) * kX4Registers + v];
      std::uint32_t const b = prod_x4[(1 * kWarp + lane) * kX4Registers + v];
      for (int h = 0; h < kHalfsPerRegister; ++h) {
        if (v < kM8Registers) {
          projected_changed += halfword(a, h) != halfword(b, h);
        } else {
          lower_changed += halfword(a, h) != halfword(b, h);
        }
      }
    }
  }

  // Compare the same-op/bad-coordinate result with the independently frozen
  // m8 ALayout. Only case 0 is needed: this is a planted-fault sensitivity
  // check, not another implementation candidate. Lanes 0 and 16 are a local
  // guard: the NVIDIA formula gives both (coord_w,coord_h)=(0,0), so their
  // complete x4 result must still equal the control-good arm. A broken bad arm
  // cannot pass merely by returning arbitrary bits everywhere.
  int red_mismatches = 0;
  int red_zero_coord_bad = 0;
  int zero_coord_lanes = 0;
  for (int lane = 0; lane < kWarp; ++lane) {
    int const nvidia_row = lane % 8;
    int const nvidia_word = ((lane / 8) * (8 / 2)) % 8;
    bool const zero_coord = nvidia_row == 0 && nvidia_word == 0;
    zero_coord_lanes += zero_coord;
    for (int reg = 0; reg < kM8Registers; ++reg) {
      std::uint32_t const got = guard_bad[lane * kX4Registers + reg];
      int const row = lane / 4;
      for (int h = 0; h < kHalfsPerRegister; ++h) {
        int const k = 2 * (lane % 4) + h + 8 * reg;
        std::uint16_t const want = guard_input[row * kCubeW + k];
        red_mismatches += halfword(got, h) != want;
      }
    }
    if (zero_coord) {
      for (int c = 0; c < kCases; ++c) {
        for (int v = 0; v < kX4Registers; ++v) {
          std::uint32_t const good =
              guard_good[(c * kWarp + lane) * kX4Registers + v];
          std::uint32_t const bad =
              guard_bad[(c * kWarp + lane) * kX4Registers + v];
          for (int h = 0; h < kHalfsPerRegister; ++h) {
            red_zero_coord_bad += halfword(good, h) != halfword(bad, h);
          }
        }
      }
    }
  }

  int const green_mismatches =
      x4_bad + guard_x4_bad + projected_changed + (kLowerValues - lower_changed);
  constexpr int kExpectedRedMismatches =
      kProjectedValues - 2 * kM8Registers * kHalfsPerRegister;
  bool const green_pass = green_mismatches == 0;
  bool const red_pass =
      red_mismatches == kExpectedRedMismatches &&
      guard_bad_map_bad == 0 && zero_coord_lanes == 2 &&
      red_zero_coord_bad == 0;

  std::printf("[G2-green-detail] x4_values=%d x4_bad=%d projected_changed=%d/%d "
              "lower_poison_changed=%d/%d guard_x4_values=%d guard_x4_bad=%d\n",
              kCases * kWarp * kX4Registers * kHalfsPerRegister,
              x4_bad, projected_changed, kProjectedValues,
              lower_changed, kLowerValues,
              kCases * kWarp * kX4Registers * kHalfsPerRegister,
              guard_x4_bad);
  std::printf("[G2-green] mismatches=%d %s\n",
              green_mismatches, green_pass ? "PASS" : "FAIL");
  std::printf("[G2-negative-detail] same_op=x4-swzl bad_map_values=%d "
              "bad_map_bad=%d zero_coord_lanes=%d zero_coord_bad=%d "
              "red_expected=%d/%d\n",
              kCases * kWarp * kX4Registers * kHalfsPerRegister,
              guard_bad_map_bad, zero_coord_lanes, red_zero_coord_bad,
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

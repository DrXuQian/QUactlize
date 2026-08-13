// L178 -- exhaustive production-bound oracle for standalone Marlin address-state hoisting.
//
// The production helpers are exercised directly, but their answers are checked
// against two independently written coordinate systems:
//   * the scalar pointer equations in local marlin_classic_ppu.cuh; and
//   * Awesome-CuTe's row-major A, [K/16,N/64,32,32] B, and [K/gs,N] scale
//     tensors plus its G2S thread layouts.
// No expected address calls a production transform/rebase helper.

// The fixed target has 256 threads, 32 output tiles and 32 K tiles.  For every
// (thread,q,k) this oracle also sweeps every legal segment count [1,32-k].

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"

using Tile = cute::Shape<cute::_16, cute::_128, cute::_128>;
using Warp = cute::Shape<cute::_16, cute::_64, cute::_32>;
using Main = cutlass::gemm::collective::MarlinCollectivePPU<
    Tile, Warp, 4, 128, cute::Stride<int64_t, cute::_1, int64_t>,
    cute::Stride<int64_t, cute::_1, int64_t>,
    cute::Stride<cute::_1, int64_t, int64_t>>;
using Vector128 = cutlass::gemm::collective::marlin_ppu_detail::Vector128;

namespace {

constexpr int kM = 1;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr int kNTiles = kN / Main::TileN;
constexpr int kKTiles = kK / Main::TileK;
constexpr int kVectorsPerBTileN = (Main::TileN / 64) * 32;
constexpr int kCodesPerVector = 32;

struct Work {
  int N_idx = -1;
  int K_idx = 0;
  int k_tile_count = 0;
  bool valid = true;
  constexpr bool is_valid() const { return valid; }
};

struct AddressReference {
  std::ptrdiff_t a_thread = 0;
  std::ptrdiff_t b_thread = 0;
  std::ptrdiff_t scale_thread = 0;
  int b_inner_delta = 0;
  int b_k_delta = 0;
  int scale_k_delta = 0;
  int a_smem_write = 0;
  std::array<int, Main::BInnerIters> a_smem_read{};
  int scale_smem_read = 0;
  bool a_copy_pred = false;
  bool scale_copy_pred = false;
  std::ptrdiff_t a = 0;
  std::array<std::ptrdiff_t, Main::BInnerIters> b{};
  std::ptrdiff_t scale = 0;
};

char const* option(int argc, char** argv, char const* prefix) {
  std::size_t const n = std::strlen(prefix);
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], prefix, n) == 0) return argv[i] + n;
  }
  return nullptr;
}

std::string read_file(char const* path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

bool contains(std::string const& text, char const* token) {
  return text.find(token) != std::string::npos;
}

int fail(char const* plant, char const* reason) {
  std::fprintf(stderr,
               "[l178:red] plant=%s caught=1 reason=%s result=RED\n",
               plant, reason);
  return 1;
}

// Independent spelling of classic's XOR transform.  Do not replace this with
// Main::transform_a_index: the point is to catch a production-map regression.
constexpr int reference_a_swizzle(int index) {
  int const row = index / 16;
  return 16 * row + ((index % 16) ^ row);
}

// Direct transcription of local PPU classic's scalar equations, in Vector128
// units.  B is int4: one Vector128 is 32 logical codes, not 16 bytes of codes.
AddressReference classic_reference(int tid, int q, int k) {
  AddressReference r;
  int const a_stride = kK / 8;
  int const b_stride = 16 * kN / 32;
  int const a_sh_write = 16 * (tid / 16) + tid % 16;
  int const a_sh_read =
      16 * ((tid % 32) % 16) + (tid % 32) / 16 +
      2 * ((tid / 32) / 2);
  r.a_thread = a_stride * (tid / 16) + tid % 16;
  r.b_thread = b_stride * (tid / 64) + tid % 64;
  r.scale_thread = tid;
  r.b_inner_delta = b_stride * (Main::Threads / Main::BSharedStride);
  r.b_k_delta = b_stride * Main::KBlocks;
  r.scale_k_delta = kN / 8;
  r.a_smem_write = reference_a_swizzle(a_sh_write);
  for (int i = 0; i < Main::BInnerIters; ++i) {
    r.a_smem_read[i] =
        reference_a_swizzle(Main::ASharedReadOuter * i + a_sh_read);
  }
  int const warp_n = (tid / 32) % 2;
  int const lane = tid % 32;
  r.scale_smem_read = 8 * warp_n + lane / 4;
  r.a_copy_pred = a_sh_write < 16 * kM;
  r.scale_copy_pred = tid < 16;
  r.a = r.a_thread + 16 * k;
  for (int i = 0; i < Main::BInnerIters; ++i) {
    r.b[i] = r.b_thread + 64 * q + r.b_k_delta * k +
             r.b_inner_delta * i;
  }
  r.scale = r.scale_thread + 16 * q + r.scale_k_delta * k;
  return r;
}

// The same addresses reconstructed from Awesome-CuTe tensor coordinates,
// rather than from classic's pointer deltas.  A vector owns 8 half values;
// a B vector owns the final 32-code mode of [K/16,N/64,32,32]; scale owns
// 8 adjacent N columns of [K/gs,N].
AddressReference awesome_reference(int tid, int q, int k) {
  AddressReference r;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const warp_n = warp % 2;
  int const warp_k = warp / 2;
  int const a_row = tid / 16;
  int const a_k_vector = tid % 16;
  int const b_n64 = 2 * q + warp_n;
  int const b_k16 = 8 * k + warp_k;
  r.a_thread = a_row * (kK / 8) + a_k_vector;
  r.b_thread = (warp_k * (kN / 64) + warp_n) * 32 + lane;
  r.scale_thread = tid;
  r.b_inner_delta = 4 * (kN / 64) * 32;
  r.b_k_delta = 8 * (kN / 64) * 32;
  r.scale_k_delta = kN / 8;
  int const a_sh_write = 16 * a_row + a_k_vector;
  int const a_sh_read = 16 * (lane % 16) + lane / 16 + 2 * warp_k;
  r.a_smem_write = reference_a_swizzle(a_sh_write);
  r.a_smem_read[0] = reference_a_swizzle(a_sh_read);
  r.a_smem_read[1] = reference_a_swizzle(8 + a_sh_read);
  r.scale_smem_read = 8 * warp_n + lane / 4;
  r.a_copy_pred = a_row < kM;
  r.scale_copy_pred = tid < kN / 256;
  r.a = r.a_thread + k * (Main::TileK / 8);
  for (int i = 0; i < Main::BInnerIters; ++i) {
    int const global_k16 = b_k16 + 4 * i;
    r.b[i] = (global_k16 * (kN / 64) + b_n64) * 32 + lane;
  }
  r.scale = (k * (kN / 8)) + q * (Main::TileN / 8) + tid;
  return r;
}

bool same_reference(AddressReference const& a, AddressReference const& b) {
  return a.a_thread == b.a_thread && a.b_thread == b.b_thread &&
         a.scale_thread == b.scale_thread &&
         a.b_inner_delta == b.b_inner_delta &&
         a.b_k_delta == b.b_k_delta &&
         a.scale_k_delta == b.scale_k_delta &&
         a.a_smem_write == b.a_smem_write &&
         a.a_smem_read == b.a_smem_read &&
         a.scale_smem_read == b.scale_smem_read &&
         a.a_copy_pred == b.a_copy_pred &&
         a.scale_copy_pred == b.scale_copy_pred && a.a == b.a &&
         a.b == b.b && a.scale == b.scale;
}

}  // namespace

int main(int argc, char** argv) {
  char const* plant = option(argc, argv, "--plant=");
  if (plant == nullptr) plant = "none";
  char const* classic_path = option(argc, argv, "--classic=");
  char const* awesome_path = option(argc, argv, "--awesome=");
  if (classic_path == nullptr || awesome_path == nullptr) {
    return fail(plant, "classic and Awesome-CuTe source paths are required");
  }
  std::string const classic_source = read_file(classic_path);
  std::string const awesome_source = read_file(awesome_path);
  for (char const* token : {
           "int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o)",
           "b_gl_rd += b_sh_stride * slice_col",
           "b_gl_rd += b_gl_rd_delta_o * slice_row",
           "int s_gl_rd = (group_blocks != -1)",
           "const int s_sh_rd = 8 * ((threadIdx.x / 32)",
       }) {
    if (!contains(classic_source, token)) {
      return fail(plant, "local classic source anchor drifted");
    }
  }
  for (char const* token : {
           "make_shape(k / _16{}, n / _64{}, _32{}, _32{})",
           "using G2SBThrLayout = decltype(make_layout(",
           "auto g2s_tBgB_copy = thr_g2s_b_copy.partition_S(gB)",
           "s2r_sScale_tile_idx =",
           "_8{} * ((tidx >> 5) % (kMmaThrLayoutN))",
       }) {
    if (!contains(awesome_source, token)) {
      return fail(plant, "Awesome-CuTe source anchor drifted");
    }
  }

  static_assert(kVectorsPerBTileN == Main::BSharedStride);
  static_assert(kCodesPerVector == 32);
  std::vector<Vector128> a(8192);
  std::vector<Vector128> b(524288);
  // Non-writer threads still carry a final pointer.  Pad the allocation so
  // host pointer arithmetic remains defined while the copy predicate is false.
  std::vector<Vector128> scale(16624);
  typename Main::Params params{
      reinterpret_cast<cutlass::half_t const*>(a.data()),
      reinterpret_cast<cutlass::int4b_t const*>(b.data()),
      reinterpret_cast<cutlass::half_t const*>(scale.data())};

  typename Main::SharedStorage shared{};
  auto shared_bases = Main::make_shared_bases(shared);
  if (reinterpret_cast<char*>(shared_bases.a) -
              reinterpret_cast<char*>(shared.storage) != 0 ||
      reinterpret_cast<char*>(shared_bases.b) -
              reinterpret_cast<char*>(shared.storage) != 16384 ||
      reinterpret_cast<char*>(shared_bases.scale) -
              reinterpret_cast<char*>(shared.storage) != 49152) {
    return fail(plant, "SharedBases byte offsets differ from classic stage ledger");
  }

  std::uint64_t state_checks = 0;
  std::uint64_t qk_checks = 0;
  std::uint64_t segment_checks = 0;
  for (int tid = 0; tid < Main::Threads; ++tid) {
    auto state = Main::init_cta_state(params, kM, kN, kK, tid);
    AddressReference classic = classic_reference(tid, 0, 0);
    AddressReference awesome = awesome_reference(tid, 0, 0);
    if (!same_reference(classic, awesome)) {
      return fail(plant, "independent classic and Awesome-CuTe equations disagree");
    }

    if (std::strcmp(plant, "b-pitch-codes") == 0) {
      state.b_thread_base += (kCodesPerVector - 1) * (tid % 64);
    } else if (std::strcmp(plant, "b-inner-n-cohort") == 0) {
      state.b_inner_delta += 16 * kN / 32;
    } else if (std::strcmp(plant, "b-k-missing-wk") == 0) {
      state.b_k_delta /= Main::WarpOnK;
    } else if (std::strcmp(plant, "scale-k-byte-half") == 0) {
      state.scale_k_delta /= 2;
    } else if (std::strcmp(plant, "tight-a-smem") == 0) {
      int const logical = 16 * (tid / 16) + tid % 16;
      int const row = logical / 8;
      state.a_smem_write = 8 * row + ((logical % 8) ^ row);
    } else if (std::strcmp(plant, "a-predicate-all-threads") == 0) {
      state.a_copy_pred = tid < Main::Threads;
    } else if (std::strcmp(plant, "scale-predicate-cohort") == 0) {
      state.scale_copy_pred = tid < 32;
    }

    if (state.tid != tid ||
        state.a_thread_base - a.data() != classic.a_thread ||
        state.b_thread_base - b.data() != classic.b_thread ||
        state.scale_thread_base - scale.data() != classic.scale_thread ||
        state.b_inner_delta != classic.b_inner_delta ||
        state.b_k_delta != classic.b_k_delta ||
        state.scale_k_delta != classic.scale_k_delta ||
        state.a_smem_write != classic.a_smem_write ||
        state.a_smem_read[0] != classic.a_smem_read[0] ||
        state.a_smem_read[1] != classic.a_smem_read[1] ||
        state.scale_smem_read != classic.scale_smem_read ||
        state.a_copy_pred != classic.a_copy_pred ||
        state.scale_copy_pred != classic.scale_copy_pred) {
      return fail(plant, "production CTA invariants diverged from both references");
    }
    ++state_checks;

    for (int q = 0; q < kNTiles; ++q) {
      for (int k = 0; k < kKTiles; ++k) {
        AddressReference const c = classic_reference(tid, q, k);
        AddressReference const aw = awesome_reference(tid, q, k);
        if (!same_reference(c, aw)) {
          return fail(plant, "independent q/K address equations disagree");
        }
        ++qk_checks;
        for (int count = 1; count <= kKTiles - k; ++count) {
          Work work{q, k, count, true};
          auto segment = Main::rebase_segment(state, work);
          if (std::strcmp(plant, "a-k-missing-step") == 0) {
            segment.a -= Main::AGlobalOuter * k;
          } else if (std::strcmp(plant, "q-local-not-global") == 0) {
            int const local_q = q % Main::WarpOnN;
            int const q_delta = q - local_q;
            for (auto& pointer : segment.b) {
              pointer -= Main::BSharedStride * q_delta;
            }
            segment.scale -= Main::ScaleSharedStride * q_delta;
          }
          bool b_matches = true;
          for (int i = 0; i < Main::BInnerIters; ++i) {
            b_matches = b_matches && segment.b[i] - b.data() == c.b[i];
          }
          if (segment.k_tiles_remaining != count ||
              segment.a - a.data() != c.a || !b_matches ||
              segment.scale - scale.data() != c.scale) {
            return fail(plant, "production rebased pointers diverged from both references");
          }
          ++segment_checks;
        }
      }
    }
  }

  if (state_checks != 256 || qk_checks != 256ull * 32 * 32 ||
      segment_checks != 256ull * 32 * 528) {
    return fail(plant, "exhaustive census is incomplete");
  }
  if (std::strcmp(plant, "none") != 0) {
    return fail(plant, "named plant did not perturb its intended invariant");
  }
  std::printf(
      "[l178] PASS: cta=%llu qk=%llu legal-segments=%llu "
      "classic=exact awesome-cute=exact shared-bytes={A:0 B:16384 S:49152} "
      "address-units={A:half8 B:int4x32 S:half8}\n",
      static_cast<unsigned long long>(state_checks),
      static_cast<unsigned long long>(qk_checks),
      static_cast<unsigned long long>(segment_checks));
  return 0;
}

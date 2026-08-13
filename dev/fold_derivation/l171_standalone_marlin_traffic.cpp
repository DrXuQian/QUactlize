// L171 -- production-bound byte ledger for the fixed standalone Marlin PPU mainloop.
//
// This oracle deliberately counts logical global-to-shared transfers, not cache
// transactions.  The box postcondition is KVD -> TSM ~= 8.91 MB; cache-line
// amplification remains a device measurement.  All counts below are derived
// from the production collective constants and are independently anchored to
// the corresponding Awesome-CuTe and PPU-classic source formulas.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"

using Main = cutlass::gemm::collective::MarlinCollectivePPU<
    cute::Shape<cute::_16, cute::_128, cute::_128>,
    cute::Shape<cute::_16, cute::_64, cute::_32>, 4, 128,
    void, void, void>;

namespace {

using u64 = std::uint64_t;

constexpr u64 kM = 1;
constexpr u64 kN = 4096;
constexpr u64 kK = 4096;
constexpr u64 kVectorBytes = 16;
constexpr u64 kTilesN = kN / Main::TileN;
constexpr u64 kTilesK = kK / Main::TileK;
constexpr u64 kCells = kTilesN * kTilesK;
constexpr u64 kValidRows = kM;

constexpr u64 kAVectors = Main::ASharedStride * kValidRows;
constexpr u64 kBVectors = Main::Threads * Main::BInnerIters;
constexpr u64 kScaleVectors = Main::ScaleSharedStride;
constexpr u64 kABytes = kAVectors * kVectorBytes;
constexpr u64 kBBytes = kBVectors * kVectorBytes;
constexpr u64 kScaleBytes = kScaleVectors * kVectorBytes;
constexpr u64 kCellBytes = kABytes + kBBytes + kScaleBytes;
constexpr u64 kLaunchBytes = kCellBytes * kCells;

static_assert(Main::TileM == 16 && Main::TileN == 128 && Main::TileK == 128);
static_assert(Main::WarpM == 16 && Main::WarpN == 64 && Main::WarpK == 32);
static_assert(Main::WarpOnM == 1 && Main::WarpOnN == 2 && Main::WarpOnK == 4);
static_assert(Main::Threads == 256 && Main::Stages == 4 && Main::GroupSize == 128);
static_assert(Main::ASharedStride == 16 && Main::ASharedStage == 256);
static_assert(Main::BSharedStride == 64 && Main::BSharedStage == 512 &&
              Main::BInnerIters == 2);
static_assert(Main::ScaleSharedStride == 16 && Main::ScaleSharedStage == 16);
static_assert(kTilesN == 32 && kTilesK == 32 && kCells == 1024);
static_assert(kAVectors == 16 && kBVectors == 512 && kScaleVectors == 16);
static_assert(kABytes == 256 && kBBytes == 8192 && kScaleBytes == 256);
static_assert(kCellBytes == 8704 && kLaunchBytes == 8912896);

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
  std::fprintf(stderr, "[l171:red] plant=%s caught=1 reason=%s result=RED\n",
               plant, reason);
  return 1;
}

char const* option(int argc, char** argv, char const* prefix) {
  std::size_t const n = std::strlen(prefix);
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], prefix, n) == 0) {
      return argv[i] + n;
    }
  }
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  char const* plant = option(argc, argv, "--plant=");
  if (plant == nullptr) plant = "none";
  char const* collective_path = option(argc, argv, "--collective=");
  char const* awesome_path = option(argc, argv, "--awesome=");
  char const* classic_path = option(argc, argv, "--classic=");
  char const* kernel_path = option(argc, argv, "--kernel=");
  if (collective_path == nullptr || awesome_path == nullptr ||
      classic_path == nullptr || kernel_path == nullptr) {
    return fail(plant, "four source paths are required");
  }

  std::string const collective = read_file(collective_path);
  std::string const awesome = read_file(awesome_path);
  std::string const classic = read_file(classic_path);
  std::string const kernel = read_file(kernel_path);
  if (collective.empty() || awesome.empty() || classic.empty() || kernel.empty()) {
    return fail(plant, "one source anchor could not be read");
  }

  // Production anchors: one predicate-scoped copy point for each plane.  The
  // stage ring changes only the destination buffer; it is not a transfer
  // multiplier.  The M residue predicate, two B writes/thread and 16 scale
  // writers determine the byte count above.
  for (char const* token : {
           "if (predicate) {",
           "logical < ASharedStride * (problem_m - m_tile * TileM)",
           "for (int i = 0; i < BInnerIters; ++i)",
           "&b_stage[Threads * i + tid], b_pointer[i]",
           "if (tid < ScaleSharedStride)",
           "&smem_scale[ScaleSharedStage * pipe + tid]",
           "TileK == GroupSize",
       }) {
    if (!contains(collective, token)) {
      return fail(plant, "production copy anchor drifted");
    }
  }

  // Independent reference anchors.  These are formulas and load sites from
  // two separate implementations, not a second copy of the numeric ledger.
  for (char const* token : {
           "using G2SAcopyOp = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>",
           "using G2SBCopyOp = SM80_CP_ASYNC_CACHEGLOBAL_EVICT<cute::uint128_t>",
           "using G2SScaleCopyAtom = Copy_Atom<G2SBCopyTraits, Atype>",
           "copy(g2s_b_copy, g2s_tBgB_copy",
           "copy_if(\n                g2s_s_copy",
       }) {
    if (!contains(awesome, token)) {
      return fail(plant, "Awesome-CuTe copy anchor drifted");
    }
  }
  for (char const* token : {
           "constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks)",
           "constexpr int b_sh_stage = b_sh_stride * thread_k_blocks",
           "constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta",
           "bool s_sh_wr_pred = threadIdx.x < s_sh_stride",
           "cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i])",
       }) {
    if (!contains(classic, token)) {
      return fail(plant, "PPU-classic copy anchor drifted");
    }
  }

  std::size_t const handoff_begin = kernel.find("static void global_handoff(");
  std::size_t const handoff_end = kernel.find("static void write_result(", handoff_begin);
  if (handoff_begin == std::string::npos || handoff_end == std::string::npos ||
      handoff_end <= handoff_begin) {
    return fail(plant, "standalone handoff source scope drifted");
  }
  std::string const handoff = kernel.substr(handoff_begin, handoff_end - handoff_begin);
  for (char const* token : {
           "accum(index) += __half2float(d[offset])",
           "d[offset] = __float2half(accum(index))",
       }) {
    if (!contains(handoff, token)) {
      return fail(plant, "standalone scalar D-chain anchor drifted");
    }
  }
  if (contains(handoff, "cp_async")) {
    return fail(plant, "standalone D-chain unexpectedly contributes KVD-to-TSM traffic");
  }

  u64 a = kABytes;
  u64 b = kBBytes;
  u64 s = kScaleBytes;
  if (std::strcmp(plant, "a-per-k-cohort") == 0) {
    a *= Main::WarpOnK;
  } else if (std::strcmp(plant, "b-two-source-duplicate") == 0) {
    b *= 2;
  } else if (std::strcmp(plant, "stage-ring-refill") == 0) {
    a *= Main::Stages;
    b *= Main::Stages;
    s *= Main::Stages;
  } else if (std::strcmp(plant, "scale-all-threads") == 0) {
    s = Main::Threads * kVectorBytes;
  } else if (std::strcmp(plant, "none") != 0) {
    return fail(plant, "unknown plant");
  }

  u64 const cell = a + b + s;
  u64 const launch = cell * kCells;
  if (cell != kCellBytes || launch != kLaunchBytes) {
    return fail(plant, "logical global-to-shared byte ledger changed");
  }

  std::printf(
      "[l171] PASS: cells=%llu per-cell={A:%llu B:%llu scale:%llu total:%llu} "
      "launch=%llu per-N-warp={A:%llu B:%llu scale:%llu} "
      "per-K-cohort={A:%llu B:%llu scale:%llu} "
      "anchors=production+awesome-cute+ppu-classic model=logical-G2S\n",
      static_cast<unsigned long long>(kCells),
      static_cast<unsigned long long>(kABytes),
      static_cast<unsigned long long>(kBBytes),
      static_cast<unsigned long long>(kScaleBytes),
      static_cast<unsigned long long>(kCellBytes),
      static_cast<unsigned long long>(kLaunchBytes),
      static_cast<unsigned long long>(kABytes / Main::WarpOnN),
      static_cast<unsigned long long>(kBBytes / Main::WarpOnN),
      static_cast<unsigned long long>(kScaleBytes / Main::WarpOnN),
      static_cast<unsigned long long>(kABytes / Main::WarpOnK),
      static_cast<unsigned long long>(kBBytes / Main::WarpOnK),
      static_cast<unsigned long long>(kScaleBytes / Main::WarpOnK));
  return 0;
}

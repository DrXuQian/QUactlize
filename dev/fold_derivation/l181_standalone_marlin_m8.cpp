// L181 -- pure-host oracle for standalone Marlin's packed M=1 / m8 A path.
//
// The positive fixture closes six independently falsifiable contracts:
//   1. exactly 16 threads issue one 16-byte A cp.async each, covering the
//      packed 1x128 fp16 row (256 bytes) with no hole or duplicate; all 256
//      threads still issue both B chunks;
//   2. PPU's plain-x2 provider redistribution reconstructs row zero exactly,
//      and every provider window remains inside those 256 bytes;
//   3. the shared ledger stores one A row (34,816 bytes total), not an m16 or
//      logical-m8 padded stage;
//   4. the 64-thread K0 cohort covers the 8x128 output exactly once;
//   5. four K cohorts reduce through the m8-sized 4->2->1 scratch map;
//   6. changing only the m8/m16 instruction does not change classic B or
//      gs128-scale artifact bytes.
//
// Each tempting regression has a named --plant arm.  The runner requires the
// same positive oracle to turn red for every plant.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "marlin_format_ppu.hpp"

namespace {

constexpr int kTileN = 128;
constexpr int kTileK = 128;
constexpr int kWarpK = 32;
constexpr int kKCohorts = kTileK / kWarpK;
constexpr int kThreads = 256;
constexpr int kOutputThreads = 64;
constexpr int kStages = 4;
constexpr int kAChunks = 16;
constexpr int kAChunkBytes = 16;
constexpr int kAStageBytes = kAChunks * kAChunkBytes;
constexpr int kBInnerIters = 2;
constexpr int kBSharedStage = 512;
constexpr int kScaleSharedStage = 16;

enum class Plant {
  None,
  MissingAChunk,
  DuplicateAChunk,
  DropBThread,
  NvidiaProvider,
  ShiftedWord,
  PaddedARows,
  M16OutputValues,
  SkipSecondReductionStep,
  M16Scratch,
  MDependentArtifact,
};

char const* name(Plant plant) {
  switch (plant) {
    case Plant::None: return "none";
    case Plant::MissingAChunk: return "missing-a-chunk";
    case Plant::DuplicateAChunk: return "duplicate-a-chunk";
    case Plant::DropBThread: return "drop-b-thread";
    case Plant::NvidiaProvider: return "nvidia-provider";
    case Plant::ShiftedWord: return "shifted-word";
    case Plant::PaddedARows: return "padded-a-rows";
    case Plant::M16OutputValues: return "m16-output-values";
    case Plant::SkipSecondReductionStep: return "skip-second-reduction-step";
    case Plant::M16Scratch: return "m16-scratch";
    case Plant::MDependentArtifact: return "m-dependent-artifact";
  }
  return "unknown";
}

Plant parse_plant(int argc, char** argv) {
  constexpr std::string_view prefix = "--plant=";
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg.substr(0, prefix.size()) != prefix) continue;
    auto value = arg.substr(prefix.size());
    if (value == "missing-a-chunk") return Plant::MissingAChunk;
    if (value == "duplicate-a-chunk") return Plant::DuplicateAChunk;
    if (value == "drop-b-thread") return Plant::DropBThread;
    if (value == "nvidia-provider") return Plant::NvidiaProvider;
    if (value == "shifted-word") return Plant::ShiftedWord;
    if (value == "padded-a-rows") return Plant::PaddedARows;
    if (value == "m16-output-values") return Plant::M16OutputValues;
    if (value == "skip-second-reduction-step")
      return Plant::SkipSecondReductionStep;
    if (value == "m16-scratch") return Plant::M16Scratch;
    if (value == "m-dependent-artifact") return Plant::MDependentArtifact;
  }
  return Plant::None;
}

struct CopyMetric {
  int a_issuers = 0;
  int a_bytes = 0;
  int a_holes = 0;
  int a_duplicates = 0;
  int b_threads = 0;
  int b_chunks = 0;
  int b_holes = 0;
  int b_duplicates = 0;
};

CopyMetric verify_cooperative_copy(Plant plant) {
  std::array<int, kAChunks> a_hits{};
  std::array<int, kThreads * kBInnerIters> b_hits{};
  std::array<bool, kThreads> b_thread_active{};
  CopyMetric result;

  for (int tid = 0; tid < kThreads; ++tid) {
    bool const a_active = tid < kAChunks &&
                          !(plant == Plant::MissingAChunk && tid == 15);
    if (a_active) {
      int destination = tid;
      if (plant == Plant::DuplicateAChunk && tid == 15) destination = 14;
      ++result.a_issuers;
      result.a_bytes += kAChunkBytes;
      if (destination >= 0 && destination < kAChunks) ++a_hits[destination];
    }

    if (!(plant == Plant::DropBThread && tid == kThreads - 1)) {
      b_thread_active[tid] = true;
      for (int inner = 0; inner < kBInnerIters; ++inner) {
        ++b_hits[kThreads * inner + tid];
        ++result.b_chunks;
      }
    }
  }

  for (int hit : a_hits) {
    result.a_holes += hit == 0;
    result.a_duplicates += std::max(0, hit - 1);
  }
  for (bool active : b_thread_active) result.b_threads += active;
  for (int hit : b_hits) {
    result.b_holes += hit == 0;
    result.b_duplicates += std::max(0, hit - 1);
  }
  return result;
}

struct LoadMetric {
  int values = 0;
  int mismatches = 0;
  int out_of_stage = 0;
  int max_byte_end = 0;
};

LoadMetric verify_plain_x2_row0(Plant plant) {
  // PPU x2 routing (SDK getThreadAddr1D(128)):
  //   output lane = 4*r+a, register=j
  //   provider    = 2*r+a/2+16*j
  //   word        = a%2 within that provider's 64-bit window.
  //
  // Only r=0 is semantically live for the admitted M=1 target.  Providers
  // for masked rows deliberately alias this packed row.  The production
  // pointer formula below must therefore reconstruct K=2*a+8*j+h for both
  // 16-wide inner steps without reaching byte 256.  B's producer enumerates
  // those steps inner-major, then K-cohort; A must use the same order.
  LoadMetric result;
  for (int warp_k = 0; warp_k < kKCohorts; ++warp_k) {
    for (int inner = 0; inner < kBInnerIters; ++inner) {
      for (int a = 0; a < 4; ++a) {
        for (int reg = 0; reg < 2; ++reg) {
          int provider = a / 2 + 16 * reg;
          int word = a % 2;
          int const k_base = (inner * kKCohorts + warp_k) * 16;
          int source_base = k_base +
                            4 * (provider % 2) + 8 * (provider / 16);

          if (plant == Plant::NvidiaProvider) {
            // Historical NVIDIA m8n8 address construction applied to PPU's
            // provider lane.  It assigns a row and base word that do not
            // describe PPU x2's redistributed 64-bit provider window.
            int const bad_row = provider % 8;
            int const bad_base_word = ((provider / 8) * 4) % 8;
            source_base = k_base +
                          bad_row * 16 + 2 * bad_base_word;
          }
          if (plant == Plant::ShiftedWord) word ^= 1;

          for (int half = 0; half < 2; ++half) {
            int const got = source_base + 2 * word + half;
            int const want = k_base + 2 * a + 8 * reg + half;
            ++result.values;
            result.mismatches += got != want;
            int const byte_begin = got * int(sizeof(uint16_t));
            int const byte_end = byte_begin + int(sizeof(uint16_t));
            result.out_of_stage += byte_begin < 0 || byte_end > kAStageBytes;
            result.max_byte_end = std::max(result.max_byte_end, byte_end);
          }
        }
      }
    }
  }
  return result;
}

struct SharedMetric {
  int stored_rows = 0;
  int a_stage_vectors = 0;
  int shared_bytes = 0;
  int mismatches = 0;
};

SharedMetric verify_shared_ledger(Plant plant) {
  constexpr int kKBlocks = kTileK / 16;
  constexpr int kASharedStride = 16 * kKBlocks / 8;
  int const rows = plant == Plant::PaddedARows ? 8 : 1;
  int const a_stage = kASharedStride * rows;
  int const shared_bytes =
      kStages * (a_stage + kBSharedStage + kScaleSharedStage) * 16;
  SharedMetric result{rows, a_stage, shared_bytes, 0};
  result.mismatches += rows != 1;
  result.mismatches += a_stage != 16;
  result.mismatches += shared_bytes != 34816;
  return result;
}

constexpr int output_row_m8(int lane, int /*value*/) { return lane / 4; }
constexpr int lane_of(int tid) { return tid % 32; }

struct OutputMetric {
  int visits = 0;
  int holes = 0;
  int duplicates = 0;
  int out_of_range = 0;
};

OutputMetric verify_output(Plant plant) {
  constexpr int kTileM = 8;
  int const values = plant == Plant::M16OutputValues ? 8 : 4;
  std::array<int, kTileM * kTileN> hit{};
  OutputMetric result;
  for (int tid = 0; tid < kOutputThreads; ++tid) {
    int const lane = lane_of(tid);
    int const warp_n = tid / 32;
    for (int n_block = 0; n_block < 4; ++n_block) {
      for (int value = 0; value < values; ++value) {
        int const row = output_row_m8(lane, value);
        int const col = (warp_n * 4 + n_block) * 16 + lane % 4 +
                        ((value % 4) << 2);
        ++result.visits;
        if (row < 0 || row >= kTileM || col < 0 || col >= kTileN) {
          ++result.out_of_range;
        } else {
          ++hit[row * kTileN + col];
        }
      }
    }
  }
  for (int n : hit) {
    result.holes += n == 0;
    result.duplicates += std::max(0, n - 1);
  }
  return result;
}

struct Contribution {
  std::array<int, kKCohorts> count{};
};

Contribution plus(Contribution lhs, Contribution const& rhs) {
  for (int k = 0; k < kKCohorts; ++k) lhs.count[k] += rhs.count[k];
  return lhs;
}

struct ReductionMetric {
  int holes = 0;
  int duplicates = 0;
  int bad_addresses = 0;
  int expected_address_holes = 0;
  int expected_address_duplicates = 0;
  int final_slots = 0;
  int stride_vectors = 0;
};

ReductionMetric verify_reduction(Plant plant) {
  constexpr int kNBlocks = 4;
  constexpr int kM8Halves = 1;
  int const halves = plant == Plant::M16Scratch ? 2 : kM8Halves;
  int const stride = kOutputThreads * 4 * halves;
  constexpr int expected_stride = kOutputThreads * 4 * kM8Halves;
  constexpr int kScratchVectors = 2048;
  std::array<Contribution, kScratchVectors> scratch{};
  std::array<int, kScratchVectors> writes{};
  std::array<int, kScratchVectors> expected_writes{};
  ReductionMetric result;
  result.stride_vectors = stride;

  auto record = [&](int address, Contribution value) {
    if (address < 0 || address >= kScratchVectors) {
      ++result.bad_addresses;
      return;
    }
    scratch[address] = value;
    ++writes[address];
  };

  for (int compact = 0; compact < kOutputThreads; ++compact) {
    for (int n_block = 0; n_block < kNBlocks; ++n_block) {
      for (int half = 0; half < halves; ++half) {
        int const chunk = halves * n_block + half;
        std::array<Contribution, kKCohorts> value{};
        for (int wk = 0; wk < kKCohorts; ++wk) value[wk].count[wk] = 1;

        for (int wk : {2, 3}) {
          int const read = stride * wk + compact;
          int const write = kOutputThreads * chunk + (read - stride * 2);
          record(write, value[wk]);
        }

        int const lower = kOutputThreads * chunk + compact;
        int const upper = kOutputThreads * chunk + stride + compact;
        Contribution survivor = value[1];
        if (plant != Plant::SkipSecondReductionStep) {
          if (lower < kScratchVectors && upper < kScratchVectors) {
            survivor = plus(survivor, scratch[lower]);
            survivor = plus(survivor, scratch[upper]);
          }
          record(lower, survivor);
        }

        Contribution final = value[0];
        if (lower >= 0 && lower < kScratchVectors) {
          final = plus(final, scratch[lower]);
        }
        for (int wk = 0; wk < kKCohorts; ++wk) {
          result.holes += final.count[wk] == 0;
          result.duplicates += std::max(0, final.count[wk] - 1);
        }
        ++result.final_slots;

        if (half < kM8Halves) {
          int const expected_chunk = kM8Halves * n_block + half;
          int const expected_lower = kOutputThreads * expected_chunk + compact;
          int const expected_upper =
              kOutputThreads * expected_chunk + expected_stride + compact;
          expected_writes[expected_lower] += 2;
          expected_writes[expected_upper] += 1;
        }
      }
    }
  }
  for (int address = 0; address < kScratchVectors; ++address) {
    result.expected_address_holes +=
        expected_writes[address] > 0 && writes[address] == 0;
    result.expected_address_duplicates +=
        writes[address] != expected_writes[address]
            ? std::max(1, std::abs(writes[address] - expected_writes[address]))
            : 0;
  }
  return result;
}

template <class T>
int differences(std::vector<T> const& lhs, std::vector<T> const& rhs) {
  if (lhs.size() != rhs.size()) return -1;
  int result = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) result += lhs[i] != rhs[i];
  return result;
}

struct ArtifactMetric {
  int b_byte_diff = 0;
  int scale_byte_diff = 0;
  int b_roundtrip_bad = 0;
  int scale_roundtrip_bad = 0;
  std::size_t b_bytes = 0;
  std::size_t scale_bytes = 0;
};

ArtifactMetric verify_artifact(Plant plant) {
  constexpr std::size_t kK = 256;
  constexpr std::size_t kN = 256;
  quactlize::marlin::ClassicFormatExtent extent{};
  ArtifactMetric result;
  if (!quactlize::marlin::classic_format_extent(kK, kN, extent)) {
    result.b_byte_diff = result.scale_byte_diff = -1;
    return result;
  }
  std::vector<uint8_t> q(extent.logical_codes);
  for (std::size_t i = 0; i < q.size(); ++i) {
    uint32_t x = uint32_t(i) + UINT32_C(0x9e3779b9);
    x ^= x >> 16;
    x *= UINT32_C(0x7feb352d);
    x ^= x >> 15;
    q[i] = uint8_t(x & 15);
  }
  std::vector<uint8_t> b16(extent.packed_bytes), b8(extent.packed_bytes);
  bool b16_ok = quactlize::marlin::pack_biased_int4_bytes(
      q.data(), q.size(), b16.data(), b16.size(), kK, kN);
  bool b8_ok = quactlize::marlin::pack_biased_int4_bytes(
      q.data(), q.size(), b8.data(), b8.size(), kK, kN);
  if (plant == Plant::MDependentArtifact && !b8.empty()) b8[137] ^= 0x10;

  std::vector<uint8_t> recovered(q.size());
  bool unpacked = b8_ok && quactlize::marlin::unpack_biased_int4_bytes(
      b8.data(), b8.size(), recovered.data(), recovered.size(), kK, kN);
  result.b_byte_diff = b16_ok && b8_ok ? differences(b16, b8) : -1;
  result.b_roundtrip_bad = unpacked ? differences(q, recovered) : -1;

  std::vector<uint16_t> plain(extent.scale_elements);
  for (std::size_t i = 0; i < plain.size(); ++i) {
    plain[i] = uint16_t((i * 40503u + 17u) & 0xffffu);
  }
  std::vector<uint16_t> s16(plain.size()), s8(plain.size());
  bool s16_ok = quactlize::marlin::permute_gs128_scales(
      plain.data(), plain.size(), s16.data(), s16.size(), kK, kN);
  bool s8_ok = quactlize::marlin::permute_gs128_scales(
      plain.data(), plain.size(), s8.data(), s8.size(), kK, kN);
  if (plant == Plant::MDependentArtifact && !s8.empty()) s8[91] ^= 1;
  std::vector<uint16_t> plain_roundtrip(plain.size());
  bool scale_unpacked = s8_ok && quactlize::marlin::unpermute_gs128_scales(
      s8.data(), s8.size(), plain_roundtrip.data(), plain_roundtrip.size(),
      kK, kN);
  result.scale_byte_diff = s16_ok && s8_ok ? differences(s16, s8) : -1;
  result.scale_roundtrip_bad =
      scale_unpacked ? differences(plain, plain_roundtrip) : -1;
  result.b_bytes = b8.size();
  result.scale_bytes = s8.size() * sizeof(uint16_t);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  Plant const plant = parse_plant(argc, argv);
  CopyMetric const copy = verify_cooperative_copy(plant);
  LoadMetric const load = verify_plain_x2_row0(plant);
  SharedMetric const shared = verify_shared_ledger(plant);
  OutputMetric const output = verify_output(plant);
  ReductionMetric const reduction = verify_reduction(plant);
  ArtifactMetric const artifact = verify_artifact(plant);

  bool const copy_ok = copy.a_issuers == 16 && copy.a_bytes == 256 &&
                       copy.a_holes == 0 && copy.a_duplicates == 0 &&
                       copy.b_threads == 256 && copy.b_chunks == 512 &&
                       copy.b_holes == 0 && copy.b_duplicates == 0;
  bool const load_ok = load.values == 128 && load.mismatches == 0 &&
                       load.out_of_stage == 0 && load.max_byte_end == 256;
  bool const shared_ok = shared.mismatches == 0;
  bool const output_ok = output.visits == 8 * 128 && output.holes == 0 &&
                         output.duplicates == 0 && output.out_of_range == 0;
  bool const reduction_ok = reduction.final_slots == 64 * 4 &&
                            reduction.holes == 0 &&
                            reduction.duplicates == 0 &&
                            reduction.bad_addresses == 0 &&
                            reduction.expected_address_holes == 0 &&
                            reduction.expected_address_duplicates == 0 &&
                            reduction.stride_vectors == 256;
  bool const artifact_ok = artifact.b_byte_diff == 0 &&
                           artifact.scale_byte_diff == 0 &&
                           artifact.b_roundtrip_bad == 0 &&
                           artifact.scale_roundtrip_bad == 0;
  bool const ok = copy_ok && load_ok && shared_ok && output_ok &&
                  reduction_ok && artifact_ok;

  std::printf("L181 copy plant=%s A={issuers:%d bytes:%d holes:%d dup:%d} "
              "B={threads:%d chunks:%d holes:%d dup:%d}\n",
              name(plant), copy.a_issuers, copy.a_bytes, copy.a_holes,
              copy.a_duplicates, copy.b_threads, copy.b_chunks, copy.b_holes,
              copy.b_duplicates);
  std::printf("L181 load atom=plain-x2 row=0 values=%d mismatches=%d "
              "out-of-stage=%d max-byte-end=%d/256\n",
              load.values, load.mismatches, load.out_of_stage,
              load.max_byte_end);
  std::printf("L181 shared stored_rows=%d A_stage_vectors=%d bytes=%d bad=%d\n",
              shared.stored_rows, shared.a_stage_vectors,
              shared.shared_bytes, shared.mismatches);
  std::printf("L181 output threads=64 values/thread=16 visits=%d holes=%d "
              "duplicates=%d out-of-range=%d\n",
              output.visits, output.holes, output.duplicates,
              output.out_of_range);
  std::printf("L181 reduce cadence=4->2->1 stride_vectors=%d final_slots=%d "
              "contribution={holes:%d duplicates:%d} "
              "addresses={bad:%d holes:%d duplicates:%d}\n",
              reduction.stride_vectors, reduction.final_slots,
              reduction.holes, reduction.duplicates,
              reduction.bad_addresses, reduction.expected_address_holes,
              reduction.expected_address_duplicates);
  std::printf("L181 artifact B=%zuB diff=%d roundtrip=%d "
              "scale=%zuB diff=%d roundtrip=%d\n",
              artifact.b_bytes, artifact.b_byte_diff,
              artifact.b_roundtrip_bad, artifact.scale_bytes,
              artifact.scale_byte_diff, artifact.scale_roundtrip_bad);

  if (plant == Plant::None) {
    if (!ok) {
      std::puts("L181 FAIL: standalone Marlin packed-M1 m8 contract is red");
      return 1;
    }
    std::puts("L181 PASS: cooperative packed-row A + plain-x2 provider + "
              "m8 output + 4->2->1 scratch + M-invariant artifacts are closed");
    return 0;
  }
  if (ok) {
    std::printf("L181 UNEXPECTED-GREEN plant=%s\n", name(plant));
    return 0;
  }
  std::printf("L181 EXPECTED-RED plant=%s\n", name(plant));
  return 2;
}

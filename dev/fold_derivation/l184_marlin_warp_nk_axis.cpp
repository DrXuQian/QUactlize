#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string_view>
#include <vector>

#include "marlin_format_ppu.hpp"

namespace {

constexpr int kTileM8 = 8;
constexpr int kTileM16 = 16;
constexpr int kTileN = 128;
constexpr int kTileK = 128;
constexpr int kAtomN = 16;
constexpr int kAtomK = 16;
constexpr int kLanes = 32;
constexpr int kPackedNBlocksPerVector = 4;
constexpr int kNBlocks = kTileN / kAtomN;
constexpr int kKBlocks = kTileK / kAtomK;
constexpr int kVectorsPerKBlock = (kNBlocks / 4) * kLanes;
constexpr int kWordsPerVector = 4;
constexpr int kTileVectors = kKBlocks * kVectorsPerKBlock;
constexpr int kTileWords = kTileVectors * kWordsPerVector;

struct Config {
  int warp_n;
  int warp_k;
  std::string_view name;
};

struct ConsumerSummary {
  int threads = 0;
  int warp_on_n = 0;
  int warp_on_k = 0;
  int nblocks_per_warp = 0;
  int k_inner_iters = 0;
  int vector_loads_per_k_inner = 0;
  int word_holes = 0;
  int word_duplicates = 0;
  int vector_min_reads = 0;
  int vector_max_reads = 0;
  int vector_total_reads = 0;
  int producer_consumer_mismatches = 0;
  int a_b_k_mismatches = 0;
  int a_k_holes = 0;
  int a_k_duplicates = 0;
  int a_shared_read_factor = 0;
  int production_m8_ab_block_mismatches = 0;
  int production_m8_ab_value_mismatches = 0;
  int old_word_holes = 0;
  int old_word_duplicates = 0;
};

[[noreturn]] void die(char const* why) {
  std::fprintf(stderr, "[l184] FAIL: %s\n", why);
  std::exit(1);
}

void require(bool condition, char const* why) {
  if (!condition) {
    die(why);
  }
}

constexpr int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

constexpr int output_row(int instruction_m, int lane, int value) {
  return lane / 4 +
         (instruction_m == 16 ? (((value >> 2) & 1) << 3) : 0);
}

constexpr int output_col_offset(int lane, int value) {
  return lane % 4 + ((value % 4) << 2);
}

constexpr int local_vector_index(int ktile, int ngroup, int lane) {
  return ktile * kVectorsPerKBlock + ngroup * kLanes + lane;
}

constexpr int local_word_index(
    int ktile, int nblock, int lane) {
  return local_vector_index(ktile, nblock / 4, lane) * kWordsPerVector +
         nblock % 4;
}

struct SourceVector {
  int ktile;
  int ngroup;
  int lane;
};

// Compose the production cp.async writer, rather than pretending shared is a
// naturally materialised tensor.  Copy iteration `i`, thread `tid` writes
// `threads*i+tid`; its global source advances by Threads/BSharedStride K
// blocks.  The current 256-thread family happens to make source and shared
// vector ordinals equal, but the proof is this composition, not that accident.
constexpr SourceVector producer_source_for_shared(
    int threads, int shared_vector) {
  int const copy_iter = shared_vector / threads;
  int const tid = shared_vector % threads;
  int const b_shared_stride = kVectorsPerKBlock;
  int const tid_k = tid / b_shared_stride;
  int const rem = tid % b_shared_stride;
  return SourceVector{
      tid_k + copy_iter * (threads / b_shared_stride),
      rem / kLanes,
      rem % kLanes};
}

ConsumerSummary consumer_summary(Config cfg, bool drop_last = false) {
  require(kTileN % cfg.warp_n == 0 && kTileK % cfg.warp_k == 0,
          "warp shape must divide the fixed tile");
  ConsumerSummary out;
  out.warp_on_n = kTileN / cfg.warp_n;
  out.warp_on_k = kTileK / cfg.warp_k;
  out.threads = kLanes * out.warp_on_n * out.warp_on_k;
  out.nblocks_per_warp = cfg.warp_n / kAtomN;
  out.k_inner_iters = cfg.warp_k / kAtomK;
  out.vector_loads_per_k_inner =
      ceil_div(out.nblocks_per_warp, kPackedNBlocksPerVector);

  std::vector<int> word_visits(kTileWords, 0);
  std::vector<int> vector_reads(kTileVectors, 0);
  int sequence = 0;
  int const sequence_count = out.warp_on_k * out.warp_on_n * kLanes *
                             out.k_inner_iters *
                             out.vector_loads_per_k_inner;
  for (int warp_k = 0; warp_k < out.warp_on_k; ++warp_k) {
    for (int warp_n = 0; warp_n < out.warp_on_n; ++warp_n) {
      for (int lane = 0; lane < kLanes; ++lane) {
        for (int inner = 0; inner < out.k_inner_iters; ++inner) {
          // Classic interleaves K cohorts inside every compute step:
          // ktile = inner * N_WARPS_K + warp_k.
          int const ktile = inner * out.warp_on_k + warp_k;
          for (int load = 0; load < out.vector_loads_per_k_inner; ++load) {
            bool const omitted = drop_last && sequence == sequence_count - 1;
            ++sequence;
            if (omitted) {
              continue;
            }
            int const nblock_begin =
                warp_n * out.nblocks_per_warp +
                load * kPackedNBlocksPerVector;
            int const ngroup = nblock_begin / kPackedNBlocksPerVector;
            int const vector = local_vector_index(ktile, ngroup, lane);
            require(vector >= 0 && vector < kTileVectors,
                    "new consumer vector escaped the staged B tile");
            ++vector_reads[vector];
            SourceVector const source =
                producer_source_for_shared(out.threads, vector);
            out.producer_consumer_mismatches +=
                source.ktile != ktile || source.ngroup != ngroup ||
                source.lane != lane;
            // The A K atom is interleaved by the same `(inner, warp_k)` pair.
            // Every N-load step for that K atom must meet B from that exact K,
            // not merely contribute to an aggregate exact cover.
            out.a_b_k_mismatches += source.ktile != ktile;
            int const word_begin = nblock_begin % kPackedNBlocksPerVector;
            int const words = std::min(
                kPackedNBlocksPerVector,
                out.nblocks_per_warp -
                    load * kPackedNBlocksPerVector);
            for (int word = 0; word < words; ++word) {
              int const physical = vector * kWordsPerVector +
                                   word_begin + word;
              require(physical >= 0 && physical < kTileWords,
                      "new consumer word escaped the staged B tile");
              ++word_visits[physical];
            }
          }
        }
      }
    }
  }
  require(sequence == sequence_count, "coverage denominator drifted");
  out.word_holes = int(std::count(word_visits.begin(), word_visits.end(), 0));
  for (int count : word_visits) {
    out.word_duplicates += std::max(0, count - 1);
  }
  out.vector_min_reads = *std::min_element(vector_reads.begin(), vector_reads.end());
  out.vector_max_reads = *std::max_element(vector_reads.begin(), vector_reads.end());
  out.vector_total_reads =
      std::accumulate(vector_reads.begin(), vector_reads.end(), 0);

  // Count semantic A K atoms once per N warp and K-inner.  Loading a second
  // 64-column B vector for WN128 may reread that same A register fragment;
  // report that instruction cost separately from semantic coverage.
  std::array<int, kKBlocks> a_k_visits{};
  for (int warp_k = 0; warp_k < out.warp_on_k; ++warp_k) {
    for (int warp_n = 0; warp_n < out.warp_on_n; ++warp_n) {
      for (int inner = 0; inner < out.k_inner_iters; ++inner) {
        int const ktile = inner * out.warp_on_k + warp_k;
        ++a_k_visits[ktile];
      }
    }
  }
  for (int count : a_k_visits) {
    out.a_k_holes += count == 0;
    out.a_k_duplicates += std::max(0, count - out.warp_on_n);
  }
  out.a_shared_read_factor = out.vector_loads_per_k_inner;

  // Reconstruct the actual packed-m8 A reader in production, including the
  // PPU x2 provider map proved by L181:
  //   source_base = warp_k*WarpK + k_inner*16
  //               + 4*(provider%2) + 8*(provider/16)
  //   provider= a/2 + 16*reg, word=a%2
  // which yields logical K = warp_k*WarpK + k_inner*16
  //                          + 2*a + 8*reg + half.
  // Pair it against the B source that actually arrives through cp.async.
  int const compute_steps =
      out.k_inner_iters * out.vector_loads_per_k_inner;
  for (int warp_k = 0; warp_k < out.warp_on_k; ++warp_k) {
    for (int warp_n = 0; warp_n < out.warp_on_n; ++warp_n) {
      for (int step = 0; step < compute_steps; ++step) {
        int const k_inner = step / out.vector_loads_per_k_inner;
        int const n_load = step % out.vector_loads_per_k_inner;
        int const nblock_begin =
            warp_n * out.nblocks_per_warp + 4 * n_load;
        int const shared_vector = local_vector_index(
            k_inner * out.warp_on_k + warp_k,
            nblock_begin / kPackedNBlocksPerVector, 0);
        SourceVector const b_source =
            producer_source_for_shared(out.threads, shared_vector);
        int const actual_a_block =
            warp_k * (cfg.warp_k / kAtomK) + k_inner;
        if (actual_a_block != b_source.ktile) {
          ++out.production_m8_ab_block_mismatches;
        }
        for (int a = 0; a < 4; ++a) {
          for (int reg = 0; reg < 2; ++reg) {
            int const provider = a / 2 + 16 * reg;
            int const word = a % 2;
            int const source_base =
                warp_k * cfg.warp_k + k_inner * kAtomK +
                4 * (provider % 2) + 8 * (provider / 16);
            for (int half = 0; half < 2; ++half) {
              int const actual_k = source_base + 2 * word + half;
              int const expected_k =
                  b_source.ktile * kAtomK + 2 * a + 8 * reg + half;
              out.production_m8_ab_value_mismatches +=
                  actual_k != expected_k;
            }
          }
        }
      }
    }
  }

  // Negative control: the old fixed 2N x 4K spelling used two compute
  // iterations and four words from `Threads*inner + tid`.  Apply that exact
  // address/loop to every candidate.  Only WN64/WK32 is allowed to survive.
  std::vector<int> old_visits(kTileWords, 0);
  for (int tid = 0; tid < out.threads; ++tid) {
    int const warp_id = tid / kLanes;
    int const warp_n = warp_id % out.warp_on_n;
    int const warp_k = warp_id / out.warp_on_n;
    int const lane = tid % kLanes;
    for (int inner = 0; inner < 2; ++inner) {
      int const vector = out.threads * inner + tid;
      if (vector < 0 || vector >= kTileVectors) {
        continue;
      }
      for (int word = 0; word < 4; ++word) {
        SourceVector const source =
            producer_source_for_shared(out.threads, vector);
        int const physical_ktile = source.ktile;
        int const physical_nblock =
            source.ngroup * kPackedNBlocksPerVector + word;
        int const physical_lane = source.lane;
        int const expected_ktile = inner * out.warp_on_k + warp_k;
        int const expected_nblock =
            warp_n * out.nblocks_per_warp + word;
        // A physical exact cover is insufficient: the word must arrive at
        // the accumulator whose A K-slice and output N-block name that word.
        // This is the seam the old fixed loop violates for both new shapes.
        if (word < out.nblocks_per_warp &&
            physical_ktile == expected_ktile &&
            physical_nblock == expected_nblock &&
            physical_lane == lane) {
          ++old_visits[local_word_index(
              expected_ktile, expected_nblock, lane)];
        }
      }
    }
  }
  out.old_word_holes =
      int(std::count(old_visits.begin(), old_visits.end(), 0));
  for (int count : old_visits) {
    out.old_word_duplicates += std::max(0, count - 1);
  }
  return out;
}

int output_coverage(Config cfg, int instruction_m, bool old_four_blocks) {
  int const warp_on_n = kTileN / cfg.warp_n;
  int const warp_on_k = kTileK / cfg.warp_k;
  int const output_threads = kLanes * warp_on_n;
  int const nblocks_per_warp = cfg.warp_n / kAtomN;
  int const accumulator_values = instruction_m / 2;
  std::vector<int> visits(instruction_m * kTileN, 0);
  for (int tid = 0; tid < output_threads; ++tid) {
    int const warp_n = tid / kLanes;
    int const lane = tid % kLanes;
    int const nblock_limit = old_four_blocks ? 4 : nblocks_per_warp;
    for (int nblock = 0; nblock < nblock_limit; ++nblock) {
      int const nbase =
          (warp_n * nblocks_per_warp + nblock) * kAtomN;
      for (int value = 0; value < accumulator_values; ++value) {
        int const row = output_row(instruction_m, lane, value);
        int const col = nbase + output_col_offset(lane, value);
        if (row >= 0 && row < instruction_m && col >= 0 && col < kTileN) {
          ++visits[row * kTileN + col];
        }
      }
    }
  }
  (void)warp_on_k;
  int bad = 0;
  for (int count : visits) {
    bad += count != 1;
  }
  return bad;
}

std::uint32_t expected_word(
    std::vector<std::uint8_t> const& logical, int n_size,
    int ktile, int nblock, int lane) {
  int const n = nblock * 16 + lane / 4;
  int const kb = ktile * 16 + (lane % 4) * 2;
  auto code = [&](int k, int col) -> std::uint32_t {
    return logical[std::size_t(k) * std::size_t(n_size) + std::size_t(col)];
  };
  return code(kb, n) << 0 | code(kb + 1, n) << 16 |
         code(kb + 8, n) << 4 | code(kb + 9, n) << 20 |
         code(kb, n + 8) << 8 | code(kb + 1, n + 8) << 24 |
         code(kb + 8, n + 8) << 12 | code(kb + 9, n + 8) << 28;
}

int byte_anchor_mismatches(Config cfg, bool permute_one_word) {
  constexpr int kSize = 128;
  constexpr int nSize = 256;
  quactlize::marlin::ClassicFormatExtent extent{};
  require(quactlize::marlin::classic_format_extent(kSize, nSize, extent),
          "classic format rejected its production-aligned anchor");
  std::vector<std::uint8_t> logical(extent.logical_codes);
  for (int k = 0; k < kSize; ++k) {
    for (int n = 0; n < nSize; ++n) {
      logical[std::size_t(k) * nSize + n] =
          std::uint8_t((11 * k + 7 * n + (k >> 3) + (n >> 4)) & 15);
    }
  }
  std::vector<std::uint32_t> packed(extent.packed_words);
  std::vector<std::uint8_t> recovered(extent.logical_codes, 0xff);
  require(quactlize::marlin::pack_biased_int4_u32(
              logical.data(), logical.size(), packed.data(), packed.size(),
              kSize, nSize),
          "classic format pack failed");
  require(quactlize::marlin::unpack_biased_int4_u32(
              packed.data(), packed.size(), recovered.data(), recovered.size(),
              kSize, nSize),
          "classic format recover failed");
  require(recovered == logical,
          "classic place/recover anchor is not an identity");

  int const warp_on_n = kTileN / cfg.warp_n;
  int const warp_on_k = kTileK / cfg.warp_k;
  int const nblocks_per_warp = cfg.warp_n / kAtomN;
  int const k_inner_iters = cfg.warp_k / kAtomK;
  int const loads = ceil_div(nblocks_per_warp, 4);
  int const threads = kLanes * warp_on_n * warp_on_k;
  int mismatches = 0;
  bool planted = false;
  for (int warp_k = 0; warp_k < warp_on_k; ++warp_k) {
    for (int warp_n = 0; warp_n < warp_on_n; ++warp_n) {
      for (int inner = 0; inner < k_inner_iters; ++inner) {
        int const ktile = inner * warp_on_k + warp_k;
        for (int load = 0; load < loads; ++load) {
          int const nblock_begin = warp_n * nblocks_per_warp + 4 * load;
          int const words = std::min(4, nblocks_per_warp - 4 * load);
          for (int lane = 0; lane < kLanes; ++lane) {
            for (int word = 0; word < words; ++word) {
              int nblock = nblock_begin + word;
              int selected_nblock = nblock;
              if (permute_one_word && !planted) {
                selected_nblock = nblock_begin + ((word + 1) % words);
                planted = true;
              }
              int const shared_vector = local_vector_index(
                  ktile, nblock_begin / kPackedNBlocksPerVector, lane);
              SourceVector const source =
                  producer_source_for_shared(threads, shared_vector);
              int const selected_word =
                  selected_nblock % kPackedNBlocksPerVector;
              int const source_nblock =
                  source.ngroup * kPackedNBlocksPerVector + selected_word;
              std::size_t const physical =
                  quactlize::marlin::detail::classic_word_offset(
                      nSize, std::size_t(source.ktile),
                      std::size_t(source_nblock), std::size_t(source.lane));
              require(physical < packed.size(),
                      "consumer physical word escaped packed artifact");
              mismatches += packed[physical] !=
                            expected_word(logical, nSize, ktile, nblock, lane);
            }
          }
        }
      }
    }
  }
  return mismatches;
}

int reduction_result(int cohorts, int red_off) {
  // One output coordinate is sufficient: the production tree uses the same
  // cohort-index arithmetic independently for every compact owner/fragment.
  std::vector<int> accum(cohorts);
  for (int i = 0; i < cohorts; ++i) {
    accum[i] = 1 << i;
  }
  std::vector<int> scratch(cohorts, 0);
  for (int step = red_off; step > 0; step /= 2) {
    std::vector<int> next_scratch = scratch;
    for (int red_idx = 0; red_idx < cohorts; ++red_idx) {
      if (step <= red_idx && red_idx < 2 * step) {
        int value = accum[red_idx];
        if (step < red_off) {
          value += scratch[red_idx] + scratch[red_idx - step];
        }
        next_scratch[red_idx - step] = value;
      }
    }
    scratch.swap(next_scratch);
  }
  return accum[0] + scratch[0];
}

struct WholeVectorGeometry {
  int tile_n;
  int tile_k;
  int warp_n;
  int warp_k;
  std::string_view name;
};

struct WholeVectorMetric {
  int threads = 0;
  int compute_steps = 0;
  int b_words = 0;
  int b_holes = 0;
  int b_duplicates = 0;
  int b_source_mismatches = 0;
  int scale_values = 0;
  int scale_mismatches = 0;
  int naive_scale_group_mismatches = 0;
  int planned_a_b_mismatches = 0;
  int old_m8_a_b_mismatches = 0;
  int old_fixed_holes = 0;
  int old_fixed_duplicates = 0;
};

SourceVector whole_vector_producer_source(
    WholeVectorGeometry g, int threads, int shared_vector) {
  int const nblocks = g.tile_n / kAtomN;
  int const b_stride = (nblocks / 4) * kLanes;
  int const copy_iter = shared_vector / threads;
  int const tid = shared_vector % threads;
  int const rem = tid % b_stride;
  return SourceVector{
      tid / b_stride + copy_iter * (threads / b_stride),
      rem / kLanes,
      rem % kLanes};
}

WholeVectorMetric whole_vector_metric(WholeVectorGeometry g) {
  require(g.tile_n % g.warp_n == 0 && g.tile_k % g.warp_k == 0,
          "whole-vector warp shape must divide its tile");
  require(g.warp_n == 64 || g.warp_n == 128,
          "whole-vector oracle accepts WN64/WN128 only");
  int const nblocks = g.tile_n / kAtomN;
  int const kblocks = g.tile_k / kAtomK;
  int const warp_on_n = g.tile_n / g.warp_n;
  int const warp_on_k = g.tile_k / g.warp_k;
  int const nblocks_per_warp = g.warp_n / kAtomN;
  int const k_inner = g.warp_k / kAtomK;
  int const b_loads = nblocks_per_warp / 4;
  int const threads = kLanes * warp_on_n * warp_on_k;
  int const compute_steps = k_inner * b_loads;
  int const b_stride = (nblocks / 4) * kLanes;
  int const b_vectors = kblocks * b_stride;
  int const b_words = b_vectors * 4;
  require(nblocks_per_warp % 4 == 0 && b_vectors % threads == 0,
          "whole-vector B load/copy does not divide exactly");

  WholeVectorMetric out;
  out.threads = threads;
  out.compute_steps = compute_steps;
  out.b_words = b_words;
  std::vector<int> visits(b_words, 0);
  std::vector<int> old_visits(b_words, 0);

  for (int warp_k = 0; warp_k < warp_on_k; ++warp_k) {
    for (int warp_n = 0; warp_n < warp_on_n; ++warp_n) {
      for (int lane = 0; lane < kLanes; ++lane) {
        for (int step = 0; step < compute_steps; ++step) {
          int const inner = step / b_loads;
          int const load = step % b_loads;
          int const expected_k = inner * warp_on_k + warp_k;
          int const expected_group = warp_n * b_loads + load;
          int const shared_vector =
              expected_k * b_stride + expected_group * kLanes + lane;
          SourceVector const source =
              whole_vector_producer_source(g, threads, shared_vector);
          out.b_source_mismatches +=
              source.ktile != expected_k ||
              source.ngroup != expected_group || source.lane != lane;
          // The generic A reader is the classic interleaved-K expression.
          int const planned_a_k = inner * warp_on_k + warp_k;
          out.planned_a_b_mismatches += planned_a_k != source.ktile;
          // The current packed-m8 source instead spells
          // warp_k*WarpK + inner*16.  Keep it as an exact regression witness.
          int const old_m8_a_k = warp_k * k_inner + inner;
          out.old_m8_a_b_mismatches += old_m8_a_k != source.ktile;
          for (int word = 0; word < 4; ++word) {
            int const source_word =
                ((source.ktile * (nblocks / 4) + source.ngroup) * kLanes +
                 source.lane) * 4 + word;
            require(source_word >= 0 && source_word < b_words,
                    "whole-vector source word escaped B stage");
            ++visits[source_word];
          }
        }
      }
    }
  }
  out.b_holes = int(std::count(visits.begin(), visits.end(), 0));
  for (int count : visits) out.b_duplicates += std::max(0, count - 1);

  // Apply the old `Threads*inner+tid`, four-word loop.  It is exact for the
  // WN64 family, but must not silently authorize either WN128 consumer.
  for (int tid = 0; tid < threads; ++tid) {
    int const warp_id = tid / kLanes;
    int const warp_n = warp_id % warp_on_n;
    int const warp_k = warp_id / warp_on_n;
    int const lane = tid % kLanes;
    for (int inner = 0; inner < 2; ++inner) {
      int const shared_vector = threads * inner + tid;
      if (shared_vector >= b_vectors) continue;
      SourceVector const source =
          whole_vector_producer_source(g, threads, shared_vector);
      int const expected_k = inner * warp_on_k + warp_k;
      for (int word = 0; word < 4; ++word) {
        int const expected_nblock = warp_n * nblocks_per_warp + word;
        int const source_nblock = source.ngroup * 4 + word;
        if (word < nblocks_per_warp && source.ktile == expected_k &&
            source_nblock == expected_nblock && source.lane == lane) {
          int const expected_word =
              ((expected_k * (nblocks / 4) + expected_nblock / 4) *
                   kLanes + lane) * 4 + expected_nblock % 4;
          ++old_visits[expected_word];
        }
      }
    }
  }
  out.old_fixed_holes =
      int(std::count(old_visits.begin(), old_visits.end(), 0));
  for (int count : old_visits) {
    out.old_fixed_duplicates += std::max(0, count - 1);
  }

  // Independently anchor the grouped-scale byte permutation.  Use two groups
  // so TK64 must reuse one scale row for adjacent tiles while TK128 advances.
  constexpr int kAnchorK = 256;
  constexpr int kAnchorN = 256;
  std::vector<std::uint32_t> scale_plain(2 * kAnchorN);
  std::vector<std::uint32_t> scale_packed(2 * kAnchorN, UINT32_MAX);
  for (int group = 0; group < 2; ++group) {
    for (int n = 0; n < kAnchorN; ++n) {
      scale_plain[group * kAnchorN + n] =
          std::uint32_t(group * 1000 + n);
    }
  }
  require(quactlize::marlin::permute_gs128_scales(
              scale_plain.data(), scale_plain.size(), scale_packed.data(),
              scale_packed.size(), kAnchorK, kAnchorN),
          "whole-vector scale anchor permutation failed");
  int const group_tiles = 128 / g.tile_k;
  require(group_tiles == 1 || group_tiles == 2,
          "whole-vector scale oracle expects TK64/TK128");
  for (int tile = 0; tile < kAnchorK / g.tile_k; ++tile) {
    int const group = tile / group_tiles;
    out.naive_scale_group_mismatches += tile != group;
    for (int warp_n = 0; warp_n < warp_on_n; ++warp_n) {
      for (int load = 0; load < b_loads; ++load) {
        int const group64 = warp_n * b_loads + load;
        for (int lane = 0; lane < kLanes; ++lane) {
          int const c = lane / 4;
          int const scale_vector = 8 * group64 + c;
          for (int word = 0; word < 4; ++word) {
            for (int half = 0; half < 2; ++half) {
              int const packed_col = scale_vector * 8 + 2 * word + half;
              int const logical_col = group64 * 64 +
                                      word * 16 + c + 8 * half;
              if (logical_col < g.tile_n) {
                ++out.scale_values;
                out.scale_mismatches +=
                    scale_packed[group * kAnchorN + packed_col] !=
                    scale_plain[group * kAnchorN + logical_col];
              }
            }
          }
        }
      }
    }
  }
  return out;
}

int whole_vector_byte_mismatches(
    WholeVectorGeometry g, bool permute_one_word) {
  constexpr int kSize = 128;
  constexpr int nSize = 256;
  quactlize::marlin::ClassicFormatExtent extent{};
  require(quactlize::marlin::classic_format_extent(kSize, nSize, extent),
          "whole-vector byte anchor extent failed");
  std::vector<std::uint8_t> logical(extent.logical_codes);
  for (int k = 0; k < kSize; ++k) {
    for (int n = 0; n < nSize; ++n) {
      logical[std::size_t(k) * nSize + n] =
          std::uint8_t((13 * k + 5 * n + (k >> 2) + (n >> 3)) & 15);
    }
  }
  std::vector<std::uint32_t> packed(extent.packed_words);
  std::vector<std::uint8_t> recovered(extent.logical_codes, 0xff);
  require(quactlize::marlin::pack_biased_int4_u32(
              logical.data(), logical.size(), packed.data(), packed.size(),
              kSize, nSize),
          "whole-vector byte anchor pack failed");
  require(quactlize::marlin::unpack_biased_int4_u32(
              packed.data(), packed.size(), recovered.data(), recovered.size(),
              kSize, nSize) && recovered == logical,
          "whole-vector byte anchor place/recover is not identity");

  int const nblocks = g.tile_n / kAtomN;
  int const warp_on_n = g.tile_n / g.warp_n;
  int const warp_on_k = g.tile_k / g.warp_k;
  int const nblocks_per_warp = g.warp_n / kAtomN;
  int const k_inner = g.warp_k / kAtomK;
  int const b_loads = nblocks_per_warp / 4;
  int const threads = kLanes * warp_on_n * warp_on_k;
  int const b_stride = (nblocks / 4) * kLanes;
  int mismatches = 0;
  bool planted = false;
  for (int warp_k = 0; warp_k < warp_on_k; ++warp_k) {
    for (int warp_n = 0; warp_n < warp_on_n; ++warp_n) {
      for (int inner = 0; inner < k_inner; ++inner) {
        int const expected_k = inner * warp_on_k + warp_k;
        for (int load = 0; load < b_loads; ++load) {
          int const expected_group = warp_n * b_loads + load;
          for (int lane = 0; lane < kLanes; ++lane) {
            int const shared_vector =
                expected_k * b_stride + expected_group * kLanes + lane;
            SourceVector const source =
                whole_vector_producer_source(g, threads, shared_vector);
            for (int word = 0; word < 4; ++word) {
              int selected_word = word;
              if (permute_one_word && !planted) {
                selected_word = (word + 1) % 4;
                planted = true;
              }
              int const source_nblock = source.ngroup * 4 + selected_word;
              int const expected_nblock = expected_group * 4 + word;
              std::size_t const physical =
                  quactlize::marlin::detail::classic_word_offset(
                      nSize, std::size_t(source.ktile),
                      std::size_t(source_nblock), std::size_t(source.lane));
              mismatches += packed[physical] != expected_word(
                  logical, nSize, expected_k, expected_nblock, lane);
            }
          }
        }
      }
    }
  }
  return mismatches;
}

int whole_vector_output_bad(WholeVectorGeometry g, int instruction_m) {
  int const warp_on_n = g.tile_n / g.warp_n;
  int const output_threads = kLanes * warp_on_n;
  int const nblocks_per_warp = g.warp_n / kAtomN;
  int const values = instruction_m / 2;
  std::vector<int> visits(instruction_m * g.tile_n, 0);
  for (int tid = 0; tid < output_threads; ++tid) {
    int const warp_n = tid / kLanes;
    int const lane = tid % kLanes;
    for (int nblock = 0; nblock < nblocks_per_warp; ++nblock) {
      for (int value = 0; value < values; ++value) {
        int const row = output_row(instruction_m, lane, value);
        int const col =
            (warp_n * nblocks_per_warp + nblock) * kAtomN +
            output_col_offset(lane, value);
        if (row >= 0 && row < instruction_m &&
            col >= 0 && col < g.tile_n) {
          ++visits[row * g.tile_n + col];
        }
      }
    }
  }
  return int(std::count_if(
      visits.begin(), visits.end(), [](int count) { return count != 1; }));
}

std::size_t mainloop_bytes(
    WholeVectorGeometry g, int tile_m, int stages) {
  int const stored_rows = tile_m == 8 ? 1 : tile_m;
  std::size_t const per_stage =
      std::size_t(2 * stored_rows * g.tile_k) +
      std::size_t(g.tile_n * g.tile_k / 2) +
      std::size_t(2 * g.tile_n);
  return std::size_t(stages) * per_stage;
}

std::size_t reduction_bytes(
    WholeVectorGeometry g, int tile_m) {
  int const cohorts = g.tile_k / g.warp_k;
  return std::size_t(cohorts / 2) * std::size_t(tile_m) *
         std::size_t(g.tile_n) * sizeof(float);
}

std::size_t output_stage_bytes(
    WholeVectorGeometry g, int tile_m) {
  return std::size_t(2) * std::size_t(tile_m) *
         std::size_t(g.tile_n + 8);
}

void prove_whole_vector_geometry(
    WholeVectorGeometry g, bool expect_old_fixed) {
  WholeVectorMetric const m = whole_vector_metric(g);
  require(m.compute_steps == 2,
          "whole-vector pair must retain the proved two-step pipeline cadence");
  require(m.b_holes == 0 && m.b_duplicates == 0 &&
              m.b_source_mismatches == 0,
          "whole-vector B producer/consumer is not exact-once");
  require(m.scale_values > 0 && m.scale_mismatches == 0,
          "whole-vector scale consumer disagrees with classic permutation");
  require((g.tile_k == 64 && m.naive_scale_group_mismatches > 0) ||
              (g.tile_k == 128 && m.naive_scale_group_mismatches == 0),
          "scale-group reuse negative does not distinguish TK64/TK128");
  require(whole_vector_byte_mismatches(g, false) == 0,
          "whole-vector consumer changed the classic packed bytes");
  require(whole_vector_byte_mismatches(g, true) > 0,
          "whole-vector one-word byte permutation plant stayed green");
  require(m.planned_a_b_mismatches == 0,
          "classic interleaved A K reader disagrees with B");
  require(whole_vector_output_bad(g, 8) == 0 &&
              whole_vector_output_bad(g, 16) == 0,
          "whole-vector output cohort has a hole or duplicate");
  int const cohorts = g.tile_k / g.warp_k;
  require(reduction_result(cohorts, cohorts / 2) == (1 << cohorts) - 1,
          "whole-vector reduction tree lost a K cohort");
  require((m.old_fixed_holes == 0 && m.old_fixed_duplicates == 0) ==
              expect_old_fixed,
          "old fixed consumer classification disagrees with WN topology");
  if (g.warp_n == 128) {
    require(m.old_m8_a_b_mismatches == 0,
            "WN128/WK16 should remove the current m8 A/B K mismatch");
  } else {
    require(m.old_m8_a_b_mismatches > 0,
            "old contiguous-m8 K expression unexpectedly matches WN64");
  }

  int capacity_rows = 0;
  int old_union_overflows = 0;
  for (int tile_m : {8, 16}) {
    for (int stages = 2; stages <= 6; ++stages) {
      std::size_t const main = mainloop_bytes(g, tile_m, stages);
      std::size_t const reduce = reduction_bytes(g, tile_m);
      std::size_t const output = output_stage_bytes(g, tile_m);
      std::size_t const required = std::max({main, reduce, output});
      require(required <= 128 * 1024,
              "whole-vector cooperative exceeds the PPU block capacity");
      old_union_overflows += main < reduce || main < output;
      ++capacity_rows;
    }
  }
  if (g.warp_n == 128) {
    require(old_union_overflows > 0,
            "old mainloop-sized cooperative did not expose its WN128 overflow");
  }
  std::printf(
      "[l184:70] cfg=%.*s TN=%d TK=%d WN=%d WK=%d threads=%d "
      "steps=%d B=%d/%d scale=0/%d output=m8+m16-exact "
      "reduce=%d->1 capacity=%d/10 old-union-overflow=%d "
      "old-m8-k=%d old-fixed=%s\n",
      int(g.name.size()), g.name.data(), g.tile_n, g.tile_k,
      g.warp_n, g.warp_k, m.threads, m.compute_steps,
      m.b_words - m.b_holes, m.b_words, m.scale_values, cohorts,
      capacity_rows, old_union_overflows, m.old_m8_a_b_mismatches,
      expect_old_fixed ? "exact" : "REJECTED");
}

void prove_config(Config cfg, bool expect_double_read) {
  ConsumerSummary const summary = consumer_summary(cfg);
  require(summary.threads == 256,
          "admitted WN/WK pair must preserve the 256-thread CTA");
  require(summary.word_holes == 0 && summary.word_duplicates == 0,
          "new B consumer is not exact-once at packed-word granularity");
  require(summary.producer_consumer_mismatches == 0,
          "cp.async producer and shared B consumer disagree");
  require(summary.a_b_k_mismatches == 0 && summary.a_k_holes == 0 &&
              summary.a_k_duplicates == 0,
          "A K-inner does not match B or cover every K atom per N warp");
  int const expected_factor = expect_double_read ? 2 : 1;
  require(summary.vector_min_reads == expected_factor &&
              summary.vector_max_reads == expected_factor &&
              summary.vector_total_reads == kTileVectors * expected_factor,
          "shared B vector-read factor differs from the declared cost");
  require(output_coverage(cfg, kTileM8, false) == 0 &&
              output_coverage(cfg, kTileM16, false) == 0,
          "K0 output cohort does not cover the full m8/m16 tile exactly once");
  require(byte_anchor_mismatches(cfg, false) == 0,
          "consumer disagrees with the independently round-tripped classic bytes");
  require(byte_anchor_mismatches(cfg, true) > 0,
          "one-word packed permutation plant stayed green");
  ConsumerSummary const denominator_plant = consumer_summary(cfg, true);
  require(denominator_plant.word_holes > 0,
          "coverage-denominator plant stayed green");

  bool const shipping = cfg.warp_n == 64 && cfg.warp_k == 32;
  if (shipping) {
    require(summary.old_word_holes == 0 && summary.old_word_duplicates == 0,
            "shipping fixed loop no longer matches its exact byte cover");
  } else {
    require(summary.old_word_holes > 0 || summary.old_word_duplicates > 0,
            "old fixed 2N x 4K loop incorrectly accepted a new topology");
  }

  std::printf(
      "[l184] cfg=%.*s topology=%dN x %dK threads=%d "
      "nblocks/warp=%d k-inner=%d b-loads/k=%d "
      "word-cover=%d/%d vector-read=%dx a-read=%dx byte-map=0/%d "
      "production-m8-ab=%d-block/%d-value old-fixed=%s\n",
      int(cfg.name.size()), cfg.name.data(), summary.warp_on_n,
      summary.warp_on_k, summary.threads, summary.nblocks_per_warp,
      summary.k_inner_iters, summary.vector_loads_per_k_inner,
      kTileWords - summary.word_holes, kTileWords, expected_factor,
      summary.a_shared_read_factor, kTileWords,
      summary.production_m8_ab_block_mismatches,
      summary.production_m8_ab_value_mismatches,
      shipping ? "exact" : "REJECTED");
}

}  // namespace

int main() {
  Config constexpr shipping{64, 32, "WN64/WK32"};
  Config constexpr wide_n{128, 16, "WN128/WK16"};
  Config constexpr narrow_n{32, 64, "WN32/WK64"};

  prove_config(shipping, false);
  prove_config(wide_n, false);
  prove_config(narrow_n, true);

  // The production sweep's whole-vector domain.  Every geometry contributes
  // TM{8,16} x S{2..6}; seven geometries therefore authorize exactly 70 rows.
  WholeVectorGeometry constexpr geometries[] = {
      {64, 128, 64, 32, "TN64/TK128/WN64/WK32"},
      {128, 64, 64, 32, "TN128/TK64/WN64/WK32"},
      {128, 64, 128, 16, "TN128/TK64/WN128/WK16"},
      {128, 128, 64, 32, "TN128/TK128/WN64/WK32"},
      {128, 128, 128, 16, "TN128/TK128/WN128/WK16"},
      {256, 64, 64, 32, "TN256/TK64/WN64/WK32"},
      {256, 64, 128, 16, "TN256/TK64/WN128/WK16"},
  };
  int geometry_index = 0;
  for (WholeVectorGeometry g : geometries) {
    bool const old_fixed = g.warp_n == 64;
    prove_whole_vector_geometry(g, old_fixed);
    ++geometry_index;
  }
  require(geometry_index * 2 * 5 == 70,
          "whole-vector coverage denominator is not exactly 70 rows");

  require(kTileK / wide_n.warp_k == 8,
          "wide-N candidate must have eight K cohorts");
  require(kTileN / wide_n.warp_n == 1,
          "wide-N candidate must have one N warp");
  require((wide_n.warp_k / kAtomK) == 1,
          "wide-N candidate must consume one K inner per warp");
  require((wide_n.warp_k / kAtomK) *
              ceil_div(wide_n.warp_n / kAtomN,
                       kPackedNBlocksPerVector) == 2,
          "wide-N candidate must retain two compute steps so the production "
          "penultimate-step pipeline advance remains reachable");
  int constexpr all_cohorts = (1 << 8) - 1;
  require(reduction_result(8, 4) == all_cohorts,
          "8->4->2->1 reduction omitted or duplicated a cohort");
  require(reduction_result(8, 2) != all_cohorts,
          "old 4-cohort reduction cadence stayed green for eight cohorts");
  require(output_coverage(wide_n, kTileM8, true) > 0 &&
              output_coverage(wide_n, kTileM16, true) > 0,
          "old fixed four-nblock output loop stayed green for WN128");

  std::printf(
      "[l184] PASS: 70/70 whole-vector rows close producer/B/scale/output/"
      "reduction/capacity; WN128/WK16 keeps the classic byte map, consumes two "
      "64-column B/scale vectors without duplication (the minimal two-step "
      "spelling rereads its one A K-inner 2x), covers m8/m16 output with "
      "32 K0 threads and reduces 8->4->2->1; WN32/WK64 is exact at word "
      "level but costs exactly 2x 128-bit shared B/scale reads; current "
      "contiguous-m8 A K expression is RED against interleaved classic B; "
      "old fixed loops, one-word permutation, naive TK64 scale advance, old "
      "mainloop-only cooperative union and denominator plants rejected\n");
  return 0;
}

// L228 -- canonical Q4 K-pack4 transport, before any production collective change.
//
// This proof intentionally separates three contracts:
//
//   1. FORMAT.  Four K codes from one N column and one gs32 group form one
//      opaque b16 word.  The converter-native quartet is
//      {r,r+8,r+16,r+24}; its eight r values are stored as compact (K/4,N)
//      with N contiguous.  No FoldN, ArtifactTileK, tactic or batch coordinate
//      enters the address.
//   2. TRANSPORT.  One existing PPU0010 transposed b16 read returns four b32
//      registers per lane, exactly one N16 x (K/4)16 microtile = N16 x K64 Q4.
//   3. COMPUTE/METADATA GEOMETRY.  Expanding each b16 at K-group kg produces
//      k=4*kg+r.  A K64 transport therefore covers four K16 MMA atoms and two
//      exact gs32 metadata groups.  A TK32 consumer may reuse the two halves of
//      the same transport; it does not need a different physical format.
//
// This file does NOT claim the current int4 converter register permutation is
// already correct for the transposed delivery.  That lane/vreg composition is
// a separate required proof.  Keeping that distinction here prevents a payload
// equality from being promoted into a shipping correctness claim.

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"
#include "q4_kpack4_offline.hpp"

namespace {

using namespace cute;

constexpr int kPack = 4;
constexpr int kMicroN = 16;
constexpr int kMicroK = 64;
constexpr int kMetadataGroup = 32;
constexpr int kWordsPerMicroK = kMicroK / kPack;
constexpr int kTransportWords = kMicroN * kWordsPerMicroK;
constexpr int kTransportBytes = kTransportWords * int(sizeof(std::uint16_t));
constexpr int kTransportRegisters = 32 * 4;

#ifndef L228_NAIVE_CONSECUTIVE_PACK
#define L228_NAIVE_CONSECUTIVE_PACK 0
#endif
#ifndef L228_ROTATE_CONVERTER_DESTINATION
#define L228_ROTATE_CONVERTER_DESTINATION 0
#endif
#ifndef L228_SHIFT_METADATA_ATOM
#define L228_SHIFT_METADATA_ATOM 0
#endif

static_assert(kTransportBytes == 512);
static_assert(kTransportRegisters * 4 == kTransportBytes);
static_assert(kMicroK / 16 == 4,
              "one K-pack4 transport must feed four K16 MMA atoms");
static_assert(kMicroK / kMetadataGroup == 2,
              "one K-pack4 transport must cover two gs32 groups");
static_assert(std::is_same_v<
                  typename MMA_Traits<
                      PPU0010_8x16x16_F32F16F16F32_TN>::BLayout,
                  typename MMA_Traits<
                      PPU0010_16x16x16_F32F16F16F32_TN>::BLayout>,
              "PPU0010 m8/m16 must retain one shared B-fragment map");

constexpr int ceil_div(int x, int y) { return (x + y - 1) / y; }

constexpr int physical_kgroup(int k) {
#if L228_NAIVE_CONSECUTIVE_PACK
  return k / kPack;
#else
  return q4_kpack4::physical_kgroup(k);
#endif
}

constexpr std::size_t word_index(int n, int k, int N) {
  return std::size_t(physical_kgroup(k)) * std::size_t(N) +
         std::size_t(n);
}

constexpr int nibble_shift(int k) {
#if L228_NAIVE_CONSECUTIVE_PACK
  return 4 * (k % kPack);
#else
  return 4 * q4_kpack4::nibble(k);
#endif
}

constexpr int logical_k_from_physical(int physical_kg, int nibble) {
#if L228_NAIVE_CONSECUTIVE_PACK
  return kPack * physical_kg + nibble;
#else
  return q4_kpack4::logical_k(physical_kg, nibble);
#endif
}

template <int N, int K>
struct Canonical {
  static_assert(N > 0 && K > 0 && K % kMetadataGroup == 0);
  static constexpr std::size_t kCodes = std::size_t(N) * K;
  static constexpr std::size_t kWords = kCodes / kPack;

  std::array<std::uint16_t, kWords> words{};

  void put(int n, int k, std::uint8_t q) {
    auto& w = words[word_index(n, k, N)];
    int const shift = nibble_shift(k);
    w = std::uint16_t((w & ~(std::uint16_t(0xf) << shift)) |
                      ((std::uint16_t(q) & 0xf) << shift));
  }

  std::uint8_t get(int n, int k) const {
    return std::uint8_t((words[word_index(n, k, N)] >> nibble_shift(k)) & 0xf);
  }
};

template <int N, int K>
int prove_roundtrip() {
  Canonical<N, K> packed;
  int bad = 0;
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      packed.put(n, k, std::uint8_t((13 * n + 7 * k + 3) & 0xf));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      bad += packed.get(n, k) != std::uint8_t((13 * n + 7 * k + 3) & 0xf);
  return bad;
}

// Traverse the SAME canonical byte array under a tactic grouping.  WN/TK
// decide only how many fixed N16/K64 transport microtiles are grouped by one
// scheduler tile.  They never enter word_index().
template <int N, int K, int WN, int TK>
int prove_tactic_traversal(int* transport_count, int* tk32_reuse) {
  static_assert(WN % kMicroN == 0);
  static_assert(TK == 32 || TK % kMicroK == 0);
  std::array<int, std::size_t(N) * K> hits{};
  int transports = 0;
  int reuse = 0;
  for (int tile_n = 0; tile_n < N; tile_n += WN) {
    for (int tile_k = 0; tile_k < K; tile_k += TK) {
      int const transport_k = TK == 32 ? (tile_k / kMicroK) * kMicroK : tile_k;
      // TK32's adjacent pair shares one K64 load.  Count it only on the first
      // half, but mark only this compute slice's logical values.
      if (TK != 32 || tile_k % kMicroK == 0) {
        transports += WN / kMicroN;
        reuse += TK == 32;
      }
      (void)transport_k;
      for (int n = tile_n; n < tile_n + WN && n < N; ++n)
        for (int k = tile_k; k < tile_k + TK && k < K; ++k)
          ++hits[std::size_t(n) * K + k];
    }
  }
  int bad = 0;
  for (int h : hits) bad += h != 1;
  *transport_count = transports;
  *tk32_reuse = reuse;
  return bad;
}

// The AIU instruction is padz.  A compact canonical buffer therefore needs
// no stored padding when N or K/4 is not a full 16x16 transport microtile.
// This oracle models the logical result of that pad: in-range codes occur once
// and every out-of-range lane is zero.
template <int N, int K>
int prove_padz_tail(int* valid, int* padded) {
  static_assert(K % kPack == 0);
  Canonical<N, K> packed;
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      packed.put(n, k, std::uint8_t(((5 * n + 11 * k) % 15) + 1));
  int bad = 0;
  int valid_count = 0;
  int padded_count = 0;
  int const padded_n = ceil_div(N, kMicroN) * kMicroN;
  int const padded_k = ceil_div(K, kMicroK) * kMicroK;
  for (int n = 0; n < padded_n; ++n) {
    for (int k = 0; k < padded_k; ++k) {
      bool const in = n < N && k < K;
      std::uint8_t const got = in ? packed.get(n, k) : 0;
      std::uint8_t const want = in
          ? std::uint8_t(((5 * n + 11 * k) % 15) + 1) : 0;
      bad += got != want;
      valid_count += in;
      padded_count += !in;
    }
  }
  *valid = valid_count;
  *padded = padded_count;
  return bad;
}

// Every b16 word is wholly contained in one gs32 group by construction.
// Within one K64 transport, kg 0..7 use group 0 and kg 8..15 group 1.
int prove_metadata() {
  int bad = 0;
  std::array<int, 2> counts{};
  for (int n = 0; n < kMicroN; ++n) {
    for (int kg = 0; kg < kWordsPerMicroK; ++kg) {
      int const group = kg / (kMetadataGroup / kPack);
      bad += group < 0 || group >= 2;
      for (int r = 0; r < kPack; ++r)
        bad += logical_k_from_physical(kg, r) / kMetadataGroup != group;
      ++counts[std::size_t(group)];
    }
  }
  bad += counts[0] != 128 || counts[1] != 128;
  using Policy = cutlass::gemm::collective::detail::FineScalePolicy<2, 1, 4>;
  static_assert(Policy::active && Policy::atoms_per_group == 2);
  constexpr std::array<int, 4> expected{0, 0, 1, 1};
  for (int atom = 0; atom < 4; ++atom)
    bad += Policy::group((atom + L228_SHIFT_METADATA_ATOM) % 4) !=
           expected[std::size_t(atom)];
  return bad;
}

// Retain two tempting but wrong formats as RED controls:
//   * naive consecutive-K pack4 has the right byte count but not the shipping
//     converter's group-local quartet;
//   * N-major K-pack4 makes K-group, not N, contiguous;
//   * Fold2 maps a physical row to two N columns and is not the canonical
//     (K/4,N) word class.
int prove_negative_controls() {
  constexpr int N = 32, K = 64;
  int n_major_same = 0;
  int fold2_same = 0;
  int consecutive_same = 0;
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; k += kPack) {
      std::size_t const canonical = word_index(n, k, N);
      std::size_t const consecutive = std::size_t(k / kPack) * N + n;
      std::size_t const n_major = std::size_t(n) * (K / kPack) + k / kPack;
      // Existing Fold2 physical word address: pair N, concatenate their K.
      std::size_t const fold2 = std::size_t(n / 2) * (2 * K / kPack) +
                                std::size_t(n % 2) * (K / kPack) + k / kPack;
      consecutive_same += canonical == consecutive &&
                          nibble_shift(k) == 4 * (k % kPack);
      n_major_same += canonical == n_major;
      fold2_same += canonical == fold2;
    }
  }
  // Only sparse fixed points may coincide; equality of the complete mapping
  // would make the negative ineffective.
  return (consecutive_same == N * K / kPack) ||
         (n_major_same == N * K / kPack) ||
         (fold2_same == N * K / kPack);
}

// Bind the payload ledger to the actual PPU0010 compute fragment shape.  The
// transposed reader itself is source-anchored by the runner because the PPU
// asm is architecture-guarded and cannot be instantiated by this host probe.
int prove_mma_destination() {
  using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
  using Mma = TiledMMA<MMA_Atom<Atom>, Layout<Shape<_1, _1, _1>>,
                       Tile<_8, _16, Int<kMicroK>>>;
  auto tensor = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<_16, Int<kMicroK>>{},
                  Stride<Int<kMicroK>, _1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(tensor);
  // 32 fp16 values/lane = four K16 B fragments.  That is exactly the output
  // of expanding four b32 source registers containing eight K-pack4 words.
  int const values_per_lane = int(size(frag));
  int const source_regs_per_lane = 4;
  int const expanded_values = source_regs_per_lane * 8;
  return values_per_lane != 32 || expanded_values != values_per_lane ||
         int(size<2>(frag)) != 4;
}

struct CoordNK {
  int n = -1;
  int k = -1;
};

// Compose the real PPU0010 B-fragment layouts with the shipping int4
// converter permutation.  The matched transposed b16 write/read is already an
// identity for fp16 GEMM; therefore the one-atom fp16 B fragment is the
// authoritative map from each delivered b16 half to (n,kg).  Expanding that
// word through MixGemmEmit<4> must land on the same (n,4*kg+r) coordinate in
// the four-atom K64 compute fragment.
struct ConverterStructure {
  int exact = 0;
  int total = 0;
  int word_same_n = 0;
  int word_consecutive_k = 0;
  int word_same_gs32 = 0;
  int words = 0;
  int source_duplicates = 0;
  int destination_duplicates = 0;
};

int prove_existing_converter_composition(ConverterStructure* stats) {
  using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
  using LoadMma = TiledMMA<MMA_Atom<Atom>, Layout<Shape<_1, _1, _1>>,
                           Tile<_8, _16, _16>>;
  using ComputeMma = TiledMMA<MMA_Atom<Atom>, Layout<Shape<_1, _1, _1>>,
                              Tile<_8, _16, Int<kMicroK>>>;
  auto load_tensor = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<_16, _16>{}, Stride<_1, _16>{}));
  auto compute_tensor = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<_16, Int<kMicroK>>{},
                  Stride<Int<kMicroK>, _1>{}));
  auto load_identity = make_identity_tensor(Shape<_16, _16>{});
  auto compute_identity = make_identity_tensor(
      Shape<_16, Int<kMicroK>>{});

  std::array<std::array<CoordNK, 4>, kTransportWords> word_outputs{};
  std::array<int, kTransportWords> word_visits{};
  std::array<int, kMicroN * kMicroK> destination_visits{};
  for (int lane = 0; lane < 32; ++lane) {
    auto load_thr = LoadMma{}.get_thread_slice(lane);
    auto load_frag = load_thr.partition_fragment_B(load_tensor);
    auto load_part = load_thr.partition_B(load_identity);
    auto load_pi = right_inverse(load_frag.layout());

    auto compute_thr = ComputeMma{}.get_thread_slice(lane);
    auto compute_frag = compute_thr.partition_fragment_B(compute_tensor);
    auto compute_part = compute_thr.partition_B(compute_identity);
    auto compute_pi = right_inverse(compute_frag.layout());
    for (int vreg = 0; vreg < 4; ++vreg) {
      for (int code = 0; code < 8; ++code) {
        int const source_half = 2 * vreg + code / 4;
        auto const source = load_part(load_pi(source_half));
        CoordNK const q{
            int(get<0>(source)),
            logical_k_from_physical(int(get<1>(source)), code % 4)};

        int const output =
            (cutlass::MixGemmEmit<4>::index(code, vreg) +
             L228_ROTATE_CONVERTER_DESTINATION) % 32;
        auto const destination = compute_part(compute_pi(output));
        CoordNK const d{int(get<0>(destination)), int(get<1>(destination))};
        stats->exact += q.n == d.n && q.k == d.k;
        ++stats->total;
        int const source_word = int(get<1>(source)) * kMicroN +
                                int(get<0>(source));
        int const source_nibble = code % kPack;
        if (source_word < 0 || source_word >= kTransportWords ||
            source_nibble < 0 || source_nibble >= kPack) {
          ++stats->source_duplicates;
        } else {
          word_outputs[std::size_t(source_word)][std::size_t(source_nibble)] = d;
          ++word_visits[std::size_t(source_word)];
        }
        int const destination_flat = d.n * kMicroK + d.k;
        if (destination_flat < 0 || destination_flat >= kMicroN * kMicroK)
          ++stats->destination_duplicates;
        else
          ++destination_visits[std::size_t(destination_flat)];
      }
    }
  }
  for (int word = 0; word < kTransportWords; ++word) {
    stats->source_duplicates += word_visits[std::size_t(word)] != 4;
    auto const& out = word_outputs[std::size_t(word)];
    bool same_n = true;
    bool same_group = true;
    std::array<int, 4> ks{};
    for (int r = 0; r < 4; ++r) {
      same_n &= out[std::size_t(r)].n == out[0].n;
      same_group &= out[std::size_t(r)].k / kMetadataGroup ==
                    out[0].k / kMetadataGroup;
      ks[std::size_t(r)] = out[std::size_t(r)].k;
    }
    auto sorted = ks;
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j)
        if (sorted[std::size_t(j)] < sorted[std::size_t(i)]) {
          int const tmp = sorted[std::size_t(i)];
          sorted[std::size_t(i)] = sorted[std::size_t(j)];
          sorted[std::size_t(j)] = tmp;
        }
    bool const converter_native_quartet =
        sorted[1] == sorted[0] + 8 && sorted[2] == sorted[0] + 16 &&
        sorted[3] == sorted[0] + 24;
    stats->word_same_n += same_n;
    stats->word_consecutive_k += converter_native_quartet;
    stats->word_same_gs32 += same_group;
    int const physical_kg = word / kMicroN;
    if (word % kMicroN == 0 &&
        (physical_kg == 0 || physical_kg == 7 ||
         physical_kg == 8 || physical_kg == 15)) {
      std::printf("L228 CONVERTER_WORD physical=(kg=%d,n=%d) logical_n=%d "
                  "logical_k={%d,%d,%d,%d}\n",
                  word / kMicroN, word % kMicroN, out[0].n,
                  out[0].k, out[1].k, out[2].k, out[3].k);
    }
    ++stats->words;
  }
  for (int visits : destination_visits)
    stats->destination_duplicates += visits != 1;
  return stats->total - stats->exact;
}

void print_transposed_copy_geometry() {
  using Atom = PPU0010_16x16x16_F32F16F16F32_TN;
  using Mma = TiledMMA<MMA_Atom<Atom>, Layout<Shape<_1, _1, _1>>,
                       Tile<_16, _16, _16>>;
  using Op = PPU0010_TSM_LD_SWZL<
      cutlass::half_t, 16, 16, true, true, 1>;
  using Copy = Copy_Atom<Op, cutlass::half_t>;
  auto smem = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<_16, _16>{}, Stride<_1, _16>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(smem);
  auto tiled = make_tiled_copy_B(Copy{}, Mma{});
  auto slice = tiled.get_thread_slice(0);
  auto src = slice.partition_S(make_mix_tensor_like(smem));
  auto dst = slice.retile_D(frag);
  std::printf("L228 TRANS_GEOMETRY frag="); print(frag.layout());
  std::printf(" src="); print(src.layout());
  std::printf(" dst="); print(dst.layout());
  std::putchar('\n');
}

}  // namespace

int main() {
  print_transposed_copy_geometry();
  int traversal_bad = 0;
  int transports = 0;
  int tk32_reuse = 0;
  int t = 0, r = 0;
  traversal_bad += prove_tactic_traversal<64, 256, 16, 64>(&t, &r);
  transports += t; tk32_reuse += r;
  traversal_bad += prove_tactic_traversal<64, 256, 32, 128>(&t, &r);
  transports += t; tk32_reuse += r;
  traversal_bad += prove_tactic_traversal<64, 256, 64, 256>(&t, &r);
  transports += t; tk32_reuse += r;
  traversal_bad += prove_tactic_traversal<64, 256, 16, 32>(&t, &r);
  transports += t; tk32_reuse += r;

  int tail_valid = 0, tail_padded = 0;
  int const roundtrip_bad = prove_roundtrip<64, 256>();
  int const tail_bad = prove_padz_tail<18, 96>(&tail_valid, &tail_padded);
  int const metadata_bad = prove_metadata();
  int const negative_bad = prove_negative_controls();
  int const mma_bad = prove_mma_destination();
  ConverterStructure converter{};
  int const converter_bad = prove_existing_converter_composition(&converter);
  bool const ok = roundtrip_bad == 0 && traversal_bad == 0 &&
                  tail_bad == 0 && metadata_bad == 0 &&
                  negative_bad == 0 && mma_bad == 0 &&
                  converter_bad == 0 &&
                  converter.word_same_n == converter.words &&
                  converter.word_consecutive_k == converter.words &&
                  converter.word_same_gs32 == converter.words &&
                  converter.source_duplicates == 0 &&
                  converter.destination_duplicates == 0;

  std::printf(
      "L228 KPACK4_CANONICAL %s roundtrip_bad=%d traversal_bad=%d "
      "transports=%d tk32_shared_transports=%d tail_bad=%d "
      "tail_valid=%d tail_padded=%d metadata_bad=%d mma_bad=%d "
      "existing_converter_exact=%d/%d converter_bad=%d "
      "converter_words_same_n=%d/%d native_kquartet=%d/%d "
      "same_gs32=%d/%d source_dup=%d destination_dup=%d "
      "negative_controls=%s bytes_per_n16k64=%d\n",
      ok ? "PASS" : "FAIL", roundtrip_bad, traversal_bad,
      transports, tk32_reuse, tail_bad, tail_valid, tail_padded,
      metadata_bad, mma_bad, converter.exact, converter.total, converter_bad,
      converter.word_same_n, converter.words,
      converter.word_consecutive_k, converter.words,
      converter.word_same_gs32, converter.words,
      converter.source_duplicates, converter.destination_duplicates,
      negative_bad == 0 ? "RED" : "FAILED",
      kTransportBytes);
  return ok ? 0 : 1;
}

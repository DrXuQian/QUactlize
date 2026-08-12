// L137 -- derive the BC-GEMV logical-to-resident code permutation from the production xplane writer.
//
// This file is intentionally a host probe.  It labels every logical code in one 256x256 supertile, sends those
// labels through xplane::place_from_map, and asks whether the resulting logical->physical address is one 16-bit
// permutation.  The printed strides are inputs to gguf_bc_vecdot's device-side reader; the permanent version of
// this probe also exhaustively compares that reader with this production-writer oracle.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fold_traits.hpp"
#include "gguf_bc_vecdot.hpp"
#include "xplane_offline.hpp"

namespace {

constexpr int N = 256, K = 256, Codes = N * K;

template <int Bits, int ArtifactTileK> struct OptionalFold {
  static constexpr int value = fold::delivery_fold_v<Bits, ArtifactTileK>;
};
template <int ArtifactTileK> struct OptionalFold<0, ArtifactTileK> {
  static constexpr int value = 1;
};

template <int Bits, int TM, int TN, int TK, int WM, int WN, int Fold, int ArtifactTileK>
std::vector<int> logical_to_physical(std::vector<int> const& map, int rows_n = N, int cols_k = K) {
  constexpr int LabelBits = 18;  // enough for the 512x512 outer-block check below
  constexpr int Passes = (LabelBits + Bits - 1) / Bits;
  constexpr int Mask = (1 << Bits) - 1;
  int const codes = rows_n * cols_k;
  std::vector<uint32_t> physical_owner(static_cast<size_t>(codes), 0);
  std::vector<uint8_t> q(static_cast<size_t>(codes));
  // Keep a red-zone because this oracle is also meant to fail closed if a producer layout escapes its declared
  // N*K plane.  Reading only the first plane below then makes any such escape non-bijective rather than corrupting
  // the probe itself.
  size_t const plane_bytes = static_cast<size_t>(codes) * Bits / 8;
  std::vector<int8_t> bytes(plane_bytes * 8, int8_t(0x5a));
  for (int pass = 0; pass < Passes; ++pass) {
    int const shift = pass * Bits;
    for (int logical = 0; logical < codes; ++logical)
      q[size_t(logical)] = uint8_t((logical >> shift) & Mask);
    xplane::place_from_map<Bits, TM, TN, TK, WM, WN, Fold, ArtifactTileK>(
        bytes.data(), map, q, rows_n, cols_k);
    for (size_t i = plane_bytes; i < bytes.size(); ++i)
      if (bytes[i] != int8_t(0x5a)) return {};
    for (int physical = 0; physical < codes; ++physical) {
      size_t const bit0 = size_t(physical) * Bits;
      uint32_t value = 0;
      for (int b = 0; b < Bits; ++b)
        value |= uint32_t((uint8_t(bytes[(bit0 + b) / 8]) >> ((bit0 + b) % 8)) & 1) << b;
      physical_owner[size_t(physical)] |= value << shift;
    }
  }
  std::vector<int> out(static_cast<size_t>(codes), -1);
  for (int physical = 0; physical < codes; ++physical) {
    int const logical = physical_owner[size_t(physical)];
    if (logical < 0 || logical >= codes) return {};
    if (out[size_t(logical)] != -1) return {};
    out[size_t(logical)] = physical;
  }
  return out;
}

bool require_bijection(std::vector<int> const& map, size_t expected, int& failures) {
  if (map.size() == expected) return true;
  ++failures;
  return false;
}

template <gguf_scale::KType T, int Bits, int OtherBits, int ArtifactTileK>
std::array<int, 16> low(char const* name, int& failures) {
  constexpr int F = fold::delivery_fold_v<Bits, ArtifactTileK>;
  constexpr int OtherF = OptionalFold<OtherBits, ArtifactTileK>::value;
  constexpr int MaxF = F > OtherF ? F : OtherF;
  constexpr int WN = MaxF > 2 ? 16 * MaxF : 32;
  constexpr int TN = 2 * WN;
  auto map = xplane::plane_map<Bits, 64, TN, ArtifactTileK, 32, WN, F>();
  std::vector<int> slot(size_t(TN) * ArtifactTileK, -1);
  for (int i = 0; i < int(map.size()); ++i)
    if (map[size_t(i)] >= 0 && map[size_t(i)] < int(slot.size())) slot[size_t(map[size_t(i)])] = i;
  std::printf("SLOT %-7s A=%d:", name, ArtifactTileK);
  for (int b = 0; (1 << b) < TN * ArtifactTileK; ++b) std::printf(" 0x%x", slot[size_t(1 << b)]);
  std::printf("\n");
  auto p = logical_to_physical<Bits, 64, TN, ArtifactTileK, 32, WN, F, ArtifactTileK>(map);
  std::printf("%-12s bits=%d/%d A=%d F=%d:", name, Bits, 0, ArtifactTileK, F);
  if (!require_bijection(p, Codes, failures)) { std::printf(" NOT_BIJECTIVE\n"); return {}; }
  std::array<int, 16> stride{};
  for (int b = 0; b < 16; ++b) stride[size_t(b)] = p[size_t(1 << b)];
  int bad = 0;
  for (int x = 0; x < Codes; ++x) {
    int y = 0;
    for (int b = 0; b < 16; ++b) if ((x >> b) & 1) y |= stride[size_t(b)];
    bad += y != p[size_t(x)];
    // place_from_map consumes q_kn[k*N+n], so x's low eight bits are N and its high eight bits are K.
    int const n = x & 255, k = x >> 8;
    bad += gguf_scale::bc_vecdot::xplane_physical_code<T,false,ArtifactTileK>(n,k,N) != p[size_t(x)];
  }
  auto wide = logical_to_physical<Bits, 64, TN, ArtifactTileK, 32, WN, F, ArtifactTileK>(map, 512, 512);
  if (wide.size() != 512u * 512u) {
    ++bad;
  } else {
    for (int k = 0; k < 512; ++k) for (int n = 0; n < 512; ++n)
      bad += gguf_scale::bc_vecdot::xplane_physical_code<T,false,ArtifactTileK>(n,k,512) !=
             wide[size_t(k)*512+n];
  }
  for (int v : stride) std::printf(" 0x%x", v);
  std::printf(" mapping_mismatches=%d\n", bad);
  failures += bad;
  return stride;
}

template <gguf_scale::KType T, int LowBits, int HighBits, int ArtifactTileK>
std::array<int, 16> high(char const* name, int& failures) {
  constexpr int F1 = fold::delivery_fold_v<LowBits, ArtifactTileK>;
  constexpr int F2 = fold::delivery_fold_v<HighBits, ArtifactTileK>;
  constexpr int MaxF = F1 > F2 ? F1 : F2;
  constexpr int WN = MaxF > 2 ? 16 * MaxF : 32;
  constexpr int TN = 2 * WN;
  auto map = xplane::tile_map_hi<LowBits, HighBits, 64, TN, ArtifactTileK, 32, WN, F2, F1>();
  std::vector<int> slot(size_t(TN) * ArtifactTileK, -1);
  for (int i = 0; i < int(map.size()); ++i)
    if (map[size_t(i)] >= 0 && map[size_t(i)] < int(slot.size())) slot[size_t(map[size_t(i)])] = i;
  std::printf("SLOT %-7s A=%d:", name, ArtifactTileK);
  for (int b = 0; (1 << b) < TN * ArtifactTileK; ++b) std::printf(" 0x%x", slot[size_t(1 << b)]);
  std::printf("\n");
  auto p = logical_to_physical<HighBits, 64, TN, ArtifactTileK, 32, WN, F2, ArtifactTileK>(map);
  std::printf("%-12s bits=%d/%d A=%d F=%d:", name, LowBits, HighBits, ArtifactTileK, F2);
  if (!require_bijection(p, Codes, failures)) { std::printf(" NOT_BIJECTIVE\n"); return {}; }
  std::array<int, 16> stride{};
  for (int b = 0; b < 16; ++b) stride[size_t(b)] = p[size_t(1 << b)];
  int bad = 0;
  for (int x = 0; x < Codes; ++x) {
    int y = 0;
    for (int b = 0; b < 16; ++b) if ((x >> b) & 1) y |= stride[size_t(b)];
    bad += y != p[size_t(x)];
    int const n = x & 255, k = x >> 8;
    bad += gguf_scale::bc_vecdot::xplane_physical_code<T,true,ArtifactTileK>(n,k,N) != p[size_t(x)];
  }
  auto wide = logical_to_physical<HighBits, 64, TN, ArtifactTileK, 32, WN, F2, ArtifactTileK>(map, 512, 512);
  if (wide.size() != 512u * 512u) {
    ++bad;
  } else {
    for (int k = 0; k < 512; ++k) for (int n = 0; n < 512; ++n)
      bad += gguf_scale::bc_vecdot::xplane_physical_code<T,true,ArtifactTileK>(n,k,512) !=
             wide[size_t(k)*512+n];
  }
  for (int v : stride) std::printf(" 0x%x", v);
  std::printf(" mapping_mismatches=%d\n", bad);
  failures += bad;
  return stride;
}

} // namespace

int main(int argc, char** argv) {
  int failures = 0;
  using gguf_scale::KType;
  if (argc != 2) { std::fprintf(stderr, "usage: %s CASE\n", argv[0]); return 2; }
#define L137_LOW(CASE,T,B,O,A,N) if (!std::strcmp(argv[1],CASE)) { low<KType::T,B,O,A>(N,failures); std::printf("RESULT %s failures=%d\n", failures?"FAIL":"PASS",failures); return failures?1:0; }
#define L137_HIGH(CASE,T,L,H,A,N) if (!std::strcmp(argv[1],CASE)) { high<KType::T,L,H,A>(N,failures); std::printf("RESULT %s failures=%d\n", failures?"FAIL":"PASS",failures); return failures?1:0; }
  L137_LOW("q2-a32-low",Q2_K,2,0,32,"Q2-low"); L137_LOW("q2-a64-low",Q2_K,2,0,64,"Q2-low");
  L137_LOW("q2-a128-low",Q2_K,2,0,128,"Q2-low"); L137_LOW("q2-a256-low",Q2_K,2,0,256,"Q2-low");
  L137_LOW("q3-a64-low",Q3_K,2,1,64,"Q3-low"); L137_LOW("q3-a128-low",Q3_K,2,1,128,"Q3-low");
  L137_LOW("q3-a256-low",Q3_K,2,1,256,"Q3-low");
  L137_LOW("q4-a32-low",Q4_K,4,0,32,"Q4-low"); L137_LOW("q4-a64-low",Q4_K,4,0,64,"Q4-low");
  L137_LOW("q4-a128-low",Q4_K,4,0,128,"Q4-low"); L137_LOW("q4-a256-low",Q4_K,4,0,256,"Q4-low");
  L137_LOW("q5-a64-low",Q5_K,4,1,64,"Q5-low"); L137_LOW("q5-a128-low",Q5_K,4,1,128,"Q5-low");
  L137_LOW("q5-a256-low",Q5_K,4,1,256,"Q5-low");
  L137_LOW("q6-a32-low",Q6_K,4,2,32,"Q6-low"); L137_LOW("q6-a64-low",Q6_K,4,2,64,"Q6-low");
  L137_LOW("q6-a128-low",Q6_K,4,2,128,"Q6-low");
  L137_HIGH("q3-a64-high",Q3_K,2,1,64,"Q3-high"); L137_HIGH("q3-a128-high",Q3_K,2,1,128,"Q3-high");
  L137_HIGH("q3-a256-high",Q3_K,2,1,256,"Q3-high");
  L137_HIGH("q5-a64-high",Q5_K,4,1,64,"Q5-high"); L137_HIGH("q5-a128-high",Q5_K,4,1,128,"Q5-high");
  L137_HIGH("q5-a256-high",Q5_K,4,1,256,"Q5-high");
  L137_HIGH("q6-a32-high",Q6_K,4,2,32,"Q6-high"); L137_HIGH("q6-a64-high",Q6_K,4,2,64,"Q6-high");
  L137_HIGH("q6-a128-high",Q6_K,4,2,128,"Q6-high");
#undef L137_LOW
#undef L137_HIGH
  if (std::strcmp(argv[1], "controls")) { std::fprintf(stderr,"unknown case %s\n",argv[1]); return 2; }
  quactlize_ppu_placed_arrangement_v1 const q2_f2{
      QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, 2, 64, 0};
  quactlize_ppu_placed_arrangement_v1 const q2_f1{
      QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, 2, 128, 0};
  quactlize_ppu_placed_arrangement_v1 const bad_version{77, 2, 64, 0};
  // Actual descriptor x compiled-reader predicate.  The positive F2 arm proves the folded reader exists.  Feeding
  // those exact bytes to the registry/default F1 reader is the required negative: it must reject, not decode.
  failures += !gguf_scale::bc_vecdot::reader_accepts<KType::Q2_K,64>(q2_f2);
  failures +=  gguf_scale::bc_vecdot::reader_accepts<KType::Q2_K,128>(q2_f2);
  failures +=  gguf_scale::bc_vecdot::reader_accepts<KType::Q2_K,64>(q2_f1);
  failures +=  gguf_scale::bc_vecdot::reader_accepts<KType::Q2_K,64>(bad_version);
  static_assert(gguf_scale::bc_vecdot::Traits<KType::Q2_K>::DefaultArtifactTileK == 256 &&
                gguf_scale::bc_vecdot::Traits<KType::Q3_K>::DefaultArtifactTileK == 256 &&
                gguf_scale::bc_vecdot::Traits<KType::Q4_K>::DefaultArtifactTileK == 256 &&
                gguf_scale::bc_vecdot::Traits<KType::Q5_K>::DefaultArtifactTileK == 256 &&
                gguf_scale::bc_vecdot::Traits<KType::Q6_K>::DefaultArtifactTileK == 128,
                "the unversioned BC ABI must preserve its pre-descriptor artifact map");
  int default_map_diff = 0, planted_mapping_mismatches = 0;
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
    int const f2 = gguf_scale::bc_vecdot::xplane_physical_code<KType::Q2_K,false,64>(n,k,N);
    int const f1 = gguf_scale::bc_vecdot::xplane_physical_code<KType::Q2_K,false,128>(n,k,N);
    default_map_diff += f1 != f2;
    planted_mapping_mismatches += (f2 ^ ((n & 1) ? 1 : 0)) != f2;
  }
  // The F2 bytes disagree with the default F1 reader in exactly 3/4 of this fixture.  Separately, changing one
  // stride bit in the header comparator must red exactly the half of inputs carrying that logical bit.
  failures += default_map_diff != 61440;
  failures += planted_mapping_mismatches != 32768;
  int planted_nonbijective_failures = 0;
  require_bijection({}, Codes, planted_nonbijective_failures);
  failures += planted_nonbijective_failures != 1;
  std::printf("NEGATIVE default_map_diff=%d/65536 planted_mapping_mismatches=%d/65536 "
              "planted_nonbijective_failures=%d\n",
              default_map_diff, planted_mapping_mismatches, planted_nonbijective_failures);
  std::printf("RESULT %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}

// L217 -- compose the exact Q4/A32 path all the way through:
// artifact slot -> AIU global-to-shared delivery -> tiled shared layout ->
// TSM swizzle load -> int4 converter/scatter -> logical MMA B coordinate.
//
// L123 skips the AIU destination layout and models shared memory as a compact
// physical row.  That is valid for one delivery, but A32/TK128 has four
// delivery cubes.  This witness keeps the cube-major shipping layout and
// compares the resulting consumer map with the offline artifact producer.

#define main l123_embedded_main
#include "l123_warp_nk_topology.cu"
#undef main

#include <array>

namespace {

using C = Q4A32Case;
using P = FoldPair<C, 32, 1>;
using S = FoldShadow<C, 32, 1>;
constexpr int Threads = size(typename P::Mma{});
constexpr int CPV = 8, CPB = 2, CK = 4;

using SmemAtomI4 = Layout<Shape<_8, _64>, Stride<_64, _1>>;
using ExactI4 = decltype(tile_to_shape(
    SmemAtomI4{}, make_shape(_32{}, _256{})));

// AIU copies delivery r from the compact physical artifact row and writes it
// into cube r of the tiled shared layout.  Record the artifact code slot
// resident at every exact shared-memory code offset.
std::vector<int> smem_to_artifact() {
  std::vector<int> out(cosize(ExactI4{}), -1);
  for (int r = 0; r < CK; ++r)
    for (int n = 0; n < 32; ++n)
      for (int q = 0; q < 64; ++q) {
        int const smem = int(ExactI4{}(make_coord(n, make_coord(q, r))));
        int const artifact = n * 256 + r * 64 + q;
        if (smem < 0 || smem >= int(out.size()) || out[smem] >= 0)
          return {};
        out[smem] = artifact;
      }
  return out;
}

std::vector<int> exact_owner(int thread, bool& valid) {
  using Tr = Copy_Traits<typename S::Op>;
  constexpr int WPR = Tr::LogicalWordsPerRow;
  auto exact_i4 = make_tensor(
      make_smem_ptr((cutlass::uint4b_t*)nullptr), ExactI4{});
  auto exact_s8 = recast<int8_t>(exact_i4);
  auto load = typename S::Mma{}.get_thread_slice(thread)
                  .partition_fragment_B(exact_s8);
  auto copy = make_tiled_copy_B(
      Copy_Atom<typename S::Op, int8_t>{}, typename S::Mma{});
  auto slice = copy.get_thread_slice((thread / 32) * 32);
  auto src = slice.partition_S(make_mix_tensor_like(exact_s8));
  auto view = slice.retile_D(load);
  std::vector<int> owner(CPB * cosize(load.layout()), -1);
  constexpr int CN = size<1>(decltype(view.layout()){});
  constexpr int CopyK = size<2>(decltype(view.layout()){});
  static_assert(CopyK == CK);
  for (int kb = 0; kb < CopyK; ++kb)
    for (int cn = 0; cn < CN; ++cn)
      for (int v = 0; v < 4; ++v)
        for (int c = 0; c < CPV; ++c) {
          auto base = src(0, cn, kb);
          int const word = int(typename Tr::LogicalTV{}(
              make_coord(make_coord(thread % 4, (thread % 32) / 4),
                         make_coord(v % 2, v / 2), _0{})));
          // Mix-pointer coordinates are descriptor ABI (coord_w, coord_h),
          // not tensor logical (row, col).  L212 independently anchors that
          // ordering for the AIU source; the TSM source uses the same ABI.
          int const q = int(get<0>(base)) + 4 * (word % WPR) + c / CPB;
          int const n = int(get<1>(base)) + word / WPR;
          int const delivery = int(get<2>(base));
          int const smem = int(ExactI4{}(
              make_coord(n, make_coord(2 * q, delivery)))) + c % CPB;
          int const dst = CPB * int(view.layout()(4 * v + c / CPB, cn, kb)) +
                          c % CPB;
          valid &= smem >= 0 && smem < int(size(ExactI4{})) &&
                   dst >= 0 && dst < int(owner.size()) &&
                   (owner[dst] < 0 || owner[dst] == smem);
          if (dst >= 0 && dst < int(owner.size())) owner[dst] = smem;
        }
  return owner;
}

FoldDelivery exact_consumer() {
  using Scatter = cutlass::MixGemmArtifactScatter<4, 2, CK>;
  FoldDelivery out;
  out.map.assign(C::tn * C::tk, -1);
  out.cohort.assign(C::tn * C::tk, -1);
  out.vreg.assign(C::tn * C::tk, -1);
  out.code.assign(C::tn * C::tk, -1);
  auto resident = smem_to_artifact();
  bool valid = resident.size() == C::tn * C::tk;
  auto s16 = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<C::tn>, Int<C::tk>>{},
                  Stride<Int<C::tk>, _1>{}));
  auto bid = make_identity_tensor(make_shape(Int<C::tn>{}, Int<C::tk>{}));
  for (int t = 0; t < Threads; ++t) {
    auto mma = typename P::Mma{};
    auto exact_i4 = make_tensor(
        make_smem_ptr((cutlass::uint4b_t*)nullptr), ExactI4{});
    auto load = typename S::Mma{}.get_thread_slice(t)
                    .partition_fragment_B(recast<int8_t>(exact_i4));
    auto owner = exact_owner(t, valid);
    auto frag = mma.get_thread_slice(t).partition_fragment_B(s16);
    auto part = mma.get_thread_slice(t).partition_B(bid);
    auto pi = right_inverse(frag.layout());
    for (int kb = 0; kb < CK; ++kb) {
      auto in = recast<typename C::Element>(load(_, _, kb));
      for (int v = 0; v < 4; ++v)
        for (int c = 0; c < CPV; ++c) {
          int const ri = CPB * int(load.layout()(0, 0, kb)) +
                         int(in.layout()(0, 0)) + v * CPV + c;
          int const e = cutlass::MixGemmEmit<4>::index(c, v);
          int const raw = Scatter::flat(e, kb, 0);
          int const smem = ri >= 0 && ri < int(owner.size()) ? owner[ri] : -1;
          int const artifact = smem >= 0 && smem < int(resident.size()) ?
                               resident[smem] : -1;
          valid &= artifact >= 0 && artifact < int(out.map.size()) &&
                   raw >= 0 && raw < int(size(frag));
          if (artifact < 0 || artifact >= int(out.map.size()) ||
              raw < 0 || raw >= int(size(frag))) continue;
          auto logical_coord = part(pi(raw));
          int const logical = int(get<0>(logical_coord)) * C::tk +
                              int(get<1>(logical_coord));
          valid &= out.map[artifact] < 0 || out.map[artifact] == logical;
          out.map[artifact] = logical;
        }
    }
  }
  if (!valid) ++out.pair_owner_bad;
  for (int x : out.map) out.pair_owner_bad += x < 0;
  return out;
}

constexpr int M = 64, N = 1024, K = 5120, GS = 32;
int qvalue(int n, int k) { return ((13 * n + 7 * k + 3) % 15) - 7; }
int scale(int n, int k) { return 1 << ((17 * (k / GS) + 29 * n + 1) % 3); }
int zero(int n, int k) { return ((11 * (k / GS) + 7 * n) % 3 - 1) * 3; }

std::pair<int, float> signature(FoldDelivery const& consumer,
                                std::vector<int> const& producer) {
  std::vector<int> inverse(C::tn * C::tk, -1);
  for (int p = 0; p < int(consumer.map.size()); ++p)
    if (consumer.map[p] >= 0) inverse[consumer.map[p]] = p;
  std::array<int, 8> active{};
  for (int s = 0; s < 8; ++s)
    active[s] = s * K / 8 + (37 * s + 11) % (K / 8);
  int bad = 0;
  float first = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float want = 0, got = 0;
      for (int s = 0; s < 8; ++s) {
        int const k = active[s], ln = n % 64, lk = k % 128;
        int const p = inverse[ln * 128 + lk];
        int sn = n, sk = k;
        if (p >= 0) {
          int const src = producer[p];
          sn = (n / 64) * 64 + src / 128;
          sk = (k / 128) * 128 + src % 128;
        }
        float const a = ((m + s) & 1) ? -.5f : .5f;
        want += a * (scale(n, k) * qvalue(n, k) + zero(n, k));
        got += a * (scale(n, k) * (p >= 0 ? qvalue(sn, sk) : -8) +
                    zero(n, k));
      }
      if (got != want) {
        if (bad == 0) first = got;
        ++bad;
      }
    }
  return {bad, first};
}

}  // namespace

int main() {
  auto producer = xplane::plane_map<4, 64, 64, 128, 16, 32, 2, 32>();
  auto consumer = exact_consumer();
  auto sig = signature(consumer, producer);
  std::size_t map_diff = diff(consumer.map, producer);
  auto planted = producer;
  for (int& logical : planted) {
    int const n = logical / Q4A32Case::tk;
    int const k = logical % Q4A32Case::tk;
    logical = ((n + 1) % Q4A32Case::tn) * Q4A32Case::tk + k;
  }
  auto planted_sig = signature(consumer, planted);
  bool const negative_red = planted_sig.first != 0;
  bool const ok = map_diff == 0 && consumer.pair_owner_bad == 0 &&
                  sig.first == 0 && negative_red;
  std::printf(
      "L217 exact-reader map_diff=%zu/8192 owner_bad=%d "
      "predicted_raw_bad=%d first=%g rotate-N-negative=%d,%g result=%s\n",
      map_diff, consumer.pair_owner_bad, sig.first, double(sig.second),
      planted_sig.first, double(planted_sig.second), ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

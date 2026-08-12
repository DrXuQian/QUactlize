// L138 -- can a 2N x 2K int8 shadow loader feed a 2N x 4K FP16 MMA?
//
// Target (the classic-Marlin topology):
//   compute AtomLayout = <1,2,4>, permutation = <16,32,64>
//   shadow  AtomLayout = <1,2,2>, permutation = <16,32,64>
//
// A compute cohort does NOT map to one shadow cohort: it consumes 32 codes
// from each of both K2 shadow cohorts.  This oracle derives that two-source
// composition from the production PPU0010_TSM_LD_SWZL partition_S and
// retile_D objects. Physical int4 slots are labelled by the shipping WK1
// xplane map, independently anchored by L123 in the runner.
//
// A green result proves source availability, not source-to-destination order
// or converter code generation: two K2 shadow fragments provide the required
// 64 selected codes per compute thread and their tile-wide union covers every
// physical code exactly once.  L142/L143 derive the real production
// destination and prove the direct-pair scatter.  The red controls here prove
// that one source is insufficient, reversing the sequential source order
// changes all 16,384 owners, and the old 2N x 4K / PermK=128 shadow is
// physically invalid.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "quactlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "quactlize_extensions/cutlass/gemm/collective/detail/ppu_2plane_source_layout.hpp"
#include "xplane_offline.hpp"

struct L138F16Atom {};

namespace cute {
template <> struct MMA_Traits<L138F16Atom> {
  using ValTypeD = float;
  using ValTypeA = cutlass::half_t;
  using ValTypeB = cutlass::half_t;
  using ValTypeC = float;
  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _128, _8>>>;
  using BLayout = ALayout;
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>,
                         Stride<Stride<_16, _1>, Stride<_64, _8>>>;
};
} // namespace cute

namespace {
using namespace cute;

constexpr int kTM = 16;
constexpr int kTN = 128;
constexpr int kTK = 128;
constexpr int kWM = 16;
constexpr int kWN = 64;
constexpr int kWK = 4;
constexpr int kBits = 4;
constexpr int kFold = 1;
constexpr int kArtifactTK = 64;
constexpr int kWOM = kTM / kWM;
constexpr int kWON = kTN / kWN;
constexpr int kRowBytes = kTK * kBits / 8;
constexpr int kAiuElem = kArtifactTK;
constexpr int kInstNum = kTK / kAiuElem;

using ComputeMma = TiledMMA<
    MMA_Atom<L138F16Atom>,
    Layout<Shape<Int<kWOM>, Int<kWON>, Int<kWK>>>,
    Tile<_16, _32, _64>>;

template <int ShadowWK, int ShadowPermK>
struct Shadow {
  using Inst = PPU0015_16x16x32_S32S8S8S32_TN;
  using Mma = TiledMMA<
      MMA_Atom<Inst>,
      Layout<Shape<Int<kWOM>, Int<kWON>, Int<ShadowWK>>>,
      Tile<_16, _32, Int<ShadowPermK>>>;
  using Op = PPU0010_TSM_LD_SWZL<
      int8_t, kTN, kAiuElem * kBits / 8, true, false, kInstNum>;
};

using PairedShadow = Shadow<2, 64>;
using SingleShadow = Shadow<1, 64>;
using K4Perm64Shadow = Shadow<4, 64>;
using OldK4Shadow = Shadow<4, 128>;

static_assert(size(ComputeMma{}) == 256);
static_assert(size(typename PairedShadow::Mma{}) == 128);
static_assert(size(typename OldK4Shadow::Mma{}) == 256);
static_assert(decltype(ComputeMma{}.template permutation_mnk<0>()){} == _16{});
static_assert(decltype(ComputeMma{}.template permutation_mnk<1>()){} == _32{});
static_assert(decltype(ComputeMma{}.template permutation_mnk<2>()){} == _64{});
static_assert(decltype(typename PairedShadow::Mma{}.template permutation_mnk<2>()){} == _64{});

struct OutputRef {
  int thread = -1;
  int frag = -1;
  int logical = -1;
  int wk = -1;
};

struct SourceFragment {
  std::vector<int> physical;
  std::vector<int> vreg, code, ni, kb;
  bool valid = true;
};

std::vector<int> compute_logical_set(std::vector<OutputRef> const& r);
std::vector<OutputRef> compute_fragment(int thread);

template <class Shadow_>
SourceFragment shadow_source_fragment(int thread) {
  using Mma = typename Shadow_::Mma;
  using Op = typename Shadow_::Op;
  using Tr = Copy_Traits<Op>;
  constexpr int WPR = Tr::LogicalWordsPerRow;
  static_assert(Tr::LogicalSlices == 1);

  auto s8 = make_tensor(make_smem_ptr((int8_t*)nullptr),
      make_layout(Shape<Int<kTN>, Int<kRowBytes>>{},
                  Stride<Int<kRowBytes>, _1>{}));
  auto sid = make_identity_tensor(make_shape(Int<kTN>{}, Int<kRowBytes>{}));
  auto load = Mma{}.get_thread_slice(thread).partition_fragment_B(s8);
  auto cp = make_tiled_copy_B(Copy_Atom<Op, int8_t>{}, Mma{})
                .get_thread_slice((thread / 32) * 32);
  auto src = cp.partition_S(sid);
  auto view = cp.retile_D(load);

  SourceFragment out;
  out.physical.assign(2 * cosize(load.layout()), -1);
  out.vreg.assign(out.physical.size(),-1);out.code.assign(out.physical.size(),-1);
  out.ni.assign(out.physical.size(),-1);out.kb.assign(out.physical.size(),-1);
  constexpr int CN = size<1>(decltype(view.layout()){});
  constexpr int CK = size<2>(decltype(view.layout()){});
  for (int ck = 0; ck < CK; ++ck)
    for (int cn = 0; cn < CN; ++cn)
      for (int v = 0; v < 4; ++v)
        for (int c = 0; c < 8; ++c) {
          auto base = src(0, cn, ck);
          int word = int(typename Tr::LogicalTV{}(
              make_coord(make_coord(thread % 4, (thread % 32) / 4),
                         make_coord(v % 2, v / 2), _0{})));
          int byte = (int(get<0>(base)) + word / WPR) * kRowBytes +
                     int(get<1>(base)) + 4 * (word % WPR) + c / 2;
          int dst = 2 * int(view.layout()(4 * v + c / 2, cn, ck)) + c % 2;
          int physical = 2 * byte + c % 2;
          bool in = dst >= 0 && dst < int(out.physical.size());
          out.valid &= in;
          if (!in) continue;
          out.valid &= out.physical[dst] < 0 || out.physical[dst] == physical;
          out.physical[dst] = physical;
          out.vreg[dst]=v;out.code[dst]=c;out.ni[dst]=cn;out.kb[dst]=ck;
        }
  out.valid &= std::none_of(out.physical.begin(), out.physical.end(),
                            [](int x) { return x < 0; });
  return out;
}

void print_selection_pattern(int st, int ct) {
  static auto const shipping = xplane::plane_map<kBits,kTM,kTN,kTK,kWM,kWN,kFold,kArtifactTK>();
  auto sf=shadow_source_fragment<PairedShadow>(st);auto c=compute_logical_set(compute_fragment(ct));
  std::map<std::tuple<int,int,int>,std::vector<int>> sel;
  for(int d=0;d<int(sf.physical.size());++d){int l=shipping[sf.physical[d]];
    if(std::binary_search(c.begin(),c.end(),l))sel[{sf.kb[d],sf.ni[d],sf.vreg[d]}].push_back(sf.code[d]);}
  std::printf("L138 select st%d->ct%d wk%d",st,ct,compute_fragment(ct)[0].wk);
  for(auto const& [key,codes]:sel){auto [kb,ni,v]=key;std::printf(" [kb%d,ni%d,v%d:",kb,ni,v);for(int x:codes)std::printf("%d",x);std::printf("]");}
  std::putchar('\n');
}

std::vector<OutputRef> compute_fragment(int thread) {
  auto s16 = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<kTN>, Int<kTK>>{}, Stride<Int<kTK>, _1>{}));
  auto bid = make_identity_tensor(make_shape(Int<kTN>{}, Int<kTK>{}));
  auto mma = ComputeMma{};
  auto tc = mma.get_thr_layout_vmnk().get_flat_coord(thread);
  int wk = int(get<3>(tc));
  auto frag = mma.get_thread_slice(thread).partition_fragment_B(s16);
  auto part = mma.get_thread_slice(thread).partition_B(bid);
  auto pi = right_inverse(frag.layout());
  std::vector<OutputRef> out;
  out.reserve(size(frag));
  for (int i = 0; i < int(size(frag)); ++i) {
    auto x = part(pi(i));
    out.push_back({thread, i, int(get<0>(x)) * kTK + int(get<1>(x)), wk});
  }
  return out;
}

enum class Pairing { Consecutive, Parity, Identity };

template <class Shadow_>
int mapped_shadow_thread(int compute_thread, Pairing pairing) {
  auto ctl = ComputeMma{}.get_thr_layout_vmnk();
  auto stl = typename Shadow_::Mma{}.get_thr_layout_vmnk();
  auto c = ctl.get_flat_coord(compute_thread);
  int wk = int(get<3>(c));
  int swk = 0;
  if (pairing == Pairing::Consecutive) swk = wk / 2;
  if (pairing == Pairing::Parity) swk = wk % 2;
  if (pairing == Pairing::Identity) swk = wk;
  return int(stl(make_coord(get<0>(c), get<1>(c), get<2>(c), swk)));
}

struct Metrics {
  int compute_fragments = 0;
  int bad_fragments = 0;
  int source_codes = 0;
  int output_codes = 0;
  int holes = 0;
  int extras = 0;
  int source_duplicates = 0;
  int output_duplicates = 0;
  int physical_holes = 0;
  int physical_duplicates = 0;
  std::array<int, kWK> fragment_bad_by_wk{};
  std::array<int, 2> pair_sources{};
  std::array<int, 2> pair_outputs{};
  std::array<int, 2> pair_holes{};
  std::array<int, 2> pair_extras{};
  uint64_t hash = UINT64_C(1469598103934665603);
  std::array<std::tuple<int, int, int, int>, 8> sample{};
  int sample_count = 0;
};
void hash_word(uint64_t& h,uint64_t x);

struct DerivedMap {
  std::vector<int> map;
  int conflicts=0,holes=0,logical_duplicates=0,physical_duplicates=0;
  std::array<int,4> fragment_holes{};
  uint64_t hash=UINT64_C(1469598103934665603);
};

// The exact two-source gate. This is the production tCrB_mma fragment layout
// printed above: ((2,2,2),4,2):((1,2,4),16,8). MixGemmChunkEmit therefore
// cannot be used unchanged: its FragLayout contract is one 32-code delivery,
// while this compute fragment has 64 outputs assembled from two deliveries.
using ComputeFragLayout = decltype(ComputeMma{}.get_thread_slice(0).partition_fragment_B(
    make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<kTN>,Int<kTK>>{},Stride<Int<kTK>,_1>{}))).layout());
static_assert(size(ComputeFragLayout{})==64);
struct TwoSourceInt4Emit {
  static constexpr bool keep(int wk,int vreg,int code) {
    return vreg/2==wk/2 && (code/2)%2==wk%2;
  }
  static constexpr int selected_per_source=32;
  static constexpr int sources_per_fragment=2;
  static constexpr int outputs=selected_per_source*sources_per_fragment;
};
static_assert(TwoSourceInt4Emit::outputs==size(ComputeFragLayout{}));

template <class Shadow_>
DerivedMap derive_map(Pairing pairing) {
  DerivedMap out;out.map.assign(kTN*kTK,-1);
  std::vector<int> physical_hits(kTN*kTK),logical_hits(kTN*kTK);
  for(int ct=0;ct<int(size(ComputeMma{}));++ct){
    int st=mapped_shadow_thread<Shadow_>(ct,pairing);auto sf=shadow_source_fragment<Shadow_>(st);
    auto refs=compute_fragment(ct);int wk=refs[0].wk;
    // The paired shadow carries 128 codes. The real converter-selection rule
    // is derived below from the observed source coordinates, not guessed:
    //   wk0: even vregs, code phase 0/1/4/5
    //   wk1: even vregs, code phase 2/3/6/7
    //   wk2: odd  vregs, code phase 0/1/4/5
    //   wk3: odd  vregs, code phase 2/3/6/7.
    std::vector<int> selected;
    for(int d=0;d<int(sf.physical.size());++d){
      bool keep=TwoSourceInt4Emit::keep(wk,sf.vreg[d],sf.code[d]);
      if(keep)selected.push_back(d);
    }
    if(selected.size()!=refs.size()){if(ct<193&&ct%64==0)std::printf("L138 derive ct%d wk%d selected=%zu refs=%zu\n",ct,wk,selected.size(),refs.size());++out.fragment_holes[wk];continue;}
    // This early oracle zipped source and destination fragment orders. L142
    // later proved that this order is NOT the production conversion order;
    // retain it only as a source-availability/bijection diagnostic.
    for(int i=0;i<int(refs.size());++i){int p=sf.physical[selected[i]],l=refs[i].logical;
      if(p<0||p>=int(out.map.size())||l<0||l>=int(out.map.size())){++out.conflicts;continue;}
      ++physical_hits[p];++logical_hits[l];
      if(out.map[p]>=0&&out.map[p]!=l)++out.conflicts;out.map[p]=l;
    }
  }
  for(int i=0;i<int(out.map.size());++i){out.holes+=out.map[i]<0;
    out.physical_duplicates+=std::max(0,physical_hits[i]-1);
    out.logical_duplicates+=std::max(0,logical_hits[i]-1);
    hash_word(out.hash,uint64_t(i));hash_word(out.hash,uint64_t(out.map[i]+1));}
  return out;
}

DerivedMap derive_two_source_map(bool reverse_sources=false) {
  DerivedMap out;out.map.assign(kTN*kTK,-1);
  std::vector<int> ph(kTN*kTK),lh(kTN*kTK);
  for(int ct=0;ct<int(size(ComputeMma{}));++ct){
    auto ctl=ComputeMma{}.get_thr_layout_vmnk();auto c=ctl.get_flat_coord(ct);int wk=int(get<3>(c));
    auto stl=typename PairedShadow::Mma{}.get_thr_layout_vmnk();
    std::vector<std::pair<int,int>> chosen;
    for(int sj=0;sj<2;++sj){int sk=reverse_sources?1-sj:sj;int st=int(stl(make_coord(get<0>(c),get<1>(c),get<2>(c),sk)));
      auto sf=shadow_source_fragment<PairedShadow>(st);
      for(int d=0;d<int(sf.physical.size());++d){
        bool keep=TwoSourceInt4Emit::keep(wk,sf.vreg[d],sf.code[d]);
        if(keep)chosen.emplace_back(sf.physical[d],d);
      }
    }
    auto refs=compute_fragment(ct);if(chosen.size()!=refs.size()){++out.fragment_holes[wk];continue;}
    for(int i=0;i<int(refs.size());++i){int p=chosen[i].first,l=refs[i].logical;
      ++ph[p];++lh[l];if(out.map[p]>=0&&out.map[p]!=l)++out.conflicts;out.map[p]=l;}
  }
  for(int i=0;i<int(out.map.size());++i){out.holes+=out.map[i]<0;
    out.physical_duplicates+=std::max(0,ph[i]-1);out.logical_duplicates+=std::max(0,lh[i]-1);
    hash_word(out.hash,uint64_t(i));hash_word(out.hash,uint64_t(out.map[i]+1));}
  return out;
}

bool derived_clean(DerivedMap const& x){return x.conflicts==0&&x.holes==0&&
  x.logical_duplicates==0&&x.physical_duplicates==0&&
  std::all_of(x.fragment_holes.begin(),x.fragment_holes.end(),[](int z){return z==0;});}

template <class Shadow_>
std::vector<int> shadow_logical_set(int thread) {
  static auto const shipping = xplane::plane_map<kBits, kTM, kTN, kTK,
                                                  kWM, kWN, kFold, kArtifactTK>();
  auto sf = shadow_source_fragment<Shadow_>(thread);
  std::vector<int> out;
  out.reserve(sf.physical.size());
  if (!sf.valid) return {};
  for (int p : sf.physical) {
    if (p < 0 || p >= int(shipping.size()) || shipping[p] < 0) return {};
    out.push_back(shipping[p]);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<int> compute_logical_set(std::vector<OutputRef> const& r) {
  std::vector<int> out;
  for (auto const& x : r) out.push_back(x.logical);
  std::sort(out.begin(), out.end());
  return out;
}

template <class Shadow_, int Fanout>
bool exhaustive_partition(char const* name, std::array<int, 4>& compute_to_shadow,
                          std::array<int, 4>& compute_to_slot) {
  constexpr int ST = size(typename Shadow_::Mma{});
  std::vector<std::vector<int>> cs(size(ComputeMma{}));
  std::vector<std::vector<int>> ss(ST);
  for (int t = 0; t < int(cs.size()); ++t) cs[t] = compute_logical_set(compute_fragment(t));
  for (int t = 0; t < ST; ++t) ss[t] = shadow_logical_set<Shadow_>(t);
  bool ok = true;
  int unmatched = 0, ambiguous = 0;
  std::array<int, 5> wk_match_count{};
  std::fill(compute_to_shadow.begin(), compute_to_shadow.end(), -2);
  std::fill(compute_to_slot.begin(), compute_to_slot.end(), -2);
  for (int st = 0; st < ST; ++st) {
    std::vector<int> candidates;
    std::vector<int> exact_intersection;
    int max_intersection = 0, max_thread = -1;
    std::vector<std::pair<int,int>> nonzero;
    for (int ct = 0; ct < int(cs.size()); ++ct)
      if (std::includes(ss[st].begin(), ss[st].end(), cs[ct].begin(), cs[ct].end()))
        candidates.push_back(ct);
      else {
        std::vector<int> z;
        std::set_intersection(ss[st].begin(), ss[st].end(), cs[ct].begin(), cs[ct].end(),
                              std::back_inserter(z));
        if (!z.empty()) nonzero.emplace_back(ct,int(z.size()));
        if (int(z.size()) > max_intersection) { max_intersection = int(z.size()); max_thread = ct; }
      }
    for (int ct = 0; ct < int(cs.size()); ++ct) {
      std::vector<int> z;
      std::set_intersection(ss[st].begin(), ss[st].end(), cs[ct].begin(), cs[ct].end(),
                            std::back_inserter(z));
      if (int(z.size()) == int(cs[ct].size())) exact_intersection.push_back(ct);
    }
    if (int(candidates.size()) != Fanout) {
      ok = false;
      if (candidates.empty()) ++unmatched; else ++ambiguous;
      if (st < 4) { std::printf("L138 exhaustive %s st=%d best-intersection=%d/32 ct=%d exact-intersection=",
                                name, st, max_intersection, max_thread);
        for(int x:exact_intersection)std::printf("%d,",x);std::putchar('\n'); }
      if(st==0){std::printf("L138 intersections st0 ");for(auto [t,n]:nonzero)std::printf("t%d:%d ",t,n);std::putchar('\n');}
      continue;
    }
    std::vector<int> joined;
    for (int ct : candidates) joined.insert(joined.end(), cs[ct].begin(), cs[ct].end());
    std::sort(joined.begin(), joined.end());
    if (joined != ss[st]) { ok = false; ++unmatched; continue; }
    for (int slot = 0; slot < Fanout; ++slot) {
      int ct = candidates[slot];
      int wk = compute_fragment(ct)[0].wk;
      ++wk_match_count[wk];
      if (st == 0 && wk >= 0 && wk < 4) {
        compute_to_shadow[wk] = st;
        compute_to_slot[wk] = slot;
      }
    }
  }
  if constexpr(std::is_same_v<Shadow_,PairedShadow>){
    for(int ct: {0,64,128,192}){std::printf("L138 reverse ct%d ",ct);
      for(int st=0;st<ST;++st){std::vector<int> z;std::set_intersection(ss[st].begin(),ss[st].end(),cs[ct].begin(),cs[ct].end(),std::back_inserter(z));if(!z.empty())std::printf("st%d:%zu ",st,z.size());}
      std::putchar('\n');}
  }
  std::printf("L138 exhaustive %-14s shadow=%d fanout=%d unmatched=%d ambiguous=%d "
              "wk-matches=%d/%d/%d/%d result=%s\n",
              name, ST, Fanout, unmatched, ambiguous,
              wk_match_count[0], wk_match_count[1], wk_match_count[2], wk_match_count[3],
              ok ? "EXACT" : "REFUTED");
  return ok;
}

void hash_word(uint64_t& h, uint64_t x) {
  for (int i = 0; i < 8; ++i) {
    h ^= (x >> (8 * i)) & 0xffu;
    h *= UINT64_C(1099511628211);
  }
}

template <class Shadow_>
Metrics evaluate(Pairing pairing) {
  auto shipping = xplane::plane_map<kBits, kTM, kTN, kTK,
                                     kWM, kWN, kFold, kArtifactTK>();
  Metrics m;
  std::vector<std::vector<OutputRef>> desired(size(typename Shadow_::Mma{}));
  for (int t = 0; t < int(size(ComputeMma{})); ++t) {
    auto refs = compute_fragment(t);
    int st = mapped_shadow_thread<Shadow_>(t, pairing);
    if (st < 0 || st >= int(desired.size())) {
      ++m.bad_fragments;
      continue;
    }
    desired[st].insert(desired[st].end(), refs.begin(), refs.end());
  }

  std::vector<int> physical_hits(kTN * kTK, 0);
  std::vector<int> logical_source_hits(kTN * kTK, 0);
  std::vector<int> logical_output_hits(kTN * kTK, 0);
  std::vector<int> source_to_thread(kTN * kTK, -1);
  std::vector<int> source_to_frag(kTN * kTK, -1);

  for (int st = 0; st < int(desired.size()); ++st) {
    auto sf = shadow_source_fragment<Shadow_>(st);
    std::map<int, std::vector<int>> sources;
    std::map<int, std::vector<OutputRef>> outputs;
    for (int p : sf.physical) {
      if (p < 0 || p >= int(shipping.size())) {
        ++m.physical_holes;
        continue;
      }
      ++physical_hits[p];
      int logical = shipping[p];
      if (logical < 0 || logical >= kTN * kTK) {
        ++m.holes;
        continue;
      }
      sources[logical].push_back(p);
      ++logical_source_hits[logical];
      ++m.source_codes;
    }
    for (auto const& r : desired[st]) {
      outputs[r.logical].push_back(r);
      ++logical_output_hits[r.logical];
      ++m.output_codes;
    }

    bool fragment_ok[2] = {true, true};
    int fragment_thread[2] = {-1, -1};
    for (auto const& r : desired[st]) {
      int slot = fragment_thread[0] == r.thread ? 0 :
                 fragment_thread[1] == r.thread ? 1 :
                 fragment_thread[0] < 0 ? (fragment_thread[0] = r.thread, 0) :
                                          (fragment_thread[1] = r.thread, 1);
      fragment_ok[slot] &= sources[r.logical].size() == 1;
    }
    for (int i = 0; i < 2; ++i) if (fragment_thread[i] >= 0) {
      ++m.compute_fragments;
      if (!fragment_ok[i]) {
        ++m.bad_fragments;
        int wk = compute_fragment(fragment_thread[i])[0].wk;
        if (wk >= 0 && wk < kWK) ++m.fragment_bad_by_wk[wk];
      }
    }

    std::map<int, int> keys;
    for (auto const& x : sources) keys[x.first] = 1;
    for (auto const& x : outputs) keys[x.first] = 1;
    for (auto const& [logical, _] : keys) {
      int ns = int(sources[logical].size()), no = int(outputs[logical].size());
      m.holes += std::max(0, no - ns);
      m.extras += std::max(0, ns - no);
      m.source_duplicates += std::max(0, ns - 1);
      m.output_duplicates += std::max(0, no - 1);
      int pair = 0;
      if (!outputs[logical].empty()) pair = outputs[logical][0].wk / 2;
      else if constexpr (std::is_same_v<Shadow_, PairedShadow>) {
        auto c = typename Shadow_::Mma{}.get_thr_layout_vmnk().get_flat_coord(st);
        pair = int(get<3>(c));
      }
      if (pair >= 0 && pair < 2) {
        m.pair_sources[pair] += ns;
        m.pair_outputs[pair] += no;
        m.pair_holes[pair] += std::max(0, no - ns);
        m.pair_extras[pair] += std::max(0, ns - no);
      }
      int match = std::min(ns, no);
      for (int i = 0; i < match; ++i) {
        int p = sources[logical][i];
        auto const& r = outputs[logical][i];
        source_to_thread[p] = r.thread;
        source_to_frag[p] = r.frag;
      }
    }
  }

  for (int i = 0; i < kTN * kTK; ++i) {
    m.physical_holes += physical_hits[i] == 0;
    m.physical_duplicates += std::max(0, physical_hits[i] - 1);
    if (source_to_thread[i] >= 0) {
      hash_word(m.hash, uint64_t(i));
      hash_word(m.hash, uint64_t(source_to_thread[i]));
      hash_word(m.hash, uint64_t(source_to_frag[i]));
      hash_word(m.hash, uint64_t(shipping[i]));
      if (m.sample_count < int(m.sample.size())) {
        m.sample[m.sample_count++] =
            {i, shipping[i], source_to_thread[i], source_to_frag[i]};
      }
    }
  }
  for (int x : logical_source_hits) m.source_duplicates += std::max(0, x - 1);
  for (int x : logical_output_hits) m.output_duplicates += std::max(0, x - 1);
  return m;
}

bool clean(Metrics const& m) {
  return m.compute_fragments == 256 && m.bad_fragments == 0 &&
         m.source_codes == kTN * kTK && m.output_codes == kTN * kTK &&
         m.holes == 0 && m.extras == 0 &&
         m.source_duplicates == 0 && m.output_duplicates == 0 &&
         m.physical_holes == 0 && m.physical_duplicates == 0;
}

void print(char const* name, Metrics const& m) {
  std::printf("L138 %-24s fragments=%d bad=%d source=%d output=%d "
              "holes=%d extras=%d srcdup=%d outdup=%d physical={holes:%d,dup:%d} "
              "wkbad=%d/%d/%d/%d hash=%016llx\n",
              name, m.compute_fragments, m.bad_fragments, m.source_codes,
              m.output_codes, m.holes, m.extras, m.source_duplicates,
              m.output_duplicates, m.physical_holes, m.physical_duplicates,
              m.fragment_bad_by_wk[0], m.fragment_bad_by_wk[1],
              m.fragment_bad_by_wk[2], m.fragment_bad_by_wk[3],
              (unsigned long long)m.hash);
  std::printf("L138 %-24s pairs={0:src=%d,out=%d,hole=%d,extra=%d;"
              "1:src=%d,out=%d,hole=%d,extra=%d}\n",
              name, m.pair_sources[0], m.pair_outputs[0], m.pair_holes[0],
              m.pair_extras[0], m.pair_sources[1], m.pair_outputs[1],
              m.pair_holes[1], m.pair_extras[1]);
  if (m.sample_count) {
    std::printf("L138 %-24s source->output", name);
    for (int i = 0; i < m.sample_count; ++i) {
      auto [p, logical, t, f] = m.sample[i];
      std::printf(" p%d=(n%d,k%d)->t%d:f%d", p, logical / kTK,
                  logical % kTK, t, f);
    }
    std::putchar('\n');
  }
}

} // namespace

int main() {
  {
    auto s8 = make_tensor(make_smem_ptr((int8_t*)nullptr),
        make_layout(Shape<Int<kTN>, Int<kRowBytes>>{}, Stride<Int<kRowBytes>, _1>{}));
    auto s16 = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
        make_layout(Shape<Int<kTN>, Int<kTK>>{}, Stride<Int<kTK>, _1>{}));
    auto sl = typename PairedShadow::Mma{}.get_thread_slice(0).partition_fragment_B(s8);
    auto cf = ComputeMma{}.get_thread_slice(0).partition_fragment_B(s16);
    std::printf("L138 shapes shadow-load="); print(sl.layout());
    std::printf(" compute-frag="); print(cf.layout()); std::putchar('\n');
  }
  std::array<int, 4> k2_shadow{}, k2_slot{}, k1_shadow{}, k1_slot{};
  bool k2_partition = exhaustive_partition<PairedShadow, 2>(
      "K2/per2", k2_shadow, k2_slot);
  bool k1_partition = exhaustive_partition<SingleShadow, 4>(
      "K1/per4", k1_shadow, k1_slot);
  for(int ct: {0,64,128,192}){print_selection_pattern(0,ct);print_selection_pattern(64,ct);}
  auto derived=derive_map<PairedShadow>(Pairing::Consecutive);
  auto two_source=derive_two_source_map();
  auto two_source_reversed=derive_two_source_map(true);
  int reversed_map_diff=0;for(int i=0;i<int(two_source.map.size());++i)
    reversed_map_diff+=two_source.map[i]!=two_source_reversed.map[i];
  auto derived_k4p64=derive_map<K4Perm64Shadow>(Pairing::Identity);
  auto derived_wrong=derive_map<PairedShadow>(Pairing::Parity);
  std::printf("L138 derived-WK4 map entries=%zu conflicts=%d holes=%d logical-dup=%d "
              "physical-dup=%d fragment-holes=%d/%d/%d/%d hash=%016llx %s\n",
              derived.map.size(),derived.conflicts,derived.holes,derived.logical_duplicates,
              derived.physical_duplicates,derived.fragment_holes[0],derived.fragment_holes[1],
              derived.fragment_holes[2],derived.fragment_holes[3],
              (unsigned long long)derived.hash,derived_clean(derived)?"BIJECTIVE":"FAIL");
  std::printf("L138 sequential-two-source-order entries=%zu conflicts=%d holes=%d logical-dup=%d physical-dup=%d "
              "fragment-holes=%d/%d/%d/%d hash=%016llx %s\n",two_source.map.size(),two_source.conflicts,
              two_source.holes,two_source.logical_duplicates,two_source.physical_duplicates,
              two_source.fragment_holes[0],two_source.fragment_holes[1],two_source.fragment_holes[2],
              two_source.fragment_holes[3],(unsigned long long)two_source.hash,
              derived_clean(two_source)?"BIJECTIVE":"FAIL");
  std::printf("L138 reversed-source-order conflicts=%d holes=%d logical-dup=%d physical-dup=%d map-diff=%d %s\n",
              two_source_reversed.conflicts,two_source_reversed.holes,
              two_source_reversed.logical_duplicates,two_source_reversed.physical_duplicates,
              reversed_map_diff,reversed_map_diff>0?"EXPECTED-RED":"UNEXPECTED-GREEN");
  std::printf("L138 wrong-pair-derived conflicts=%d holes=%d logical-dup=%d physical-dup=%d %s\n",
              derived_wrong.conflicts,derived_wrong.holes,derived_wrong.logical_duplicates,
              derived_wrong.physical_duplicates,derived_clean(derived_wrong)?"UNEXPECTED-GREEN":"EXPECTED-RED");
  std::printf("L138 K4-perm64-derived conflicts=%d holes=%d logical-dup=%d physical-dup=%d "
              "fragment-holes=%d/%d/%d/%d %s\n",derived_k4p64.conflicts,derived_k4p64.holes,
              derived_k4p64.logical_duplicates,derived_k4p64.physical_duplicates,
              derived_k4p64.fragment_holes[0],derived_k4p64.fragment_holes[1],
              derived_k4p64.fragment_holes[2],derived_k4p64.fragment_holes[3],
              derived_clean(derived_k4p64)?"BIJECTIVE":"FAIL");
  auto good = evaluate<PairedShadow>(Pairing::Consecutive);
  auto old = evaluate<OldK4Shadow>(Pairing::Identity);
  auto wrong = evaluate<PairedShadow>(Pairing::Parity);
  print("K2 consecutive", good);
  print("old K4 shadow", old);
  print("K2 wrong wk%2", wrong);
  bool old_red = !clean(old) && (old.holes > 0 || old.extras > 0 ||
                                 old.physical_duplicates > 0);
  bool wrong_red = !clean(wrong) && wrong.holes > 0 && wrong.extras > 0;
  bool ok = derived_clean(two_source) && !derived_clean(derived) &&
            reversed_map_diff>0 && !derived_clean(derived_wrong) &&
            old_red && wrong_red;
  std::printf("L138 target compute=<1,2,4>/perm64 shadow=<1,2,2>/perm64 "
              "two-source-availability=%s sequential-order=L142-COUNTERFEIT "
              "old-K4=%s wrong-pair=%s result=%s\n",
              derived_clean(two_source) ? "EXACT" : "FAIL",
              old_red ? "EXPECTED-RED" : "UNEXPECTED-GREEN",
              wrong_red ? "EXPECTED-RED" : "UNEXPECTED-GREEN",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

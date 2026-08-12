// L142 -- derive the real two-source consumer on production SmemLayoutB.
//
// L138 deliberately used a compact row-major half tensor to expose the MMA
// fragment.  The collective does not: it partitions the tiled/swizzled B
// stage.  Same shape, different strides, and converter placement is a stride
// contract.  Reuse L138's source-side Copy_Traits machinery, derive the
// destination from the exact production stage, then model the actual
// `(t,t+4)` pair reads and half2 writes.  Sequentially zipping the fragment
// orders is retained only as the ea96/17df negative.
#define main l138_compact_reference_main
#include "l138_wk_shadow_delivery.cu"
#undef main

namespace {
using namespace cute;

using ProductionSmemLayoutAtomB =
    Layout<Shape<_8, _64>, Stride<_64, _1>>;
using ProductionSmemLayoutB = decltype(tile_to_shape(
    ProductionSmemLayoutAtomB{},
    make_shape(Int<kTN>{}, Int<kTK>{}, _4{})));
using ProductionFragLayout = decltype(ComputeMma{}.get_thread_slice(0)
    .partition_fragment_B(make_tensor(
        make_smem_ptr((cutlass::half_t*)nullptr), ProductionSmemLayoutB{})(_,_,0)).layout());
using CompactFragLayout = ComputeFragLayout;

static_assert(std::is_same_v<ProductionFragLayout,
    Layout<Shape<Shape<_2,_2,_2>,_4,_2>,
           Stride<Stride<_1,_2,_4>,_8,_32>>>,
    "L142 must model the exact production WK4 B fragment");
static_assert(!std::is_same_v<ProductionFragLayout, CompactFragLayout>,
    "compact s16 is a counterfeit production anchor");

std::vector<OutputRef> production_compute_fragment(int thread) {
  auto s16 = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                         ProductionSmemLayoutB{});
  auto bid = make_identity_tensor(make_shape(Int<kTN>{}, Int<kTK>{}));
  auto mma = ComputeMma{};
  auto tc = mma.get_thr_layout_vmnk().get_flat_coord(thread);
  int wk = int(get<3>(tc));
  auto frag = mma.get_thread_slice(thread).partition_fragment_B(s16(_,_,0));
  auto part = mma.get_thread_slice(thread).partition_B(bid);
  auto pi = right_inverse(frag.layout());
  std::vector<OutputRef> out;
  for (int i = 0; i < int(size(frag)); ++i) {
    auto x = part(pi(i));
    out.push_back({thread, i, int(get<0>(x)) * kTK + int(get<1>(x)), wk});
  }
  return out;
}

DerivedMap production_map(bool compact_negative) {
  DerivedMap out;
  out.map.assign(kTN * kTK, -1);
  std::vector<int> ph(kTN*kTK), lh(kTN*kTK);
  auto ctl = ComputeMma{}.get_thr_layout_vmnk();
  auto stl = PairedShadow::Mma{}.get_thr_layout_vmnk();
  for (int ct = 0; ct < int(size(ComputeMma{})); ++ct) {
    auto c = ctl.get_flat_coord(ct);
    int wk = int(get<3>(c));
    std::vector<int> selected;
    for (int sk = 0; sk < 2; ++sk) {
      int st = int(stl(make_coord(get<0>(c),get<1>(c),get<2>(c),sk)));
      auto sf = shadow_source_fragment<PairedShadow>(st);
      for (int d=0; d<int(sf.physical.size()); ++d)
        if (TwoSourceInt4Emit::keep(wk,sf.vreg[d],sf.code[d]))
          selected.push_back(sf.physical[d]);
    }
    auto refs = compact_negative ? compute_fragment(ct)
                                 : production_compute_fragment(ct);
    if (selected.size() != refs.size()) { ++out.fragment_holes[wk]; continue; }
    for (int i=0; i<int(refs.size()); ++i) {
      int p=selected[i], l=refs[i].logical;
      ++ph[p]; ++lh[l];
      if (out.map[p]>=0 && out.map[p]!=l) ++out.conflicts;
      out.map[p]=l;
    }
  }
  for (int i=0;i<int(out.map.size());++i) {
    out.holes += out.map[i]<0;
    out.physical_duplicates += std::max(0,ph[i]-1);
    out.logical_duplicates += std::max(0,lh[i]-1);
    hash_word(out.hash,uint64_t(i)); hash_word(out.hash,uint64_t(out.map[i]+1));
  }
  return out;
}

enum class ConsumerModel { WholeConverter32, DirectShippingPairs };

int source_slot(SourceFragment const& sf, int ni, int vreg, int code) {
  int found = -1;
  for (int d = 0; d < int(sf.physical.size()); ++d) {
    if (sf.ni[d] != ni || sf.vreg[d] != vreg || sf.code[d] != code) continue;
    if (found >= 0) return -2;
    found = d;
  }
  return found;
}

int output_slot(std::vector<OutputRef> const& refs, int logical) {
  int found = -1;
  for (int i = 0; i < int(refs.size()); ++i) {
    if (refs[i].logical != logical) continue;
    if (found >= 0) return -2;
    found = i;
  }
  return found;
}

bool print_anchored_pair_destinations() {
  static auto const shipping = xplane::plane_map<kBits,kTM,kTN,kTK,
                                                   kWM,kWN,kFold,kArtifactTK>();
  auto ctl=ComputeMma{}.get_thr_layout_vmnk();
  auto stl=PairedShadow::Mma{}.get_thr_layout_vmnk();
  bool ok = true;
  for (int ct : {0,64,128,192}) {
    auto c=ctl.get_flat_coord(ct); int const wk=int(get<3>(c));
    auto refs=production_compute_fragment(ct);
    std::printf("L142 anchored wk%d", wk);
    for (int sk=0; sk<2; ++sk) {
      int const st=int(stl(make_coord(get<0>(c),get<1>(c),get<2>(c),sk)));
      auto sf=shadow_source_fragment<PairedShadow>(st);
      int const v0=wk/2, v1=v0+2, phase=wk%2;
      for (int ni=0; ni<4; ++ni)
        for (int v : {v0,v1})
          for (int t : {2*phase,2*phase+1}) {
            // One f16x2 pair takes the same nibble position from the low and
            // high 16-bit halves of the vreg: (t,t+4), not (2t,2t+1).
            int const d0=source_slot(sf,ni,v,t);
            int const d1=source_slot(sf,ni,v,t+4);
            ok &= d0>=0 && d1>=0;
            if (d0<0 || d1<0) continue;
            int const o0=output_slot(refs,shipping[sf.physical[d0]]);
            int const o1=output_slot(refs,shipping[sf.physical[d1]]);
            ok &= o0>=0 && o1==o0+1 && o0%2==0;
            std::printf(" s%dn%dv%dt%d->h%d",sk,ni,v,t,o0/2);
          }
    }
    std::putchar('\n');
  }
  return ok;
}

bool prove_shadow_lane_preservation() {
  auto const ctl = ComputeMma{}.get_thr_layout_vmnk();
  auto const stl = PairedShadow::Mma{}.get_thr_layout_vmnk();
  int lane_bad = 0;
  int coord_bad = 0;
  for (int ct = 0; ct < int(size(ComputeMma{})); ++ct) {
    auto const c = ctl.get_flat_coord(ct);
    for (int sk = 0; sk < 2; ++sk) {
      int const st = int(stl(make_coord(get<0>(c), get<1>(c), get<2>(c), sk)));
      lane_bad += (st % 32) != (ct % 32);
      auto const s = stl.get_flat_coord(st);
      coord_bad += int(get<0>(s)) != int(get<0>(c));
      coord_bad += int(get<1>(s)) != int(get<1>(c));
      coord_bad += int(get<2>(s)) != int(get<2>(c));
      coord_bad += int(get<3>(s)) != sk;
    }
  }
  bool const ok = lane_bad == 0 && coord_bad == 0;
  std::printf("L142 shadow-lane-preservation cases=%d lane-bad=%d coord-bad=%d %s\n",
              int(size(ComputeMma{})) * 2, lane_bad, coord_bad,
              ok ? "EXACT" : "FAIL");
  return ok;
}

std::vector<int> consumer_map(ConsumerModel model, bool& model_valid) {
  // Model the instructions the consumer actually issues, starting at its raw
  // source words and ending at its actual fp16 fragment offsets.  In
  // particular, do NOT call TwoSourceInt4Emit::keep(): that is the desired
  // algebra, and using it here made the old whole-Converter<32> seam falsely
  // prove itself.
  std::vector<int> out(kTN*kTK,-1), ph(kTN*kTK), lh(kTN*kTK);
  auto ctl=ComputeMma{}.get_thr_layout_vmnk();
  auto stl=PairedShadow::Mma{}.get_thr_layout_vmnk();
  bool valid=true;
  for(int ct=0;ct<int(size(ComputeMma{}));++ct){
    auto c=ctl.get_flat_coord(ct);int wk=int(get<3>(c));
    auto refs=production_compute_fragment(ct);
    valid &= refs.size() == 64;
    for(int sk=0;sk<2;++sk){int st=int(stl(make_coord(get<0>(c),get<1>(c),get<2>(c),sk)));
      auto sf=shadow_source_fragment<PairedShadow>(st);
      if (model == ConsumerModel::WholeConverter32) {
        // The rejected implementation reinterprets raw source codes [0,32)
        // as four int4 vregs, then writes the wide converter result contiguously.
        for (int d = 0; d < 32; ++d) {
          int const e = cutlass::MixGemmEmit<4>::index(d % 8, d / 8);
          int const p = sf.physical[d];
          int const l = refs[32 * sk + e].logical;
          valid &= p >= 0 && p < int(out.size()) && l >= 0 && l < int(out.size());
          if (p < 0 || p >= int(out.size()) || l < 0 || l >= int(out.size())) continue;
          valid &= out[p] < 0 || out[p] == l;
          out[p] = l; ++ph[p]; ++lh[l];
        }
      }
      else {
        // Instruction-level production model.  One f16x2 conversion reads
        // nibble T from both 16-bit halves of one raw vreg: (T,T+4).  WK picks
        // vregs {wk/2,wk/2+2} and phase T={2*(wk%2),+1}.  Production fragment
        // strides (...,8,32) then put the pair at
        //   h2 = 16*source + 4*NI + 2*vreg-in-pair + pair-in-phase.
        // No desired keep() predicate or artifact map participates here.
        int const v0 = wk / 2;
        int const v1 = v0 + 2;
        int const phase = wk % 2;
        for (int ni = 0; ni < 4; ++ni)
          for (int vi = 0; vi < 2; ++vi)
            for (int ti = 0; ti < 2; ++ti) {
              int const v = vi == 0 ? v0 : v1;
              int const t = 2 * phase + ti;
              int const h2 = 16 * sk + 4 * ni + 2 * vi + ti;
              int const d0 = source_slot(sf, ni, v, t);
              int const d1 = source_slot(sf, ni, v, t + 4);
              valid &= d0 >= 0 && d1 >= 0;
              if (d0 < 0 || d1 < 0) continue;
              for (int lane = 0; lane < 2; ++lane) {
                int const p = sf.physical[lane == 0 ? d0 : d1];
                int const l = refs[2 * h2 + lane].logical;
                valid &= p >= 0 && p < int(out.size()) && l >= 0 && l < int(out.size());
                if (p < 0 || p >= int(out.size()) || l < 0 || l >= int(out.size())) continue;
                valid &= out[p] < 0 || out[p] == l;
                out[p] = l; ++ph[p]; ++lh[l];
              }
            }
      }
    }
  }
  int holes = 0, physical_dup = 0, logical_dup = 0;
  for(int i=0;i<int(out.size());++i) {
    holes += out[i] < 0;
    physical_dup += std::max(0, ph[i] - 1);
    logical_dup += std::max(0, lh[i] - 1);
    valid &= out[i]>=0&&ph[i]==1&&lh[i]==1;
  }
  std::printf("L142 consumer-model=%s holes=%d physical-dup=%d logical-dup=%d valid=%d\n",
      model == ConsumerModel::DirectShippingPairs ? "direct-shipping-pairs" : "whole-converter32",
      holes, physical_dup, logical_dup, int(valid));
  model_valid = valid;
  return out;
}

uint64_t map_hash(std::vector<int> const& map) {
  uint64_t h=UINT64_C(1469598103934665603);
  for (int i=0;i<int(map.size());++i) {
    hash_word(h,uint64_t(i)); hash_word(h,uint64_t(map[i]+1));
  }
  return h;
}

bool prove_shipping_emitter_semantic_rank() {
  // The converter already owns the int4 LOP3/FMA sequence.  Reusing it is
  // correct only if ChunkPlace can still see the semantic K mode.  The
  // tempting rank-2 production shape has the right linear strides but no K
  // mode, so ka() is identically zero.  Keep that exact mistake as the red
  // control; the singleton-N, explicit-K view changes no address but makes
  // the converter's chunk algebra name the real cohort.
  using Ranked = Layout<Shape<Shape<_2,_2,_2>,_1,_4>,
                        Stride<Stride<_1,_2,_4>,_0,_8>>;
  using Rank2 = Layout<Shape<Shape<_2,_2,_2>,_4>,
                       Stride<Stride<_1,_2,_4>,_8>>;
  int ranked_keep_bad = 0;
  int ranked_at_bad = 0;
  int rank2_keep_bad = 0;
  int rank2_at_bad = 0;
  int ranked_kept[4] = {};
  int rank2_kept[4] = {};
  for (int wk = 0; wk < 4; ++wk) {
    for (int v = 0; v < 4; ++v) {
      for (int t = 0; t < 4; ++t) {
        bool const expected =
            (v == wk / 2 || v == wk / 2 + 2) &&
            (t == 2 * (wk % 2) || t == 2 * (wk % 2) + 1);
        int const expected_at = 2 * (v >= 2) + (t & 1);
#define L142_CHECK_EMIT(WK) do {                                                   \
          using Good = cutlass::MixGemmChunkEmit<4, WK, 4, true, Ranked>;          \
          using Bad  = cutlass::MixGemmChunkEmit<4, WK, 4, true, Rank2>;           \
          bool const good_keep = Good::keep(t, v);                                 \
          bool const bad_keep = Bad::keep(t, v);                                   \
          ranked_keep_bad += good_keep != expected;                                \
          rank2_keep_bad += bad_keep != expected;                                  \
          ranked_kept[WK] += good_keep;                                             \
          rank2_kept[WK] += bad_keep;                                               \
          if (good_keep && expected) ranked_at_bad += Good::at(t, v) != expected_at;\
          if (bad_keep && expected) rank2_at_bad += Bad::at(t, v) != expected_at;  \
        } while (false)
        switch (wk) {
          case 0: L142_CHECK_EMIT(0); break;
          case 1: L142_CHECK_EMIT(1); break;
          case 2: L142_CHECK_EMIT(2); break;
          case 3: L142_CHECK_EMIT(3); break;
        }
#undef L142_CHECK_EMIT
      }
    }
  }
  bool const ok = ranked_keep_bad == 0 && ranked_at_bad == 0 &&
                  ranked_kept[0] == 4 && ranked_kept[1] == 4 &&
                  ranked_kept[2] == 4 && ranked_kept[3] == 4 &&
                  rank2_keep_bad == 24 &&
                  rank2_kept[0] == 16 && rank2_kept[1] == 0 &&
                  rank2_kept[2] == 0 && rank2_kept[3] == 0;
  std::printf(
      "L142 shipping-emitter ranked keep-bad=%d at-bad=%d kept=%d/%d/%d/%d; "
      "rank2 keep-bad=%d at-bad=%d kept=%d/%d/%d/%d %s\n",
      ranked_keep_bad, ranked_at_bad, ranked_kept[0], ranked_kept[1],
      ranked_kept[2], ranked_kept[3], rank2_keep_bad, rank2_at_bad,
      rank2_kept[0], rank2_kept[1], rank2_kept[2], rank2_kept[3],
      ok ? "SEMANTIC-RANK-EXACT+RANK2-EXPECTED-RED" : "FAIL");
  return ok;
}
}

int main() {
  bool const emitter_rank_ok = prove_shipping_emitter_semantic_rank();
  bool const shadow_lane_ok = prove_shadow_lane_preservation();
  bool const pair_destinations_ok = print_anchored_pair_destinations();
  auto prod = production_map(false);
  auto compact = production_map(true);
  auto const shipping = xplane::plane_map<kBits,kTM,kTN,kTK,
                                           kWM,kWN,kFold,kArtifactTK>();
  bool consumer_valid = false, whole_valid = false;
  auto consumer = consumer_map(ConsumerModel::DirectShippingPairs, consumer_valid);
  auto whole = consumer_map(ConsumerModel::WholeConverter32, whole_valid);
  (void)whole_valid;
  int diff=0;
  for (int i=0;i<int(prod.map.size());++i) diff += prod.map[i]!=compact.map[i];
  int consumer_diff=consumer.size()==shipping.size()?0:-1;
  if(consumer_diff==0)for(int i=0;i<int(shipping.size());++i)consumer_diff+=consumer[i]!=shipping[i];
  int whole_diff=whole.size()==shipping.size()?0:-1;
  if(whole_diff==0)for(int i=0;i<int(shipping.size());++i)whole_diff+=whole[i]!=shipping[i];
  int ea96_vs_shipping=prod.map.size()==shipping.size()?0:-1;
  if(ea96_vs_shipping==0)for(int i=0;i<int(shipping.size());++i)ea96_vs_shipping+=prod.map[i]!=shipping[i];
  bool ok=emitter_rank_ok && shadow_lane_ok && pair_destinations_ok && derived_clean(prod) && derived_clean(compact) && diff>0 &&
          compact.hash==UINT64_C(0x17dfe6248fc38143) && consumer_valid && consumer_diff==0 &&
          map_hash(consumer)==UINT64_C(0xb89b157b5b1bd6c3) &&
          whole_diff>0 && ea96_vs_shipping>0;
  std::printf("L142 production-layout map entries=%zu conflicts=%d holes=%d logical-dup=%d physical-dup=%d hash=%016llx\n",
      prod.map.size(),prod.conflicts,prod.holes,prod.logical_duplicates,
      prod.physical_duplicates,(unsigned long long)prod.hash);
  std::printf("L142 compact-s16 hash=%016llx map-diff=%d %s result=%s\n",
      (unsigned long long)compact.hash,diff,diff?"EXPECTED-RED":"UNEXPECTED-GREEN",
      ok?"PASS":"FAIL");
  std::printf("L142 production-consumer element-map-diff=%d/%zu hash=%016llx %s\n",
      consumer_diff,shipping.size(),(unsigned long long)map_hash(consumer),
      consumer_diff==0?"SHIPPING-EXACT":"FAIL");
  std::printf("L142 whole-converter32 element-map-diff=%d/%zu %s\n",
      whole_diff,shipping.size(),whole_diff>0?"EXPECTED-RED":"UNEXPECTED-GREEN");
  std::printf("L142 ea96-artifact element-map-diff=%d/%zu %s\n",
      ea96_vs_shipping,shipping.size(),ea96_vs_shipping>0?"EXPECTED-RED":"UNEXPECTED-GREEN");
  return ok?0:1;
}

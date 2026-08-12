// L142 -- re-anchor L138's two-source map to the production SmemLayoutB.
//
// L138 deliberately used a compact row-major half tensor to expose the MMA
// fragment.  The collective does not: it partitions the tiled/swizzled B
// stage.  Same shape, different strides, and converter placement is a stride
// contract.  Reuse L138's real source-side Copy_Traits machinery, but derive
// the destination from the exact production stage layout.
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

enum class ConsumerModel { WholeConverter32, PackedSelected8 };

int source_slot(SourceFragment const& sf, int ni, int vreg, int code) {
  int found = -1;
  for (int d = 0; d < int(sf.physical.size()); ++d) {
    if (sf.ni[d] != ni || sf.vreg[d] != vreg || sf.code[d] != code) continue;
    if (found >= 0) return -2;
    found = d;
  }
  return found;
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
        // The proved consumer treats the 128-code source as four independent
        // NI groups.  For each NI it packs two selected bytes from each of two
        // vregs into one 8-code word, then invokes the shipping base-8
        // converter.  That converter's own MixGemmEmit order names the eight
        // destination slots.  This is the exact word-read/output-write shape
        // used by the production branch below.
        int const v0 = wk / 2;
        int const v1 = v0 + 2;
        int const phase = wk % 2;
        int const codes[4] = {2 * phase, 2 * phase + 1,
                              2 * phase + 4, 2 * phase + 5};
        for (int ni = 0; ni < 4; ++ni) {
          int packed_code = 0;
          for (int v : {v0, v1})
            for (int code : codes) {
              int const d = source_slot(sf, ni, v, code);
              valid &= d >= 0;
              if (d < 0) { ++packed_code; continue; }
              int const e = cutlass::MixGemmEmit<4>::index(packed_code, 0);
              int const p = sf.physical[d];
              int const l = refs[32 * sk + 8 * ni + e].logical;
              valid &= p >= 0 && p < int(out.size()) && l >= 0 && l < int(out.size());
              if (p >= 0 && p < int(out.size()) && l >= 0 && l < int(out.size())) {
                valid &= out[p] < 0 || out[p] == l;
                out[p] = l; ++ph[p]; ++lh[l];
              }
              ++packed_code;
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
      model == ConsumerModel::PackedSelected8 ? "packed-selected8" : "whole-converter32",
      holes, physical_dup, logical_dup, int(valid));
  model_valid = valid;
  return out;
}
}

int main() {
  auto prod = production_map(false);
  auto compact = production_map(true);
  bool consumer_valid = false, whole_valid = false;
  auto consumer = consumer_map(ConsumerModel::PackedSelected8, consumer_valid);
  auto whole = consumer_map(ConsumerModel::WholeConverter32, whole_valid);
  int diff=0;
  for (int i=0;i<int(prod.map.size());++i) diff += prod.map[i]!=compact.map[i];
  int consumer_diff=consumer.size()==prod.map.size()?0:-1;
  if(consumer_diff==0)for(int i=0;i<int(prod.map.size());++i)consumer_diff+=consumer[i]!=prod.map[i];
  int whole_diff=whole.size()==prod.map.size()?0:-1;
  if(whole_diff==0)for(int i=0;i<int(prod.map.size());++i)whole_diff+=whole[i]!=prod.map[i];
  bool ok=derived_clean(prod) && derived_clean(compact) && diff>0 &&
          compact.hash==UINT64_C(0x17dfe6248fc38143) && consumer_valid && consumer_diff==0 &&
          whole_diff>0;
  std::printf("L142 production-layout map entries=%zu conflicts=%d holes=%d logical-dup=%d physical-dup=%d hash=%016llx\n",
      prod.map.size(),prod.conflicts,prod.holes,prod.logical_duplicates,
      prod.physical_duplicates,(unsigned long long)prod.hash);
  std::printf("L142 compact-s16 hash=%016llx map-diff=%d %s result=%s\n",
      (unsigned long long)compact.hash,diff,diff?"EXPECTED-RED":"UNEXPECTED-GREEN",
      ok?"PASS":"FAIL");
  std::printf("L142 production-consumer element-map-diff=%d/%zu %s\n",
      consumer_diff,prod.map.size(),consumer_diff==0?"EXACT":"FAIL");
  std::printf("L142 whole-converter32 element-map-diff=%d/%zu %s\n",
      whole_diff,prod.map.size(),whole_diff>0?"EXPECTED-RED":"UNEXPECTED-GREEN");
  return ok?0:1;
}

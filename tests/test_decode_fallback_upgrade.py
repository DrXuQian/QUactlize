"""Fallback inherits implementation improvements, not a false measured label."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_q8_bucket_inherits_its_donor_upgrade(tmp_path):
    source = tmp_path / 'fallback.cpp'
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> const kImages{};\n'
                                      'static char const kJitSource[]="";\n')
    source.write_text(r'''
#include "quactlize/dispatch/binding.cpp"
#include "tests/legacy_q8_overlay.hpp"
#include <cassert>
#include <cstring>

int main() {
  using namespace quactlize::dispatch;
  int simt=0,tc=0,table_misses=0;
  for(auto const& row:matched::data::kExact) {
    auto const& choice=matched::data::kChoices[row.choice];
    qkg_simt_call_v2 d{2,sizeof(d)};auto& c=d.call;
    c.version=1;c.size=sizeof(c);c.qtype=row.q;c.input_type=QKG_F32;
    c.mode=row.mode;c.rows=row.tokens*(row.mode==QKG_INDEXED?row.topk:1);
    c.n=row.n;c.k=row.k;c.experts=row.experts;c.topk=row.topk;c.channels=row.channels;
    d.compute_type=row.compute;
    qkg_simt_config_v1 expected{};
    if(choice.kind==QKS_SMALLM_TC) {
      if(!q8_vector::select_tc(d,choice.tc,expected)) continue;
      ++tc;
    } else if(choice.kind==QKS_SMALLM_SIMT) {
      auto f=choice.reader;expected={1,sizeof(expected),f.variant,f.columns,f.warps,f.values,f.split};
      if(!q8_vector::select(d,expected)) continue;
      ++simt;
    } else continue;
    // No equality to the original N/K is necessary to inherit this donor.
    c.n+=256;
    qkg_simt_config_v1 actual{};
    assert(q8_vector::select_bucket(d,row,choice,actual));
    assert(!std::memcmp(&actual,&expected,sizeof(actual)));
    for(int field=0;field<11;++field) {
      auto b=d;
      switch(field) {
        case 0:b.call.qtype=14;break;
        case 1:b.compute_type=1-b.compute_type;break;
        case 2:b.call.mode=2-b.call.mode;break;
        case 3:b.call.input_type=QKG_F16;break;
        case 4:b.call.n=row.n*2+256;break;
        case 5:b.call.k=row.k*2+256;break;
        case 6:b.call.rows=9*b.call.topk;break;
        case 7:b.call.channels=9;break;
        case 8:b.call.experts+=1;break;
        case 9:b.call.topk=0;break;
        default:b.call.rows=0;
      }
      actual={1,sizeof(actual),1,4,4,4,1};auto saved=actual;
      assert(!q8_vector::select_bucket(b,row,choice,actual));
      assert(!std::memcmp(&saved,&actual,sizeof(actual)));
    }
    // Exercise the real bucket lookup, not only a supplied donor.
    auto selected=matched::select(d);
    if(selected.row && selected.policy==QKS_MATCHED_BUCKET) {
      auto const& donor=*selected.row;
      auto const& f=matched::data::kChoices[donor.choice];
      if(q8_vector::select_bucket(d,donor,f,actual)) {
        Runtime runtime; // Empty module catalog; a reader upgrade must not JIT.
        c.a_row_stride=c.k;c.a_token_stride=int64_t(c.k)*c.channels;
        c.ids_stride=c.topk;c.out_row_stride=c.n;
        auto arr=q8_kpack2::arrangement();qks_smallm_choice_v2 result{};
        assert(quactlize_kpack_dispatch_query_smallm_v3(&runtime,&d,&arr,&result)==QKS_OK);
        assert(result.base.kind==QKS_SMALLM_SIMT && result.base.policy==QKS_MATCHED_BUCKET);
        assert(result.base.source_n==donor.n && result.base.source_k==donor.k);
        assert(result.base.source_tokens==donor.tokens && result.compute_type==donor.compute);
        assert(!std::memcmp(&actual,&result.base.simt,sizeof(actual)));
        assert(result.base.sizes.workspace_bytes==(actual.split==1?0:uint64_t(c.rows)*c.n*actual.split*4));
        assert(runtime.plans.empty());++table_misses;
      }
    }
  }
  assert(simt>0 && tc==2 && table_misses>0);
}
''')
    binary=tmp_path/'fallback'
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT),'-I'+str(tmp_path),
                    source,'-ldl','-o',binary],check=True)
    subprocess.run([binary],check=True)


def test_bucket_upgrade_is_not_relabelled_exact():
    binding=(ROOT/'quactlize/dispatch/binding.cpp').read_text()
    body=binding.split('int quactlize_kpack_dispatch_query_smallm_v3(',1)[1].split(
        'int quactlize_kpack_dispatch_query_dense_io_v1(',1)[0]
    assert 'select_smallm(*typed,*arrangement)' in body
    assert 'q8_vector::' not in body


def test_fast_reduction_is_shared_by_both_readers():
    simt=(ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    q8=(ROOT/'quactlize/execution/simt_q8_vector.cuh').read_text()
    assert simt.count('launch_reduction<Q>(c,split,stream);')==2
    assert 'else launch_reduction<8>(c,split,stream);' in q8
    body=simt.split('void launch_reduction(',1)[1].split('template<int Q,int Variant',1)[0]
    assert 'model_gemv::vector_reduction(c,split)' in body
    assert 'register_reuse_reduce<Q>' in body
    assert all(f'reduce_decode_rows<{s}>' in body for s in (2,4,8))
    assert '2048' not in body and '4096' not in body


def test_q5_generic_and_q8_dynamic_optimized_paths_are_not_old_shape_only():
    simt=(ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    q8=(ROOT/'quactlize/execution/simt_q8_vector.cuh').read_text()
    assert 'constexpr int Changes=kBf16F32Changes<Q,Variant,Columns,Warps,P>;' in simt
    assert 'register_reuse<Q,1,Variant,Columns,Warps,P,1,Changes>' in simt
    assert 'auto strategy=q8_strategy(d,Variant,Columns,Warps,P,split);' in q8
    # Known narrow-window and constant-folded winners are still preserved.
    assert 'kernel_s1<1,0,1,4,8,4,false>' in q8
    assert 'kernel_model<1,0,1,8,4,4,false,2048,4096,8>' in q8

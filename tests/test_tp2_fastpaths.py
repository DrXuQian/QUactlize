"""Host contracts for shape-independent candidates, not PPU admission."""
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def compile_run(tmp_path, text):
    source = tmp_path / 'host.cpp'
    source.write_text(text)
    exe = tmp_path / 'host'
    subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-I'+str(ROOT), source, '-o', exe], check=True)
    subprocess.run([exe], check=True)


def test_prepare_fallback_is_structural_not_a_model_shape(tmp_path):
    text = (ROOT/'quactlize/execution/moe_prepare.cuh').read_text()
    body = text[text.index('template<class Plan>\nCUTLASS_HOST_DEVICE bool supported'):
                text.index('template<class Shape,class Stride,class Plan>\nbool launch_all_simt')]
    compile_run(tmp_path, '''#include <cassert>
#include <cstdint>
#define CUTLASS_HOST_DEVICE
struct IO { int tokens=1,topk=8,channels=1; };
struct Projection { IO io; int n=1024,k=2048,experts=256,m=8,tile_m=8; };
struct Router { int version=1,use_sigmoid=0,with_norm=1,delayed_softmax=0; void* bias=nullptr; };
struct Plan { Projection gate,down,up; Router router; bool merged=true; int mask=5; };
int moe_simt_mask(Plan const& p){return p.mask;}
''' + body + '''
int main(){
  Plan p;p.down.n=3072;p.down.k=512;p.gate.k=3072;
  for(int t=1;t<=8;++t){p.gate.io.tokens=t;p.gate.m=8*t;
    assert(all_simt_supported(p));}
  p.gate.io.tokens=1;p.gate.m=8;
  for(int bad=0;bad<13;++bad){auto b=p;
    switch(bad){
      case 0:b.gate.io.tokens=9;break;case 1:b.gate.m=9;break;
      case 2:b.gate.experts=128;break;case 3:b.gate.io.topk=4;break;
      case 4:b.gate.io.channels=8;break;case 5:b.down.tile_m=0;break;
      case 6:b.mask=1;break;case 7:b.merged=false;break;
      case 8:b.router.bias=&p;break;case 9:b.router.delayed_softmax=1;break;
      case 10:b.down.k=768;break;case 11:b.gate.n=0;break;default:b.gate.k=-256;
    }assert(!all_simt_supported(b));
  }
  p.gate.k=2048;p.down.n=2048;assert(all_simt_supported(p));
}''')
    launcher=text.split('void launch(Plan const& plan,hggcStream_t stream)',1)[1]
    assert 'if(all_simt_supported(plan))' in launcher
    assert 'admitted(' not in text


def test_actual_vector_reducer_rows_stride_and_scalar_fallback(tmp_path):
    text = (ROOT/'quactlize/execution/simt_reducer.cuh').read_text()
    body = text[text.index('template<int Splits>'):text.index('} // namespace')]
    compile_run(tmp_path, '''#include "quactlize/execution/model_gemv_scope.hpp"
#include <cassert>
#include <vector>
#include <cstring>
#include <initializer_list>
#define __global__
#define CUTLASS_PRAGMA_UNROLL
struct Dim {int x;};Dim blockIdx,blockDim,threadIdx;
struct alignas(8) float2 {float x,y;};
''' + body + r'''
template<int S> void check(int rows,int n,int stride) {
  std::vector<float> partial(size_t(rows)*S*n),out(size_t(rows)*stride+16,-999.f),gold=out;
  for(int r=0;r<rows;++r) for(int s=0;s<S;++s) for(int c=0;c<n;++c)
    partial[(r*S+s)*n+c]=float((r*r*13+c*7+s*s*31+r*s*17)%113-56)*.03125f;
  for(int r=0;r<rows;++r) for(int c=0;c<n;++c){float sum=0;
    for(int s=0;s<S;++s)sum+=partial[(r*S+s)*n+c];gold[r*stride+c]=sum;}
  blockDim.x=32;
  for(blockIdx.x=0;blockIdx.x<(rows*(n/2)+31)/32;++blockIdx.x)
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
      reduce_decode_rows<S>(partial.data(),out.data(),rows,n,stride);
  assert(!memcmp(out.data(),gold.data(),out.size()*sizeof(float)));
  if(rows>1){int bad=0;for(int r=0;r<rows;++r)for(int c=0;c<n;++c){
    float wrong=0;for(int s=0;s<S;++s)wrong+=partial[(s*rows+r)*n+c];bad+=wrong!=gold[r*stride+c];}assert(bad);}
}
int main(){
  for(int r:{1,2,3,4,5,6,7,8,64}) for(int n:{2,256,512,3072}) for(int pad:{0,2}) {
    check<2>(r,n,n+pad);check<4>(r,n,n+pad);check<8>(r,n,n+pad);
  }
  using quactlize::execution::model_gemv::vector_reduction;
  qkg_call_v1 c{};c.rows=3;c.n=512;c.out_row_stride=514;c.output=(float*)0x1000;c.workspace=(void*)0x2000;
  for(int s:{2,4,8})assert(vector_reduction(c,s));assert(!vector_reduction(c,1));
  c.output=(float*)0x1004;assert(!vector_reduction(c,4));c.output=(float*)0x1000;
  c.workspace=(void*)0x2004;assert(!vector_reduction(c,4));c.workspace=(void*)0x2000;
  c.out_row_stride=513;assert(!vector_reduction(c,4));c.out_row_stride=514;
  c.n=511;assert(!vector_reduction(c,4));c.n=512;
  c.out_row_stride=510;assert(!vector_reduction(c,4));
}''')


def test_caller_carries_local_dimensions(tmp_path):
    llama = Path(os.environ['LLAMA_CI_DIR'])
    text = (llama/'ggml/src/ggml-cuda/quactlize-execution.cu').read_text()
    body = text[text.index('qkg_gate_up_call_v1 paired_call('):text.index('qzd_call_v1 expansion_call(')]
    compile_run(tmp_path, '#include "quactlize/fusion/gate_up.h"\n#include <cassert>\n' + body + '''
int main(){
  auto d=paired_call(8,1024,3072,1,7,QKG_COMPUTE_F16,false);auto c=d.input.call;
  assert(c.n==1024 && c.k==3072 && c.rows==7 && c.a_row_stride==3072 && c.out_row_stride==1024);
  d=paired_call(12,512,3072,256,3,QKG_COMPUTE_BF16,true);c=d.input.call;
  assert(c.n==512 && c.k==3072 && c.rows==24 && c.topk==8 && c.ids_stride==8);
  assert(d.round_projection==1 && d.input.compute_type==QKG_COMPUTE_BF16);
}''')
    assert 'gate.n==1024' not in text and 'gate.k==2048' not in text
    binding=(ROOT/'quactlize/dispatch/binding.cpp').read_text().split('quactlize_kpack_dispatch_moe_bind_gate_up_v1',1)[1]
    assert 'v.n=p.gate.n/2' in binding and 'v.n=512' not in binding


def test_bounded_inventory_keeps_incumbents_and_declares_access():
    from dataclasses import asdict
    from dev.tp2_decode import plan
    from dev.gemv_model.source import source
    from dev.gemv_model.access import access
    assert len(plan.PRIORITY)==10 and len(plan.POINTS)==17
    assert plan.TOKEN_CONTROLS==tuple(range(1,9))
    assert len({p.name for p in plan.POINTS})==len(plan.POINTS)
    assert len([p for p in plan.PRIORITY if p.tc])==4
    for p in plan.POINTS:
        cs=plan.candidates(p)
        assert 1<len(cs)<=9 and len({c.name for c in cs})==len(cs)
        assert all(p.physical_n%c.tile_n==0 for c in cs)
        text=source(p,cs)
        assert text.count('    case ')==len(cs)
        assert f'c.n!={p.n} || c.k!={p.k}' in text
        if p.q==8 and cs[0].variant==1:
            assert 'register_reuse_body<8,1,1,' in text
        if p.q==12 and not p.paired:
            assert 'ppu_arrangements::q4_kpack4_transpose_v1()' in text
        for c in cs:
            a=access(p,c,dict(A=0,low=0,high=0,units=0))
            assert a['B_contiguous_bytes_per_k_worker_group']==2*c.tile_n
            assert all(s['width_bytes']>0 for s in a['streams'])
            if c.vector_reduce:
                assert not p.paired and c.split in (2,4,8)
                assert f'reduce_decode_rows<{c.split}>' in text
        if p in plan.PRIORITY and not p.tc:
            assert cs[0].name=='clone' and cs[0].split==1
    small=next(p for p in plan.PRIORITY if p.name=='tp2-q8-shared-down')
    assert {k:asdict(plan.candidates(small)[0])[k] for k in ('variant','columns','warps','values','split')}==dict(
        variant=1,columns=4,warps=2,values=2,split=1)


def test_explicit_fusion_queries_accept_local_sizes_without_promoting_them(tmp_path):
    compile_run(tmp_path, r'''#include "quactlize/fusion/validation.hpp"
#include <cassert>
#include <initializer_list>
int main(){
  using namespace quactlize::fusion;
  for(int q:{8,12})for(int n:{256,512,1024})for(int k:{512,2048,3072})for(int t=1;t<=8;++t){
    qkg_gate_up_call_v1 d{};d.version=1;d.size=sizeof(d);
    d.input.version=2;d.input.size=sizeof(d.input);d.input.compute_type=q==12;
    d.output_type=QKG_F32;d.round_projection=q==12;
    auto& c=d.input.call;c.version=1;c.size=sizeof(c);c.qtype=q;c.n=n;c.k=k;
    c.experts=q==12?256:1;c.mode=q==12?QKG_INDEXED:QKG_DENSE;c.input_type=QKG_F32;
    c.topk=q==12?8:1;c.channels=1;c.rows=t*c.topk;c.ids_stride=c.topk;
    c.a_row_stride=k+4;c.a_token_stride=k+4;c.out_row_stride=n+2;
    qkg_gate_up_layout_v1 layout{1,sizeof(layout),QKG_GATE_UP_N4_V1,
      q==8?q8_kpack2::arrangement():ppu_arrangements::q4_kpack4_transpose_v1()};
    qkg_sizes_v1 result{},expected{};
    assert(!quactlize::execution::sizes(q,2*n,k,c.experts,&layout.packing,expected));
    for(int backend:{QKG_GATE_UP_SIMT,QKG_GATE_UP_TC})for(int split:{1,2,4,8}){
      qkg_gate_up_config_v1 f{1,sizeof(f),backend,split,backend?8:0,backend?0:8};
      assert(query(d,f,layout,result)==QKG_OK);
      assert(result.low_bytes==expected.low_bytes && result.units_bytes==expected.units_bytes);
      assert(result.workspace_bytes==(split==1?0:uint64_t(c.rows)*2*n*split*4));
    }
    qkg_gate_up_config_v1 selected{};
    bool measured=(n==512 && k==2048) || (q==8 && n==1024 && k==3072 && t==1);
    assert((select(q,n,k,c.experts,t,d.input.compute_type,&selected)==QKG_OK)==measured);
  }
}''')


@pytest.mark.parametrize('topk', [1,8])
@pytest.mark.parametrize('write_padding', [False,True])
def test_reducer_control_orchestration_checks_padding_and_releases(topk,write_padding):
    from dev.gemv_model.engine import vector_reducer_controls
    from dev.gate_up_perf.bench import Buffer
    class Host:
        def __init__(self):self.allocations=[];self.memory={};self.next=4096
        def allocate(self,size):
            p=self.next;self.next+=(size+1023)//1024*1024
            self.allocations.append(p);self.memory[p]=np.zeros(size,'u1');return p
        def view(self,p,size):
            base=next(k for k,v in self.memory.items() if k<=p and p+size<=k+v.size)
            return self.memory[base][p-base:p-base+size]
        def fill(self,p,size,byte=0xA5):self.view(p,size)[:]=byte
        def download(self,p,size):return self.view(p,size).copy()
        def sync(self):pass
        def release_after(self,count):
            for p in self.allocations[count:]:del self.memory[p]
            del self.allocations[count:]
    rt=Host()
    b=SimpleNamespace(rt=rt,point=SimpleNamespace(n=256),topk=topk,a=Buffer(rt,16))
    b.call=lambda copy=0:SimpleNamespace()
    original=b.call
    def update(tokens,repeat):
        b.rows=tokens*topk
        b.gold=(np.arange(b.rows*256,dtype='f4').reshape(b.rows,256)%31+repeat+1)
        b.denom=np.ones_like(b.gold)*100
    b.update=update
    def prepare():
        c=b.call()
        def launch():
            for r,row in enumerate(b.gold):
                rt.view(c.output+r*c.out_row_stride*4,1024)[:]=row.view('u1')
            rt.fill(c.workspace,b.rows*256*4*4,0)
            if write_padding:rt.fill(c.output+1024,4,0)
            return 0
        return launch
    p=SimpleNamespace(b=b,config=SimpleNamespace(split=4),prepare=prepare,close=lambda:None)
    if write_padding:
        with pytest.raises(ValueError,match='padding'):vector_reducer_controls(p)
    else:
        rows=vector_reducer_controls(p)
        assert len(rows)==8 and all(r['error']==0 for r in rows)
    assert b.call is original and len(rt.allocations)==1

"""Fixture publication order; these host tests do not admit a device kernel."""
from pathlib import Path
import subprocess
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('unstable_ties', [False, True])
def test_actual_router_top8_and_logits_alias(tmp_path, unstable_ties):
    command=['g++','-std=c++20','-O2','-pthread','-ffp-contract=off',
             '-I'+str(ROOT),str(ROOT/'tests/moe_router_warp_host.cpp')]
    if unstable_ties:
        source=(ROOT/'quactlize/execution/moe_router_warp.cuh').read_text()
        assert source.count('key[a+1]>key[a]')==1
        source=source.replace('key[a+1]>key[a]','key[a+1]>=key[a]')
        source=source.replace('../integrations/llama/router.cuh',
                              str(ROOT/'quactlize/integrations/llama/router.cuh'))
        mutant=tmp_path/'unstable.cuh';mutant.write_text(source)
        command.append('-DQK_ROUTER_CANDIDATE="'+str(mutant)+'"')
    binary=tmp_path/'router'
    subprocess.run(command+['-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180)
    if unstable_ties:
        assert result.returncode!=0, 'non-stable top8 negative escaped'
    else:
        assert result.returncode==0,result.stderr
        assert 'MOE_ROUTER_HOST PASS cases=40 ' in result.stdout


def test_prepare_fast_scope_and_m1_publication():
    text=(ROOT/'quactlize/execution/moe_prepare.cuh').read_text()
    assert 'constexpr bool OneWarp=AllSimt && Capacity==8' in text
    assert 'AllSimt && !OneWarp && plan.router.version && !plan.router.bias' in text
    assert text.index('if (proven) publish(true)') < text.index('int mine=0')
    assert 'if constexpr(!OneWarp) if (!proven) __syncthreads()' in text
    assert '__match_any_sync(0xffffffff,id)' in text
    admission=text[text.index('CUTLASS_HOST_DEVICE bool admitted'):]
    assert '!r.delayed_softmax && !r.bias' in admission
    assert 'r.with_norm' in admission and '!r.use_sigmoid' in admission


def test_prepare_integration_compares_the_exact_previous_source(tmp_path):
    from dev.moe_prepare.build import baseline_headers
    from dev.moe_prepare.run_integration import BASELINE, CASES
    receipt=baseline_headers(BASELINE,tmp_path)
    assert receipt['commit']==BASELINE
    assert len(receipt['source_hashes'])==2
    before=subprocess.check_output(['git','show',BASELINE+':quactlize/execution/moe_prepare.cuh'],cwd=ROOT,text=True)
    assert (tmp_path/'prepare-incumbent.cuh').read_text()==before.replace(
        'prepare_detail','prepare_incumbent').replace('router_256_top8_warp','router_256_top8_warp_incumbent').replace(
        '"moe_router_warp.cuh"','"router-incumbent.cuh"').replace(
        '"../runtime/moe_chain.cuh"','"'+str(ROOT/'quactlize/runtime/moe_chain.cuh')+'"')
    assert len(CASES)==len(set(CASES))==16
    assert {c[0] for c in CASES}=={1,2,4,8} and {c[5] for c in CASES}=={0,1}
    text=(ROOT/'dev/moe_prepare/bench.cu').read_text()
    assert 'prepare_incumbent::launch<Shape,Stride>(plan,stream)' in text
    assert 'direct|=arm==0 && prepare_incumbent::admitted(plan)' in text


def test_prepare_integration_timing_denominators_and_negatives():
    from dev.moe_prepare.run_integration import timing_rows
    def row(arm,median=3,samples=None):
        sample_text=json.dumps([3.]*60 if samples is None else samples,separators=(',',':'))
        return f'MOE_PREPARE_TIME arm={arm} median_us={median} samples={sample_text}'
    text=row(0)+'\n'+row(1)
    assert len(timing_rows(text))==2
    for bad in (row(0),text+'\n'+row(1),row(0)+'\n'+row(0),
                row(0,4)+'\n'+row(1),row(0,samples=[3.]*59)+'\n'+row(1),
                row(0,samples=[float('nan')]*60)+'\n'+row(1)):
        with pytest.raises(ValueError):timing_rows(bad)


def test_production_prepare_promotes_only_measured_all_simt_domain(tmp_path):
    text=(ROOT/'quactlize/execution/moe_prepare.cuh').read_text()
    body=text[text.index('template<class Plan>\nCUTLASS_HOST_DEVICE bool admitted'):text.index('\ntemplate<class Shape,class Stride,class Plan>\nvoid launch')]
    source=tmp_path/'admission.cpp'
    source.write_text('''#include <cassert>
#define CUTLASS_HOST_DEVICE
struct IO { int tokens=1; };
struct Projection { IO io; int n=1024,k=2048; };
struct Router { int version=1,use_sigmoid=0,with_norm=1,delayed_softmax=0; void* bias=nullptr; };
struct Plan { Projection gate,down; Router router; bool merged=true,supported=true; int mask=5; };
bool supported(Plan const& p){return p.supported;}
int moe_simt_mask(Plan const& p){return p.mask;}
''' + body + '''
int main(){
  Plan p;p.down.n=2048;p.down.k=512;
  for(int t=1;t<=8;++t){p.gate.io.tokens=t;assert(admitted(p)==(t==1||t==2||t==4||t==8));}
  p.gate.io.tokens=1;
  for(int m=0;m<8;++m){p.mask=m;assert(admitted(p)==(m==5));}
  p.mask=5;p.router.bias=&p;assert(!admitted(p));p.router.bias=nullptr;
  p.router.use_sigmoid=1;assert(!admitted(p));p.router.use_sigmoid=0;
  p.router.with_norm=0;assert(!admitted(p));p.router.with_norm=1;
  p.router.delayed_softmax=1;assert(!admitted(p));p.router.delayed_softmax=0;
  p.gate.k=3072;assert(!admitted(p));p.gate.k=512;assert(admitted(p));
  p.merged=false;assert(!admitted(p));
}''')
    exe=tmp_path/'admission'
    subprocess.run(['g++','-std=c++17',source,'-o',exe],check=True)
    subprocess.run([exe],check=True)


def test_actual_stream_upload_body_orders_pageable_map_and_lifetime(tmp_path):
    text = (ROOT/'dev/moe_prepare/bench.cu').read_text()
    begin = text.index('  void put_on_stream(')
    end = text.index('  std::vector<T> get()', begin)
    method = text[begin:end]
    # Model the legal behavior where pageable H2D returns before DMA finishes.
    # The unchanged-stream negative must be observable, never a lucky copy.
    cpp = tmp_path/'stream.cpp'
    cpp.write_text(r'''
#include <cassert>
#include <cstring>
#include <vector>
using cudaStream_t=int;
constexpr int cudaMemcpyHostToDevice=1;
struct Pending { void* dst; const void* src; size_t size; int stream; };
std::vector<Pending> pending;
int cudaMemcpyAsync(void* dst,const void* src,size_t size,int,int stream) {
  pending.push_back({dst,src,size,stream});return 0;
}
int cudaStreamSynchronize(int stream) {
  for(auto it=pending.begin();it!=pending.end();) {
    if(it->stream==stream) {std::memcpy(it->dst,it->src,it->size);it=pending.erase(it);}
    else ++it;
  }
  return 0;
}
void ck(int code){assert(!code);}
void require(bool yes,const char*){assert(yes);}
template<class T> struct Buffer {T* ptr;size_t count;
''' + method + r'''
};
int main() {
  int device[2]={0,1};Buffer<int> b{device,2};
  std::vector<int> bad={1,0},original={0,1};
  cudaMemcpyAsync(device,bad.data(),sizeof(device),cudaMemcpyHostToDevice,0);
  cudaStreamSynchronize(7);
  assert(device[0]==0 && device[1]==1); // Legacy negative can remain invisible.
  cudaStreamSynchronize(0);b.put_on_stream(original,7);
  b.put_on_stream(bad,7);
  assert(device[0]==1 && device[1]==0 && pending.empty());
  b.put_on_stream(original,7);
  assert(device[0]==0 && device[1]==1 && pending.empty());
}
''')
    exe=tmp_path/'stream'
    subprocess.run(['g++','-std=c++17','-O2',cpp,'-o',exe],check=True)
    subprocess.run([exe],check=True)
    proof=text[text.index('  void activation_proof('):text.index('  void correctness(')]
    assert 'down.map.put_on_stream(bad,stream)' in proof
    assert 'down.map.get()==bad' in proof
    assert 'expected_negative!=positive' in proof and 'out==expected_negative' in proof
    assert 'down.map.put_on_stream(map,stream)' in proof
    assert 'ids.put_on_stream(bad,stream)' in text
    assert 'ids.put_on_stream(in,stream)' in text
    timing=text[text.index('  void benchmark('):]
    assert 'put_on_stream' not in timing

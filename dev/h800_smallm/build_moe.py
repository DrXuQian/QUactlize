#!/usr/bin/env python3
"""Build isolated prepare candidates using the shipping independent oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import replace_once


def source():
    router=(ROOT/'quactlize/integrations/llama/router.cuh').read_text()
    start=router.index('template<bool HasBias>')
    fast=router[start:router.index('\n} // namespace',start)]
    fast=replace_once(fast,'router_256_top8(', 'router_top8_warp(')
    fast=replace_once(fast,'int* shared_ids) {','int* shared_ids,bool publish) {')
    fast=replace_once(fast,'  if (threadIdx.x>=32) return;\n','')
    fast=replace_once(fast,'int lane=int(threadIdx.x);','int lane=int(threadIdx.x)%32;')
    fast=replace_once(fast,'      const_cast<int32_t*>(io.ids)[slot]=expert;',
                     '      if (publish) const_cast<int32_t*>(io.ids)[slot]=expert;')
    fast=replace_once(fast,'  if (lane<8) r.weights[lane]=chosen*r.scale;',
                     '  if (publish && lane<8) r.weights[lane]=chosen*r.scale;')
    generic=(ROOT/'quactlize/runtime/moe_chain.cuh').read_text()
    start=generic.index('template<class Shape,class Stride>\n__global__ void moe_chain_prepare(')
    body=generic[start:generic.index('\nCUTLASS_DEVICE float moe_projection_value',start)]
    body=replace_once(body,'void moe_chain_prepare(', 'void prepare_fast_router(')
    body=replace_once(body,'if (plan.router.version) quactlize::llama::router_256(plan.router,p.io,ids,blockIdx.x==0);', '''if (plan.router.version) {
    int warp=tid/32;
    if (warp<p.io.tokens) {
      auto router=plan.router; auto io=p.io;
      router.logits+=int64_t(warp)*256; router.weights+=warp*8;
      io.ids+=int64_t(warp)*io.ids_stride;
      if (router.bias) quactlize::llama::router_top8_warp<true>(router,io,ids+warp*8,blockIdx.x==0);
      else quactlize::llama::router_top8_warp<false>(router,io,ids+warp*8,blockIdx.x==0);
    }
  }''')
    helper='namespace quactlize::llama {\n'+fast+'\n}\n'
    # With <=4 top-8 tokens, no expert owns more than four rows. When every
    # projection's TileM covers them, gate/up/down share the same directory
    # prefix; only pointers/shapes/header TileM differ.
    ds=generic.index('template<class Shape,class Stride>\nCUTLASS_DEVICE void moe_descriptors(')
    de=generic.index('template<class Shape,class Stride>\n__global__ void moe_chain_prepare(',ds)
    desc=generic[ds:de]
    desc=replace_once(desc,'void moe_descriptors(','void compact_descriptors(')
    desc=replace_once(desc,'int const* expert_starts,int const* expert_counts,bool valid)',
        'int const* expert_starts,int const* expert_counts,bool valid,int first,int total)')
    cut=desc.index('  if (blockIdx.x==0 && tid<32) {')
    desc=desc[:cut]+'''  if (blockIdx.x==0 && tid<32) {
    if (tid<p.m) {
      p.io.row_ids[ranks[tid]]=tid;
      if (valid && starts[tid]==ranks[tid])
        static_cast<moe::BlockEntry*>(p.directory_entries)[first]=
            moe::make_entry(ids[tid],counts[tid],first,starts[tid]);
    }
    if (!tid) {
      p.offsets[p.experts]=valid?p.m:0;
      *static_cast<moe::Header*>(p.directory_header)={total,
          valid?0:int(moe::BuildStatus::InvalidArgument),p.tile_m,p.experts};
    }
  }
}
'''
    compact=replace_once(body,'void prepare_fast_router(','void prepare_compact(')
    marker='  moe_descriptors<Shape,Stride>(plan.gate,'
    prefix='''  int first=0,total=0;
  if (blockIdx.x==0 && tid<32) {
    int id=tid<p.m?ids[tid]:p.experts;
    int head=valid && tid<p.m && starts[tid]==ranks[tid];
    #pragma unroll
    for (int j=0;j<32;++j) {
      int other=__shfl_sync(0xffffffff,id,j);
      int active=__shfl_sync(0xffffffff,head,j);
      first+=other<id?active:0;total+=active;
    }
  }
'''
    compact=replace_once(compact,marker,prefix+marker)
    compact=compact.replace('moe_descriptors<Shape,Stride>','compact_descriptors<Shape,Stride>')
    compact=compact.replace('expert_starts,expert_counts,valid);','expert_starts,expert_counts,valid,first,total);')
    helper+='namespace quactlize::runtime {\n'+desc+compact+'\n}\n'
    helper+='namespace quactlize::runtime {\n'+body+'\n}\n'
    helper+=(ROOT/'dev/h800_smallm/moe_prepare.cuh').read_text()
    helper+=(ROOT/'dev/h800_smallm/moe_vector.cuh').read_text()
    first=generic.index('template<class Shape,class Stride>\n__global__ void moe_chain_prepare_m1(')
    last=generic.index('\n// All participants',first)
    m1=generic[first:last].replace('moe_chain_prepare_m1(','prepare_m1_vector(').replace('moe_m1_gather(','moe_m1_gather_vector(')
    helper+='namespace quactlize::runtime {\n'+m1+'\n}\n'
    test=(ROOT/'tests/kpack_moe_chain_cuda.cu').read_text()
    test=replace_once(test,'#include <algorithm>', helper+'\n#include <algorithm>')
    test=replace_once(test,'static bool multi_token=false;',
        'static bool multi_token=false;\nstatic int prepare_arm=0;\nstatic bool model_splits=false;')
    first=test.index('#ifdef KPACK_MOE_PREPARE_BASELINE')
    end=test.index('#endif',first)+len('#endif')
    test=test[:first]+'''    bool eligible=experts==256 && topk==8 && tokens<=4;
    if ((prepare_arm==5 || prepare_arm==6) && moe_prepare_m1_supported(plan))
      prepare_m1_vector<Shape,Stride><<<1,256,0,cudaStreamPerThread>>>(plan);
    else if ((prepare_arm==4 || prepare_arm==6) && eligible && plan.gate.tile_m>=tokens && plan.down.tile_m>=tokens && (merged || plan.up.tile_m>=tokens))
      prepare_compact<Shape,Stride><<<moe_prepare_blocks(experts,m),256,0,cudaStreamPerThread>>>(plan);
    else if (prepare_arm==3 && eligible)
      prepare_once<Shape,Stride,true><<<1,256,0,cudaStreamPerThread>>>(plan);
    else if (prepare_arm==2 && eligible)
      prepare_once<Shape,Stride><<<1,256,0,cudaStreamPerThread>>>(plan);
    else if (prepare_arm==1 && eligible)
      prepare_fast_router<Shape,Stride><<<moe_prepare_blocks(experts,m),256,0,cudaStreamPerThread>>>(plan);
    else if (!generic_prepare && moe_prepare_m1_supported(plan))
      moe_chain_prepare_m1<Shape,Stride><<<1,256,0,cudaStreamPerThread>>>(plan);
    else moe_chain_prepare<Shape,Stride><<<moe_prepare_blocks(experts,m),256,0,cudaStreamPerThread>>>(plan);
'''+test[end:]
    test=replace_once(test,'      if (std::strcmp(argv[i],"--benchmark")==0)',
        '''      if (std::strcmp(argv[i],"--fast-router")==0) prepare_arm=1;
      else if (std::strcmp(argv[i],"--once")==0) prepare_arm=2;
      else if (std::strcmp(argv[i],"--once-wide")==0) prepare_arm=3;
      else if (std::strcmp(argv[i],"--compact")==0) prepare_arm=4;
      else if (std::strcmp(argv[i],"--model-splits")==0) model_splits=true;
      else if (std::strcmp(argv[i],"--m1-vector")==0) prepare_arm=5;
      else if (std::strcmp(argv[i],"--selected")==0) prepare_arm=6;
      else if (std::strcmp(argv[i],"--benchmark")==0)''')
    # Exercise the modified warp-local router with the same 4096 exact-bit
    # cases, in addition to all seven graph replays per multi-token fixture.
    test=test.replace('quactlize::llama::router_256_top8<true>(router,io,ids);',
                      'if (threadIdx.x<32) quactlize::llama::router_top8_warp<true>(router,io,ids,true);')
    test=test.replace('else quactlize::llama::router_256_top8<false>(router,io,ids);',
                      'else if (threadIdx.x<32) quactlize::llama::router_top8_warp<false>(router,io,ids,true);')
    # Braces preserve the HasBias branch when adding its warp predicate.
    test=replace_once(test,'if (router.bias) if (threadIdx.x<32)',
                      'if (threadIdx.x<32 && router.bias)')
    test=replace_once(test,'run(merged,tokens,8,256,4,2,8,router,k);',
        'run(merged,tokens,8,256,model_splits?2:4,2,model_splits?1:8,router,k);')
    return test


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    generated=out/'moe.cu'; generated.write_text(source())
    includes=[ROOT/'dev/gemv_cuda/compat',ROOT,ROOT/'quactlize/include',ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    command=[str(a.cuda/'bin/nvcc'),'-std=c++17','-arch=sm_90','-O3','-lineinfo','-Xptxas=-v',
        '--expt-relaxed-constexpr','-DCUTLASS_USE_PACKED_TUPLE=1','-DCUTE_USE_PACKED_TUPLE=1',
        '-include',str(ROOT/'dev/gemv_cuda/compat/compiler_bridge.h'),
        *['-I'+str(x) for x in includes],str(generated),'-o',str(out/'moe')]
    with (out/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    record=dict(command=command,source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
                binary_sha256=hashlib.sha256((out/'moe').read_bytes()).hexdigest(),production_changed=False)
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print('H800_MOE_BUILD COMPILED '+str(out),flush=True)


if __name__=='__main__': main()

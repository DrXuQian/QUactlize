from pathlib import Path
import os
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def test_actual_packed_writer_and_typed_s1_query(tmp_path):
    sdk=Path(os.environ.get('PPU_SDK','/root/ppu-sdk/2.1.1'))
    executable=tmp_path/'fastpath'
    result=subprocess.run(['g++','-O2','-std=c++17',f'-I{ROOT}',
        f'-I{ROOT}/quactlize/include',f'-I{ROOT}/third_party/actlize/include',
        f'-I{sdk}/include',f'-I{sdk}/targets/x86_64-linux/include',
        str(ROOT/'tests/bf16_fastpath_host.cpp'),str(ROOT/'quactlize/execution/q4_decode.cpp'),
        '-o',str(executable)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(executable)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'BF16_FASTPATH_HOST PASS' in result.stdout


def test_typed_q4_generation_retains_every_selected_reader():
    from quactlize.execution import q4_decode_codegen
    from dev.gemv_ppu import moe_s1
    sources=q4_decode_codegen.sources()
    for (n,k),recipes in q4_decode_codegen.recipes().items():
        source=sources[f'q4decode_{n}_{k}.cu']
        for r,v,w,p,c in recipes:
            assert f'launch<{r},{v},{w},{p},{c},{n},{k}>(c)' in source
            assert f'launch<{r},{v},{w},{p},{c},{n},{k},1>(c)' in source
    for n,k in moe_s1.SHAPES:
        source=moe_s1.source_v2(n,k)
        for r in moe_s1.inventory():
            args=f'{r.reader},{r.variant},{r.warps},{r.values},{n},{k}'
            assert f'launch<{args}>(*c)' in source
            assert f'launch<{args},1>(d->call)' in source


def test_ap1_bf16_admission_keeps_the_existing_single_plane_domain(tmp_path,monkeypatch):
    from test_kpack_jit import fake_compiler
    from quactlize.decode.compiler import DecodeCompiler
    from tools.kpack_jit import parent_tuple
    sdk=fake_compiler(tmp_path)
    monkeypatch.setenv('PATH',str(sdk/'bin')+os.pathsep+os.environ['PATH'])
    compiler=DecodeCompiler(sdk,tmp_path/'cache',compute_type='bf16')
    for q in (10,12):
        for route in (0,1):
            parent=parent_tuple(f'q{q}',[q,route,8,64,256,8,16,2,1,16,-1])
            source=compiler.source(parent,'')
            assert '#define QKD_USE_BF16_COMPUTE 1' in source
            assert '#define QK_AP 1' in source
    import pytest
    with pytest.raises(ValueError,match='packed-A'):
        compiler.source(parent|dict(qtype=14),'')
    with pytest.raises(ValueError,match='packed-A'):
        compiler.source(parent|dict(tm=16,wm=16),'')


def test_ap1_proposal_retains_the_admitted_provider(tmp_path):
    source=tmp_path/'proposal.cpp';executable=tmp_path/'proposal'
    source.write_text('''#include "quactlize/dispatch/compute.hpp"
#include <cassert>
int main(){
  using namespace quactlize::dispatch;
  qks_request_v1 r{1,sizeof(r),12,0,1,512,2048,1,1,UINT64_C(0x51344b5034540001)};
  std::string name;
  auto donor=compute_proposal(r,{},name);
  donor.symbol="fqk_tc_q12_l1_a0_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1_dn16";
  donor.tm=8;donor.wm=8;donor.ap=1;
  auto result=compute_proposal(r,{&donor,QKS_RECENT},name);
  assert(result.ap==1 && name==donor.symbol);
  r.m=r.max_rows=2;
  assert(compute_proposal(r,{&donor,QKS_RECENT},name).symbol==nullptr);
}''')
    subprocess.run(['g++','-O2','-std=c++17',f'-I{ROOT}',str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True)

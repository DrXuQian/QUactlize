from pathlib import Path
import subprocess
import sys
import types
import math

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_prefill_workspace_is_shape_bound_and_small_m_declines(tmp_path):
    source=tmp_path/'layout.cpp';exe=tmp_path/'layout'
    source.write_text(r'''
#include "quactlize/prefill/layout.hpp"
#include <cassert>
int main() {
 for (int q=10;q<=14;++q) for(int experts:{1,256}) {
   auto arr=q==12?ppu_arrangements::q4_kpack4_transpose_v1():ppu_arrangements::kquant_kpack_transpose_v1(q);
   qkp_call_v1 c{};c.version=1;c.size=sizeof(c);c.m=experts==1?128:1024;
   c.a_stride=512;c.output_stride=256;c.device=0;c.a_rows=c.m;
   c.weight={1,sizeof(qzd_call_v1),q,256,512,experts,1,5};
   quactlize::prefill::Layout l{};
   assert(quactlize::prefill::layout(c,&arr,l)==0);
   assert(l.a==uint64_t(experts)*256*512*2 && l.out>l.a && l.directory<l.bytes);
   assert(l.bytes%256==0);
   for (int m:{1,2,4,8,64}) { c.m=m;assert(quactlize::prefill::layout(c,&arr,l)!=0); }
   c.m=experts==1?4096:32768;c.a_rows=c.m;assert(quactlize::prefill::layout(c,&arr,l)==0);
   c.m*=2;assert(quactlize::prefill::layout(c,&arr,l)!=0);
   c.m=experts==1?128:1024;c.a_rows=c.m;arr.mapping_id=0;assert(quactlize::prefill::layout(c,&arr,l)!=0);
 }
}
''')
    subprocess.run(['g++','-std=c++17',f'-I{ROOT}',f'-I{ROOT/"quactlize/include"}',str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_full_bf16_provider_keeps_measured_abi_and_no_hot_path_compiler():
    source=(ROOT/'quactlize/prefill/runtime.cpp').read_text()
    run=source.split('extern "C" int quactlize_kpack_prefill_run_v1(')[1].split('extern "C" void')[0]
    for forbidden in ('posix_spawn','hggcMalloc','hggcMemcpy','hggcStreamSynchronize','dlopen','prewarm('):
        assert forbidden not in run
    assert 'quactlize_kpack_dequant_indexed_v1' in run
    assert 'base+h.layout.rows' in run and 'base+h.layout.directory' in run
    assert 'h.gemm(h.blas,1,0,w.n,c.m,w.k' in run
    assert 'base+h.layout.out,14,w.n,68,-1' in run
    assert 'cublasSetWorkspace' not in source


def test_prefill_sync_only_updates_documented_public_includes(tmp_path):
    from tools.sync_kpack_llama_headers import sync, FILES, INCLUDES
    target=tmp_path/'ggml/src/ggml-cuda/quactlize';target.mkdir(parents=True)
    (target/'ABI_SHA256').write_text('')
    sync(tmp_path)
    for src,dst in FILES.items():
        text=(ROOT/src).read_text()
        if src=='quactlize/execution/q4_decode.h':text=text.replace('"api.h"','"kpack_execution.h"')
        for before,after in INCLUDES.items():text=text.replace('"'+before+'"','"'+after+'"')
        assert (target/dst).read_text()==text


def test_prefill_ctypes_and_gate_domains(tmp_path):
    import ctypes as C
    from quactlize.prefill.native import Call, Options, Choice
    from tools.run_kpack_prefill_runtime import workloads, routing, require_guards
    source = tmp_path/'abi.cpp'
    source.write_text(r'''
#include "quactlize/prefill/api.h"
#include "quactlize/dispatch/api.h"
#include <cstddef>
#include <cstdio>
int main() {
 printf("%zu %zu %zu %zu %zu %zu\n",sizeof(qkp_call_v1),sizeof(qkp_options_v1),
  sizeof(qks_prefill_choice_v1),offsetof(qkp_call_v1,a),offsetof(qkp_call_v1,offsets),offsetof(qkp_call_v1,workspace_bytes));
}
''')
    exe=tmp_path/'abi'
    subprocess.run(['g++','-std=c++17',f'-I{ROOT}',f'-I{ROOT/"quactlize/include"}',str(source),'-o',str(exe)],check=True)
    got=list(map(int,subprocess.check_output([str(exe)],text=True).split()))
    assert got == [C.sizeof(Call),C.sizeof(Options),C.sizeof(Choice),Call.a.offset,Call.offsets.offset,Call.workspace_bytes.offset]
    cases=workloads()
    assert len(cases)==7 and {w['q'] for w in cases if w['experts']==1} == set(range(10,15))
    assert {(w['n'],w['k']) for w in cases if w['experts']>1} == {(2048,512),(3072,512)}
    import numpy as np
    for channels in (1,8):
        for repeat in range(3):
            src,dst,offsets,ids,arows=routing(1024,256,channels,repeat)
            assert sorted(dst)==list(range(1024)) and np.array_equal(src,arows[dst])
            assert np.all(src>=0) and np.all(src<128*channels)
            assert offsets[0]==0 and offsets[-1]==1024 and np.all(np.diff(offsets)>=0)
            assert np.array_equal(np.repeat(np.arange(256),np.diff(offsets)), ids[dst])
    with pytest.raises(ValueError):require_guards(b'bad',0xA5,2)


@pytest.mark.parametrize('grouped', [False, True])
def test_actual_native_provider_composition_with_cpu_abi_stubs(tmp_path, grouped):
    sdk=Path('/root/ppu-sdk/2.1.1')
    provider=tmp_path/'CUDA_SDK/targets/x86_64-linux/lib'
    provider.mkdir(parents=True)
    stub=tmp_path/'blas.cpp'
    stub.write_text(r'''
#include <cassert>
#include <cstdint>
extern "C" int cublasCreate_v2(void** p) { *p=new int(1);return 0; }
extern "C" int cublasDestroy_v2(void* p) { delete static_cast<int*>(p);return 0; }
extern "C" int cublasSetStream_v2(void*,void* s) {return s!=nullptr;}
extern "C" int cublasSetMathMode(void*,int mode) {return mode!=16;}
extern "C" int cublasGemmEx(void*,int ta,int tb,int n,int m,int k,void const* alpha,
 void const* b,int bt,int ldb,void const* a,int at,int lda,void const* beta,void* out,int ct,int ldc,int compute,int algorithm) {
 assert(ta==1 && tb==0 && n==256 && m==128 && k==512);
 assert(bt==14 && at==14 && ct==14 && lda==512 && ldb==512 && ldc==256 && compute==68 && algorithm==-1);
 assert(*static_cast<float const*>(alpha)==1 && *static_cast<float const*>(beta)==0);
 assert(*static_cast<uint16_t const*>(a)==0x4000 && *static_cast<uint16_t const*>(b)==0x3f80);
 *static_cast<uint16_t*>(out)=0x4000;return 0;
}
extern "C" void launch(void const* a,void const* b,void* out,int const* rows,void* directory,
 int m,int expected_m,void* stream,int sms,int smem,void* signal,int* rc) {
 assert(m==1024 && expected_m==4 && sms==72 && smem==8192 && !stream && !signal && directory && rows);
 assert(*static_cast<uint16_t const*>(a)==0x4000 && *static_cast<uint16_t const*>(b)==0x3f80);
 *static_cast<uint16_t*>(out)=0x4000;*rc=0;
}
''')
    subprocess.run(['g++','-shared','-fPIC',str(stub),'-o',str(provider/'libcublas.so')],check=True)
    test=tmp_path/'native.cpp'
    test.write_text(r'''
#include <hggc_runtime.h>
#include "quactlize/prefill/layout.hpp"
#include "quactlize/dequant/indexed.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
using quactlize::prefill::Layout;
int expanded=0,inputs=0,outputs=0;
extern "C" hggcError_t hggcGetDevice(int* d) { *d=0;return hggcSuccess; }
extern "C" hggcError_t hggcGetDeviceProperties(hggcDeviceProp* p,int) {
 std::strcpy(p->name,"PPU-ZW810");p->multiProcessorCount=1;return hggcSuccess;
}
extern "C" hggcError_t hggcDeviceGetAttribute(int* n,hggcDeviceAttr attr,int) {
 assert(attr==hggcDevAttrMultiProcessorCount);*n=72;return hggcSuccess;
}
extern "C" hggcError_t hggcStreamIsCapturing(hggcStream_t,hggcStreamCaptureStatus* s) { *s=hggcStreamCaptureStatusNone;return hggcSuccess; }
extern "C" hggcError_t hggcGetLastError() {return hggcSuccess;}
extern "C" int qkp_stage_prepare(qkp_call_v1 const* c,Layout const* l,void*) {
 ++inputs;*reinterpret_cast<uint16_t*>(static_cast<char*>(c->workspace)+l->a)=0x4000;return 0;
}
extern "C" int qkp_stage_finish(qkp_call_v1 const* c,Layout const* l,void*) {
 ++outputs;assert(*reinterpret_cast<uint16_t*>(static_cast<char*>(c->workspace)+l->out)==0x4000);c->output[0]=2;return 0;
}
extern "C" int quactlize_kpack_dequant_v1(qzd_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const*) {
 ++expanded;assert(c->operation==1 && c->zero==nullptr && c->output_bytes==uint64_t(c->experts)*256*512*2);
 *static_cast<uint16_t*>(c->output)=0x3f80;return 0;
}
extern "C" int quactlize_kpack_dequant_indexed_v1(qzd_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const* arr,int const* ids,int const* count,int* status) {
 assert(c->experts==256 && ids && count && status);return quactlize_kpack_dequant_v1(c,arr);
}
int main(int argc,char** argv) {
 assert(argc==2 || argc==4);auto arr=ppu_arrangements::q4_kpack4_transpose_v1();
 bool grouped=argc==4;
 qkp_call_v1 c{};c.version=1;c.size=sizeof(c);c.m=128;c.a_rows=128;c.a_stride=512;c.output_stride=256;c.device=0;
 c.weight={1,sizeof(qzd_call_v1),12,256,512,1,1,5};
 if (grouped) {c.m=c.a_rows=1024;c.weight.experts=256;}
 c.weight.low_bytes=uint64_t(c.weight.experts)*256*512/2;c.weight.unit_bytes=uint64_t(c.weight.experts)*256*512/256*16;
 c.weight.low=std::aligned_alloc(256,c.weight.low_bytes);c.weight.units=std::aligned_alloc(256,c.weight.unit_bytes);
 c.a=static_cast<float*>(std::aligned_alloc(256,c.m*512*4));c.output=static_cast<float*>(std::aligned_alloc(256,c.m*256*4));
 if (grouped) {
   c.src_rows=static_cast<int*>(std::aligned_alloc(256,4096));
   c.dst_rows=static_cast<int*>(std::aligned_alloc(256,4096));
   c.offsets=static_cast<int*>(std::aligned_alloc(256,1280));
 }
 assert(quactlize_kpack_prefill_query_v1(&c,&arr,&c.workspace_bytes)==0);
 c.workspace=std::aligned_alloc(256,c.workspace_bytes);
 qkp_options_v1 o{1,sizeof(o),argv[1],grouped?argv[2]:nullptr,grouped?argv[3]:nullptr};void* h=nullptr;
 assert(quactlize_kpack_prefill_prepare_v1(&c,&arr,&o,&h)==0 && h);
 int32_t const* device_status=nullptr;
 assert(quactlize_kpack_prefill_device_status_v1(h,&device_status)==0 && device_status);
 assert(expanded==0 && inputs==0);
 for(int i=0;i<3;++i) { assert(quactlize_kpack_prefill_run_v1(h,nullptr)==0);assert(c.output[0]==2); }
 assert(expanded==3 && inputs==3 && outputs==3);
 assert(quactlize_kpack_prefill_run_v1(h,reinterpret_cast<void*>(1))!=0);
 quactlize_kpack_prefill_destroy_v1(h);
 auto original=c.output;c.output=static_cast<float*>(c.workspace);
 assert(quactlize_kpack_prefill_prepare_v1(&c,&arr,&o,&h)!=0 && !h);
 c.output=original;auto input=c.a;c.a=static_cast<float*>(c.workspace);
 assert(quactlize_kpack_prefill_prepare_v1(&c,&arr,&o,&h)!=0 && !h);c.a=input;
 std::free(original);std::free(const_cast<float*>(c.a));std::free(c.workspace);
 std::free(const_cast<void*>(c.weight.low));std::free(const_cast<void*>(c.weight.units));
 std::free(const_cast<int*>(c.src_rows));std::free(const_cast<int*>(c.dst_rows));std::free(const_cast<int*>(c.offsets));
}
''')
    exe=tmp_path/'native'
    subprocess.run(['g++','-std=c++17','-O1',f'-I{ROOT}',f'-I{ROOT/"quactlize/include"}',
        f'-I{sdk/"include"}',str(test),str(ROOT/'quactlize/prefill/runtime.cpp'),'-ldl','-pthread','-o',str(exe)],check=True)
    args = [str(exe),str(tmp_path)]
    if grouped:
        helper = tmp_path/'provider helper.py'
        helper.write_text('print("QKP_DEEPGEMM_V1 1024 256 512 256 72 8192 16384")\n' +
            f'print({str(provider/"libcublas.so")!r})\nprint({"a"*64!r})\n')
        args += [sys.executable,str(helper)]
    subprocess.run(args,check=True)


@pytest.mark.parametrize('fault', [None, 'abi', 'tuning', 'source', 'args', 'scalars', 'omitted'])
def test_installed_provider_compile_only_handoff(tmp_path, monkeypatch, fault):
    from tools.kpack_deepgemm_prewarm import resolve

    class Tensor:
        def __init__(self, shape, dtype, device):
            self.shape = (shape,) if isinstance(shape, int) else shape
            self.dtype, self.device = dtype, device

        def numel(self):
            return math.prod(self.shape)

    class Stream:
        pass

    torch = types.ModuleType('torch')
    torch.bfloat16, torch.int32 = 'bf16', 'i32'
    torch.empty = lambda shape, dtype, device: Tensor(shape, dtype, device)
    torch.cuda = types.SimpleNamespace(Stream=Stream, set_device=lambda _: None,
        get_device_properties=lambda _: types.SimpleNamespace(multi_processor_count=72, name='PPU-ZW810'))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    module = types.ModuleType('deep_gemm.jit_kernels.m_grouped_gemm')
    module.__file__ = str(tmp_path/'provider.py')
    Path(module.__file__).write_text('# test provider ABI\n')
    runtime = types.ModuleType('deep_gemm.jit.runtime')
    state = {'mode': 0, 'calls': 0}
    runtime.get_compile_mode = lambda: state['mode']
    runtime.set_compile_mode = lambda mode: state.update(mode=mode)
    runtime.CompileMode = types.SimpleNamespace(ONLY_COMPILE=types.SimpleNamespace(value=1))
    template = types.ModuleType('deep_gemm.jit.template')
    template.typename_map = {'bf16': 'torch.bfloat16', 'i32': 'torch.int32', int: 'int', Stream: 'torch.cuda.Stream'}
    template.cpp_format = lambda source, keys: source
    template.generate = lambda includes, defs, source: source

    class Runtime:
        path = str(tmp_path)

        def __call__(self, *args):
            assert state['mode'] == 1, 'an eager GPU launch escaped prewarm'
            state['calls'] += 1

    def compile_and_tune(**kw):
        defs = kw['arg_defs']
        declared = ', '.join(f"('{name}', {template.typename_map[dtype]})" for name, dtype in defs)
        (tmp_path/'kernel.args').write_text('bad' if fault=='args' else declared)
        (tmp_path/'kernel.cu').write_text('bad' if fault=='source' else kw['template'])
        (tmp_path/'kernel.so').write_bytes(b'fake-image-not-device-evidence')
        return Runtime()

    module.jit_tuner = types.SimpleNamespace(compile_and_tune=compile_and_tune)

    def entry(lhs, rhs, out, indices, m_rows):
        if fault == 'omitted':
            return
        defs = (('lhs',torch.bfloat16),('rhs',torch.bfloat16),('out',torch.bfloat16),
            ('grouped_layout',torch.int32),('block_m_info',torch.int32),('m',int),('expected_m',int),
            ('stream',Stream),('num_sms',int),('smem_size',int),('signal',torch.int32))
        if fault == 'abi':
            defs = defs[:-1]
        args = (lhs,rhs,out,m_rows,Tensor(4096,'i32','meta'),
            1 if fault=='scalars' else 1024,4,Stream(),72,8192,Tensor(0,'i32','meta'))
        image = module.jit_tuner.compile_and_tune(name='m_grouped_gemm_bf16_bf16_bf16_nt',
            space=(1,) if fault=='tuning' else (), includes=[], arg_defs=defs,
            template='provider-selected-source', keys={'N':256,'K':512}, args=args)
        image(*args)

    entry.__module__ = module.__name__
    module.m_grouped_gemm_bf16_bf16_bf16_nt_nopad = entry
    for item in (module, runtime, template):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    args = types.SimpleNamespace(m=1024,n=256,k=512,experts=256,device=0)
    if fault:
        with pytest.raises(ValueError):
            resolve(args)
        assert not (tmp_path/'quactlize-launch-m1024.json').exists()
    else:
        reply = resolve(args)
        assert reply.startswith('QKP_DEEPGEMM_V1 1024 256 512 256 72 8192 16384\n')
        assert state['calls'] == 1
        assert (tmp_path/'quactlize-launch-m1024.json').is_file()
    assert state['mode'] == 0
    assert module.jit_tuner.compile_and_tune is compile_and_tune

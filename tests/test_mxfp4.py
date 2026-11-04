"""GGUF MXFP4 bit-preserving layout and public contracts; no GPU claims."""
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import types
import math

import numpy as np
import pytest

from reference import mxfp4 as ref
from ci.support.packing import Arrangement, Sizes

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def packer(tmp_path_factory):
    root=tmp_path_factory.mktemp('mxfp4-host')
    source=root/'pack.cpp'
    source.write_text(r'''
#include "quactlize/packing/mxfp4.hpp"
#include "quactlize/packing/word_pack.hpp"
#include "quactlize/mxfp4/e8m0.hpp"
#include "quactlize/moe/matmul_validation.hpp"
extern "C" int admitted(quactlize_ppu_moe_matmul_params_v1 const* p) {
    return quactlize::moe::matmul::admission(p);
}
extern "C" unsigned short sf_bits(unsigned char e) { return quactlize::mxfp4::scale_bits(e); }
extern "C" void pack(unsigned char const* raw,unsigned* low,unsigned char* scales,int e,int n,int k) {
    for(uint64_t i=0;i<uint64_t(e)*n*k/8;++i) low[i]=quactlize::mxfp4::Map::word(raw,n,k,i);
    for(uint64_t i=0;i<uint64_t(e)*n*k/32;++i) scales[i]=quactlize::mxfp4::Map::scale(raw,n,k,i);
}
extern "C" void pair(unsigned char const* gate,unsigned char const* up,unsigned* low,unsigned char* scales,int e,int n,int k) {
    quactlize_pack::PairedRows raw{gate,up,uint64_t(n)*(k/32)*17};
    for(uint64_t i=0;i<uint64_t(e)*2*n*k/8;++i) low[i]=quactlize::mxfp4::Map::word(raw,2*n,k,i);
    for(uint64_t i=0;i<uint64_t(e)*2*n*k/32;++i) scales[i]=quactlize::mxfp4::Map::scale(raw,2*n,k,i);
}
''')
    library=root/'pack.so'
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-I'+str(ROOT),
        '-I'+str(ROOT/'quactlize/include'),'-I'+str(ROOT/'third_party/actlize/include'),
        '-I'+str(Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))/'include'),
        str(source),str(ROOT/'quactlize/packing/sizes.cpp'),'-o',str(library)],check=True)
    lib=C.CDLL(str(library))
    lib.pack.argtypes=[C.c_void_p]*3+[C.c_int]*3
    lib.pair.argtypes=[C.c_void_p]*4+[C.c_int]*3
    lib.quactlize_ppu_kpack_canonical_arrangement_v1.argtypes=[C.c_int,C.POINTER(Arrangement)]
    lib.quactlize_ppu_kpack_sizes_for_arrangement_v1.argtypes=[C.c_int]*4+[C.POINTER(Arrangement),C.POINTER(Sizes)]
    lib.sf_bits.argtypes=[C.c_ubyte];lib.sf_bits.restype=C.c_ushort
    return lib


@pytest.mark.parametrize('e,n,k',[(1,64,32),(3,128,96),(4,256,512),(2,512,4096)])
def test_exact_device_word_ownership(packer,e,n,k):
    raw=np.random.default_rng(e+n+k).integers(0,256,(e,n,k//32,17),dtype='u1')
    weights,scales=ref.pack(raw)
    low=np.empty_like(weights);sf=np.empty_like(scales)
    packer.pack(raw.ctypes.data,low.ctypes.data,sf.ctypes.data,e,n,k)
    np.testing.assert_array_equal(low,weights)
    np.testing.assert_array_equal(sf,scales)
    np.testing.assert_array_equal(ref.recover(low,sf),raw)
    # The inverse catches a wrong nibble or metadata permutation, not just shape.
    low.flat[0]^=1
    assert not np.array_equal(ref.recover(low,sf),raw)
    low.flat[0]^=1;sf.flat[-1]^=1
    assert not np.array_equal(ref.recover(low,sf),raw)


def test_all_codes_scales_and_tp_ep_slices(packer):
    e,n,k=4,256,512
    raw=np.zeros((e,n,k//32,17),'u1')
    raw[...,0]=np.arange(n,dtype='u1')[None,:,None]
    raw[...,1:]=np.arange(16,dtype='u1')*17
    for part in [raw,raw[1:3].copy(),raw[:,:128].copy(),raw[:,:,4:12].copy()]:
        e,n,kb,_=part.shape
        lo,sf=ref.pack(part);a,b=np.empty_like(lo),np.empty_like(sf)
        packer.pack(part.ctypes.data,a.ctypes.data,b.ctypes.data,e,n,kb*32)
        np.testing.assert_array_equal(ref.recover(a,b),part)


def test_paired_gate_up_is_canonical_concat_not_activation_interleave(packer):
    raw=np.random.default_rng(39).integers(0,256,(2,2,128,16,17),dtype='u1')
    gate,up=raw
    want=np.concatenate((gate,up),axis=1)
    lo,sf=ref.pack(want);a,b=np.empty_like(lo),np.empty_like(sf)
    packer.pair(gate.ctypes.data,up.ctypes.data,a.ctypes.data,b.ctypes.data,2,128,512)
    np.testing.assert_array_equal(a,lo);np.testing.assert_array_equal(b,sf)


def test_size_identity_and_old_format_rejection(packer):
    arr=Arrangement();sizes=Sizes()
    assert packer.quactlize_ppu_kpack_canonical_arrangement_v1(39,C.byref(arr))==0
    assert arr.mapping_id==ref.MAPPING_ID and arr.group_size==32 and arr.bits==4
    query=packer.quactlize_ppu_kpack_sizes_for_arrangement_v1
    assert query(256,512,4,39,C.byref(arr),C.byref(sizes))==0
    assert sizes.raw_bytes==sizes.low_bytes+sizes.units_bytes==4*256*(512//32)*17
    assert sizes.high_bytes==0
    for n,k,e in [(63,512,4),(64,31,4),(64,32,0)]:
        assert query(n,k,e,39,C.byref(arr),C.byref(sizes))!=0
    assert query(2147483584,2147483616,2147483647,39,C.byref(arr),C.byref(sizes))==26
    old=Arrangement();assert packer.quactlize_ppu_kpack_canonical_arrangement_v1(12,C.byref(old))==0
    assert query(256,512,1,39,C.byref(old),C.byref(sizes))==38
    assert query(256,512,1,12,C.byref(arr),C.byref(sizes))==38


def test_e8m0_decoder_including_zero_exponent(packer):
    bits=np.array([packer.sf_bits(e) for e in range(256)],dtype='u4')<<16
    got=bits.view('f4').astype('f8')
    np.testing.assert_array_equal(got[:255],np.exp2(np.arange(255)-127.))
    assert np.isnan(got[255])
    assert got[0]!=0  # Sail's uncorrected byte<<7 conversion fails this.


@pytest.mark.parametrize('channels',[1,6])
def test_moe_admission_all_m_topk6(packer,channels):
    from ci.support.moe import Params, Arrangement as MoeArrangement, Sizes as MoeSizes
    packer.admitted.argtypes=[C.POINTER(Params)]
    arr=Arrangement();sizes=Sizes()
    assert packer.quactlize_ppu_kpack_canonical_arrangement_v1(39,C.byref(arr))==0
    assert packer.quactlize_ppu_kpack_sizes_for_arrangement_v1(512,1024,256,39,C.byref(arr),C.byref(sizes))==0
    for m in (0,1,2,3,4,5,6,7,8,9,31,512,1024,2048,4096,5120):
        p=Params(1,C.sizeof(Params),39,512,1024,256,m,6,channels,0,
                 1024,1024*channels,6,512,MoeArrangement.from_buffer_copy(arr),MoeSizes.from_buffer_copy(sizes))
        assert packer.admitted(C.byref(p))==1
        p.arrangement.mapping_id^=1
        assert packer.admitted(C.byref(p))==0


@pytest.mark.parametrize('fault',[None,'abi','tuning','layout','args','source','scalars','cu','missing'])
def test_compile_only_provider_handoff(tmp_path,monkeypatch,fault):
    from tools.kpack_mxfp4_prewarm import resolve
    class Tensor:
        def __init__(self,shape,dtype,device):
            assert device=='meta'
            self.shape=(shape,) if isinstance(shape,int) else shape
            self.dtype=dtype
        def numel(self):return math.prod(self.shape)
    class Stream:pass
    torch=types.ModuleType('torch')
    torch.bfloat16,torch.int32,torch.uint8='bf16','i32','u8'
    torch.empty=lambda shape,dtype,device:Tensor(shape,dtype,device)
    torch.cuda=types.SimpleNamespace(Stream=Stream,set_device=lambda _:None,
        get_device_properties=lambda _:types.SimpleNamespace(name='PPU-ZW810E',multi_processor_count=64,major=8,minor=0))
    monkeypatch.setitem(sys.modules,'torch',torch)
    module=types.ModuleType('deep_gemm.jit_kernels.m_grouped_gemm_w4a16')
    module.__file__=str(tmp_path/'provider.py');Path(module.__file__).write_text('# CPU ABI double\n')
    runtime=types.ModuleType('deep_gemm.jit.runtime')
    state={'mode':0,'calls':0,'sms':20}
    runtime.get_compile_mode=lambda:state['mode']
    runtime.set_compile_mode=lambda mode:state.update(mode=mode)
    runtime.CompileMode=types.SimpleNamespace(ONLY_COMPILE=types.SimpleNamespace(value=1))
    utils=types.ModuleType('deep_gemm.jit_kernels.utils')
    utils.set_num_sms=lambda sms:state.update(sms=sms)
    template=types.ModuleType('deep_gemm.jit.template')
    template.typename_map={'bf16':'torch.bfloat16','i32':'torch.int32','u8':'torch.uint8',int:'int',Stream:'torch.cuda.Stream'}
    template.cpp_format=lambda text,keys:text
    template.generate=lambda includes,defs,text:text
    class Runtime:
        path=str(tmp_path)
        def __call__(self,*args):
            assert state['mode']==1,'GPU launch escaped prepare'
            state['calls']+=1
    def compile_and_tune(**kw):
        assert 'Quactlize E8M0 endpoints' in kw['template']
        assert kw['includes'][0].endswith('mxfp4/deepgemm.cuh"')
        declared=', '.join(f"('{name}', {template.typename_map[dtype]})" for name,dtype in kw['arg_defs'])
        (tmp_path/'kernel.args').write_text('bad' if fault=='args' else declared)
        (tmp_path/'kernel.cu').write_text('bad' if fault=='source' else kw['template'])
        (tmp_path/'kernel.so').write_bytes(b'CPU ABI double, not PPU code')
        return Runtime()
    module.jit_tuner=types.SimpleNamespace(compile_and_tune=compile_and_tune)
    def entry(lhs,rhs,out,indices,m_rows):
        if fault=='missing':return
        assert rhs[0].shape==(256,32,512) and rhs[1].shape==(256,16,256)
        defs=(('lhs','bf16'),('rhs','i32'),('rhs_scales','u8'),('out','bf16'),('m',int),
              ('expected_m',int),('stream',Stream),('num_sms',int),('m_rows','i32'),('block_m_info','i32'))
        if fault=='abi':defs=defs[:-1]
        keys=dict(N=256,K=512,NUM_GROUPS=256,GROUP_SIZE=32,GEMM_TYPE='GroupedNoPad',
            ELEMENT_B='int4_t' if fault=='layout' else 'cutlass::float4_t',ELEMENT_SCALE='uint8_t',
            BLOCK_M=16,BLOCK_N=256,BLOCK_K=64,WARP_M=16,WARP_N=64,WARP_K=64,NUM_STAGES=2)
        actual=(lhs,*rhs,out,1 if fault=='scalars' else 102,1,Stream(),
                state['sms']+int(fault=='cu'),m_rows,Tensor(1024,'i32','meta'))
        image=module.jit_tuner.compile_and_tune(name='m_grouped_gemm_w4a16',keys=keys,
            space=(1,) if fault=='tuning' else (),includes=[],arg_defs=defs,template='provider-source',
            args=actual,jit_include_dir='actlize_v1.0.0')
        image(*actual)
    entry.__module__=module.__name__;module.m_grouped_gemm_w4a16_nopad=entry
    for mod in (module,runtime,template,utils):monkeypatch.setitem(sys.modules,mod.__name__,mod)
    args=types.SimpleNamespace(m=102,n=256,k=512,experts=256,device=0)
    if fault:
        with pytest.raises(ValueError):resolve(args)
        assert not (tmp_path/'quactlize-mxfp4-m102.json').exists()
    else:
        assert resolve(args).startswith('QZ_MXFP4_V1 102 256 512 256 64 4096\n')
        assert state['calls']==1 and (tmp_path/'quactlize-mxfp4-m102.json').is_file()
    assert state['mode']==0 and module.jit_tuner.compile_and_tune is compile_and_tune

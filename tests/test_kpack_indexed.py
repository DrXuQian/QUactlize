import ctypes as C
from pathlib import Path
import subprocess

import numpy as np
import pytest
from quactlize.dispatch.native import IndexedIO

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    output=tmp_path_factory.mktemp("indexed")/"host.so"
    build=subprocess.run(["g++","-std=c++17","-O2","-shared","-fPIC",f"-I{ROOT}",
        f"-I{ROOT/'third_party/actlize/include'}","-I/root/ppu-sdk/2.1.1/include",
        "-I/root/ppu-sdk/2.1.1/targets/x86_64-linux/include",
        "-DCUTLASS_USE_PACKED_TUPLE=1","-DCUTE_USE_PACKED_TUPLE=1",
        str(ROOT/"tests/kpack_indexed_host.cpp"),"-o",str(output)],text=True,capture_output=True)
    assert build.returncode==0,build.stdout+build.stderr
    lib=C.CDLL(str(output))
    lib.row_map.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,*([C.c_void_p]*4)]
    lib.row_map.restype=C.c_int
    return lib


@pytest.mark.parametrize("tm",[8,16,32,64,128,256])
@pytest.mark.parametrize("tokens,topk,experts",[(1,8,256),(4,8,256),(16,2,8),(9,1,2),(1,1,1024)])
def test_exact_route_and_directory(library,tm,tokens,topk,experts):
    rng=np.random.default_rng(132)
    # Rotate/permutate on every replay; ranks cannot be cached by pointer.
    for repeat in range(4):
        ids=np.stack([rng.choice(experts,topk,replace=False) for _ in range(tokens)]).astype('i4').reshape(-1)
        m=ids.size
        rank=np.full(m,-1,dtype='i4')
        offsets=np.full(experts+1,-1,dtype='i4')
        shapes=np.full((experts,3),-1,dtype='i4')
        entries=np.full((m+2,4),-123,dtype='i4')
        assert library.row_map(ids.ctypes.data,m,topk,experts,tm,rank.ctypes.data,
            offsets.ctypes.data,shapes.ctypes.data,entries[1:].ctypes.data)==0
        ordered=np.argsort(ids,kind='stable')
        np.testing.assert_array_equal(rank[ordered],np.arange(m))
        counts=np.bincount(ids,minlength=experts)
        np.testing.assert_array_equal(offsets,np.r_[0,counts.cumsum()])
        np.testing.assert_array_equal(shapes[:,0],counts)
        assert np.all(shapes[:,1]==512) and np.all(shapes[:,2]==2048)
        golden=[]
        for e,count in enumerate(counts):
            begin=len(golden)
            golden.extend([(e,count,begin,offsets[e])]*((int(count)+tm-1)//tm))
        np.testing.assert_array_equal(entries[1:1+len(golden)],golden)
        assert np.all(entries[0]==-123) and np.all(entries[1+len(golden):]==-123)
        if topk>1:
            # Omitting the slot axis is a real wrong-permutation negative.
            assert np.any(ordered%topk!=0)


@pytest.mark.parametrize("ids",[[0,-1],[0,8],[3,3]])
def test_invalid_router_rejected(library,ids):
    ids=np.asarray(ids,dtype='i4')
    output=np.full(80,-123,dtype='i4')
    assert library.row_map(ids.ctypes.data,2,2,8,8,*([output.ctypes.data]*4))==1
    assert np.all(output==-123)


def test_fused_dispatch_is_additive_and_large_cases_decline():
    source=(ROOT/'quactlize/runtime/module.cuh').read_text()
    assert 'call.m>32 || call.experts>1024' in source
    assert 'hggcStreamIsCapturing' in source
    assert 'indexed_prepare<tm>' in source and 'indexed_finish<S>' in source
    # Existing runs remain the implementation for unsupported/older modules.
    assert 'grouped_splitk_device_metadata<<<' in source
    assert 'reduction.run(stream)' in source
    binding=(ROOT/'quactlize/dispatch/binding.cpp').read_text()
    assert 'if (!h.module->bind_indexed) return QKS_MISS;' in binding


def test_indexed_c_abi_matches_python(library):
    assert library.indexed_io_size()==C.sizeof(IndexedIO)


@pytest.mark.parametrize('experts',[1,2,31,32,33,256,1024])
@pytest.mark.parametrize('rows',[1,8,9,17,31,32])
def test_prepare_ctas_own_every_expert_and_gather_word_once(library,experts,rows):
    blocks=library.prepare_blocks(experts,rows)
    assert blocks>=rows and blocks*32>=experts
    experts_seen=[e for b in range(blocks) for e in range(b*32,min((b+1)*32,experts))]
    assert experts_seen==list(range(experts))
    for k in (256,2048,5120,16384):
        seen=np.zeros((rows,(k+255)//256),dtype='i4')
        for block in range(blocks):
            row,chunk=block%rows,block//rows
            chunks=(blocks-1-row)//rows+1
            seen[row,chunk::chunks]+=1
        assert np.all(seen==1)

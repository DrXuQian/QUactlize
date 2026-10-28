"""Real SDK ABI with host doubles; no PPU or GPU-performance claim."""
import ctypes as C
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from dev.native_components.kernel_nodes import KernelNodes
from dev.strategy.isolated import Captured

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def adapter(tmp_path_factory):
    sdk=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))
    if not (sdk/'CUDA_SDK/include/cuda.h').is_file():
        pytest.skip('real SDK headers required for the native node ABI test')
    folder=tmp_path_factory.mktemp('kernel-node-abi')
    source=folder/'mock.cpp'
    source.write_text(r'''
#include <string>
#include <stdexcept>
namespace nc {
thread_local std::string error;
template<class F> int guard(F f) {
 try {f();error.clear();return 0;} catch(const std::exception & e){error=e.what();return 1;}
}
}
#include "kernel_bridge.hpp"
#include <cassert>
static int mode=0,driver_queries=0,driver_adds=0,runtime_adds=0;
static void * args[1]{};
extern "C" const char * nc_error(){return nc::error.c_str();}
extern "C" void set_mock_mode(int m){mode=m;}
extern "C" int count_driver_queries(){return driver_queries;}
extern "C" int count_driver_adds(){return driver_adds;}
extern "C" int count_runtime_adds(){return runtime_adds;}
extern "C" cudaError_t CUDARTAPI cudaGraphKernelNodeGetParams(cudaGraphNode_t,cudaKernelNodeParams * p){
 p->func=reinterpret_cast<void *>(123);p->gridDim={3,1,1};p->blockDim={64,1,1};p->sharedMemBytes=16;p->kernelParams=args;
 return mode==0?cudaSuccess:mode==2?cudaErrorInvalidValue:cudaErrorSymbolNotFound;
}
extern "C" CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode,CUDA_KERNEL_NODE_PARAMS * p){
 ++driver_queries;p->func=reinterpret_cast<CUfunction>(456);
 p->gridDimX=7;p->gridDimY=p->gridDimZ=1;p->blockDimX=128;p->blockDimY=p->blockDimZ=1;
 p->sharedMemBytes=32;p->kernelParams=args;return mode==3?CUDA_ERROR_INVALID_VALUE:CUDA_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaGetLastError(){return mode==4?cudaErrorInvalidValue:cudaErrorSymbolNotFound;}
extern "C" const char * CUDARTAPI cudaGetErrorString(cudaError_t){return "host-mock-runtime-error";}
extern "C" CUresult CUDAAPI cuGetErrorName(CUresult,const char ** s){*s="host-mock-driver-error";return CUDA_SUCCESS;}
extern "C" cudaError_t CUDARTAPI cudaFuncGetName(const char ** s,const void *){*s="runtime_kernel";return cudaSuccess;}
extern "C" CUresult CUDAAPI cuFuncGetName(const char ** s,CUfunction){*s="driver_kernel";return CUDA_SUCCESS;}
extern "C" cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t * node,cudaGraph_t,
 const cudaGraphNode_t *,size_t,const cudaKernelNodeParams * p){
 ++runtime_adds;assert(p->func==reinterpret_cast<void *>(123)&&p->kernelParams==args);
 *node=reinterpret_cast<cudaGraphNode_t>(1234);return cudaSuccess;
}
extern "C" CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode * node,CUgraph,
 const CUgraphNode *,size_t,const CUDA_KERNEL_NODE_PARAMS * p){
 ++driver_adds;assert(p->func==reinterpret_cast<CUfunction>(456)&&p->kernelParams==args);
 *node=reinterpret_cast<CUgraphNode>(5678);return CUDA_SUCCESS;
}
''')
    binary=folder/'mock.so'
    subprocess.run(['g++','-std=c++17','-shared','-fPIC','-I'+str(ROOT/'dev/native_components'),
        '-I'+str(sdk/'CUDA_SDK/include'),str(source),'-o',str(binary)],check=True,capture_output=True)
    lib=C.CDLL(str(binary));lib.set_mock_mode.argtypes=[C.c_int];lib.set_mock_mode.restype=None
    return lib,KernelNodes(lib)


@pytest.mark.parametrize('mode',[0,1])
def test_runtime_and_driver_params_stay_on_their_own_api(adapter,mode):
    lib,nodes=adapter;lib.set_mock_mode(mode)
    handle,desc=nodes.capture(7)
    try:
        assert desc['parameter_api']==('DRIVER_CUFUNCTION' if mode else 'RUNTIME_HOST_STUB')
        assert desc['grid']==([7,1,1] if mode else [3,1,1])
        assert desc['block']==([128,1,1] if mode else [64,1,1])
        assert desc['shared_bytes']==(32 if mode else 16)
        out=C.c_void_p();nodes.add(handle,C.byref(out),9,None)
        assert out.value==(5678 if mode else 1234)
    finally:nodes.destroy(handle)


@pytest.mark.parametrize('mode',[2,3,4])
def test_unrelated_runtime_pending_and_driver_errors_never_admit(adapter,mode):
    lib,nodes=adapter;lib.set_mock_mode(mode);before=lib.count_driver_queries()
    with pytest.raises(RuntimeError,match='kernel-node bridge'):
        nodes.capture(7)
    assert lib.count_driver_queries()==before+int(mode==3)


def test_captured_opaque_nodes_are_retained_through_replay_and_destroyed_first(monkeypatch):
    events=[]
    class Fn:
        def __init__(self,name):self.name=name
        def __call__(self,*args):
            n=self.name
            if n=='GraphGetNodes':
                if args[1] is None:args[2]._obj.value=1
                else:args[1][0]=77
            elif n=='GraphGetEdges':args[-1]._obj.value=0
            elif n=='GraphNodeGetType':args[1]._obj.value=0
            elif n in ('GraphKernelNodeGetParams','FuncGetName','GraphAddKernelNode'):
                pytest.fail('opaque Driver node must not take the Runtime parameter/add API')
            elif n=='GraphDestroy':events.append('graph-destroy')
            elif n=='GraphCreate':args[0]._obj.value=123
            elif n=='GraphInstantiateWithFlags':args[0]._obj.value=456
            elif n=='GraphLaunch':events.append('replay')
            return 0
    class Lib:
        def __getattr__(self,n):return Fn(n.removeprefix('hggc'))
    class Nodes:
        def capture(self,node):
            events.append('capture');return C.c_void_p(88),dict(symbol='driver_kernel',grid=[1,1,1],
                block=[32,1,1],shared_bytes=0,parameter_api='DRIVER_CUFUNCTION')
        def add(self,handle,out,graph,previous):
            assert handle.value==88;out._obj.value=99;events.append('add-driver')
        def destroy(self,handle):assert handle.value==88;events.append('holder-destroy')
    lib=Lib();rt=SimpleNamespace(lib=lib,stream=C.c_void_p(2),sync=lambda:None,kernel_nodes=Nodes(),
        **{name:Fn(name) for name in ('GraphDestroy','GraphInstantiateWithFlags','GraphLaunch','GraphExecDestroy')})
    owner=object()
    cap=Captured(rt,[None],owned_graph=C.c_void_p(91),allocation_owner=owner)
    graph=cap.isolate(0,weight_ring=False,validation_only=True)
    assert cap.allocation_owner is owner and 'holder-destroy' not in events
    graph.close();cap.close();cap.close()
    assert events.count('holder-destroy')==1 and events[-2:]==['holder-destroy','graph-destroy']

"""Native enqueue/order/error tests; fake device clocks are not PPU evidence."""
import ctypes as C
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from dev.strategy.cublas_timing import NativeCalls

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def native_timer(tmp_path_factory):
    root=tmp_path_factory.mktemp('native-events')
    source=root/'timer.cpp'
    source.write_text(r'''
#include <cassert>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>
struct Event { double time=0; };
using hggcStream_t=void*;
using hggcEvent_t=Event*;
enum { hggcErrorInvalidValue=1 };
static int mode, created, destroyed, recorded, calls, synced, readbacks, drains;
static double device_time;
static std::vector<std::pair<int,int>> order;
int hggcEventCreate(Event** e) {
    assert(!calls);
    if (mode==1 && created==5) return 71;
    *e=new Event; ++created; return 0;
}
int hggcEventDestroy(Event* e) { delete e; ++destroyed; return mode==8 ? 78 : 0; }
int hggcStreamSynchronize(void*) { ++drains; return 0; }
int hggcEventRecord(Event* e, void*) {
    ++recorded;
    if (mode==3 && recorded==10) return 73;
    e->time=device_time; return 0;
}
int hggcEventSynchronize(Event*) {
    assert(calls==12*3+1+27);  // no per-phase/per-sample synchronization
    ++synced; return mode==5 ? 75 : 0;
}
int hggcEventElapsedTime(float* ms, Event* a, Event* b) {
    assert(synced==1);
    ++readbacks;
    if (mode==6) return 76;
    *ms=mode==7 ? std::numeric_limits<float>::quiet_NaN() : float(b->time-a->time);
    return 0;
}
#include "dev/strategy/prefill_events.hpp"
int main(int argc, char** argv) {
    assert(argc==2); mode=std::atoi(argv[1]);
    float gpu[27]={}; double host[27]={};
    auto call=[](int phase,int copy) {
        assert(created==56 && synced==0 && readbacks==0);
        ++calls; order.emplace_back(phase,copy);
        if (mode==2 && calls==3) return 72;
        if (mode==4 && calls==40) return 74;
        device_time+=(phase==0 ? .002 : phase==1 ? .040 : .003);
        return 0;
    };
    int rc=strategy_prefill::ordinary_events((void*)1, mode==9 ? 1 : 3,12,3,3,call,gpu,host);
    const int expected=mode==0 ? 0 : (mode==7 || mode==9) ? 1 : 70+mode;
    assert(rc==expected && created==destroyed);
    if (mode==9) assert(!created && !calls);
    if (mode==0) {
        assert(synced==1 && !drains && readbacks==27 && recorded==56);
        for (int i=0;i<36;++i) assert(order[i]==std::make_pair(i%3,(i/3)%3));
        assert(order[36]==std::make_pair(1,2));
        for (int i=0;i<27;++i) {
            assert(order[37+i]==std::make_pair(i%3,(i/3)%3));
            assert(std::abs(gpu[i]-(i%3==0 ? 2 : i%3==1 ? 40 : 3))<.001);
            assert(host[i]>=0 && std::isfinite(host[i]));
        }
    }
}
''')
    binary=root/'timer'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ROOT),
                    str(source),'-o',str(binary)],check=True)
    return binary


@pytest.mark.parametrize('mode',range(10))
def test_native_events_precreate_then_warm_enqueue_once_and_cleanup(native_timer,mode):
    r=subprocess.run([str(native_timer),str(mode)],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr


class Function:
    def __init__(self,fn): self.fn=fn
    def __call__(self,*args): return self.fn(*args)


def binding_fixture(*,stream=17):
    launches=[]
    tensor=lambda p: SimpleNamespace(data_ptr=lambda:p)
    def get_stream(handle,out): out._obj.value=stream; return 0
    # The actual provider function pointer is forwarded, never reopened by name.
    provider=SimpleNamespace(handle=C.c_void_p(99),lib=SimpleNamespace(
        cublasGemmEx=C.CFUNCTYPE(C.c_int)(lambda:0),cublasGetStream_v2=Function(get_stream)))
    def timer(*args):
        assert args[9:16]==(128,1024,5120,3,12,3,3)
        assert list(args[5])==[101,102,103]
        for i in range(27): args[16][i]=float(i+1);args[17][i]=i/10
        return 0
    lib=SimpleNamespace(strategy_prefill_cublas_launch=Function(lambda *args: launches.append(args) or 0),
                        strategy_prefill_cublas_events=Function(timer))
    b=SimpleNamespace(w=SimpleNamespace(operation='dense',experts=1,m=128,n=1024,k=5120),
        r=SimpleNamespace(stream=C.c_void_p(17)),helpers=SimpleNamespace(lib=lib),
        a=SimpleNamespace(ptr=21),out=SimpleNamespace(ptr=22),error=SimpleNamespace(ptr=23))
    core=SimpleNamespace(provider=provider,copies=3,a=tensor(41),out=tensor(42),
                         weights=[tensor(p) for p in (101,102,103)])
    return b,core,launches


def test_prebinding_keeps_weight_addresses_stream_and_sample_order():
    b,core,launches=binding_fixture()
    native=NativeCalls(b,core)
    assert native.matmul(2)==0
    assert launches[0][2:]==(41,103,42,128,1024,5120)
    rows=native.measure(12)
    assert len(rows)==27 and rows[-1]==dict(round=2,sample=2,copy=2,phase='output',
                                           event_span_us=27.,host_submit_us=2.6)
    with pytest.raises(ValueError,match='bounds'): native.matmul(3)


def test_different_event_and_provider_stream_is_rejected_before_launch():
    b,core,launches=binding_fixture(stream=18)
    with pytest.raises(ValueError,match='stream differ'): NativeCalls(b,core)
    assert not launches


def test_no_python_callback_graph_or_profiler_in_native_interval():
    source=(ROOT/'dev/strategy/prefill_events.hpp').read_text()
    assert 'hggcGraph' not in source and 'hggcEventSynchronize' in source
    assert source.index('for (int i = 0; i < slots; ++i)') < source.index('hggcEventSynchronize')
    from dev.strategy.prefill_build import HEADERS
    assert 'dev/strategy/prefill_events.hpp' in HEADERS


def test_blas_forwarder_uses_exact_sdk_abi_and_precision(tmp_path):
    sdk=os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk: pytest.skip('opt-in real SDK header/host ABI check; no GPU')
    source=tmp_path/'blas.cpp'
    source.write_text(r'''
#include <cassert>
#include "dev/strategy/prefill_blas.cpp"
static int invoked;
cublasStatus_t fake(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
    int n,int m,int k,const void* alpha,const void* b,cudaDataType bt,int ldb,
    const void* a,cudaDataType at,int lda,const void* beta,void* c,cudaDataType ct,
    int ldc,cublasComputeType_t compute,cublasGemmAlgo_t algo) {
    ++invoked;
    assert(h==(void*)9 && ta==CUBLAS_OP_T && tb==CUBLAS_OP_N);
    assert(n==1024 && m==128 && k==5120 && ldb==k && lda==k && ldc==n);
    assert(a==(void*)1 && b==(void*)2 && c==(void*)3);
    assert(bt==CUDA_R_16BF && at==bt && ct==bt && *(const float*)alpha==1.f && *(const float*)beta==0.f);
    assert(compute==CUBLAS_COMPUTE_32F && algo==CUBLAS_GEMM_DEFAULT);
    return CUBLAS_STATUS_EXECUTION_FAILED;
}
int main() {
    void* fn=reinterpret_cast<void*>(&fake);
    assert(strategy_prefill_cublas_launch(fn,(void*)9,(void*)1,(void*)2,(void*)3,128,1024,5120)
           == CUBLAS_STATUS_EXECUTION_FAILED);
    assert(invoked==1);
    assert(strategy_prefill_cublas_launch(nullptr,(void*)9,(void*)1,(void*)2,(void*)3,128,1024,5120)==1);
    assert(invoked==1);
}
''')
    binary=tmp_path/'blas'
    subprocess.run(['g++','-std=c++17','-Wall','-Werror','-I'+str(ROOT),
        '-I'+str(Path(sdk)/'CUDA_SDK/include'),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

"""Host arithmetic/API controls. These do not claim PPU execution success."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.probe_native_component_failures import check_binding,select,pack
from dev.native_components.build import between

ROOT=Path(__file__).resolve().parents[1]


class FailureProbeTest(unittest.TestCase):
    def test_bounded_selection_covers_actual_mechanisms(self):
        rows=[]
        for kind,types in (('jit-params',(10,11,12,13)),('q8-activation',(12,13)),('provider-precision',(10,11,12,13))):
            for q in types:
                for m in (8,128,512):
                    rows.append(dict(kind=kind,error=m/10000,row=dict(id=f'{kind}-{q}-{m}',
                        workload=dict(qtype=q,m=m,n=512,k=2048))))
        chosen=select(rows)
        self.assertEqual(len(chosen),9)
        self.assertTrue(all(r['row']['workload']['m']==(8 if r['kind']=='jit-params'else 512)for r in chosen))
        self.assertEqual(chosen,select(list(reversed(rows))))

    def test_only_new_driver_binding_may_be_added(self):
        old={'identity':{k:k for k in ('binding','headers','flags','response_files','compiler_sha256','llama_head')}}
        old['identity']['libraries']={'/sdk/libcudart.so':'x'}
        new=copy.deepcopy(old);new['identity']['libraries']['/sdk/libcuda.so.1']='y'
        check_binding(old,new,{'sdk':'hash'})
        for key in ('binding','headers','flags','response_files','compiler_sha256','llama_head'):
            bad=copy.deepcopy(new);bad['identity'][key]='other'
            with self.assertRaisesRegex(ValueError,'binding changed'):check_binding(old,bad,{'sdk':'hash'})
        for libraries in ({'/sdk/libcudart.so':'CHANGED'}, {'/sdk/libcudart.so':'x','/sdk/libother.so':'y'}):
            bad=copy.deepcopy(new);bad['identity']['libraries']=libraries
            with self.assertRaisesRegex(ValueError,'libraries'):check_binding(old,bad,{'sdk':'hash'})

    def test_diagnostic_does_not_become_a_timing_record(self):
        source=(ROOT/'dev/native_components/carrier.cpp').read_text()
        diagnostic=source[source.index('if(request.value("diagnostic"'):source.index('proofs.push_back(proof(0));')]
        self.assertNotIn('nc.measure',diagnostic)
        self.assertIn('continue;',diagnostic)
        self.assertIn('NUMERIC_AND_EXTRACTION_DIAGNOSTIC_NO_TIMING',diagnostic)
        script=(ROOT/'tools/probe_native_component_failures.py').read_text()
        self.assertNotIn("atomic_json(a.previous",script)
        self.assertIn("if mode!='native':child_env['GGML_CUDA_CUBLAS_COMPUTE_TYPE']=mode",script)

    def test_changed_input_snapshot_waits_for_producer_not_only_copy_stream(self):
        compiler=shutil.which('g++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        # Compile the actual download helper with two independent fake streams.
        # Waiting for tensor_get's copy alone must leave compute pending. The
        # legacy helper then deterministically reads input0 after launching input1.
        helper=between((ROOT/'dev/native_components/carrier.cpp').read_text(),
                       'static std::vector<float> download(', 'static void compare_native(')
        prefix=r'''
#include <vector>
#include <cstddef>
#include <cassert>
#include <cstring>
struct Tensor {std::vector<float> data;};
struct Stream {Tensor * output;std::vector<float> pending;bool waiting=false;int waits=0;};
struct Backend {Stream * p;};
struct Point {Backend backends[1];std::vector<Tensor *> outputs[1];};
static size_t ggml_nelements(Tensor * t) {return t->data.size();}
static void ggml_backend_synchronize(Stream * s) {
    ++s->waits;
    if(s->waiting) {s->output->data=s->pending;s->waiting=false;}
}
static void ggml_backend_tensor_get(Tensor * t,void * out,size_t off,size_t bytes) {
    std::memcpy(out,reinterpret_cast<char *>(t->data.data())+off,bytes);
}
'''
        suffix=r'''
int main() {
    Tensor first{{1,2}},second{{3,4}};Stream producer{&first,{11,12},true};
    Point p{{{&producer}},{{&first,&second}}};
    if(download(p,0)!=std::vector<float>({11,12}) || producer.waiting || producer.waits!=1)return 1;
    producer={&second,{21,22},true};
    if(download(p,1)!=std::vector<float>({21,22}) || producer.waiting || producer.waits!=1)return 2;
    if(first.data!=std::vector<float>({11,12}))return 3;
}
'''
        with tempfile.TemporaryDirectory()as td:
            path=Path(td)
            for name,body,want in (('fixed',helper,0),('missing-producer-wait',
                    helper.replace('ggml_backend_synchronize(point.backends[0].p);',''),1)):
                src=path/(name+'.cpp');src.write_text(prefix+body+suffix)
                subprocess.run([compiler,'-std=c++17',str(src),'-o',str(path/name)],check=True,capture_output=True)
                self.assertEqual(subprocess.run([str(path/name)]).returncode,want)

    def test_diagnostic_keeps_failed_phase_and_metrics_without_relaxing_gate(self):
        text=(ROOT/'dev/native_components/carrier.cpp').read_text()
        comparison=between(text,'static void compare_native(','static int helpers(')
        self.assertIn('require(relative<1e-5,',comparison)
        self.assertLess(comparison.index('if(evidence)*evidence=detail;'),comparison.index('require(relative<1e-5,'))
        diagnostic=between(text,'if(request.value("diagnostic",false)) {','            proofs.push_back(proof(0));')
        self.assertIn('"input0-replay"',diagnostic)
        self.assertIn('"input1-replay"',diagnostic)
        self.assertIn('detail["status"]="INFRASTRUCTURE_FAIL";detail["error"]=e.what();save();throw;',diagnostic)
        # publish() refuses overwrite: write one final success OR one failure,
        # not several updates to the same immutable result path.
        self.assertEqual(diagnostic.count('save();'),2)

    def test_native_q8_host_arithmetic(self):
        compiler=shutil.which('g++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        source=r'''
#include "oracle.hpp"
#include <cassert>
int main() {
    std::vector<float> x(64);
    x[0]=.5f; x[1]=.25f; x[2]=-.25f; x[3]=-.1f;
    int blocks=0;
    auto y=native_q8_1_input(x,[&](float d) {
        ++blocks; if(d==0)return 0.f;
        assert(std::abs(d-.5f/127)<1e-10f);return .003936767578125f;
    });
    assert(blocks==2 && y[0]==127*.003936767578125f);
    assert(y[1]==64*.003936767578125f && y[2]==-64*.003936767578125f);
    assert(y[3]==-25*.003936767578125f && y[63]==0);
    assert(x[1]==.25f); // reference model never changes GPU inputs
    bool rejected=false;
    try {native_q8_1_input(std::vector<float>(31),[](float d){return d;});}
    catch(const std::runtime_error &){rejected=true;} assert(rejected);
    x[0]=INFINITY;rejected=false;
    try {native_q8_1_input(x,[](float d){return d;});}
    catch(const std::runtime_error &){rejected=true;} assert(rejected);
}
'''
        with tempfile.TemporaryDirectory()as td:
            path=Path(td);(path/'test.cpp').write_text(source)
            subprocess.run([compiler,'-std=c++17','-I'+str(ROOT/'dev/native_components'),str(path/'test.cpp'),'-o',str(path/'test')],check=True,capture_output=True)
            subprocess.run([str(path/'test')],check=True)

    def test_driver_fallback_host_api_mock_real_sdk_types(self):
        sdk=Path(os.environ.get('PPU_SDK','/root/ppu-sdk/2.1.1'))
        if not (sdk/'CUDA_SDK/include/cuda.h').is_file():self.skipTest('optional SDK headers')
        source=r'''
#include "kernel_node.hpp"
#include <cassert>
static cudaError_t runtime_rc=cudaSuccess;
static CUresult driver_rc=CUDA_SUCCESS;
static int queried=0,added_runtime=0,added_driver=0;
static void * args[1]{};
extern "C" cudaError_t CUDARTAPI cudaGraphKernelNodeGetParams(cudaGraphNode_t,cudaKernelNodeParams * p) {
    p->func=reinterpret_cast<void *>(123);p->gridDim={3,1,1};p->kernelParams=args;return runtime_rc;
}
extern "C" CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode,CUDA_KERNEL_NODE_PARAMS * p) {
    ++queried;p->func=reinterpret_cast<CUfunction>(456);p->gridDimX=7;p->blockDimX=128;
    p->kernelParams=args;return driver_rc;
}
extern "C" const char * CUDARTAPI cudaGetErrorString(cudaError_t) {return "mock-runtime-error";}
extern "C" cudaError_t CUDARTAPI cudaGetLastError() {return cudaErrorSymbolNotFound;}
extern "C" CUresult CUDAAPI cuGetErrorName(CUresult,const char ** n) {*n="mock-driver-error";return CUDA_SUCCESS;}
extern "C" cudaError_t CUDARTAPI cudaFuncGetName(const char **n,const void *) {*n="runtime";return cudaSuccess;}
extern "C" CUresult CUDAAPI cuFuncGetName(const char **n,CUfunction) {*n="driver";return CUDA_SUCCESS;}
extern "C" cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t *,cudaGraph_t,const cudaGraphNode_t *,size_t,const cudaKernelNodeParams * p) {
    ++added_runtime;assert(p->func==reinterpret_cast<void *>(123));return cudaSuccess;
}
extern "C" CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode * n,CUgraph,const CUgraphNode *,size_t count,const CUDA_KERNEL_NODE_PARAMS * p) {
    ++added_driver;assert(count==1 && p->func==reinterpret_cast<CUfunction>(456));
    assert(p->gridDimX==7 && p->blockDimX==128 && p->kernelParams==args);
    *n=reinterpret_cast<CUgraphNode>(321);return CUDA_SUCCESS;
}
int main() {
    cudaGraphNode_t node=nullptr;
    nc::KernelNode a;a.capture(node);assert(!a.driver_entry && queried==0);
    assert(std::string(a.name())=="runtime");a.add(&node,nullptr,nullptr);assert(added_runtime==1);
    runtime_rc=cudaErrorSymbolNotFound;
    nc::KernelNode b;b.capture(node);assert(b.driver_entry && queried==1);
    assert(std::string(b.name())=="driver");b.add(&node,nullptr,reinterpret_cast<cudaGraphNode_t>(1));
    assert(added_driver==1 && node==reinterpret_cast<cudaGraphNode_t>(321));
    runtime_rc=cudaErrorInvalidValue;bool failed=false;
    try {nc::KernelNode c;c.capture(node);}catch(const std::runtime_error &){failed=true;}
    assert(failed && queried==1); // unrelated runtime errors do NOT fall through
    runtime_rc=cudaErrorSymbolNotFound;driver_rc=CUDA_ERROR_INVALID_VALUE;failed=false;
    try {nc::KernelNode c;c.capture(node);}catch(const std::runtime_error &){failed=true;}
    assert(failed && queried==2 && added_driver==1);
}
'''
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);(root/'test.cpp').write_text(source)
            command=['g++','-std=c++17','-I'+str(ROOT/'dev/native_components'),'-I'+str(sdk/'CUDA_SDK/include'),str(root/'test.cpp'),'-o',str(root/'test')]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            subprocess.run([str(root/'test')],check=True)

    def test_compact_diagnostic_excludes_weights_and_binaries(self):
        import tarfile
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);(root/'fixtures').mkdir();(root/'fixtures/weights.json').write_text('{}')
            (root/'test.so').write_bytes(b'not a library');(root/'summary.json').write_text('{}')
            with tarfile.open(pack(root))as tar:self.assertEqual(tar.getnames(),['summary.json'])


if __name__=='__main__':unittest.main()

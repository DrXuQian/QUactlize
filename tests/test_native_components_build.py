"""CMake cwd/options-file regression tests; no synthetic GPU admission."""
import json
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dev.native_components.build import (adapter_source, build_adapter, native_flags, response_files,
                                        run_commands, NATIVE_ONLY_BEGIN, NATIVE_ONLY_END, HERE)
from quactlize.runtime.compiler import sha
from tools.run_native_components_multi import worker_failure
from tools.run_native_components import carrier, environment


class NativeIsolationTest(unittest.TestCase):
    def test_real_host_link_excludes_hooks_but_keeps_header_abi_and_native_buffers(self):
        # The same link failure without a GPU SDK. The optional hook has no
        # definition: do not make a successful link depend on a permissive stub.
        native = r'''
#include <stdexcept>
#define __global__
using half=short;
struct Buffer { void * type; };
struct ggml_tensor { Buffer * buffer=nullptr; ggml_tensor * view_src=nullptr;
    ggml_tensor * src[3]={}; int op=1; };
struct ggml_backend_cuda_context { int device;
#ifdef GGML_NCP_QUACTLIZE
    int extension_header_abi;
#endif
};
constexpr int GGML_OP_MUL_MAT=1,GGML_OP_MUL_MAT_ID=2;
static int native_calls=0;
static int native_types[2];
void * ggml_backend_cuda_buffer_type(int d) { return &native_types[d]; }
void * ggml_backend_buffer_get_type(Buffer * b) { return b->type; }
void ggml_cuda_set_device(int) {}
int cudaGetLastError() { return 0; }
bool ggml_quactlize_is_alias(const ggml_tensor *);
void ggml_quactlize_mul_mat(ggml_backend_cuda_context &,ggml_tensor *);
int cublasSgemm(int) { return 0; }
int cublasGemmEx(int) { return 0; }
int cublasGemmStridedBatchedEx(int) { return 0; }
int cublasGemmBatchedEx(int) { return 0; }
static_assert(sizeof(half)==2);
static __global__ void k_compute_batched_ptrs() {}
static float af(float alpha) { static const float val = alpha; return val; }
static float bf(float beta) { static const float val = beta; return val; }
static half ah(half alpha) { static const half val = alpha; return val; }
static half bh(half beta) { static const half val = beta; return val; }
static void provider_sites() {
    int main_stream=0;
    cublasSgemm(0); cublasGemmEx(0); cublasGemmStridedBatchedEx(0); cublasGemmBatchedEx(0);
}
static bool ggml_cuda_should_fuse_mul_mat() { return false; }
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,const ggml_tensor *,ggml_tensor * dst) {
#ifdef GGML_NCP_QUACTLIZE
    if (ggml_quactlize_is_alias(src0)) { ggml_quactlize_mul_mat(ctx,dst); return; }
#endif
    ++native_calls;
}
static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    ggml_cuda_mul_mat(ctx,dst->src[0],dst->src[1],dst);
}
static bool ggml_cuda_compute_forward() { return false; }
'''
        for enabled in (False,True):
            with self.subTest(extension=enabled),tempfile.TemporaryDirectory()as td:
                root=Path(td);llama=root/'llama';directory=llama/'ggml/src/ggml-cuda'
                directory.mkdir(parents=True);(directory/'ggml-cuda.cu').write_text(native)
                (root/'record.hpp').write_text('namespace nc {\n'
                    'template<class F,class... A> int provider(const char*,int,F f,A... a) { return f(a...); }\n'
                    'inline void check(int,const char*) {}\n}\n')
                # Compile the shipping adapter guard, not a Python simulation.
                guard=(HERE/'bridge.cuh').read_text().split('struct Session {',1)[0]+'}\n'
                (root/'bridge.cuh').write_text(guard+r'''
extern "C" int header_abi() { return sizeof(ggml_backend_cuda_context); }
extern "C" int restored_macro() {
#ifdef GGML_NCP_QUACTLIZE
    return 1;
#else
    return 0;
#endif
}
extern "C" int exercise(int indexed,int bad) {
    ggml_backend_cuda_context ctx{};
    int foreign=0; Buffer plain{ggml_backend_cuda_buffer_type(0)},packed{&foreign};
    Buffer other{ggml_backend_cuda_buffer_type(1)};
    ggml_tensor weight{},view{},out{};weight.buffer=&plain;
    out.op=indexed?GGML_OP_MUL_MAT_ID:GGML_OP_MUL_MAT;out.src[0]=&weight;
    if (bad==1) { view.view_src=&weight;out.src[0]=&view; }
    if (bad==2) weight.buffer=&packed;
    if (bad==3) { weight.buffer=&packed;view.buffer=&plain;view.view_src=&weight;out.src[0]=&view; }
    if (bad==4) { view.buffer=&packed;view.view_src=&weight;out.src[0]=&view; }
    if (bad==5) weight.buffer=nullptr;
    if (bad==6) weight.buffer=&other;
    const int before=native_calls;
    try { nc::dispatch(ctx,bad==7?nullptr:&out); }
    catch (const std::runtime_error &) { return native_calls==before?-1:-2; }
    return native_calls-before;
}
''')
                generated,binding=adapter_source(llama)
                self.assertEqual(binding['native_extension_isolation']['weights'],'ORDINARY_CUDA_BUFFER_ONLY')
                self.assertIn(native.split('static void ggml_cuda_mul_mat(',1)[1].split(
                    'static bool ggml_cuda_compute_forward(',1)[0],generated)
                src=root/'adapter.cpp';lib=root/'adapter.so'
                command=['g++','-std=c++17','-O1','-fPIC','-shared','-Wl,--no-undefined',
                         '-I'+str(root),str(src),'-o',str(lib)]
                if enabled:command.insert(1,'-DGGML_NCP_QUACTLIZE')
                if enabled:
                    src.write_text(generated.replace(NATIVE_ONLY_BEGIN,'').replace(NATIVE_ONLY_END,''))
                    old=subprocess.run(command,capture_output=True,text=True)
                    self.assertNotEqual(old.returncode,0)
                    self.assertIn('undefined reference',old.stderr)
                    self.assertIn('ggml_quactlize_is_alias',old.stderr)
                src.write_text(generated)
                linked=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(linked.returncode,0,linked.stdout+linked.stderr)
                symbols=subprocess.check_output(['nm','-u','-C',str(lib)],text=True)
                self.assertNotIn('ggml_quactlize_',symbols)
                api=C.CDLL(str(lib));api.exercise.argtypes=[C.c_int,C.c_int]
                self.assertEqual(api.header_abi(),8 if enabled else 4)
                self.assertEqual(api.restored_macro(),int(enabled))
                for indexed in (0,1):
                    for bad in range(8):
                        self.assertEqual(api.exercise(indexed,bad),1 if bad<2 else -1)


class NativeBuildTest(unittest.TestCase):
    def fixture(self, root):
        source=root/'llama source';build=root/'llama build';sdk=root/'sdk'
        target=build/'ggml/src/ggml-cuda'
        for directory in (source/'ggml/src/ggml-cuda',source/'ggml/include',source/'vendor',
                          target/'CMakeFiles/ggml-cuda.dir',build/'bin',sdk/'bin'):
            directory.mkdir(parents=True,exist_ok=True)
        compiler=sdk/'bin/nvcc';compiler.write_text('HOST TEST COMPILER PLACEHOLDER')
        (source/'ggml/src/ggml-cuda/common.cuh').write_text('// host test')
        for name in ('libggml-cuda.so','libggml-base.so'):
            (build/'bin'/name).write_text('HOST TEST LIBRARY PLACEHOLDER')
        (build/'CMakeCache.txt').write_text(
            f'CMAKE_HOME_DIRECTORY:INTERNAL={source}\nCMAKE_CUDA_COMPILER:FILEPATH={compiler}\nGGML_USE_PPU:BOOL=ON\n')
        directory=target/'CMakeFiles/ggml-cuda.dir'
        flags=directory/'flags.make'
        flags.write_text('CUDA_DEFINES = -DGGML_USE_PPU\n'
            'CUDA_INCLUDES = --options-file CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp\n'
            'CUDA_FLAGS = -O3 -std=c++17 -Xcompiler "-Wall -Wextra"\n')
        response=directory/'includes_CUDA.rsp'
        response.write_text('-I"relative include"\n')
        return source,build,sdk,target,response

    def test_exact_flags_and_target_cwd(self):
        with tempfile.TemporaryDirectory()as td:
            source,build,sdk,target,rsp=self.fixture(Path(td))
            compiler,flags,path=native_flags(source,build)
            self.assertEqual(compiler,str(sdk/'bin/nvcc'))
            self.assertEqual(flags,['-DGGML_USE_PPU','--options-file',
                'CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp','-O3','-std=c++17','-Xcompiler','-Wall -Wextra'])
            self.assertEqual(path.parents[2],target)
            self.assertEqual(response_files(flags,target),{str(rsp):sha(rsp)})
            with self.assertRaisesRegex(ValueError,'missing native options file'):
                response_files(flags,build)  # The shipped wrong cwd reproduces the failure.

    def test_real_subprocess_reads_relative_options_from_target(self):
        with tempfile.TemporaryDirectory()as td:
            _,build,_,target,_=self.fixture(Path(td))
            code='from pathlib import Path; assert Path("CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp").is_file(), "options file missing"'
            command=[sys.executable,'-c',code]
            with self.assertRaisesRegex(RuntimeError,'options file missing'):
                run_commands([command],build,Path(td)/'old-cwd.log')
            run_commands([command],target,Path(td)/'native-cwd.log')

    def test_receipt_seals_rsp_and_build_preserves_cwd_and_flags(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);source,build,sdk,target,rsp=self.fixture(root);output=root/'output'
            invocations=[]

            def compiler(command,*,cwd,stdout,stderr):
                invocations.append((list(command),cwd))
                self.assertEqual(cwd,target)
                if len(invocations)==1:
                    self.assertEqual(response_files(command,cwd),{str(rsp):sha(rsp)})
                Path(command[command.index('-o')+1]).write_text('HOST TEST OUTPUT, NOT DEVICE CODE')
                return subprocess.CompletedProcess(command,0)

            with patch('dev.native_components.build.adapter_source',return_value=('// test source',{})),\
                 patch('dev.native_components.build.subprocess.check_output',return_value='a'*40+'\n'),\
                 patch('dev.native_components.build.subprocess.run',side_effect=compiler):
                binary,receipt=build_adapter(source,build,sdk,output)
                self.assertTrue(binary.is_file());self.assertEqual(len(invocations),2)
                self.assertEqual(receipt['identity']['compiler_workdir'],str(target))
                self.assertEqual(receipt['identity']['response_files'],{str(rsp):sha(rsp)})
                self.assertIn('--options-file',invocations[0][0])
                self.assertEqual(build_adapter(source,build,sdk,output)[1],receipt)
                self.assertEqual(len(invocations),2)  # Receipt reuses the completed adapter.
                rsp.write_text('-Ichanged\n')
                with self.assertRaisesRegex(ValueError,'native adapter identity changed'):
                    build_adapter(source,build,sdk,output)
                self.assertEqual(len(invocations),2)

    def test_newer_source_mtime_does_not_require_backend_rebuild(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);source,build,sdk,_,rsp=self.fixture(root);output=root/'output'
            header=source/'ggml/src/ggml-cuda/common.cuh'
            stamp=(build/'bin/libggml-cuda.so').stat().st_mtime_ns+10_000_000_000
            os.utime(header,ns=(stamp,stamp))
            self.assertGreater(header.stat().st_mtime_ns,(build/'bin/libggml-cuda.so').stat().st_mtime_ns)

            def compiler(command,**kwargs):
                Path(command[command.index('-o')+1]).write_text('HOST TEST ONLY')
                return subprocess.CompletedProcess(command,0)

            with patch('dev.native_components.build.adapter_source',return_value=('// test source',{})),\
                 patch('dev.native_components.build.subprocess.check_output',return_value='a'*40+'\n'),\
                 patch('dev.native_components.build.subprocess.run',side_effect=compiler)as compile_call:
                result=build_adapter(source,build,sdk,output)
                for path in (header,rsp,build/'bin/libggml-cuda.so',sdk/'bin/nvcc'):
                    os.utime(path,ns=(stamp+1,stamp+1))
                self.assertEqual(build_adapter(source,build,sdk,output),result)
                self.assertEqual(compile_call.call_count,2)

    def test_content_changes_still_reject_receipt_with_older_timestamps(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);source,build,sdk,target,rsp=self.fixture(root);output=root/'output'
            dispatch=source/'ggml/src/ggml-cuda/ggml-cuda.cu';dispatch.write_text('// native source')

            def compiler(command,**kwargs):
                Path(command[command.index('-o')+1]).write_text('HOST TEST ONLY')
                return subprocess.CompletedProcess(command,0)

            def binding(_):
                return dispatch.read_text(),{'source_sha256':sha(dispatch)}

            with patch('dev.native_components.build.adapter_source',side_effect=binding),\
                 patch('dev.native_components.build.subprocess.check_output',return_value='a'*40+'\n'),\
                 patch('dev.native_components.build.subprocess.run',side_effect=compiler)as compile_call:
                binary,receipt=build_adapter(source,build,sdk,output)
                for path in (dispatch,source/'ggml/src/ggml-cuda/common.cuh',
                             build/'bin/libggml-cuda.so',build/'bin/libggml-base.so',sdk/'bin/nvcc',
                             build/'CMakeCache.txt',target/'CMakeFiles/ggml-cuda.dir/flags.make',rsp,binary):
                    with self.subTest(path=path):
                        original=path.read_bytes()
                        try:
                            path.write_bytes(original+b'\n# changed content\n')
                            os.utime(path,ns=(1,1))
                            with self.assertRaisesRegex(ValueError,'native adapter identity changed'):
                                build_adapter(source,build,sdk,output)
                        finally:
                            path.write_bytes(original)
                self.assertEqual(build_adapter(source,build,sdk,output)[1],receipt)
                self.assertEqual(compile_call.call_count,2)

    def test_response_forms_nested_and_content_hash(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);a=root/'a.rsp';b=root/'b.rsp';c=root/'space name.rsp'
            a.write_text('-optf=b.rsp\n');b.write_text('-Irelative\n');c.write_text('-DUNCHANGED=1\n')
            for flags in (['--options-file','a.rsp'],['--options-file=a.rsp'],['-optf','a.rsp'],['-optf=a.rsp']):
                self.assertEqual(response_files(flags,root),{str(a):sha(a),str(b):sha(b)})
            self.assertEqual(response_files(['-optf=a.rsp,b.rsp'],root),{str(a):sha(a),str(b):sha(b)})
            self.assertEqual(response_files(['--options-file',str(c)],root),{str(c):sha(c)})
            before=response_files(['--options-file','a.rsp'],root)
            b.write_text('-Iother\n')
            self.assertNotEqual(before,response_files(['--options-file','a.rsp'],root))
            b.write_text('--options-file a.rsp\n')
            with self.assertRaisesRegex(ValueError,'cyclic'):response_files(['-optf=a.rsp'],root)

    def test_missing_response_arguments_fail_before_compile(self):
        with tempfile.TemporaryDirectory()as td:
            for flags in (['--options-file'],['-optf='],['--options-file','missing.rsp']):
                with self.subTest(flags=flags),self.assertRaises(ValueError):response_files(flags,Path(td))

    def test_compile_error_preserves_full_log_and_prints_diagnostic(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);log=root/'compiler.log'
            code='import sys; sys.stderr.buffer.write(b"fatal error: includes_CUDA.rsp missing\\xff\\n"); sys.exit(7)'
            with self.assertRaisesRegex(RuntimeError,'compile failed rc=7')as caught:
                run_commands([[sys.executable,'-c',code]],root,log)
            self.assertIn('includes_CUDA.rsp missing',str(caught.exception))
            self.assertIn('cwd='+str(root),str(caught.exception))
            self.assertIn(b'\xff',log.read_bytes())
            self.assertEqual(json.loads(log.read_text(errors='replace').splitlines()[0]),dict(cwd=str(root)))

    def test_coordinator_surfaces_actual_worker_error(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);log=root/'console.log'
            log.write_bytes(b'uninteresting output\ncommand fatal : Could not open options file\xff\n')
            result=worker_failure(root,1)
            self.assertIn('Could not open options file',result['diagnostic'])
            self.assertNotIn('uninteresting',result['diagnostic'])
            self.assertEqual(result['returncode'],1)
            missing=worker_failure(root/'absent',1)
            self.assertEqual(missing['diagnostic'],'')

    def test_prefill_coordinator_uses_its_exact_timestamped_log(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);log=root/'console-123.log'
            log.write_text("undefined reference to `ggml_quactlize_is_alias(ggml_tensor const*)'\n")
            from tools.run_prefill_cohort_multi import worker_failure as prefill_failure
            failure=prefill_failure(root,1,log=log)
            self.assertEqual(failure['log'],str(log))
            self.assertIn('ggml_quactlize_is_alias',failure['diagnostic'])

    def test_carrier_compile_and_link_errors_reach_worker_summary(self):
        for message in ('fatal error: ggml.h: No such file or directory',
                        "undefined reference to `native_test_symbol'"):
            with self.subTest(message=message),tempfile.TemporaryDirectory()as td:
                root=Path(td);source,build,sdk,_,_=self.fixture(root)
                output=root/'output';output.mkdir()
                args=SimpleNamespace(llama_source=source,llama_build=build,sdk=sdk)

                def compiler(command,*,cwd,env,stdout,stderr):
                    self.assertEqual(command[0],'g++')
                    self.assertEqual(cwd,build)
                    self.assertEqual(env,environment(args))
                    stdout.write(message+'\n')
                    return subprocess.CompletedProcess(command,1)

                with patch('dev.native_components.build.subprocess.run',side_effect=compiler):
                    with self.assertRaisesRegex(RuntimeError,'native carrier compile/link failed rc=1')as caught:
                        carrier(args,output)
                self.assertIn(message,str(caught.exception))
                self.assertIn(message,(output/'carrier-build.log').read_text())
                self.assertFalse((output/'carrier.json').exists())
                (root/'console.log').write_text(str(caught.exception))
                self.assertIn(message,worker_failure(root,1)['diagnostic'])

    def test_carrier_success_receipt_reused_and_binary_change_rejected(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);source,build,sdk,_,_=self.fixture(root)
            output=root/'output';output.mkdir()
            args=SimpleNamespace(llama_source=source,llama_build=build,sdk=sdk)

            def compiler(command,**kwargs):
                Path(command[command.index('-o')+1]).write_text('HOST TEST ONLY')
                return subprocess.CompletedProcess(command,0)

            with patch('dev.native_components.build.subprocess.run',side_effect=compiler)as compile_call:
                binary=carrier(args,output)
                self.assertEqual(carrier(args,output),binary)
                compile_call.assert_called_once()
                binary.write_text('CHANGED')
                with self.assertRaisesRegex(ValueError,'native fixture changed'):
                    carrier(args,output)

    def test_command_runner_passes_explicit_environment(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td)
            command=[sys.executable,'-c','import os; assert os.environ["NATIVE_BUILD_TEST"] == "carrier"']
            run_commands([command],root,root/'environment.log',
                         env={'NATIVE_BUILD_TEST':'carrier'},stages=['native carrier compile/link'])


if __name__=='__main__':unittest.main()

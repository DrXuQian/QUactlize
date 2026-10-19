"""CMake cwd/options-file regression tests; no synthetic GPU admission."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dev.native_components.build import build_adapter, native_flags, response_files, run_commands
from quactlize.runtime.compiler import sha
from tools.run_native_components_multi import worker_failure


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


if __name__=='__main__':unittest.main()

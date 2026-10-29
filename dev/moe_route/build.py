"""Build one independent route TU and one wrapper for the linked native helper."""
import os
from pathlib import Path
import subprocess
import time

from dev.native_components.build import native_flags, response_files, run_commands
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha

ROOT = Path(__file__).resolve().parents[2]


def revision(root):
    return dict(commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                dirty=bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)))


def build(sdk, llama, llama_build, output):
    sdk, llama, llama_build = (Path(p).resolve(strict=True) for p in (sdk, llama, llama_build))
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    compiler, flags, flag_file = native_flags(llama, llama_build)
    cwd = flag_file.parents[2]
    responses = response_files(flags, cwd)
    images = [llama_build/'bin'/name for name in ('libggml-cuda.so', 'libggml-base.so')]
    symbols = subprocess.check_output(['nm', '-D', '-C', str(images[0])], text=True)
    if 'ggml_cuda_launch_mm_ids_helper(' not in symbols:
        raise ValueError('linked native library does not export mm_ids_helper; do not substitute a copied kernel')
    files = list((ROOT/'quactlize/routing').glob('*')) + list((ROOT/'dev/moe_route').glob('*')) + [
        ROOT/'tools/run_moe_route.py', ROOT/'dev/prefill_asys/trace.py', ROOT/'dev/native_components/build.py',
        ROOT/'quactlize/runtime/compiler.py', ROOT/'quactlize/runtime/native.py', ROOT/'quactlize/runtime/tuning.py']
    sources = {str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file() and p.suffix != '.pyc'}
    native_headers = {str(p.relative_to(llama)): sha(p) for directory in ('ggml/include', 'ggml/src/ggml-cuda')
                      for p in (llama/directory).rglob('*') if p.suffix in ('.h', '.hpp', '.cuh')}
    images += [sdk/'lib'/('lib'+name+'.so') for name in LIBRARIES]
    identity = dict(quactlize=revision(ROOT), sources=sources, llama=revision(llama),
                    native_headers=native_headers, native_flags=flags, response_files=responses,
                    libraries={str(p.resolve()): sha(p) for p in images},
                    sdk=str(sdk), compiler_sha256=sha(sdk/'bin/hgcc'),
                    native_compiler_sha256=sha(Path(compiler).resolve()))
    qz, native = output/'libroute-candidate.so', output/'libroute-native.so'
    commands = [
        [str(sdk/'bin/hgcc'), *FLAGS, '-DQZR_TESTING', '-c', str(ROOT/'quactlize/routing/route.cu'),
         '-o', str(output/'route.o')],
        ['g++', '-shared', '-Wl,-z,defs', '-Wl,-Bsymbolic', str(output/'route.o'),
         '-L'+str(sdk/'lib'), '-Wl,-rpath,'+str(sdk/'lib'), *['-l'+n for n in LIBRARIES], '-o', str(qz)],
        [compiler, *flags, '-I'+str(ROOT), '-I'+str(llama/'ggml/src/ggml-cuda'),
         '-c', str(ROOT/'dev/moe_route/native.cu'), '-o', str(output/'native.o')],
        ['g++', '-shared', '-Wl,-z,defs', str(output/'native.o'), '-L'+str(llama_build/'bin'),
         '-Wl,-rpath,'+str(llama_build/'bin'), '-Wl,-rpath-link,'+str(llama_build/'bin'), '-lggml-cuda', '-lggml-base',
         '-L'+str(sdk/'lib'), '-Wl,-rpath-link,'+str(sdk/'lib'), '-lhggc_wrapper',
         '-L'+str(sdk/'CUDA_SDK/targets/x86_64-linux/lib'),
         '-Wl,-rpath-link,'+str(sdk/'CUDA_SDK/targets/x86_64-linux/lib'),
         '-lcublas', '-lcudart', '-lcuda', '-pthread', '-ldl', '-o', str(native)],
    ]
    start = time.monotonic()
    env = dict(os.environ, PATH=str(sdk/'bin')+':'+os.environ.get('PATH', ''))
    run_commands(commands, cwd, output/'build.log', env=env)
    if any(sha(ROOT/p) != value for p, value in sources.items()) or response_files(flags, cwd) != responses:
        raise ValueError('build inputs changed during compilation')
    identity.update(seconds=time.monotonic()-start, commands=commands,
                    candidate=str(qz), native=str(native),
                    binary_sha256={str(p): sha(p) for p in (qz, native)},
                    native_kind='LINKED_LLAMA_MM_IDS_HELPER', production_changed=False)
    atomic_json(output/'build.json', identity)
    return identity

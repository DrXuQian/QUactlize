"""Narrow test adapter from the caller's exact native host dispatcher.

No git checkout, caller patch, backend rebuild or alternate MMVQ heuristic.
The test TU links the existing native kernel implementations. Provider calls
are recorded outside capture; their scalar pointers in this source span are
the native traits' static alpha/beta values, not temporary test-stack data.
"""
import json
from pathlib import Path
import re
import shlex
import subprocess

from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from dev.strategy.journal import atomic_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def response_files(flags, workdir):
    """Seal nvcc options files without rewriting native compiler arguments.

    CMake emits these paths relative to the target binary directory, NOT
    the top-level build or the directory containing flags.make. Nested
    options files use the same compiler cwd. Missing/cyclic files fail before
    compilation, rather than silently dropping the target's include paths.
    """
    result={}

    def visit(arguments, active):
        i=0
        while i<len(arguments):
            argument=arguments[i];value=None
            if argument in ('--options-file','-optf'):
                i+=1
                if i==len(arguments):raise ValueError('native options-file argument is missing')
                value=arguments[i]
            elif argument.startswith(('--options-file=','-optf=')):
                value=argument.split('=',1)[1]
            if value is not None:
                for name in value.split(','):
                    if not name:raise ValueError('empty native options-file path')
                    path=(workdir/name).resolve()
                    if path in active:raise ValueError('cyclic native options file: '+str(path))
                    if not path.is_file():
                        raise ValueError(f'missing native options file: {path} (compiler cwd={workdir})')
                    if str(path)not in result:
                        result[str(path)]=sha(path)
                        visit(shlex.split(path.read_text()),active|{path})
            i+=1

    visit(flags,set())
    return result


def run_commands(commands, workdir, log_path, *, env=None, stages=None):
    """Keep the full log and expose the compiler's own diagnostic on failure."""
    if stages is None:
        stages=['native adapter '+('compile'if i==0 else'link')for i in range(len(commands))]
    if len(stages)!=len(commands):
        raise ValueError('build commands and diagnostic stages differ')
    with log_path.open('x')as log:
        log.write(json.dumps(dict(cwd=str(workdir)))+'\n')
        for index,command in enumerate(commands):
            log.write(json.dumps(command)+'\n');log.flush()
            options={}if env is None else dict(env=env)
            result=subprocess.run(command,cwd=workdir,stdout=log,stderr=subprocess.STDOUT,**options)
            if result.returncode:
                log.flush()
                with log_path.open('rb')as reader:
                    reader.seek(max(0,log_path.stat().st_size-8192))
                    tail='\n'.join(reader.read().decode(errors='replace').splitlines()[-24:])
                raise RuntimeError(f'{stages[index]} failed '
                                   f'rc={result.returncode}; cwd={workdir}; log={log_path}\n{tail}')


def between(text, start, end):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('native source seam changed: '+start)
    a, b = text.index(start), text.index(end)
    if b <= a:
        raise ValueError('native source spans overlap')
    return text[a:b]


def wrap_providers(text):
    names = ('cublasSgemm', 'cublasGemmEx', 'cublasGemmStridedBatchedEx', 'cublasGemmBatchedEx')
    counts = {}
    for name in names:
        spans = []
        for match in re.finditer(r'\b'+name+r'\s*\(', text):
            depth, i = 1, match.end()
            while i < len(text) and depth:
                depth += (text[i] == '(') - (text[i] == ')'); i += 1
            if depth:
                raise ValueError('unclosed provider call')
            spans.append((match.start(), i, text[match.end():i-1]))
        if len(spans) != 1:
            raise ValueError('native provider inventory changed: '+name)
        counts[name] = len(spans)
        for start, end, args in reversed(spans):
            text = text[:start]+('nc::provider("'+name+'", main_stream, '
                '[](auto... v) { return '+name+'(v...); }, '+args+')')+text[end:]
    return text, counts


def adapter_source(llama):
    source = llama/'ggml/src/ggml-cuda/ggml-cuda.cu'
    text = source.read_text()
    headers = text[:text.index('static_assert(sizeof(half)')]
    blas = between(text, 'static __global__ void k_compute_batched_ptrs(',
                   'static bool ggml_cuda_should_fuse_mul_mat(')
    # Recording provider arguments must not preserve pointers to dead scalars.
    for kind in ('float', 'half'):
        if f'static const {kind} val = alpha;' not in blas or f'static const {kind} val = beta;' not in blas:
            raise ValueError('native scalar ownership changed; review recording adapter')
    if re.search(r'(?:^|[;{])\s*(?:float|half)\s+(?:alpha|beta)\s*=', blas):
        raise ValueError('unreviewed local provider scalar')
    dispatch = between(text, 'static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,',
                       'static bool ggml_cuda_compute_forward(')
    adapted, providers = wrap_providers(blas)
    result = headers+'\n#include "record.hpp"\n'+adapted+dispatch+'\n#include "bridge.cuh"\n'
    return result, dict(native_dispatch_sha256=digest(dispatch), native_provider_source_sha256=digest(blas),
                        provider_sites=providers, source_file=str(source), source_sha256=sha(source),
                        arithmetic_changed=False, selector_changed=False, llama_tree_modified=False)


def native_flags(llama, build):
    cache = (build/'CMakeCache.txt').read_text()
    home = re.search(r'^CMAKE_HOME_DIRECTORY:[^=]+=(.*)$', cache, re.M)
    compiler = re.search(r'^CMAKE_CUDA_COMPILER:[^=]+=(.*)$', cache, re.M)
    if not home or Path(home[1]).resolve() != llama.resolve() or not compiler:
        raise ValueError('native source/build/compiler binding differs')
    if not re.search(r'^GGML_USE_PPU:BOOL=ON$', cache, re.M):
        raise ValueError('native component adapter requires the existing PPU build')
    path = build/'ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make'
    if not path.is_file():
        raise ValueError('missing native target flags.make (Makefiles build required)')
    flags = []
    for name in ('CUDA_DEFINES', 'CUDA_INCLUDES', 'CUDA_FLAGS'):
        found = re.search(r'^'+name+r'\s*=\s*(.*)$', path.read_text(), re.M)
        if not found:
            raise ValueError('missing exact native flags: '+name)
        flags += shlex.split(found[1])
    if any(x in flags for x in ('-o', '-c', '--compile')):
        raise ValueError('unexpected compiler action in flags.make')
    return compiler[1], flags, path


def build_adapter(llama, build, sdk, output):
    llama, build, sdk = (Path(p).resolve(strict=True) for p in (llama, build, sdk))
    output.mkdir(parents=True, exist_ok=True)
    source, binding = adapter_source(llama)
    compiler, flags, flag_file = native_flags(llama, build)
    # Make's CUDA compile rule runs "cd <target binary dir> && nvcc ...".
    # Keep that cwd for relative -I paths and --options-file arguments alike.
    workdir=flag_file.parents[2]
    responses=response_files(flags,workdir)
    compiler_path=Path(compiler).resolve(strict=True)
    if not compiler_path.is_relative_to(sdk):
        raise ValueError('native compiler is outside the requested PPU SDK; use its matching SDK')
    libs = [build/'bin/libggml-cuda.so', build/'bin/libggml-base.so']
    if any(not p.is_file() for p in libs):
        raise ValueError('build the native backend with its existing .aoneci/scripts/build.sh first')
    # Checkout/copy timestamps do not establish source/DSO compatibility.
    # Seal contents below; the carrier checks extracted vs linked execution.
    images=set(p.resolve()for p in (build/'bin').glob('*.so*')if p.is_file())
    for directory in (sdk/'lib',sdk/'targets/x86_64-linux/lib',sdk/'CUDA_SDK/targets/x86_64-linux/lib'):
        for name in ('libcublas.so','libcudart.so','libcuda.so','libhggc_wrapper.so','libhggc.so','libhggcrt1.so'):
            p=directory/name
            if p.is_file():images.add(p.resolve())
    identity = dict(binding=binding, flags_sha256=sha(flag_file), flags=flags,
        compiler_workdir=str(workdir),response_files=responses,
        cmake_sha256=sha(build/'CMakeCache.txt'),
        llama_head=subprocess.check_output(['git','-C',str(llama),'rev-parse','HEAD'],text=True).strip(),
        headers={str(p.relative_to(llama)):sha(p) for base in ('ggml/include','ggml/src/ggml-cuda')
                 for p in (llama/base).rglob('*') if p.is_file() and p.suffix in ('.h','.hpp','.cuh')},
        libraries={str(p):sha(p) for p in sorted(images)},
        test_sources={p.name:sha(p) for p in HERE.iterdir() if p.is_file()},
        compiler=str(compiler_path),compiler_sha256=sha(compiler_path), sdk=str(sdk))
    manifest = output/'adapter.json'
    binary = output/'libnative-components.so'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['identity'] != identity or old['binary_sha256'] != sha(binary):
            raise ValueError('native adapter identity changed; use a new output directory')
        return binary, old
    generated, obj = output/'native-dispatch.cu', output/'native-dispatch.o'
    if any(p.exists() for p in (binary, generated, obj)):
        raise ValueError('unreceipted adapter build exists; use a new output directory')
    generated.write_text(source)
    commands = [[str(compiler), *flags, '-I'+str(HERE), '-I'+str(llama/'vendor'),
                 '-I'+str(llama/'ggml/src'), '-c', str(generated), '-o', str(obj)],
        ['g++','-shared','-Wl,--no-undefined',str(obj),'-o',str(binary),'-L'+str(build/'bin'),
         '-Wl,-rpath,'+str(build/'bin'),'-lggml-cuda','-lggml-base',
         '-L'+str(sdk/'CUDA_SDK/targets/x86_64-linux/lib'),'-lcublas','-lcudart','-lcuda',
         '-L'+str(sdk/'lib'),'-L'+str(sdk/'targets/x86_64-linux/lib'),
         '-Wl,-rpath-link,'+str(sdk/'CUDA_SDK/targets/x86_64-linux/lib'),
         '-Wl,-rpath-link,'+str(sdk/'targets/x86_64-linux/lib'),'-lhggc_wrapper','-pthread','-ldl']]
    run_commands(commands,workdir,output/'adapter-build.log')
    if response_files(flags,workdir)!=responses:
        raise ValueError('native response files changed during adapter build')
    receipt = dict(identity=identity, binary_sha256=sha(binary), commands=commands,
                   scope='TEST_ONLY_EXACT_NATIVE_HOST_DISPATCH_NOT_DEVICE_ADMISSION')
    atomic_json(manifest, receipt)
    return binary, receipt


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source',type=Path,required=True)
    parser.add_argument('--emit-only',type=Path,required=True)
    args=parser.parse_args()
    text,binding=adapter_source(args.llama_source)
    with args.emit_only.open('x') as out:out.write(text)
    print(json.dumps(binding))

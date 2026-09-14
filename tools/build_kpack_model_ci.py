#!/usr/bin/env python3
"""Run the pinned llama joint CI build without modifying the supplied checkouts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def clone_checkout(source, target, revision):
    source = source.resolve(strict=True)
    if Path(git(source, 'rev-parse', '--show-toplevel')).resolve() != source:
        raise ValueError(f'not a source checkout: {source}')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('source revision must be a full commit ID')
    git(source, 'cat-file', '-e', revision + '^{commit}')
    subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', '--', str(source), str(target)], check=True)
    git(target, 'checkout', '--detach', revision)
    if not (target / '.gitmodules').is_file():
        return
    entries = git(target, 'config', '-f', '.gitmodules', '--get-regexp', r'^submodule\..*\.path$')
    for entry in entries.splitlines():
        _, name = entry.split(maxsplit=1)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'invalid submodule path: {name}')
        mode, _, pin, _ = git(target, 'ls-tree', 'HEAD', '--', name).split(maxsplit=3)
        if mode != '160000':
            raise ValueError(f'not a pinned submodule: {name}')
        # Clone committed objects, not locally patched files or existing build trees.
        clone_checkout(source / relative, target / relative, pin)
    git(target, 'submodule', 'init')
    git(target, 'submodule', 'absorbgitdirs')


def build_receipt(llama, ncp, sdk, jobs):
    build = llama / 'build-ci'
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if '=' in line and ':' in line and not line.startswith(('//', '#')):
            key, value = line.split('=', 1)
            cache[key.split(':', 1)[0]] = value
    expected = dict(GGML_CUDA='ON', GGML_USE_PPU='ON', GGML_NCP_QUACTLIZE='ON',
        GGML_NCP_FA='ON', GGML_NCP_MOE='ON', GGML_NCP_GDN='OFF',
        CMAKE_CUDA_COMPILER=str(sdk / 'CUDA_SDK/bin/nvcc'))
    if any(cache.get(k) != v for k, v in expected.items()):
        raise ValueError('joint CI CMake profile differs from the requested build')
    binary = build / 'bin'
    required = ['llama-server', 'llama-batched-bench', 'llama-perplexity',
                'libncp_fa.so', 'libncp_moe.so', 'libggml-cuda.so']
    if not all((binary / name).is_file() for name in required):
        raise ValueError('joint CI build is missing model binaries or NCP libraries')
    headers = sorted((binary / 'deep_gemm/include').rglob('*'))
    if not (binary / 'deep_gemm/include/deep_gemm').is_dir() or not any(p.is_file() for p in headers):
        raise ValueError('joint CI build is missing the DeepGEMM JIT include tree')
    paths = {binary / name for name in required} | set(binary.glob('lib*.so*')) | set(headers)
    files = {}
    for path in sorted(paths):
        if path.is_dir():
            continue
        path.resolve(strict=True).relative_to(binary.resolve())
        files[str(path.relative_to(build))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(entry='.aoneci/scripts/build.sh', llama_source_commit=git(llama, 'rev-parse', 'HEAD'),
        ncp_source_commit=git(ncp, 'rev-parse', 'HEAD'), ncp_submodules=git(ncp, 'submodule', 'status', '--recursive'),
        requested_jobs=jobs, cmake=expected, files=files, build=str(build), device_admission='PENDING')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('llama', 'ncp', 'sdk', 'output', 'receipt'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--jobs', type=int, default=192)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError('jobs must be positive')
    sdk = args.sdk.resolve(strict=True)
    if not (sdk / 'CUDA_SDK/bin/nvcc').is_file():
        raise ValueError('SDK compiler is missing')
    llama_rev = git(args.llama, 'rev-parse', 'HEAD')
    pin = git(args.llama, 'show', llama_rev + ':.aoneci/NCP_LIB_VERSION')
    revisions = [line.split('#', 1)[0].strip() for line in pin.splitlines()]
    revisions = [line for line in revisions if line]
    if len(revisions) != 1 or not re.fullmatch(r'[0-9a-f]{40}', revisions[0]):
        raise ValueError('CI NCP pin is missing or ambiguous')
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    llama, ncp = output / 'llama', output / 'ncp_flash_lib'
    clone_checkout(args.llama, llama, llama_rev)
    clone_checkout(args.ncp, ncp, revisions[0])
    env = os.environ.copy()
    # All submodules are cloned locally; no global credential rewrite is needed.
    for key in ('USRNAME', 'TOKEN', 'DG_LIBRARY_ROOT'):
        env.pop(key, None)
    python_bin = output / 'python-bin'
    python_bin.mkdir()
    (python_bin / 'python').symlink_to(sys.executable)
    env.update(LLAMA_CI_DIR=str(llama), NCP_LIB_DIR=str(ncp), NCP_LIB_REV=revisions[0],
        PPU_NVCC=str(sdk / 'CUDA_SDK/bin/nvcc'), CUDA_HOME=str(sdk / 'CUDA_SDK'),
        JOBS=str(args.jobs), DG_JIT_CACHE_DIR=str(output / 'ncp-jit-cache'),
        PATH=str(python_bin) + os.pathsep + env['PATH'])
    print(f'KPACK_MODEL_CI source={llama_rev} ncp={revisions[0]} jobs={args.jobs} output={output}', flush=True)
    subprocess.run(['bash', str(llama / '.aoneci/scripts/build.sh')], cwd=llama, env=env, check=True)
    receipt = build_receipt(llama, ncp, sdk, args.jobs)
    if receipt['ncp_source_commit'] != revisions[0] or receipt['llama_source_commit'] != llama_rev:
        raise ValueError('CI build changed its pinned source')
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(f'KPACK_MODEL_CI COMPLETE build={receipt["build"]} receipt={args.receipt}', flush=True)


if __name__ == '__main__':
    main()

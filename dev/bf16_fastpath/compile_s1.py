#!/usr/bin/env python3
"""Compile and inspect one complete typed Q4 S1 reader family; no device run."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_ppu import moe_s1
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha


def build(sdk, output, n, k):
    sdk = sdk.resolve(strict=True)
    source = moe_s1.source_v2(n, k)
    includes = [ROOT / 'quactlize/execution', ROOT / 'quactlize/include',
                ROOT / 'third_party/actlize/include']
    # The quoted project dependency closure binds the actual typed readers,
    # validation and arrangement. System headers are bound by the SDK receipt.
    pending = [includes[0] / 'q4_s1_kernel.cuh', Path(moe_s1.__file__), Path(__file__)]
    inputs = set()
    while pending:
        path = pending.pop().resolve(strict=True)
        if path in inputs:
            continue
        inputs.add(path)
        for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(), re.M):
            matches = [p for p in [path.parent / name, *[d / name for d in includes]] if p.is_file()]
            if matches:
                pending.append(matches[0])
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)}
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cu = output / 'q4_s1.cu'
    cu.write_text(source)
    library = output / 'libq4_s1_bf16.so'
    commands = [
        [str(sdk / 'bin/hgcc'), *FLAGS, *[f'-I{p}' for p in includes],
         '-c', str(cu), '-o', str(output / 'q4_s1.o')],
        ['g++', '-shared', '-Wl,-Bsymbolic', str(output / 'q4_s1.o'),
         '-o', str(library), f'-L{sdk / "lib"}', *[f'-l{x}' for x in LIBRARIES]],
        [str(sdk / 'bin/hgobjdump'), '--dump-isa', str(library)],
    ]
    env = dict(os.environ)
    env['PATH'] = str(sdk / 'bin') + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(sdk / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    start = time.monotonic()
    for label, command in zip(('compile', 'link', 'isa'), commands):
        with (output / f'{label}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        print(f'BF16_Q4_S1_COMPILE phase={label}', flush=True)
    isa = (output / 'isa.log').read_text()
    sections = re.findall(r'Disassembly of section \.text\.kernel\.([^\n]+):', isa)
    f16 = {s for s in sections if 'indexed_kernelI' in s}
    bf16 = {s for s in sections if 'indexed_kernel_bf16I' in s}
    recipes = len(moe_s1.inventory())
    if len(f16) != 2 * recipes or len(bf16) != 3 * recipes:
        raise ValueError(f'kernel denominator differs: f16={len(f16)}, bf16={len(bf16)}')
    if 'v.fma.f32' not in isa:
        raise ValueError('missing FP32 dot instructions')
    exports = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
    for version in (1, 2):
        for operation in ('query', 'run'):
            if f' quactlize_q4_s1_{operation}_v{version}\n' not in exports:
                raise ValueError('missing typed S1 export')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('source changed during compilation')
    manifest = dict(schema='quactlize.bf16-q4-s1-compile.v1', shape=[n, k],
                    recipes=recipes, f16_kernels=len(f16), bf16_kernels=len(bf16),
                    library=library.name, sha256=sha(library), source_hashes=hashes,
                    generated_source_sha256=sha(cu), compiler_sha256=sha(sdk / 'bin/hgcc'),
                    inspector_sha256=sha(sdk / 'bin/hgobjdump'),
                    runtime={x: sha(sdk / 'lib' / f'lib{x}.so') for x in LIBRARIES},
                    commands=commands, seconds=time.monotonic() - start,
                    device_validated=False, performance_admitted=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('BF16_Q4_S1_COMPILE PASS f16=32 bf16=48 device_validated=0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--k', type=int, default=2048)
    args = parser.parse_args()
    build(args.sdk, args.output, args.n, args.k)

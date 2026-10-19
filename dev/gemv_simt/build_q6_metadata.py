#!/usr/bin/env python3
"""Compile two exact Q6 bodies per precision, no production library changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt.q6_metadata import source
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import FLAGS, LIBRARIES, HEADER_SUFFIXES, sha


def inputs():
    paths = {p for folder in ('quactlize/execution', 'quactlize/include',
                              'third_party/actlize/include', 'third_party/actlize/tools/util/include')
             for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in HEADER_SUFFIXES}
    for name in ('q6_metadata.py', 'q6_metadata.cuh', 'build_q6_metadata.py'):
        paths.add(ROOT/'dev/gemv_simt'/name)
    paths.update(ROOT/p for p in ('dev/gemv_simt/model_followup.py', 'quactlize/runtime/compiler.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def build(sdk, output, object_only=False, *, generator=source, source_inputs=inputs,
          schema='q6-metadata-v1', kernel_names=None):
    sdk = sdk.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    includes = [ROOT, ROOT/'quactlize/execution', ROOT/'quactlize/include',
                ROOT/'third_party/actlize/include', ROOT/'third_party/actlize/tools/util/include']
    if not (includes[-2]/'cute/tensor.hpp').is_file():
        raise ValueError('missing pinned actlize headers; initialize the local gitlink first')
    frozen = source_inputs()
    env = dict(os.environ, PATH=str(sdk/'bin')+':'+os.environ.get('PATH', ''),
               LD_LIBRARY_PATH=str(sdk/'lib')+':'+os.environ.get('LD_LIBRARY_PATH', ''))
    started = time.monotonic()

    def one(compute):
        src, obj, lib = (output/(compute+ext) for ext in ('.cu', '.o', '.so'))
        src.write_text(generator(compute))
        commands = [[str(sdk/'bin/hgcc'), *FLAGS, *[f'-I{p}' for p in includes],
                     '-c', str(src), '-o', str(obj)]]
        if not object_only:
            commands.append(['g++', '-shared', '-Wl,-Bsymbolic', '-Wl,-z,defs', str(obj),
                             f'-L{sdk}/lib', *[f'-l{x}' for x in LIBRARIES], '-o', str(lib)])
        atomic_json(output/(compute+'.commands.json'), commands)
        log_path = output/(compute+'.build.log')
        try:
            with log_path.open('x') as log:
                for command in commands:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(str(exc)+'\n'+log_path.read_text(errors='replace')[-6000:]) from exc
        target = obj if object_only else lib
        for option, ext in (('--dump-resource-usage=all', '.resources.txt'), ('--dump-isa', '.isa.txt')):
            with (output/(compute+ext)).open('x') as log:
                subprocess.run([sdk/'bin/hgobjdump', option, target], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        row=dict(compute=compute, image=target.name, sha256=sha(target), source_sha256=sha(src))
        if kernel_names is not None: row['kernel_names']=list(kernel_names)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(one, ('f16', 'bf16')))
    if source_inputs() != frozen:
        raise ValueError('source changed during compilation')
    receipt = dict(schema=schema, object_only=object_only, records=records,
                   compiler_sha256=sha(sdk/'bin/hgcc'), flags=FLAGS, sources=frozen,
                   seconds=time.monotonic()-started, device_admission='NOT_RUN',
                   production_changed=False)
    if not object_only:
        from quactlize.runtime.native import sdk_identity
        receipt['sdk'] = sdk_identity(sdk)
    atomic_json(output/'manifest.json', receipt)
    print('Q6_METADATA_BUILD '+json.dumps({k:v for k,v in receipt.items() if k!='sources'}), flush=True)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--object-only', action='store_true')
    a = p.parse_args()
    build(a.sdk, a.output.resolve(), a.object_only)

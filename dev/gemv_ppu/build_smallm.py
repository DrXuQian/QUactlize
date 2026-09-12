#!/usr/bin/env python3
"""Compile the six Q4 small-M readers and the current five-parent TC union."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu import smallm as spec
from quactlize.runtime.compiler import Compiler, FLAGS, LIBRARIES
from tools.build_kpack_dispatch import plan as policy_plan


def isa(text, n, k):
    sections = list(re.finditer(r'Disassembly of section \.text\.kernel\.[^\n]+:', text))
    rows = {}
    expected = {spec.template_args(c, n, k): c.key for c in spec.inventory(n, k)}
    for i, match in enumerate(sections):
        if 'q4_smallm_' not in match[0]:
            continue
        args = tuple(map(int, re.findall(r'(?:I|E)Li(\d+)', match[0])))
        if args not in expected or expected[args] in rows:
            raise ValueError('unexpected/duplicate small-M native specialization')
        body = text[match.end():sections[i+1].start() if i+1 < len(sections) else None]
        ops = Counter(re.findall(r'\t([a-z][\w.]+)\s', body))
        rows[expected[args]] = dict(symbol=match[0],
            code_fastpath_present=bool(ops['v.lop3.b32'] and any('f16x2' in op for op in ops)),
            fp32_fma_present=any(op.startswith('v.fma.f32') for op in ops),
            scope='STATIC_NATIVE_ISA_NOT_DYNAMIC_COUNTS',
            operations={op:v for op,v in ops.items() if op.startswith(('vmem.', 'tsm.', 's.cbr', 's.blksyn'))
                        or any(x in op for x in ('shuffle', 'f16x2', 'fma.f32', 'lop3'))})
    if set(rows) != set(expected.values()) or any(not v['code_fastpath_present'] or not v['fp32_fma_present'] for v in rows.values()):
        raise ValueError('missing SIMT native code/FP32 fast path')
    return rows


def build(a):
    sdk = a.sdk.resolve(strict=True)
    old = spec.medium_refine.verify(*(ROOT/'prebuilt/ppu0010'/p for p in spec.PACKAGES))
    if digest(sdk/'bin/hgcc') != old['compiler_sha256']:
        raise ValueError('the M1 control compiler is required')
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parents, selections = policy_plan(output, spec.requests())
    if any(s['status'] != 'SELECTED' for s in selections):
        raise ValueError('TC selector omitted a requested shape')
    for parent in parents:
        if parent['route'] != 'fq-dense' or parent['qtype'] != 12 or parent['ap'] != 0:
            raise ValueError('TC control must retain FP16 A and canonical Q4 FQ')
    sources = set(old['source_hashes']) | {
        'dev/gemv_ppu/smallm.py', 'dev/gemv_ppu/build_smallm.py',
        'tools/build_kpack_dispatch.py', 'tools/kpack_native_policy.cpp',
        'quactlize/runtime/compiler.py', 'quactlize/runtime/native.py',
        'quactlize/dispatch/policy.hpp', 'quactlize/dispatch/api.h'}
    sources.update(str(p.relative_to(ROOT)) for p in (ROOT/'quactlize/dispatch').rglob('*.hpp'))
    hashes = {name:digest(ROOT/name) for name in sorted(sources)}
    os.environ['PATH'] = str(sdk/'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = str(sdk/'lib') + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    includes = [output, ROOT/'dev/gemv_ppu', ROOT/'quactlize/execution', ROOT/'quactlize/include',
                ROOT/'benchmarks', ROOT/'third_party/actlize/include', ROOT/'third_party/actlize/tools/util/include']
    commands = []

    def run(label, cmd):
        cmd = list(map(str, cmd))
        commands.append(dict(label=label, argv=cmd))
        with (output/(label+'.log')).open('w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc:
            raise ValueError(f'{label} rc={rc}; see {output/(label+".log")}')
        print(f'Q4_SMALLM_BUILD phase={label} status=PASS elapsed_s={time.monotonic()-started:.1f}', flush=True)

    units = []
    for n, k in spec.SHAPES:
        label = f'n{n}_k{k}'
        path = output/(label+'.cu')
        path.write_text(spec.source(n, k))
        units.append((label, path, n, k))
    reference = output/'reference.cu'
    reference.write_text(spec.reference_source())
    units += [('reference', reference, 0, 0), ('probe', ROOT/'dev/gemv_ppu/probe.cu', 0, 0)]

    def compile_one(unit):
        label, path = unit[:2]
        run(label, [sdk/'bin/hgcc', *FLAGS, '-DQKG_QTYPE=12', *[f'-I{p}' for p in includes],
                    '-c', path, '-o', output/(label+'.o')])

    with ThreadPoolExecutor(max_workers=min(a.jobs, len(units))) as pool:
        list(pool.map(compile_one, units))
    payloads, stats = {}, {}
    for label, path, n, k in units[:-1]:
        library = output/(spec.payload(n,k) if n else 'libq4_smallm_reference.so')
        run('link-'+label, ['g++', '-shared', '-Wl,-Bsymbolic', output/(label+'.o'), output/'probe.o',
                           '-o', library, f'-L{sdk/"lib"}', *[f'-l{lib}' for lib in LIBRARIES]])
        run('isa-'+label, [sdk/'bin/hgobjdump', '--dump-isa', library])
        text = (output/('isa-'+label+'.log')).read_text()
        if 'q4_ppu_marker' not in text:
            raise ValueError('marker missing')
        if n:
            stats[f'{n}x{k}'] = isa(text, n, k)
        elif 'q4k_gemv_kernel' not in text:
            raise ValueError('raw reference kernel missing')
        payloads[library.name] = dict(sha256=digest(library), isa_sha256=digest(output/('isa-'+label+'.log')))
    compiler = Compiler(sdk, a.cache or output/'tc-cache', min(a.jobs, 5))
    records = compiler.compile_only(parents, progress=lambda *v: print('Q4_SMALLM_TC_BUILD', *v, flush=True))
    modules = []
    for record in records:
        target = output/'modules'/record['key']/'kernel.so'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record['path'], target)
        modules.append(record | dict(path=str(target.relative_to(output))))
    if any(digest(ROOT/name) != sha for name, sha in hashes.items()):
        raise ValueError('source changed during build')
    (output/'isa-stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    manifest = dict(schema=spec.SCHEMA, plan=spec.plan(), production_changed=False, device_validated=False,
        source_hashes=hashes, compiler_sha256=digest(sdk/'bin/hgcc'), inspector_sha256=digest(sdk/'bin/hgobjdump'),
        runtime={f'lib{lib}.so':digest(sdk/'lib'/f'lib{lib}.so') for lib in LIBRARIES},
        review_sha256=digest(spec.REVIEW), payloads=payloads, modules=modules, tc_selection=selections,
        isa_sha256=digest(output/'isa-stats.json'), commands=sorted(commands, key=lambda r:r['label']),
        seconds=time.monotonic()-started)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    spec.verify(output)
    print(f'Q4_SMALLM_BUILD status=COMPILED simt_contexts={sum(len(v) for v in stats.values())} tc_parents={len(modules)} seconds={manifest["seconds"]:.1f} device_validated=0', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--jobs', type=int, default=6)
    a = p.parse_args()
    if a.jobs < 1:
        p.error('jobs must be positive')
    build(a)

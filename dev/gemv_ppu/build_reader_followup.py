#!/usr/bin/env python3
"""Compile only the four-shape followup and its small-reader control."""
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
from dev.gemv_cuda.build import digest
from dev.gemv_ppu import reader_followup as spec
from dev.gemv_ppu import reader_reuse as frozen
from quactlize.runtime.compiler import FLAGS, LIBRARIES


def build(a):
    sdk = a.sdk.resolve(strict=True)
    old = frozen.verify(a.reuse, a.config_bundle, a.previous, a.controls, a.baseline)
    if digest(sdk/'bin/hgcc') != old['compiler_sha256']:
        raise ValueError('use the original control compiler')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = dict(old['source_hashes'])
    for name in ('dev/gemv_ppu/reader_followup.py', 'dev/gemv_ppu/build_reader_followup.py'):
        hashes[name] = digest(ROOT/name)
    units, generated = [], {}
    for n, k, family in [*((n, k, 'affine') for n, k in spec.SHAPES), (512, 2048, 'small')]:
        name = f'{family}_n{n}_k{k}'
        path = out/(name+'.cu')
        path.write_text(spec.small_source() if family == 'small' else spec.source(n, k))
        generated[path.name] = digest(path)
        units.append((name, path, n, k, family))
    env = dict(os.environ)
    env['PATH'] = str(sdk/'bin')+os.pathsep+env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(sdk/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
    includes = [out, ROOT/'dev/gemv_ppu', ROOT/'quactlize/execution', ROOT/'quactlize/include',
                ROOT/'benchmarks', ROOT/'third_party/actlize/include', ROOT/'third_party/actlize/tools/util/include']
    commands, started = [], time.monotonic()

    def run(label, cmd):
        cmd = list(map(str, cmd))
        commands.append(dict(label=label, argv=cmd))
        with (out/(label+'.log')).open('w') as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            raise ValueError(f'{label}: rc={rc} log={out/(label+".log")}')
        print(f'Q4_FOLLOWUP_BUILD phase={label} status=PASS elapsed_s={time.monotonic()-started:.1f}', flush=True)

    def compile_one(unit):
        name, path = unit[:2]
        run(name, [sdk/'bin/hgcc', *FLAGS, '-DQKG_QTYPE=12', *[f'-I{p}' for p in includes],
                   '-c', path, '-o', out/(name+'.o')])

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(compile_one, [*units, ('probe', ROOT/'dev/gemv_ppu/probe.cu')]))
    payloads, stats = {}, {}
    for name, _, n, k, family in units:
        library = out/spec.payload(n, k, family)
        run(name+'-link', ['g++', '-shared', '-Wl,-Bsymbolic', out/(name+'.o'), out/'probe.o', '-o', library,
                          f'-L{sdk/"lib"}', *[f'-l{lib}' for lib in LIBRARIES]])
        run(name+'-isa', [sdk/'bin/hgobjdump', '--dump-isa', library])
        isa = (out/(name+'-isa.log')).read_text()
        rows = spec.small_isa(isa) if family == 'small' else frozen.isa_statistics(isa)
        if 'q4_ppu_marker' not in isa:
            raise ValueError('missing native marker')
        for candidate in spec.inventory(n, k):
            if candidate.family != family:
                continue
            key = f'n{n}-k{k}-{candidate.key}'
            row = rows.get(candidate.key if family == 'small' else key)
            if not row or not row['code_fastpath_present'] or not row['fp32_fma_present']:
                raise ValueError('native specialization/fast dequant/FP32 FMA missing: '+key)
            stats[key] = row
        payloads[library.name] = dict(sha256=digest(library), isa_sha256=digest(out/(name+'-isa.log')))
    if any(digest(ROOT/name) != sha for name, sha in hashes.items()):
        raise ValueError('source changed during build')
    (out/'isa-stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    m = dict(schema=spec.SCHEMA, plan=spec.plan(), device_validated=False, production_changed=False,
        reuse_manifest_sha256=digest(a.reuse/'manifest.json'), reader_review_sha256=digest(spec.READER_REVIEW),
        compiler_sha256=digest(sdk/'bin/hgcc'), inspector_sha256=digest(sdk/'bin/hgobjdump'),
        runtime={f'lib{lib}.so': digest(sdk/'lib'/f'lib{lib}.so') for lib in LIBRARIES},
        source_hashes=hashes, generated=generated, payloads=payloads, isa_sha256=digest(out/'isa-stats.json'),
        commands=sorted(commands, key=lambda r:r['label']), seconds=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')
    spec.verify(out, a.reuse, a.config_bundle, a.previous, a.controls, a.baseline)
    print(f'Q4_FOLLOWUP_BUILD status=COMPILED candidates={len(stats)} device_validated=0 seconds={m["seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for name, folder in [('reuse','q4-reader-reuse-v1'), ('config-bundle','q4-config-sweep-v1'),
                         ('previous','q4-cold-shapes-v1'), ('controls','q4-h800-port-v1'), ('baseline','q4-simt-ab-v1')]:
        p.add_argument('--'+name, type=Path, default=ROOT/'prebuilt/ppu0010'/folder)
    build(p.parse_args())

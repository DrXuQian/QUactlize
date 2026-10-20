#!/usr/bin/env python3
"""Compare production Q8 instruction bytes with the measured bodies. No GPU work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt.general_plan import configs
from dev.gemv_simt.q8_address_source import source as measured_source, symbol_name
from dev.gemv_simt.q6_schedule_audit import instructions
from dev.strategy.journal import atomic_json
from quactlize.execution.q8_refresh import capability, rows
from quactlize.execution.simt_codegen import source as production_source
from quactlize.runtime.compiler import FLAGS, sha


def audit(measured, production, resources):
    old, new = instructions(measured), instructions(production)
    configs_by_columns = {c['columns']: i for i, c in enumerate(configs(8))}
    records = []
    for b, c, w in sorted({(r['implementation']['body'], r['config']['columns'],
                            r['config']['warps']) for r in rows()}):
        name = symbol_name(configs_by_columns[c], b)
        before, = [s for s in old if f'{len(name)}{name}E' in s]
        name = f'kernel_{b}_{c}_{w}'
        after, = [s for s in new if f'{len(name)}{name}E' in s]
        raw = lambda code: b''.join(r[1] for r in code)
        if raw(old[before]) != raw(new[after]):
            raise ValueError('measured/production native body differs: '+name)
        fields = resources.split(after+' RESOURCE INFO:', 1)[1].split('.ARGUMENT:', 1)[0]
        usage = {k: int(re.search(k+r':(\d+)', fields)[1]) for k in
                 ('vreg_number', 'sreg_number', 'shared_memory_size', 'STACK SIZE')}
        if usage['STACK SIZE'] or not any(r[2].startswith('v.fma.f32') for r in new[after]):
            raise ValueError('production Q8 spill or missing FP32 accumulator')
        records.append(dict(body=b, columns=c, warps=w, values=4, symbol=after,
                            instruction_sha256=hashlib.sha256(raw(new[after])).hexdigest(),
                            resources=usage, measured_native_bytes_equal=True))
    return records


def build(sdk, output):
    sdk = sdk.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PATH=str(sdk/'bin')+':'+os.environ.get('PATH', ''),
               LD_LIBRARY_PATH=str(sdk/'lib')+':'+os.environ.get('LD_LIBRARY_PATH', ''))
    includes = [ROOT, ROOT/'quactlize/execution', ROOT/'quactlize/include',
                ROOT/'third_party/actlize/include', ROOT/'third_party/actlize/tools/util/include']
    started = time.monotonic()
    frozen = capability()
    from dev.gemv_simt.build_general import inputs
    source_hashes = inputs()
    source_hashes.update({str(p.relative_to(ROOT)): sha(p) for p in
                         (Path(__file__), ROOT/'quactlize/execution/simt_codegen.py')})

    def one(item):
        name, text = item
        src, obj = output/(name+'.cu'), output/(name+'.o')
        src.write_text(text)
        command = [str(sdk/'bin/hgcc'), *FLAGS, *['-I'+str(p) for p in includes],
                   '-c', str(src), '-o', str(obj)]
        atomic_json(output/(name+'.command.json'), command)
        with (output/(name+'.build.log')).open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        for option, ext in [('--dump-isa', '.isa.txt'), ('--dump-resource-usage=all', '.resources.txt')]:
            with (output/(name+ext)).open('x') as log:
                subprocess.run([sdk/'bin/hgobjdump', option, obj], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        return dict(name=name, object_sha256=sha(obj), source_sha256=sha(src))

    with ThreadPoolExecutor(max_workers=2) as pool:
        images = list(pool.map(one, [('measured', measured_source(8, 'f16', configs(8))),
                                    ('production', production_source(8))]))
    if capability() != frozen or any(sha(ROOT/p) != h for p, h in source_hashes.items()):
        raise ValueError('Q8 inputs changed during compilation')
    records = audit((output/'measured.isa.txt').read_text(),
                    (output/'production.isa.txt').read_text(),
                    (output/'production.resources.txt').read_text())
    result = dict(schema='quactlize.q8-refresh-native.v1', images=images, audit=records,
                  capability=frozen, sources=source_hashes, flags=FLAGS,
                  compiler_sha256=sha(sdk/'bin/hgcc'), seconds=time.monotonic()-started,
                  kernel_launches=0, device_validated=False,
                  scope='LOCAL_SAME_COMPILER_CODE_IDENTITY_NOT_MODEL_PERFORMANCE')
    atomic_json(output/'manifest.json', result)
    print('Q8_REFRESH_NATIVE '+json.dumps(dict(status='PASS', bodies=len(records),
          keys=len(rows()), seconds=result['seconds'], kernel_launches=0)), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.sdk, a.output.resolve())

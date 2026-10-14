"""Small adapter/indexed-expansion carrier, separate from all TC modules."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import Compiler, FLAGS, LIBRARIES, sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('dev/strategy/prefill_helpers.cu', 'quactlize/dequant/indexed.cu')


def identity(sdk):
    compiler = Compiler(sdk, Path(sdk)/'unused-strategy-identity')
    return dict(compiler=compiler.identity, sources={p: sha(ROOT/p) for p in SOURCES},
                adapters={str(p.relative_to(ROOT)): sha(p)
                    for folder in ('quactlize/dense', 'quactlize/prefill', 'quactlize/dequant')
                    for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.h', '.hpp', '.cuh', '.cu')})


def commands(sdk, output):
    includes = [ROOT, ROOT/'quactlize/include', ROOT/'third_party/actlize/include']
    return [[str(sdk/'bin/hgcc'), *FLAGS, *['-I'+str(p) for p in includes],
             '-c', str(ROOT/source), '-o', str(output/(Path(source).stem+'.o'))] for source in SOURCES]


def verify(sdk, output):
    row = json.loads((output/'manifest.json').read_text())
    if (row['identity'] != identity(sdk) or row['sha256'] != sha(output/'libstrategy_prefill.so') or
            row.get('record_sha256') != digest({k: v for k, v in row.items() if k != 'record_sha256'})):
        raise ValueError('prefill helper source/SDK/image differs')
    return row


def build(sdk, output):
    sdk, output = Path(sdk).resolve(strict=True), Path(output).resolve()
    if (output/'manifest.json').exists():
        return verify(sdk, output)
    output.mkdir(parents=True, exist_ok=False)
    before = identity(sdk)
    env = dict(os.environ, PATH=str(sdk/'bin')+':'+os.environ.get('PATH', ''),
               LD_LIBRARY_PATH=str(sdk/'lib')+':'+os.environ.get('LD_LIBRARY_PATH', ''))
    start = time.monotonic()
    def run(item):
        name, command = item
        with (output/(name+'.log')).open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    jobs = commands(sdk, output)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [(Path(p).stem, c) for p, c in zip(SOURCES, jobs)]))
    link = ['g++', '-shared', '-Wl,-Bsymbolic', '-Wl,-z,defs',
        *[str(output/(Path(p).stem+'.o')) for p in SOURCES], '-L'+str(sdk/'lib'),
        *['-l'+n for n in LIBRARIES], '-o', str(output/'libstrategy_prefill.so')]
    run(('link', link))
    if before != identity(sdk):
        raise ValueError('prefill helper sources changed during build')
    row = dict(identity=before, sha256=sha(output/'libstrategy_prefill.so'), commands=[*jobs, link],
               seconds=time.monotonic()-start, device_admission='PENDING', production_changed=False)
    row['record_sha256'] = digest(row)
    atomic_json(output/'manifest.json', row)
    return row

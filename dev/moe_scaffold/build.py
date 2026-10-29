"""Two device TUs and host selection; no llama, actlize, TC or JIT sweep."""
import os
from pathlib import Path
import subprocess
import time

from dev.moe_route.build import revision
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha

ROOT = Path(__file__).resolve().parents[2]


def build(sdk, output):
    sdk = Path(sdk).resolve(strict=True); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(p.relative_to(ROOT)): sha(p) for directory in
        ('quactlize/moe', 'quactlize/routing', 'dev/moe_scaffold', 'dev/moe_route')
        for p in (ROOT/directory).glob('*') if p.is_file()}
    for name in ('tools/run_moe_scaffold.py', 'dev/prefill_asys/trace.py', 'quactlize/runtime/compiler.py',
                 'dev/strategy/refresh_clock.py', 'dev/strategy/journal.py'):
        sources[name] = sha(ROOT/name)
    target = output/'libmoe-scaffold.so'
    commands = [[str(sdk/'bin/hgcc'), *FLAGS, '-DQZR_TESTING', '-DQZM_TESTING', '-c',
        str(ROOT/source), '-o', str(output/(Path(source).stem+'.o'))]
        for source in ('quactlize/routing/route.cu', 'quactlize/moe/scaffold.cu')]
    commands.append(['g++','-std=c++17','-O2','-fPIC','-c',str(ROOT/'quactlize/moe/selection.cpp'),
                     '-o',str(output/'selection.o')])
    commands.append(['g++', '-shared', '-Wl,-z,defs', '-Wl,-Bsymbolic',
        str(output/'route.o'), str(output/'scaffold.o'), str(output/'selection.o'), '-L'+str(sdk/'lib'),
        '-Wl,-rpath,'+str(sdk/'lib'), *['-l'+name for name in LIBRARIES], '-o', str(target)])
    start = time.monotonic()
    env = dict(os.environ, PATH=str(sdk/'bin')+':'+os.environ.get('PATH', ''))
    for i, command in enumerate(commands):
        log = output/f'build-{i}.log'
        with log.open('w') as stream:
            rc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc: raise RuntimeError(f'helper build rc={rc}: {log}\n'+log.read_text(errors='replace')[-8000:])
    if any(sha(ROOT/p) != value for p, value in sources.items()): raise ValueError('build sources changed')
    result = dict(quactlize=revision(ROOT), sources=sources, commands=commands, candidate=str(target),
        candidate_sha256=sha(target), compiler_sha256=sha(sdk/'bin/hgcc'),
        libraries={str((sdk/'lib'/('lib'+name+'.so')).resolve()): sha(sdk/'lib'/('lib'+name+'.so')) for name in LIBRARIES},
        sdk=str(sdk), seconds=time.monotonic()-start, production_changed=False)
    atomic_json(output/'receipt.json', result)
    return result


def verify(receipt):
    for path, value in receipt['sources'].items():
        if sha(ROOT/path) != value: raise ValueError('helper source changed: '+path)
    for path, value in receipt['libraries'].items():
        if sha(path) != value: raise ValueError('runtime changed: '+path)
    if sha(receipt['candidate']) != receipt['candidate_sha256']: raise ValueError('helper binary changed')

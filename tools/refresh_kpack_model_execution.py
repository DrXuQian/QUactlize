#!/usr/bin/env python3
"""Refresh a small dispatcher/execution pair while reusing unchanged TC images."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.bf16_compute.package import execution_receipt
from quactlize.decode.compiler import DecodeCompiler
from quactlize.decode.grouped_compiler import GroupedComputeCompiler
from quactlize.runtime.compiler import Compiler, sha, source_contract
from quactlize.runtime.tuning import digest
from tools.kpack_jit import module_source
from tools.build_kpack_dispatch import catalog
from tools.build_kpack_model_package import attach_q4_bf16
from tools.verify_kpack_dispatch import verify, pack_paths, prefill_paths, router_alias_paths


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def dispatcher_refresh_scope(changed, historical_caller=False):
    scopes = {
        ('quactlize/dispatch/policy.hpp',): 'DENSE_N_EXTENSION_PREDICTED',
        ('quactlize/dispatch/moe.hpp',): 'M1_ROUTER_SNAPSHOT_ALIAS_ADMISSION',
    }
    scope = scopes.get(tuple(sorted(changed)))
    if historical_caller or scope is None:
        raise ValueError('dispatcher-only refresh requires one reviewed host-only change')
    return scope


def refresh_dispatcher(base, out, sdk, alias_gate=None):
    old = verify(base, sdk=sdk)
    changed = sorted(name for name, value in old['policy_hashes'].items() if sha(ROOT/name) != value)
    scope = dispatcher_refresh_scope(changed, 'model' in old)
    if alias_gate is not None:
        alias_gate = alias_gate.resolve(strict=True)
        gate = json.loads((alias_gate/'manifest.json').read_text())
        if (scope != 'M1_ROUTER_SNAPSHOT_ALIAS_ADMISSION' or 'router_alias_gate' in old or
                gate.get('platform') != 'ppu' or sha(alias_gate/'bench') != gate['binary_sha256'] or
                any(sha(ROOT/name) != value for name,value in gate['source_hashes'].items())):
            raise ValueError('router alias gate source/build differs')
    if any(sha(ROOT/name) != value for name,value in old['execution_receipt']['source_hashes'].items()):
        raise ValueError('dispatcher-only refresh cannot relabel changed GPU source')
    if any(p.is_symlink() or (not p.is_dir() and not p.is_file()) for p in base.rglob('*')):
        raise ValueError('source package contains links or special files')
    shutil.copytree(base, out)
    (out/'catalog.inc').write_text(catalog(old['modules'], old['jit_source_contract']))
    command = ['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',
        '-I'+str(out),str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl',
        '-o',str(out/'libquactlize_kpack_dispatch.so')]
    subprocess.run(command, check=True)
    old['dispatch_sha256'] = sha(out/'libquactlize_kpack_dispatch.so')
    old['host_command'] = command
    for name in changed:
        old['policy_hashes'][name] = sha(ROOT/name)
    old['dispatcher_refresh'] = dict(base_manifest_sha256=sha(base/'manifest.json'),
        changed_policy_inputs=changed, scope=scope, gpu_compilations=0)
    old['device_validated'] = False
    if alias_gate is not None:
        (out/'router-alias').mkdir()
        for name in ('manifest.json','bench'):
            shutil.copy2(alias_gate/name,out/'router-alias'/name)
        old['router_alias_gate'] = dict(path='router-alias/manifest.json',
            sha256=sha(out/'router-alias/manifest.json'),binary_sha256=sha(out/'router-alias/bench'),
            cases=360,device_validated=False)
        router_alias_paths(out,old['router_alias_gate'])
    save(out/'manifest.json', old)
    verify(out, sdk=sdk)
    for path in base.rglob('*'):
        if path.is_file() and path.name not in ('libquactlize_kpack_dispatch.so','manifest.json'):
            if sha(path) != sha(out/path.relative_to(base)):
                raise ValueError('dispatcher refresh changed another payload: '+str(path))
    print(f'MODEL_DISPATCH_REFRESH PASS GPU_IMAGES=UNCHANGED output={out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','base','output'):
        p.add_argument('--'+name, type=Path, required=True)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--execution', type=Path)
    action.add_argument('--dispatcher-only', action='store_true')
    p.add_argument('--router-alias-gate', type=Path, help='attach the bounded M1 alias proof to a host-only refresh')
    a = p.parse_args()
    base, out = a.base.resolve(strict=True), a.output.resolve()
    if out.exists():
        raise ValueError('use a fresh output directory')
    if a.dispatcher_only:
        refresh_dispatcher(base, out, a.sdk, a.router_alias_gate)
        return
    if a.router_alias_gate:
        raise ValueError('--router-alias-gate requires --dispatcher-only')
    # Old execution sources intentionally differ. Validate only the reused
    # payloads here; the assembled package gets full current-source checks.
    old = json.loads((base/'manifest.json').read_text())
    if old.get('schema') != 'quactlize.kpack-native-dispatch.v1':
        raise ValueError('invalid source model package')
    for record in old['modules']:
        if (sha(base/record['path']) != record['sha256'] or
                digest(dict(identity=record['identity'], parent=record['parent'],
                            source=module_source(record))) != record['key']):
            raise ValueError('source TC payload/identity differs')
    if sha(base/'mixed-stages') != old['moe_mixed_gate']['simt_binaries'][0]['sha256']:
        raise ValueError('mixed stage control differs')
    library, execution = execution_receipt(a.execution)
    compiler = Compiler(a.sdk, out/'unused-cache')
    if source_contract(compiler.identity) != old['jit_source_contract']:
        raise ValueError('TC source changed; this operation cannot relabel old images')
    gate = json.loads((base/'bf16/manifest.json').read_text())
    for record in gate['modules'].values():
        if sha(base/'bf16'/record['path']) != record['sha256']:
            raise ValueError('BF16 gate TC payload differs')
        cls = GroupedComputeCompiler if record['parent']['route'].endswith('grouped') else DecodeCompiler
        current = cls(a.sdk, out/'unused-cache', compute_type=record['spec']['compute']).identity
        if current != record['identity']:
            raise ValueError('BF16 gate TC module identity changed')
    subprocess.run([sys.executable, ROOT/'tools/build_kpack_dispatch.py', '--sdk', a.sdk,
                    '--output', out, '--execution-bundle', a.execution, '--jit-only'], check=True)
    m = json.loads((out/'manifest.json').read_text())

    def copy(name):
        source = base/name
        if source.is_symlink() or not source.resolve(strict=True).is_relative_to(base):
            raise ValueError('payload escapes its source package: '+name)
        target = out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for record in old['modules']:
        copy(record['path'])
    copy('mixed-stages')
    for name in pack_paths(base, old['pack']) + prefill_paths(base, old['prefill'], sdk=a.sdk):
        copy(name)
    for field in ('modules','compiler_identities','moe_mixed_gate','pack','prefill'):
        m[field] = old[field]
    (out/'catalog.inc').write_text(catalog(m['modules'], m['jit_source_contract']))
    cmd = ['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',
           '-I'+str(out),str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl',
           '-o',str(out/'libquactlize_kpack_dispatch.so')]
    subprocess.run(cmd, check=True)
    m['host_command'], m['dispatch_sha256'] = cmd, sha(out/'libquactlize_kpack_dispatch.so')
    m['reused_packages'] = [dict(path=str(base), manifest_sha256=sha(base/'manifest.json'),
                                scope='UNCHANGED_TC_PREFILL_PACK_AND_STAGE_CONTROL')]
    # The gate now calls the new execution image. Its TC images and oracle
    # stay byte-identical; no previous device verdict is carried forward.
    (out/'bf16').mkdir()
    for record in gate['modules'].values():
        copy('bf16/'+record['path'])
    shutil.copy2(library, out/'bf16'/library.name)
    gate['simt'] = gate['moe'] = dict(path=library.name, sha256=sha(library))
    gate['reused_execution'] = execution
    gate['device_validated'] = False
    for field in ('harness_repair','oracle_followup'):
        gate.pop(field, None)
    save(out/'bf16/manifest.json', gate)
    m['bf16_gate'] = dict(path='bf16/manifest.json', sha256=sha(out/'bf16/manifest.json'),
                         cases=len(gate['cases']), modules=len(gate['modules']), device_validated=False)
    save(out/'manifest.json', m)
    q4 = out/'q4-gate-build'
    subprocess.run([sys.executable, ROOT/'dev/bf16_fastpath/build_gate.py', '--platform','ppu',
                    '--sdk',a.sdk,'--execution',a.execution,'--output',q4], check=True)
    attach_q4_bf16(out, q4, a.sdk)
    verify(out, sdk=a.sdk)
    print(f'MODEL_EXECUTION_REFRESH PASS modules={len(m["modules"])} TC_DEVICE_COMPILATIONS=0 output={out}')


if __name__ == '__main__':
    main()

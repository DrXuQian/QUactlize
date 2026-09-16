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
from tools.verify_kpack_dispatch import verify, pack_paths, prefill_paths


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','base','execution','output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    base, out = a.base.resolve(strict=True), a.output.resolve()
    if out.exists():
        raise ValueError('use a fresh output directory')
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

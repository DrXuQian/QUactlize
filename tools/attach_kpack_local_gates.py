#!/usr/bin/env python3
"""Attach two isolated reader/prepare experiments; never arm production defaults."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha


def payload_paths(root, receipt, *, sdk=None, check_source=False):
    root=Path(root).resolve(strict=True)
    if receipt.get('path')!='local-gates/manifest.json' or sha(root/receipt['path'])!=receipt.get('sha256'):
        raise ValueError('local gate manifest identity differs')
    gate=json.loads((root/receipt['path']).read_text())
    if (gate.get('schema')!='quactlize.local-decode-experiments.v1' or
            gate.get('production_selection_changed') is not False):
        raise ValueError('not an isolated local optimization gate')
    expected={'q8/q8.so','q8/manifest.json','moe/bench','moe/manifest.json'}
    if set(gate.get('files',{}))!=expected:raise ValueError('local gate payload set differs')
    for name,want in gate['files'].items():
        path=root/'local-gates'/name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root/'local-gates') or sha(path)!=want:
            raise ValueError('local gate payload differs: '+name)
    if sdk is not None:
        for name,want in gate['runtime'].items():
            if sha(Path(sdk)/'lib'/name)!=want:raise ValueError('local gate SDK runtime differs: '+name)
    if check_source:
        for name,want in gate['harness'].items():
            if sha(ROOT/name)!=want:raise ValueError('local gate harness differs: '+name)
    return ['local-gates/manifest.json',*['local-gates/'+n for n in sorted(expected)]]


def attach(bundle, q8, moe, sdk):
    from tools.verify_kpack_dispatch import verify
    bundle=bundle.resolve(strict=True);model=verify(bundle,sdk=sdk)
    if 'local_optimization_gate' in model:raise ValueError('local gate is already attached')
    q=json.loads((q8/'manifest.json').read_text());m=json.loads((moe/'manifest.json').read_text())
    if (q.get('schema')!='quactlize.q8-vector.v1' or q.get('platform')!='ppu' or
            m.get('platform')!='ppu' or sha(q8/'q8.so')!=q['library_sha256'] or
            sha(moe/'bench')!=m['binary_sha256']):
        raise ValueError('experimental PPU build differs')
    for build in (q,m):
        for name,want in build['source_hashes'].items():
            if sha(ROOT/name)!=want:raise ValueError('experimental build source changed: '+name)
    target=bundle/'local-gates';target.mkdir()
    files={}
    for label,source,binary in (('q8',q8,'q8.so'),('moe',moe,'bench')):
        (target/label).mkdir()
        for name in ('manifest.json',binary):
            dest=target/label/name;shutil.copy2(source/name,dest)
            files[f'{label}/{name}']=sha(dest)
    harness=('dev/gemv_simt/q8_vector_run.py','dev/gemv_simt/q8_vector_campaign.py',
             'dev/gemv_simt/q8_vector_access.py','dev/gemv_simt/run.py','dev/gemv_simt/native.py',
             'dev/gemv_simt/fixture.py','dev/bf16_compute/fixture.py','dev/moe_prepare/run.py')
    gate=dict(schema='quactlize.local-decode-experiments.v1',files=files,
        runtime={'libhggc_wrapper.so':sha(sdk/'lib/libhggc_wrapper.so')},
        harness={n:sha(ROOT/n) for n in harness},production_selection_changed=False,
        expected=dict(q8_numeric=12480,q8_performance=50,moe_numeric=3840,moe_performance=48),
        admission='PPU_PENDING',baseline_control_warning='NVIDIA_M1_F16_TC_GATHER_FAULT_NOT_A_PROVED_PPU_BUG')
    (target/'manifest.json').write_text(json.dumps(gate,indent=2)+'\n')
    model['local_optimization_gate']=dict(path='local-gates/manifest.json',sha256=sha(target/'manifest.json'))
    payload_paths(bundle,model['local_optimization_gate'],sdk=sdk,check_source=True)
    (bundle/'manifest.json').write_text(json.dumps(model,indent=2)+'\n')
    verify(bundle,sdk=sdk)
    print('LOCAL_GATES_ATTACHED Q8=50 MOE=48 production_changes=0 PPU=PENDING')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','q8','moe','sdk'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();attach(a.bundle,a.q8,a.moe,a.sdk)

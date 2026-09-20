#!/usr/bin/env python3
"""Add the scoped paired-N4 runtime; preserve every existing GPU payload."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha
from tools.build_kpack_dispatch import catalog
from tools.run_kpack_gate_up import verify as verify_fusion
from tools.verify_kpack_dispatch import verify, paired_paths
from tools import paired_decode_selection


def attach(base, fusion, output, sdk):
    base=base.resolve(strict=True);fusion=fusion.resolve(strict=True);output=output.resolve()
    if output.exists(): raise ValueError('use a fresh output directory')
    # Replaced fusion sources intentionally differ; reused images keep their
    # original receipts and are checked again in the complete new package.
    old=verify(base)
    library,build=verify_fusion(fusion,sdk)
    changes={name for name,h in old['policy_hashes'].items() if sha(ROOT/name)!=h}
    allowed={'quactlize/dispatch/binding.cpp','quactlize/dispatch/api.h',
             'quactlize/fusion/gate_up.h','quactlize/fusion/validation.hpp','tools/build_kpack_dispatch.py'}
    if not changes<=allowed: raise ValueError('unexpected host policy changes: '+str(sorted(changes)))
    if any(sha(ROOT/name)!=h for name,h in old['execution_receipt']['source_hashes'].items()):
        raise ValueError('existing execution GPU source changed')
    if any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in base.rglob('*')):
        raise ValueError('source bundle contains a link or special file')
    shutil.copytree(base,output)
    previous=old.get('paired_gate_up')
    if previous:
        control=output/'paired-control';control.mkdir()
        shutil.copy2(base/previous['library'],control/previous['library'])
        shutil.copy2(base/previous['receipt'],control/'manifest.json')
    (output/'catalog.inc').write_text(catalog(old['modules'],old['jit_source_contract']))
    command=['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',
             '-I'+str(output),str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl',
             '-o',str(output/'libquactlize_kpack_dispatch.so')]
    subprocess.run(command,check=True)
    shutil.copy2(library,output/library.name)
    shutil.copy2(fusion/'manifest.json',output/'gate-up-runtime.json')
    old['dispatch_sha256']=sha(output/'libquactlize_kpack_dispatch.so')
    old['host_command']=command
    for name in allowed: old['policy_hashes'][name]=sha(ROOT/name)
    for name in ('quactlize/fusion/selection.hpp','quactlize/fusion/recipes.inc',
                 'policies/kpack_gate_up_selection_v2.json'):
        old['policy_hashes'][name]=sha(ROOT/name)
    selection=json.loads(paired_decode_selection.EVIDENCE.read_text())
    if paired_decode_selection.header(selection)!=paired_decode_selection.OUTPUT.read_text():
        raise ValueError('paired generated selection is stale')
    if build['source_hashes'].get('quactlize/fusion/recipes.inc')!=sha(paired_decode_selection.OUTPUT):
        raise ValueError('fusion was built with another recipe catalog')
    old['paired_gate_up']=dict(schema='quactlize.paired-model.v2',library=library.name,
        sha256=sha(library),receipt='gate-up-runtime.json',receipt_sha256=sha(fusion/'manifest.json'),
        layout_id=build['layout_id'],q8_shared='F16_N512_K2048_T1_8_N1024_K3072_T1',
        q4_routed='GENERATED_PAIRED_SELECTION_V2',canonical_retained=True,device_validated=False,
        selection=selection,cost_authority=selection['archive_sha256'])
    if previous:
        old['paired_gate_up']['control']=dict(path='paired-control/'+previous['library'],
            sha256=previous['sha256'],receipt='paired-control/manifest.json',
            receipt_sha256=previous['receipt_sha256'])
    old['paired_gate_up']['additional_cost_authority']=sha(ROOT/'policies/kpack_decode_winners_v1.json')
    old['dispatcher_refresh']=dict(base_manifest_sha256=sha(base/'manifest.json'),
        scope='PAIRED_RECIPE_V2_MEASURED_CHAIN',changed_policy_inputs=sorted(changes),
        existing_gpu_payloads='BYTE_IDENTICAL',new_library=library.name)
    old['device_validated']=False
    (output/'manifest.json').write_text(json.dumps(old,indent=2)+'\n')
    paired_paths(output,old['paired_gate_up'],sdk=sdk)
    verify(output,sdk=sdk)
    for p in base.rglob('*'):
        if p.is_file() and p.relative_to(base).as_posix() not in (
            'libquactlize_kpack_dispatch.so','manifest.json','libquactlize_ppu_gate_up.so','gate-up-runtime.json','catalog.inc'):
            if sha(p)!=sha(output/p.relative_to(base)): raise ValueError('changed reused payload: '+str(p))
    print('GATE_UP_MODEL_PACKAGE PASS existing_gpu_images=UNCHANGED new_fusion=1 caller_binaries=0 output='+str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','fusion','output','sdk'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();attach(a.base,a.fusion,a.output,a.sdk)

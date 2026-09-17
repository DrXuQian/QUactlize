#!/usr/bin/env python3
"""Require actual paired shared/routed kernels in the warmed Qwen3.5 trace."""
import argparse
import json
from pathlib import Path


def check(results):
    paths=sorted((results/'trace').rglob('native/proof.json'))
    if not paths:raise ValueError('no native Asys proof')
    rows=[]
    for path in paths:
        proof=json.loads(path.read_text())
        if (set(proof.get('paired_observed_ops',[]))!={'dense','grouped'} or
                proof.get('paired_missing_ops') or proof.get('kernel_execution')!='PASS_SHORT_REQUEST' or
                proof.get('capture_scope')!='SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED'):
            raise ValueError('shared/routed paired execution not proven: '+str(path))
        rows.append(dict(path=str(path),paired_ops=proof['paired_observed_ops'],
                         capture_scope=proof['capture_scope']))
    (results/'paired-model-proof.json').write_text(json.dumps(dict(status='PASS',traces=rows),indent=2)+'\n')
    print('KPACK_PAIRED_MODEL PASS shared=DEVICE_OBSERVED routed=DEVICE_OBSERVED first_use=EXCLUDED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    check(p.parse_args().results)

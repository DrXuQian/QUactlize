#!/usr/bin/env python3
"""Require actual paired shared/routed kernels in the warmed Qwen3.5 trace."""
import argparse
import json
from pathlib import Path


def check(results, selected_scope=False):
    paths=sorted((results/'trace').rglob('native/proof.json'))
    if not paths:raise ValueError('no native Asys proof')
    rows=[]
    for path in paths:
        proof=json.loads(path.read_text())
        expected={'dense','grouped'}
        if selected_scope:
            selection=proof.get('selection',{})
            expected={row['op'] for row in selection.get('paired_plans',[])}
            if (selection.get('fully_selected') is not True or
                    not proof.get('expected_ops') or
                    set(proof.get('observed_ops',[]))!=set(proof['expected_ops']) or
                    proof.get('missing_ops') or not expected<=set(proof['expected_ops'])):
                raise ValueError('new model selected compute not proven: '+str(path))
        if (set(proof.get('paired_observed_ops',[]))!=expected or
                proof.get('paired_missing_ops') or proof.get('kernel_execution')!='PASS_SHORT_REQUEST' or
                proof.get('capture_scope')!='SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED'):
            raise ValueError('shared/routed paired execution not proven: '+str(path))
        rows.append(dict(path=str(path),paired_ops=proof['paired_observed_ops'],
                         paired_status='DEVICE_OBSERVED' if expected else 'NOT_SELECTED',
                         capture_scope=proof['capture_scope']))
    (results/'paired-model-proof.json').write_text(json.dumps(dict(status='PASS',traces=rows),indent=2)+'\n')
    print('KPACK_PAIRED_MODEL PASS scope='+('SELECTED_MODEL_OPERATIONS' if selected_scope else 'SHARED_AND_ROUTED')+
          ' first_use=EXCLUDED details=paired-model-proof.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--selected-scope',action='store_true',help='require each selected fusion, without assuming every model has MoE/shared experts')
    a=p.parse_args();check(a.results,a.selected_scope)

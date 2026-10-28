"""Build-time MoE exact admission; runtime reads the existing flat catalog.

Router labels are evidence only. Do not change bucket donors or enable the
experimental pair reader through a geometry that denotes a shipping reader.
"""
import copy
import json
import math
from pathlib import Path
import statistics

from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[2]
TABLE=ROOT/'policies/kpack_moe_decode_exact_v1.json'
ARCHIVE='709144bef858b1cef681f0e559285a43f5edadfeb638c7c2639f9ad2048d6a82'
COMMIT='98f5cd4b94922653f928555d96c54a9abeb9cba1'
EVIDENCE='35f20d9c9d8bce30d3e0b594d22dd6e8e1407753a8081901ae70e7e5d345e725'


def verify(data):
    if (data.get('schema')!='quactlize.moe-decode-exact.v1' or data.get('archive_sha256')!=ARCHIVE or
            data.get('evidence_sha256')!=EVIDENCE or
            data.get('evidence_sha256')!=digest({k:v for k,v in data.items() if k!='evidence_sha256'}) or
            data['source_revision']['quactlize_commit']!=COMMIT or
            data['timing_method']!='ASYS_GPU_ACTIVITY_MEAN_COMPONENT_SUM' or
            data['runtime_router_readback'] or data['bucket_changes'] or data['model_tpot_admitted']):
        raise ValueError('MoE exact evidence identity/scope differs')
    keys=[r['key'] for r in data['rows']]
    if keys!=[[13,2,2048,512,256,8,8,m,1] for m in (1,4,8)]:
        raise ValueError('MoE exact measured denominator differs')
    if len(data['body_hashes'])<5 or any(sha(ROOT/p)!=h for p,h in data['body_hashes'].items()):
        raise ValueError('MoE exact measured body changed; renew device evidence')
    for row in data['rows']:
        c=row['config']
        if (set(c)!={'kind','variant','columns','warps','values','split'} or c['kind']!='simt' or
                c['split']!=1 or c['variant']!=3 or
                (c['columns'],c['warps'],c['values']) not in ((4,2,8),(8,4,4))):
            raise ValueError('MoE exact recipe is not a supported single-row reader')
        profiles=row['profiles']
        expected={1:['spread'],4:['repeat2'],8:['cluster','spread']}[row['key'][7]]
        if [p['router'] for p in profiles]!=expected:
            raise ValueError('MoE exact router coverage differs')
        if row['changed']!=(row['config']!=row['original_config']):
            raise ValueError('MoE exact changed flag differs')
        for p in profiles:
            for label in ('original','selected'):
                value=p[label];samples=value['samples_us']
                if (len(samples)!=6 or any(len(r)!=5 for r in samples) or
                        any(not math.isfinite(t) or t<=0 for r in samples for t in r) or
                        not math.isclose(statistics.fmean(t for r in samples for t in r),value['mean_us'],rel_tol=1e-12) or
                        not 0<=value['max_relative_l1']<.005):
                    raise ValueError('MoE exact component samples/numerics differ')
            if p['selected']['mean_us']>p['original']['mean_us']:
                raise ValueError('MoE exact replacement regresses a measured router')
            if row['changed'] and p['selected']['producer']!='quactlize::execution::simt::common_reader':
                raise ValueError('MoE exact replacement is not the measured common reader')


def load():
    data=json.loads(TABLE.read_bytes());verify(data);return data


def apply(baseline):
    data=load();result=copy.deepcopy(baseline)
    exact={tuple(r['key']):r for r in result['exact']}
    for measured in data['rows']:
        row=exact.get(tuple(measured['key']))
        if row is None or row['config']!=measured['original_config'] or row.get('dequant_config',-1)!=-1:
            raise ValueError('MoE exact incumbent changed; review the transition')
        if not measured['changed']:continue
        row['bucket_config']=copy.deepcopy(row.get('bucket_config',row['config']))
        row['bucket_dequant_config']=row.get('bucket_dequant_config',row.get('dequant_config',-1))
        row.update(config=copy.deepcopy(measured['config']),guard=copy.deepcopy(measured['config']),
                   policy=18,dequant_config=-1,status='MEASURED_MOE_SINGLE_ROW',
                   profiles=[p['router'] for p in measured['profiles']],
                   moe_evidence_sha256=data['evidence_sha256'])
    return result

#!/usr/bin/env python3
"""Import the reviewed 40-key MoE confirmation, never change decode buckets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.moe_confirm.plan import choose,key
from quactlize.dispatch import moe_prefill_policy as policy
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def read(path):return json.loads(Path(path).read_bytes())
def sealed(path):
    r=read(path)
    if r['sha256']!=digest({k:v for k,v in r.items() if k!='sha256'}):raise ValueError('broken receipt: '+str(path))
    return r


def collect(folder):
    folder=Path(folder);run=sealed(folder/'run.json');summary=read(folder/'summary.json')
    transfer=read(folder/'transfer-manifest.json');box=Path(transfer['source'])
    if (summary['status']!='PASS' or summary['passed']!=344 or any(summary['returncodes']) or
            run['source_revision']['quactlize_commit']!=policy.COMMIT or run.get('retained_points')):
        raise ValueError('expected the complete fresh 344-point confirmation')
    groups=defaultdict(list)
    for card in run['cards']:
        for path in (folder/('gpu-'+card)/'points').glob('*.json'):
            point=sealed(path)
            if point['workload']['m']<=8:continue
            if point['status']!='PASS' or point['run']!=run['sha256']:raise ValueError('unconfirmed point')
            groups[key(point['workload'])].append(point)
    rows=[]
    for host,points in sorted(groups.items()):
        points.sort(key=lambda p:policy.ROUTERS.index(p['workload']['router']))
        selected=choose(points);recipe=points[0]['bindings'][selected]['recipe']
        profiles=[]
        for p in points:
            if p['bindings'][selected]['recipe']!=recipe:raise ValueError('router-dependent recipe')
            numerical=[]
            for name,want in p['batch_receipts'].items():
                path=folder/Path(name).relative_to(box)
                if sha(path)!=want:raise ValueError('changed batch receipt')
                batch=sealed(path)
                if selected in batch['data']['arms']:numerical+=batch['data']['arms'][selected]['numerical']
            if len(numerical)!=6:raise ValueError('missing selected accuracy checks')
            s=p['decision']['statistics'];parts=[]
            for cid in p['arm_components'][selected]:
                r=p['component_records'][cid]
                parts.append(dict(id=cid,kind=r['kind'],mean_us=r['mean_us']))
            profiles.append(dict(point=p['point'],router=p['workload']['router'],
                selected_us=s[selected]['mean_us'],original_us=s['original']['mean_us'],
                selected_samples_us=s[selected]['samples_us'],components=parts,
                max_relative_l1=max(n['relative_l1_error'] for n in numerical),
                authority=p['authority'],point_receipt=p['sha256']))
        rows.append(dict(key=list(host),selected=selected,recipe=recipe,profiles=profiles))
    result=dict(schema='quactlize.moe-prefill-exact.v1',archive_sha256=policy.ARCHIVE,
        source_revision=run['source_revision'],source_sha256=run['source']['sha256'],run_sha256=run['sha256'],
        timing='ASYS_GPU_ACTIVITY_MEAN_COMPONENT_SUM',prepare='PUBLIC_QUACTLIZE_PREPARE_V2',
        runtime_router_readback=False,native_no_regression_admitted=False,model_tpot_admitted=False,
        scope='ORDINARY_Q4_GATE_UP_Q5_DOWN_F32_IO_BF16_COMPUTE_NOT_FUSED_FFN',rows=rows)
    result['sha256']=digest(result);policy.validate(result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path)
    p.add_argument('--check',action='store_true');a=p.parse_args()
    data=collect(a.input) if a.input else policy.load()
    rendered=policy.header(data)
    if a.check:
        if rendered!=policy.HEADER.read_text():raise ValueError('stale MoE prefill header')
    else:
        policy.TABLE.write_text(json.dumps(data,indent=2)+'\n');policy.HEADER.write_text(rendered)
    print('MOE_PREFILL_POLICY rows=40 profiles=200 decode_changed=0 buckets_changed=0')


if __name__=='__main__':main()

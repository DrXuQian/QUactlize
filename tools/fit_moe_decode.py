#!/usr/bin/env python3
"""Revalidate the reviewed Q5 Asys archive and bind shipping single-row wins."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.smallm_asys import validate,check_producers
from dev.strategy.refresh_gaps import producer_entry
from quactlize.execution.moe_decode_policy import ARCHIVE,COMMIT,TABLE,verify
from quactlize.runtime.tuning import digest


def capture(path):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ARCHIVE:raise ValueError('unreviewed MoE archive')
    raw={}
    with tarfile.open(path) as archive:
        for member in archive:
            p=PurePosixPath(member.name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or member.name in raw:
                raise ValueError('unsafe/duplicate MoE archive member')
            raw[member.name]=archive.extractfile(member).read()
    def read(name):return json.loads(raw[name])
    def sealed(name):
        value=read(name)
        if value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):
            raise ValueError('MoE receipt seal differs: '+name)
        return value
    transfer=read('transfer-manifest.json')
    if set(raw)!={r['path'] for r in transfer['included']}|{'transfer-manifest.json'}:
        raise ValueError('MoE transfer inventory differs')
    for r in transfer['included']:
        b=raw[r['path']]
        if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:
            raise ValueError('MoE transfer payload differs')
    plan,run=sealed('plan.json'),sealed('run.json')
    if (run['plan']!=plan['sha256'] or run['source_revision']['quactlize_commit']!=COMMIT or
            plan['cohort']!='moe-token-pair' or len(plan['points'])!=8 or
            len(run['devices'])!=1 or run['devices'][0]['device']['compute_units']!=72):
        raise ValueError('MoE cohort/source/device differs')
    groups=defaultdict(list);deferred=[]
    for point in plan['points']:
        w=point['workload']
        if w['qtype']!=13:continue  # Failed Q4 measurements are NOT admitted.
        filename,=[n for n in raw if '/points/' in n and n.endswith(point['id']+'.json')]
        receipt=sealed(filename)
        if receipt['status']!='PASS' or receipt['run']!=run['sha256']:raise ValueError('MoE point failed')
        data=receipt['data'];validate(data);check_producers(point,data)
        if data['workload']!=w:raise ValueError('MoE receipt workload differs')
        arms={a['id']:a for a in point['arms']}
        stats=data['decision']['statistics']
        candidates={}
        for aid,arm in arms.items():
            p=dict(arm['recipe']['parameters'])
            if aid!='original' and (arm['recipe']['kind']!='simt' or p.pop('moe_pair_rows',None)!=1):continue
            c=dict(kind=arm['recipe']['kind'],**p)
            if set(c)!={'kind','variant','columns','warps','values','split'}:
                raise ValueError('cannot alias experimental body to a shipping config')
            nodes=[data['component_records'][cid]['identity']['node'] for cid in data['arm_components'][aid]
                   if data['component_records'][cid]['kind']=='producer']
            node,=nodes
            if aid!='original' and producer_entry(node)!='quactlize::execution::simt::common_reader':
                raise ValueError('single-row candidate has the wrong body')
            proof=data['arms'][aid]
            value=dict(mean_us=stats[aid]['mean_us'],samples_us=stats[aid]['samples_us'],
                max_relative_l1=max(p['relative_l1_error'] for p in proof['numerical']),
                producer=producer_entry(node),module_sha256=proof['receipt']['module_sha256'])
            if aid=='original' or value['mean_us']<=stats['original']['mean_us']:
                key=digest(c)
                old=candidates.get(key)
                if old is None or value['mean_us']<old['measurement']['mean_us']:
                    candidates[key]=dict(config=c,arm=aid,measurement=value)
        old=dict(kind='simt',**point['selection']['config'])
        original=dict(mean_us=stats['original']['mean_us'],samples_us=stats['original']['samples_us'],
            max_relative_l1=max(p['relative_l1_error'] for p in data['arms']['original']['numerical']),
            producer=producer_entry(next(data['component_records'][cid]['identity']['node']
                for cid in data['arm_components']['original'] if data['component_records'][cid]['kind']=='producer')),
            module_sha256=data['arms']['original']['receipt']['module_sha256'])
        groups[tuple(point['selection']['request'])].append(dict(point=point['id'],router=w['router'],
            candidates=candidates,original_config=old,original=original,
            record_sha256=hashlib.sha256(raw[filename]).hexdigest(),latency_provenance=data['latency_provenance']))
        paired=[dict(arm=aid,mean_us=s['mean_us']) for aid,s in stats.items()
                if dict(arms[aid]['recipe']['parameters']).get('moe_pair_rows')==2]
        if paired:deferred.append(dict(point=point['id'],router=w['router'],candidates=paired,
            reason='PAIR_READER_NOT_ADMITTED_AS_A_SHIPPING_SINGLE_ROW_CONFIG'))
    rows=[]
    for key,profiles in sorted(groups.items()):
        common=set.intersection(*(set(p['candidates']) for p in profiles))
        def score(cid):
            ratios=[p['candidates'][cid]['measurement']['mean_us']/p['original']['mean_us'] for p in profiles]
            return max(ratios),sum(ratios),not all(p['candidates'][cid]['arm']=='original' for p in profiles),cid
        chosen=min(common,key=score);first=profiles[0]
        config=first['candidates'][chosen]['config']
        rows.append(dict(key=list(key),config=config,original_config=first['original_config'],
            changed=config!=first['original_config'],profiles=[dict(point=p['point'],router=p['router'],
                original=p['original'],selected=p['candidates'][chosen]['measurement'],
                selected_arm=p['candidates'][chosen]['arm'],record_sha256=p['record_sha256'],
                latency_provenance=p['latency_provenance']) for p in sorted(profiles,key=lambda p:p['router'])]))
    # Retain the complete compiled execution-header closure, not only the
    # top-level reader. Additional non-header policy files do not change it.
    bodies={p:h for p,h in run['source']['files'].items()
            if p.startswith('quactlize/execution/') and Path(p).suffix in ('.h','.hpp','.cuh','.inc')}
    result=dict(schema='quactlize.moe-decode-exact.v1',archive_sha256=ARCHIVE,
        source_revision=run['source_revision'],run_sha256=run['sha256'],device=run['devices'][0],
        body_hashes=bodies,rows=rows,deferred=deferred,
        timing_method='ASYS_GPU_ACTIVITY_MEAN_COMPONENT_SUM',selection='MINIMAX_MEAN_RATIO_OF_COMMON_PUBLIC_RECIPES',
        runtime_router_readback=False,bucket_changes=False,model_tpot_admitted=False,
        scope='Q5_INDEXED_F32_STORAGE_BF16_COMPUTE_EXACT_M1_M8_NOT_GROUPED_OR_MODEL_AB')
    result['evidence_sha256']=digest(result);verify(result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--check',action='store_true')
    a=p.parse_args();data=capture(a.review)
    if a.check:
        if json.loads(TABLE.read_bytes())!=data:raise ValueError('stored MoE evidence differs')
    else:TABLE.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print('MOE_EXACT '+json.dumps(dict(rows=len(data['rows']),changed=sum(r['changed'] for r in data['rows']),
          model_tpot_admitted=False,bucket_changes=False)))


if __name__=='__main__':main()

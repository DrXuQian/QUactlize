#!/usr/bin/env python3
"""Import the reviewed Asys token cohort and regenerate exact launch bindings."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.production import config, key, require, serialize
from dev.strategy.smallm_asys import validate, check_producers
from quactlize.execution import token_cta_policy as policy
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

COHORTS = {
    'models': ('1f45f264e91da0473fd97718f2fa2c2366f7ecf76d05eaac1c14b11072edd065',
               'bbcf4d6be6dbaa9e258db3fd22dbfe7425f476ec', 328),
    'q6': ('c43f3b1a549c92a5d7ec241276519300c20c20199a404dde2d942a1a94b16224',
           '724b1ed290f6f8468135bf847662c256ef20ce13', 8),
}


def capture(path):
    require(sha(path)==policy.ARCHIVE,'unreviewed token CTA archive')
    records, hashes = {}, {}
    with tarfile.open(path,'r|xz') as stream:
        for member in stream:
            name=member.name;p=PurePosixPath(name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts and
                    name not in hashes,'unsafe/duplicate member')
            body=stream.extractfile(member).read()
            hashes[name]=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
            if name.endswith('.json'):records[name]=json.loads(body)
    transfer=records['transfer-manifest.json'];root=PurePosixPath(transfer['source'])
    require({r['path']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in transfer['included']}==
            {k:v for k,v in hashes.items() if k!='transfer-manifest.json'},'transfer hashes differ')
    plan,run=records['plan.json'],records['run.json']
    for r in (plan,run):require(r['sha256']==digest({k:v for k,v in r.items() if k!='sha256'}),'run/plan seal')
    require(run['plan']==plan['sha256'] and plan['cohort']=='token-cta-formats' and
            records['summary.json']['status']=='COMPLETE','incomplete cohort')
    require(run['source_revision']['quactlize_commit']=='1048a176a468e07abf30685f25344a4bae7c3410',
            'unreviewed source revision')
    points={p['id']:p for p in plan['points']};rows=[]
    headers=next(iter(records['images.json']['vectors'].values()))['identity']['headers']
    require(all(image['identity']['headers']==headers for image in records['images.json']['vectors'].values()),
            'vector source closure differs across cells')
    originals={tuple(r['request']):r for r in records['cpp-originals.json']['requests']}
    for name,r in records.items():
        if '/points/' not in name:continue
        require(r['status']=='PASS' and r['run']==run['sha256'] and
                r['sha256']==digest({k:v for k,v in r.items() if k!='sha256'}),'point seal/status')
        p=points[r['point']];v=r['data'];validate(v);check_producers(p,v)
        for source,h in r['evidence'].items():
            local=str(PurePosixPath(source).relative_to(root))
            if local in hashes:require(hashes[local]['sha256']==h,'point capture hash differs')
        arms={a['id']:a for a in p['arms']};winner=v['decision']['winner']
        arm=arms[winner];recipe=arm['recipe'];c=config(recipe,p['workload'])
        tokens=c.pop('cta_tokens',None);window=c.pop('cta_window',0)
        impl=dict(kind='token-cta',tokens=tokens,window=window) if tokens else dict(kind='shipping')
        stats=v['decision']['statistics'][winner];k=list(key(p['workload']))
        components=[]
        for cid in v['arm_components'][winner]:
            component=v['component_records'][cid]
            components.append(dict(id=cid,kind=component['kind'],mean_us=component['mean_us'],
                samples_sha256=digest(component['samples_us']),identity=component['identity']))
        measured=v['arms'][winner]
        selected=dict(id=winner,recipe=recipe,config=c,implementation=impl,
            dequant_config=dict(recipe['parameters']).get('dequant_config',-1),
            mean_us=stats['mean_us'],round_means_us=[statistics.fmean(s) for s in stats['samples_us']],
            components=components,execution=measured['receipt'],
            numerical_error=max(n['relative_l1_error'] for n in measured['numerical']))
        rows.append(dict(key=k,point=r['point'],original_cpp=originals[tuple(k)],
            selected=selected,changed=recipe!=p['original_recipe'],
            candidate_means_us={a:s['mean_us'] for a,s in v['decision']['statistics'].items()},
            receipt_sha256=hashes[name]['sha256'],latency_provenance=v['latency_provenance']))
    from tools.generate_decode_selection import effective,EVIDENCE
    baseline=effective(json.loads(EVIDENCE.read_text()),include_token_cta=False)
    data=dict(schema='quactlize.token-cta-exact.v1',archive_sha256=policy.ARCHIVE,
        baseline_sha256=digest(baseline),source_revision=run['source_revision'],
        run_sha256=run['sha256'],plan_sha256=plan['sha256'],sdk=run['sdk'],devices=run['devices'],
        body_hashes={name:headers[name] for name in headers if
            name.startswith('quactlize/execution/') and name.endswith(('.cuh','.h','.hpp','.inc')) and
            name!='quactlize/execution/simt_strategy.hpp'},
        rows=sorted(rows,key=lambda r:r['key']),
        scope='DENSE_N1024_K5120_M1_TO8_F32_STORAGE_F16_COMPUTE',
        metric='ASYS_ARITHMETIC_MEAN_SUM_OF_ISOLATED_COMPONENTS',
        native_no_regression_admitted=False,model_tpot_admitted=False,bucket_changed=False)
    # Timing is bound to the measured device bodies, not future edited headers.
    require(data['body_hashes'] and all(h==sha(ROOT/name)
        for name,h in data['body_hashes'].items() if name!='quactlize/execution/q8_refresh.inc'),
        'measured kernel body changed')
    data['evidence_sha256']=digest(data)
    return data


def cohort(path, name):
    """Import only the audited archives, including the sealed Q6 M4 reparse."""
    archive, commit, count = COHORTS[name]
    require(sha(path) == archive, 'unreviewed '+name+' archive')
    records, hashes = {}, {}
    with tarfile.open(path, 'r|xz') as stream:
        for member in stream:
            p = PurePosixPath(member.name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts and
                    member.name not in hashes, 'unsafe/duplicate member')
            body = stream.extractfile(member).read()
            hashes[member.name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            if member.name.endswith('.json'):
                records[member.name] = json.loads(body)
    transfer = records['transfer-manifest.json']; root = PurePosixPath(transfer['source'])
    require({r['path']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in transfer['included']} ==
            {k:v for k,v in hashes.items() if k != 'transfer-manifest.json'}, 'transfer hashes differ')
    plan, run = records['plan.json'], records['run.json']
    for r in (plan, run):
        require(r['sha256'] == digest({k:v for k,v in r.items() if k != 'sha256'}), 'plan/run seal')
    require(run['plan'] == plan['sha256'] and
            run['source_revision']['quactlize_commit'] == commit, 'measured source/run differs')
    summary = records['summary.json']
    if name == 'q6':
        reparsed = [v for k,v in records.items() if k.startswith('reparse-') and k.endswith('/summary.json')]
        require(len(reparsed) == 1 and reparsed[0]['old_receipts_modified'] is False, 'Q6 reparse scope')
        summary = reparsed[0]
    require(summary['status'] == 'COMPLETE' and len(summary['points']) == count, 'incomplete '+name)
    points = {p['id']:p for p in plan['points']}
    originals = {tuple(r['request']):r for r in records['cpp-originals.json']['requests']}
    members = {r['point']:n for n,r in records.items() if '/points/' in n}
    rows = []
    for item in summary['points']:
        member = (str(PurePosixPath(item['receipt']).relative_to(root)) if 'receipt' in item
                  else members[item['point']])
        r = records[member]
        require(r['status'] == 'PASS' and r['sha256'] == digest(
            {k:v for k,v in r.items() if k != 'sha256'}), 'point seal/status')
        p, v = points[r['point']], r['data']
        validate(v); check_producers(p, v)
        require(v['latency_provenance']['quactlize_commit'] == commit, 'point source differs')
        for source, h in r['evidence'].items():
            local = str(PurePosixPath(source).relative_to(root))
            if local in hashes:
                require(hashes[local]['sha256'] == h, 'point capture hash differs')
        arms = {a['id']:a for a in p['arms']}; winner = v['decision']['winner']
        recipe = arms[winner]['recipe']; c = config(recipe, p['workload'])
        tokens, window, body = c.pop('cta_tokens', None), c.pop('cta_window', 0), c.pop('q8_body', None)
        impl = (dict(kind='token-cta', tokens=tokens, window=window) if tokens else
                dict(kind='q8-address', body=body) if body else dict(kind='shipping'))
        stats = v['decision']['statistics'][winner]; k = list(key(p['workload']))
        measured = v['arms'][winner]
        components = []
        for cid in v['arm_components'][winner]:
            component = v['component_records'][cid]
            components.append(dict(id=cid, kind=component['kind'], mean_us=component['mean_us'],
                samples_sha256=digest(component['samples_us']), identity=component['identity']))
        selected = dict(id=winner, recipe=recipe, config=c, implementation=impl,
            dequant_config=dict(recipe['parameters']).get('dequant_config', -1),
            mean_us=stats['mean_us'], round_means_us=[statistics.fmean(s) for s in stats['samples_us']],
            components=components, execution=measured['receipt'],
            numerical_error=max(n['relative_l1_error'] for n in measured['numerical']))
        rows.append(dict(key=k, point=r['point'], cohort=name, original_cpp=originals[tuple(k)],
            selected=selected, changed=recipe != p['original_recipe'],
            candidate_means_us={a:s['mean_us'] for a,s in v['decision']['statistics'].items()},
            receipt_sha256=hashes[member]['sha256'], latency_provenance=v['latency_provenance']))
    headers = next(iter(records['images.json']['vectors'].values()))['identity']['headers']
    require(all(image['identity']['headers'] == headers for image in records['images.json']['vectors'].values()),
            'vector source closure differs')
    metadata = dict(archive_sha256=archive, source_revision=run['source_revision'],
        run_sha256=run['sha256'], plan_sha256=plan['sha256'], sdk=run['sdk'], devices=run['devices'],
        measured_headers_sha256=digest(headers))
    return rows, metadata, headers


def capture_all(formats, models, q6):
    legacy = capture(formats)
    data = copy.deepcopy(legacy)
    data['schema'] = 'quactlize.token-cta-exact.v2'
    data['cohorts'] = {'formats': {k:legacy[k] for k in
        ('archive_sha256','source_revision','run_sha256','plan_sha256','sdk','devices')}}
    for r in data['rows']:
        r['cohort'] = 'formats'
    for name, path in (('models', models), ('q6', q6)):
        rows, metadata, headers = cohort(path, name)
        data['rows'].extend(rows); data['cohorts'][name] = metadata
        if name == 'models':
            # These generated HOST catalogs change in this admission. Device
            # arithmetic headers remain byte-bound; catalogs have separate
            # capability seals, not obsolete measured-body hashes.
            catalogs = {'quactlize/execution/q8_refresh.inc', 'quactlize/execution/token_cta.inc'}
            data['body_hashes'] = {p:h for p,h in headers.items() if p not in catalogs and
                p.startswith('quactlize/execution/') and p.endswith(('.cuh','.h','.hpp','.inc'))}
            require(all(sha(ROOT/p) == h for p,h in data['body_hashes'].items()), 'measured body changed')
    data['rows'].sort(key=lambda r:r['key'])
    data['scope'] = 'DENSE_MODEL_360_PLUS_PRIOR_16_FORMAT_CONTROLS_M1_TO8_F32_STORAGE_F16_COMPUTE'
    data['evidence_sha256'] = digest({k:v for k,v in data.items() if k != 'evidence_sha256'})
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path);parser.add_argument('--check',action='store_true')
    parser.add_argument('--models',type=Path);parser.add_argument('--q6',type=Path)
    args=parser.parse_args()
    if bool(args.models) != bool(args.q6) or (args.models and not args.capture):
        parser.error('--models and --q6 require each other and --capture of the original format cohort')
    if args.capture:
        if args.check:parser.error('capture is not a --check operation')
        if not args.models:parser.error('current admission requires --capture, --models and --q6 archives')
        data=capture_all(args.capture,args.models,args.q6)
        policy.verify(data)
        policy.TABLE.write_text(serialize(data))
        print('TOKEN_CTA_CAPTURE evidence='+data['evidence_sha256'])
        return
    text=policy.catalog()
    if args.check:
        if policy.CATALOG.read_text()!=text:raise ValueError('stale token CTA binding')
    else:policy.CATALOG.write_text(text)
    policy.capability()
    print(f'TOKEN_CTA_BINDING exact={len(policy.load()["rows"])} token_bodies={len(policy.rows())} buckets_changed=0')


if __name__=='__main__':main()

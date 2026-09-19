#!/usr/bin/env python3
"""Generate final decode choices from reviewed, complete-call measurements.

Runtime consumes one catalog. Old matched/Q8 boards are immutable import
sources, not successive runtime heuristics. Timing revisions and implementation
identity are separate; a geometry tuple alone is not a measured kernel.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fit_smallm_closure import header as matched_header

FIELDS = ('variant', 'columns', 'warps', 'values', 'split')
EVIDENCE = ROOT/'policies/kpack_decode_winners_v1.json'
OUTPUT = ROOT/'policies/kpack_smallm_effective_v1.hpp'
IMPLEMENTATIONS = ROOT/'quactlize/execution/measured_decode.inc'
ARCHIVES = {
    'tp2': '5944aa7547e4dd5c9b87dae0f906102d7df5d4e39f42d7007c25731408fee075',
    'fallback': '6a1b416f7ec91683b15ac886b0d364bcc275a5aa19ca5db8ce5d70ef46ade4c6',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(path, expected):
    if digest(path.read_bytes()) != expected:
        raise ValueError('unreviewed result archive')
    result = {}
    with tarfile.open(path) as stream:
        for item in stream:
            p = PurePosixPath(item.name)
            if p.is_absolute() or '..' in p.parts or not (item.isdir() or item.isfile()):
                raise ValueError('unsafe result member')
            if item.isfile():
                if item.name in result:
                    raise ValueError('duplicate result member')
                result[item.name] = stream.extractfile(item).read()
    return result


def measure(value):
    rounds = value['samples_us']
    if len(rounds) != 6 or any(len(r) != 15 for r in rounds):
        raise ValueError('confirmation denominator differs')
    flat = [t for r in rounds for t in r]
    if not all(math.isfinite(t) and t > 0 for t in flat):
        raise ValueError('invalid timing')
    median = statistics.median(map(statistics.median, rounds))
    if not math.isclose(median, value['median_us'], rel_tol=1e-12):
        raise ValueError('reported median differs')
    return dict(median_us=median, round_medians_us=list(map(statistics.median, rounds)),
                samples_sha256=digest(json.dumps(rounds, separators=(',', ':')).encode()))


def key(p):
    return [p['q'], p['mode'], p['n'], p['k'], 256 if p['mode'] else 1,
            8 if p['mode'] else 1, p['channels'], 1, p['compute']]


def numerical(proof):
    if isinstance(proof, dict):
        if proof['status']!='PASS':raise ValueError('numerical proof failed')
        proof=proof['controls']
    if not set(range(1,9)) <= {r['tokens'] for r in proof} or any(
            not math.isfinite(r['error']) or r['error']>=.005 for r in proof):
        raise ValueError('numerical coverage/error differs')


def capture(tp2, fallback):
    old, new = archive(tp2, ARCHIVES['tp2']), archive(fallback, ARCHIVES['fallback'])
    manifests = {name: json.loads(data['results/gemv-manifest.json'])
                 for name, data in [('tp2', old), ('fallback', new)]}
    # Recheck saved counters; do not require the box to repeat parser failures.
    from dev.tp2_decode.run_fallback import profile_check
    for r in manifests['fallback']['records']:
        for arm in ('reference', 'fallback'):
            profile_check(new[f"results/sweep/{r['point']['name']}-{arm}.acu.csv"].decode(), r, arm)
    old_summary = json.loads(old['results/sweep/summary.json'])
    if not old_summary['complete'] or old_summary['status'] != 'PASS' or len(old_summary['records']) != 17:
        raise ValueError('TP2 confirmed pool incomplete')
    winners = []
    for record in manifests['tp2']['records']:
        p = record['point']; name = p['name']; member = f'results/sweep/{name}.json'
        raw = json.loads(old[member]); values = raw['summary']; source = 'tp2'
        if raw['status'] != 'PASS' or raw['failures']:
            raise ValueError('invalid source point: '+name)
        for arm in values:
            numerical(raw['numerics'][arm])
        summaries = {arm: measure(v) for arm, v in values.items()}
        arm = min(summaries, key=lambda a: summaries[a]['median_us'])
        # A newer paired comparison takes precedence, not an across-run
        # comparison of absolute microseconds from different epochs.
        if not p['paired'] and member in new:
            source = 'fallback'; raw = json.loads(new[member]); values = raw['timing']
            if raw['status'] != 'PASS':
                raise ValueError('fallback numerical evidence differs')
            for v in raw['numerics'].values():numerical(v)
            summaries = {a: measure(v) for a, v in values.items()}
            latest = min(summaries, key=lambda a: summaries[a]['median_us'])
            arm = values[latest].get('arm', 'incumbent') if latest == 'reference' else 'production'
            selected = values[latest]
        else:
            selected = values[arm]
        c = copy.deepcopy(selected['config'])
        kind = selected['kind']
        if kind == 'tc':
            c.update(kind='tc', route=0 if c['route']=='fq-dense' else 1,
                     parent_persistent=c.pop('persistent'), grid_mode=0, grid_b=0)
        else:
            c = dict(kind='simt', **{f: c[f] for f in FIELDS})
        impl = 'existing' if arm in ('incumbent', 'production') else 'constant-split'
        winners.append(dict(point=p, key=key(p), config=c, implementation=impl,
            body=selected['config'], scope='GATE_UP_SWIGLU' if p['paired'] else 'GEMV_WITH_REDUCER',
            # Q4 paired and unfused+activation still need comparable chain costs.
            automatic=not (p['paired'] and name=='tp2-q4-paired-routed'),
            source=source, arm=arm, measurements=summaries,
            result_sha256=digest((old if source=='tp2' else new)[member])))
    for record in manifests['fallback']['records']:
        p = record['point']
        if not p['name'].startswith('unseen-'):
            continue
        member=f"results/sweep/{p['name']}.json";raw=json.loads(new[member])
        summaries={a:measure(v) for a,v in raw['timing'].items()}
        if raw['status']!='PASS' or min(summaries,key=lambda a:summaries[a]['median_us'])!='fallback':
            raise ValueError('table-miss promotion is not a winner')
        c=raw['timing']['fallback']['config']
        winners.append(dict(point=p,key=key(p),config=dict(kind='simt',**{f:c[f] for f in FIELDS}),
            implementation='existing',body=c,scope='GEMV_WITH_REDUCER',automatic=True,source='fallback',
            arm='production',measurements=summaries,result_sha256=digest(new[member])))
    result=dict(schema='quactlize.decode-winners.v1',archives=ARCHIVES,rows=winners,
        timing='ROTATING_M1_COMPLETE_CALL',numerical='TOKENS_1_TO_8',
        admission='COMPONENT_CONFIRMED_PRODUCTION_INTEGRATION_PENDING',
        idle_audit='NO_INDEPENDENT_AUDIT',device='PPU-ZW810',policy_tokens=[1])
    return result | dict(evidence_sha256=digest(json.dumps(result,sort_keys=True).encode()))


def effective(evidence):
    if evidence['schema']!='quactlize.decode-winners.v1' or evidence['archives']!=ARCHIVES:
        raise ValueError('winner evidence authority differs')
    if evidence.get('evidence_sha256')!=digest(json.dumps(
            {k:v for k,v in evidence.items() if k!='evidence_sha256'},sort_keys=True).encode()):
        raise ValueError('winner evidence was edited; recapture the reviewed archives')
    base=json.loads((ROOT/'policies/kpack_smallm_matched_v1.json').read_text())
    vector=json.loads((ROOT/'policies/kpack_q8_vector_v1.json').read_text())
    result=copy.deepcopy(base)
    for row in result['exact']:
        row['policy']=14 if row['status']=='ROUTER_SENSITIVE_PARETO' else 12
        row['guard']=copy.deepcopy(row['config'])
        if row['key'][0]!=8:
            continue
        cfg=row['config']
        for v in vector['rows']:
            if row['key']==v['key'] and cfg['kind']=='simt' and [cfg[f] for f in FIELDS]==v['baseline']:
                row['config']=dict(kind='simt',**dict(zip(FIELDS,v['candidate'])));row['policy']=15
        for v in vector.get('tc_rows',[]):
            expected=dict(v['baseline']);expected['route']=1
            expected['parent_persistent']=expected.pop('persistent')
            if row['key']==v['key'] and cfg['kind']=='tc' and all(cfg.get(f)==x for f,x in expected.items()):
                row['config']=dict(kind='simt',**dict(zip(FIELDS,v['candidate'])));row['policy']=15
    by_key={tuple(r['key']):i for i,r in enumerate(result['exact'])}
    for w in evidence['rows']:
        if w['point']['paired'] or not w['automatic']:
            continue
        if w['key'][7]!=1:
            raise ValueError('M1 evidence cannot promote another token count')
        row=dict(key=w['key'],config=w['config'],guard=w['config'],policy=12,
                 status='MEASURED_MINIMUM',worst_regret_pct=0,point=w['point']['name'])
        index=by_key.get(tuple(w['key']))
        if index is None:
            by_key[tuple(w['key'])]=len(result['exact']);result['exact'].append(row)
        else:
            result['exact'][index]=row
    # Retain existing bucket geometry/order. Refresh those donors in place and
    # add only previously unoccupied M1 buckets. Exact winners always precede it.
    occupied={tuple(b['key']) for b in result['buckets']}
    for i,r in enumerate(result['exact']):
        if 'point' not in r:
            continue
        q,mode,n,k,e,top,ch,m,compute=r['key']
        b=[q,mode,e,top,ch,(m-1).bit_length(),n.bit_length()-1,k.bit_length()-1,compute]
        if tuple(b) not in occupied:
            occupied.add(tuple(b));result['buckets'].append(dict(key=b,row=i))
    # Successful new exact evidence supersedes an old open cell only for the
    # identical semantic key; all other explicit exclusions remain fail-closed.
    promoted={tuple(w['key']) for w in evidence['rows'] if w['automatic'] and not w['point']['paired']}
    result['excluded']=[r for r in base['excluded'] if tuple(r['key'][:-1]+[int(r['key'][-1]=='bf16')]) not in promoted]
    return result


def header(policy):
    text=matched_header(policy).replace('tools/fit_smallm_closure.py','tools/generate_decode_selection.py')
    text=text.replace('smallm_matched_data','smallm_effective_data')
    # Guard choices preserve old donor admission unless a new measured winner
    # has explicitly replaced it. They never select a second kernel.
    guards=matched_header(dict(exact=[dict(r,config=r['guard']) for r in policy['exact']],buckets=[],excluded=[]))
    guard_choices=guards.split('inline constexpr Choice kChoices[]={\n',1)[1].split('\n};',1)[0]
    guard_rows=guards.split('inline constexpr Row kExact[]={\n',1)[1].split('\n};',1)[0]
    indices=[int(line.strip().strip('{},').split(',')[-2]) for line in guard_rows.splitlines()]
    extra='inline constexpr Choice kGuards[]={\n'+guard_choices+'\n};\n'
    extra+='inline constexpr int kGuardIndex[]={'+','.join(map(str,indices))+'};\n'
    extra+='inline constexpr int kPolicy[]={'+','.join(str(r['policy']) for r in policy['exact'])+'};\n'
    return text.replace('} // namespace quactlize::smallm_effective_data',extra+'} // namespace quactlize::smallm_effective_data')


def implementations(evidence):
    lines=['// Generated measured implementations. QDM is supplied by the consumer.',
           '// id, q, mode, N, K, E, topk, channels, compute, V, C, W, P, S, changes, hoist, fixed']
    for w in evidence['rows']:
        if w['point']['paired'] or w['implementation']!='constant-split' or w['config']['kind']!='simt':
            continue
        q,mode,n,k,e,top,ch,m,compute=w['key'];b=w['body']
        fields=[q,mode,n,k,e,top,ch,compute,*[b[f] for f in FIELDS],b['changes'],int(b['hoist']),int(b['fixed'])]
        lines.append('QDM('+w['point']['name'].replace('-','_')+','+','.join(map(str,fields))+')')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tp2',type=Path);p.add_argument('--fallback',type=Path)
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.tp2 or a.fallback:
        if not a.tp2 or not a.fallback or a.check:p.error('capture requires both archives')
        EVIDENCE.write_text(json.dumps(capture(a.tp2,a.fallback),indent=2)+'\n')
    e=json.loads(EVIDENCE.read_text());policy=effective(e)
    for path,text in ((OUTPUT,header(policy)),(IMPLEMENTATIONS,implementations(e))):
        if a.check:
            if path.read_text()!=text:raise ValueError('stale generated selection: '+str(path))
        else:path.write_text(text)
    print(f'DECODE_SELECTION_GENERATED exact={len(policy["exact"])} buckets={len(policy["buckets"])} winners={len(e["rows"])}')


if __name__=='__main__':main()

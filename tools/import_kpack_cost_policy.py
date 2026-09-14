#!/usr/bin/env python3
"""Audit finite component receipts and emit a bounded, measurement-based policy.

No GPU calls or timing estimates from peak bandwidth. Complete TC calls already
include reduction. Generated tables contain measured knots, not shape rules.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.tuning import digest
from tools.run_kpack_cost_supplement import keys, validate, stable, timing_valid
from tools.kpack_cost_supplement import dequant_key, validate_point


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


class Archive:
    """Read only regular JSON files, without extracting untrusted paths."""
    def __init__(self, path):
        self.path = Path(path)
        self.sha = sha_bytes(self.path.read_bytes())
        self.data = {}
        with tarfile.open(path) as source:
            seen = set()
            for m in source:
                p = PurePosixPath(m.name)
                if p.is_absolute() or '..' in p.parts or m.name in seen:
                    raise ValueError('unsafe/duplicate archive path: ' + m.name)
                seen.add(m.name)
                if not (m.isfile() or m.isdir()):
                    raise ValueError('nonregular archive member: ' + m.name)
                if m.isfile() and m.name.endswith('.json'):
                    self.data[m.name] = source.extractfile(m).read()

    def read(self, name, expected=None):
        data = self.data['results/' + name]
        if expected and sha_bytes(data) != expected:
            raise ValueError('receipt digest differs: ' + name)
        return json.loads(data)


def audit(source):
    result = source.read('result.json')
    authority = source.read('authority.json')
    plan = source.read('plan.json')
    if (result['status'] != 'COMPLETE' or result['missing'] or
            result['authority_sha256'] != digest(authority) or
            digest({k:v for k,v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256'] or
            plan['plan_sha256'] != authority['plan_sha256']):
        raise ValueError('incomplete/unbound campaign')
    expected = {k for p in plan['points'] for k in keys(p)} | {'dense-m1-n4096-s4-recheck'}
    if set(result['components']) != expected or result['complete'] != len(expected):
        raise ValueError('component denominator differs')
    records = {}
    weight_authorities = {}
    for key, ref in result['components'].items():
        row = source.read(ref['path'], ref['sha256'])
        weight = str(PurePosixPath(ref['path']).parent.parent)
        if weight not in weight_authorities:
            a = source.read(weight + '/authority.json')
            task = result['assignment'][PurePosixPath(weight).name]
            probe = authority['probes'][str(task['device'])]
            if (a['device'] != probe['device'] or a['runtime'] != authority['runtime'] or
                    a['python_packages'] != probe['packages'] or a['sources'] != authority['sources'] or
                    a['plan_sha256'] != authority['plan_sha256'] or a['bundle_sha256'] != authority['bundle_sha256']):
                raise ValueError('weight authority differs: ' + weight)
            weight_authorities[weight] = a
        a = weight_authorities[weight]
        validate(row, key, digest(a))
        if row['device'] != a['device'] or ref['device'] != a['device']:
            raise ValueError('component device differs: ' + key)
        records[key] = row
    for p in plan['points']:
        validate_point(p)
        rows = [records[k] for k in keys(p)]
        if any(r.get('fixture') != rows[0]['fixture'] for r in rows):
            raise ValueError('weight/oracle bytes differ: ' + p['id'])
        for r in rows:
            if r['kind'] in ('gemm','bf16') and (r['rows'] != p['rows'] or r['routes_sha256'] != p['routes_sha256']):
                raise ValueError('router differs: ' + p['id'])
    return plan, records, authority


def config(p, route, mode, row):
    selected = p['routes'][str(route)]['selected' if mode == 'current' else 'historical']
    symbol = row['selection']['parent']
    if symbol != selected.get('parent', selected.get('symbol')) or row['selection']['split'] != selected['split']:
        raise ValueError('measured tactic differs from planned tactic')
    fields = re.search(r'_tm(\d+)_tn(\d+)_tk(\d+)_wm(\d+)_wn(\d+)_s(\d+)(?:_bc\d+)?_ap(\d+)_dn(\d+)', symbol)
    if not fields:
        raise ValueError('unknown parent symbol: ' + symbol)
    grid_mode = selected['grid_mode']
    if isinstance(grid_mode, str):
        grid_mode = {'implicit':0,'ordinary':0,'capacity':2,'balanced':3}[grid_mode]
    return dict(symbol=symbol, qtype=p['q'], route=route,
        **dict(zip(('tm','tn','tk','wm','wn','stages','ap','dn'),map(int,fields.groups()))),
        parent_persistent=int(symbol.endswith('_persistent')) if symbol.startswith('fqg_') else -1,
        split=selected['split'], grid_mode=grid_mode, grid_b=selected.get('grid_b',0), occupancy=0,
        mapping_id='0x51344b5034540001' if p['q']==12 else '0x514b504b54000001')


def candidate(route, gemm, dq=0, dq_config=-1, tactic=None, source=None, stable_timing=True):
    return dict(route=route, gemm_us=gemm, dequant_us=dq, cost_us=gemm+dq,
                dequant_config=dq_config, tactic=tactic, source=source, stable=stable_timing)


def points_from_supplement(plan, records):
    out = []
    for p in plan['points']:
        candidates = []
        sf = min((r for r in records[dequant_key(p,0)]['rows'] if stable(r)), key=lambda r:r['median_us'])
        for route, info in p['routes'].items():
            route = int(route)
            for mode in ('current','historical'):
                if mode == 'historical' and info['historical'] is None:
                    continue
                key = f'{p["id"]}-r{route}-{mode}'
                row = records[key]
                candidates.append(candidate(route%2,row['median_us'],sf['median_us'] if route%2 else 0,
                    sf['config'] if route%2 else -1,config(p,route,mode,row),key,stable(row)))
        bf = records[p['id']+'-bf16']
        # Even an all-active measurement must use an indexed-capable recipe in
        # production: next invocation can have a different GPU expert set.
        allowed = {4,5,10,11} if p['q'] in (12,13) else {4,5}
        full = [r for r in records[dequant_key(p,1)]['rows'] if stable(r) and
                (p['experts']==1 or r['config'] in allowed)]
        if full:
            dq = min(full,key=lambda r:r['median_us'])
            candidates.append(candidate(2,bf['median_us'],dq['median_us'],dq['config'],
                                        source=p['id']+'-bf16',stable_timing=stable(bf)))
        out.append(dict(q=p['q'],n=p['n'],k=p['k'],experts=p['experts'],tokens=p['tokens'],
                        profile=p['profile'],id=p['id'],candidates=candidates))
    return out


def prior_points(source, board, supplement_plan):
    """Join the 88 prior complete calls with the previously audited 44 sums."""
    report = source.read('result.json')
    authority = source.read('authority.json',report['files']['authority.json'])
    if report['completed'] != 88 or report['failed'] or report['status'] != 'PASS':
        raise ValueError('prior selected GEMM denominator differs')
    # The original composition is bound by the selected measurement authority.
    if authority['board_sha256'] != sha_bytes((ROOT/'docs/measurements/kpack_prefill_components_20260914/result.json').read_bytes()):
        raise ValueError('prior component board differs')
    bundle=ROOT/'prebuilt/ppu0010/kpack-prefill-measure-v1'
    manifest=json.loads((bundle/'manifest.json').read_text())
    if sha_bytes((bundle/'manifest.json').read_bytes())!=authority['bundle_sha256']:
        raise ValueError('prior build manifest differs')
    archived_plan=(bundle/'plan.json').read_bytes()
    if sha_bytes(archived_plan)!=manifest['files']['plan.json']:
        raise ValueError('prior selected plan differs')
    requests={tuple(r['request']):r for r in json.loads(archived_plan)['requests']}
    out = []
    for b in board['rows']:
        w = b['workload']; e=w['experts']; t=b['tokens']
        ident=f'q{w["q"]}-n{w["n"]}-k{w["k"]}-e{e}-t{t}'
        candidates=[]
        for route in ((0,1) if e==1 else (2,3)):
            name=f'{ident}-r{route}.json'
            row=source.read(name,report['files'][name]);timing_valid(row)
            if (row['status']!='PASS' or row['error']>=.005 or not row['reducer_included'] or row['sf_prepass_timed'] or
                    row['fixture']['bf16_golden_sha256']!=b['golden_sha256'] or row['routes_sha256']!=b['routes_sha256'] or
                    any(row[k]!='PASS' for k in ('guards','zero_a','changed_a_graph','changed_a_sequence'))):
                raise ValueError('prior GEMM binding/controls differ')
            c=requests[(w['q'],route,row['total_rows'],w['n'],w['k'],e,t)]
            p=dict(q=w['q'],routes={str(route):dict(selected=c)})
            tc=config(p,route,'current',row)
            sf=b['sf_dequant']
            candidates.append(candidate(route%2,row['median_us'],sf['us'] if route%2 else 0,
                sf['config'] if route%2 else -1,tc,name,stable(row)))
        dq=b['full_dequant']
        if e==1 or dq['config'] in (4,5,10,11):
            candidates.append(candidate(2,b['bf16_gemm']['us'],dq['us'],dq['config'],source=ident+'-old-full'))
        out.append(dict(q=w['q'],n=w['n'],k=w['k'],experts=e,tokens=t,profile='real',id=ident+'-real',candidates=candidates))
    return out


def candidate_key(c):
    return (c['route'], c['dequant_config'], digest(c['tactic']))


def choose_profiles(profiles):
    """Minimax measured regret over actual profiles; no router readback."""
    pools=[]
    for p in profiles:
        pool={}
        for c in p['candidates']:
            if c['stable']:
                k=candidate_key(c)
                if k not in pool or c['cost_us']<pool[k]['cost_us']:
                    pool[k]=c
        pools.append(pool)
    common=set.intersection(*(set(p) for p in pools))
    if not common:
        raise ValueError('no stable common candidate: '+str([p['id'] for p in profiles]))
    best=[min(c['cost_us'] for c in p.values()) for p in pools]
    ranked=[]
    for k in common:
        regret=max(p[k]['cost_us']/b for p,b in zip(pools,best))
        ranked.append((regret,k))
    minimum=min(v for v,k in ranked)
    # Prefer FQ, then SF, within the allowed five-percent envelope.
    tied=[(k,v) for v,k in ranked if v<=max(1.05,minimum)]
    k,regret=min(tied,key=lambda x:(x[0][0],x[1],x[0]))
    rows=[p[k] for p in pools]
    return dict(route=k[0],dequant_config=k[1],tactic=rows[0]['tactic'],
                gemm_us=max(r['gemm_us'] for r in rows),dequant_us=max(r['dequant_us'] for r in rows),
                worst_measured_regret_pct=(regret-1)*100,
                profiles=[p['profile'] for p in profiles])


def policy_rows(points):
    groups=defaultdict(list)
    for p in points:
        groups[tuple(p[f] for f in ('q','n','k','experts','tokens'))].append(p)
    out=[]
    for key, ps in sorted(groups.items()):
        row=dict(zip(('q','n','k','experts','tokens'),key))
        # Availability masks have separate measured fallbacks. No SF timing
        # may use the old, slower prepass when the new expansion DSO is absent.
        row['choices']={}
        for mask in (1,2,3,7):
            filtered=[p|dict(candidates=[c for c in p['candidates'] if mask&(1<<c['route'])]) for p in ps]
            row['choices'][str(mask)]=choose_profiles(filtered)
        for mask in ('3','7'):
            c=row['choices'][mask]
            if c['route']<2 and c['tactic']!=row['choices'][str(1<<c['route'])]['tactic']:
                raise ValueError('cross-route choice differs from the actual fixed-route dispatcher')
        out.append(row)
    return out


def header(report):
    tactics={digest(c['tactic']):c['tactic'] for row in report['knots'] for c in row['choices'].values() if c['tactic']}
    names={k:i for i,k in enumerate(sorted(tactics))}
    lines=['// Generated by tools/import_kpack_cost_policy.py. Measured component costs.', '#pragma once',
        '#include "kpack_zw810_runtime_v1.hpp"','namespace quactlize_kpack_cost_v1 {',
        'using Config = quactlize_kpack_runtime_v1::Config;',
        'struct Choice { int route, dequant, config; double gemm_us, dequant_us; };',
        'struct Knot { int q,n,k,experts,tokens; Choice choices[4]; };',
        'inline constexpr char kEvidence[] = '+json.dumps(report['evidence_sha256'])+';',
        'inline constexpr Config kConfigs[] = {']
    for k in sorted(tactics):
        c=tactics[k]
        values=[json.dumps(k),json.dumps(c['symbol']),json.dumps('MEASURED_COMPONENT')]
        values += [str(c[f]) for f in ('qtype','route','tm','tn','tk','wm','wn','stages','ap','dn','parent_persistent','split','grid_mode','grid_b','occupancy')]
        values += [c['mapping_id']+'ULL']
        lines.append('  {'+','.join(values)+'},')
    lines+=['};','inline constexpr Knot kKnots[] = {']
    for r in report['knots']:
        cells=[]
        for mask in ('1','2','3','7'):
            c=r['choices'][mask];idx=names[digest(c['tactic'])] if c['tactic'] else -1
            cells.append('{'+','.join(map(str,[c['route'],c['dequant_config'],idx,c['gemm_us'],c['dequant_us']]))+'}')
        lines.append('  {'+','.join(str(r[f]) for f in ('q','n','k','experts','tokens'))+',{'+','.join(cells)+'}},')
    return '\n'.join(lines+['};','} // namespace quactlize_kpack_cost_v1',''])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supplement',type=Path,required=True)
    p.add_argument('--prior-gemm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--header',type=Path,required=True)
    a=p.parse_args()
    source=Archive(a.supplement);plan,records,authority=audit(source)
    prior=Archive(a.prior_gemm)
    board=json.loads((ROOT/'docs/measurements/kpack_prefill_components_20260914/result.json').read_text())
    points=points_from_supplement(plan,records)+prior_points(prior,board,plan)
    if len({p['id'] for p in points})!=len(points):raise ValueError('duplicate workload')
    report=dict(schema='quactlize.component-policy.v1',sources={source.path.name:source.sha,prior.path.name:prior.sha},
        points=points,knots=policy_rows(points),components_audited=len(records),
        reducer_recheck=records['dense-m1-n4096-s4-recheck'],
        scope='SUM_OF_ISOLATED_COMPONENTS_NOT_MEASURED_E2E',
        small_m_full_dequant=False,external_adapters_included=False,device_admission='PENDING_COMPOSED_RUNTIME')
    report['evidence_sha256']=digest(report)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    a.header.write_text(header(report))
    print(f'KPACK_COST_POLICY audited={len(records)} points={len(points)} knots={len(report["knots"])}')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Generate exact and bucket decode tables from recorded PPU measurements.

No online timing. Historical TC calls, current calibration and the SIMT sweep
remain distinct cohorts. Cross-cohort choices are proposals for a model gate,
not contemporaneous performance admission. Q4 keeps its confirmed joint board.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_simt_formats as sweep
from quactlize.execution.simt_codegen import Config as SimtConfig, runtime_inventory

EVIDENCE = ROOT / 'docs/measurements/smallm_20260915.json.gz'
POLICY = ROOT / 'policies/kpack_smallm_v1.json'
SCHEMA = 'quactlize.smallm-table.v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_members(path, wanted):
    result = {}
    with tarfile.open(path) as source:
        for member in source:
            if wanted(member.name):
                if not member.isfile() or member.name in result:
                    raise ValueError('duplicate/nonregular evidence: ' + member.name)
                result[member.name] = source.extractfile(member).read()
    return result


def key(q, mode, n, k, tokens, channels=1):
    return (q, mode, n, k, tokens, channels)


def tc_config(symbol, split, algorithm, grid=0):
    match = re.fullmatch(r'(?:fqk_tc|fqg|sf|sfg)_q(\d+)(?:_l\d+)?(?:_a0)?'
                        r'_tm(\d+)_tn(\d+)_tk(\d+)_wm(\d+)_wn(\d+)_s(\d+)'
                        r'(?:_bc0)?_ap(\d+)_dn(\d+)(_persistent|_nonpersistent)?', symbol)
    if not match:
        raise ValueError('unrecognized parent: ' + symbol)
    q, tm, tn, tk, wm, wn, stages, ap, dn = map(int, match.groups()[:9])
    grouped = symbol.startswith(('fqg_', 'sfg_'))
    # Sparse grouped directory grids must be recomputed for the real request.
    persistent = algorithm in ('PERSISTENT', 'GROUPED_PERSISTENT')
    return dict(kind='tc', symbol=symbol, qtype=q, route=int(grouped)*2+int(symbol.startswith('sf')),
                tm=tm, tn=tn, tk=tk, wm=wm, wn=wn, stages=stages, ap=ap, dn=dn,
                parent_persistent=int(match[10]=='_persistent') if grouped and symbol.startswith('fqg_') else -1,
                split=int(split), grid_mode=2 if persistent else 0,
                grid_b=max(1, math.ceil(grid/72)) if persistent else 0)


def capture(simt, overnight, trace):
    raw = read_members(simt, lambda n: n in ('results/authority.json', 'results/summary.json') or n.endswith('/attempt-1.json'))
    authority = json.loads(raw['results/authority.json'])
    summary = json.loads(raw['results/summary.json'])
    if summary.get('status') != 'PASS' or summary.get('passed') != 200 or summary.get('failed'):
        raise ValueError('requires the complete 200-case SIMT cohort')
    points = []
    class Text:
        def __init__(self, data): self.data = data
        def read_text(self): return self.data.decode()
    for p in sweep.cases(sweep.FORMATS, range(1, 9)):
        data = raw[f'results/{p["id"]}/attempt-1.json']
        if not sweep.result_valid(Text(data), p, authority):
            raise ValueError('invalid SIMT receipt: ' + p['id'])
        r = json.loads(data)['result']
        new = r['best']['new']
        rounds = [x['samples_us'] for x in sorted(r['confirmation'], key=lambda x:x['round'])
                  if x['arm']=='new' and x['key']==new['key']]
        points.append(dict(key=key(p['qtype'], 0 if p['mode']=='dense' else 2,
                                   p['n'], p['k'], p['tokens'], p['channels']),
                           recipe=new['key'], samples=rounds, median_us=new['median_us'],
                           result_sha256=sha(data), proof=r['replay_proof'],
                           cache=r['cache'], ring_bytes=r['ring_bytes']))
    historical = read_members(overnight, lambda n:n=='results/heuristic-input.json')
    raw_tc = json.loads(historical['results/heuristic-input.json'])
    if raw_tc['scope'] != 'PER_ROUTE_FULL_OUTPUT_NO_PREPASS_AMORTIZATION':
        raise ValueError('historical TC scope differs')
    tc = []
    for item in raw_tc['rows']:
        r, m = item['request'], item['measurement']
        p, g = r['problem'], r['grouped']
        if r['route'] not in ('fq-dense', 'fq-grouped') or p['qtype']==12:
            continue
        if m['status'] != 'CONFIRMED_SELECTED_SET':
            continue
        grouped = r['route']=='fq-grouped'
        if grouped and (g['experts']!='256' or g['topk']!='8'):
            continue
        tokens = int(g['tokens']) if grouped else p['m']
        if not 1 <= tokens <= 8:
            continue
        times = m['round_medians_us']
        if len(times)!=3 or not all(math.isfinite(t) and t>0 for t in times):
            raise ValueError('historical TC samples differ')
        c = tc_config(m['symbol'], m['split'], m['algorithm'], m['grid'])
        # Grouped TC observations do not distinguish shared/slot-specific A.
        # They are core-call costs, not an indexed F32 endpoint admission.
        for channels in ((1,8) if grouped else (1,)):
            tc.append(dict(key=key(p['qtype'], 2 if grouped else 0, p['n'], p['k'], tokens, channels),
                           config=c, median_us=m['median_us'], samples=times,
                           source='overnight', scope='COMPLETE_FP16_TC_NO_EXTERNAL_ADAPTERS'))
    calibration = json.loads((ROOT/'policies/kpack_zw810_heuristic_v1.json').read_text())
    base = json.loads((ROOT/'policies/kpack_zw810_runtime_v1.json').read_text())
    configs = base['configurations'] | calibration['configurations']
    for e in calibration['entries']:
        q, op, n, k, experts, m, *_ = e['key']
        if q==12 or op!=0 or not 1<=m<=8:
            continue
        c = configs[e['config_id']]
        tc.append(dict(key=key(q,op,n,k,m), config=tc_config(c['symbol'],c['split'],c['algorithm']),
                       median_us=e['selected_us'], samples=e['selected_samples_us'],
                       source='calibration', scope='COMPLETE_FP16_TC_NO_EXTERNAL_ADAPTERS'))
    # Q8's existing TC pool has no sweep board; use actual decode trace costs.
    # These are profiled model observations, explicitly not cold-cohort peers.
    trace_rows = list(csv.DictReader(io.StringIO(Path(trace).read_text())))
    traces = defaultdict(list)
    for r in trace_rows:
        if r['arm']=='native' and r['q']=='8' and r['matrices']=='1':
            traces[int(r['n']),int(r['k'])].append(float(r['producer_us'])+float(r['extra_us']))
    splits = {(8192,2048):2,(4096,2048):4,(2048,4096):8,(512,2048):8,(2048,512):4}
    if set(traces)!=set(splits) or any(not v or not all(math.isfinite(t) and t>0 for t in v) for v in traces.values()):
        raise ValueError('Q8 model trace inventory differs')
    for (n,k), values in sorted(traces.items()):
        c = tc_config('sf_q8_a0_tm16_tn64_tk64_wm16_wn16_s2_bc0_ap0_dn64',splits[n,k],'TC_SPLIT')
        tc.append(dict(key=key(8,0,n,k,1),config=c,median_us=statistics.median(values),samples=values,
                       source='model-trace',scope='PROFILED_MODEL_PRODUCER_PLUS_REDUCER'))
    sources = {str(p.name):sha(p.read_bytes()) for p in (simt,overnight,trace)}
    sources.update({str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in (
        ROOT/'policies/kpack_zw810_heuristic_v1.json',ROOT/'policies/kpack_zw810_runtime_v1.json')})
    return dict(schema=SCHEMA, sources=sources, authority=authority, simt=points, tc=tc)


def bin2(n):
    return int(n).bit_length()-1


def fit(evidence):
    if evidence['schema']!=SCHEMA or len(evidence['simt'])!=200:
        raise ValueError('small-M evidence denominator differs')
    tc = defaultdict(list)
    for row in evidence['tc']:
        tc[tuple(row['key'])].append(row)
    simt = {tuple(r['key']):r for r in evidence['simt']}
    if len(simt)!=200:
        raise ValueError('duplicate SIMT context')
    rows, review = [], []
    for k in sorted(tc.keys() | simt.keys()):
        controls = tc[k]
        s = simt.get(k)
        best = min(controls, key=lambda r:r['median_us']) if controls else None
        choice = best['config'] if best else None
        decision = 'TC_RETAINED' if best else 'NO_TC_COMPARISON'
        if s:
            samples = s['samples']
            median = statistics.median(t for r in samples for t in r)
            if (len(samples)!=4 or any(len(r)!=11 for r in samples) or
                    not math.isclose(median,s['median_us'],rel_tol=1e-12) or
                    not all(math.isfinite(t) and t>0 for r in samples for t in r)):
                raise ValueError('SIMT confirmation differs')
            # Headroom against the fastest old core observation, not a slower
            # scalar baseline. No model-speedup or common-cache claim follows.
            if best and max(map(statistics.median,samples))*1.10 < min(min(c['samples']) for c in controls):
                fields = list(map(int,re.fullmatch(r'reuse-v(\d+)-c(\d+)-w(\d+)-p(\d+)-s(\d+)',s['recipe']).groups()))
                if SimtConfig(*fields) not in runtime_inventory(k[0]):
                    raise ValueError('SIMT recipe absent from compiled inventory')
                choice = dict(kind='simt', variant=fields[0], columns=fields[1], warps=fields[2], values=fields[3], split=fields[4])
                decision = 'CROSS_COHORT_SIMT_PROPOSAL'
        if choice:
            rows.append(dict(key=list(k),config=choice,decision=decision))
        if s:
            review.append(dict(key=list(k),simt_us=s['median_us'],tc_us=best['median_us'] if best else None,decision=decision))
    # An occupied logarithmic bucket chooses a medoid of its exact-table
    # winners. Runtime only searches these records; no fitted timing equation.
    groups = defaultdict(list)
    for i,r in enumerate(rows):
        q,op,n,k,m,ch = r['key']
        groups[q,op,ch,bin2(m-1)+1 if m>1 else 0,bin2(n),bin2(k)].append(i)
    buckets=[]
    for b,indices in sorted(groups.items()):
        def distance(i,j):
            return sum(abs(a-b)/max(a,b) for a,b in zip(rows[i]['key'][2:5],rows[j]['key'][2:5]))
        i=min(indices,key=lambda i:(sum(distance(i,j) for j in indices),i))
        buckets.append(dict(key=list(b),row=i))
    return dict(schema=SCHEMA,exact=rows,buckets=buckets,review=review,
                sources=evidence['sources'],scope='DECODE_1_TO_8_F32_ENDPOINTS_F16_ACTIVATIONS',
                bucket_scope='SAME_FORMAT_AND_OPERATOR_NEAREST_LOG_BUCKET_RUNTIME_VALIDATION_REQUIRED',
                performance_admission='PENDING_CONTEMPORANEOUS_MODEL_GATE',
                summary=dict(exact=len(rows),buckets=len(buckets),simt=sum(r['config']['kind']=='simt' for r in rows),
                             simt_candidates=len(simt),q4='UNCHANGED_CONFIRMED_JOINT_TABLE'))


def header(policy):
    configs = sorted({json.dumps(r['config'],sort_keys=True) for r in policy['exact']})
    lines=['// Generated by tools/fit_kpack_smallm.py. Do not edit table rows.', '#pragma once',
           '#include "kpack_zw810_heuristic_v1.hpp"',
           'namespace quactlize::smallm_data {',
           'using Config=quactlize_kpack_heuristic_v1::Config;',
           'struct Simt { int variant,columns,warps,values,split; };',
           'struct Choice { bool simt; Simt reader; Config tc; };',
           'inline constexpr Choice kChoices[]={']
    for text in configs:
        c=json.loads(text)
        if c['kind']=='simt':
            lines.append('    {true,{'+','.join(str(c[f]) for f in ('variant','columns','warps','values','split'))+'},{}},')
        else:
            mapping='0x51384b5032540001' if c['qtype']==8 else '0x514b504b54000001'
            values=[c[f] for f in ('qtype','route','tm','tn','tk','wm','wn','stages','ap','dn','parent_persistent','split','grid_mode','grid_b')]+[0]
            lines.append('    {false,{}, {'+','.join([json.dumps(c['symbol']),json.dumps(c['symbol']),'"SMALLM_TABLE"',*map(str,values),f'UINT64_C({mapping})'])+'}},')
    lines+=['};','struct Row { int q,mode,n,k,tokens,channels,choice; };','inline constexpr Row kExact[]={']
    for r in policy['exact']:
        lines.append('    {'+','.join(map(str,r['key']+[configs.index(json.dumps(r['config'],sort_keys=True))]))+'},')
    lines+=['};','struct Bucket { int q,mode,channels,m,n,k,row; };','inline constexpr Bucket kBuckets[]={']
    for b in policy['buckets']:
        lines.append('    {'+','.join(map(str,b['key']+[b['row']]))+'},')
    lines+=['};','} // namespace quactlize::smallm_data','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('simt','overnight','trace'):p.add_argument('--'+name,type=Path)
    p.add_argument('--evidence',type=Path,default=EVIDENCE)
    p.add_argument('--output',type=Path,default=POLICY)
    a=p.parse_args()
    if a.simt:
        e=capture(a.simt,a.overnight,a.trace)
        a.evidence.write_bytes(gzip.compress(json.dumps(e,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    else:e=json.loads(gzip.decompress(a.evidence.read_bytes()))
    policy=fit(e);policy['evidence_sha256']=sha(a.evidence.read_bytes())
    a.output.write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
    a.output.with_suffix('.hpp').write_text(header(policy))
    print('KPACK_SMALLM_TABLE '+json.dumps(policy['summary']))


if __name__=='__main__':main()

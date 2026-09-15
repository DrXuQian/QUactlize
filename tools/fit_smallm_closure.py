#!/usr/bin/env python3
"""Import sealed matched PPU samples and emit compute-specific decode tables.

The table is a measured-pool policy, not global optimality or model admission.
Open cells retain the previous route. GPU-only routing histograms never become
host lookup keys: a grouped row minimizes worst regret over measured profiles.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import random
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.smallm_closure.plan import candidate_id
from dev.smallm_closure.results import adjudicate, policy_review
from quactlize.runtime.tuning import digest, ROUTES
from quactlize.execution.simt_codegen import Config, runtime_inventory

SCHEMA = 'quactlize.smallm-matched.v1'
KEY = ('q', 'mode', 'n', 'k', 'experts', 'topk', 'channels', 'tokens', 'compute')


def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def proof_valid(r):
    if r['status'] != 'MEASURED':
        return
    proof = r['proof']
    full = r['phase'].endswith('-r0')
    if full and proof.get('guards') != 'PASS':
        raise ValueError('missing guard proof')
    for p in [proof['eager'] if full else proof, *proof.get('graph_replays', [])]:
        if p['bad'] or p['nonfinite'] or not p['cells'] or not math.isfinite(p['relative_l1_error']) or p['relative_l1_error'] >= p['tolerance']:
            raise ValueError('numerical proof is not admitted')
    if full and (
            len(proof.get('graph_replays', [])) != 3 or proof.get('zero_a_negative') != 'RED' or
            not {'low', 'units'} <= set(proof.get('ring_pointer_negatives', []))):
        raise ValueError('missing changed-graph / wrong-pointer / zero-A proof')


def capture(archive):
    """Stream, never extract, the large archive. Recompute every sealed receipt."""
    points, cells, saved, receipt_hashes = {}, defaultdict(dict), {}, []
    names = set()
    authority = {}
    with tarfile.open(archive) as source:
        for member in source:
            name = member.name
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or name in names or member.issym() or member.islnk():
                raise ValueError('unsafe/duplicate archive member: ' + name)
            names.add(name)
            if not member.isfile() or not name.endswith('.json'):
                continue
            if name in ('results/plan.json', 'results/policy-review.json', 'results/devices.json'):
                saved[name] = json.load(source.extractfile(member))
                continue
            match = re.fullmatch(r'results/cells/([^/]+)/(result|screen-[^/]+|confirm-[^/]+)\.json', name)
            if not match:
                continue
            r = json.load(source.extractfile(member))
            sealed = r.pop('receipt_sha256')
            if digest(r) != sealed:
                raise ValueError('corrupt result: ' + name)
            receipt_hashes.append((name, sealed))
            pid = match[1]
            a = r['authority']
            if a['point'] != pid or a['device']['compute_units'] != 72 or a['device']['warp'] != 32:
                raise ValueError('point/device identity differs')
            f = a['fixture']
            if f['cache'] != 'ROTATING_ACTIVE_WEIGHT_ADDRESSES' or f['active_ring_bytes'] < 2.25 * f['l2_bytes'] or f['l2_bytes'] <= 0:
                raise ValueError('cold ring receipt differs')
            ah = digest(a)
            if pid in authority and authority[pid] != ah:
                raise ValueError('mixed authority within one point')
            authority[pid] = ah
            if match[2] == 'result':
                points[pid] = r
            else:
                proof_valid(r)
                item = {k: r[k] for k in ('candidate', 'phase', 'status')}
                if 'samples_us' in r:
                    item['samples_us'] = r['samples_us']
                cells[pid][r['phase'], r['candidate']] = item
    plan = saved['results/plan.json']
    if digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256']:
        raise ValueError('plan digest differs')
    if any(candidate_id(c) != cid for cid, c in plan['candidates'].items()):
        raise ValueError('candidate identity differs')
    by_id = {p['id']: p for p in plan['points']}
    if set(points) - by_id.keys():
        raise ValueError('unexpected point')
    compact = []
    for pid in (p['id'] for p in plan['points'] if p['id'] in points):
        row=points[pid]
        p = by_id[pid]
        if row['point_spec'] != p or row['authority']['plan'] != plan['plan_sha256']:
            raise ValueError('point spec/source differs')
        dev = row['authority']['device']
        if saved['results/devices.json'][dev['visible_devices']] != dev:
            raise ValueError('assigned physical device differs')
        f = row['authority']['fixture']['fixture']
        if any(f[k] != p[k] for k in ('q', 'n', 'k', 'experts')):
            raise ValueError('fixture shape differs')
        # Equal screen times retain the runner's randomized insertion order,
        # not tar/directory order (the original shortlist uses a stable sort).
        order=list(p['candidates']);random.Random(pid).shuffle(order)
        screen = {cid: cells[pid]['screen',cid] for cid in order if ('screen',cid) in cells[pid]}
        confirms = {}
        for c in row['confirmed']:
            cid = c['candidate']
            epochs = defaultdict(dict)
            for (phase, candidate), r in cells[pid].items():
                m = re.fullmatch(r'confirm-e(\d+)-r(\d+)', phase)
                if candidate == cid and m and r['status'] == 'MEASURED':
                    epochs[int(m[1])][int(m[2])] = r['samples_us']
            matches = [e for e in epochs.values() if sorted(e) == list(range(plan['protocol']['rounds'])) and
                       [statistics.median(e[i]) for i in sorted(e)] == c['round_medians_us'] and
                       statistics.median(v for samples in e.values() for v in samples) == c['median_us']]
            if not matches:
                raise ValueError('summary has no matching complete raw confirmation')
            confirms[cid] = dict(status='MEASURED', rounds=[matches[0][i] for i in sorted(matches[0])])
        rebuilt = adjudicate(p, plan['candidates'], screen, confirms, plan['protocol'])
        for field in ('winner', 'confirmed', 'status', 'issues', 'structural'):
            left, right = rebuilt[field], row[field]
            if field in ('issues', 'structural'):
                left, right = (sorted(x, key=lambda v: json.dumps(v, sort_keys=True)) for x in (left, right))
            if left != right:
                raise ValueError(f'summary replay differs: {pid}/{field}: {left!r} != {right!r}; issues={rebuilt["issues"]!r}')
        compact.append({k: row[k] for k in ('point', 'status', 'winner', 'confirmed', 'issues')} |
                       dict(point_spec={k: p[k] for k in (*KEY, 'id', 'router')}))
    if policy_review(compact) != saved['results/policy-review.json']:
        raise ValueError('policy summary replay differs')
    result = dict(schema=SCHEMA, archive_sha256=sha_file(archive), plan_sha256=plan['plan_sha256'],
        receipt_count=len(receipt_hashes), receipt_root=digest(sorted(receipt_hashes)),
        expected_points=len(by_id), missing=sorted(by_id.keys()-points.keys()),
        points=compact, candidates=plan['candidates'], inventory=plan['inventory'],
        scope=plan['scope'], protocol=plan['protocol'])
    return result | dict(evidence_sha256=digest(result))


def config(c, q):
    if c['kind'] == 'simt':
        if Config(**c['recipe']) not in runtime_inventory(q):
            raise ValueError('measured SIMT recipe absent from production')
        return dict(kind='simt', **c['recipe'])
    if c['kind'] == 'q4':
        if q != 12:
            raise ValueError('Q4 reader on a different format')
        return dict(kind='q4', **c['recipe'])
    p = c['parent']
    return p | dict(kind='tc', route=ROUTES.index(p['route']), parent_persistent=p['persistent'],
                split=c['split'], grid_mode=c['grid_mode'], grid_b=c['grid_b'])


def fit(e):
    if e['schema'] != SCHEMA or digest({k: v for k, v in e.items() if k != 'evidence_sha256'}) != e['evidence_sha256']:
        raise ValueError('matched evidence identity differs')
    exact, excluded = [], []
    for r in policy_review(e['points']):
        q, mode, n, k, experts, topk, channels, tokens, compute = r['key']
        if r['status'] not in ('WITHIN_5_PERCENT', 'ROUTER_SENSITIVE_PARETO') or n % 256 or k % (512 if q in (11, 14) else 256):
            excluded.append(r)
            continue
        cid = r['minimax']['candidate']
        key = [q, mode, n, k, experts, topk, channels, tokens, int(compute == 'bf16')]
        exact.append(dict(key=key, config=config(e['candidates'][cid], q), candidate=cid,
            status=r['status'], worst_regret_pct=r['minimax']['worst_regret_pct'], profiles=r['profiles']))
    # Representative measured winner per occupied bucket. Runtime bounds the
    # donor distance and rechecks the actual shape, compute and ABI eligibility.
    groups = defaultdict(list)
    for i, r in enumerate(exact):
        q, mode, n, k, experts, topk, ch, m, compute = r['key']
        b = [q, mode, experts, topk, ch, (m-1).bit_length(), n.bit_length()-1, k.bit_length()-1, compute]
        if r['config']['kind'] != 'q4':
            groups[tuple(b)].append(i)
    buckets = []
    for key, indices in sorted(groups.items()):
        def score(i):
            return sum(sum(abs(math.log2(exact[i]['key'][a]/exact[j]['key'][a])) for a in (2,3,7)) for j in indices)
        donor = min(indices, key=lambda i: (score(i), exact[i]['worst_regret_pct'], i))
        buckets.append(dict(key=list(key), row=donor))
    return dict(schema=SCHEMA, exact=exact, buckets=buckets, excluded=excluded, missing=e['missing'],
        evidence_sha256=e['evidence_sha256'], archive_sha256=e['archive_sha256'],
        scope='F32_STORAGE_EXPLICIT_F16_BF16_COMPUTE_MATCHED_COMPLETE_DECODE_CALL',
        bucket_scope='SAME_FORMAT_COMPUTE_OPERATOR_EXPERTS_CHANNELS_TOKEN_BUCKET_MAX_2X_N_AND_K',
        admission='PPU_MATCHED_POOL_MODEL_INTEGRATION_PENDING',
        summary=dict(exact=len(exact), buckets=len(buckets), excluded=len(excluded),
                     router_sensitive=sum(r['status']=='ROUTER_SENSITIVE_PARETO' for r in exact)))


def header(policy):
    configs = sorted({json.dumps(r['config'], sort_keys=True) for r in policy['exact']})
    lines = ['// Generated by tools/fit_smallm_closure.py; no online tuning.', '#pragma once',
             '#include "kpack_zw810_heuristic_v1.hpp"', 'namespace quactlize::smallm_matched_data {',
             'using Config=quactlize_kpack_heuristic_v1::Config;',
             'struct Reader { int reader,variant,columns,warps,values,split; };',
             'struct Choice { int kind; Reader reader; Config tc; };',
             'inline constexpr Choice kChoices[]={']
    for text in configs:
        c = json.loads(text)
        if c['kind'] != 'tc':
            values = [c.get(f, 0 if f=='reader' else 1) for f in ('reader','variant','columns','warps','values','split')]
            lines.append('    {'+str(2 if c['kind']=='q4' else 1)+',{'+','.join(map(str,values))+'},{}},')
        else:
            mapping = {8:'0x51384b5032540001',12:'0x51344b5034540001'}.get(c['qtype'],'0x514b504b54000001')
            values = [c[f] for f in ('qtype','route','tm','tn','tk','wm','wn','stages','ap','dn','parent_persistent','split','grid_mode','grid_b')]+[0]
            lines.append('    {0,{}, {'+','.join([json.dumps(c['symbol']),json.dumps(c['symbol']),'"MATCHED_SMALLM"',*map(str,values),f'UINT64_C({mapping})'])+'}},')
    lines += ['};','struct Row { int q,mode,n,k,experts,topk,channels,tokens,compute,choice,router_sensitive; };','inline constexpr Row kExact[]={']
    for r in policy['exact']:
        index = configs.index(json.dumps(r['config'],sort_keys=True))
        lines.append('    {'+','.join(map(str,r['key']+[index,int(r['status']=='ROUTER_SENSITIVE_PARETO')]))+'},')
    lines += ['};','struct Bucket { int q,mode,experts,topk,channels,m,n,k,compute,row; };','inline constexpr Bucket kBuckets[]={']
    for b in policy['buckets']:
        lines.append('    {'+','.join(map(str,b['key']+[b['row']]))+'},')
    lines += ['};','inline constexpr Row kOpen[]={']
    for r in policy['excluded']:
        key=r['key'][:-1]+[int(r['key'][-1]=='bf16')]
        lines.append('    {'+','.join(map(str,key+[0,0]))+'},')
    lines += ['};','} // namespace quactlize::smallm_matched_data','']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path)
    p.add_argument('--evidence',type=Path,default=ROOT/'docs/measurements/smallm_matched_20260915.json.gz')
    p.add_argument('--output',type=Path,default=ROOT/'policies/kpack_smallm_matched_v1.json')
    a = p.parse_args()
    if a.archive:
        e = capture(a.archive)
        a.evidence.parent.mkdir(parents=True,exist_ok=True)
        a.evidence.write_bytes(gzip.compress(json.dumps(e,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    else:
        e = json.loads(gzip.decompress(a.evidence.read_bytes()))
    policy = fit(e)
    a.output.write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
    a.output.with_suffix('.hpp').write_text(header(policy))
    print('SMALLM_MATCHED_POLICY '+json.dumps(policy['summary']),flush=True)


if __name__ == '__main__':
    main()

"""Production ABI/catalog support."""
import copy
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))


from tools.fit_smallm_closure import header as matched_header
FIELDS = ('variant', 'columns', 'warps', 'values', 'split')


EVIDENCE = ROOT/'policies/kpack_decode_winners_v1.json'


COMPONENTS = ROOT/'policies/kpack_component_decode_v1.json'


OUTPUT = ROOT/'policies/kpack_smallm_effective_v1.hpp'


IMPLEMENTATIONS = ROOT/'quactlize/execution/measured_decode.inc'


ARCHIVES = {
    'tp2': '5944aa7547e4dd5c9b87dae0f906102d7df5d4e39f42d7007c25731408fee075',
    'fallback': '6a1b416f7ec91683b15ac886b0d364bcc275a5aa19ca5db8ce5d70ef46ade4c6',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def effective(evidence, *, include_components=True, include_refresh=True, include_dense=True, include_token_cta=True,
              include_moe=True):
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
    if include_components:
        from quactlize.dispatch.catalog_components import apply
        result = apply(json.loads(COMPONENTS.read_text()), result)
        if include_refresh:
            from quactlize.execution.q8_refresh import apply as bind_q8
            result = bind_q8(result)
            if include_dense:
                from quactlize.dispatch.catalog_dense import apply as bind_dense
                result = bind_dense(result)
                if include_token_cta:
                    from quactlize.execution.token_cta_policy import apply as bind_tokens
                    result = bind_tokens(result)
                    if include_moe:
                        from quactlize.execution.moe_decode_policy import apply as bind_moe
                        result = bind_moe(result)
    return result


def header(policy):
    prefix=next((i for i,r in enumerate(policy['exact']) if r.get('source')=='dense-m-exact-v1'),len(policy['exact']))
    seed=[]
    if any('previous_config' in r for r in policy['exact']):
        prior=[json.dumps(r.get('previous_config',r['config']),sort_keys=True) for r in policy['exact']]
        seed=sorted(set(prior[:prefix]))+sorted(set(prior)-set(prior[:prefix]))
    text=matched_header(policy,config_prefix=prefix,config_seed=seed).replace('tools/fit_smallm_closure.py','tools/generate_decode_selection.py')
    text=text.replace('smallm_matched_data','smallm_effective_data')
    # Generic donor choices preserve coverage where an exact Q4 body bakes N/K.
    guards=matched_header(dict(exact=[dict(r,config=r['guard']) for r in policy['exact']],buckets=[],excluded=[]),config_prefix=prefix)
    guard_choices=guards.split('inline constexpr Choice kChoices[]={\n',1)[1].split('\n};',1)[0]
    guard_rows=guards.split('inline constexpr Row kExact[]={\n',1)[1].split('\n};',1)[0]
    indices=[int(line.strip().strip('{},').split(',')[-2]) for line in guard_rows.splitlines()]
    extra='inline constexpr Choice kGuards[]={\n'+guard_choices+'\n};\n'
    extra+='inline constexpr int kGuardIndex[]={'+','.join(map(str,indices))+'};\n'
    extra+='inline constexpr int kPolicy[]={'+','.join(str(r['policy']) for r in policy['exact'])+'};\n'
    extra+='inline constexpr int kDequant[]={'+','.join(str(r.get('dequant_config',-1)) for r in policy['exact'])+'};\n'
    # Exact-only Q8 admission must not change an unseen shape's old donor.
    donors = [dict(r, config=r.get('bucket_config', r['guard'] if r['config']['kind']=='q4' else r['config']))
              for r in policy['exact']]
    buckets = matched_header(dict(exact=donors, buckets=[], excluded=[]),config_prefix=prefix)
    choices = buckets.split('inline constexpr Choice kChoices[]={\n',1)[1].split('\n};',1)[0]
    rows = buckets.split('inline constexpr Row kExact[]={\n',1)[1].split('\n};',1)[0]
    indices = [int(line.strip().strip('{},').split(',')[-2]) for line in rows.splitlines()]
    extra += 'inline constexpr Choice kBucketChoices[]={\n'+choices+'\n};\n'
    extra += 'inline constexpr int kBucketIndex[]={'+','.join(map(str,indices))+'};\n'
    extra += 'inline constexpr int kBucketDequant[]={'+','.join(str(r.get(
        'bucket_dequant_config', -1 if r['config']['kind']=='q4' else r.get('dequant_config',-1)))
        for r in policy['exact'])+'};\n'
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
    import argparse
    from tools import paired_decode_selection as paired
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    evidence = json.loads(EVIDENCE.read_text())
    policy = effective(evidence)
    outputs = ((OUTPUT, header(policy)), (IMPLEMENTATIONS, implementations(evidence)),
               (paired.OUTPUT, paired.header(json.loads(paired.EVIDENCE.read_text()))))
    for path, text in outputs:
        if args.check:
            if path.read_text() != text:
                raise ValueError('stale generated selection: ' + str(path))
        else:
            path.write_text(text)
    print(f'DECODE_SELECTION_GENERATED exact={len(policy["exact"])} buckets={len(policy["buckets"])}')


if __name__ == '__main__':
    main()

"""Score a FROZEN blind pool against same-point historical confirmations.

Never called by Prescreen. This is a development diagnostic, not fresh PPU
evidence, component samples, a prediction for unmeasured recipes, or proof
against later production promotions. Do not compare microseconds across eras.
"""
from collections import Counter
import gzip
import json
import math

from dev.strategy.space import factored_recall
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest
from tools.export_kpack_strategy import recipe as normalize


def compare(w, generated, measured):
    """measured contains only compatible, independently admitted old rows."""
    for _, cost in measured:
        if not math.isfinite(cost) or cost<=0:
            raise ValueError('nonfinite/nonpositive historical confirmation')
    if not measured:
        return dict(status='NO_COMPATIBLE_HISTORICAL_REFERENCE',regret_pct=None)
    best=min(cost for _,cost in measured)
    matched=[(r,cost) for r,cost in measured if factored_recall(w,generated,[r])['recalled']]
    if not matched:
        return dict(status='NO_MEASURED_RULE_CANDIDATE',historical_best_us=best,regret_pct=None)
    selected,cost=min(matched,key=lambda x:(x[1],x[0].key))
    regret=100*(cost/best-1)
    return dict(status='HISTORICAL_BEST_INCLUDED' if regret<=1e-7 else
                       'WITHIN_5PCT_NOT_STRICT_PARITY' if regret<=5 else 'KNOWN_HISTORICAL_GAP',
                historical_best_us=best,best_measured_generated_us=cost,regret_pct=regret,
                recipe=selected.key)


def replay(plan,root):
    """Reveal old targets after generation; never supplement the candidate list."""
    if not plan.get('factorized_search'):
        return dict(status='UNSUPPORTED_FLAT_PLAN',performance_proven=False)
    path=root/'docs/measurements/smallm_matched_20260915.json.gz'
    decode=json.loads(gzip.decompress(path.read_bytes()))
    prefill=json.loads((root/'docs/measurements/kpack_component_policy_20260914.json').read_text())
    if decode['scope']!='TP1_DENSE_AND_INDEXED_TOKENS_1_TO_8_F32_IO':
        raise ValueError('historical decode contract differs')
    if prefill['scope']!='SUM_OF_ISOLATED_COMPONENTS_NOT_MEASURED_E2E':
        raise ValueError('historical prefill contract differs')
    index={}
    for row in decode['points']:
        p=row['point_spec']
        if row['status']!='MEASURED_POOL_CLOSED':continue
        key=(p['q'],p['n'],p['k'],p['tokens'],p['mode'],p['compute'],p['experts'],p['topk'],p['channels'],
             p.get('router','real') if p['mode'] else '')
        if key in index:raise ValueError('duplicate historical decode point')
        measured=[]
        for entry in row['confirmed']:
            c=decode['candidates'][entry['candidate']]
            r=(normalize(dict(c['parent'],kind='tc',split=c['split'],grid_mode=c['grid_mode'],grid_b=c['grid_b']))
               if c['kind']=='tc' else normalize(dict(c['recipe'],kind=c['kind'])))
            measured.append((r,entry['median_us']))
        if not measured or min(cost for _,cost in measured)!=row['winner']['median_us']:
            raise ValueError('historical confirmation winner differs')
        index[key]=measured
    prefills={}
    for row in prefill['points']:
        if row['experts']!=1:continue  # old F16 MoE is NOT current BF16 evidence
        key=(row['q'],row['n'],row['k'],row['tokens'])
        if key in prefills:raise ValueError('duplicate historical prefill point')
        measured=[]
        for c in row['candidates']:
            if not c['stable']:continue
            if not math.isclose(c['cost_us'],c['gemm_us']+c['dequant_us'],rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError('historical component sum differs')
            r=(Recipe.make('full','full',dequant_config=c['dequant_config'],provider='bf16-default')
               if c['route']==2 else normalize(c['tactic'],c['route'],c['dequant_config']))
            measured.append((r,c['cost_us']))
        prefills[key]=measured
    rows=[]
    for point in plan['points']:
        w=Workload(**point['workload']);scope='NO_COMPATIBLE_SCOPE';measured=[]
        if w.m<=8 and w.storage=='f32' and w.fusion=='none':
            mode=2 if w.operation=='indexed' else 0
            key=(w.qtype,w.n,w.k,w.m,mode,w.compute,w.experts,w.topk,w.channels,w.router or ('real' if mode else ''))
            if w.operation in ('dense','indexed'):measured=index.get(key,[])
            scope='OLD_MATCHED_COMPLETE_CALL'
        elif w.operation=='dense' and w.compute=='f16' and w.storage=='f32' and w.fusion=='none':
            measured=prefills.get((w.qtype,w.n,w.k,w.m),[])
            scope='OLD_F16_COMPONENT_SUM_NOT_E2E'
        result=compare(w,[Recipe.from_dict(c) for c in point['candidates']],measured)
        rows.append(dict(point=point['id'],scope=scope,**result))
    counts={scope:dict(Counter(r['status'] for r in rows if r['scope']==scope)) for scope in sorted({r['scope'] for r in rows})}
    return dict(schema='quactlize.historical-blind-replay.v1',counts=counts,rows=rows,
        evidence=dict(decode=digest(decode),prefill=digest(prefill)),
        scope='DEVELOPMENT_REPLAY_OLD_CONFIRMED_POOL_NOT_CURRENT_PRODUCTION_OR_NEW_DEVICE',
        generated_pool_was_not_mutated=True,unmeasured_candidates='UNKNOWN_NOT_LOSERS',
        original_wiring_ab_still_required=True,performance_proven=False,production_changed=False)

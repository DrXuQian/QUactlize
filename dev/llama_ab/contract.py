"""Frozen workload denominator and strict, paired native regression verdicts."""
from collections import Counter
from dataclasses import asdict, replace
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics
import re

from quactlize.runtime.tuning import digest
from quactlize.dispatch.prefill_policy import EVIDENCE as PREFILL_EVIDENCE

ROOT = Path(__file__).resolve().parents[2]
SCOPE = 'LLAMA_EAGER_COMPLETE_OPERATOR_EVENTS_NOT_KERNEL_ONLY_OR_MODEL_TPOT'
MEASURED_STATUSES = frozenset(('PASS', 'REGRESSION', 'INCONCLUSIVE'))


@lru_cache(maxsize=1)
def confirmed_prefill():
    evidence=json.loads(PREFILL_EVIDENCE.read_text())
    return {p['point']:p for p in evidence['points']}


def inventory():
    from tools.export_kpack_strategy import load_sources, small_workload
    from dev.strategy.plan import workloads
    evidence = json.loads((ROOT / 'policies/kpack_component_decode_v1.json').read_text())
    rows = []
    for row in evidence['rows']:
        for profile in row['profiles']:
            w = replace(small_workload(row['key']), router=profile['router'])
            if w.key != profile['point']:
                raise ValueError('frozen decode profile identity differs')
            rows.append(dict(id=w.key, cohort='decode', workload=asdict(w)))
    table, _ = load_sources()
    for w in workloads(table, 'prefill'):
        rows.append(dict(id=w.key, cohort='prefill', workload=asdict(w)))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate workload')
    counts = Counter((r['cohort'], r['workload']['operation']) for r in rows)
    if counts != {('decode', 'dense'): 530, ('decode', 'indexed'): 584,
                  ('prefill', 'dense'): 280, ('prefill', 'indexed'): 150}:
        raise ValueError('workload denominator changed; review before extending the gate')
    return sorted(rows, key=lambda r:r['id'])


def family(row):
    w = row['workload']
    return digest({k:w[k] for k in ('qtype', 'n', 'k', 'experts', 'operation', 'compute')})


def pilot(rows):
    # Format/operator/cohort plus every selected prefill route. The old
    # smallest-shape-only pilot never exercised FULL, even when it won most
    # dense points. No new config or workload is introduced here.
    selected = {}
    for row in sorted(rows, key=lambda r:(r['workload']['n'] * r['workload']['k'],
                                         r['workload']['m'], r['id'])):
        w = row['workload']
        route=confirmed_prefill()[row['id']]['recipe']['route'] if row['cohort']=='prefill' else ''
        selected.setdefault((row['cohort'], w['operation'], w['qtype'],route), row)
    return sorted(selected.values(), key=lambda r:r['id'])


def phase_points(rows, phase):
    if phase == 'all':
        return list(rows)
    if phase == 'pilot':
        return pilot(rows)
    raise ValueError('unknown A/B phase: ' + phase)


def coverage_breakdown(rows, records):
    """Expose missing/failed/slow points in each format/operator cohort."""
    groups = {}
    for row in rows:
        w = row['workload']
        groups.setdefault((row['cohort'], w['operation'], w['qtype']), []).append(row)
    return [dict(cohort=key[0], operation=key[1], qtype=key[2],
                 **coverage(points, {r['id']: records[r['id']] for r in points if r['id'] in records}))
            for key, points in sorted(groups.items())]


def judge(raw):
    if raw.get('correctness') != 'PASS' or raw.get('scope') != SCOPE:
        raise ValueError('timing lacks numerical proof or full-call scope')
    rounds = raw['rounds']
    if len(rounds) < 6:
        raise ValueError('at least six paired rounds required')
    ratios, native, candidate, order_drift = [], [], [], []
    for row in rounds:
        if [b['arm'] for b in row] != [0, 1, 1, 0]:
            raise ValueError('round is not ABBA')
        by_arm = [[], []]
        for block in row:
            values = block['event_us']
            if len(values) < 3 or any(not isinstance(x, (int,float)) or
                                     not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError('missing/nonfinite samples')
            by_arm[block['arm']].extend(values)
        a, b = map(statistics.median, by_arm)
        native.append(a); candidate.append(b); ratios.append(b/a)
        for i,j in ((0,3),(1,2)):
            x,y=statistics.median(row[i]['event_us']),statistics.median(row[j]['event_us'])
            order_drift.append(abs(x-y)/min(x,y))
    ratio = statistics.median(ratios)
    # The entire observed round envelope must be on the passing side. This
    # is an empirical stability gate, NOT a statistical confidence interval.
    status = ('INCONCLUSIVE' if max(order_drift) > .05 else 'PASS' if max(ratios) <= 1 else
              'REGRESSION' if min(ratios) > 1 else 'INCONCLUSIVE')
    return dict(status=status, native_us=statistics.median(native),
                candidate_us=statistics.median(candidate), ratio=ratio,
                round_ratios=ratios, ratio_range=[min(ratios), max(ratios)],
                maximum_abba_order_drift=max(order_drift),allowed_slowdown_pct=0, confidence_interval=False)


def coverage(rows, records):
    expected = {r['id'] for r in rows}
    if set(records) - expected:
        raise ValueError('unexpected result outside the frozen inventory')
    counts = Counter(records[k]['status'] for k in expected if k in records)
    missing = sorted(expected - set(records))
    return dict(status='PASS' if counts.get('PASS') == len(expected) else 'NOT_ADMITTED',
                expected=len(expected), completed=len(expected)-len(missing),
                measured=sum(counts.get(s, 0) for s in MEASURED_STATUSES),
                counts=dict(counts), missing=missing, scope=SCOPE,
                model_tpot_admitted=False, production_changed=False)


def route_proof(text, row):
    """Require a selected K-pack plan, never infer compute from buffer placement."""
    name = 'ab_' + row['id'][:20] + '_'
    lines = [s for s in text.splitlines() if ('tensor='+name) in s and
             ('[quactlize-plan]' in s or '[quactlize-dense]' in s)]
    # Public dense caller prints a different format from the historical PoC.
    if not lines:
        lines = [s for s in text.splitlines() if name in s and 'quactlize:' in s and
                 'route=' in s and 'config=' in s]
    if not lines:
        raise ValueError('candidate has no selected compute plan; load placement alone is insufficient')
    if any(name in s and ('retain legacy' in s.lower() or 'policy miss' in s.lower() or 'route=legacy' in s.lower())
           for s in text.splitlines()):
        raise ValueError('candidate silently retained a legacy route')
    w=row['workload']
    for line in lines:
        fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
        if any(fields.get(k)!=str(w[key]) for k,key in (('q','qtype'),('n','n'),('k','k'))):
            raise ValueError('logged selected shape differs')
        bf16=fields.get('activation')=='BF16' or fields.get('route')=='full-bf16'
        if w['operation']=='indexed' and (fields.get('op')!='grouped' or not bf16):
            raise ValueError('indexed selected plan is not BF16 grouped compute')
        if w['operation']=='dense' and fields.get('activation') not in (None,'FP16','BF16'):
            raise ValueError('unexpected dense arithmetic')
        if row.get('cohort')=='prefill':
            expected=confirmed_prefill().get(row['id'])
            if expected is None:raise ValueError('prefill AB point lacks confirmed selector evidence')
            recipe=expected['recipe'];p=dict(recipe['parameters'])
            route='full-bf16' if recipe['kind']=='full' else recipe['route']
            if (fields.get('policy')!='19' or fields.get('route')!=route or fields.get('channels')!='1' or
                    fields.get('dequant')!=str(p['dequant_config'])):
                raise ValueError('actual prefill route/helper differs from confirmed composition')
            if recipe['kind']=='tc' and (fields.get('parent')!=p['symbol'] or fields.get('split')!=str(p['split'])):
                raise ValueError('actual prefill TC recipe differs from confirmed selection')
    return sorted(set(lines))

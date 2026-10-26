#!/usr/bin/env python3
"""Audit final choices locally against confirmed evidence; no GPU or tuning."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.generate_decode_selection import EVIDENCE, COMPONENTS, FIELDS, effective, header, implementations, OUTPUT, IMPLEMENTATIONS
from tools.verify_kpack_dispatch import verify
from quactlize.dispatch.planning import plan_smallm, validate_inventory
from quactlize.dispatch.native import Dispatch
from quactlize.execution.native import Call, arrangement
from quactlize.runtime.compiler import sha


def check_winner(selected, winner):
    if selected['status'] or selected['policy'] != winner.get('policy',12):
        raise ValueError('confirmed winner was not selected: ' + winner['point']['name'])
    c = winner['config']
    if c['kind'] == 'tc':
        good = (selected['kind'] == 0 and selected['parent']['symbol'] == c['symbol']
                and all(selected[f] == c[f] for f in ('split','grid_mode','grid_b'))
                and all(selected['parent'][f] == c[f] for f in ('route','tm','tn','tk','wm','wn','stages','ap','dn')))
    elif c['kind']=='q4':
        good = selected['kind']==2 and selected['config']=={k:v for k,v in c.items() if k!='kind'}
    else:
        good = selected['kind'] == 1 and selected['config'] == {f: c[f] for f in FIELDS}
        if winner['implementation'] == 'constant-split':
            impl = selected['implementation']
            good &= impl['measured'] == winner['point']['name'].replace('-', '_')
            good &= all(impl[f] == winner['body'][f] for f in ('fixed', 'changes', 'hoist'))
    if not good:
        raise ValueError('selected implementation differs from evidence: ' + winner['point']['name'])


def audit(bundle, output):
    m = verify(bundle)
    if m['policy_hashes'].get('policies/kpack_smallm_effective_v1.hpp')!=sha(OUTPUT):
        raise ValueError('package uses a different effective catalog')
    e = json.loads(EVIDENCE.read_text())
    policy = effective(e)
    if header(policy) != OUTPUT.read_text() or implementations(e) != IMPLEMENTATIONS.read_text():
        raise ValueError('generated catalog/implementation is stale')
    output.mkdir(parents=True, exist_ok=False)
    plan = plan_smallm(output)
    if len(plan['requests']) != len(policy['exact']) or any(r['status'] for r in plan['requests']):
        raise ValueError('production selector omitted a catalog entry')
    available = validate_inventory(plan, m['execution_receipt'], m['modules'], jit=m.get('jit_required', False))
    rows = {tuple(r['request']): r for r in plan['requests']}
    winners = [w for w in e['rows'] if w['automatic'] and not w['point']['paired']]
    catalog={tuple(r['key']):r for r in policy['exact']}
    for w in winners:
        current=catalog[tuple(w['key'])]
        if current['policy']==18:
            w=dict(w,config=current['config'],policy=18,implementation='existing')
        check_winner(rows[tuple(w['key'])], w)
    components=json.loads(COMPONENTS.read_text())
    from quactlize.execution.token_cta_policy import load as token_table
    tokens=token_table()
    replacements={tuple(w['key']):w for w in tokens['rows'] if w['changed']}
    for w in components['rows']:
        if not w['changed']:continue
        r=rows[tuple(w['key'])]
        if tuple(w['key']) in replacements:
            c=replacements[tuple(w['key'])]['selected']
            w=dict(w,config=c['config'],dequant_config=c['dequant_config'])
        check_winner(r,dict(w,policy=18,implementation='existing',point={'name':str(w['key'])}))
        if r['dequant_config']!=w['dequant_config']:
            raise ValueError('selected SF helper differs from component confirmation')
    for w in tokens['rows']:
        r=rows[tuple(w['key'])];c=w['selected'];live=catalog[tuple(w['key'])]
        check_winner(r,dict(config=c['config'],policy=live['policy'],implementation='existing',point={'name':str(w['key'])}))
        if c['implementation']['kind']=='token-cta' and (
                r['implementation'].get('cta_tokens')!=w['key'][7] or
                r['implementation'].get('cta_window')!=c['implementation']['window']):
            raise ValueError('selected token CTA implementation differs from confirmed body')
        if c['implementation']['kind']=='q8-address' and (
                r['implementation'].get('q8_body')!=c['implementation']['body']):
            raise ValueError('selected Q8 address implementation differs from confirmed body')
    public_count = 0
    d = Dispatch(bundle)
    try:
        # SIMT queries use only the published host dispatcher. TC decisions
        # use the same pure selector above; loading a TC module is a GPU task.
        for r in plan['requests']:
            if r['kind'] != 1:
                continue
            q, mode, n, k, experts, top, channels, tokens, compute = r['request']
            call = Call(version=1, size=C.sizeof(Call), qtype=q, mode=mode, n=n, k=k,
                        experts=experts, topk=top, channels=channels, rows=tokens*top,
                        input_type=1, a_row_stride=k, a_token_stride=k*channels,
                        ids_stride=top, out_row_stride=n)
            choice = d.query_smallm_matched(call, arrangement(q), compute)
            if (choice is None or choice.base.kind != 1 or choice.base.policy != r['policy'] or
                    {f: getattr(choice.base.simt, f) for f in FIELDS} != r['config'] or
                    [choice.base.source_n, choice.base.source_k, choice.base.source_tokens] != r['donor']):
                raise ValueError('published C ABI differs from selector: ' + str(r['request']))
            public_count += 1
    finally:
        d.close()
    result = dict(schema='quactlize.decode-selection-audit.v1', status='PASS', gpu_used=False,
                  evidence_sha256=sha(EVIDENCE), manifest_sha256=sha(bundle/'manifest.json'),
                  component_evidence_sha256=sha(COMPONENTS), component_confirmations=components['summary'],
                  token_cta_evidence_sha256=tokens['evidence_sha256'], token_cta_exact_points=len(tokens['rows']),
                  catalog_rows=len(rows), public_simt_queries=public_count,
                  confirmed_ordinary_winners=len(winners), required=available['required'],
                  tc_module_loading='DEFERRED_TO_DEVICE_GATE',
                  scope='LOCAL_SELECTION_AND_INVENTORY_NOT_NEW_DEVICE_PERFORMANCE')
    (output/'selection.json').write_text(json.dumps(plan, indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('DECODE_SELECTION_AUDIT PASS catalog={} public_simt={} confirmed_winners={} GPU=0'.format(
        len(rows), public_count, len(winners)), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit(a.bundle.resolve(strict=True), a.output.resolve())

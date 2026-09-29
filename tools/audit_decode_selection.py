#!/usr/bin/env python3
"""Audit final choices locally against confirmed evidence; no GPU or tuning."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.generate_decode_selection import EVIDENCE, FIELDS, effective, header, implementations, OUTPUT, IMPLEMENTATIONS
from tools.verify_kpack_dispatch import verify
from quactlize.dispatch.planning import plan_smallm, validate_inventory
from quactlize.dispatch.native import Dispatch
from quactlize.execution.native import Call, arrangement
from quactlize.runtime.compiler import sha


def check_winner(selected, winner):
    if selected['status'] or selected['policy'] != 12:
        raise ValueError('confirmed winner was not selected: ' + winner['point']['name'])
    c = winner['config']
    if c['kind'] == 'tc':
        good = (selected['kind'] == 0 and selected['parent']['symbol'] == c['symbol']
                and selected['split'] == c['split'])
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
    for w in winners:
        check_winner(rows[tuple(w['key'])], w)
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

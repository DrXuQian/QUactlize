#!/usr/bin/env python3
"""Rank prototype configs per shape against the in-run production baseline."""
import json, sys, collections
BPE = {8: 34/32, 10: 84/256, 11: 110/256, 12: 144/256, 13: 176/256, 14: 210/256}
top = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rows:
    if r['status'] != 'OK':
        print('BAD', r['case']['id'], r['status'], r.get('error', ''))
        continue
    cid = r['case']['id']
    sid = cid.rsplit('-', 1)[0]
    by[sid].append(r)
for sid, rs in by.items():
    base = [r for r in rs if r['case']['id'].endswith('-base')]
    prot = sorted([r for r in rs if not r['case']['id'].endswith('-base')], key=lambda r: r['call_sum_us']['mean'])
    c = rs[0]['case']
    b = c['n'] * c['k'] * BPE[c['qtype']] * (8 * c['m'] if c['kind'] == 'moe' else 1)
    bm = base[0]['call_sum_us']['mean'] if base else float('nan')
    print(f"== {sid}: base {bm:8.2f}us ({b/bm/1e3:5.0f} GB/s) [{base[0]['config'][:44] if base else ''}] n={len(prot)}")
    for r in prot[:top]:
        t = r['call_sum_us']['mean']
        print(f"     {r['case']['id'].split('-')[-1]:14s} {t:8.2f}us ratio {t/bm:5.3f} ({b/t/1e3:5.0f} GB/s) "
              f"comps {[round(x['us']['mean'], 2) for x in r['components']]} regs {r['components'][0]['regs']}")

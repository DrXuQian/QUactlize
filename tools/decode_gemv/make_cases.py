#!/usr/bin/env python3
"""Expand the workload manifest into qgemv_bench case lists (unique shape x M)."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BPE = {8: 34 / 32, 10: 84 / 256, 11: 110 / 256, 12: 144 / 256, 13: 176 / 256, 14: 210 / 256}


def case_id(s, m):
    if s['kind'] == 'moe':
        return f"e-q{s['qtype']}-n{s['n']}-k{s['k']}-e{s['experts']}-t{s['topk']}-c{s['channels']}-m{m}"
    return f"d-q{s['qtype']}-n{s['n']}-k{s['k']}-m{m}"


def cases(manifest, ms, shape_filter=None):
    out = []
    for s in manifest['unique_shapes']:
        if shape_filter and not shape_filter(s):
            continue
        for m in ms:
            c = dict(id=case_id(s, m), kind=s['kind'], qtype=s['qtype'], n=s['n'], k=s['k'], m=m)
            if s['kind'] == 'moe':
                c.update(experts=s['experts'], topk=s['topk'], channels=s['channels'])
            else:
                c.update(compute=0)
            c['bytes_m1'] = s['n'] * s['k'] * BPE[s['qtype']] * (s['topk'] if s['kind'] == 'moe' else 1)
            out.append(c)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', default='1-8')
    p.add_argument('--out', required=True)
    p.add_argument('--batches', type=int, default=1, help='split into N balanced files (out_<i>.json)')
    p.add_argument('--kind', choices=('dense', 'moe'))
    p.add_argument('--qtype', type=int)
    args = p.parse_args()
    lo, _, hi = args.m.partition('-')
    ms = list(range(int(lo), int(hi or lo) + 1))
    manifest = json.loads((ROOT / 'inventory/workload_manifest.json').read_text())
    flt = lambda s: (not args.kind or s['kind'] == args.kind) and (not args.qtype or s['qtype'] == args.qtype)
    cs = cases(manifest, ms, flt)
    if args.batches == 1:
        Path(args.out).write_text(json.dumps(cs, indent=0) + '\n')
        print(len(cs), 'cases ->', args.out)
        return
    # Balance by weight bytes (dominates generation/packing wall time).
    buckets = [[] for _ in range(args.batches)]
    load = [0.0] * args.batches
    for c in sorted(cs, key=lambda c: -c['bytes_m1']):
        i = load.index(min(load))
        buckets[i].append(c)
        load[i] += c['bytes_m1'] + 2e7
    stem = Path(args.out)
    for i, b in enumerate(buckets):
        path = stem.with_name(f'{stem.stem}_{i}.json')
        path.write_text(json.dumps(sorted(b, key=lambda c: c['id']), indent=0) + '\n')
        print(len(b), 'cases ->', path)


if __name__ == '__main__':
    main()

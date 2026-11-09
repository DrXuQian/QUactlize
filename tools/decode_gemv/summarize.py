#!/usr/bin/env python3
"""Aggregate parsed Asys rounds into per-shape and per-model decode GEMV costs.

Input: one or more run directories produced by ab_rounds.py (round*/<arm>/parsed.jsonl).
Per case and arm: pooled samples over rounds -> mean/median of complete-call time
(sum of the call's kernel durations). T_model(M) = sum(calls x mean complete-call time).
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def case_key(c):
    return c['id']


def load_runs(dirs):
    data = defaultdict(lambda: defaultdict(list))  # arm -> case id -> list of parsed records
    for d in dirs:
        for f in sorted(Path(d).glob('round*/*/parsed.jsonl')):
            arm = f.parent.name
            if '.contaminated' in arm:
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get('status') == 'OK':
                    r['round'] = f.parent.parent.name
                    data[arm][case_key(r['case'])].append(r)
    return data


def pooled(records):
    means = [r['call_sum_us']['mean'] for r in records]
    medians = [r['call_sum_us']['median'] for r in records]
    comps = defaultdict(list)
    for r in records:
        for c in r['components']:
            comps[c['position']].append((c['name'], c['us']['mean']))
    return dict(mean=statistics.fmean(means), median=statistics.median(medians), round_means=means,
                min_round=min(means), max_round=max(means), config=records[0]['config'],
                kernels=records[0]['kernels_per_call'],
                components=[dict(position=p, name=v[0][0], mean=statistics.fmean(x[1] for x in v))
                            for p, v in sorted(comps.items())])


def model_tables(manifest, stats, arm, ms):
    out = {}
    for mod in manifest['models']:
        for m in ms:
            total, missing = 0.0, []
            for f in mod['families']:
                cid = (f"e-q{f['qtype']}-n{f['n']}-k{f['k']}-e{f['experts']}-t{f['topk']}-c{f['channels']}-m{m}"
                       if f['kind'] == 'moe' else f"d-q{f['qtype']}-n{f['n']}-k{f['k']}-m{m}")
                s = stats.get(arm, {}).get(cid)
                if s is None:
                    missing.append(cid)
                    continue
                total += f['calls_per_token'] * s['mean']
            out[(mod['model'], m)] = dict(total_us=total, missing=missing)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--base', default='base')
    p.add_argument('--cand', default=None)
    p.add_argument('--m', default='1-8')
    p.add_argument('--csv', default=None)
    p.add_argument('--md', default=None)
    args = p.parse_args()
    lo, _, hi = args.m.partition('-')
    ms = list(range(int(lo), int(hi or lo) + 1))
    manifest = json.loads((ROOT / 'inventory/workload_manifest.json').read_text())
    data = load_runs(args.runs)
    stats = {arm: {cid: pooled(rs) for cid, rs in cases.items()} for arm, cases in data.items()}
    arms = [args.base] + ([args.cand] if args.cand else [])
    tables = {arm: model_tables(manifest, stats, arm, ms) for arm in arms}
    lines = []
    for m in ms:
        lines.append(f'\n### M={m}\n')
        hdr = '| model | ' + ' | '.join(f'{a} us' for a in arms) + (' | ratio | reduction | missing |' if args.cand else ' | missing |')
        lines.append(hdr)
        lines.append('|' + '---|' * (hdr.count('|') - 1))
        reds = []
        for mod in manifest['models']:
            b = tables[args.base][(mod['model'], m)]
            row = f"| {mod['model']} | {b['total_us']:.1f}"
            miss = len(b['missing'])
            if args.cand:
                c = tables[args.cand][(mod['model'], m)]
                miss += len(c['missing'])
                ratio = c['total_us'] / b['total_us'] if b['total_us'] else float('nan')
                reds.append(1 - ratio)
                row += f" | {c['total_us']:.1f} | {ratio:.4f} | {100 * (1 - ratio):.2f}%"
            lines.append(row + f' | {miss} |')
        if args.cand and reds:
            lines.append(f'\nM={m} equal-weight model mean reduction: {100 * statistics.fmean(reds):.2f}%')
    text = '\n'.join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + '\n')
    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['case', 'arm', 'config', 'kernels', 'mean_us', 'median_us', 'min_round_mean', 'max_round_mean',
                        'rounds', 'components'])
            for arm in arms:
                for cid, s in sorted(stats.get(arm, {}).items()):
                    w.writerow([cid, arm, s['config'], s['kernels'], f"{s['mean']:.3f}", f"{s['median']:.3f}",
                                f"{s['min_round']:.3f}", f"{s['max_round']:.3f}", len(s['round_means']),
                                ' + '.join(f"{c['mean']:.2f}" for c in s['components'])])


if __name__ == '__main__':
    main()

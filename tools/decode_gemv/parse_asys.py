#!/usr/bin/env python3
"""Attribute Asys GPU kernel activities to qgemv_bench cases.

Each case's timed region is bracketed by two memsets of unique byte counts on the
benchmark stream. All kernels between them belong to the case's graph replays.
They are grouped by their position inside one public call; every position must keep
one kernel identity. Complete-call cost = sum of the per-position durations of one call.
"""
import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
from pathlib import Path


def load(db_path):
    uri = Path(db_path).resolve(strict=True).as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        kt = 'HGPTI_ACTIVITY_KIND_KERNEL'
        mt = 'HGPTI_ACTIVITY_KIND_MEMSET'
        if kt not in tables:
            raise ValueError('no GPU kernel activity table; capture failed')
        strings = dict(db.execute('SELECT id,value FROM StringIds'))
        rows = []
        for r in db.execute(f'SELECT * FROM {kt}'):
            r = dict(r)
            rows.append(dict(kind='kernel', start=r['start'], end=r['end'], name=strings.get(r['demangledName']),
                             grid=(r['gridX'], r['gridY'], r['gridZ']), block=(r['blockX'], r['blockY'], r['blockZ']),
                             regs=r.get('registersPerThread'), smem=r.get('staticSharedMemory'),
                             stream=r['streamId'], device=r['deviceId']))
        if mt in tables:
            for r in db.execute(f'SELECT * FROM {mt}'):
                r = dict(r)
                rows.append(dict(kind='memset', start=r['start'], end=r['end'], bytes=r['bytes'], stream=r['streamId'],
                                 device=r['deviceId']))
    rows.sort(key=lambda x: (x['start'], x['end']))
    return rows


def stats(values):
    v = sorted(values)
    n = len(v)
    q = lambda f: v[min(n - 1, max(0, int(round(f * (n - 1)))))]
    return dict(n=n, mean=statistics.fmean(v), median=statistics.median(v), p10=q(.1), p90=q(.9), min=v[0],
                max=v[-1], std=statistics.pstdev(v) if n > 1 else 0.0)


def attribute(results, rows):
    import bisect
    out = []
    memsets = {}
    for r in rows:
        if r['kind'] == 'memset':
            memsets.setdefault(r['bytes'], []).append(r)
    by_stream = {}
    for r in rows:
        if r['kind'] == 'kernel':
            by_stream.setdefault(r['stream'], []).append(r)
    starts = {s: [k['start'] for k in ks] for s, ks in by_stream.items()}
    all_kernels = [r for r in rows if r['kind'] == 'kernel']
    all_starts = [k['start'] for k in all_kernels]
    for res in results:
        if res.get('status') != 'OK':
            out.append(dict(case=res['case'], status=res.get('status'), error=res.get('error')))
            continue
        begin, end = res['marker']
        mb, me = memsets.get(begin, []), memsets.get(end, [])
        if len(mb) != 1 or len(me) != 1:
            out.append(dict(case=res['case'], status='NO_MARKERS'))
            continue
        mb, me = mb[0], me[0]
        stream = mb['stream']
        ks = by_stream.get(stream, [])
        lo, hi = bisect.bisect_left(starts.get(stream, []), mb['end']), bisect.bisect_left(starts.get(stream, []), me['start'])
        kernels = [r for r in ks[lo:hi] if r['end'] <= me['start']]
        lo, hi = bisect.bisect_left(all_starts, mb['end'] - 10**7), bisect.bisect_left(all_starts, me['start'])
        foreign = [r for r in all_kernels[lo:hi] if r['stream'] != stream and r['end'] > mb['end']]
        samples = res['samples']
        if not kernels or len(kernels) % samples:
            out.append(dict(case=res['case'], status='KERNEL_COUNT', kernels=len(kernels), samples=samples))
            continue
        per = len(kernels) // samples
        positions = []
        ok = True
        for p in range(per):
            ks = kernels[p::per]
            ids = {(k['name'], k['grid'], k['block']) for k in ks}
            if len(ids) != 1:
                ok = False
                break
            durs = [(k['end'] - k['start']) / 1000.0 for k in ks]
            positions.append(dict(position=p, name=ks[0]['name'], grid=ks[0]['grid'], block=ks[0]['block'],
                                  regs=ks[0]['regs'], smem=ks[0]['smem'], us=stats(durs)))
        if not ok:
            out.append(dict(case=res['case'], status='POSITION_IDENTITY'))
            continue
        call_sums, spans = [], []
        for s in range(samples):
            ks = kernels[s * per:(s + 1) * per]
            call_sums.append(sum((k['end'] - k['start']) for k in ks) / 1000.0)
            spans.append((ks[-1]['end'] - ks[0]['start']) / 1000.0)
        rec = dict(case=res['case'], status='OK', config=res.get('config'), route=res.get('route'),
                   policy=res.get('policy'), split=res.get('split'), ring=res.get('ring'),
                   weight_bytes=res.get('weight_bytes'), samples=samples, kernels_per_call=per,
                   components=positions, call_sum_us=stats(call_sums), call_span_us=stats(spans),
                   component_mean_sum_us=sum(p['us']['mean'] for p in positions),
                   concurrent_foreign_kernels=len(foreign), check=res.get('check'), package=res.get('package'))
        out.append(rec)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--sqlite', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    results = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    rows = load(args.sqlite)
    parsed = attribute(results, rows)
    digest = hashlib.sha256(Path(args.sqlite).read_bytes()).hexdigest()
    with open(args.out, 'w') as fh:
        for p in parsed:
            p['sqlite_sha256'] = digest
            fh.write(json.dumps(p) + '\n')
    for p in parsed:
        if p['status'] != 'OK':
            print(p['case']['id'], p['status'], p.get('error', ''), file=sys.stderr)
            continue
        comps = ' + '.join(f"{c['us']['mean']:.2f}" for c in p['components'])
        print(f"{p['case']['id']:40s} {p['config'][:48]:48s} call mean {p['call_sum_us']['mean']:9.2f} "
              f"med {p['call_sum_us']['median']:9.2f} [{comps}] foreign={p['concurrent_foreign_kernels']}")


if __name__ == '__main__':
    main()

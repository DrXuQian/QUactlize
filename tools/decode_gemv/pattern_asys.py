#!/usr/bin/env python3
"""Asys attribution for tools/pattern_probe.cu: per-config median GPU kernel duration.

    python3 tools/pattern_asys.py --sqlite trace.sqlite --stdout probe.out [--json out.jsonl]
Kernels between the memset marker of config i (1 MiB + 64*i bytes) and the next marker
belong to config i; the first `ring` launches (the probe's warm pass) are dropped.
"""
import argparse
import json
import re
import sqlite3
import statistics
from pathlib import Path


def ring_for(nbytes):
    slot = (nbytes + (1 << 20) - 1) >> 20 << 20
    return max(4, ((160 << 20) + slot - 1) // slot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sqlite', required=True)
    ap.add_argument('--stdout', required=True)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    configs = {}
    for line in Path(a.stdout).read_text().splitlines():
        m = re.match(r'(\d+) (\w+) n=(\d+) rows=(\d+) T=(\d+) WN=(\d+) WK=(\d+) U=(\d+) bytes=(\d+) event_us=([\d.]+)', line)
        if m:
            i = int(m[1])
            configs[i] = dict(index=i, pattern=m[2], n=int(m[3]), rows=int(m[4]), T=int(m[5]), WN=int(m[6]),
                              WK=int(m[7]), U=int(m[8]), bytes=int(m[9]), event_us=float(m[10]))
    uri = Path(a.sqlite).resolve(strict=True).as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        kernels = sorted(db.execute('SELECT start,end FROM HGPTI_ACTIVITY_KIND_KERNEL'))
        marks = sorted(db.execute('SELECT start,bytes FROM HGPTI_ACTIVITY_KIND_MEMSET'))
    marks = [(s, (b - (1 << 20)) // 64) for s, b in marks if b >= (1 << 20) and (b - (1 << 20)) % 64 == 0]
    out = []
    for j, (start, i) in enumerate(marks):
        if i not in configs:
            continue
        stop = marks[j + 1][0] if j + 1 < len(marks) else float('inf')
        ks = [(e - s) / 1e3 for s, e in kernels if start <= s < stop]
        c = configs[i]
        ks = ks[ring_for(c['bytes']):]
        if not ks:
            continue
        med = statistics.median(ks)
        c.update(kernel_us=med, n_kernels=len(ks), gbps=c['bytes'] / med / 1e3)
        out.append(c)
        print(f"{i:4d} {c['pattern']:4s} n={c['n']:6d} rows={c['rows']:6d} T={c['T']} WN={c['WN']:5d} WK={c['WK']:2d} "
              f"U={c['U']} MB={c['bytes'] / 2**20:7.2f} kernel_us={med:8.2f} GB/s={c['gbps']:7.1f} "
              f"(event {c['event_us']:.2f})")
    if a.json:
        Path(a.json).write_text(''.join(json.dumps(c) + '\n' for c in out))


if __name__ == '__main__':
    main()

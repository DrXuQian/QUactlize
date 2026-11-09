#!/usr/bin/env python3
"""Summarize hgobjdump --dump-resource-usage for mma_reader kernels: VREG, stack, static shared.

    python3 tools/resources.py LIB.resources.log [--max-vreg 128]
Prints one line per kernel <Q,Mode,Compute,T,WN,WK,Tokens,NB> and flags any stack use (spill).
"""
import argparse
import re
from pathlib import Path

HEAD = re.compile(r'^Func \d+ (\S+) RESOURCE INFO:')
ARGS = re.compile(r"mma_reader(?:6kernel|13staged_kernel)I((?:Li-?\d+E)+)E")


def parse(text):
    out, cur = [], None
    for line in text.splitlines():
        m = HEAD.match(line)
        if m:
            cur = dict(name=m[1])
            out.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith('vreg_number:'):
            cur['vreg'] = int(line.split(':')[1])
        elif line.startswith('shared_memory_size:'):
            cur['smem'] = int(line.split(':')[1])
        elif line.startswith('STACK SIZE:'):
            cur['stack'] = int(line.split(':')[1], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--max-vreg', type=int, default=0, help='also list kernels above this VREG count')
    a = ap.parse_args()
    rows = []
    for f in parse(Path(a.log).read_text()):
        m = ARGS.search(f['name'])
        if not m:
            continue
        q, mode, comp, t, wn, wk, tok, nb = map(int, re.findall(r'Li(-?\d+)E', m[1]))
        rows.append((q, mode, comp, t, wn, wk, tok, nb, f.get('vreg'), f.get('stack', 0), f.get('smem', 0)))
    rows.sort()
    spills = [r for r in rows if r[9]]
    print(f'{len(rows)} kernels, {len(spills)} with stack')
    worst = {}
    for r in rows:
        key = r[:4] + (r[6],)
        worst[key] = max(worst.get(key, 0), r[8])
    print('max VREG by (Q,Mode,Compute,T,Tokens):')
    for k, v in sorted(worst.items()):
        print('  ', k, v)
    for r in spills:
        print('STACK', r)
    if a.max_vreg:
        for r in rows:
            if r[8] > a.max_vreg:
                print('VREG', r)


if __name__ == '__main__':
    main()

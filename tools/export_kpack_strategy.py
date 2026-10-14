#!/usr/bin/env python3
"""Normalize checked-in PPU evidence without rerunning a sweep or changing routing.

The latest effective decode board remains authoritative. Older observations
are retained, not compared by absolute latency across unrelated epochs.
Component sums and resident-F16 timings are never called complete F32 calls.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.dispatch.table import Observation, Recipe, StrategyTable, Workload
from quactlize.runtime.tuning import digest
from tools.generate_decode_selection import effective


def recipe(config, route=None, dequant=-1):
    c = dict(config)
    kind = c.pop('kind', 'tc')
    if kind != 'tc':
        return Recipe.make(kind, 'fq', **c)
    r = c.pop('route', 0) if route is None else route
    if isinstance(r, str):
        r = int(r.startswith('sf'))
    c['persistent'] = c.pop('parent_persistent', c.get('persistent', -1))
    if isinstance(c.get('grid_mode'), str):
        c['grid_mode'] = {'implicit': 0, 'ordinary': 0, 'capacity': 2, 'balanced': 3}[c['grid_mode']]
    # Execution identity, not reported occupancy/shared-memory or source labels.
    fields = ('symbol', 'qtype', 'tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn',
              'persistent', 'split', 'grid_mode', 'grid_b')
    return Recipe.make('tc', ('fq', 'sf')[r % 2],
                       **{k: c[k] for k in fields if k in c}, dequant_config=dequant)


def small_workload(key, *, compute=None):
    q, mode, n, k, experts, topk, channels, tokens, dtype = key
    return Workload(q, n, k, tokens, 'indexed' if mode else 'dense',
                    compute or ('bf16' if dtype else 'f16'), experts=experts,
                    topk=topk, channels=channels)


def load_sources(root=ROOT):
    root = Path(root)
    observations, sources = [], {}

    def read(name):
        data = json.loads((root / name).read_text())
        sources[name] = digest(data)
        return data

    def add(w, r, source, scope, priority, status, evidence, us=None, **details):
        observations.append(Observation(w, r, source, scope, priority, status, evidence, us, details))

    # Use the SAME effective-board generator as the C++ production selector.
    # Do not accidentally undo Q8/TP/fallback promotions by importing only the
    # older matched board. effective() validates its reviewed evidence digest.
    winner = read('policies/kpack_decode_winners_v1.json')
    read('policies/kpack_smallm_matched_v1.json')
    read('policies/kpack_q8_vector_v1.json')
    if root.resolve() != ROOT.resolve():
        raise ValueError('effective board must be imported with its own source checkout')
    board = effective(winner)
    board_id = digest({name: value for name, value in sources.items()})
    for row in board['exact']:
        add(small_workload(row['key']), recipe(row['config']), 'effective-decode',
            'COMPLETE_CALL', 50, row['status'], board_id,
            policy=row['policy'], router_sensitive=row.get('status') == 'ROUTER_SENSITIVE_PARETO',
            profiles=row.get('profiles', []), guard=row['guard'])

    old = read('policies/kpack_smallm_v1.json')
    for row in old['exact']:
        q, mode, n, k, tokens, channels = row['key']
        decision = row['decision']
        # TC_RETAINED retained an older tactic; it did not establish a matched
        # F32 endpoint measurement. Do not upgrade it to complete-call timing.
        scope = 'PROPOSAL'
        key = [q, mode, n, k, 256 if mode else 1, 8 if mode else 1, channels, tokens, 0]
        add(small_workload(key), recipe(row['config']), 'legacy-decode', scope, 20,
            decision, sources['policies/kpack_smallm_v1.json'],
            timing_reuse=False, original_decision=decision)

    q4 = read('policies/kpack_q4_decode_v1.json')
    for row in q4['ranges']:
        spelling = row['recipe']
        if spelling.startswith('simt:'):
            match = re.fullmatch(r'simt:r(\d+)-v(\d+)-w(\d+)-p(\d+)-c(\d+)', spelling)
            if not match:
                raise ValueError('unknown Q4 recipe: ' + spelling)
            r = Recipe.make('q4', 'fq', **dict(zip(('reader', 'variant', 'warps', 'values', 'columns'),
                                                 map(int, match.groups()))))
        else:
            match = re.fullmatch(r'tc:(.+):s(\d+):b(\d+):g(\d+)', spelling)
            if not match:
                raise ValueError('unknown TC recipe: ' + spelling)
            parent, split, blocks, grid = match.groups()
            parts = dict((name, int(value)) for name, value in re.findall(
                r'(?:^|_)(tm|tn|tk|wm|wn|s|ap|dn)(\d+)', parent))
            parts['stages'] = parts.pop('s')
            parts.update(symbol=parent, qtype=12, split=int(split), grid_b=int(blocks),
                         grid_mode=int(grid), persistent=0 if 'nonpersistent' in parent else
                         1 if 'persistent' in parent else -1)
            r = Recipe.make('tc', 'fq', **parts)
        grouped = row['operator'] != 'dense'
        for m in range(row['first'], row['last'] + 1):
            w = Workload(12, row['n'], row['k'], m, 'indexed' if grouped else 'dense',
                         experts=256 if grouped else 1, topk=8 if grouped else 1)
            add(w, r, 'q4-decode', 'COMPLETE_CALL', 30 if row['role'] == 'auto' else 10,
                'MEASURED_RANGE' if row['first'] != row['last'] else 'MEASURED_EXACT',
                sources['policies/kpack_q4_decode_v1.json'], role=row['role'])

    configurations = {}
    for name, source, priority in (
            ('policies/kpack_zw810_runtime_v1.json', 'historical-resident', 5),
            ('policies/kpack_zw810_heuristic_v1.json', 'calibrated-resident', 15)):
        data = read(name)
        configurations.update(data['configurations'])
        for row in data['entries']:
            q, route, n, k, experts, m, *_ = row['key']
            rows = (row.get('context', {}).get('rows') or
                    (data.get('row_vectors', [])[row['row_vector']] if row.get('row_vector', -1) >= 0 else ()))
            w = Workload(q, n, k, m, 'grouped' if route >= 2 else 'dense', storage='f16',
                         experts=experts, router=digest(rows) if route >= 2 else '')
            add(w, recipe(configurations[row['config_id']]), source, 'RESIDENT_GEMM', priority,
                row.get('status', 'MEASURED_CALIBRATION'), sources[name], row.get('selected_us'),
                excludes_sf_prepass=True, excludes_f32_adapters=True)

    name = 'docs/measurements/kpack_component_policy_20260914.json'
    cost = read(name)
    for row in cost['knots']:
        # This is the original F16 component board. No BF16 timing is invented.
        grouped = row['experts'] != 1
        w = Workload(row['q'], row['n'], row['k'], row['tokens'],
                     'indexed' if grouped else 'dense', experts=row['experts'], topk=8 if grouped else 1)
        for mask, c in row['choices'].items():
            r = (Recipe.make('full', 'full', dequant_config=c['dequant_config'], provider='bf16-default')
                 if c['route'] == 2 else recipe(c['tactic'], c['route'], c['dequant_config']))
            add(w, r, 'prefill-components', 'COMPONENT_SUM', 40 if mask == '7' else 10,
                'ISOLATED_SUM_NOT_COMPLETE_CALL', sources[name], c['gemm_us'] + c['dequant_us'],
                mask=int(mask), gemm_us=c['gemm_us'], dequant_us=c['dequant_us'],
                external_adapters_included=False, profiles=c['profiles'])
    return StrategyTable(observations), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new JSON file; no existing evidence is overwritten')
    args = parser.parse_args()
    table, sources = load_sources()
    result = table.as_dict() | {'sources': sources, 'production_selection_changed': False}
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, sort_keys=True, separators=(',', ':'))
            stream.write('\n')
    print('KPACK_STRATEGY_IMPORT ' + json.dumps(dict(
        observations=len(table.observations), exact_keys=len(table.exact),
        sources=dict(Counter(r.source for r in table.observations)),
        scopes=dict(Counter(r.scope for r in table.observations)),
        table_sha256=table.identity, device_work=0, production_selection_changed=False)))


if __name__ == '__main__':
    main()

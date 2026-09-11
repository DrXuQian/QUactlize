#!/usr/bin/env python3
"""Validate an extracted PPU replay and import ACU counters without a device."""
import argparse
import collections
import csv
import gzip
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.run_h800_port import POLICY, ROUNDS, VARIANTS, parse_result, recipes, summarize


def receipt_entry(folder, entry, arm, recipe, shape, mode, samples, device):
    log = folder / entry['log']
    if log.parent != folder or digest(log) != entry['log_sha256']:
        raise ValueError('child log path/hash differs')
    text = 'Q4_PPU_CELL ' + json.dumps(entry['row'])
    row = parse_result(text, arm, recipe, shape, mode, samples)
    if row['device'] != device or text not in log.read_text():
        raise ValueError('child row/device absent from log')
    return row


def review(folder):
    report = json.loads((folder / 'summary.json').read_text())
    authority = json.loads((folder / 'authority.json').read_text())
    inputs = {'runner': 'dev/gemv_ppu/run_h800_port.py', 'protocol': 'dev/gemv_ppu/run.py',
              'parser': 'dev/gemv_ppu/campaign.py',
              'candidate': 'prebuilt/ppu0010/q4-h800-port-v1/manifest.json',
              'baseline': 'prebuilt/ppu0010/q4-simt-ab-v1/manifest.json'}
    if authority != report['authority'] or any(digest(ROOT / p) != authority[k] for k, p in inputs.items()):
        raise ValueError('source/package authority differs')
    expected = {(tuple(shape), mode) for shape in POLICY for mode in ('warm', 'rotating')}
    seen = [(tuple(c['shape']), c['mode']) for c in report['cases']]
    if (len(seen) != len(expected) or set(seen) != expected or report['failures']
            or authority['rounds'] != 6 or authority['samples'] != 15):
        raise ValueError('incomplete comparison matrix')
    counts = collections.Counter()
    samples = []
    spreads = []
    for case in report['cases']:
        _, n, k = case['shape']
        mode = case['mode']
        records = {}
        for arm in VARIANTS:
            wanted = recipes(arm, n, k)
            rounds = []
            for label in ['screen'] + [f'confirm{i}' for i in range(ROUNDS)]:
                path = folder / f'n{n}-k{k}-{mode}-{arm}-{label}.json'
                cached = json.loads(path.read_text())
                if set(cached) != {json.dumps(list(r)) for r in wanted}:
                    raise ValueError('recipe set differs: ' + path.name)
                rows = [receipt_entry(folder, cached[json.dumps(list(r))], arm, list(r), case['shape'],
                                      mode, 5 if label == 'screen' else 15, authority['device']) for r in wanted]
                counts[arm + ('/screen' if label == 'screen' else '/confirm')] += len(rows)
                samples.extend(dict(row, phase=label) for row in rows)
                if label == 'screen':
                    wanted = [r['recipe'] for r in sorted(rows, key=lambda r: r['median_us'])[:2]]
                else:
                    rounds.append(rows)
            records[arm] = rounds
        if summarize(n, k, mode, records) != case:
            raise ValueError('summary differs from raw receipts')
        for arm, best in case['winners'].items():
            values = [r['median_us'] for r in best['rounds']]
            spreads.append(dict(shape=case['shape'], mode=mode, arm=arm,
                                span_pct=100 * (max(values) / min(values) - 1)))
    expected_profiles = {(n, k, arm) for n, k in ((512, 2048), (1024, 5120), (5120, 8192)) for arm in VARIANTS}
    actual_profiles = [(*p['shape'][1:], p['variant']) for p in report['profiles']]
    if len(actual_profiles) != 9 or set(actual_profiles) != expected_profiles:
        raise ValueError('profile matrix differs')
    for profile in report['profiles']:
        if profile['status'] != 'PASS' or profile['timing_authority'] or profile['cache'] != 'FORCED_COLD':
            raise ValueError('missing/mislabeled profile')
        log = folder / profile['log']
        if log.parent != folder:
            raise ValueError('profile log escapes result directory')
        parse_result(log.read_text(), profile['variant'], profile['recipe'], profile['shape'], 'warm', 0)
        for entry in profile['files']:
            path = folder / entry['file']
            if path.parent != folder or digest(path) != entry['sha256']:
                raise ValueError('profile hash/path differs')
    return report, samples, dict(counts), spreads


def import_profiles(folder, profiles, command):
    out = []
    for profile in profiles:
        files = [x for x in profile['files'] if x['file'].endswith('.acurep')]
        if len(files) != 1:
            raise ValueError('expected one ACU report per kernel')
        result = subprocess.run(command + ['--import', str(folder / files[0]['file']), '--page', 'raw', '--csv'],
                                check=True, capture_output=True, text=True)
        metrics = list(csv.DictReader(io.StringIO(result.stdout)))
        if len(metrics) != 1:
            raise ValueError('expected one profiled kernel')
        log = (folder / profile['log']).read_text()
        # A context-owner warning does not prove simultaneous GPU work. Keep
        # exact owners; never label it clean or contaminated from names alone.
        owners = sorted(set(re.findall(r'^\s*(\d+):([^\n]+)$', log, re.M)))
        out.append(dict(profile, metrics=metrics[0], other_context_owners=owners))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--acu', type=Path, required=True)
    p.add_argument('--loader', type=Path)
    p.add_argument('--library-path')
    a = p.parse_args()
    if bool(a.loader) != bool(a.library_path):
        p.error('an alternate loader requires its private library path')
    command = ([str(a.loader), '--library-path', a.library_path] if a.loader else []) + [str(a.acu)]
    report, samples, counts, spreads = review(a.results.resolve())
    profiles = import_profiles(a.results, report['profiles'], command)
    summary = dict(report, archive_sha256=digest(a.archive), records=len(samples), record_counts=counts,
                   max_conditioned_error=max(r['error'] for r in samples),
                   max_round_median_span_pct=max(r['span_pct'] for r in spreads),
                   within_5pct={arm: sum(c['delta_pct'][arm] < 5 for c in report['cases']) for arm in VARIANTS[:2]},
                   both_within_5pct=sum(c['verdict'] == 'WITHIN_5_PERCENT' for c in report['cases']),
                   interference='NO_EXCLUSIVITY_PROOF_FROM_CONTEXT_WARNING', profiles=profiles)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (a.output / 'samples.json.gz').write_bytes(gzip.compress(json.dumps(samples).encode(), mtime=0))
    print(f'Q4_PPU_REVIEW VERIFIED records={len(samples)} profiles={len(profiles)} both_within_5pct={summary["both_within_5pct"]}/12')


if __name__ == '__main__':
    main()

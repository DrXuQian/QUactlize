#!/usr/bin/env python3
"""Export recipes, not timings, from the reviewed large-M finalist archive."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.journal import atomic_json
from dev.strategy.prefill_closure import require
from dev.strategy.prefill_plan import family
from dev.strategy.resource_pilot import verified_plan
from quactlize.dispatch.table import Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ARCHIVE_SHA = 'b43e1d28bada3d28d15aed2ebd5e92d693c63cb4429924e24ab79fbf5d5ce8fb'


def export(archive, policy_path):
    require(sha(archive) == ARCHIVE_SHA, 'unreviewed prefill finalist archive')
    policy = json.loads(policy_path.read_text())
    policies = {p['point']: p for p in policy['points']}
    with tarfile.open(archive) as tar:
        plan = verified_plan(json.load(tar.extractfile('finalists.json')))
        require(policy['plan_sha256'] == plan['plan_sha256'], 'policy/finalists provenance differs')
        # The old family's PROCESS duration is only a scheduling hint. Never
        # import old component times into a fresh round or selection verdict.
        hints = {}
        for shard in range(7):
            entries = sorted((int(m[2]), m[1]) for n in tar.getnames()
                if (m := re.fullmatch(fr'shard-{shard}/family-([0-9a-f]{{64}})-(\d+)\.log', n)))
            for (start, key), (end, _) in zip(entries, entries[1:]):
                hints[key] = max(1., (end-start)/1e9)
            text = tar.extractfile(f'shard-{shard}/console.log').read().decode(errors='replace')
            if entries:
                key = entries[-1][1]
                waits = re.findall(fr'phase=finalists-{key[:12]} minutes=([\d.]+)', text)
                require(waits, 'last family lacks scheduling evidence')
                hints[key] = float(waits[-1])*60+30
        recipes, points = {}, []
        for point in plan['points']:
            selected = policies[point['id']]
            require(selected['workload'] == point['workload'], 'policy workload changed')
            require(set(selected['statistics']) == set(point['arms']), 'finalist coverage changed')
            arms = {}
            for aid, arm in point['arms'].items():
                r = Recipe.from_dict(arm['recipe'])
                recipes[r.key] = asdict(r)
                arms[aid] = dict(recipe=r.key, historical_roles=arm['roles'],
                    qualified=aid in selected['decision']['qualified'])
            incumbent = selected['decision']['winner']
            require(incumbent in arms and
                    Recipe.from_dict(point['arms'][incumbent]['recipe']) == Recipe.from_dict(selected['recipe']),
                    'current incumbent not in historical finalists')
            points.append(dict(id=point['id'], workload=point['workload'], arms=arms,
                incumbent=incumbent, historical_original=point['original']))
    require(len(points) == 430 and sum(len(p['arms']) for p in points) == 1138,
            'unexpected finalist denominator')
    require({family(p) for p in points} == set(hints), 'family scheduling evidence incomplete')
    value = dict(schema='quactlize.prefill-refresh-seed.v1', points=points, recipes=recipes,
        archive_sha256=ARCHIVE_SHA, historical_plan=plan['plan_sha256'], policy_sha256=sha(policy_path),
        scheduling_seconds=hints, scheduling_scope='HISTORICAL_PROCESS_DURATION_ONLY_NOT_TIMING_REUSE',
        scope='HISTORICAL_FINALISTS_NOT_NEW_BLIND_SEARCH', production_changed=False)
    value['sha256'] = digest(value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, export(args.archive, ROOT/'policies/kpack_prefill_confirmation_v1.json'))

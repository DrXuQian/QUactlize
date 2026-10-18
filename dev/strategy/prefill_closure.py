"""Freeze measured finalists after blind-rule replay; never install a screen.

The reviewed archive is immutable. A control may qualify under a new general
rule, but its original role and raw record stay unchanged in the evidence.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import tarfile

from dev.strategy.prefill_selection import control_helper_config
from dev.strategy.prescreen import Prescreen, semantic_config
from dev.strategy.space import factored_recall
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

REVIEW_SHA256 = '7b9a96cd4d01393de96cae0eef71f988c7be3775a03d9f20343bd6048a89aebe'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_compact(stream):
    tables, groups = defaultdict(dict), defaultdict(list)
    counts, checksum, ended = Counter(), hashlib.sha256(), False
    with lzma.open(stream, 'rb') as src:
        for raw in src:
            require(not ended, 'records after compact footer')
            def reject(value):
                raise ValueError('nonfinite JSON: '+value)
            r = json.loads(raw, parse_constant=reject)
            kind = r['type']
            if kind == 'end':
                require(r['prior_lines_sha256'] == checksum.hexdigest() and
                        r['counts'] == dict(counts) and r['export_status'] == 'COMPLETE',
                        'compact checksum/count/coverage differs')
                ended = True
            else:
                checksum.update(raw); counts[kind] += 1
            if kind == 'definition':
                require(r['index'] not in tables[r['table']], 'duplicate definition')
                tables[r['table']][r['index']] = r['value']
            else:
                groups[kind].append(r)
    require(ended and not groups.get('export_issue'), 'incomplete compact review')
    return tables, groups


def recipe_with_helper(recipe, config):
    p = dict(recipe.parameters)
    p['dequant_config'] = config
    return Recipe.make(recipe.kind, recipe.route, **p)


def point_finalists(w, arms, components, best_helpers, generated, *, limit=4):
    """Top two distinct rule calls, best control and original, at most four.

    The complete screen already compared all generated candidates. This is
    confirmation, not a new top-four search. Duplicate measured stages are
    executed once; their roles remain aliases, not invented extra samples.
    """
    require('original' in arms, 'original wiring control missing')
    choices = {}
    for aid, arm in arms.items():
        r = arm['recipe']
        qualify = factored_recall(w, generated, [r])['status'] == 'PASS'
        role = 'full_dequant' if r.kind == 'full' else 'sf' if r.route == 'sf' and w.qtype != 8 else None
        old_config = control_helper_config(w, r) if role else -1
        configs = {old_config}
        if qualify and role:
            configs.add(best_helpers[role])
        for config in sorted(configs):
            ids = list(arm['core_components'])
            if role:
                require(config in arm['helpers'][role], 'missing measured helper')
                ids += arm['helpers'][role][config]
            require(ids and all(c in components for c in ids), 'incomplete measured composition')
            key = digest(sorted(ids))
            value = choices.setdefault(key, dict(components=ids, roles=[], aliases=[],
                recipe=asdict(recipe_with_helper(r, config)), source_arm=aid,
                cost_us=math.fsum(components[c]['median_us'] for c in ids),
                noisy=any(components[c]['spread_pct'] > 5 for c in ids),
                original=False))
            if qualify and (not role or config == best_helpers[role]):
                value['roles'].append('rule')
                value['rule_proof'] = 'R27_LEAVE_NK_OUT_GENERATED_NOT_CHAMPION_INJECTION'
            if config == old_config and ('control' in arm['roles'] or 'original' in arm['roles']):
                value['roles'].append('control')
            if aid == 'original' and config == old_config:
                value['original'] = True
                value['roles'].append('original')
                # Keep the explicit original source binding for the shared call.
                value['source_arm'] = aid
                value['recipe'] = asdict(recipe_with_helper(r, config))
            value['aliases'].append(dict(arm=aid, helper=config, raw_roles=arm['roles']))
    rules = sorted((k for k,v in choices.items() if 'rule' in v['roles']),
                   key=lambda k:(choices[k]['cost_us'], k))
    controls = sorted((k for k,v in choices.items() if 'control' in v['roles']),
                      key=lambda k:(choices[k]['cost_us'], k))
    original = [k for k,v in choices.items() if v['original']]
    require(rules and controls and len(original) == 1, 'missing rule/control denominator')
    selected = set(rules[:2]) | {controls[0], original[0]}
    require(len(selected) <= limit, 'confirmation shortlist exceeds bound')
    for value in choices.values():
        value['roles'] = sorted(set(value['roles']))
    shortlist = {k:choices[k] for k in sorted(selected)}
    return dict(workload=asdict(w), id=w.key, arms=shortlist, original=original[0],
        best_rule=rules[0], best_control=controls[0],
        screen_compositions=len(choices), screen_rule_compositions=len(rules),
        rule_gap_pct=100*(choices[rules[0]]['cost_us']/choices[controls[0]]['cost_us']-1),
        noisy=any(v['noisy'] for v in shortlist.values()),
        omitted_rule_finalists=len(set(rules)-selected),
        status='CONFIRM_REQUIRED', performance_admitted=False)


def export(archive, table):
    require(sha(archive) == REVIEW_SHA256, 'archive is not the completed reviewed supplement')
    selector = Prescreen(table)
    result, shards = [], []
    with tarfile.open(archive, 'r|gz') as tar:
        seen = set()
        for member in tar:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and '..' not in path.parts and member.isfile() and
                    member.name not in seen, 'unsafe/duplicate review member')
            seen.add(member.name)
            if not member.name.endswith('.xz'):
                continue
            tables, groups = read_compact(tar.extractfile(member))
            run = groups['run'][0]; shard = run['summary']['counts']['shard']
            require(len(groups['run']) == 1, 'duplicate run')
            aa = {(r['point'],r['id']):r for r in groups['arm']}
            require(len(aa) == len(groups['arm']), 'duplicate arm')
            for binding in groups.get('reuse_binding', []):
                aa[binding['point'],binding['id']] = dict(aa[binding['point'],binding['source_job']],
                                                         recipe=binding['recipe'])
            cc = {r['id']:r for r in groups['component']}
            pp = {r['id']:r for r in groups['point']}
            for screen in groups['screen']:
                pid = screen['point']; w = Workload(**pp[pid]['workload'])
                require(w.key == pid and screen['status'] == 'PASS' and not screen['missing'],
                        'incomplete/wrong workload')
                generated, report = selector.select(w, blind=True, per_stratum=2, stencil='warps')
                helpers = {'sf':{}, 'full_dequant':{}}
                provider = []
                for (p, aid), row in aa.items():
                    if p != pid: continue
                    require(row['status'] == 'PASS', 'failed measured job')
                    if row['roles'] in (['sf'], ['full_dequant']):
                        helpers[row['roles'][0]][row['expansion_config']] = row['components']
                    if row['roles'] == ['bf16']:
                        provider = row['components']
                # screen.components contains full arms, including FULL calls
                # without their own journal job. Their recipes are definitions.
                arms = {}
                for aid, ids in screen['components'].items():
                    if (pid, aid) in aa:
                        raw = aa[pid,aid]
                        recipe = Recipe.from_dict(tables['config'][raw['recipe']])
                        core = raw['components']
                    else:
                        # FULL has one provider job and separately timed helpers.
                        if aid == 'original':
                            configs = [screen['control_helper_bindings'][aid]['requested']]
                        else:
                            configs = [c for c in range(-1, 13) if semantic_config(Recipe.make(
                                'full', 'full', dequant_config=c, provider='bf16-default')) == aid]
                        require(len(configs) == 1, 'unbound composed arm')
                        recipe = Recipe.make('full', 'full', dequant_config=configs[0], provider='bf16-default')
                        core = provider
                    roles = screen['selection']['roles'][aid]
                    arms[aid] = dict(recipe=recipe, roles=roles, core_components=core, helpers=helpers)
                row = point_finalists(w, arms, cc, screen['best_helpers'], generated)
                row.update(shard=shard, rule_revision=report['rule_revision'],
                           delivery_peers=report['prefill_delivery_peers'],
                           source_screen_sha256=digest(screen))
                result.append(row)
            shards.append(dict(shard=shard, resource=run['authority']['resource'],
                               authority=run['authority_sha256']))
            print(f'STRATEGY_CLOSURE_REPLAY shard={shard} points={len(result)}/430', flush=True)
    require(len(result) == len({p['id'] for p in result}) == 430 and len(shards) == 7,
            'closure workload/shard denominator differs')
    plan = dict(schema='quactlize.prefill-finalists.v1', archive_sha256=REVIEW_SHA256,
        catalog=table.identity, rule_revision=27, points=sorted(result, key=lambda p:p['id']),
        shards=sorted(shards,key=lambda s:s['shard']),
        counts=dict(points=len(result), arms=sum(len(p['arms']) for p in result),
                    rule_gaps=sum(p['rule_gap_pct']>1e-10 for p in result),
                    noisy=sum(p['noisy'] for p in result)),
        protocol=dict(rounds=6, order='FORWARD_REVERSE_ALTERNATING',
                      timing='SEPARATE_COMPONENTS_CUBLAS_ORDINARY_EVENTS',
                      new_measurements=True, max_calls_per_point=4),
        production_changed=False, performance_admitted=False,
        scope='SHORTLIST_NOT_ADMISSION_NO_OLD_SELECTOR_FALLBACK')
    plan['plan_sha256'] = digest(plan)
    return plan

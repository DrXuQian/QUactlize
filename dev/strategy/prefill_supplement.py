"""Append-only rule supplementation of the reviewed b0e813c large-M screen.

Only the proposal/composition layer changes. Imported receipts keep their
original authorities, samples and device images. Alias matching requires the
SAME image and actual launch, not just a similar parent name or geometry.
"""
import ast
from dataclasses import asdict
import math
from pathlib import Path
import statistics

from dev.strategy.journal import ArmJournal, atomic_json
from dev.strategy.prefill_retry import PrefillJournal, checked_component, copy_exact, read
from dev.strategy.resource_pilot import verified_plan
from dev.strategy.resources import module_spec
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest


BASE_HARNESS = 'a315ce458e00a8e25ed489f917163139061dbf78fd24f97a3c82a4385b173fbd'
BASE_RUNNER = 'b108f8a6c871711b4493ff9062e8530a5ab0495add898fe689b30a87ed379a20'
# Exact child/jobs source from BASE_RUNNER, not a version-dependent ast.dump.
MEASUREMENT_SOURCE = 'b8a70944ead8a134bd60718e17968ba58c52c1a61322b5fb460905afe9bf15ae'
RUNNER = 'tools/run_kpack_strategy_prefill.py'
HOST_ONLY = {
    'dev/strategy/prescreen.py', 'dev/strategy/prefill_candidates.py',
    'dev/strategy/prefill_plan.py', 'dev/strategy/prefill_selection.py',
    'dev/strategy/prefill_supplement.py', 'tools/pack_kpack_strategy_review.py',
    'tools/run_kpack_strategy_prefill_multigpu.py',
}


def previous_root(path):
    root = Path(path).resolve(strict=True)
    if (root/'measurement').is_dir(): root = root/'measurement'
    for name in ('authority.json', 'execution-plan.json', 'plan/plan.json'):
        read(root, name)
    if (root/'supplement/index.json').exists():
        raise ValueError('resume this supplement, not a chain of rule migrations')
    return root


def measurement_signature(source):
    """Seal reviewed function text, including decorators, across Python versions.

    ast.dump adds type_params in Python3.12 even for unchanged functions. Use
    the parser only to locate source spans; do not normalize executable text.
    """
    names = {'child', 'jobs'}
    lines = source.splitlines(keepends=True)
    functions = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in names:
            continue
        if node.name in functions:
            raise ValueError('duplicate measured function: '+node.name)
        start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
        functions[node.name] = ''.join(lines[start:node.end_lineno])
    if functions.keys() != names:
        raise ValueError('missing measured functions: '+', '.join(sorted(names-functions.keys())))
    return digest(functions)


def check_harness(old, current, root):
    for value in (old, current):
        if value.get('sha256') != digest(value['files']):
            raise ValueError('supplement harness digest differs')
    if old['sha256'] != BASE_HARNESS or old['files'].get(RUNNER) != BASE_RUNNER:
        raise ValueError('supplement requires the reviewed b0e813c ordinary-event screen')
    changed = {p for p in old['files'].keys() | current['files'].keys()
               if old['files'].get(p) != current['files'].get(p)}
    if changed - HOST_ONLY - {RUNNER}:
        raise ValueError('supplement changed measurement inputs: '+', '.join(sorted(changed-HOST_ONLY-{RUNNER})))
    signature = measurement_signature((Path(root)/RUNNER).read_text(encoding='utf-8'))
    if signature != MEASUREMENT_SOURCE:
        raise ValueError('supplement changed the measured child or job source: '
                         f'got={signature} expected={MEASUREMENT_SOURCE}')


def geometry(spec):
    return digest(dict(spec, parent={k:v for k,v in spec['parent'].items() if k != 'symbol'}))


def reuse_event_helper(previous, output, sdk):
    from dev.strategy.prefill_build import verify
    before = verify(sdk, previous/'cublas-helpers')
    for name in ('manifest.json', 'libstrategy_prefill.so'):
        path = previous/'cublas-helpers'/name
        if path.is_symlink(): raise ValueError('event helper must not be a symlink')
        copy_exact(output/'cublas-helpers'/name, path.read_bytes())
    if verify(sdk, output/'cublas-helpers') != before:
        raise ValueError('copied event helper changed')


def canonicalize(points, old_plan):
    """Reuse a symbol only AFTER blind generation chose its exact geometry."""
    verified_plan(old_plan)
    specs = {}
    for point in old_plan['points']:
        for arm in point['arms']:
            if arm['recipe']['kind'] == 'tc':
                spec = arm['spec']
                specs.setdefault(geometry(spec), set()).add(spec['parent']['symbol'])
    old = {p['id']:p for p in old_plan['points']}
    if set(old) != {p['id'] for p in points}:
        raise ValueError('supplement must revisit every point in the original shard')
    aliases = []
    for point in points:
        if point['workload'] != old[point['id']]['workload']:
            raise ValueError('supplement changed a workload contract')
        w = Workload(**point['workload'])
        pool = {}
        for value in point['candidates']:
            r = Recipe.from_dict(value)
            if r.kind == 'tc':
                names = specs.get(geometry(module_spec(w, r)), ())
                p = dict(r.parameters)
                if names and p['symbol'] not in names:
                    before = p['symbol']; p['symbol'] = min(names)
                    aliases.append(dict(point=point['id'], before=before, after=p['symbol']))
                    r = Recipe.make(r.kind, r.route, **p)
            pool[r.key] = asdict(r)
        point['candidates'] = list(pool.values())
        from dev.strategy.space import grid_search
        point['grid_search'] = {r.key:g for value in point['candidates']
                                for r in [Recipe.from_dict(value)] if (g:=grid_search(w,r))}
    return aliases


def compatible(old, new, root):
    check_harness(old['harness'], new['harness'], root)
    exceptions = {'harness', 'plan', 'resource', 'tc'}
    if {k:v for k,v in old.items() if k not in exceptions} != {k:v for k,v in new.items() if k not in exceptions}:
        raise ValueError('supplement changed helpers/provider/oracle/cache/timing authority')
    for key in ('device', 'sdk', 'cache'):
        if old['resource'][key] != new['resource'][key]:
            raise ValueError('supplement requires the same physical device/SDK/cache: '+key)
    if any(new['tc'].get(mid) != image for mid, image in old['tc'].items()):
        raise ValueError('supplement changed or omitted an existing compiled TC image')


def job_key(point, job, authority):
    if job['recipe'] is None:
        return digest([point['workload'], job['roles'], job['config']])
    if job['recipe']['kind'] != 'tc':
        raise ValueError('supplement jobs must be TC or independent helpers')
    # Workload includes mapping-relevant qtype, precision, storage, router,
    # strides and channels. Identical images + launch bind all TC config axes.
    return digest([point['workload'], authority['tc'][job['module']],
                   job['spec']['endpoint'], job['spec']['compute'], job['binding']['launch']])


def import_passes(previous, output, plan, authority, jobs, source_root):
    old, _ = read(previous, 'authority.json')
    compatible(old, authority, source_root)
    old_plan, _ = read(previous, 'execution-plan.json'); verified_plan(old_plan)
    originals = {p['id']:p for p in old_plan['points']}
    if set(originals) != {p['id'] for p in plan['points']} or plan['blocked']:
        raise ValueError('supplement has changed or resource-blocked point coverage')
    path = output/'supplement/index.json'
    if path.exists():
        index, _ = read(output, 'supplement/index.json')
        if index['previous_output'] != str(previous) or index['authority_sha256'] != digest(authority):
            raise ValueError('supplement resume authority changed')
        for cid, seal in index['components'].items():
            row, _ = read(output, Path('components')/(cid+'.json'))
            checked_component(output, cid, row['identity']['authority'])
            if row['record_sha256'] != seal: raise ValueError('imported component changed')
        return index
    prior = PrefillJournal(previous/'arms', dict(sha256=digest(old)), '0'*32)
    components, receipts = {}, {}
    for point in plan['points']:
        op = originals[point['id']]
        if op['workload'] != point['workload']: raise ValueError('supplement workload differs')
        pool = {}
        for job in jobs(op):
            row = prior.read(op, job)
            if not row or row['status'] != 'PASS': continue
            pool.setdefault(job_key(op, job, old), []).append((job, row))
        for job in jobs(point):
            matches = pool.get(job_key(point, job, authority), [])
            if not matches: continue
            # Deterministic identity preference, never choose the fastest old
            # sample among aliases. Preserve independent confirmations later.
            oj, row = min(matches, key=lambda pair:(pair[0]['id'] != job['id'], pair[0]['id']))
            for cid in row['data']['components']:
                if cid in components: continue
                record, raw = checked_component(previous, cid, row['identity']['authority'])
                copy_exact(output/'components'/(cid+'.json'), raw)
                components[cid] = record['record_sha256']
            relative = Path('supplement/arms')/point['id']/(job['id']+'.json')
            source = (Path('prior/arms') if prior.index and point['id']+'/'+oj['id'] in prior.index['receipts']
                      else Path('arms'))/point['id']/oj['id']/'current.json'
            original, raw = read(previous, source)
            if original != row: raise ValueError('previous receipt changed during import')
            copy_exact(output/relative, raw)
            receipts[point['id']+'/'+job['id']] = dict(path=str(relative), seal=row['record_sha256'],
                expected=ArmJournal(output/'arms', {}, '0'*32).identity(point, job)['arm'],
                source_job=oj['id'], source_authority=row['identity']['authority'])
    index = dict(schema='quactlize.prefill-rule-supplement.v1', previous_output=str(previous),
        authority_sha256=digest(authority), receipts=receipts, components=components,
        passed_jobs=len(receipts), production_changed=False,
        scope='SAME_IMAGE_LAUNCH_FIXTURE_DEVICE_PRIOR_PASS_NOT_NEW_MEASUREMENTS')
    authorities = {digest(old):old}
    if prior.index:
        prior_authority, _ = read(previous, 'prior/authority.json')
        authorities[digest(prior_authority)] = prior_authority
    index['source_authorities'] = authorities
    index['record_sha256'] = digest(index)
    atomic_json(path, index)
    return index


def estimate(previous, plan, journal, jobs):
    """Advisory wall-time range, not kernel-us extrapolation or a guarantee."""
    old_plan, _ = read(previous, 'execution-plan.json')
    old, _ = read(previous, 'authority.json')
    prior = PrefillJournal(previous/'arms', dict(sha256=digest(old)), '0'*32)
    times, current_wall = {}, 0.
    for p in old_plan['points']:
        times[p['id']] = []
        for j in jobs(p):
            row = prior.read(p,j)
            if not row or row['status'] != 'PASS': continue
            seconds = row.get('data',{}).get('wall_seconds')
            if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds <= 0: continue
            if row['identity']['authority'] == prior.authority: current_wall += seconds
            if j['recipe'] is not None: times[p['id']].append(seconds)
    summary, _ = read(previous, 'summary.json')
    residual = max(0., summary['elapsed_seconds']-current_wall)
    low = high = 0.; pending = missing = 0
    for p in plan['points']:
        due = [j for j in jobs(p) if journal.due(p,j)]
        pending += len(due)
        samples = times.get(p['id'], [])
        if not samples or any(j['recipe'] is None for j in due):
            missing += len(due); continue
        ordered = sorted(samples)
        low += len(due)*statistics.median(samples)
        high += len(due)*ordered[min(len(ordered)-1, int(.9*len(ordered)))]
    return dict(pending_jobs=pending, jobs_without_prior_wall_model=missing,
        remaining_hours_low=(low+residual)/3600 if not missing else None,
        remaining_hours_high=(high+residual)/3600 if not missing else None,
        prior_unattributed_wall_seconds=residual,
        method='SAME_POINT_TC_JOB_MEDIAN_TO_P90_PLUS_PRIOR_NONJOB_WALL',
        exclusions=['NEW_COMPILE_MISSES', 'FINALIST_CONFIRMATION', 'MODEL_AB'],
        duration_guaranteed=False, advisory_only=True)


class SupplementJournal(ArmJournal):
    def __init__(self, directory, authority, invocation):
        super().__init__(directory, authority, invocation)
        self.index = None
        if (self.root.parent/'supplement/index.json').exists():
            self.index, _ = read(self.root.parent, 'supplement/index.json')
            if self.index['authority_sha256'] != authority['sha256']:
                raise ValueError('supplement journal authority differs')

    def read(self, point, arm):
        row = super().read(point, arm)
        prior = self.index['receipts'].get(point['id']+'/'+arm['id']) if self.index else None
        if prior is None: return row
        if row is not None: raise ValueError('supplement remeasured an imported PASS')
        if self.identity(point, arm)['arm'] != prior['expected']:
            raise ValueError('supplement target arm changed')
        old, _ = read(self.root.parent, prior['path'])
        if (old['record_sha256'] != prior['seal'] or old['status'] != 'PASS' or
                old['identity']['point'] != point['id'] or old['identity']['workload'] != point['workload'] or
                old['identity']['authority'] != prior['source_authority']):
            raise ValueError('supplement imported PASS changed')
        return old

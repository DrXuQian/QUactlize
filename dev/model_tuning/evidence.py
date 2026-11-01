"""Normalize existing verified Asys receipts, not historical summary latencies."""
from collections import defaultdict
from pathlib import Path
import json
import statistics

from dev.model_tuning.policy import record, samples
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def read(path):
    return json.loads(Path(path).read_bytes())


def provenance(run, device=None):
    revision = run.get('source_revision', {})
    source = run.get('source', {})
    if not revision:
        # Dense Asys keeps its commit inside the source receipt.
        revision = run.get('revision', run.get('source_revision', {}))
    if not source or not run.get('harness') or not run.get('images',run.get('payload')):
        raise ValueError('run lacks source, harness or compiled binary identity; cannot export policy')
    device = device or run.get('device', {})
    props = device.get('device', device)
    target = dict(sdk=run.get('sdk', device.get('sdk')), device={k:v for k,v in props.items()
        if k not in ('host','ordinal','pci','uuid','clock_mhz','frequency_hz')})
    return dict(quactlize_commit=revision.get('quactlize_commit'), source_revision=revision,
        source_identity=digest(dict(source=source, harness=run.get('harness'))),
        target_identity=digest(target), target=target, physical_device=device,
        source_manifest_sha256=digest(source), harness_manifest_sha256=digest(run['harness']),
        binary_manifest_sha256=digest(run.get('images',run.get('payload'))),
        timing_method='ASYS_GPU_ACTIVITY')


def indexed_or_smallm(root, points, *, smallm):
    from tools.run_smallm_regressions_asys import sealed, checked_result
    from tools.run_moe_confirmation import completed
    root = Path(root); run = sealed(root/'run.json'); plan = sealed(root/'plan.json')
    specs = {p['id']:p for p in plan['points']}; out = {}; issues = []
    images = read(root/'images.json') if smallm else None
    for worker, ids in enumerate(run['assignments']):
        for key in ids:
            path = root/('gpu-'+run['cards'][worker])/'points'/(key+'.json')
            if not path.is_file():
                issues.append(dict(execution=key, reason='MISSING_CONFIRMATION')); continue
            row = sealed(path)
            if row.get('status') != 'PASS':
                issues.append(dict(execution=key, reason='FAILED_CONFIRMATION', log=str(path))); continue
            if smallm:
                checked_result(row, specs[key], run, images)
                data = row['data']; execution = row.get('execution_point', specs[key])
                definitions = {a['id']:dict(recipe=a['recipe']) for a in execution['arms']}
            else:
                completed(path, run)  # Includes raw batch/numerical receipt hashes.
                data = row; definitions = data['bindings']
            configs = [dict(binding=definitions[cid], samples_us=value['samples_us'], correctness='PASS',
                            components=[dict(data['component_records'][k], uses=1)
                                        for k in data['arm_components'][cid]]+
                                ([dict(data['arm_resets'][cid],kind='error_reset',uses=1)]
                                 if cid in data.get('arm_resets',{}) else []))
                       for cid, value in data['decision']['statistics'].items()]
            proof = dict(receipt=str(path), sha256=sha(path), run=run['sha256'])
            prov = provenance(run, run['devices'][worker])
            for point in points.get(key, ()):
                out[point['id']] = record(point, configs, data['decision']['statistics']['original']['mean_us'],
                                          prov, proof=proof)
    return list(out.values()), issues


def dense(root, points):
    from tools.review_prefill_cohort import ingest
    result = ingest(root); out = []; issues = []
    for key, point in result['points'].items():
        authority = result['authorities'][point['authority']]
        configs, controls = [], []
        for value in point['measurements']:
            if value['status'] != 'PASS':
                continue
            cost = statistics.fmean(samples(value['complete_samples_us']))
            if 'model-config' in value.get('roles', ()):
                controls.append(cost)
            if value['arm'] not in ('fq','sf','full-fp16'):
                continue
            binding = result['candidates'][value['candidate']]
            configs.append(dict(binding=binding, samples_us=value['complete_samples_us'],
                                correctness='PASS', components=[dict(result['components'][cid],
                                    component_id=cid,samples_us=ts,mean_us=statistics.fmean(ts),uses=1)
                                    for cid,ts in zip(value['components'],value['component_samples_us'])]))
        if not controls:
            issues.append(dict(execution=key, reason='MISSING_CURRENT_CONTROL')); continue
        for p in points.get(key, ()):
            out.append(record(p, configs, min(controls), provenance(authority),
                proof=dict(authority=point['authority'], measurement_sha256=digest(point), root=str(root))))
    issues += [dict(execution=key, reason='MISSING_CONFIRMATION') for key in result['incomplete']]
    return out, issues


def collect(inventory, campaign, run_root):
    by_backend = defaultdict(lambda:defaultdict(list))
    for point in inventory['points']:
        binding = campaign['bindings'].get(point['id'])
        if binding:
            by_backend[binding['backend']][binding['execution']].append(point)
    records, issues = [], list(campaign['pending'])
    run_root = Path(run_root)
    for job in campaign['jobs']:
        backend = job['backend']; directory = run_root/backend
        if backend == 'dense-prefill':
            roots = sorted(directory.glob('gpu-*/results'))
            roots = [p for p in roots if (p/'authority.json').is_file()]
            for root in roots:
                values, missing = dense(root, by_backend[backend]); records.extend(values); issues.extend(missing)
            if not roots:
                issues.append(dict(backend=backend, reason='NO_DENSE_RUN'))
        elif (directory/'run.json').is_file():
            values, missing = indexed_or_smallm(directory, by_backend[backend], smallm=backend.startswith('smallm-'))
            records.extend(values); issues.extend(missing)
        else:
            issues.append(dict(backend=backend, reason='NO_MEASURED_RUN'))
    return records, issues

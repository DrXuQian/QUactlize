#!/usr/bin/env python3
"""Import reviewed MoE Asys confirmations into one router-independent table."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.asys_reparse import measurement_provenance
from dev.strategy.moe_sweep import resolve, finalists, subset
from dev.strategy.smallm_asys import validate, check_producers
from quactlize.execution.moe_decode_policy import (
    ARCHIVE, COMMIT, TABLE, choose, configuration, worst_regret, validate as validate_policy, verify)
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sealed(value):
    require(value.get('sha256') == digest({k:v for k,v in value.items() if k != 'sha256'}),
            'MoE receipt seal differs')
    return value


def capture(path):
    """Stream the archive; do not keep multi-GB capture/trace samples in memory."""
    path = Path(path)
    require(sha(path) == ARCHIVE, 'unreviewed MoE archive')
    metadata, payloads, confirmations, screens, evidence = {}, {}, {}, {}, {}
    roots = {'plan.json', 'run.json', 'summary.json', 'images.json',
             'moe-inventory.json', 'moe-headers.json', 'transfer-manifest.json'}
    with tarfile.open(path, 'r|xz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            require(member.isfile() and not name.is_absolute() and '..' not in name.parts and
                    member.name not in payloads, 'unsafe/duplicate MoE archive member')
            raw = archive.extractfile(member).read()
            payloads[member.name] = dict(path=member.name, bytes=len(raw),
                                       sha256=hashlib.sha256(raw).hexdigest())
            if member.name in roots:
                metadata[member.name] = json.loads(raw)
                continue
            if not (len(name.parts) == 3 and name.parts[0].startswith('gpu-') and
                    name.parts[1] in ('points', 'screens') and name.suffix == '.json'):
                continue
            plan, run, images = (metadata[n+'.json'] for n in ('plan', 'run', 'images'))
            # Python 3.12 changed float sum(). Replay the measured interpreter
            # arithmetic rather than weaken the raw multi-component verifier.
            measured_python = tuple(map(int, run['python_version'].split()[0].split('.')[:2]))
            require(sys.version_info[:2] == measured_python,
                    'raw MoE recapture requires Python '+'.'.join(map(str, measured_python)))
            point, = [p for p in plan['points'] if p['id'] == name.stem]
            receipt = sealed(json.loads(raw))
            shard = next(i for i, ids in enumerate(run['assignments']) if point['id'] in ids)
            require(receipt['status'] == 'PASS' and receipt['point'] == point['id'] and
                    receipt['run'] == run['sha256'] and receipt['shard'] == shard and
                    name.parts[0] == 'gpu-'+run['cards'][shard], 'MoE point/worker binding differs')
            data = receipt['data']
            phase = 'screen' if name.parts[1] == 'screens' else 'confirm'
            require(data['workload'] == point['workload'] and data['measurement_phase'] == phase,
                    'MoE workload/phase differs')
            validate(data)
            check_producers(receipt['execution_point'], data)
            resolved = resolve(point, receipt['resource_queries'], images)
            authority = dict(run=run['sha256'], shard=shard, device=run['devices'][shard])
            provenance = data['latency_provenance']
            require(all(provenance.get(k) == v for k,v in measurement_provenance(run, shard).items()),
                    'MoE latency provenance differs')
            for component in data['component_records'].values():
                require(component['identity']['authority'] == authority and
                        component['identity']['implementation'] == run['source']['sha256'],
                        'MoE component authority differs')
            for filename, expected in receipt['evidence'].items():
                require(filename not in evidence or evidence[filename] == expected, 'mixed evidence hash')
                evidence[filename] = expected
                if filename.endswith('.asysrep'):
                    require(provenance['trace_sha256'] == expected, 'MoE trace binding differs')
                if filename.endswith('.sqlite'):
                    require(provenance['sqlite_sha256'] == expected, 'MoE SQLite binding differs')
            binding = dict(resolved=digest(resolved), queries=digest(receipt['resource_queries']))
            if phase == 'screen':
                require(receipt['execution_point'] == resolved and receipt['screen_sha256'] is None,
                        'MoE resource-screen inventory differs')
                screens[point['id']] = dict(binding, sha256=receipt['sha256'],
                    selected=digest(subset(resolved, finalists(resolved, data))), arms=len(data['arms']))
                continue
            arms = {a['id']:a for a in receipt['execution_point']['arms']}
            candidates = {}
            for aid, measured in data['arms'].items():
                arm = arms[aid]
                recipe = arm['recipe']
                image = images['tc' if recipe['kind'] == 'tc' else 'vectors'][arm['module']]
                proof = measured['receipt']
                require(proof['module_sha256'] == image['sha256'] and proof['build_key'] == image['key'],
                        'MoE measured module differs')
                require(measured['fixture']['l2_bytes'] == run['l2_bytes'], 'MoE cold-ring L2 differs')
                cid = digest(recipe)
                require(cid not in candidates, 'duplicate recipe in MoE confirmation')
                stats = data['decision']['statistics'][aid]
                candidates[cid] = dict(recipe=recipe, mean_us=stats['mean_us'], samples_us=stats['samples_us'],
                    components=[dict(id=k, kind=data['component_records'][k]['kind'],
                        mean_us=data['component_records'][k]['mean_us']) for k in data['arm_components'][aid]],
                    max_relative_l1=max(p['relative_l1_error'] for p in measured['numerical']),
                    module_sha256=proof['module_sha256'], producer=arm['expected_producer'],
                    noise_advisory=stats['noise_advisory'])
            old = configuration(arms['original']['recipe'])
            require(old == dict(kind=old['kind'], **point['selection']['config']),
                    'fresh original differs from public C++ query')
            confirmations[point['id']] = dict(binding, execution=digest(receipt['execution_point']),
                screen_sha256=receipt['screen_sha256'], decision=data['decision'],
                key=point['selection']['request'], original_config=old,
                original_policy=point['selection']['policy'],
                profile=dict(point=point['id'], router=point['workload']['router'],
                    original=digest(arms['original']['recipe']), candidates=candidates,
                    record_sha256=payloads[member.name]['sha256'], latency_provenance=provenance))

    transfer = metadata['transfer-manifest.json']
    members = {p['path']:p for p in transfer['included']}
    require(len(members) == len(transfer['included']) and
            set(payloads) == set(members) | {'transfer-manifest.json'} and
            all(payloads[n] == v for n,v in members.items()), 'MoE transfer inventory/hash differs')
    plan, run = (sealed(metadata[n+'.json']) for n in ('plan', 'run'))
    summary = metadata['summary.json']
    require(plan['cohort'] == 'moe-config-models' and run['plan'] == plan['sha256'] and
            run['source_revision']['quactlize_commit'] == COMMIT and summary['status'] == 'COMPLETE' and
            not any(summary['returncodes']) and summary['expected'] == 144 and
            run['timing_method'] == 'ASYS_GPU_ACTIVITY', 'MoE cohort/source/completion differs')
    ids = [pid for group in run['assignments'] for pid in group]
    require(len(ids) == len(set(ids)) == 144 and set(ids) == set(confirmations) == set(screens) ==
            {p['id'] for p in plan['points']}, 'MoE confirmation coverage differs')
    require(digest(metadata['moe-inventory.json']) == plan['coverage']['inventory_sha256'] and
            digest(metadata['moe-headers.json']) == metadata['moe-inventory.json']['headers_snapshot_sha256'],
            'MoE model/header inventory differs')
    for row in summary['points']:
        require(row['status'] == 'PASS' and row['decision'] == confirmations[row['point']]['decision'],
                'MoE summary differs from raw confirmation')
    for name, expected in evidence.items():
        relative = PurePosixPath(name).relative_to(transfer['source']).as_posix()
        if relative in payloads:
            require(payloads[relative]['sha256'] == expected, 'MoE capture hash differs')
        else:
            require(relative.endswith(('.asysrep', '.sqlite')), 'missing retained MoE evidence')
    groups = defaultdict(list)
    for pid, row in confirmations.items():
        screen = screens[pid]
        require(all(row[k] == screen[k] for k in ('resolved', 'queries')) and
                row['execution'] == screen['selected'] and row['screen_sha256'] == screen['sha256'],
                'MoE confirmation is not its fresh screen finalist pool')
        groups[tuple(row['key'])].append(row)
    # The other catalog layers are independent inputs. Bind their exact current
    # rows separately from the actual measurement's (previous MoE) control.
    from tools.generate_decode_selection import EVIDENCE, effective
    base = effective(json.loads(EVIDENCE.read_bytes()), include_moe=False)
    base_rows = {tuple(r['key']):r for r in base['exact']}
    rows = []
    for key, values in sorted(groups.items()):
        profiles = sorted((v['profile'] for v in values), key=lambda p:p['router'])
        selected = choose(profiles)
        config = configuration(profiles[0]['candidates'][selected]['recipe'])
        original = values[0]['original_config']
        require(all(v['original_config'] == original for v in values), 'MoE original changes across routers')
        row = base_rows[key]
        rows.append(dict(key=list(key), base_config=row['config'],
            base_dequant_config=row.get('dequant_config', -1), original_config=original,
            original_policy=values[0]['original_policy'], config=config, selected=selected,
            changed=config != row['config'], measurement_changed=config != original,
            worst_regret_pct=worst_regret(profiles, selected), profiles=profiles))
    bodies = {p:h for p,h in run['source']['files'].items()
        if p.startswith('quactlize/execution/') and Path(p).suffix in ('.h','.hpp','.cuh','.inc')}
    result = dict(schema='quactlize.moe-decode-exact.v1', archive_sha256=ARCHIVE,
        source_revision=run['source_revision'], run_sha256=run['sha256'], devices=run['devices'],
        body_hashes=bodies, rows=rows, coverage=summary['models'], deferred_models=summary['deferred_models'],
        timing_method='ASYS_GPU_ACTIVITY_MEAN_COMPONENT_SUM',
        selection='CONFIRMED_COMMON_MINIMAX_NO_PROFILE_REGRESSION',
        runtime_router_readback=False, bucket_changes=False, model_tpot_admitted=False,
        native_no_regression_admitted=False, bf16_storage_admitted=False,
        raw_reports_reopened=False, raw_report_location='RETAINED_ON_MEASUREMENT_BOX',
        scope='QWEN_TP_LOCAL_INDEXED_M1_8_F32_ENDPOINTS_BF16_COMPUTE_NOT_FUSION_OR_MODEL_AB')
    result['evidence_sha256'] = digest(result)
    validate_policy(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = capture(args.review)
    if args.check:
        verify(data)
        require(json.loads(TABLE.read_bytes()) == data, 'stored MoE evidence differs')
    else:
        TABLE.write_text(table_text(data))
    print('MOE_EXACT '+json.dumps(dict(rows=len(data['rows']), profiles=144,
        changed_vs_measured=sum(r['measurement_changed'] for r in data['rows']),
        evidence_sha256=data['evidence_sha256'], model_tpot_admitted=False, bucket_changes=False)))


def table_text(data):
    # One exact key per line keeps a machine-generated evidence refresh
    # reviewable without expanding every raw 6x5 sample into separate lines.
    head = json.dumps({k:v for k,v in data.items() if k != 'rows'},
                      sort_keys=True, indent=2, allow_nan=False)
    rows = ',\n'.join('    '+json.dumps(r, sort_keys=True, allow_nan=False) for r in data['rows'])
    return head[:-2]+',\n  "rows": [\n'+rows+'\n  ]\n}\n'


if __name__ == '__main__':
    main()

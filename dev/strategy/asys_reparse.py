"""Offline parser migration, retaining old measurement authority and verdicts."""
from pathlib import Path
import time

from dev.strategy.asys_components import parse
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from dev.strategy.smallm_asys import check_producers, METHOD
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


OVERLAP_ERRORS = {
    'isolated kernel activities overlap or are unordered',
    'isolated kernel activities overlap',
}


def measurement_provenance(run, shard):
    return dict(run['source_revision'], harness_sha256=run['harness']['sha256'],
                source_sha256=run['source']['sha256'], images_sha256=digest(run['images']),
                sdk=run['sdk'], device=run['devices'][shard], asys_sha256=run['asys_sha256'],
                python_version=run['python_version'])


def reparse(output, parser_root):
    # Import without opening the SDK or probing a device. This deliberately
    # does not compare an old measurement's source against today's checkout.
    from tools.run_smallm_regressions_asys import sealed, done
    from tools.run_decode_refresh import sealed as write_sealed

    output = Path(output).resolve(strict=True)
    plan, run = (sealed(output / name) for name in ('plan.json', 'run.json'))
    if plan['schema'] != 'quactlize.smallm-asys.v1' or run['plan'] != plan['sha256']:
        raise ValueError('old run/plan binding differs')
    ids = [pid for group in run['assignments'] for pid in group]
    if (len(ids) != len(set(ids)) or set(ids) != {p['id'] for p in plan['points']} or
            len(run['cards']) != len(run['assignments']) or run['timing_method'] != METHOD):
        raise ValueError('old run assignment/method differs')
    by_id = {p['id']: p for p in plan['points']}
    target = output / ('reparse-' + str(time.time_ns()))
    target.mkdir()  # Exclusive; never rewrite an old PASS, FAIL or summary.
    parser = harness_identity(parser_root)
    rows = []
    for shard, group in enumerate(run['assignments']):
        folder = output / ('gpu-' + run['cards'][shard])
        for pid in group:
            point = by_id[pid]
            old_path = folder / 'points' / (pid + '.json')
            try:
                old = sealed(old_path)
                if old['run'] != run['sha256'] or old['point'] != pid:
                    raise ValueError('old point binding differs')
                evidence = {str(old_path): sha(old_path)}
                if old['status'] == 'PASS':
                    done(old_path, point, run)
                    rows.append(dict(point=pid, status='PASS', action='PRESERVED_OLD_PASS',
                                     receipt=str(old_path), evidence=evidence,
                                     decision=old['data']['decision']))
                    continue
                if old.get('error') not in OVERLAP_ERRORS:
                    raise ValueError('failure is not an eligible timestamp-only parser rejection: ' + old.get('error', ''))
                attempt = Path(old['attempt']).resolve(strict=True)
                if not attempt.is_relative_to((folder / 'attempts').resolve(strict=True)):
                    raise ValueError('old attempt is outside its worker directory')
                failure_path = attempt / 'failure.json'
                if sealed(failure_path) != old:
                    raise ValueError('attempt failure and point receipt differ')
                capture_path = attempt / 'capture.json'
                captured = sealed(capture_path)
                authority = dict(run=run['sha256'], shard=shard, device=run['devices'][shard])
                if captured['point'] != pid or captured['authority'] != authority:
                    raise ValueError('old capture authority differs')
                db, report = attempt / 'trace.sqlite', attempt / 'trace.asysrep'
                for path in (failure_path, capture_path, db, report):
                    evidence[str(path)] = sha(path)
                data = parse(captured['data'], db, report, measurement_provenance(run, shard))
                check_producers(point, data)
                # Hashes first observed here are not retroactively claimed to
                # have been checked by the old failed parser.
                for path, expected in evidence.items():
                    if sha(Path(path)) != expected:
                        raise ValueError('evidence changed during offline parsing: ' + path)
                result_path = target / (pid + '.json')
                write_sealed(result_path, dict(status='PASS', point=pid, run=run['sha256'], data=data,
                    parser=parser, evidence=evidence, old_failure=str(failure_path),
                    raw_hash_scope='OBSERVED_DURING_OFFLINE_REPARSE', gpu_work=False,
                    production_changed=False, native_no_regression_admitted=False))
                rows.append(dict(point=pid, status='PASS', action='OFFLINE_REPARSED',
                                 receipt=str(result_path), evidence={str(result_path): sha(result_path)},
                                 decision=data['decision']))
            except (OSError, ValueError, KeyError) as exc:
                rows.append(dict(point=pid, status='FAIL', error=str(exc), action='NOT_ADMITTED'))
    result = dict(schema='quactlize.smallm-asys-reparse.v1', run=run['sha256'],
        status='COMPLETE' if all(r['status']=='PASS' for r in rows) else 'INCOMPLETE',
        expected=len(ids), points=rows, parser=parser, source_revision=run['source_revision'],
        timing_method=METHOD, gpu_work=False, old_receipts_modified=False,
        production_changed=False, native_no_regression_admitted=False, model_tpot_admitted=False)
    atomic_json(target / 'summary.json', result)
    return target, result

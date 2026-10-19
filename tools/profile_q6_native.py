#!/usr/bin/env python3
"""Fresh ABBA component timings, then three exact Q6 MoE ACU reports.

No llama rebuild, production edit, config search or full-model TPOT claim.
All profiled children have their own output: profiler timings are never used
as ordinary event samples. Old measurements remain read-only.
"""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.native_components.profile import POINT, WORKLOAD, selected, check_native, acu_command, check_report
from dev.native_components.plan import plan
from dev.native_components.fixture import weights, activation
from dev.native_components.accounting import checked_component
from dev.native_components.build import build_adapter
from dev.native_components.history import check_cohort
from dev.llama_ab.contract import family
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_native_components import carrier, environment, validated, device, point_request
from tools.probe_native_component_failures import check_binding


def execute(command, log, env):
    atomic_json(log.with_suffix('.command.json'), dict(argv=list(map(str, command))))
    start = time.monotonic()
    with log.open('x') as stream:
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        last = start
        try:
            while child.poll() is None:
                time.sleep(1)
                if time.monotonic() - last >= 30:
                    print(f'Q6_ACU_WAIT seconds={time.monotonic()-start:.1f} log={log}', flush=True)
                    last = time.monotonic()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    atomic_json(log.with_suffix('.process.json'), dict(rc=child.returncode, seconds=time.monotonic()-start))
    if child.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-25:])
        raise RuntimeError(f'child rc={child.returncode}; log={log}\n{tail}')


def candidate(a):
    import numpy as np
    from dev.strategy.vector import Compiler, spec
    from dev.strategy.cold_decode import Ring, prepared_arm, component_capture, publish_components
    from dev.strategy.indexed import point as indexed_point
    from dev.smallm_closure.bench import Context
    from quactlize.dispatch.table import Workload
    from quactlize.runtime.native import SDK, checked
    from tools.profile_kpack_gpu_compact import AcuRange
    row, recipe = selected(plan()['points'])
    request = json.loads(a.request.read_text())
    if request['row'] != row or request['authority']['device'] != device(a):
        raise ValueError('candidate plan/device changed')
    w = Workload(**WORKLOAD)
    item = spec(w, recipe)
    record = request['vector']
    Compiler(a.sdk, Path(record['path']).parent.parent).verify(item, record)
    sdk = SDK(a.sdk)
    ring = Ring(sdk, w, a.l2_bytes)
    b = None
    try:
        if (ring.w.record['raw_sha256'] != request['raw_fixture_receipt']['raw_sha256'] or
                ring.copies != request['native_point']['copies']):
            raise ValueError('candidate/native raw GGUF bytes or cold ring differ')
        b = Context(sdk, ring, indexed_point(w))
        fixture = activation(row)
        coeff = np.asarray(fixture['coefficients'][0], dtype='<f4').reshape(w.m*w.channels, 4)
        expected = np.ascontiguousarray(coeff[:, fixture['categories']])
        if not np.array_equal(b.data['a'], expected) or b.data['ids'].ravel().tolist() != request['native_point']['owners']:
            raise ValueError('candidate/native input bytes or expert routing differ')
        input_proof = dict(activation_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                           owners=request['native_point']['owners'], repeat=0)
        arm = dict(recipe=asdict(recipe), spec=item)
        with prepared_arm(b, w, arm, record, None, None) as session:
            # prepared_arm validates changed-input replay; restore input0 for
            # timing/profiling to match the native carrier's timed input.
            b.update(0)
            for i in range(ring.copies):
                checked(session.core.launch(i), 'matched input0 ring setup')
            session.proof()
            if len(session.capture.groups) != 1:
                raise ValueError('Q6 S1 unexpectedly added helper/reducer kernels')
            node = session.capture.descriptions[0]
            if 'quactlize::execution::simt::' not in node['name']:
                raise ValueError('selected Q6 call did not execute the shipping SIMT body')
            if a.child == 'profile':
                session.rt.sync()
                with AcuRange(sdk):
                    checked(session.core.launch(0), 'profile one warmed ordinary SIMT call')
                    session.rt.sync()
                rows, pending = [], []
            else:
                rows, pending = component_capture(session.rt, session.capture, w, recipe,
                    digest(request['authority']), record['sha256'], a.output, session.proof)
        publish_components(pending)
        result = dict(session.data, stages=rows, node=node, input=input_proof,
                      sum_us=sum(r['median_us'] for r in rows) if rows else None,
                      profiled=a.child == 'profile', vector_sha256=record['sha256'],
                      authority=request['authority'])
    finally:
        if b:
            b.close()
        ring.close()
    atomic_json(a.output/'candidate.json', result)
    print('Q6_ACU_CANDIDATE ' + json.dumps(dict(correctness=result['correctness'],
          sum_us=result['sum_us'], profiled=result['profiled'])), flush=True)
    return 0


def archive(output):
    target = output/'review.tgz'
    with tarfile.open(target, 'w:gz') as tar:
        for path in sorted(output.rglob('*')):
            parts = path.relative_to(output).parts
            if (path.is_file() and not path.is_symlink() and
                    (path.suffix in ('.json', '.log', '.csv', '.acurep') or path.name == 'profile')
                    and not {'fixtures', 'vector-cache', 'source'} & set(parts)):
                tar.add(path, arcname=str(path.relative_to(output)), recursive=False)
    return target


def profile_one(a, name, expected, native_point, request, binary, candidate_cmd, env):
    out = a.output/name
    out.mkdir()
    is_native = name.startswith('native-')
    if is_native:
        # Two allocations, original input0 during excluded warmups.
        # Diagnostics verify both replays, but deliberately do not time.
        diagnostic = dict(native_point, copies=2, ring=dict(scope='PROFILE_NOT_ROTATING_TIMING'))
        atomic_json(out/'request.json', dict(request, points=[diagnostic], diagnostic=True))
        command = [str(binary), str(out/'request.json'), str(out)]
    else:
        command = candidate_cmd(out, 'profile')
    command = acu_command(a.acu, out/'profile', command,
                          symbol=expected['symbol'] if is_native else None)
    print(f'Q6_ACU_PROFILE arm={name} log={out / "acu.log"}', flush=True)
    execute(command, out/'acu.log', env)
    proof_path = out/(POINT+'.diagnostic.json') if is_native else out/'candidate.json'
    proof = json.loads(proof_path.read_text())
    if proof.get('authority') != request['authority']:
        raise ValueError('profiled child authority differs')
    if is_native:
        if proof['status'] != 'DIAGNOSTIC_COMPLETE' or proof['linked_backend_comparison'] != 'PASS_TWO_INPUTS':
            raise ValueError('profiled native output/replay check failed')
        check_native(proof['inventory'])
    elif proof['correctness'] != 'PASS' or not proof['profiled'] or proof['node'] != expected:
        raise ValueError('profiled candidate entry/correctness differs')
    paths = [p for p in (out/'profile', out/'profile.acurep') if p.is_file()]
    if len(paths) != 1:
        raise ValueError('ACU produced missing/ambiguous report')
    report = paths[0]
    export = [str(a.acu), '--import', str(report), '--page', 'raw', '--csv']
    execute(export, out/'raw.csv', env)
    observed = check_report((out/'raw.csv').read_text(errors='replace'), expected)
    return dict(name=name, status='PASS', report=str(report), report_sha256=sha(report), **observed)


def profiles(a, jobs, native_point, request, binary, candidate_cmd, env):
    reports = []
    for name, expected in jobs:
        try:
            reports.append(profile_one(a, name, expected, native_point, request, binary, candidate_cmd, env))
        except Exception as exc:
            failure = dict(name=name, status='FAIL', error=str(exc), traceback=traceback.format_exc())
            atomic_json(a.output/name/'failure.json', failure)
            reports.append(failure)
            print(f'Q6_ACU_PROFILE_FAIL arm={name} error={exc} remaining_continue=1', flush=True)
    return reports


def collect(a):
    from dev.strategy.vector import Compiler, spec
    from quactlize.dispatch.table import Workload
    row, recipe = selected(plan()['points'])
    # Record exact historical selection; do not silently pick a different Q6
    # shape if the original run is missing or the caller has since changed.
    matches = list(a.previous.glob('gpu-*/results/' + POINT + '.json'))
    if len(matches) != 1:
        raise ValueError('previous run must contain exactly one measured Q6 point')
    old_path = matches[0]
    old = json.loads(old_path.read_text())
    if old.get('status') != 'MEASURED' or old['data']['workload'] != WORKLOAD:
        raise ValueError('previous Q6 point was not measured with this workload')
    check_native(old['data']['components'])
    worker = old_path.parent.parent
    frozen = dict(row=row, previous_record=str(old_path), previous_sha256=sha(old_path),
                  native_historical_us=old['sum_us'], sequence=['native', 'kpack', 'kpack', 'native'],
                  profiles=['kpack-producer', 'native-producer', 'native-adapter'],
                  production_changed=False, scope='ONE_Q6_MOE_POINT_NOT_SWEEP_OR_MODEL_TPOT')
    atomic_json(a.output/'plan.json', frozen)
    if a.plan_only:
        print('Q6_ACU_PLAN ' + json.dumps(frozen), flush=True)
        return 0
    if not a.confirm_idle:
        raise ValueError('confirm an otherwise idle GPU; CUDA Graphs do not isolate other workloads')
    dev = device(a)
    check_cohort(dev)
    old_dev = old['data']['authority']['device']
    if old_dev['sdk'] != dev['sdk'] or old_dev['device']['compute_units'] != dev['device']['compute_units']:
        raise ValueError('Q6 historical SDK/device class differs; do not attribute an upgrade to the kernel')
    held = (Path('/tmp')/('quactlize-components-'+digest(dev['device']['pci'])[:24]+'.lock')).open('a')
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        harness = harness_identity(ROOT)
        execute([str(a.acu), '--version'], a.output/'acu-version.log', environment(a))
        execute([str(a.acu), '--help'], a.output/'acu-help.log', environment(a))
        adapter, receipt = build_adapter(a.llama_source, a.llama_build, a.sdk, a.output/'build')
        check_binding(json.loads((worker/'build/adapter.json').read_text()), receipt, dev)
        binary = carrier(a, a.output/'build')
        vector = Compiler(a.sdk, a.output/'vector-cache').build(spec(Workload(**WORKLOAD), recipe))
        atomic_json(a.output/'vector.json', vector)
        prior = worker/'fixtures'/family(row)
        folder = prior if (prior/'weights.raw').is_file() and (prior/'weights.json').is_file() else a.output/'fixtures'
        raw, raw_receipt = weights(WORKLOAD, folder)
        native_point = dict(point_request(row, a.l2_bytes), input_fixture=activation(row))
        runtime = next(p for p in (a.sdk/'lib/libhggc_wrapper.so',
            a.sdk/'targets/x86_64-linux/lib/libhggc_wrapper.so') if p.is_file())
        authority = dict(device=dev, adapter=receipt['binary_sha256'], carrier=sha(binary),
                         backend=receipt['identity']['libraries'], vector=vector['sha256'],
                         harness=harness, previous_sha256=sha(old_path), rounds=3, samples=3,
                         profiler=dict(path=str(a.acu.resolve()), sha256=sha(a.acu)))
        request = dict(runtime_library=str(runtime), adapter_library=str(adapter),
                       raw_fixture=str(raw), raw_fixture_receipt=raw_receipt,
                       points=[native_point], authority=authority, rounds=3, samples=3)
        candidate_request = dict(row=row, native_point=native_point, raw_fixture_receipt=raw_receipt,
                                 vector=vector, authority=authority)
        atomic_json(a.output/'candidate-request.json', candidate_request)
        atomic_json(a.output/'authority.json', authority)
        env = environment(a)

        def candidate_cmd(output, mode):
            return [sys.executable, '-u', str(Path(__file__)), '--child', mode,
                    '--request', str(a.output/'candidate-request.json'), '--output', str(output),
                    '--sdk', str(a.sdk), '--l2-bytes', str(a.l2_bytes)]

        measured = []
        for index, arm in enumerate(frozen['sequence']):
            if harness_identity(ROOT) != harness:
                raise ValueError('profile harness changed during execution')
            out = a.output/(str(index)+'-'+arm)
            out.mkdir()
            print(f'Q6_ACU_TIMING arm={arm} pass={index+1}/4 log={out / "run.log"}', flush=True)
            if arm == 'native':
                atomic_json(out/'request.json', request)
                execute([str(binary), str(out/'request.json'), str(out)], out/'run.log', env)
                data = validated(out/(POINT+'.json'), authority)
                check_native(data['components'])
                cost = sum(checked_component(c)['median_us'] for c in data['components'])
            else:
                execute(candidate_cmd(out, 'timing'), out/'run.log', env)
                data = json.loads((out/'candidate.json').read_text())
                if data['correctness'] != 'PASS' or data['profiled'] or data['authority'] != authority:
                    raise ValueError('candidate event receipt did not pass')
                cost = data['sum_us']
            measured.append(dict(arm=arm, sum_us=cost, output=str(out)))
        native = statistics.median(r['sum_us'] for r in measured if r['arm'] == 'native')
        kpack = statistics.median(r['sum_us'] for r in measured if r['arm'] == 'kpack')
        timing = dict(runs=measured, native_us=native, kpack_us=kpack, delta_pct=100*(kpack/native-1),
                      scope='SUM_OF_SEPARATELY_TIMED_KERNELS_NOT_MODEL_E2E', no_regression_admitted=False)
        atomic_json(a.output/'timing.json', timing)
        print('Q6_ACU_TIMING_DONE ' + json.dumps(timing), flush=True)

        candidate_node = json.loads((a.output/'1-kpack/candidate.json').read_text())['node']
        native_nodes = json.loads((a.output/'0-native'/(POINT+'.json')).read_text())['components']
        reports = profiles(a, [('kpack-producer', candidate_node), ('native-producer', native_nodes[1]),
                               ('native-adapter', native_nodes[0])], native_point, request, binary, candidate_cmd, env)
        if harness_identity(ROOT) != harness or sha(old_path) != authority['previous_sha256']:
            raise ValueError('harness or prior evidence changed')
        status = 'PASS' if all(r['status'] == 'PASS' for r in reports) else 'INCOMPLETE'
        atomic_json(a.output/'summary.json', dict(status=status, timing=timing, reports=reports,
                    production_changed=False, model_tpot_admitted=False,
                    scope='ONE_Q6_MOE_ACU_DIAGNOSIS_NOT_ALL_FORMAT_REGRESSION_ADMISSION'))
        print(f'Q6_ACU_DONE status={status} reports={sum(r["status"]=="PASS" for r in reports)}/3 production_changed=0', flush=True)
    finally:
        held.close()
    return 0 if status == 'PASS' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path)
    p.add_argument('--llama-source', type=Path, default=Path('/sim/eec/shared/junfu.qx/llama.cpp.ncp'))
    p.add_argument('--llama-build', type=Path)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    p.add_argument('--acu', type=Path)
    p.add_argument('--l2-bytes', type=int, default=67108864)
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--child', choices=('timing', 'profile'))
    p.add_argument('--request', type=Path)
    a = p.parse_args()
    a.output = a.output.resolve()
    if not a.child and a.output.exists() and any(a.output.iterdir()):
        p.error('use a new output directory; never overwrite prior measurements')
    a.output.mkdir(parents=True, exist_ok=True)
    if a.child:
        if not a.request:
            p.error('--child requires --request')
        if (a.output/'candidate.json').exists():
            p.error('candidate result already exists; use a new output')
        return candidate(a)
    if not a.previous:
        p.error('--previous is required')
    a.previous = a.previous.resolve(strict=True)
    if a.output == a.previous or a.output.is_relative_to(a.previous):
        p.error('profile output must be separate from the original run')
    a.llama_build = (a.llama_build or a.llama_source/'build-ci').resolve()
    a.sdk = a.sdk.resolve()
    a.acu = a.acu or a.sdk/'asight/bin/acu'
    if ',' in os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        p.error('one visible GPU only')
    if not a.plan_only:
        if not a.acu.is_file():
            p.error('ACU executable missing; set --acu')
        os.environ.update(environment(a))
    try:
        return collect(a)
    except Exception as exc:
        atomic_json(a.output/'failure.json', dict(status='FAIL', error=str(exc), traceback=traceback.format_exc(),
                                                production_changed=False, performance_admitted=False))
        traceback.print_exc()
        return 1
    finally:
        print('results=' + str(archive(a.output)), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

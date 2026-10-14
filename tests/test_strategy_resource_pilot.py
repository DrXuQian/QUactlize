"""Resource-reuse and bounded pilot contracts, without a PPU."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dev.strategy.journal import atomic_json
from dev.strategy.resource_pilot import load_probe, smoke_cases, module_record, current_module
from dev.strategy.resources import inventory, resolve_point
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.probe_kpack_strategy import write
from test_strategy_prescreen import tc


def passed_result(case):
    from dev.strategy.components import statistics_checked
    w = Workload(**case['workload'])
    proof = dict(cells=w.m*w.topk*w.n, bad=0, nonfinite=0, tolerance=.005, relative_l1_error=.0001)
    component = dict(statistics_checked([[1.]*3 for _ in range(3)]), role='producer',
                     node=dict(name='void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<Shape>>(Params)'),
                     timed_kernel_types=1, first_use_excluded=True, cache='WARM_SMOKE_NO_COLD_OR_PERFORMANCE_ADMISSION')
    return dict(status='PASS', scope='WARM_TC_MECHANISM_GATE_NOT_COLD_SELECTION',
        production_changed=False, performance_admitted=False, numerical=[proof]*5, extracted=proof,
        negatives=dict(zero_input='DETECTED', omitted_producer='DETECTED', ring_planes=['low','units']),
        components=[component])


def example(tmp_path):
    w = Workload(12, 1024, 2048, 1)
    r = tc()
    recipes = [Recipe.make('tc', 'fq', **(dict(r.parameters) | dict(split=s, grid_mode=0, grid_b=0))) for s in (1, 2, 4)]
    point = dict(id=w.key, workload=asdict(w), candidates=[asdict(c) for c in recipes], champion_controls=[])
    plan = dict(cohort='pilot', points=[point], source_identity=dict(sha256='old-rule-source'))
    plan['plan_sha256'] = digest(plan)
    atomic_json(tmp_path / 'plan/plan.json', plan)
    plan = json.loads((tmp_path / 'plan/plan.json').read_text())
    authority = dict(plan=plan['plan_sha256'], source='old-rule-source', sdk='old-sdk',
                     device=dict(ordinal=0, compute_units=72, pci='host-pci'))
    inv = inventory(plan)
    atomic_json(tmp_path / 'resources/authority.json', authority)
    atomic_json(tmp_path / 'resources/inventory.json', inv)
    receipts = {}
    for key, query in inv['queries'].items():
        row = dict(status='READY', request=query['request'], occupancy=3, device=dict(ordinal=0, compute_units=72),
                   workspace_bytes=1024, shared_bytes=2048)
        receipts[key] = row
        write(tmp_path / 'resources/queries' / (key + '.json'), dict(authority=authority, query=key), row)
    for mid, spec in inv['modules'].items():
        path = tmp_path / 'cache/mock-key/kernel.so'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not-a-device-library; host-only path/hash test')
        record = dict(parent=spec['parent'], key='mock-key', identity=dict(kernel='old'), sha256=sha(path))
        atomic_json(path.parent / 'manifest.json', record)
        write(tmp_path / 'resources/modules' / (mid + '.json'), dict(authority=authority, module=mid),
              dict(status='PASS', record=dict(record, path=str(path))))
    resolved = dict(plan, points=[resolve_point(p, receipts) for p in plan['points']], resource_authority=authority,
                    resource_receipts_sha256=digest(receipts))
    resolved['plan_sha256'] = digest({k: v for k, v in resolved.items() if k != 'plan_sha256'})
    atomic_json(tmp_path / 'resources/resolved-plan.json', resolved)
    atomic_json(tmp_path / 'resources/summary.json', dict(status='COMPLETE', missing=[], counts=inv['counts'],
                ready=len(receipts), unavailable=0, kernel_launches=0))
    return load_probe(tmp_path)


def test_import_complete_resource_probe_and_bound_s1_split(tmp_path):
    probe = example(tmp_path)
    smoke = smoke_cases(probe)
    assert len(smoke['cases']) == 2
    assert [c['binding']['launch']['split'] for c in smoke['cases']] == [1, 2]
    assert smoke['recommendation'] is None and not smoke['performance_admitted']
    assert smoke == json.loads(json.dumps(smoke))
    assert module_record(probe, smoke['cases'][0])['key'] == 'mock-key'


def test_resource_summary_is_not_enough_without_query_receipts(tmp_path):
    probe = example(tmp_path)
    path = next((tmp_path / 'resources/queries').glob('*.json'))
    data = json.loads(path.read_text())
    data['data']['occupancy'] = 100
    atomic_json(path, data)
    with pytest.raises(ValueError, match='digest'):
        load_probe(tmp_path)


def test_forged_resolved_grid_is_rejected_even_if_resealed(tmp_path):
    example(tmp_path)
    path = tmp_path / 'resources/resolved-plan.json'
    data = json.loads(path.read_text())
    binding = next(iter(data['points'][0]['resource_resolution']['bindings'].values()))
    binding['launch']['grid'] = 123456
    data['plan_sha256'] = digest({k: v for k, v in data.items() if k != 'plan_sha256'})
    atomic_json(path, data)
    with pytest.raises(ValueError, match='grids'):
        load_probe(tmp_path)


def test_changed_cached_payload_cannot_be_loaded(tmp_path):
    probe = example(tmp_path)
    case = smoke_cases(probe)['cases'][0]
    record = module_record(probe, case)
    Path(record['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        module_record(probe, case)


def test_current_module_check_never_builds_and_rejects_stale_compiler():
    spec = dict(parent=dict(symbol='host-test'))
    calls = []
    compiler = SimpleNamespace(identity=dict(kernel='same', sdk='same'),
        source=lambda parent, key: 'body', check_inputs=lambda: calls.append('checked'),
        build=lambda parent: pytest.fail('no JIT allowed in resource reuse'))
    record = dict(identity=compiler.identity, key=digest(dict(identity=compiler.identity, parent=spec['parent'], source='body')))
    current_module(record, spec, compiler)
    assert calls == ['checked']
    compiler.identity = dict(kernel='changed', sdk='same')
    with pytest.raises(ValueError, match='compiled source'):
        current_module(record, spec, compiler)


def test_plan_only_needs_no_sdk_or_gpu_packages(tmp_path):
    resource = tmp_path / 'resource'
    example(resource)
    result = subprocess.run([sys.executable, 'tools/run_kpack_strategy_pilot.py', '--resource-run', str(resource),
        '--output', str(tmp_path / 'output'), '--plan-only', '--sdk', str(tmp_path / 'NO_SDK')],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"compile_modules": 0' in result.stdout
    assert not (tmp_path / 'output/authority.json').exists()


def test_supplemental_warm_timing_cannot_claim_cold_or_pick_winner(tmp_path):
    smoke = smoke_cases(example(tmp_path))
    assert 'COLD_WEIGHT_COMPARISON' in smoke['excluded']
    assert 'ORIGINAL_WIRING_AB' in smoke['excluded']
    assert 'KQUANT_SF_EXPANSION' in smoke['excluded']
    assert all(c['timing'] == 'WARM_SINGLE_KERNEL_SMOKE_NOT_COLD_PERFORMANCE' for c in smoke['cases'])


def test_pass_requires_complete_oracles_negatives_and_recomputed_statistics(tmp_path):
    import copy
    from dev.strategy.resource_pilot import validate_pass
    case = smoke_cases(example(tmp_path))['cases'][0]
    result = passed_result(case)
    assert validate_pass(case, result) == result
    for change in ('numerical', 'negative', 'statistic', 'combined', 'wrong_kernel'):
        bad = copy.deepcopy(result)
        if change == 'numerical': bad['numerical'] = []
        if change == 'negative': bad['negatives']['omitted_producer'] = 'NOT_TESTED'
        if change == 'statistic': bad['components'][0]['median_us'] = .01
        if change == 'combined': bad['components'][0]['timed_kernel_types'] = 3
        if change == 'wrong_kernel': bad['components'][0]['node']['name'] = 'indexed_prepare<8>'
        with pytest.raises(ValueError): validate_pass(case, bad)


def previous_pilot(tmp_path):
    probe = example(tmp_path / 'resource')
    plan = smoke_cases(probe)
    plan['resource_plan'] = probe['resolved']['plan_sha256']
    plan['plan_sha256'] = digest(plan)
    directory = tmp_path / 'previous'
    authority = dict(plan=plan['plan_sha256'], harness=dict(sha256='old-harness'),
                     resource=probe['authority'], oracle='same-oracle', protocol='same-protocol')
    atomic_json(directory / 'plan.json', plan)
    atomic_json(directory / 'authority.json', authority)
    for case in plan['cases']:
        result = passed_result(case) if case['split_kind'] == 's1' else dict(status='FAIL', error='classification')
        write(directory / 'cases' / (case['id'] + '.json'), dict(authority=authority, case=case), result)
    return plan, directory, authority


def test_failed_only_retry_preserves_raw_pass_without_relabelling(tmp_path):
    from dev.strategy.resource_pilot import retry_failed, validate_retry_authority
    plan, directory, authority = previous_pilot(tmp_path)
    before = {p: p.read_bytes() for p in directory.rglob('*.json')}
    retry = retry_failed(plan, directory)
    assert len(retry['cases']) == 1 and retry['cases'][0]['split_kind'] == 'split'
    assert retry['retry']['previous_passed'] == [plan['cases'][0]['id']]
    assert retry['retry']['original_expected'] == 2
    assert retry['retry']['previous_harness_sha256'] == 'old-harness'
    assert before == {p: p.read_bytes() for p in directory.rglob('*.json')}
    current = dict(authority, plan=retry['plan_sha256'], harness=dict(sha256='fixed-harness'))
    validate_retry_authority(retry, current)
    with pytest.raises(ValueError, match='authority changed'):
        validate_retry_authority(retry, dict(current, oracle='different-oracle'))


def test_retry_rejects_changed_plan_and_damaged_previous_receipt(tmp_path):
    from dev.strategy.resource_pilot import retry_failed
    plan, directory, authority = previous_pilot(tmp_path)
    with pytest.raises(ValueError, match='identical original'):
        retry_failed(dict(plan, scope='changed'), directory)
    path = directory / 'cases' / (plan['cases'][0]['id'] + '.json')
    damaged = json.loads(path.read_text())
    damaged['data']['numerical'] = []
    atomic_json(path, damaged)
    with pytest.raises(ValueError, match='digest'):
        retry_failed(plan, directory)


def test_box_failed_only_plan_needs_no_device_or_rebuild(tmp_path):
    import os
    plan, directory, authority = previous_pilot(tmp_path)
    # Box convention is run/pilot, while the Python runner accepts pilot.
    run = tmp_path / 'old-run'
    run.mkdir()
    directory.rename(run / 'pilot')
    env = dict(os.environ, PROBE_RUN=str(tmp_path / 'resource'), RUN_ROOT=str(tmp_path),
               RETRY_FROM=str(run), PLAN_ONLY='1', PYTHON=sys.executable)
    result = subprocess.run(['bash', 'tools/run_kpack_strategy_pilot_box.sh'], env=env,
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"cases": 1' in result.stdout and '"previous_passed": 1' in result.stdout
    assert '"compile_modules": 0' in result.stdout


def test_box_plan_only_has_no_runtime_dependency_and_preserves_shell(tmp_path):
    import os
    resource = tmp_path / 'resource'
    example(resource)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PROBE_RUN=str(resource), RUN_ROOT=str(tmp_path), PLAN_ONLY='1', PYTHON=sys.executable)
    result = subprocess.run(['bash', str(root/'tools/run_kpack_strategy_pilot_box.sh')],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'runner_rc=0 stage=complete' in result.stdout
    assert 'Current Docker shell is preserved' in result.stdout
    assert len(list(tmp_path.glob('kpack-rule-exec.*.results.tgz'))) == 1


def test_grouped_fused_reducer_finish_has_explicit_timing_scope():
    from dev.strategy.components import stage_role
    assert stage_role('indexed_prepare<8>', 2) == 'prepare'
    assert stage_role('indexed_finish<2>', 2) == 'reducer_finish'
    assert stage_role('indexed_finish<1>', 1) == 'postprocess'
    assert stage_role('reduce_decode<2>', 2) == 'reducer'


def test_large_bf16_fixture_preserves_f32_upload_and_f64_oracle():
    pytest.importorskip('torch')
    import numpy as np
    from dev.strategy.pilot_native import scale_activation_data
    data = dict(a=np.array([.25, -.5], dtype='f4'), gold=np.array([2., -3.], dtype='f8'),
                denom=np.array([4., 5.], dtype='f8'))
    scale_activation_data(data, 2**20)
    assert data['a'].dtype == np.dtype('f4') and data['gold'].dtype == data['denom'].dtype == np.dtype('f8')
    assert np.isfinite(data['a']).all() and np.max(np.abs(data['a'])) > 65504
    assert data['a'].tolist() == [262144., -524288.]


@pytest.mark.parametrize('native', [False, True])
def test_dense_carrier_uses_exact_typed_or_native_pointer_abi(monkeypatch, native):
    pytest.importorskip('torch')
    import ctypes as C
    import dev.strategy.pilot_native as carrier
    from dev.strategy.resources import module_spec, probe_spec
    from quactlize.runtime.native import Resources as Query, Call
    from quactlize.decode.native_compute import DenseComputeCall
    w = Workload(12, 1024, 2048, 512 if native else 8)
    recipe = tc(split=2)
    spec = module_spec(w, recipe)
    seen = []
    class Owner:
        def __init__(self, sdk): pass
        def close(self): pass
    class Buffer:
        ptr_next = 0x10000
        def __init__(self, owner, size):
            self.ptr, self.size = Buffer.ptr_next, size
            Buffer.ptr_next += max(256, size)
        def upload(self, value): pass
    class Probe:
        device = dict(ordinal=0, compute_units=72)
        lib = object()
        def __init__(self, spec, record): pass
        def _bind(self, name, args, result=C.c_int):
            def invoke(*values):
                seen.append(name)
                if 'query' in name or 'prepare' in name:
                    request = C.cast(values[0], C.POINTER(Call if native else DenseComputeCall)).contents
                    call = request if native else request.dense.call
                    assert call.m == w.m and call.compute_units == 72
                    if not native:
                        assert request.dense.input_type == request.dense.output_type == 1
                    if 'query' in name:
                        assert not call.a
                        query = C.cast(values[2], C.POINTER(Query)).contents
                        query.occupancy, query.workspace_bytes, query.shared_bytes = 3, 4096, 8192
                    else:
                        assert call.a and call.output and call.workspace and call.workspace_bytes == 4096
                        assert call.stream == 17 and call.low in (0x30000, 0x40000)
                        assert call.metadata == call.low + 128 and not call.zero
                        C.cast(values[2], C.POINTER(C.c_void_p))[0] = C.c_void_p(42)
                return 0
            return invoke
    monkeypatch.setattr(carrier, 'Resources', Owner)
    monkeypatch.setattr(carrier, 'Buffer', Buffer)
    monkeypatch.setattr(carrier, 'NativeProbe', Probe)
    b = SimpleNamespace(rows=w.m, p=dict(k=w.k, n=w.n), r=SimpleNamespace(stream=C.c_void_p(17)),
        sdk=SimpleNamespace(synchronize=lambda stream: None), a=Buffer(None, 4), out=Buffer(None, 4),
        ring=SimpleNamespace(planes=[dict(low=p, high=None, units=p+128) for p in (0x30000, 0x40000)]))
    case = dict(spec=spec, binding=dict(launch=dict(algorithm=0, split=2, grid=0)))
    resources = dict(request=probe_spec(w, recipe, 0), occupancy=3, workspace_bytes=4096, shared_bytes=8192)
    core = carrier.DenseCore(b, dict(key='mock', sha256='mock'), case, resources)
    assert core.receipt['storage'] == ('F16_NATIVE' if native else 'F32')
    assert len(core.handles) == 2 and core.launch(1) == 0
    core.close()
    assert seen.count('destroy_v1' if native else 'decode_dense_destroy_v1') == 2

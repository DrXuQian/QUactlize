"""Host contracts for the new-clock refresh; never stand in for device results."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import pytest

from dev.strategy.refresh import make_plan, request, family, assign, Q8_BODIES
from dev.strategy.refresh_clock import ADVISORY, parse, observe, request_lock
from dev.strategy.vector import spec, source
from dev.llama_ab.contract import inventory
from quactlize.dispatch.planning import plan_smallm
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    directory = tmp_path_factory.mktemp('decode-refresh-host')
    points = [p for p in inventory() if p['cohort'] == 'decode']
    original = plan_smallm(directory, [request(Workload(**p['workload'])) for p in points])
    return make_plan(original['requests'])


def test_full_measured_registry_not_reduced_to_previous_passing_subset(plan):
    wanted = [p for p in inventory() if p['cohort'] == 'decode']
    assert len(wanted) == len(plan['points']) == 1114
    assert {p['id'] for p in wanted} == {p['id'] for p in plan['points']}
    assert plan['counts']['operations'] == dict(dense=530, indexed=584)
    assert plan['counts']['families'] == 104
    assert plan['historical_times_reused'] is False
    assert plan['production_changed'] is False
    assert plan['sha256'] == digest({k: v for k, v in plan.items() if k != 'sha256'})
    for p in plan['points']:
        arms = p['arms']
        assert len(arms) <= 48
        assert len({a['id'] for a in arms}) == len(arms)
        original = next(a for a in arms if a['id'] == 'original')
        assert original['required']
        assert original['recipe'] == p['original_recipe']
        assert 'CURRENT_CPP_SELECTED_RECIPE' in original['reasons']
        assert any(a['recipe']['kind'] == 'tc' for a in arms)
        assert any(a['recipe']['kind'] == 'simt' for a in arms)
        for a in arms:
            table = plan['tc'] if a['recipe']['kind'] == 'tc' else plan['vectors']
            assert table[a['module']] == a['spec']
            assert digest(a['spec']) == a['module']


def test_q8_address_variants_present_for_every_dense_point_not_blanket_promotion(plan):
    total = 0
    for p in plan['points']:
        q8 = [a for a in p['arms'] if 'q8_body' in a['spec']]
        if p['workload']['qtype'] == 8 and p['workload']['operation'] == 'dense':
            total += 1
            assert len(q8) == 2 * len(Q8_BODIES)
            assert {a['spec']['q8_body'] for a in q8} == set(Q8_BODIES)
            assert {a['spec']['config']['columns'] for a in q8} == {4, 8}
            assert all(a['id'] != 'original' for a in q8)
            assert all(dict(Recipe.from_dict(a['recipe']).parameters)['split'] == 1 for a in q8)
        else:
            assert not q8
    assert total == 73


def test_family_assignments_disjoint_complete_stable(plan):
    a = assign(plan, 7)
    assert a == assign(deepcopy(plan), 7)
    flat = [key for rows in a for key in rows]
    assert len(flat) == len(set(flat)) == 104
    assert all(a)
    for p in plan['points']:
        assert sum(family(Workload(**p['workload'])) in keys for keys in a) == 1
    with pytest.raises(ValueError):
        assign(plan, 0)
    with pytest.raises(ValueError):
        assign(plan, 105)


def test_resource_rejection_is_visible_not_missing_incumbent_or_measured_loser(plan, monkeypatch):
    import dev.strategy.resources as resource
    from tools.run_decode_refresh import bind_point
    p = next(p for p in plan['points'] if any(not a['required'] and a['recipe']['kind'] == 'tc' for a in p['arms']))
    arms = [deepcopy(a) for a in p['arms'] if not a['required'] and a['recipe']['kind'] == 'tc'][:1]
    item = dict(p, arms=arms)
    records = {'tc': {arms[0]['module']: {}}, 'vectors': {}}
    class Unsupported:
        def __init__(self, *args):
            pass
        def query(self, request):
            return dict(status='UNAVAILABLE', rc=1, runtime_status=0, request=request)
    monkeypatch.setattr(resource, 'NativeProbe', Unsupported)
    result, bindings = bind_point(item, records)
    assert not result['arms'] and not bindings
    assert list(result['resource_unavailable']) == [arms[0]['id']]
    arms[0]['required'] = True
    with pytest.raises(ValueError, match='required incumbent'):
        bind_point(item, records)


@pytest.mark.parametrize('columns,warps', [(8, 4), (4, 8)])
@pytest.mark.parametrize('compute', ['f16', 'bf16'])
def test_new_adapter_reuses_measured_body_and_keeps_arm_identity(columns, warps, compute):
    from dev.gemv_simt.q8_address_source import source as measured
    from dev.strategy.components import stage_role
    cfg = dict(variant=5, columns=columns, warps=warps, values=4)
    text = measured(8, compute, [cfg | dict(split=1, hoist=True)])
    for body in Q8_BODIES:
        r = Recipe.make('simt', 'fq', **cfg, split=1, q8_body=body)
        item = spec(Workload(8, 4096, 2048, 1, compute=compute), r)
        generated = source(item)
        assert generated.startswith(text.replace('extern "C" int general_reader_run(', 'static int refresh_q8_run('))
        assert f'return refresh_q8_run(d,a,0,{body});' in generated
        assert digest(item) in generated
        assert stage_role(f'quactlize::execution::simt::q8_address_0_{body}(qkg_call_v1)', 1) == 'producer'
        assert 'd->call.mode!=QKG_DENSE' in generated
    # Shipping launch_v2 and measured_decode remain the actual incumbent.
    baseline = source(spec(Workload(8, 4096, 2048, 1, compute=compute),
                           Recipe.make('simt', 'fq', **cfg, split=1)))
    assert 'measured_decode_launch<8>' in baseline and 'q8_address_' not in baseline


@pytest.mark.parametrize('changes', [dict(split=2), dict(q8_body=2), dict(q8_body=True),
                                      dict(q8_body='4'), dict(values=8), dict(variant=4)])
def test_new_q8_candidate_rejects_unadmitted_domains(changes):
    p = dict(variant=5, columns=8, warps=4, values=4, split=1, q8_body=4) | changes
    with pytest.raises(ValueError):
        spec(Workload(8, 512, 2048, 1), Recipe.make('simt', 'fq', **p))
    with pytest.raises(ValueError):
        spec(Workload(8, 512, 2048, 1, 'indexed', 'bf16', experts=256, topk=8),
             Recipe.make('simt', 'fq', variant=5, columns=8, warps=4, values=4, split=1, q8_body=4))


def test_clock_parser_binds_physical_card_and_both_clocks():
    assert parse('00000000:08:00.0, 1700, 1800\n', '0000:08:00.0', 1700)['cu_mhz'] == 1700
    for text in ('0000:08:00.0, 1000, 1800', '0000:08:00.0, N/A, 1800',
                 '0000:09:00.0, 1700, 1800', '0000:08:00.0, nan, 1800',
                 '0000:08:00.0, 1700, 1200', '0000:08:00.0, 1700, 1800\n0000:09:00.0, 1700, 1800'):
        with pytest.raises(ValueError):
            parse(text, '0000:08:00.0', 1700, 1800)


def test_clock_observation_is_read_only_and_outside_gpu_timing(monkeypatch):
    class Result:
        returncode = 0
        stdout = '0000:08:00.0, 1700, 1800\n'
        stderr = ''
    commands = []
    def fake(command, **kwargs):
        commands.append(command)
        assert kwargs['timeout'] == 20
        return Result()
    monkeypatch.setattr(subprocess, 'run', fake)
    result = observe(Path('/sdk/ppu-smi'), '0000:08:00.0', 1700)
    assert commands == [result['command']]
    assert result['scope'].endswith('NOT_CONTINUOUS_CLOCK_LOCK_PROOF')
    assert '--id=0000:08:00.0' in commands[0]
    assert not any(x.startswith(('--lock', '--reset', '--applications', '--set')) for x in commands[0])


def test_requested_clock_does_not_turn_actual_throttling_into_failure():
    for actual in (1000, 1537, 1687, 1700, 1712):
        value = parse(f'0000:08:00.0, {actual}, 1750', '0000:08:00.0', 1700, 1800,
                      advisory=True)
        assert value['cu_mhz'] == actual and value['memory_mhz'] == 1750
        assert value['clock_policy'] == ADVISORY
        assert value['affects_verdict'] is False
        assert value['frequency_normalization'] is False
    # Prior strict receipts keep their old interpretation, not a retroactive PASS.
    with pytest.raises(ValueError, match='current clocks differ'):
        parse('0000:08:00.0, 1537, 1800', '0000:08:00.0', 1700, 1800)
    for text in ('0000:09:00.0, 1700, 1800', '0000:08:00.0, nan, 1800'):
        with pytest.raises(ValueError):
            parse(text, '0000:08:00.0', 1700, advisory=True)


def test_lock_request_uses_verified_ppu_option_and_only_physical_target(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 20
        return SimpleNamespace(returncode=0, stdout='accepted', stderr='')
    monkeypatch.setattr(subprocess, 'run', execute)
    result = request_lock('/sdk/ppu-smi', '0001:CE:00.0', 1700)
    assert calls == [['/sdk/ppu-smi', '--id=0001:CE:00.0',
                      '--lock-ppu-clocks=1700,1700', '--error-on-warning']]
    assert result['returncode'] == 0 and result['command'] == calls[0]
    for mhz in (float('nan'), float('inf'), 0, -1, 1700.5):
        with pytest.raises(ValueError):
            request_lock('/sdk/ppu-smi', '0001:CE:00.0', mhz)
    assert len(calls) == 1
    with pytest.raises(ValueError):
        request_lock('/sdk/ppu-smi', '', 1700)


@pytest.mark.parametrize('refused', [False, True])
def test_preflight_records_accepted_or_failed_requests_without_reset(tmp_path, monkeypatch, refused):
    from types import SimpleNamespace
    import tools.run_decode_refresh as runner
    calls, requests = [], []
    def lock(fd, flags):
        calls.append(flags)
    def request(binary, pci, mhz):
        assert len(calls) == 2  # All assigned devices acquired before ANY mutation.
        requests.append(pci)
        return dict(returncode=int(refused), stdout='lock test', stderr='', pci=pci)
    monkeypatch.setattr(runner.fcntl, 'flock', lock)
    monkeypatch.setattr(runner, 'request_lock', request)
    monkeypatch.setattr(runner, 'observe', lambda binary, pci, mhz, **kw:
                        parse(f'{pci}, 1537, 1800', pci, mhz, **kw))
    a = SimpleNamespace(output=tmp_path, smi=Path('/unused'), expected_ce_ghz=1.7)
    devices = [dict(device=dict(pci=p)) for p in ('0000:08:00.0', '0001:CE:00.0')]
    if refused:
        with pytest.raises(ValueError, match='lock request failed'):
            runner.lock_clocks(a, devices)
        assert requests == ['0000:08:00.0']
    else:
        runner.lock_clocks(a, devices)
        assert requests == ['0000:08:00.0', '0001:CE:00.0']
    receipt = json.loads(next(tmp_path.glob('clock-lock-*.json')).read_bytes())
    assert receipt['status'] == ('FAIL' if refused else 'REQUESTS_ACCEPTED')
    assert receipt['settings_after_run'] == 'REQUEST_LEFT_IN_PLACE'
    assert receipt['memory_and_power_settings_changed'] is False


def test_clock_retry_reuses_passes_only_without_modifying_originals(plan, tmp_path):
    from dev.strategy.refresh_retry import make_retry, OLD_FILES, NEW_FILES, check_transition
    from tools.run_decode_refresh import authority_for, sealed, saved
    old_harness = dict(files=OLD_FILES | {'dev/strategy/confirm.py': 'unchanged-timer'})
    old_harness['sha256'] = digest(old_harness['files'])
    new_harness = dict(files={k: 'reviewed-clock-edit' for k in OLD_FILES} |
                      {k: 'new-file' for k in NEW_FILES} |
                      {'dev/strategy/confirm.py': 'unchanged-timer'})
    new_harness['sha256'] = digest(new_harness['files'])
    check_transition(old_harness, new_harness)
    for changed in ('dev/strategy/confirm.py', 'tools/run_decode_refresh.py'):
        bad = deepcopy(old_harness if changed.endswith('run_decode_refresh.py') else new_harness)
        bad['files'][changed] = 'unreviewed-change'
        bad['sha256'] = digest(bad['files'])
        with pytest.raises(ValueError):
            check_transition(bad, new_harness) if changed.endswith('run_decode_refresh.py') else \
                check_transition(old_harness, bad)
    assignments = assign(plan, 7)
    run = dict(plan=plan['sha256'], harness=old_harness, assignments=assignments,
               cards=list(map(str, range(1, 8))), cu_mhz=1700, memory_mhz=1800,
               devices=[dict(device=dict(pci=f'0000:0{i}:00.0')) for i in range(7)])
    sealed(tmp_path/'plan.json', plan)
    sealed(tmp_path/'run.json', run)
    run = json.loads((tmp_path/'run.json').read_bytes())
    # Host stand-in tests only orchestration, not numerical/timing admission.
    called = []
    def completed(path, authority, point):
        called.append(point['id'])
        if point == plan['points'][0]:
            return dict(status='FAIL', sha256='old-failure')
        return dict(status='PASS', sha256='old-passing-proof')
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    retry = make_retry(tmp_path, plan, new_harness, completed, authority_for)
    assert len(called) == 1114
    assert retry['points'] == plan['points'][:1]
    assert retry['counts']['points'] == 1
    assert retry['clock_retry']['previous_passed'] == 1113
    assert retry['clock_retry']['raw_results_modified'] is False
    assert retry['clock_retry']['previous_points'][0]['status'] == 'FAIL'
    assert all(p.read_bytes() == data for p, data in before.items())
    assert authority_for(run, 0).get('clock_policy') is None
    assert authority_for(run | dict(clock_policy=ADVISORY), 0)['clock_policy'] == ADVISORY
    # A corrupt old receipt must still fail through the real reader.
    point = plan['points'][0]
    shard = next(i for i, keys in enumerate(assignments) if family(Workload(**point['workload'])) in keys)
    path = tmp_path/('gpu-'+run['cards'][shard])/'points'/(point['id']+'.json')
    sealed(path, dict(status='FAIL', point=point['id'], authority=authority_for(run, shard)))
    data = json.loads(path.read_bytes())
    data['status'] = 'PASS'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash'):
        make_retry(tmp_path, plan, new_harness, saved, authority_for)


def test_aggregate_keeps_missing_and_failures_in_denominator(plan, tmp_path):
    from tools.run_decode_refresh import aggregate, authority_for, sealed
    assignments = assign(plan, 7)
    hardware = dict(device=dict(pci='0000:08:00.0'))
    run = dict(assignments=assignments, cards=list(map(str, range(1, 8))),
               devices=[hardware]*7, sha256='current-run', cu_mhz=1700, memory_mhz=1800)
    p = plan['points'][0]
    shard = next(i for i, keys in enumerate(assignments) if family(Workload(**p['workload'])) in keys)
    target = tmp_path/('gpu-'+run['cards'][shard])/'points'/(p['id']+'.json')
    sealed(target, dict(point=p['id'], status='FAIL', error='guard failed', authority=authority_for(run, shard)))
    result = aggregate(plan, run, tmp_path)
    assert result['expected'] == 1114 and result['completed'] == 0
    assert result['counts'] == dict(FAIL=1, MISSING=1113)
    assert result['status'] == 'INCOMPLETE'
    r = json.loads(target.read_text())
    r['error'] = 'silently changed'
    target.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='hash'):
        aggregate(plan, run, tmp_path)


def test_frozen_binding_rejects_partial_or_changed_inputs(tmp_path):
    from tools.run_decode_refresh import freeze
    p = tmp_path/'inputs.json'
    freeze(p, {'plan': 'a', 'cards': ['1', '2']})
    freeze(p, {'plan': 'a', 'cards': ['1', '2']})
    with pytest.raises(ValueError, match='binding changed'):
        freeze(p, {'plan': 'a', 'cards': ['2', '1']})


def test_real_failed_children_continue_without_looping_and_retry_on_resume(tmp_path, monkeypatch):
    """Fault injection is orchestration evidence, not synthetic GPU admission."""
    import sys
    from types import SimpleNamespace
    import tools.run_decode_refresh as runner
    from dev.strategy.journal import atomic_json
    w = Workload(8, 512, 2048, 1)
    points = [dict(id=str(i), workload=asdict(w)) for i in range(2)]
    fam = family(w)
    run = dict(assignments=[[fam]], cards=['1'], sha256='fault-test',
               devices=[dict(device=dict(pci='fault-test-'+tmp_path.name))],
               cu_mhz=1700, memory_mhz=1800)
    atomic_json(tmp_path/'run.json', run)
    atomic_json(tmp_path/'plan.json', dict(points=points))
    launches = []
    original_popen = subprocess.Popen
    def child(command, **kwargs):
        invocation = command[command.index('--invocation')+1]
        target_dir = tmp_path/'gpu-1/points'
        left = [p for p in points if not (target_dir/(p['id']+'.json')).exists() or
                json.loads((target_dir/(p['id']+'.json')).read_text()).get('invocation') != invocation]
        point = left[0]
        authority = runner.authority_for(run, 0)
        program = (
            "import sys\n"
            "from pathlib import Path\n"
            "from tools.run_decode_refresh import sealed\n"
            f"sealed(Path({str(target_dir/(point['id']+'.json'))!r}), "
            f"dict(status='FAIL',point={point['id']!r},authority={authority!r},"
            f"invocation={invocation!r},error='injected child failure'))\n"
            "raise SystemExit(1)\n")
        launches.append((invocation, point['id']))
        return original_popen([sys.executable, '-c', program], cwd=ROOT, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', child)
    a = SimpleNamespace(output=tmp_path, sdk=Path('/unused'), smi=Path('/unused'),
                        l2_bytes=64, shard=0, invocation='first')
    assert runner.worker(a) == 0
    assert launches == [('first', '0'), ('first', '1')]
    assert runner.worker(a) == 0
    assert len(launches) == 2  # Do not retry a runtime failure forever in the same invocation.
    a.invocation = 'resume'
    assert runner.worker(a) == 0
    assert launches[-2:] == [('resume', '0'), ('resume', '1')]


def test_light_pack_recognizes_refresh_excludes_images_and_reports(tmp_path):
    from tools.pack_simt_split_review import package, result_root
    import tarfile
    results = tmp_path/'results'
    results.mkdir()
    (results/'plan.json').write_text(json.dumps({'schema': 'quactlize.decode-clock-refresh.v1'}))
    (results/'test.so').write_bytes(b'not needed')
    (results/'test.acurep').write_bytes(b'not needed')
    (results/'raw-samples.json').write_text('{"samples_us": [[1,2,3]]}')
    assert result_root(tmp_path) == results
    archive = package(results)
    with tarfile.open(archive) as f:
        names = f.getnames()
        assert 'raw-samples.json' in names
        assert 'test.so' not in names and 'test.acurep' not in names


@pytest.mark.parametrize('compute', ['f16', 'bf16'])
@pytest.mark.parametrize('columns,warps', [(8, 4), (4, 8)])
def test_native_q8_adapter_compile_only(tmp_path, compute, columns, warps):
    sdk = os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk:
        pytest.skip('opt-in real HGCC object compilation, not GPU performance')
    from quactlize.runtime.compiler import FLAGS
    item = spec(Workload(8, 5120, 8192, 1, compute=compute),
                Recipe.make('simt', 'fq', variant=5, columns=columns, warps=warps,
                            values=4, split=1, q8_body=4))
    src, obj = tmp_path/'kernel.cu', tmp_path/'kernel.o'
    src.write_text(source(item))
    command = [str(Path(sdk)/'bin/hgcc'), *FLAGS,
               *['-I'+str(ROOT/p) for p in ('.', 'quactlize/execution', 'quactlize/include',
                                            'third_party/actlize/include', 'third_party/actlize/tools/util/include')],
               '-c', str(src), '-o', str(obj)]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    assert obj.stat().st_size > 0

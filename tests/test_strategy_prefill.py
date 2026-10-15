"""Large-M offline orchestration contracts; no mock timing is device evidence."""
from copy import deepcopy
import ctypes as C
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dev.strategy.prefill_plan import bind_original, family, partition, execution_plan, full_configs
from dev.strategy.resources import inventory, resolve_point
from dev.strategy.components import stage_role
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest
from tools.run_kpack_strategy_prefill import jobs

ROOT = Path(__file__).resolve().parents[1]


def point(w=None):
    w = w or Workload(12, 1024, 5120, 512)
    r = Recipe.make('tc', 'fq', symbol='fqk_tc_q12_l1_a0_tm16_tn64_tk128_wm16_wn16_s2_bc0_ap0_dn16',
        qtype=12, tm=16, tn=64, tk=128, wm=16, wn=16, stages=2, ap=0, dn=16,
        persistent=-1, split=1, grid_mode=0, grid_b=0, dequant_config=-1)
    return dict(id=w.key, workload=asdict(w), original_recipe=asdict(r), candidates=[asdict(r)],
                champion_controls=[], baseline='original')


def test_sharding_owns_complete_weight_families_and_no_candidate_cross_card():
    points = [point(Workload(q, n, 512, m)) for q in (10, 12, 14)
              for n in (256, 512) for m in (128, 512, 4096)]
    shards = [partition(points, i, 3) for i in range(3)]
    assert sorted(p['id'] for s in shards for p in s) == sorted(p['id'] for p in points)
    assert all(set(map(family, a)).isdisjoint(map(family, b)) for i, a in enumerate(shards) for b in shards[i+1:])
    assert all(len(s) == 6 for s in shards)
    for args in ((-1, 2), (2, 2), (0, 0), (0, 257)):
        with pytest.raises(ValueError): partition(points, *args)


@pytest.mark.parametrize('probe_rc', [0, 7])
def test_resource_child_can_lock_device_and_exposes_its_failure(tmp_path, monkeypatch, capsys, probe_rc):
    """Real subprocess/flock regression, not a mocked successful lock."""
    torch = pytest.importorskip('torch')
    import tools.run_kpack_strategy_prefill as mod
    import tools.kpack_bf16_providers as providers
    out = tmp_path/'run'
    (tmp_path/'asight/bin').mkdir(parents=True)
    (tmp_path/'asight/bin/asys').write_text('host-only-profiler-identity')
    lock = tmp_path/('quactlize-components-'+digest('host-only-pci')[:24]+'.lock')
    monkeypatch.setattr(mod, 'Path', lambda value: tmp_path if str(value) == '/tmp' else Path(value))
    monkeypatch.setattr(mod, 'freeze', lambda args: dict(points=[point()], total_points=1, total_families=1))
    monkeypatch.setattr(mod, 'inventory', lambda plan: dict(counts=dict(modules=1, queries=1)))
    monkeypatch.setattr(mod, 'device_probe', lambda sdk: dict(pci='host-only-pci'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(sys, 'argv', ['run', '--output', str(out), '--sdk', str(tmp_path), '--idle-confirmed'])
    original_wait = mod.wait
    probe_status = []

    class ProbeReached(Exception):
        pass

    def check_measurement_lock(*args):
        assert probe_status == [0]
        code = 'import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
        other = subprocess.run([sys.executable, '-c', code, str(lock)], capture_output=True, text=True)
        assert other.returncode != 0 and 'BlockingIOError' in other.stderr
        raise ProbeReached

    monkeypatch.setattr(providers, 'Cublas', check_measurement_lock)

    def run_probe(command, log, *, held=None, phase='setup'):
        assert 'probe_kpack_strategy.py' in command[2]
        code = ('import fcntl,sys; f=open(sys.argv[1],"a"); '
                'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); '
                'print("RESOURCE_CHILD_LOCK_OK",flush=True); '
                'print("TEST_RESOURCE_FAILURE",flush=True) if int(sys.argv[2]) else None; '
                'sys.exit(int(sys.argv[2]))')
        rc = original_wait([sys.executable, '-u', '-c', code, str(lock), str(probe_rc)], log,
                           held=held, phase=phase)
        assert 'RESOURCE_CHILD_LOCK_OK' in log.read_text(), log.read_text()
        assert rc == probe_rc
        probe_status.append(rc)
        return rc

    monkeypatch.setattr(mod, 'wait', run_probe)
    if probe_rc:
        with pytest.raises(ValueError, match='resource phase incomplete.*rc=7.*resources-'):
            mod.main()
        console = capsys.readouterr().out
        assert 'STRATEGY_PREFILL_CHILD_FAILURE' in console and 'TEST_RESOURCE_FAILURE' in console
    else:
        with pytest.raises(ProbeReached):
            mod.main()


def test_failed_child_log_tail_is_bounded_and_decodes_invalid_bytes(tmp_path, capsys):
    from tools.run_kpack_strategy_prefill import wait
    log = tmp_path/'child.log'
    code = 'import sys; sys.stdout.buffer.write(b"OMITTED_PREFIX"+b"x"*20000+b"\\xffFAIL_TAIL"); sys.exit(9)'
    assert wait([sys.executable, '-c', code], log, phase='test') == 9
    console = capsys.readouterr().out
    assert 'FAIL_TAIL' in console and 'OMITTED_PREFIX' not in console and len(console) < 14000
    assert log.stat().st_size > 20000


def test_helper_compile_failure_prints_real_log_tail_and_keeps_artifact(tmp_path, monkeypatch, capsys):
    import dev.strategy.prefill_build as mod
    monkeypatch.setattr(mod, 'identity', lambda sdk: dict(test='HOST_ONLY_FAILURE_INJECTION'))
    code = 'import sys; sys.stdout.buffer.write(b"OMITTED_PREFIX"+b"x"*20000+b"\\xffHGCC_FAILURE_TAIL"); sys.exit(5)'
    monkeypatch.setattr(mod, 'commands', lambda sdk, output: [[sys.executable, '-c', code]])
    output = tmp_path/'attempt'
    with pytest.raises(subprocess.CalledProcessError) as error:
        mod.build(tmp_path, output)
    assert error.value.returncode == 5
    console = capsys.readouterr().out
    assert 'STRATEGY_PREFILL_BUILD_FAILURE unit=prefill_helpers rc=5' in console
    assert 'HGCC_FAILURE_TAIL' in console and 'OMITTED_PREFIX' not in console and len(console) < 14000
    assert (output/'prefill_helpers.log').stat().st_size > 20000
    assert not (output/'manifest.json').exists()


def test_live_original_is_revealed_after_blind_generation_not_injected(tmp_path):
    points = [point(), point(Workload(12, 1024, 2048, 128, 'indexed', 'bf16', experts=256, topk=8)),
              point(Workload(8, 2048, 4096, 128))]
    before = [deepcopy(p['candidates']) for p in points]
    bind_original(points, tmp_path)
    assert before == [p['candidates'] for p in points]
    assert all(p['original_selector']['status'] == 0 for p in points)
    assert all(p['original_recipe'] for p in points)
    assert all(p['champion_controls'] == [] for p in points)
    assert points[-1]['original_recipe']['route'] == 'sf'


def test_full_expansion_keeps_sparse_supported_domain_not_all_experts():
    dense = Workload(12, 512, 2048, 128)
    grouped = Workload(12, 512, 2048, 128, 'indexed', 'bf16', experts=256, topk=8)
    assert full_configs(dense) == [0, 1, 2, 3, 4, 5, 10, 11, 12]
    assert full_configs(grouped) == [4, 5, 10, 11]
    assert full_configs(replace(grouped, qtype=14)) == [4, 5]
    assert full_configs(replace(dense, m=8)) == []


def test_full_helper_reuse_requires_exact_active_ids_not_just_same_expert_count():
    from dev.strategy.components import component_key
    w = Workload(12, 512, 2048, 128, 'indexed', 'bf16', experts=256, topk=8)
    node = dict(name='quactlize::dequant::full_wide', symbol='exact-native-symbol',
                active_expert_ids=[1, 3, 9], grid=[16, 16, 256], block=[128, 1, 1])
    def key(w, node): return component_key(w, None, node, authority='mock', source='source', cache='cold')[0]
    assert key(w, node) == key(replace(w, m=4096), node)
    assert key(w, node) != key(w, dict(node, active_expert_ids=[1, 3, 10]))
    for domain in ([], [1, 1], [3, 1], [256], [1.5]):
        with pytest.raises(ValueError): key(w, dict(node, active_expert_ids=domain))


def probe(points):
    inv = inventory(dict(points=points))
    receipts = {key: dict(status='READY', request=value['request'], occupancy=3,
                device=dict(compute_units=64, ordinal=0)) for key, value in inv['queries'].items()}
    return dict(inventory=inv, receipts=receipts,
                resolved=dict(points=[resolve_point(p, receipts) for p in points]))


def test_prefill_plan_keeps_general_endpoint_and_original_baseline():
    p = point(); p['candidates'].append(asdict(Recipe.make('full', 'full', provider='bf16-default', dequant_config=4)))
    data = execution_plan(probe([p])); got = data['points'][0]
    assert not data['production_changed'] and not data['performance_admitted']
    assert got['arms'][0]['roles'] == ['original']
    tc = [a for a in got['arms'] if a['recipe']['kind'] == 'tc']
    assert all(a['spec']['endpoint'] == 'native' for a in tc)
    all_jobs = jobs(got)
    assert sum(j['roles'] == ['bf16'] for j in all_jobs) == 1
    assert sum(j['roles'] == ['full_dequant'] for j in all_jobs) == 9
    assert all(j.get('recipe', {}).get('kind') != 'full' for j in all_jobs if j['recipe'])
    # Provider runs once; dequant configurations are a separate axis.
    assert len(all_jobs) == len(tc)+10


def test_unavailable_original_is_not_silently_dropped():
    p = point(); frozen = probe([p])
    for row in frozen['receipts'].values(): row['status'] = 'UNAVAILABLE'
    data = execution_plan(frozen)
    assert data['points'] == [] and data['counts']['requested_points'] == 1
    assert data['counts']['blocked_points'] == 1 and data['blocked']


def test_prefill_roles_include_provider_directory_and_no_nested_misclassification():
    names = ['void quactlize::prefill::active_experts(int*)',
             'void quactlize::prefill::gather_bf16(float*, void*)',
             'void computeBlockInfoKernel(int*)',
             'void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<Directory>>(Params)',
             'void quactlize::prefill::finish_bf16(void*)']
    assert [stage_role(n, 1) for n in names] == ['prepare', 'adapter', 'prepare', 'producer', 'postprocess']


def test_grouped_bf16_splitk_inventory_has_its_real_reducer(tmp_path, monkeypatch):
    from dev.strategy.prefill_measure import stages
    from types import SimpleNamespace
    names = [
        'quactlize::prefill::gather_bf16(float const*, __ppu_bfloat16*, int const*, int, long, int, int*)',
        'void quactlize::runtime::grouped_splitk_device_metadata<Shape, Stride>(Params)',
        'void quactlize::moe_directory::build_kernel<16, 256>(Params)',
        'void cutlass::device_kernel<cutlass::gemm::kernel::GroupDirectoryMixedInputKernel<Shape, Metadata>>(Params)',
        'void quactlize::runtime::reduce_bf16_grouped<2>(float const*, cutlass::bfloat16_t*, long)',
        'quactlize::prefill::finish_bf16(__ppu_bfloat16 const*, float*, int const*, int, int)',
    ]
    assert [stage_role(n, 2) for n in names] == ['adapter', 'prepare', 'prepare', 'producer', 'reducer', 'postprocess']
    class InventoryPassed(Exception):
        pass
    def replay(*_):
        raise InventoryPassed
    monkeypatch.setattr('dev.strategy.prefill_measure.replay_extracted', replay)
    w = Workload(12, 256, 512, 128, 'indexed', 'bf16', experts=256, topk=8)
    recipe = Recipe.make('tc', 'fq', split=2)
    capture = SimpleNamespace(descriptions=[dict(name=n) for n in names], groups=list(range(len(names))))
    with pytest.raises(InventoryPassed):
        stages(capture, SimpleNamespace(w=w), recipe, 'host-test', 'host-test', tmp_path, None)
    capture.descriptions[4] = dict(name='unknown_kernel(reduce_bf16_grouped)')
    with pytest.raises(ValueError, match='inventory differs'):
        stages(capture, SimpleNamespace(w=w), recipe, 'host-test', 'host-test', tmp_path, None)


def test_router_resident_maps_have_exact_semantics_and_real_default():
    pytest.importorskip('torch')
    from dev.strategy.prefill_fixture import routing
    for channels in (1, 8):
        w = Workload(12, 256, 512, 128, 'indexed', 'bf16', experts=256, topk=8, channels=channels)
        a = routing(w); b = routing(replace(w, router='real'))
        assert a['sha256'] == b['sha256']
        assert np.array_equal(np.sort(a['order']), np.arange(1024))
        assert np.array_equal(a['sorted_owners'], np.repeat(np.arange(256), a['counts']))
        assert a['offsets'][0] == 0 and a['offsets'][-1] == 1024 and max(a['counts']) <= 128
        assert np.array_equal(a['sorted_source'], a['source'][a['order']])
        assert max(a['source']) < 128*channels
        assert routing(w, 1)['sha256'] != a['sha256']


def test_verified_prefill_active_bound_does_not_shrink_small_m_or_metadata_rings():
    pytest.importorskip('torch')
    from dev.strategy.cold_decode import ring_geometry
    w = Workload(12, 512, 2048, 128, 'indexed', 'bf16', experts=256, topk=8)
    sizes = dict(low=256*512*2048//2, high=0, units=256*512*2048//16)
    before = ring_geometry(w, sizes, 64*2**20)
    after = ring_geometry(w, sizes, 64*2**20, minimum_active=230)
    assert before['minimum_active'] == 8 and before['copies'] > after['copies']
    assert after['active_ring_bytes'] >= 2.25*64*2**20
    for active in (0, 7, 257, 1.5):
        with pytest.raises(ValueError): ring_geometry(w, sizes, 64*2**20, minimum_active=active)
    with pytest.raises(ValueError): ring_geometry(w, sizes, 64*2**20, metadata_only=True, minimum_active=230)


def test_general_tc_never_binds_decode_indexed_and_uses_typed_gpu_maps(monkeypatch):
    pytest.importorskip('torch')
    import dev.strategy.prefill_native as mod
    from quactlize.runtime.native import Resources as Query
    events = []
    w = Workload(12, 256, 512, 128, 'indexed', 'bf16', experts=256, topk=8)
    p = point(w); r = Recipe.from_dict(p['original_recipe'])
    spec = dict(parent={}, endpoint='grouped', compute='bf16')
    request = dict(module=spec, m=1024, n=256, k=512, experts=256, max_rows=128,
                   mapping=w.mapping, group_size=32)
    spec['parent']['qtype'] = 12
    class Probe:
        def __init__(self, *_): self.lib = 'fake'; self.device = dict(ordinal=0, compute_units=64)
        def _bind(self, name, args, result=C.c_int):
            assert 'bind_llama' not in name
            def fn(*values):
                events.append(name)
                if name == 'grouped_query_v4':
                    q = values[2]._obj
                    q.version, q.size, q.workspace_bytes, q.shared_bytes, q.occupancy = 1, C.sizeof(Query), 256, 8192, 3
                if name == 'grouped_prepare_v4':
                    d = values[0]._obj
                    assert d.version == 4 and d.compute_type == 1 and d.metadata_type == 1
                    assert d.device_call.max_rows == 128 and d.device_call.call.offsets_device == 88
                    assert not d.device_call.call.rows_host
                    values[-1]._obj.value = 123
                return 0
            return fn
    monkeypatch.setattr(mod, 'NativeProbe', Probe)
    monkeypatch.setattr(mod, 'Resources', lambda sdk: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(mod, 'Buffer', lambda res, size: SimpleNamespace(ptr=16, size=size))
    b = SimpleNamespace(w=w, rows=1024, sdk=SimpleNamespace(synchronize=lambda _: None),
        r=SimpleNamespace(stream=C.c_void_p(4)), offsets=SimpleNamespace(ptr=88),
        ring=SimpleNamespace(planes=[dict(low=16, high=None, units=32)]*2),
        input=lambda ptr, bf: events.append(('input', bf)) or 0,
        finish=lambda ptr, bf: events.append(('finish', bf)) or 0)
    arm = dict(recipe=asdict(r), spec=spec, binding=dict(launch=dict(algorithm=0, split=1, grid=0)))
    core = mod.TensorCore(b, arm, dict(key='key', sha256='binary'),
        dict(request=request, workspace_bytes=256, shared_bytes=8192, occupancy=3))
    core.launch(); core.close()
    assert events.count('grouped_prepare_v4') == 2
    assert ('input', True) in events and ('finish', True) in events


def test_plan_only_box_is_resumable_and_rejects_repartition(tmp_path):
    env = dict(os.environ, PYTHON=sys.executable, RUN_ROOT=str(tmp_path), PLAN_ONLY='1', SHARD='0', SHARDS='2',
               PPU_SDK=str(tmp_path/'MISSING_SDK'), CUDA_VISIBLE_DEVICES='')
    cmd = ['bash', str(ROOT/'tools/run_kpack_strategy_prefill_box.sh')]
    r = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    assert r.returncode == 0, r.stdout+r.stderr
    runs = [p for p in tmp_path.glob('kpack-rule-prefill.*') if p.is_dir()]
    assert len(runs) == 1
    counts = json.loads((runs[0]/'measurement/counts.json').read_text())
    assert counts['families'] == 43 and counts['points'] == 215 and counts['all_points'] == 430
    assert counts['modules'] <= 256 and not (runs[0]/'measurement/authority.json').exists()
    r = subprocess.run(cmd, cwd=ROOT, env=dict(env, RESUME_DIR=str(runs[0])), text=True, capture_output=True)
    assert r.returncode == 0, r.stdout+r.stderr
    r = subprocess.run(cmd, cwd=ROOT, env=dict(env, RESUME_DIR=str(runs[0]), SHARD='1'), text=True, capture_output=True)
    assert r.returncode != 0 and 'resume shard changed' in r.stdout+r.stderr


def test_external_graph_clone_keeps_torch_pool_until_after_destroy(monkeypatch):
    from dev.strategy.isolated import Captured, KernelParams
    events = []
    class Fn:
        def __init__(self, name): self.name = name
        def __call__(self, *args):
            name = self.name
            if name == 'GraphGetNodes':
                if args[1] is None: args[2]._obj.value = 1
                else: args[1][0] = 77
            elif name == 'GraphGetEdges': args[-1]._obj.value = 0
            elif name == 'GraphNodeGetType': args[1]._obj.value = 0
            elif name == 'GraphKernelNodeGetParams':
                value = args[1]._obj
                value.func = 123; value.grid.x = value.block.x = 1
                value.grid.y = value.grid.z = value.block.y = value.block.z = 1
            elif name == 'FuncGetName': args[0]._obj.value = b'gemm_kernel'
            elif name == 'GraphDestroy': events.append(('destroy', args[0].value))
            return 0
    class Lib:
        def __getattr__(self, name): return Fn(name.removeprefix('hggc'))
    owner = object()
    def unexpected(*_): raise AssertionError('must not begin a second capture')
    rt = SimpleNamespace(lib=Lib(), stream=C.c_void_p(2), sync=lambda: None,
        StreamBeginCapture=unexpected, StreamEndCapture=unexpected, GraphDestroy=Fn('GraphDestroy'))
    capture = Captured(rt, [None], owned_graph=C.c_void_p(91), allocation_owner=owner)
    assert capture.allocation_owner is owner and capture.groups == [[0]]
    assert isinstance(capture.params[0], KernelParams)
    capture.close()
    assert events == [('destroy', 91)] and capture.allocation_owner is None


def test_prefill_control_failure_is_not_swallowed_as_expected_negative():
    pytest.importorskip('torch')
    from dev.strategy.prefill_measure import expect_oracle_failure
    def bad(): raise ValueError('prefill oracle: nonfinite output')
    expect_oracle_failure(bad, 'negative')
    with pytest.raises(ValueError, match='was not detected'):
        expect_oracle_failure(lambda: None, 'zero output')
    def infra(): raise RuntimeError('driver lost')
    with pytest.raises(RuntimeError, match='driver lost'):
        expect_oracle_failure(infra, 'negative')


def test_prefill_summary_composes_full_provider_once_with_separate_dequant(tmp_path):
    from dev.strategy.journal import ArmJournal, atomic_json
    from dev.strategy.components import component_key, statistics_checked
    from tools.run_kpack_strategy_prefill import summarize
    p = point()
    full = asdict(Recipe.make('full', 'full', provider='bf16-default', dequant_config=4))
    p['candidates'] = [full]
    plan = execution_plan(probe([p])); p = plan['points'][0]
    journal = ArmJournal(tmp_path/'arms', dict(test='mock-not-PPU'), 'a'*32)
    for j in jobs(p):
        role = j['roles'][0]
        value = 9. if role == 'original' else 2. if role == 'bf16' else 1.+j['config']
        node = dict(name='gemm' if role in ('original', 'bf16') else 'dequant_full', symbol=j['id'])
        cid, ident = component_key(Workload(**p['workload']), None, node,
                                   authority=journal.authority, source='mock', cache='mock')
        row = statistics_checked([[value]*3 for _ in range(3)])
        row.update(identity=ident, component_id=cid, kind=stage_role(node['name'], 1))
        row['record_sha256'] = digest(row)
        atomic_json(tmp_path/'components'/(cid+'.json'), row)
        journal.start(p, j)
        journal.finish(p, j, 'PASS', data=dict(correctness='PASS', components=[cid]))
    result = summarize(tmp_path, plan, journal)[0]
    assert result['selection']['costs_us']['original'] == 9.
    rule = next(a for a, r in result['selection']['roles'].items() if 'rule' in r)
    assert result['selection']['costs_us'][rule] == 3.
    assert len(result['components'][rule]) == 2
    assert result['best_helpers']['full_dequant'] == 0


@pytest.mark.parametrize('q', [10, 11, 12, 13, 14])
def test_full_expansion_oracle_checks_selected_bits_and_inactive_experts(q):
    pytest.importorskip('torch')
    from dev.strategy.prefill_full import Full
    from dev.bf16_compute.fixture import raw_weight, bf16_bits
    n, k, e = 256, 512, 4
    contents = np.full((e, n, k), 0x7f7f, dtype='<u2')
    for expert in (1, 3):
        _, gold = raw_weight(q, n, k, 710+expert*31+q*1009)
        contents[expert] = bf16_bits(gold)
    base, error = 8192, 100000000
    def download(ptr, count):
        if ptr == error: return b'\0'*4
        return contents.tobytes()[ptr-base:ptr-base+count]
    full = Full.__new__(Full)
    full.b = SimpleNamespace(w=SimpleNamespace(qtype=q, n=n, k=k, experts=e),
        route=dict(active=np.array([1, 3])), error=SimpleNamespace(ptr=error),
        rt=SimpleNamespace(sync=lambda: None), sdk=SimpleNamespace(download=download))
    full.weights = [SimpleNamespace(data_ptr=lambda: base)]
    full.guards = lambda: None
    assert full.check_weights()['bad'] == 0
    contents[0, 0, 0] = 0
    with pytest.raises(ValueError, match='inactive-write guard'): full.check_weights()
    contents[0, 0, 0] = 0x7f7f
    contents[1, 0, 0] ^= 1
    with pytest.raises(ValueError, match='oracle'): full.check_weights()


def test_full_capture_clones_inside_torch_capture_and_retains_owner(monkeypatch):
    pytest.importorskip('torch')
    from contextlib import contextmanager
    import dev.strategy.prefill_full as f
    events = []; owner = object()
    @contextmanager
    def graph(obj, stream):
        assert obj is owner and stream == 'stream'
        events.append('begin')
        yield
        events.append('end')
    def clone(stream, out):
        assert events[-1] == 'call1'
        out._obj.value = 123; events.append('clone'); return 0
    def captured(rt, calls, *, owned_graph, allocation_owner):
        assert events[-1] == 'end' and owned_graph.value == 123 and allocation_owner is owner
        assert len(calls) == 2
        return 'capture-with-owner'
    full = f.Full.__new__(f.Full)
    full.torch = SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=lambda: owner, graph=graph))
    full.stream = 'stream'; full.copies = 2
    full.b = SimpleNamespace(w=SimpleNamespace(experts=256),
                             rt=SimpleNamespace(sync=lambda: None), r=SimpleNamespace(stream=1),
                             helpers=SimpleNamespace(clone=clone))
    full.launch = lambda copy: events.append('call'+str(copy)) or 0
    monkeypatch.setattr(f, 'Captured', captured)
    assert full.capture() == 'capture-with-owner'
    assert events == ['begin', 'call0', 'call1', 'clone', 'end']


def test_native_helper_objects_compile_without_touching_production(tmp_path):
    sdk = os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk: pytest.skip('opt-in real HGCC object compile, no GPU')
    from dev.strategy.prefill_build import commands
    for command in commands(Path(sdk), tmp_path):
        subprocess.run(command, check=True, capture_output=True, text=True)
    isa = subprocess.check_output([str(Path(sdk)/'bin/hgobjdump'), '--dump-isa',
                                   str(tmp_path/'prefill_helpers.o')], text=True)
    assert 'gather_bf16' in isa and 'finish_bf16' in isa and 'convert_rows' in isa
    assert 'bf16.f32' in isa or 'v.pcnvt.bf16x2' in isa

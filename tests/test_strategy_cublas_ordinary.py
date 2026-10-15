"""Host-only no-Graph measurement contracts; these do not admit PPU timing."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from dev.strategy.cublas_trace import Ranges, label, parse_trace, profile_command, summarize_trace, wait_process
from dev.strategy.prefill_full import Full


def fixture(path, *, reducer=True):
    con = sqlite3.connect(path)
    con.executescript('''
    CREATE TABLE StringIds (id INTEGER, value TEXT);
    CREATE TABLE HGTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER,
                             globalTid INTEGER, eventType INTEGER);
    CREATE TABLE HGPTI_ACTIVITY_KIND_KERNEL (
        start INTEGER, end INTEGER, demangledName INTEGER, mangledName INTEGER,
        deviceId INTEGER, contextId INTEGER, streamId INTEGER,
        gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER,
        staticSharedMemory INTEGER, dynamicSharedMemory INTEGER);
    ''')
    con.executemany('INSERT INTO StringIds VALUES (?,?)', [
        (1, 'gather_bf16'), (2, 'some_bf16_gemm'), (3, 'splitk_reduction'), (4, 'finish_bf16')])
    rows = []
    for r in range(3):
        for sample in range(3):
            for phase, kind in (('input', 1), ('provider', 2), ('output', 4)):
                text = label(r, sample, phase)
                start = (len(rows)+1)*100000
                con.execute('INSERT INTO HGTX_EVENTS VALUES (?,?,?,?,?,?)',
                            (start, start+90000, text, None, 7<<24, 59))
                con.execute('INSERT INTO HGPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (start+1000, start+6000, kind, kind, 0, 1, 10, 32, 1, 1, 256, 1, 1, 0, 1024))
                if reducer and phase == 'provider':
                    con.execute('INSERT INTO HGPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                (start+7000, start+9000, 3, 3, 0, 1, 10, 16, 1, 1, 128, 1, 1, 0, 0))
                rows.append(dict(label=text, phase=phase, round=r, sample=sample, copy=sample))
    # Warmup/JIT activities exist but must never enter a timed result.
    con.execute('INSERT INTO HGPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (10, 99999, 2, 2, 0, 1, 10, 32, 1, 1, 256, 1, 1, 0, 1024))
    con.commit(); con.close()
    return rows


def test_each_gpu_kernel_is_timed_not_the_range_or_whole_call(tmp_path):
    path = tmp_path/'trace.sqlite'; expected = fixture(path)
    result = parse_trace(path, expected)
    summary = summarize_trace(result)
    assert result['status'] == 'PASS' and not result['cuda_graph']
    assert len(result['ranges']) == 27
    assert summary['phase_kernel_sum_median_us'] == dict(input=5., provider=7., output=5.)
    assert sorted((r['role'], r['median_us'], r['instances']) for r in summary['kernels']) == [
        ('adapter', 5., 9), ('postprocess', 5., 9), ('producer', 5., 9), ('reducer', 2., 9)]
    assert all(r['host_range_us'] == 90 for r in result['ranges'])
    assert not summary['unknown_kernels']


@pytest.mark.parametrize('sql,message', [
    ('DELETE FROM HGPTI_ACTIVITY_KIND_KERNEL', 'empty device'),
    ('DELETE FROM HGTX_EVENTS WHERE rowid=1', 'coverage'),
    ('INSERT INTO HGTX_EVENTS SELECT * FROM HGTX_EVENTS WHERE rowid=1', 'coverage'),
    ('UPDATE HGTX_EVENTS SET end=NULL WHERE rowid=1', 'incomplete'),
    ('UPDATE HGTX_EVENTS SET globalTid=123 WHERE rowid=1', 'multiple host'),
    ('UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET start=99999 WHERE rowid=1', 'crosses'),
    ('UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET streamId=11 WHERE rowid=1', 'stream'),
    ('UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET deviceId=1 WHERE rowid=1', 'device'),
    ('UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET contextId=2 WHERE rowid=1', 'context'),
    ('UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET demangledName=999 WHERE rowid=1', 'resolved name'),
    ('DELETE FROM HGPTI_ACTIVITY_KIND_KERNEL WHERE rowid=1', 'no GPU kernel'),
    ('DROP TABLE HGPTI_ACTIVITY_KIND_KERNEL', 'schema missing'),
])
def test_incomplete_or_foreign_activity_is_not_admitted(tmp_path, sql, message):
    path = tmp_path/'trace.sqlite'; expected = fixture(path)
    with sqlite3.connect(path) as con: con.execute(sql)
    with pytest.raises(ValueError, match=message): parse_trace(path, expected)


def test_string_table_ranges_and_varying_provider_dispatch_remain_visible(tmp_path):
    path = tmp_path/'trace.sqlite'; expected = fixture(path)
    with sqlite3.connect(path) as con:
        con.execute('INSERT INTO StringIds SELECT 500,text FROM HGTX_EVENTS WHERE rowid=1')
        con.execute('UPDATE HGTX_EVENTS SET text=NULL,textId=500 WHERE rowid=1')
        con.execute('DELETE FROM HGPTI_ACTIVITY_KIND_KERNEL WHERE rowid=3')
        con.execute("INSERT INTO StringIds VALUES (800,'provider_internal_unknown')")
        con.execute('UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET demangledName=800 WHERE rowid=6')
    result = parse_trace(path, expected)
    summary = summarize_trace(result)
    assert summary['unknown_kernels'] == ['provider_internal_unknown']
    assert sum(len(r['kernels']) for r in result['ranges']) == 35


def test_cublas_capture_rejected_before_any_cuda_api():
    full = Full.__new__(Full); full.b = SimpleNamespace(w=SimpleNamespace(experts=1))
    with pytest.raises(ValueError, match='not CUDA Graph'): full.capture()


def test_range_finishes_device_work_before_pop_and_pairs_on_exception():
    events = []
    ranges = Ranges.__new__(Ranges)
    ranges.push = lambda x: events.append(('push', x))
    ranges.pop = lambda: events.append('pop')
    sync = lambda: events.append('sync')
    with ranges.range('test', sync): events.append('launch')
    assert events == ['sync', ('push', b'test'), 'launch', 'sync', 'pop']
    events.clear()
    with pytest.raises(RuntimeError):
        with ranges.range('bad', sync): raise RuntimeError('driver error')
    assert events == ['sync', ('push', b'bad'), 'pop']


def test_profile_has_device_activity_no_graph_and_no_overwrite(tmp_path):
    cmd = profile_command('/sdk/asys', tmp_path/'trace.asysrep', ['python3', 'worker.py'])
    assert cmd[cmd.index('--trace')+1] == 'hggc,hgtx'
    assert cmd[cmd.index('--hggc-trace-set')+1] == 'kernel-activity'
    assert cmd[cmd.index('--force-overwrite')+1] == 'false'
    assert cmd[-2:] == ['python3', 'worker.py']


def test_ordinary_worker_never_uses_capture_and_retains_numeric_negatives(monkeypatch):
    import dev.strategy.cublas_ordinary as m
    events, controls = [], []
    phase = dict(repeat=0, zero=False, written=False, bad_weight=False, copy=0, poison=False)
    def update(repeat, **kw):
        phase.update(repeat=repeat, zero=kw.get('zero', False), written=False)
        events.append(('update', repeat, kw))
    def setup(config): phase.update(bad_weight=False); events.append('dequant')
    def launch(copy=0):
        phase.update(written=True, copy=copy, poison=False); events.append('gemm'); return 0
    def check(**kw):
        if kw.get('zero'): assert phase['zero']; return {'pass': True}
        if not phase['written'] or phase['zero'] or (phase['bad_weight'] and phase['copy']==2) or phase['poison']:
            raise ValueError('prefill oracle: injected mismatch')
        return {'pass': True}
    def fill(ptr, *_):
        if ptr == 42: phase['bad_weight'] = True
        else: phase['poison'] = True
    def sampled(fn, count):
        assert count==1
        controls.append(len(events)); fn(); return [20.]
    b = SimpleNamespace(w=SimpleNamespace(operation='dense', experts=1), update=update, check=check,
        rt=SimpleNamespace(sync=lambda: None), r=SimpleNamespace(fill=fill, samples=sampled),
        ring=SimpleNamespace(buffers=[], receipt={'unchanged': True}),
        input=lambda *args: 0, finish=lambda *args: 0)
    tensor = lambda ptr: SimpleNamespace(data_ptr=lambda: ptr, numel=lambda: 16)
    core = SimpleNamespace(copies=3, weights=[tensor(40),tensor(41),tensor(42)], a=tensor(5), out=tensor(9),
        setup=setup, launch=launch, matmul=launch, update=lambda: None, guards=lambda: None,
        check_weights=lambda i: {'bad':0}, receipt={'provider':'cuBLAS'})
    monkeypatch.setattr(m, 'loaded_images', lambda: {'blas.so':'sha'})
    r = m.measure(b, core)
    assert r['correctness']=='PASS' and not r['cuda_graph'] and r['warmup_calls']==12
    assert r['omitted_producer']=='DETECTED' and r['pointer_negatives']==['BF16_LAST_COPY_ZERO']
    assert len(r['measured'])==27 and len(controls)==28  # one untimed event warmup
    assert all(item['event_span_us']==20. for item in r['measured'])
    assert r['profiler'] is False and r['annotations'] is None
    # No fixture/expansion/warmup work after the first measured range.
    assert all(e=='gemm' for e in events[controls[1]:])
    assert any(e[0]=='update' and e[2].get('amplitude')==2**20 for e in events if isinstance(e,tuple))


def test_profiled_components_remain_distinct_from_event_evidence(tmp_path):
    from dev.strategy.cublas_components import components
    from dev.strategy.components import statistics_checked, recommendation
    from dev.strategy.journal import atomic_json
    from tools.run_kpack_strategy import record_load
    from quactlize.dispatch.table import Workload
    from quactlize.runtime.tuning import digest
    path = tmp_path/'trace.sqlite'; expected = fixture(path)
    trace = parse_trace(path, expected)
    rows = components(trace, Workload(11, 1024, 5120, 128), {'sha256':'authority'}, 'source')
    assert len(rows)==4
    for i, row in enumerate(rows):
        dest = tmp_path/f'{i}.json'; atomic_json(dest, row)
        assert record_load(dest, row['identity']) == row
        assert row['scope']=='PROFILED_SINGLE_KERNEL' and row['timing_method']=='ASYS_GPU_ACTIVITY'
    records = {r['component_id']:r for r in rows}
    records['baseline'] = statistics_checked([[20.]*3 for _ in range(3)])
    result = recommendation({'original':['baseline'], 'full':[r['component_id'] for r in rows]}, records)
    assert result['costs_us']['full']==17. and 'MIXED_METHODS' in result['scope']
    bad = copy.deepcopy(rows[0]); bad['identity']['cuda_graph']=True
    bad['record_sha256']=digest({k:v for k,v in bad.items() if k!='record_sha256'})
    atomic_json(tmp_path/'bad.json', bad)
    with pytest.raises(ValueError,match='no-Graph trace'): record_load(tmp_path/'bad.json', bad['identity'])


def test_variable_inventory_is_kept_raw_but_not_misrepresented_as_one_recipe(tmp_path):
    from dev.strategy.cublas_components import components
    from quactlize.dispatch.table import Workload
    path = tmp_path/'trace.sqlite'; expected = fixture(path)
    with sqlite3.connect(path) as con:
        con.execute('DELETE FROM HGPTI_ACTIVITY_KIND_KERNEL WHERE rowid=3')
    trace = parse_trace(path, expected)  # Profiling itself succeeded.
    with pytest.raises(ValueError,match='multiple launch inventories'):
        components(trace, Workload(11,1024,5120,128), 'auth', 'source')


def test_campaign_dense_provider_uses_no_graph_child_and_inherits_lock():
    import inspect
    from tools.run_kpack_strategy_prefill import child, main
    source = inspect.getsource(child)
    assert "role == 'bf16' and w.operation == 'dense'" in source
    assert 'measure_cublas(args, w, authority, held)' in source
    assert 'os.dup(args.lock_fd)' in source
    assert "'--lock-fd', str(held.fileno())" in inspect.getsource(main)


def test_owned_process_timeout_does_not_leak_its_child(tmp_path):
    import subprocess
    import sys
    marker = tmp_path/'must-not-appear'
    # The grandchild would leave a marker after the parent timeout if only
    # the profiler PID, rather than its owned group, were killed.
    code = ('import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'
            +repr(f'import time;time.sleep(1);open({str(marker)!r},"x").close()')
            +']);time.sleep(20)')
    with (tmp_path/'lock').open('a') as held:
        result = wait_process([sys.executable,'-c',code],tmp_path/'wait.log',held,.2)
    assert result['timeout'] and result['rc']<0
    subprocess.run([sys.executable,'-c','import time;time.sleep(1.2)'],check=True)
    assert not marker.exists()


def test_event_worker_failure_cannot_publish_success(tmp_path, monkeypatch):
    import tools.probe_kpack_cublas_ordinary as probe
    from dev.strategy.journal import atomic_json
    from quactlize.runtime.tuning import digest
    output = tmp_path/'output'; output.mkdir()
    authority = dict(profiler=False)
    calls = []
    def wait(cmd, log, held, timeout):
        calls.append(cmd); Path(log).write_text('mock child\n')
        arm = cmd[cmd.index('--worker')+1]
        atomic_json(output/arm/'result.json', dict(status='PASS', authority_sha256=digest(authority),
            data=dict(correctness='PASS')))
        return dict(rc=23, timeout=False, seconds=1.)
    monkeypatch.setattr(probe,'wait_process',wait)
    with (tmp_path/'lock').open('a') as held:
        with pytest.raises(ValueError,match='events failed'):
            probe.run_events('/sdk',output,authority,held)
    assert len(calls)==1 and not (output/'summary.json').exists()


def test_reuse_builds_queries_l2_from_actual_reused_directory(monkeypatch, tmp_path):
    import tools.probe_kpack_cublas_ordinary as probe
    import inspect
    assert 'l2_authority(SimpleNamespace(output=builds, l2_bytes=args.l2_bytes))' in inspect.getsource(probe.run)


def test_complete_event_handoff_has_no_profiler_process(tmp_path, monkeypatch):
    import tools.probe_kpack_cublas_ordinary as probe
    from dev.strategy.journal import atomic_json
    from quactlize.runtime.tuning import digest
    output = tmp_path/'output'; output.mkdir()
    authority = dict(profiler=False)
    calls=[]
    def wait(cmd, log, held, timeout):
        calls.append(cmd)
        Path(log).write_text('mock only\n')
        arm=cmd[cmd.index('--worker')+1]
        atomic_json(output/arm/'result.json', dict(status='PASS', authority_sha256=digest(authority),data={'test':True}))
        return dict(rc=0, timeout=False, seconds=1.)
    monkeypatch.setattr(probe,'wait_process',wait)
    monkeypatch.setattr(probe,'summarize',lambda data: dict(status='PASS', received=data))
    with (tmp_path/'lock').open('a') as held:
        result=probe.run_events('/sdk',output,authority,held)
    assert result['status']=='PASS' and result['received']=={'test':True}
    assert len(calls)==1 and calls[0][-2:]==['--worker','events']
    assert (output/'summary.json').is_file() and not (output/'ordinary.sqlite').exists()


def test_box_wrapper_is_non_destructive_and_requires_idle_device():
    import subprocess
    path=Path(__file__).resolve().parents[1]/'tools/run_kpack_cublas_ordinary_box.sh'
    subprocess.run(['bash','-n',str(path)],check=True)
    text=path.read_text()
    assert 'rm ' not in text and 'IDLE_CONFIRMED' in text and 'CUDA_VISIBLE_DEVICES:?' in text
    assert 'REUSE_BUILDS' in text and 'probe.results' not in text

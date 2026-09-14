import copy
import ctypes as C
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools import run_kpack_decode_reducer as bench


def test_plan_only_reuses_selected_or_confirmed_small_m_reducers():
    rows = bench.cases()
    assert len(rows) == 138
    assert sum(not r['compact'] for r in rows) == 122
    assert sum(r['compact'] for r in rows) == 16
    assert len({tuple(r[k] for k in ('compact', 'm', 'n', 'split')) for r in rows}) == len(rows)
    assert all(r['references'] and r['split'] in (2, 4, 8) for r in rows)
    assert all(1 <= r['m'] <= (64 if r['compact'] else 8) for r in rows)
    assert max(2 * r['m'] * r['n'] * (4*r['split'] + 2) for r in rows) < 32 * 1024**2
    assert all('q' not in r and 'k' not in r for r in rows)


def record(w=None):
    w = w or bench.cases()[0]
    size = w['m'] * w['n'] * w['split'] * 4
    return dict(status='PASS', workload=w, scope=bench.SCOPE, samples_us=[2.]*15,
        median_us=2., round_medians_us=[2.]*3, device={'l2_bytes':64*1024**2},
        partial_bytes=size, copies=2, calls_per_graph=64,
        resident_bytes=2*(size+w['m']*w['n']*2),
        cache='REUSED_TWO_BUFFERS_NO_FLUSH_HIT_RATE_NOT_MEASURED',
        partial_layout='FP32_S_M_N', output_dtype='FP16', addition_order='INCREASING_S',
        raw_bad=0, negative_bad=100, guards='PASS', changed_partial_graph='PASS',
        fast_path=bool(w['compact'] or w['m']==1),
        producer_timed=False, dequant_timed=False, scatter_timed=False, production_changed=False)


@pytest.mark.parametrize('key,value', [
    ('scope','FULL_OUTPUT'), ('producer_timed',True), ('dequant_timed',True),
    ('scatter_timed',True), ('production_changed',True), ('output_dtype','FP32'),
    ('partial_layout','FP32_M_N_S'), ('addition_order','UNORDERED'),
    ('samples_us',[1.]*14), ('samples_us',[float('nan')]*15), ('samples_us',[True]*15),
    ('median_us',3.), ('round_medians_us',[3.]*3), ('copies',1), ('calls_per_graph',32),
    ('partial_bytes',8), ('resident_bytes',8), ('cache','ROTATING_HBM'),
    ('raw_bad',1), ('negative_bad',0), ('guards','SKIPPED'), ('changed_partial_graph','MISSING'),
    ('fast_path',False), ('device',{'l2_bytes':16}),
])
def test_rejects_wrong_scope_or_incomplete_proof(key, value):
    r = record()
    bench.validate(r, r['workload'])
    broken = copy.deepcopy(r)
    broken[key] = value
    with pytest.raises(ValueError):
        bench.validate(broken, r['workload'])


@pytest.mark.parametrize('compact,m,fast', [(0,1,True), (0,2,False), (0,8,False), (1,8,True), (1,64,True)])
def test_fast_path_contract(compact, m, fast):
    w = dict(compact=compact,m=m,n=512,split=4,id='test',references=['unit-test'])
    r = record(w)
    assert r['fast_path'] == fast
    bench.validate(r,w)


class HostSDK:
    """CPU test double for ordering/oracle controls, not device validation."""
    def __init__(self):
        self.lib = self

    def hggcMemcpy(self, dst, src, size, kind):
        assert kind == 1
        C.memmove(dst, src, size)
        return 0

    def synchronize(self, stream):
        pass

    def download(self, pointer, size):
        return C.string_at(pointer, size)


class HostResources:
    closed = 0

    def __init__(self, sdk):
        self.sdk, self.stream, self.buffers = sdk, C.c_void_p(1), []

    def alloc(self, size):
        buf = C.create_string_buffer(size + 256)
        self.buffers.append(buf)
        return (C.addressof(buf) + 255) // 256 * 256

    def fill(self, pointer, byte, size):
        C.memset(pointer, byte, size)

    def samples(self, fn, count):
        for _ in range(count):
            assert fn() == 0
        return [128.] * count

    def close(self):
        self.buffers.clear()
        type(self).closed += 1


class HostLibrary:
    def __init__(self, stale=False):
        self.args, self.stale, self.written = {}, stale, set()

    def prepare(self, compact, m, n, split, pointer, size, out, result):
        key = len(self.args) + 1
        self.args[key] = compact,m,n,split,pointer,size,out
        result._obj.value = key
        return 0

    def run(self, h, stream):
        if self.stale and h.value in self.written:
            return 0
        self.written.add(h.value)
        _,m,n,s,pointer,size,out = self.args[h.value]
        data = np.frombuffer(C.string_at(pointer,size),dtype='<f4').reshape(s,m*n)
        total = np.zeros(m*n,dtype='<f4')
        for row in data:
            np.add(total,row,out=total)
        value = total.astype('<f2')
        C.memmove(out,value.ctypes.data,value.nbytes)
        return 0

    def fast(self, h):
        compact,m,*_ = self.args[h.value]
        return int(compact or m == 1)

    def destroy(self, h):
        self.args.pop(h.value)


class HostGraph:
    def __init__(self, sdk, stream, launch, repeats):
        assert repeats == 32
        self.launch = launch

    def __call__(self):
        return self.launch()

    def close(self):
        pass


@pytest.mark.parametrize('compact,m', [(0,1),(0,7),(1,16)])
def test_measurement_driver_controls_on_cpu(monkeypatch, compact, m):
    monkeypatch.setattr(bench,'Resources',HostResources)
    monkeypatch.setattr(bench,'Replay',HostGraph)
    w = dict(compact=compact,m=m,n=256,split=4,id='cpu-control',references=['unit-test'])
    lib = HostLibrary()
    got = bench.measure(HostSDK(),lib,{'l2_bytes':64*1024**2},w)
    assert got['median_us'] == 2. and got['negative_bad'] > 0 and not lib.args
    stale = HostLibrary(stale=True)
    with pytest.raises(ValueError,match='negative=True'):
        bench.measure(HostSDK(),stale,{'l2_bytes':64*1024**2},w)
    assert not stale.args


def test_shell_and_no_build_or_full_dequant():
    path = Path('tools/run_kpack_decode_reducer_ppu_box.sh')
    subprocess.run(['bash','-n',str(path)],check=True)
    run = subprocess.run(['bash','-c',f'PPU_SDK=/missing-sdk bash {path}; printf "PARENT_ALIVE\\n"'],
                         capture_output=True,text=True)
    assert run.returncode == 0 and 'PARENT_ALIVE' in run.stdout
    src = path.read_text()
    assert src.index('PYTHON=$(command') < src.index('source "$SDK/envsetup.sh"')
    assert 'build_kpack' not in src and 'run_kpack_dequant_gate.py' not in src

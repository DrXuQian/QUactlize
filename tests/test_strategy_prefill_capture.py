"""Host contracts only: these are not PPU capture/performance admission."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from dev.strategy.prefill_full import Full, ResidentWorkspace
from tools.kpack_bf16_providers import Cublas
from tools.probe_kpack_prefill_capture import arm_pass, package, summarize, wait_arm

ROOT = Path(__file__).resolve().parents[1]


class Function:
    def __init__(self, rc=0): self.calls, self.rc = [], rc
    def __call__(self, *args): self.calls.append(args); return self.rc


@pytest.mark.parametrize('entry', ['cublasSetWorkspace_v2', 'cublasSetWorkspace'])
def test_explicit_workspace_supports_installed_abi_spelling(entry):
    c = Cublas.__new__(Cublas); c.handle = C.c_void_p(9)
    fn = Function(); c.lib = SimpleNamespace(**{entry: fn})
    row = c.set_workspace(0x2000, 32*1024*1024)
    assert row['entry'] == entry and row['mode'] == 'CALLER_OWNED'
    assert fn.argtypes == [C.c_void_p, C.c_void_p, C.c_size_t]
    assert fn.calls == [(c.handle, 0x2000, 32*1024*1024)]
    assert row['algorithm'] == 'CUBLAS_GEMM_DEFAULT' and not row['allocation_timed']


@pytest.mark.parametrize('pointer,size', [(0, 1024), (257, 1024), (-256, 1024),
    (256., 1024), (True, 1024), (256, 0), (256, -1), (256, True), (256, 1.)])
def test_workspace_rejects_bad_alignment_and_sizes(pointer, size):
    c = Cublas.__new__(Cublas)
    with pytest.raises(ValueError): c.set_workspace(pointer, size)


def test_workspace_does_not_silently_fallback_on_error_or_missing_api():
    c = Cublas.__new__(Cublas); c.handle = C.c_void_p(9)
    preferred, fallback = Function(801), Function()
    c.lib = SimpleNamespace(cublasSetWorkspace_v2=preferred, cublasSetWorkspace=fallback)
    with pytest.raises(RuntimeError, match='status=801'): c.set_workspace(256, 1024)
    assert len(preferred.calls) == 1 and not fallback.calls
    c.lib = SimpleNamespace()
    with pytest.raises(ValueError, match='no workspace fallback'): c.set_workspace(256, 1024)


@pytest.mark.parametrize('base', [0x1000, 0x1001, 0x1080, 0x10ff])
def test_workspace_both_guards_stay_inside_the_owned_allocation(base):
    calls = []
    mem = bytearray(4096+768)
    def alloc(size): assert size == len(mem); return base
    def fill(ptr, value, size): mem[ptr-base:ptr-base+size] = bytes([value])*size
    def download(ptr, size): return bytes(mem[ptr-base:ptr-base+size])
    r = SimpleNamespace(alloc=alloc, fill=fill, stream='init-stream',
        sdk=SimpleNamespace(synchronize=lambda s: calls.append(s), download=download))
    w = ResidentWorkspace(r, 4096)
    assert w.ptr % 256 == 0 and w.ptr-256 >= base and w.ptr+4096+256 <= base+len(mem)
    assert calls == ['init-stream']
    fill(w.ptr, 42, w.size)  # cuBLAS may write every byte of its workspace.
    w.guard()
    fill(w.ptr+w.size, 0, 1)
    with pytest.raises(ValueError, match='redzone'): w.guard()


@pytest.mark.parametrize('size', [-1, True, 1.5, '1024'])
def test_full_rejects_invalid_workspace_before_allocating(size):
    with pytest.raises(ValueError, match='nonnegative'): Full(None, None, None, None, workspace_bytes=size)


def test_workspace_lifetime_extends_past_handle_destruction():
    events = []
    f = Full.__new__(Full)
    f.b = SimpleNamespace(rt=SimpleNamespace(sync=lambda: events.append('sync')))
    f.provider = SimpleNamespace(close=lambda: events.append('destroy-cublas'))
    f.r = SimpleNamespace(close=lambda: events.append('free-workspace'))
    f.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append('empty-cache')))
    f.weights, f.tensors, f.backs = [], [], []
    f.close()
    assert events == ['sync', 'destroy-cublas', 'free-workspace', 'empty-cache']


def test_capture_api_failure_and_node_inventory_are_not_conflated(tmp_path):
    code = r'''
#include <cassert>
#include <cstdlib>
using hggcStream_t = void*; using hggcGraph_t = void*; using hggcGraphNode_t = void*;
enum E {hggcSuccess=0, hggcErrorInvalidValue=1, hggcErrorNotSupported=801};
enum hggcStreamCaptureStatus {hggcStreamCaptureStatusNone=0, hggcStreamCaptureStatusActive=1};
enum hggcGraphNodeType {hggcGraphNodeTypeKernel=0};
static int mode, clones;
int hggcStreamGetCaptureInfo(void*, hggcStreamCaptureStatus* s, void*, void** g, void*, void*, void*) {
    *s = mode==2 ? hggcStreamCaptureStatusNone : hggcStreamCaptureStatusActive;
    *g = (void*)17; return mode==1 ? 801 : 0;
}
int hggcGraphGetNodes(void* g, void** nodes, size_t* count) {
    assert(g==(void*)17);
    if (mode==3 || (mode==6 && nodes)) return 801;
    if (!nodes) { *count = mode==8 ? 0 : 2; return 0; }
    nodes[0]=(void*)1; nodes[1]=(void*)2; if (mode==13) *count=1; return 0;
}
int hggcGraphNodeGetType(void*, hggcGraphNodeType* t) {
    if (mode==7) return 801;
    *t=(hggcGraphNodeType)(mode==4 ? 10 : mode==9 ? 5 : mode==10 ? 11 : mode==11 ? 13 : 0);
    return 0;
}
int hggcGraphClone(void** clone, void* graph) {
    assert(graph==(void*)17); ++clones; if (mode==5) return 801; *clone=(void*)23; return 0;
}
#include "dev/strategy/prefill_capture.hpp"
int main(int argc, char** argv) {
    mode=std::atoi(argv[1]); void* out=(void*)99;
    int rc=strategy_prefill::clone_capture(nullptr, mode==12 ? nullptr : &out);
    int expected=mode==0 ? 0 : (mode==2 || mode==8 || mode==12 || mode==13) ? 1 : 801;
    assert(rc==expected); assert(clones==(mode==0 || mode==5 ? 1 : 0));
    if (mode!=12) assert(out==(mode==0 ? (void*)23 : nullptr));
}
'''
    src = tmp_path/'bridge.cpp'; src.write_text(code)
    binary = tmp_path/'bridge'
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Werror', '-I'+str(ROOT), str(src), '-o', str(binary)], check=True)
    for mode in range(14):
        p = subprocess.run([str(binary), str(mode)], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        if mode == 1: assert 'stage=capture-info rc=801' in p.stderr and 'stage=clone' not in p.stderr
        if mode == 5: assert 'stage=clone rc=801' in p.stderr
        if mode in (4, 9, 10, 11):
            assert 'origin=HARNESS_NOT_CLONE_API' in p.stderr and 'stage=clone' not in p.stderr


def result(rc=0, status='PASS', correctness='PASS', timeout=False):
    return dict(process=dict(rc=rc, timeout=timeout), result=dict(status=status, data=dict(correctness=correctness)))


def test_control_failure_is_not_a_candidate_pass_and_exit_status_is_required():
    for bad in (result(rc=-6), result(status='FAIL'), result(correctness='PENDING'), result(timeout=True)):
        assert not arm_pass(bad)
        assert summarize(dict(default=result(), resident=bad))['status'] == 'FAIL'
    s = summarize(dict(default=result(rc=-6), resident=result()))
    assert s['status'] == 'CANDIDATE_PASS' and s['default_reproduced_failure']
    assert not s['production_changed'] and not s['performance_admitted']
    assert not summarize(dict(default=result(), resident=result()))['default_reproduced_failure']


def test_real_child_failure_does_not_stop_the_next_process(tmp_path):
    with (tmp_path/'lock').open('a') as held:
        first = wait_arm([sys.executable, '-c', 'raise SystemExit(23)'], tmp_path/'bad.log', held, 5)
        second = wait_arm([sys.executable, '-c', 'print("CONTROL_OK")'], tmp_path/'ok.log', held, 5)
    assert first['rc'] == 23 and second['rc'] == 0 and not second['timeout']
    assert 'CONTROL_OK' in (tmp_path/'ok.log').read_text()


def test_timeout_kills_only_owned_child_and_retains_result(tmp_path):
    with (tmp_path/'lock').open('a') as held:
        r = wait_arm([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path/'slow.log', held, .1)
    assert r['timeout'] and r['rc'] < 0 and r['seconds'] < 5


def test_result_archive_omits_binaries_but_keeps_failed_logs_and_samples(tmp_path):
    out = tmp_path/'probe'; out.mkdir()
    for name in ('authority.json', 'native.json', 'bad.log', 'weights.bin', 'helper.so', 'helper.o'):
        (out/name).write_text('{}')
    (out/'resident').mkdir(); (out/'resident/result.json').write_text(json.dumps(result()))
    package(out)
    with tarfile.open(str(out)+'.results.tgz') as archive:
        assert set(archive.getnames()) == {'authority.json', 'bad.log', 'resident/result.json'}
    with pytest.raises(FileExistsError): package(out)

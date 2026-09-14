from pathlib import Path
import ctypes as C
import os
import subprocess
import numpy as np
import pytest
from test_kpack_native_dispatch import build_stub, probe, Choice, Call
from test_kpack_jit import enable, helper_file

ROOT = Path(__file__).resolve().parents[1]


def test_actual_typed_device_probe_uses_the_same_sm_attribute_as_query(tmp_path):
    text = (ROOT / 'quactlize/decode/dense.cuh').read_text()
    begin = text.index('extern "C" int quactlize_kpack_decode_dense_device_v1(')
    body = text[begin:text.index('extern "C" int quactlize_kpack_decode_dense_query_v1(', begin)]
    # PPU property structure reports 1, but the explicit SM attribute is 72.
    # Compile the actual host entry, without constructing any GPU collective.
    source = r'''
#include <cstdint>
#include <cstring>
#include <cassert>
enum { QK_OK=0,QK_INVALID=2,QK_RUNTIME_ERROR=3,hggcSuccess=0 };
struct hggcDeviceProp { char name[256]; int multiProcessorCount; };
int attribute=72;
int hggcGetDevice(int* d) { *d=0; return 0; }
int hggcGetDeviceProperties(hggcDeviceProp* p,int) {
    std::strcpy(p->name,"PPU-ZW810");p->multiProcessorCount=1;return 0;
}
namespace cutlass { struct KernelHardwareInfo {
    static int query_device_multiprocessor_count(int) { return attribute; }
}; }
''' + body + r'''
int main() {
    char name[128]{};int device=-1,cu=-1;
    assert(quactlize_kpack_decode_dense_device_v1(name,128,&device,&cu)==0);
    assert(cu==72 && device==0 && !std::strcmp(name,"PPU-ZW810"));
    attribute=0;
    assert(quactlize_kpack_decode_dense_device_v1(name,128,&device,&cu)==QK_RUNTIME_ERROR);
    assert(quactlize_kpack_decode_dense_device_v1(name,2,&device,&cu)==QK_INVALID);
}
'''
    path=tmp_path/'probe.cpp';path.write_text(source)
    exe=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17',str(path),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_decode_device_preflight_retains_mismatch_values(tmp_path, capsys):
    import json
    from types import SimpleNamespace
    from tools.probe_kpack_decode_device import probe
    from quactlize.runtime.compiler import sha
    source=tmp_path/'probe.cpp'
    source.write_text(r'''
#include <cstring>
extern "C" int hggcDeviceGetAttribute(int* v,int a,int d) { *v=72; return a!=16||d!=0; }
extern "C" int quactlize_kpack_device_v1(char* n,int,int* d,int* c) { std::strcpy(n,"PPU-ZW810");*d=0;*c=72;return 0; }
extern "C" int quactlize_kpack_decode_dense_device_v1(char* n,int,int* d,int* c) { std::strcpy(n,"PPU-ZW810");*d=0;*c=1;return 0; }
''')
    lib=tmp_path/'probe.so'
    subprocess.run(['g++','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    manifest=dict(modules=[dict(identity=dict(endpoints='typed') if typed else {},
        path=lib.name,sha256=sha(lib),parent=dict(symbol='control'),key='a'*64) for typed in (False,True)])
    rows,valid=probe(SimpleNamespace(lib=C.CDLL(str(lib))),tmp_path,manifest)
    assert not valid and rows[0]['reported_cu']==72 and rows[1]['reported_cu']==1
    assert rows[1]['attribute_cu']==72
    assert 'IDENTITY_MISMATCH_NOT_NUMERICAL' in capsys.readouterr().out


def test_typed_input_matches_independent_tsm_reader(tmp_path):
    binary = tmp_path / "decode-input"
    sdk = Path(os.environ.get("PPU_SDK", "/root/ppu-sdk/2.1.1"))
    result = subprocess.run([
        "g++", "-O2", "-std=c++17", f"-I{ROOT/'quactlize/include'}",
        f"-I{ROOT/'third_party/actlize/include'}", f"-I{sdk/'include'}",
        f"-I{sdk/'targets/x86_64-linux/include'}",
        str(ROOT / "tests/kpack_decode_input_host.cpp"), "-o", str(binary),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KPACK_DECODE_INPUT PASS" in result.stdout
    assert "negatives=6" in result.stdout


class TypedCall(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32), ("call", Call),
                ("input_type", C.c_int32), ("output_type", C.c_int32)]


def test_typed_query_reuses_policy_but_not_half_module_or_ticket(tmp_path, probe):
    f, runtime, request = build_stub(tmp_path, probe, values=(12, 0, 1, 1024, 5120, 1, 1), jit=True)
    directory = tmp_path / "modules" / ("f" * 64)
    directory.mkdir()
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-DQK_TEST_TYPED",
        f"-I{ROOT}", f"-I{tmp_path}", str(ROOT / "tests/kpack_dispatch_stub.cpp"),
        "-o", str(directory / "kernel.so")], check=True)
    helper = helper_file(tmp_path,
        f"count=Path({str(tmp_path/'calls')!r})\n"
        "typed='--dense-io' in sys.argv\n"
        "count.write_text((count.read_text() if count.exists() else '')+('T' if typed else 'H'))\n"
        f"print('QK_JIT_V1 '+('f' if typed else 'a')*64+' {'b'*64} {'d'*64}')\n")
    assert enable(f, runtime, tmp_path, helper) == 0
    handles = []
    try:
        choices = []
        for storage in (1, 2):
            choice = Choice()
            for _ in range(2):
                assert f["query_dense_io"](runtime, C.byref(request), storage, 0, C.byref(choice)) == 0
            assert choice.build_key == b"f" * 64
            assert choice.shared_bytes == 4096 + storage * 128
            call = Call(version=1, size=C.sizeof(Call), m=request.m, n=request.n, k=request.k,
                experts=1, group_size=32, device=0, compute_units=72, mapping_id=request.mapping_id,
                a=4096, low=8192, metadata=12288, output=16384, workspace=24576,
                workspace_bytes=choice.workspace_bytes)
            typed = TypedCall(1, C.sizeof(TypedCall), call, storage, storage)
            handle = C.c_void_p()
            assert f["prepare"](runtime, C.byref(choice), C.byref(call), C.byref(handle)) == 2
            typed.output_type = 3 - storage
            assert f["prepare_dense_io"](runtime, C.byref(choice), C.byref(typed), C.byref(handle)) == 2
            typed.output_type = storage
            assert f["prepare_dense_io"](runtime, C.byref(choice), C.byref(typed), C.byref(handle)) == 0
            handles.append(handle)
            choices.append(choice)
        assert choices[0].ticket != choices[1].ticket
        assert (tmp_path / "calls").read_text() == "T"
        old = Choice()
        assert f["query"](runtime, C.byref(request), C.byref(old)) == 0
        assert old.parent == choices[0].parent
        assert (old.policy, old.algorithm, old.split) == (choices[0].policy, choices[0].algorithm, choices[0].split)
        assert old.build_key == b"a" * 64 and old.ticket not in [c.ticket for c in choices]
        assert f["prepare_dense_io"](runtime, C.byref(old), C.byref(typed), C.byref(C.c_void_p())) == 2
        assert (tmp_path / "calls").read_text() == "TH"
        request.m = request.max_rows = 128
        miss = Choice()
        assert f["query_dense_io"](runtime, C.byref(request), 1, 0, C.byref(miss)) == 1
        assert not miss.ticket
        assert f["query_dense_io"](runtime, C.byref(request), 3, 0, C.byref(miss)) == 2
        assert f["query_dense_io"](runtime, C.byref(request), 1, 2, C.byref(miss)) == 2
        for handle in handles:
            for _ in range(3):
                assert f["run"](handle, None) == 0
        assert (tmp_path / "calls").read_text() == "TH"
    finally:
        for handle in handles:
            f["destroy"](handle)
        f["close"](runtime)


def test_typed_cache_identity_and_catalog_are_separate(tmp_path, monkeypatch):
    from test_kpack_jit import fake_compiler
    from quactlize.decode.compiler import DecodeCompiler
    from quactlize.runtime.compiler import Compiler, source_contract
    from tools.kpack_jit import parent_tuple, inspect_modules
    from tools.build_kpack_dispatch import catalog
    sdk = fake_compiler(tmp_path)
    monkeypatch.setenv("PATH", str(sdk / "bin") + os.pathsep + os.environ["PATH"])
    parent = parent_tuple("parent", [12, 0, 8, 64, 64, 8, 16, 2, 0, 16, -1])
    normal = Compiler(sdk, tmp_path / "cache")
    typed = DecodeCompiler(sdk, tmp_path / "cache")
    old, new = normal.build(parent), typed.build(parent)
    assert old["key"] != new["key"]
    assert typed.build(parent)["cache_hit"]
    contract = source_contract(normal.identity)
    assert new["identity"]["base_source_contract"] == contract
    assert len(inspect_modules(tmp_path / "cache", [old["key"], new["key"]], contract)) == 2
    source = catalog([old, new], contract)
    assert source.count(',{},true') == 1
    with pytest.raises(ValueError):
        catalog([new, new], contract)


def test_q8_gate_has_independent_n_by_k_sums():
    from tools.run_kpack_decode_io import fixture
    from tools.run_q8_kpack2_gate import fixture as raw_fixture
    n, k = 256, 512
    w = fixture(8, n, k)
    _, _, _, weight = raw_fixture(n, k, 1)
    assert w.sums.shape == (1, 4, n)
    np.testing.assert_array_equal(w.sums[0].sum(0), weight[0].astype('f8').sum(1))


def test_bf16_storage_and_numerical_negatives():
    from tools.run_kpack_decode_io import bf16_bits, bf16_float, output_check
    values = np.array([1.0, -1.0, 1.00390625, 1.01171875, 0.0], dtype='f4')
    np.testing.assert_array_equal(bf16_bits(values), [0x3f80, 0xbf80, 0x3f80, 0x3f82, 0])
    np.testing.assert_array_equal(bf16_float(bf16_bits(values))[:2], values[:2])
    assert output_check(np.array([1.]), np.array([1.]), np.array([1.])) == 0
    for wrong in (0., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            output_check(np.array([wrong]), np.array([1.]), np.array([1.]))


def test_llama_api_mirrors_and_dense_direct_pointers():
    llama = Path('/root/llama.cpp/ggml/src/ggml-cuda')
    if not llama.is_dir():
        pytest.skip('private llama integration checkout is not present')
    expected = (ROOT/'quactlize/dispatch/api.h').read_text().replace(
        '#include "../runtime/abi.h"', '#include "kpack_module.h"').replace(
        '#include "../integrations/llama/indexed.h"', '#include "kpack_indexed.h"').replace(
        '#include "../decode/api.h"', '#include "kpack_decode_io.h"')
    assert (llama/'quactlize/kpack_dispatch.h').read_text() == expected
    expected = (ROOT/'quactlize/decode/api.h').read_text().replace(
        '#include "../runtime/abi.h"', '#include "kpack_module.h"')
    assert (llama/'quactlize/kpack_decode_io.h').read_text() == expected
    source = (llama/'quactlize-execution.cu').read_text()
    assert '!ids && tokens <= 8' in source
    assert 'c.a = p->dense_io ? input->data : p->a' in source
    assert 'c.output = p->dense_io ? output->data : p->out' in source
    assert 'p->dense_io ? 0 : align256' in source
    run = source.split('bool ggml_quactlize_execution_run(')[1]
    assert run.index('if (p.indexed || p.dense_io)') < run.index('gather<<<')


def test_box_package_matches_all_gate_requests_without_jit():
    import json
    from tools.build_kpack_decode_io import native_simt_source
    from tools.verify_kpack_dispatch import verify
    from quactlize.runtime.compiler import sha
    root = ROOT/'prebuilt/ppu0010/kpack-decode-io-v1'
    manifest = verify(root)
    typed = {r['parent']['symbol'] for r in manifest['modules'] if 'endpoints' in r['identity']}
    ordinary = {r['parent']['symbol'] for r in manifest['modules'] if 'endpoints' not in r['identity']}
    points = manifest['decode_io_gate']
    assert len(typed) == 27 and len(ordinary) == 5 and len(points['requests']) == 104
    for point in points['requests']:
        assert point['status'] == 'SELECTED' and point['parent'] in typed
    for point in points['grouped']:
        assert point['status'] == 'SELECTED' and point['parent'] in ordinary
    for item in points['simt_binaries']:
        assert sha(root/item['path']) == item['sha256']
        source = ROOT/'tests'/(item['path']+'.cu')
        assert sha(source) == item['source_sha256']
        native = native_simt_source(source.read_text())
        assert '<hggc_runtime.h>' in native and 'cudaStream' not in native
        assert 'quactlize/runtime/' in native
    assert json.loads((root/manifest['decode_policy']['path']).read_text())


@pytest.mark.parametrize('missing_index', range(4))
def test_moe_pack_preflight_requires_every_entry_before_host_queries(tmp_path, missing_index):
    import json
    from tools.run_kpack_moe_gate import load_pack_library, PACK_SYMBOLS
    from quactlize.runtime.compiler import sha
    # Any query call here would return 99: all symbols must be resolved first.
    source = tmp_path / 'partial.cpp'
    source.write_text('\n'.join(f'extern "C" int {name}() {{ return 99; }}'
                                for i, name in enumerate(PACK_SYMBOLS) if i != missing_index))
    library = tmp_path / 'partial.so'
    subprocess.run(['g++', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        schema='quactlize.kpack-device-pack-build.v1', library=library.name, sha256=sha(library))))
    with pytest.raises(ValueError, match='lacks the MoE producer ABI') as error:
        load_pack_library(library)
    assert PACK_SYMBOLS[missing_index] in str(error.value)


def test_actual_published_moe_pack_abi_and_old_producer_negative(tmp_path):
    """Real LFS host queries; runtime registration is stubbed, launches abort."""
    # Loading the published PPU DSO registers its device images. Those calls
    # are inert here so this host-only proof also runs without the SDK's
    # Ubuntu 24.04 runtime. The exported producer queries remain unmodified.
    source = tmp_path / 'registration.cpp'
    source.write_text(r'''
#include <cstdlib>
extern "C" void** __hggcRegisterFatBinary(void*) { static void* image; return &image; }
extern "C" void __hggcUnregisterFatBinary(void**) {}
extern "C" void __hggcRegisterFunction(void**, ...) {}
extern "C" void __hggcRegisterVar(void**, ...) {}
extern "C" int __hggcPushCallConfiguration(...) { std::abort(); }
extern "C" int __hggcPopCallConfiguration(...) { std::abort(); }
extern "C" int hggcLaunchKernel(...) { std::abort(); }
extern "C" int hggcGetLastError() { std::abort(); }
''')
    subprocess.run(['g++', '-shared', '-fPIC', str(source),
                    '-Wl,-soname,libhggc_wrapper.so', '-o', str(tmp_path/'libhggc_wrapper.so')], check=True)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(tmp_path) + ':' + env.get('LD_LIBRARY_PATH', '')
    source = r'''
from pathlib import Path
from tools.run_kpack_moe_gate import load_pack_library
root = Path('prebuilt/ppu0010')
try:
    load_pack_library(root/'kpack-pack-v1/libquactlize_ppu_pack.so')
except ValueError as error:
    assert 'canonical_arrangement_v1' in str(error)
    assert 'prepare_gate_up_dev_for_arrangement_v1' in str(error)
else:
    raise AssertionError('old single-source producer was admitted')
_, proof = load_pack_library(root/'kpack-fusion-v1/libquactlize_ppu_pack.so')
assert proof['status'] == 'PASS' and proof['scope'] == 'HOST_ABI_ONLY'
assert proof['formats'] == [8, 10, 11, 12, 13, 14]
assert len(proof['symbols']) == 4
print('DECODE_PACK_DEPENDENCY PASS old=REJECTED paired=HOST_ABI_PASS kernels_launched=0')
'''
    import sys
    result = subprocess.run([sys.executable, '-c', source], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'DECODE_PACK_DEPENDENCY PASS' in result.stdout
    launcher = (ROOT/'tools/run_kpack_decode_io_device_fix_box.sh').read_text()
    runner = (ROOT/'tools/run_kpack_decode_io_ppu_box.sh').read_text()
    dependency = 'prebuilt/ppu0010/kpack-fusion-v1/libquactlize_ppu_pack.so'
    assert dependency in launcher and dependency in runner
    assert 'kpack-pack-v1' not in launcher + runner
    assert runner.index('phase=pack-library') < runner.index('phase=indexed-stages')


def test_decode_pack_failure_stops_before_fixtures_and_records_reason(tmp_path, monkeypatch):
    import json
    import sys
    import tools.run_kpack_decode_io as gate
    import tools.run_kpack_moe_gate as moe
    monkeypatch.setattr(sys, 'argv', ['gate', '--sdk', str(tmp_path), '--bundle', str(tmp_path),
                                    '--output', str(tmp_path/'results'), '--pack-library', str(tmp_path/'old.so')])
    monkeypatch.setattr(gate, 'verify', lambda *args, **kwargs: {})
    def reject(*args):
        raise ValueError('missing paired producer')
    def device(*args):
        raise AssertionError('device initialized before dependency validation')
    monkeypatch.setattr(moe, 'load_pack_library', reject)
    monkeypatch.setattr(gate, 'SDK', device)
    assert gate.main() == 2
    result = json.loads((tmp_path/'results/summary.json').read_text())
    assert result['status'] == 'INFRASTRUCTURE_FAIL'
    assert result['phase'] == 'pack-library' and result['numerical_cases_started'] == 0
    assert result['error'] == 'missing paired producer'

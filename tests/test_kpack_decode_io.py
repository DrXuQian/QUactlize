from pathlib import Path
import ctypes as C
import os
import subprocess
import numpy as np
import pytest
from test_kpack_native_dispatch import build_stub, probe, Choice, Call
from test_kpack_jit import enable, helper_file

ROOT = Path(__file__).resolve().parents[1]


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

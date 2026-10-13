"""Complete provider host contracts. Not PPU numeric/performance evidence."""
from pathlib import Path
import ctypes as C
import os
import subprocess

import pytest
from test_kpack_native_dispatch import build_stub, probe, Choice, Request
from test_kpack_jit import enable, helper_file
from test_kpack_dynamic import dynamic_functions

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('q,route,m', [(12, 0, 9), (12, 0, 127), (12, 1, 128),
                                      (12, 0, 8192), (8, 1, 9), (8, 1, 4096)])
def test_general_dynamic_does_not_expand_legacy_decode_domain(tmp_path, probe, q, route, m):
    f, runtime, request = build_stub(tmp_path, probe, values=(q, route, m, 1024, 5120, 1, m), jit=True)
    module = tmp_path / 'modules' / ('f' * 64)
    module.mkdir()
    subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-DQK_TEST_DYNAMIC', '-DQK_TEST_TYPED',
        '-DQK_TEST_GENERAL', f'-I{ROOT}', f'-I{tmp_path}', str(ROOT / 'tests/kpack_dispatch_stub.cpp'),
        '-o', str(module / 'kernel.so')], check=True)
    helper = helper_file(tmp_path, f"print('QK_JIT_V1 {'f'*64} {'b'*64} {'d'*64}')\n")
    assert enable(f, runtime, tmp_path, helper) == 0
    lib = C.CDLL(str(tmp_path / 'dispatch.so'))
    query = lib.quactlize_kpack_dispatch_query_dense_dynamic_v1
    query.argtypes = [C.c_void_p, C.POINTER(Request), C.c_int, C.POINTER(Choice)]
    contract = lib.quactlize_kpack_dispatch_dense_contract_v1
    contract.argtypes = [C.c_void_p, C.POINTER(Choice), C.POINTER(Request), C.POINTER(C.c_int), C.POINTER(C.c_int)]
    choice, handle = Choice(), C.c_void_p()
    try:
        assert f['query_dense_io'](runtime, C.byref(request), 1, 0, C.byref(choice)) == 1
        assert query(runtime, C.byref(request), 0, C.byref(choice)) == 0
        observed, compute, metadata = Request(), C.c_int(-1), C.c_int(-1)
        assert contract(runtime, C.byref(choice), C.byref(observed), C.byref(compute), C.byref(metadata)) == 0
        assert bytes(observed) == bytes(request)
        assert compute.value == metadata.value == 0
        assert dynamic_functions(tmp_path)[0](runtime, C.byref(choice), C.byref(handle)) == 0
        choice.split = 99
        assert contract(runtime, C.byref(choice), C.byref(observed), C.byref(compute), C.byref(metadata)) == 2
    finally:
        if handle.value:
            f['destroy'](handle)
        f['close'](runtime)


def test_public_provider_orchestration_and_poc_selection(tmp_path):
    sdk = Path(os.environ.get('PPU_SDK', '/root/ppu-sdk/2.1.1'))
    exe = tmp_path / 'provider'
    (tmp_path / 'catalog.inc').write_text('static std::vector<Image> kImages{};\nstatic char const kJitSource[]="";\n')
    subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-rdynamic', f'-I{ROOT}', f'-I{tmp_path}',
        f'-I{ROOT / "quactlize/include"}', f'-I{ROOT / "third_party/actlize/include"}', f'-I{sdk / "include"}',
        str(ROOT / 'tests/dense_provider_host.cpp'), str(ROOT / 'quactlize/packing/sizes.cpp'),
        str(ROOT / 'quactlize/execution/simt.cpp'), str(ROOT / 'quactlize/execution/q4_decode.cpp'),
        '-ldl', '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_public_header_is_c_and_cpp_compatible(tmp_path):
    for compiler, language in [('gcc', 'c'), ('g++', 'c++')]:
        subprocess.run([compiler, '-x', language, '-fsyntax-only', f'-I{ROOT}',
            f'-I{ROOT / "quactlize/include"}', '-'],
            input='#include "quactlize/dense/api.h"\n', text=True, check=True)


def test_auto_delivery_is_only_admitted_for_legacy_dense_fq():
    from quactlize.runtime.compiler import validate_parent
    parent = dict(symbol='legacy', qtype=12, route='fq-dense', tm=64, tn=128,
                  tk=256, wm=64, wn=16, stages=2, ap=0, dn=0, persistent=-1)
    validate_parent(parent)
    with pytest.raises(ValueError):
        validate_parent(dict(parent, route='sf-dense'))
    with pytest.raises(ValueError):
        validate_parent(dict(parent, route='fq-grouped', persistent=0))
    with pytest.raises(ValueError):
        validate_parent(dict(parent, tm=8, wm=8, ap=1))

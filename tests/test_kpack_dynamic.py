"""Dynamic execution ABI tests. Host tests are NOT PPU numerical admission."""
import ctypes as C
from pathlib import Path
import os
import subprocess

import numpy as np
import pytest

from test_kpack_native_dispatch import build_stub, probe, Choice, Call
from test_kpack_jit import enable, helper_file

ROOT = Path(__file__).resolve().parents[1]


def dynamic_functions(path):
    lib = C.CDLL(str(path / 'dispatch.so'))
    prepare = lib.quactlize_kpack_dispatch_prepare_dynamic_v1
    prepare.argtypes, prepare.restype = [C.c_void_p, C.POINTER(Choice), C.POINTER(C.c_void_p)], C.c_int
    run = lib.quactlize_kpack_dispatch_run_dynamic_v1
    run.argtypes, run.restype = [C.c_void_p, C.POINTER(Call)], C.c_int
    return prepare, run


@pytest.mark.parametrize('typed', [False, True])
@pytest.mark.parametrize('q,route', [(12, 0), (12, 1), (8, 1)])
def test_dispatch_dynamic_keeps_ticket_but_changes_all_bindings(tmp_path, probe, typed, q, route):
    f, runtime, request = build_stub(tmp_path, probe, values=(q, route, 1, 1024, 5120, 1, 1), jit=typed)
    key = 'f' if typed else 'a'
    directory = tmp_path / 'modules' / (key * 64)
    directory.mkdir(exist_ok=True)
    subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-DQK_TEST_DYNAMIC',
                    *(['-DQK_TEST_TYPED'] if typed else []), f'-I{ROOT}', f'-I{tmp_path}',
                    str(ROOT / 'tests/kpack_dispatch_stub.cpp'), '-o', str(directory / 'kernel.so')], check=True)
    if typed:
        helper = helper_file(tmp_path, f"print('QK_JIT_V1 {'f'*64} {'b'*64} {'d'*64}')\n")
        assert enable(f, runtime, tmp_path, helper) == 0
    choice = Choice()
    if typed:
        assert f['query_dense_io'](runtime, C.byref(request), 1, 0, C.byref(choice)) == 0
    else:
        assert f['query'](runtime, C.byref(request), C.byref(choice)) == 0
    prepare, run = dynamic_functions(tmp_path)
    handle = C.c_void_p()
    original = bytes(choice)
    choice.split = 99
    assert prepare(runtime, C.byref(choice), C.byref(handle)) == 2 and not handle.value
    choice = Choice.from_buffer_copy(original)
    assert prepare(runtime, C.byref(choice), C.byref(handle)) == 0 and handle.value
    f['close'](runtime)  # The live dynamic handle keeps its module alive.
    try:
        buffers = []
        for stream in (1, 2):
            a, output, scratch, low, scale, zero = [np.zeros(64, dtype='f4') for _ in range(6)]
            assert all(v.ctypes.data % 16 == 0 for v in (a, output, scratch, low, scale, zero))
            a[0] = 4 * stream
            c = Call(version=1, size=C.sizeof(Call), m=1, n=request.n, k=request.k, experts=1,
                     group_size=32, device=choice.device, compute_units=choice.compute_units,
                     mapping_id=request.mapping_id, a=a.ctypes.data, output=output.ctypes.data,
                     low=low.ctypes.data, metadata=scale.ctypes.data,
                     zero=zero.ctypes.data if route == 1 and q != 8 else None,
                     workspace=scratch.ctypes.data, workspace_bytes=scratch.nbytes, stream=stream)
            buffers.append((c, a, output, scratch, low, scale, zero))
        for c, a, output, scratch, *_ in buffers * 3:
            assert run(handle, C.byref(c)) == 0
            assert output[0] == a[0] + c.stream
            assert scratch.view('i4')[0] == choice.split
        assert f['run'](handle, None) == 2  # Never mix bound/dynamic lifecycles.
        call = buffers[0][0]
        for field, bad in [('m', 2), ('n', 512), ('mapping_id', 0), ('device', 7),
                           ('workspace_bytes', 0), ('a', None), ('output', None), ('workspace', None)]:
            changed = Call.from_buffer_copy(call)
            setattr(changed, field, bad)
            assert run(handle, C.byref(changed)) != 0
    finally:
        f['destroy'](handle)


def test_old_modules_remain_usable_but_do_not_fake_dynamic_support(tmp_path, probe):
    f, runtime, request = build_stub(tmp_path, probe, values=(12, 0, 1, 1024, 5120, 1, 1))
    prepare, _ = dynamic_functions(tmp_path)
    choice, handle = Choice(), C.c_void_p()
    try:
        assert f['query'](runtime, C.byref(request), C.byref(choice)) == 0
        assert prepare(runtime, C.byref(choice), C.byref(handle)) == 3 and not handle.value
        assert b'rebuild' in f['error']()
    finally:
        f['close'](runtime)


def test_grouped_is_not_advertised_as_dense_dynamic(tmp_path, probe):
    f, runtime, request = build_stub(tmp_path, probe)
    prepare, _ = dynamic_functions(tmp_path)
    choice, handle = Choice(), C.c_void_p()
    try:
        assert f['query'](runtime, C.byref(request), C.byref(choice)) == 0
        assert prepare(runtime, C.byref(choice), C.byref(handle)) == 1 and not handle.value
    finally:
        f['close'](runtime)


def test_actual_prefill_dynamic_composition_and_stream_isolation(tmp_path):
    sdk = Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK'))
    provider = tmp_path / 'CUDA_SDK/targets/x86_64-linux/lib'
    provider.mkdir(parents=True)
    subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC',
                    str(ROOT / 'tests/kpack_prefill_provider_stub.cpp'),
                    '-o', str(provider / 'libcublas.so')], check=True)
    exe = tmp_path / 'prefill-dynamic'
    subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', f'-I{ROOT}',
                    f'-I{ROOT / "quactlize/include"}', f'-I{sdk / "include"}',
                    str(ROOT / 'tests/kpack_prefill_dynamic_host.cpp'),
                    str(ROOT / 'quactlize/prefill/runtime.cpp'), '-ldl', '-o', str(exe)], check=True)
    subprocess.run([str(exe), str(tmp_path)], check=True)

import ctypes as C
import inspect
from types import SimpleNamespace

import pytest

from tools import run_kpack_decode_sweep as sweep
from tools.run_kpack_gemv_gate import Resources


@pytest.mark.parametrize("ordered", [False, True])
def test_delayed_default_poison_is_red_and_actual_same_stream_helper_is_green(ordered):
    # A legal queue interleaving: the nonblocking stream drains first, then
    # delayed work from the default stream. Use the real fixture helper.
    memory = C.create_string_buffer(16)
    pointer = C.addressof(memory)
    queues = {0: [], 7: []}

    def memset_async(dst, byte, size, stream):
        queues[stream.value].append(lambda: C.memset(dst, byte, size))
        return 0

    r = Resources.__new__(Resources)
    r.sdk = SimpleNamespace(lib=SimpleNamespace(hggcMemsetAsync=memset_async))
    r.stream = C.c_void_p(7)
    if ordered:
        r.fill(pointer, 0xA5, 16)
    else:
        queues[0].append(lambda: C.memset(pointer, 0xA5, 16))
    queues[7].append(lambda: C.memset(pointer, 0x3C, 16))
    for stream in (7, 0):
        for operation in queues[stream]:
            operation()
    assert memory.raw == bytes([0x3C if ordered else 0xA5]) * 16


def test_fixture_fill_propagates_runtime_error():
    r = Resources.__new__(Resources)
    r.sdk = SimpleNamespace(lib=SimpleNamespace(hggcMemsetAsync=lambda *args: 200))
    r.stream = C.c_void_p(7)
    with pytest.raises(RuntimeError, match="status=200"):
        r.fill(4096, 0xA5, 16)


def test_same_stream_poison_is_outside_captured_timing_body():
    for function in (sweep.gemm_cell, sweep.simt_cell):
        source = inspect.getsource(function)
        assert "sdk.fill(" not in source
        assert "r.fill(" in source
        assert "sdk.synchronize(None)" in source
        assert "lambda:" in source and "Replay(" in source
        # The capture callable remains module.run(), not a poison wrapper.
        launch = next(line for line in source.splitlines() if "launch = lambda:" in line)
        assert "fill" not in launch and "synchronize" not in launch


def test_pageable_fixture_upload_finishes_before_source_lifetime_ends():
    observed = []
    sdk = SimpleNamespace(
        lib=SimpleNamespace(hggcMemcpy=lambda *args: (observed.append("H2D"), 0)[1]),
        synchronize=lambda stream: observed.append(("drain", stream)),
    )
    sweep.upload_into(sdk, 4096, [1, 2, 3])
    assert observed == ["H2D", ("drain", None)]

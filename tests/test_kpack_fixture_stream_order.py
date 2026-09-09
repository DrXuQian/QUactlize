import ctypes as C
import inspect
from types import SimpleNamespace

import pytest

from tools import run_kpack_decode_sweep as sweep
from tools import run_kpack_native_gate as native
from tools.run_kpack_gemv_gate import Resources
import numpy as np


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


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("prepass_present", [False, True])
def test_native_sf_poison_cannot_overwrite_prepass(ordered, prepass_present):
    output, scale, zero = [np.zeros(8, dtype="<f2") for _ in range(3)]
    queues = {0: [], 7: []}

    def memset_async(dst, byte, size, stream):
        queues[stream.value].append(lambda: C.memset(dst, byte, size))
        return 0

    r = Resources.__new__(Resources)
    r.sdk = SimpleNamespace(lib=SimpleNamespace(hggcMemsetAsync=memset_async))
    r.stream = C.c_void_p(7)
    if ordered:
        # Execute the native gate's actual fill helper, not a copied sequence.
        native.poison_call(r, output.ctypes.data, output.nbytes,
                           scale.ctypes.data, zero.ctypes.data, scale.nbytes)
    else:
        for array, byte in ((output, 0xA5), (scale, 0x7E), (zero, 0x7E)):
            queues[0].append(lambda a=array, b=byte: C.memset(a.ctypes.data, b, a.nbytes))
    if prepass_present:
        queues[7].append(lambda: (scale.fill(1), zero.fill(0)))
    queues[7].append(lambda: np.copyto(output, scale + zero))
    # A legal nonblocking/default interleaving: prepass, delayed default
    # poison, then GEMM. The correct same-stream order forbids this overwrite.
    while len(queues[7]) > 1:
        queues[7].pop(0)()
    for operation in queues[0] + queues[7]:
        operation()
    if ordered and prepass_present:
        assert np.array_equal(output, np.ones_like(output))
    else:
        assert np.isnan(output).all()


def test_native_gate_uses_same_stream_poison_and_checks_eager_before_capture():
    source = inspect.getsource(native.run)
    assert "sdk.fill(" not in source
    assert source.count("sdk.synchronize(None)") == 2  # Weights, then each A/router upload.
    assert source.index('check("eager", index)') < source.index("hggcStreamBeginCapture")
    assert 'check("graph_replay", index)' in source
    launch = source[source.index("def launch():"):source.index("def check(")]
    assert "fill(" not in launch and "synchronize(" not in launch


@pytest.mark.parametrize("fault", ["nan", "poison", "finite"])
def test_native_output_failure_preserves_phase_and_first_bad(fault):
    gold = np.ones((2, 8), dtype="f8")
    got = gold.astype("<f2")
    if fault == "nan":
        got.view("<u2")[1, 3] = 0x7E7E
    elif fault == "poison":
        got.view("<u2")[1, 3] = 0xA5A5
    else:
        got[1, 3] = 2
    with pytest.raises(native.OutputFailure) as error:
        native.check_output(got, gold, np.ones_like(gold), dict(q=14, m=128), "graph_replay", 2)
    proof = error.value.proof
    assert proof["first"]["coord"] == [1, 3] and proof["bad"] == 1
    assert proof["phase"] == "graph_replay" and proof["profile"] == 2
    assert proof["nonfinite"] == int(fault == "nan")
    assert proof["metadata_poison"] == int(fault == "nan")
    assert proof["output_poison"] == int(fault == "poison")
    assert len(proof["golden_sha256"]) == len(proof["output_sha256"]) == 64

import ctypes as C
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reference import gguf_kpack as ref
from tests.test_kpack_execution import host, raw_fixture
from tests.test_kpack_execution import (
    test_independent_offline_bytes_and_affine_reader as check_reader,
)
from quactlize.execution.native import Call, Config, Sizes, arrangement
from tools import build_kpack_decode_sweep as builder
from tools import run_kpack_decode_sweep as sweep
from tools.kpack_execution_fixture import IndexedWeights


@pytest.mark.parametrize("q", range(10, 15))
def test_reused_words_cover_every_group_and_both_planes(host, q):
    n, k, e = 256, 512, 3
    art = ref.prepare_grouped(torch.from_numpy(raw_fixture(q, n, k, e)), n, k, q, e)
    fn = host.qkg_host_reuse
    fn.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_bool]
    args = (
        q,
        art.low.data_ptr(),
        art.high.data_ptr() if art.high.numel() else None,
        n,
        k,
        e,
    )
    assert fn(*args, False) == 0
    assert fn(*args, True) > n * k * e // 4


def test_magic_pair_is_exact_before_scale_and_zero():
    raw = np.arange(64, dtype="u2")
    magic = (raw + 0x6400).view("f2")
    for offset in (0, 8):
        converted = np.subtract(magic, np.float16(1024 + offset), dtype=np.float16)
        assert np.array_equal(converted, (raw.astype("i4") - offset).astype("f2"))
        s, z = np.float16(0.01), np.float16(0.1)
        baseline = np.add(
            np.multiply(converted, s, dtype=np.float16), z, dtype=np.float16
        )
        wrong = (magic.astype("f4") * s + (z - np.float16((1024 + offset) * s))).astype(
            "f2"
        )
        assert np.any(baseline != wrong)


def test_pair_fused_affine_subnormals_and_zero_signs(host):
    assert host.qkg_host_pair_arithmetic() == 0


@pytest.mark.parametrize("q", range(10, 15))
def test_fused_pair_reader_against_independent_gguf(host, q):
    host.qkg_host_pair_read.argtypes = host.qkg_host_read.argtypes
    check_reader(SimpleNamespace(qkg_host_read=host.qkg_host_pair_read), q)


class Layout(C.Structure):
    _fields_ = [
        (x, C.c_uint64)
        for x in (
            "shapes",
            "outputs",
            "strides",
            "rows",
            "partials",
            "partial_bytes",
            "total",
        )
    ]


@pytest.mark.parametrize("split", [1, 2, 4, 8])
@pytest.mark.parametrize("device", [False, True])
def test_workspace_retains_s1_abi_and_places_slice_major_fp32(host, split, device):
    fn = host.qkg_host_grouped_workspace
    fn.argtypes = [C.c_int] * 4 + [C.c_uint64] * 3 + [C.c_bool, C.POINTER(Layout)]
    fn.restype = C.c_bool
    out = Layout()
    assert fn(256, split, 8, 512, 1040, 12, 8, device, C.byref(out))
    assert out.shapes == 1040
    assert out.outputs == 1040 + 256 * 12
    assert out.strides == out.outputs + 256 * split * 8
    assert out.rows == out.strides + 256 * split * 8
    assert out.partials == out.rows + (256 * 4 if device else 0)
    assert out.partial_bytes == (8 * 512 * split * 4 if split > 1 else 0)
    assert out.total == out.partials + out.partial_bytes
    if split == 1:
        assert out.total == (9232 if device else 8208)
    for field, _ in Layout._fields_:
        assert getattr(out, field) % 16 == 0
    assert not fn(256, 3, 8, 512, 1040, 12, 8, device, C.byref(out))
    assert not fn(256, 8, 2**31 - 1, 2**31 - 1, 1040, 12, 8, device, C.byref(out))
    assert not fn(0, 1, 8, 512, 1040, 12, 8, device, C.byref(out))


def test_expert_descriptor_slots_cover_only_real_partial_rows():
    rows = [9, 0, 3, 1]
    offsets = np.r_[0, np.cumsum(rows)]
    n, m, split = 64, sum(rows), 4
    owners = np.zeros((split, m, n), dtype="i4")
    descriptors = [None] * (len(rows) * split)
    for s in range(split):
        for e, count in enumerate(rows):
            descriptors[e + s * len(rows)] = (s * m + offsets[e]) * n
            owners[s, offsets[e] : offsets[e + 1], :] += 1
    assert np.all(owners == 1)
    assert descriptors[2 + 3 * 4] == (3 * m + 9) * n
    assert descriptors[2 + 3 * 4] != (2 * split + 3) * n


def test_parent_plan_keeps_incumbents_tm8_and_no_new_tactic_authority():
    parents = builder.parents()
    assert len(parents) == 11
    assert {p["qtype"] for p in parents} == {12, 13}
    for q in (12, 13):
        subset = [p for p in parents if p["qtype"] == q]
        assert any(p["tm"] == 16 and p["tn"] == 64 and p["dn"] == 64 for p in subset)
        assert any(p["tm"] == 8 for p in subset)
        assert all(
            p["persistent"] == 0 and p["ap"] == 0 and p["tk"] == 256 for p in subset
        )


def test_partial_oracle_is_strided_logical_k_and_rejects_plane_loss():
    w = IndexedWeights(
        12, 256, 512, 4, partial_specs=[(256, 2)], partial_experts=range(4)
    )
    data = sweep.grouped_data(w, [9, 0, 3, 1])
    gold = sweep.partial_gold(w, data, 256, 2)
    assert np.allclose(gold["golden"].sum(0), data["golden"], atol=1e-10)
    partial = gold["golden"].astype("f4")
    out = np.add(partial[0], partial[1], dtype=np.float32).astype("f2")
    order = np.arange(13)
    assert sweep.check_partials(partial.tobytes(), out, w, data, 256, 2, order) < 0.005
    wrong = partial.copy()
    wrong[1].fill(0)
    with pytest.raises(ValueError):
        sweep.check_partials(wrong.tobytes(), out, w, data, 256, 2, order)
    with pytest.raises(ValueError):
        sweep.check_partials(partial.tobytes(), out + 1, w, data, 256, 2, order)
    with pytest.raises(ValueError):
        sweep.check_partials(partial[::-1].tobytes(), out, w, data, 256, 2, order)


def test_pair_extension_does_not_widen_scalar_abi(host):
    c = Call(
        version=1,
        size=C.sizeof(Call),
        qtype=12,
        n=512,
        k=2048,
        experts=256,
        rows=8,
        mode=2,
        input_type=1,
        channels=1,
        topk=8,
        a_row_stride=2048,
        a_token_stride=2048,
        ids_stride=8,
        out_row_stride=512,
    )
    f = Config(16, 2, 2)
    arr = arrangement(12)
    out = Sizes()
    host.qkg_host_pair_query.argtypes = host.qkg_host_query.argtypes
    assert host.qkg_host_query(C.byref(c), C.byref(f), C.byref(arr), C.byref(out)) != 0
    assert (
        host.qkg_host_pair_query(C.byref(c), C.byref(f), C.byref(arr), C.byref(out))
        == 0
    )
    assert out.workspace_bytes == 8 * 512 * 2 * 4
    f.warps = 3
    assert (
        host.qkg_host_pair_query(C.byref(c), C.byref(f), C.byref(arr), C.byref(out))
        != 0
    )


def test_job_denominator():
    manifest = dict(modules=[dict(parent=p) for p in builder.parents()])
    assert len(sweep.jobs(manifest)) == 16
    assert (
        sum(sweep.expected_cells(job, manifest) for job in sweep.jobs(manifest)) == 260
    )


@pytest.mark.parametrize(
    "arm,gpu_directory,grid_b",
    [
        ("host-compact", False, 0),
        ("device-only", False, 0),
        ("device-only", True, 0),
        ("device-only", True, 1),
        ("device-only", True, 2),
    ],
)
@pytest.mark.parametrize("split", [1, 2])
def test_gemm_driver_mutable_router_and_fp32_partial_boundary(
    monkeypatch, arm, split, gpu_directory, grid_b
):
    w = IndexedWeights(
        12, 256, 512, 4, partial_specs=[(256, 2)], partial_experts=range(4)
    )
    profiles = [sweep.grouped_data(w, rows) for rows in ([9, 0, 3, 1], [0, 3, 1, 9])]
    memory = []
    captured = {}

    class SDK:
        def synchronize(self, stream):
            pass

        def fill(self, p, value, count):
            C.memset(p, value, count)

        def download(self, p, count):
            return C.string_at(p, count)

        lib = SimpleNamespace(
            hggcMemcpy=lambda dst, src, size, kind: (C.memmove(dst, src, size), 0)[1]
        )

    class Resources:
        def __init__(self, sdk):
            self.stream = C.c_void_p(1)

        def alloc(self, size):
            buf = C.create_string_buffer(size)
            memory.append(buf)
            return C.addressof(buf)

        def upload(self, array):
            pointer = self.alloc(array.nbytes)
            C.memmove(pointer, array.ctypes.data, array.nbytes)
            return pointer

        def close(self):
            pass

        def samples(self, fn, count):
            for _ in range(count):
                assert fn() == 0
            return [160.0] * count

    def query(cp, rp, qp):
        c = C.cast(cp, C.POINTER(sweep.Call)).contents
        r = C.cast(rp, C.POINTER(sweep.Recipe)).contents
        q = C.cast(qp, C.POINTER(sweep.Query)).contents
        q.workspace_bytes = 512 + (c.m * c.n * r.split * 4 if r.split > 1 else 0)
        q.shared_bytes = 38912
        q.occupancy = 10
        return 0

    def prepare(cp, rp, hp):
        c = C.cast(cp, C.POINTER(sweep.Call)).contents
        captured["call"] = sweep.Call.from_buffer_copy(c)
        C.cast(hp, C.POINTER(C.c_void_p))[0] = C.c_void_p(1)
        return 0

    def query2(dp, rp, qp):
        d = C.cast(dp, C.POINTER(sweep.DeviceCall)).contents
        assert not d.call.rows_host and not d.call.rows_device and d.max_rows == 9
        return query(C.byref(d.call), rp, qp)

    def prepare2(dp, rp, hp):
        d = C.cast(dp, C.POINTER(sweep.DeviceCall)).contents
        return prepare(C.byref(d.call), rp, hp)

    class Module:
        record = dict(parent=dict(symbol="frozen-parent", tk=256, tm=8, tn=64))
        lib = SimpleNamespace(
            quactlize_kpack_grouped_query_v2=query2,
            quactlize_kpack_grouped_prepare_v2=prepare2,
        )

        def device_identity(self):
            return dict(ordinal=0, compute_units=72)

        def destroy(self, handle):
            pass

        def run(self, handle, stream):
            c = captured["call"]
            bounds = np.frombuffer(C.string_at(c.offsets_device, 20), dtype="i4")
            data = sweep.grouped_data(w, np.diff(bounds))
            if gpu_directory:
                rows = np.diff(bounds)
                prefix = np.r_[0, np.cumsum((rows + 7) // 8)]
                header = np.array([prefix[-1], 0, 8, 4], dtype="<i4")
                entries = np.array(
                    [
                        [e, rows[e], prefix[e], bounds[e]]
                        for e in range(4)
                        for _ in range(int(prefix[e + 1] - prefix[e]))
                    ],
                    dtype="<i4",
                )
                C.memmove(c.workspace, header.ctypes.data, header.nbytes)
                C.memmove(c.workspace + 16, entries.ctypes.data, entries.nbytes)
            if split == 1:
                out = data["golden"].astype("<f2")
            else:
                planes = sweep.partial_gold(w, data, 256, split)["golden"].astype("<f4")
                C.memmove(
                    c.workspace + c.workspace_bytes - planes.nbytes,
                    planes.ctypes.data,
                    planes.nbytes,
                )
                out = np.add(planes[0], planes[1], dtype="f4").astype("<f2")
            C.memmove(c.output, out.ctypes.data, out.nbytes)
            return 0

    Module.query = staticmethod(query)
    Module.prepare = staticmethod(prepare)

    class Replay:
        def __init__(self, sdk, stream, fn, repeats):
            self.fn = fn

        def __call__(self):
            return self.fn()

        def close(self):
            pass

    monkeypatch.setattr(sweep, "Resources", Resources)
    monkeypatch.setattr(sweep, "Replay", Replay)
    args = SimpleNamespace(
        graph_repeats=16, correctness_repeats=3, warmups=1, samples=3, rounds=2
    )
    result = sweep.gemm_cell(
        args,
        SDK(),
        Module(),
        w,
        profiles,
        split,
        arm,
        gpu_directory=gpu_directory,
        grid_b=grid_b,
    )
    assert result["status"] == "PASS" and result["median_us"] == 10.0
    assert result["profiles_checked"] == (2 if arm == "device-only" else 1)
    assert result["partial_bytes"] == (2 * 13 * 256 * 4 if split == 2 else 0)
    if gpu_directory:
        assert result["directory_capacity"] == 5
        assert result["grid"] == [20 * split, 1, 1]
    c = captured["call"]
    assert np.array_equal(
        np.frombuffer(C.string_at(c.offsets_device, 20), dtype="i4"), [0, 9, 9, 12, 13]
    )

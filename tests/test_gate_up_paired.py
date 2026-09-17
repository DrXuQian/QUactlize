"""Real packer bytes and CuTe register ownership, without a device claim."""

import ctypes as C
from pathlib import Path
import os
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest

from quactlize.fusion.native import Layout, FusionCall, Config, Call, Sizes
from tests.test_kpack_device_pack import library

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def paired(library):
    library.host_pack_paired_n4.argtypes = [
        C.c_int,
        *[C.c_void_p] * 5,
        C.c_int,
        C.c_int,
        C.c_int,
    ]
    library.quactlize_gate_up_layout_v1.argtypes = [C.c_int, C.POINTER(Layout)]
    library.host_gate_up_query.argtypes = [
        C.POINTER(FusionCall),
        C.POINTER(Config),
        C.POINTER(Layout),
        C.POINTER(Sizes),
    ]
    library.host_gate_up_buffers.argtypes = [C.POINTER(FusionCall), C.POINTER(Sizes)]
    return library


@pytest.mark.parametrize("q", [8, 10, 11, 12, 13, 14])
@pytest.mark.parametrize("experts", [1, 3])
def test_paired_word_metadata_bytes_and_expert_slices(paired, q, experts):
    n, k = 256, 512
    block, block_bytes = (
        (32, 34) if q == 8 else (256, {10: 84, 11: 110, 12: 144, 13: 176, 14: 210}[q])
    )
    row_bytes = k // block * block_bytes
    rng = np.random.default_rng(418 + q + experts)
    gate, up = [
        rng.integers(0, 256, (experts, n, row_bytes), dtype="u1") for _ in range(2)
    ]
    # Independent construction: pair whole four-row blocks, not production index functions.
    merged = np.stack(
        (
            gate.reshape(experts, n // 4, 4, row_bytes),
            up.reshape(experts, n // 4, 4, row_bytes),
        ),
        axis=2,
    ).reshape(experts, 2 * n, row_bytes)
    low_bits = {8: 8, 10: 2, 11: 2, 12: 4, 13: 4, 14: 4}[q]
    high_bits = {8: 0, 10: 0, 11: 1, 12: 0, 13: 1, 14: 2}[q]
    count = experts * 2 * n * k
    sizes = [
        count // 8 * low_bits,
        count // 8 * high_bits,
        merged.nbytes - count // 8 * (low_bits + high_bits),
    ]
    outputs = [
        [np.full(size + 32, 0xA5, dtype="u1") for size in sizes] for _ in range(2)
    ]
    views = [[x[16:-16] for x in group] for group in outputs]
    ptr = lambda x: x.ctypes.data if x.size else None
    assert (
        paired.host_pack(q, merged.ctypes.data, *map(ptr, views[0]), 2 * n, k, experts)
        == 0
    )
    assert (
        paired.host_pack_paired_n4(
            q, gate.ctypes.data, up.ctypes.data, *map(ptr, views[1]), n, k, experts
        )
        == 0
    )
    for baseline, candidate in zip(views[0], views[1]):
        assert np.array_equal(baseline, candidate)
    for output in outputs:
        for x in output:
            assert np.all(x[:16] == 0xA5) and np.all(x[-16:] == 0xA5)


def test_actual_cute_full_tile_register_pairs(tmp_path):
    sdk = Path(os.environ.get("PPU_SDK", "/root/ppu-sdk/2.1.1"))
    lib = tmp_path / "ownership.so"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f'-I{ROOT/"third_party/actlize/include"}',
            f'-I{sdk/"include"}',
            str(ROOT / "tests/gate_up_layout_host.cpp"),
            "-o",
            str(lib),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert C.CDLL(str(lib)).gate_up_ownership() == 0


def request(q=8):
    c = Call(
        version=1,
        size=C.sizeof(Call),
        qtype=q,
        n=512,
        k=2048,
        experts=1,
        rows=1,
        mode=0,
        input_type=1,
        channels=1,
        topk=1,
        a_row_stride=2048,
        a_token_stride=2048,
        ids_stride=1,
        out_row_stride=512,
    )
    return FusionCall(c)


@pytest.mark.parametrize("q", [8, 10, 11, 12, 13, 14])
def test_query_sizes_and_no_old_layout_alias(paired, q):
    d = request(q)
    layout = Layout()
    assert paired.quactlize_gate_up_layout_v1(q, C.byref(layout)) == 0
    for f in (
        Config(),
        Config(split=8),
        Config(backend=1, split=8, tile_m=8, warps=0),
        Config(backend=1, tile_m=16, warps=0),
    ):
        out = Sizes()
        assert (
            paired.host_gate_up_query(
                C.byref(d), C.byref(f), C.byref(layout), C.byref(out)
            )
            == 0
        )
        assert out.workspace_bytes == (0 if f.split == 1 else 1 * 8 * 1024 * 4)
    for field in ("version", "size", "layout_id"):
        bad = Layout.from_buffer_copy(layout)
        setattr(bad, field, getattr(bad, field) ^ 1)
        out = Sizes(1, 2, 3, 4, 5)
        before = bytes(out)
        assert (
            paired.host_gate_up_query(
                C.byref(d), C.byref(Config()), C.byref(bad), C.byref(out)
            )
            == 38
        )
        assert bytes(out) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 0),
        ("n", 2147483647),
        ("out_row_stride", 511),
        ("out_row_stride", 2**63 - 1),
        ("a_row_stride", 2049),
    ],
)
def test_invalid_geometry(paired, field, value):
    d = request()
    setattr(d.input.call, field, value)
    layout = Layout()
    assert paired.quactlize_gate_up_layout_v1(8, C.byref(layout)) == 0
    assert (
        paired.host_gate_up_query(
            C.byref(d), C.byref(Config()), C.byref(layout), C.byref(Sizes())
        )
        != 0
    )


def test_public_header_is_c_compatible(tmp_path):
    subprocess.run(
        [
            "cc",
            "-x",
            "c",
            "-std=c11",
            "-fsyntax-only",
            "-I" + str(ROOT),
            "-I" + str(ROOT / "quactlize/include"),
            "-",
        ],
        input='#include "quactlize/fusion/gate_up.h"\n',
        text=True,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("compute,storage", [(0, 2), (1, 0), (2, 1)])
def test_no_silent_precision_change(paired, compute, storage):
    d = request()
    d.input.compute_type = compute
    d.input.call.input_type = storage
    layout = Layout()
    assert paired.quactlize_gate_up_layout_v1(8, C.byref(layout)) == 0
    assert (
        paired.host_gate_up_query(
            C.byref(d), C.byref(Config()), C.byref(layout), C.byref(Sizes())
        )
        != 0
    )


@pytest.mark.parametrize("backend", [0, 1])
def test_device_test_inventory_is_admissible(paired, backend):
    from tools.run_kpack_gate_up import cases, arithmetic, configurations

    layout = Layout()
    assert paired.quactlize_gate_up_layout_v1(8, C.byref(layout)) == 0
    for mode, m, channels, profile in cases("tc" if backend else "simt"):
        for compute, storage, output, rounding in arithmetic():
            d = request()
            c = d.input.call
            c.rows = m * 8 if mode == 2 else m
            c.mode = mode
            c.experts = 1 if mode == 0 else 8
            c.channels = channels
            c.topk = 8 if mode == 2 else 1
            c.ids_stride = 11
            c.a_token_stride = channels * c.a_row_stride
            c.input_type = storage
            d.input.compute_type = compute
            d.output_type = output
            d.round_projection = rounding
            for split in (1, 2, 4, 8):
                for config in configurations("tc" if backend else "simt", split):
                    assert (
                        paired.host_gate_up_query(
                            C.byref(d),
                            C.byref(config),
                            C.byref(layout),
                            C.byref(Sizes()),
                        )
                        == 0
                    )


def test_output_extent_is_logical_not_physical_and_rejects_overlap(paired):
    d = request()
    c = d.input.call
    c.a = 0x1000000
    c.low = 0x2000000
    c.units = 0x3000000
    c.output = 0x4000000
    layout = Layout()
    assert paired.quactlize_gate_up_layout_v1(8, C.byref(layout)) == 0
    sizes = Sizes()
    assert (
        paired.host_gate_up_query(
            C.byref(d), C.byref(Config(split=8)), C.byref(layout), C.byref(sizes)
        )
        == 0
    )
    c.workspace = 0x4000000 + c.n * 4
    c.workspace_bytes = sizes.workspace_bytes
    assert paired.host_gate_up_buffers(C.byref(d), C.byref(sizes)) == 0
    c.workspace -= 4
    assert paired.host_gate_up_buffers(C.byref(d), C.byref(sizes)) != 0
    c.workspace = 0x5000000
    c.workspace_bytes = sizes.workspace_bytes - 1
    assert paired.host_gate_up_buffers(C.byref(d), C.byref(sizes)) != 0
    c.workspace_bytes = sizes.workspace_bytes
    c.low += 2
    assert paired.host_gate_up_buffers(C.byref(d), C.byref(sizes)) != 0


class HostMemory:
    """Only test the Python oracle and guards, never emulate a GPU verdict."""

    def __init__(self):
        self.allocations = []
        self.stream = C.c_void_p()

    def allocate(self, n):
        a = np.empty(max(n, 1), dtype="u1")
        self.allocations.append(a)
        return a.ctypes.data

    def copy(self, p, a):
        a = np.ascontiguousarray(a)
        C.memmove(p, a.ctypes.data, a.nbytes)

    def upload(self, a):
        if not a.size:
            return None
        p = self.allocate(a.nbytes)
        self.copy(p, a)
        return p

    def fill(self, p, n, byte=0xA5):
        C.memset(p, byte, n)

    def download(self, p, n):
        return np.frombuffer(C.string_at(p, n), dtype="u1").copy()

    def sync(self):
        pass

    def release_after(self, n):
        del self.allocations[n:]


@pytest.mark.parametrize("q", [8, 10, 11, 12, 13, 14])
def test_device_fixture_oracle_guards_and_replay_reset_on_host(paired, q):
    from tools.run_kpack_gate_up import Weights, Bench

    def layout(q):
        result = Layout()
        assert paired.quactlize_gate_up_layout_v1(q, C.byref(result)) == 0
        return result

    def pack(g, u, l, h, units, n, k, e, q, arr, stream):
        return paired.host_pack_paired_n4(q, g, u, l, h, units, n, k, e)

    rt = HostMemory()
    lib = SimpleNamespace(arrangement=layout, pack=pack)
    w = Weights(rt, lib, q, 256, 512, 8)
    for mode, m, channels, profile in ((1, 17, 1, "single"), (2, 2, 8, "indexed")):
        bench = Bench(rt, lib, w, mode, m, 1, 1, 1, 1, channels, profile)
        bench.update(3, large=True)
        assert np.max(np.abs(bench.host_a)) > 65504
        bench.poison()
        host = rt.download(bench.output, bench.output_bytes)
        host[16:-16].view("<f4").reshape(bench.rows, w.n + 8)[:, : w.n] = bench.gold
        rt.copy(bench.output, host)
        assert bench.check(1) == 0
        rt.fill(bench.output + 16, 4, 0)
        host[16:-16].view("<f4").reshape(bench.rows, w.n + 8)[:, : w.n] = 0
        rt.copy(bench.output, host)
        with pytest.raises(ValueError, match="independent GGUF"):
            bench.check(1)
        bench.update(0)
        assert np.max(np.abs(bench.host_a)) < 1
        assert np.array_equal(
            rt.download(bench.a, bench.host_a.nbytes).view("<f4"),
            bench.host_a.reshape(-1),
        )
        bench.close()


def test_resume_requires_unique_complete_numeric_and_replay_evidence():
    from tools.run_kpack_gate_up import (
        cases,
        arithmetic,
        configurations,
        RECORD_KEYS,
        complete_part,
    )

    records = [
        dict(
            zip(RECORD_KEYS, (*case, *types, split, config.tile_m, config.warps)),
            error=0.0,
        )
        for case in cases("simt")
        for types in arithmetic()
        for split in (1, 2, 4, 8)
        for config in configurations("simt", split)
    ]
    part = dict(
        status="PASS",
        q=12,
        backend="simt",
        identity={"hash": "fixture"},
        records=records,
        replays=80,
        negative_controls=32,
    )
    assert complete_part(part, 12, "simt", part["identity"])
    assert not complete_part(part, 12, "simt", {"hash": "other"})
    original = records[-1]
    records[-1] = records[0]
    assert not complete_part(part, 12, "simt", part["identity"])
    records[-1] = original
    original["error"] = float("nan")
    assert not complete_part(part, 12, "simt", part["identity"])
    original["error"] = 0.0
    part["negative_controls"] = 0
    assert not complete_part(part, 12, "simt", part["identity"])

"""Local reader/size proofs. No device timing or device admission."""

import ctypes as C
from pathlib import Path
import os
import subprocess

import gguf
import numpy as np
import pytest
import torch

from reference import gguf_kpack as ref
from quactlize.execution.native import Call, Config, Sizes, Arrangement, arrangement

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    sdk = Path(os.environ.get("PPU_SDK", "/root/ppu-sdk/2.1.1"))
    target = tmp_path_factory.mktemp("execution-host") / "host.so"
    result = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{ROOT/'quactlize/execution'}",
            f"-I{ROOT/'quactlize/include'}",
            f"-I{ROOT/'third_party/actlize/include'}",
            f"-I{sdk/'include'}",
            str(ROOT / "tests/kpack_execution_host.cpp"),
            "-o",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lib = C.CDLL(str(target))
    lib.qkg_host_read.argtypes = (
        [C.c_int] + [C.c_void_p] * 3 + [C.c_int] * 3 + [C.c_void_p] * 3
    )
    lib.qkg_host_query.argtypes = [
        C.POINTER(Call),
        C.POINTER(Config),
        C.POINTER(Arrangement),
        C.POINTER(Sizes),
    ]
    return lib


def raw_fixture(q, n, k, experts):
    spec = ref.SPECS[q]
    rng = np.random.default_rng(81203 + q)
    raw = rng.integers(
        0, 256, (experts * n * (k // 256), spec.raw_bytes), dtype=np.uint8
    )
    for offset in (spec.d_offset, spec.dmin_offset):
        if offset >= 0:
            raw[:, offset : offset + 2] = (
                rng.uniform(0.001, 0.009, len(raw))
                .astype("<f2")
                .view("u1")
                .reshape(-1, 2)
            )
    return raw


@pytest.mark.parametrize("q", range(10, 15))
def test_independent_offline_bytes_and_affine_reader(host, q):
    n, k, e = 256, 512, 3
    raw = raw_fixture(q, n, k, e)
    art = ref.prepare_grouped(torch.from_numpy(raw), n, k, q, e)
    w = np.zeros((e, n, k), dtype=np.float16)
    s = np.zeros((e, k // ref.SPECS[q].group_size, n), dtype=np.float16)
    z = np.zeros_like(s)
    pointers = [
        x.data_ptr() if x.numel() else None for x in (art.low, art.high, art.units)
    ]
    assert (
        host.qkg_host_read(
            q, *pointers, n, k, e, w.ctypes.data, s.ctypes.data, z.ctypes.data
        )
        == 0
    )
    official = gguf.quants.dequantize(
        raw.reshape(-1), gguf.GGMLQuantizationType(q)
    ).reshape(e, n, k)
    assert np.isfinite(w).all() and np.isfinite(s).all() and np.isfinite(z).all()
    from tools.run_kpack_gemv_gate import metadata_oracle

    want_s, want_z = metadata_oracle(art.units.numpy(), q, n, k, e)
    assert np.array_equal(s.view("u2"), want_s.view("u2"))
    assert np.array_equal(z.view("u2"), want_z.view("u2"))
    assert np.max(np.abs(w.astype("f4") - official)) / np.max(np.abs(official)) < 0.005
    # Signed A and distinct expert rows test numeric impact instead of unstable
    # element-relative division at weights near zero.
    a = (
        np.random.default_rng(73 + q)
        .normal(0, 0.2, (e, 7, k))
        .astype("f2")
        .astype("f8")
    )
    denom = np.abs(a) @ np.abs(official.astype("f8")).transpose(0, 2, 1)
    error = np.abs(a @ (w.astype("f8") - official.astype("f8")).transpose(0, 2, 1))
    assert np.max(error / np.maximum(denom, 1e-30)) < 0.005
    # Replacing a code plane must be visible to this exact reader.
    badlow = torch.zeros_like(art.low)
    old = w.copy()
    host.qkg_host_read(
        q,
        badlow.data_ptr(),
        pointers[1],
        pointers[2],
        n,
        k,
        e,
        w.ctypes.data,
        s.ctypes.data,
        z.ctypes.data,
    )
    assert np.count_nonzero(w != old) > w.size // 2


def call(q=12, mode=0, e=1):
    return Call(
        version=1,
        size=C.sizeof(Call),
        qtype=q,
        n=256,
        k=512,
        experts=e,
        rows=8,
        mode=mode,
        input_type=0,
        channels=1,
        topk=1,
        a_row_stride=512,
        a_token_stride=512,
        ids_stride=1,
        out_row_stride=256,
    )


@pytest.mark.parametrize("q", range(10, 15))
def test_sizes_and_device_only_query(host, q):
    c = call(q, 1, 256)
    f = Config(split=4)
    s = Sizes()
    spec = ref.SPECS[q]
    assert (
        host.qkg_host_query(C.byref(c), C.byref(f), C.byref(arrangement(q)), C.byref(s))
        == 0
    )
    assert s.low_bytes == 256 * 256 * 512 * spec.low_bits // 8
    assert s.high_bytes == 256 * 256 * 512 * spec.high_bits // 8
    assert s.units_bytes == 256 * 256 * 2 * spec.sb_bytes
    assert s.sf_plane_bytes == 2 * 256 * 256 * 512 // spec.group_size
    assert s.workspace_bytes == 8 * 256 * 4 * 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("size", 0),
        ("qtype", 9),
        ("n", 255),
        ("k", 128),
        ("rows", 0),
        ("experts", 0),
        ("input_type", 3),
        ("mode", 4),
        ("a_row_stride", 511),
        ("out_row_stride", 255),
        ("topk", 2),
        ("channels", 2),
        ("a_row_stride", 2**63 - 1),
    ],
)
def test_invalid_query(host, field, value):
    c = call()
    setattr(c, field, value)
    f = Config()
    s = Sizes()
    assert (
        host.qkg_host_query(
            C.byref(c), C.byref(f), C.byref(arrangement(12)), C.byref(s)
        )
        != 0
    )


def test_indexed_strides_and_canonical_identity(host):
    c = call(13, 2, 256)
    c.topk = 8
    c.ids_stride = 10
    c.a_token_stride = 512
    f = Config()
    s = Sizes()
    a = arrangement(13)
    assert host.qkg_host_query(C.byref(c), C.byref(f), C.byref(a), C.byref(s)) == 0
    c.channels = 8
    assert host.qkg_host_query(C.byref(c), C.byref(f), C.byref(a), C.byref(s)) != 0
    c.a_token_stride = 4096
    assert host.qkg_host_query(C.byref(c), C.byref(f), C.byref(a), C.byref(s)) == 0
    a.mapping_id ^= 1
    assert host.qkg_host_query(C.byref(c), C.byref(f), C.byref(a), C.byref(s)) == 38


def test_bounded_gate_inventory_and_routing():
    from tools.run_kpack_gemv_gate import plan, fixture, CONFIGS
    from tools.kpack_execution_fixture import IndexedWeights

    assert len(CONFIGS) == len(set(CONFIGS)) == 8
    assert len(plan(False)) == 5 and len(plan(True)) == 7
    assert sum(len(x["cases"]) for x in plan(True)) == 31
    w = IndexedWeights(13, 256, 512, 4)
    assert np.all(w.categories == w.categories[0])
    for case in plan(False)[3]["cases"]:
        d = fixture(w, case)
        assert d["golden"].shape == (case["rows"], 256)
        assert np.all(np.isfinite(d["golden"])) and np.all(d["denom"] > 0)
        if case["mode"] == 1:
            assert list(d["offsets"]) == [0, 2, 2, 5, 6]
        if case["mode"] == 2:
            ids = d["ids"]
            assert all(len(set(r)) == case["topk"] for r in ids)
            for i, e in enumerate(d["expert"]):
                assert e == ids[i // case["topk"], i % case["topk"]]
                assert (
                    d["arows"][i]
                    == i // case["topk"] * case["channels"]
                    + i % case["topk"] % case["channels"]
                )

"""Host-only contract tests for the versioned fully-quantized artifact descriptor.

These deliberately replace the torch ops with recording callables. They test the Python ABI seam -- descriptor
ownership, validation and forwarding -- without pretending a host mock proves a PPU kernel accepts an arrangement.
The device-side descriptor x tactic predicate has its own compiled controls.
"""
import importlib.util
import copy
import pickle
import pathlib
import re
import os
import subprocess
import sys

import pytest
import torch

from quactlize import formats, routes

_PACK_SPEC = importlib.util.spec_from_file_location(
    "pack_gguf_artifact_abi_test", pathlib.Path(__file__).parents[1] / "tools" / "pack_gguf.py")
pack_gguf = importlib.util.module_from_spec(_PACK_SPEC)
_PACK_SPEC.loader.exec_module(pack_gguf)


def _planes(arrangement, n=2, k=256):
    low = torch.zeros((1, n, k * arrangement.bits // 8), dtype=torch.uint8)
    high = (torch.zeros((1, n, k * arrangement.high_bits // 8), dtype=torch.uint8)
            if arrangement.high_bits else torch.empty((0,), dtype=torch.uint8))
    units = torch.zeros((1, n, 16), dtype=torch.uint8)
    return low, high, units


def _artifact(qtype=formats.QuantType.Q2_K, tile_k=64, version=routes.PLACED_ARTIFACT_VERSION):
    arrangement = formats.placed_arrangement(qtype, tile_k)
    return routes.PlacedArtifact(_planes(arrangement), arrangement, version)


def test_format_plane_descriptor_matches_the_shipping_cpp_registry():
    """The Python format fact and C++ X-macro must not become two plausible descriptors."""
    inc = pathlib.Path(formats.__file__).parent / "include" / "ppu_format_config.inc"
    rows = {}
    for m in re.finditer(r"^\s*X\((.*?)\)\s*\\?\s*$", inc.read_text(), re.M):
        fields = [x.strip().strip('"') for x in m.group(1).split(",")]
        rows[int(fields[2])] = (int(fields[3]), int(fields[4]), int(fields[6]))
    assert rows, "the registry parser found no X(...) rows -- a vacuous comparison is not a contract"
    for qtype in formats.PLACED_CODE_PLANES:
        low, high = formats.placed_code_planes(qtype)
        assert rows[int(qtype)] == (low, high, formats.placed_arrangement(qtype).tile_k)


def test_dense_producer_attaches_the_exact_arrangement(monkeypatch):
    calls = []
    arrangement = formats.placed_arrangement(formats.QuantType.Q2_K, 64)

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            return _planes(arrangement)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    blocks = torch.zeros((256, 84), dtype=torch.uint8)
    artifact = routes.prepare_fully_quantized_dense(blocks, 256, 256, formats.QuantType.Q2_K, tile_k=64)
    assert isinstance(artifact, routes.PlacedArtifact)
    assert artifact.arrangement_version == routes.PLACED_ARTIFACT_VERSION
    assert artifact.arrangement == arrangement
    assert artifact.requested_tile_k == arrangement.tile_k  # compatibility view, not a second stored field
    assert calls[0][0] == "gguf_prepare_fully_quantized_dense_for_tile"
    assert calls[0][1][-1] == arrangement.tile_k


@pytest.mark.parametrize("reader", ["dense", "bc", "dequant"])
def test_tuple_stripping_and_unknown_versions_fail_before_reader_dispatch(monkeypatch, reader):
    called = []
    monkeypatch.setattr(routes, "_op", lambda name: lambda *args: called.append((name, args)))
    artifact = _artifact()
    a = torch.zeros((1, 256), dtype=torch.float16)

    def read(x):
        if reader == "dense":
            return routes.matmul_fully_quantized_dense(a, x, formats.QuantType.Q2_K)
        if reader == "bc":
            return routes.matmul_bc_gemv(a, x, formats.QuantType.Q2_K)
        return routes.dequantize_fully_quantized(x, formats.QuantType.Q2_K)

    with pytest.raises(TypeError, match="strips.*bits,tile_k,high_bits"):
        read(tuple(artifact))
    with pytest.raises(ValueError, match="unsupported placed-artifact descriptor version"):
        read(routes.PlacedArtifact(artifact, artifact.arrangement, version=99))
    assert called == []


@pytest.mark.parametrize("reader", ["dense", "bc", "dequant"])
def test_qtype_descriptor_mismatch_fails_before_reader_dispatch(monkeypatch, reader):
    called = []
    monkeypatch.setattr(routes, "_op", lambda name: lambda *args: called.append((name, args)))
    artifact = _artifact(formats.QuantType.Q2_K, 64)
    a = torch.zeros((1, 256), dtype=torch.float16)
    with pytest.raises(ValueError, match=r"Q4_K requires 4\+0"):
        if reader == "dense":
            routes.matmul_fully_quantized_dense(a, artifact, formats.QuantType.Q4_K)
        elif reader == "bc":
            routes.matmul_bc_gemv(a, artifact, formats.QuantType.Q4_K)
        else:
            routes.dequantize_fully_quantized(artifact, formats.QuantType.Q4_K)
    assert called == []


def test_folded_descriptor_reaches_dense_and_bc_readers_without_a_caller_tile_argument(monkeypatch):
    calls = []
    # Q3/TK64 is the shipping folded tensor-reader control: low/high folds are 2/4 and the two-plane collective
    # has the packed-metadata seam needed to consume it.  Single-plane Q2/Q4 F>1 remains explicitly fail-closed
    # in l138 until that collective gets the equivalent metadata staging.
    artifact = _artifact(formats.QuantType.Q3_K, 64)
    assert artifact.arrangement.fold == 2
    assert artifact.arrangement.high_fold == 4

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            return torch.empty((1, 2), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    a = torch.zeros((1, 256), dtype=torch.float16)
    routes.matmul_fully_quantized_dense(a, artifact, formats.QuantType.Q3_K)
    routes.matmul_bc_gemv(a, artifact, formats.QuantType.Q3_K)
    want_tail = (int(formats.QuantType.Q3_K), routes.PLACED_ARTIFACT_VERSION, 2, 64, 1)
    assert calls[0][0] == "gguf_dense_fully_quantized_for_arrangement"
    assert calls[1][0] == "gguf_gemv_bc_for_arrangement"
    assert calls[0][1][-5:] == want_tail
    assert calls[1][1][-5:] == want_tail


def test_q2_f2_descriptor_reaches_the_bc_reader_and_not_the_legacy_f1_contract(monkeypatch):
    """Q2/A64 is the required single-plane F2 BC arm; its reader identity comes only from the artifact."""
    calls = []
    artifact = _artifact(formats.QuantType.Q2_K, 64)
    assert artifact.arrangement.fold == 2

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            return torch.empty((1, 2), dtype=torch.float32)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    routes.matmul_bc_gemv(torch.zeros((1, 256), dtype=torch.float16), artifact, formats.QuantType.Q2_K)
    assert len(calls) == 1 and calls[0][0] == "gguf_gemv_bc_for_arrangement"
    assert calls[0][1][-5:] == (int(formats.QuantType.Q2_K), routes.PLACED_ARTIFACT_VERSION, 2, 64, 0)


def test_folded_descriptor_selects_the_existing_tile_aware_inverse(monkeypatch):
    calls = []
    artifact = _artifact(formats.QuantType.Q2_K, 64)

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            if name == "gguf_packed_scale_prepass":
                low = artifact[0]
                # [experts, scale_k, n], enough for the route contract mock.
                return (torch.ones((1, low.shape[2] * 4 // 256, low.shape[1]), dtype=torch.float16),) * 2
            return torch.empty((1, 2, 256), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    routes.dequantize_fully_quantized(artifact, formats.QuantType.Q2_K)
    assert calls[-1][0] == "gguf_dense_artifact_dequantize_for_tile"
    assert calls[-1][1][-1] == 64


def test_f2_artifact_cannot_silently_reach_an_f1_only_reader(monkeypatch):
    """The descriptor makes backend incompatibility an explicit rejection, rather than an implicit F=1 decode."""
    artifact = _artifact(formats.QuantType.Q3_K, 64)

    def fake_op(name):
        def f1_only(*args):
            _qtype, _version, bits, tile_k, _high_bits = args[-5:]
            if formats.fold_for(bits, tile_k) != 1:
                raise RuntimeError("compiled reader accepts F=1; artifact records F=2")
            return torch.empty((1, 1), dtype=torch.float16)
        return f1_only

    monkeypatch.setattr(routes, "_op", fake_op)
    with pytest.raises(RuntimeError, match="artifact records F=2"):
        routes.matmul_fully_quantized_dense(
            torch.zeros((1, 256), dtype=torch.float16), artifact, formats.QuantType.Q3_K)


@pytest.mark.parametrize("reader", ["bc_moe", "grouped_tensor"])
def test_grouped_readers_reject_a_descriptor_their_legacy_ops_cannot_carry(monkeypatch, reader):
    called = []
    monkeypatch.setattr(routes, "_op", lambda name: lambda *args: called.append((name, args)))
    artifact = _artifact(formats.QuantType.Q3_K, 64)
    rows = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError, match="grouped.*descriptor|descriptor.*grouped"):
        if reader == "bc_moe":
            routes.matmul_bc_gemv_moe(
                torch.zeros((1, 256), dtype=torch.float16), artifact, formats.QuantType.Q3_K, 1, rows)
        else:
            routes.matmul_fully_quantized_grouped(
                torch.zeros((1, 256), dtype=torch.float16), artifact, formats.QuantType.Q3_K, rows)
    assert called == []


def test_manifest_roundtrip_restores_descriptor_and_rejects_legacy_or_future_versions(tmp_path):
    artifact = _artifact(formats.QuantType.Q2_K, 64)
    tensor_dir = tmp_path / "weight"
    tensor_dir.mkdir()
    pack_gguf._write(tensor_dir, *artifact)
    record = {
        "name": "blk.0.weight",
        "dir": "weight",
        "ggml_type": int(formats.QuantType.Q2_K),
        "arrangement_version": artifact.arrangement_version,
        "arrangement": artifact.arrangement._asdict(),
    }
    restored = pack_gguf.restore_artifact(tmp_path, record)
    assert isinstance(restored, routes.PlacedArtifact)
    assert restored.arrangement == artifact.arrangement
    assert restored.arrangement_version == artifact.arrangement_version
    assert all(torch.equal(a, b) for a, b in zip(restored, artifact))

    for bad_version in (None, routes.PLACED_ARTIFACT_VERSION + 1):
        bad = dict(record, arrangement_version=bad_version)
        with pytest.raises(ValueError, match="arrangement_version"):
            pack_gguf.restore_artifact(tmp_path, bad)

    bad_arrangement = dict(record, arrangement={"bits": 4, "tile_k": 64, "high_bits": 0})
    with pytest.raises(ValueError, match="disagrees with ggml_type"):
        pack_gguf.restore_artifact(tmp_path, bad_arrangement)


def test_copy_pickle_and_identity_keep_descriptor_attached():
    folded = _artifact(formats.QuantType.Q3_K, 64)
    unfolded = routes.PlacedArtifact(folded, formats.placed_arrangement(formats.QuantType.Q3_K, 256))
    for restored in (copy.copy(folded), copy.deepcopy(folded), pickle.loads(pickle.dumps(folded))):
        assert isinstance(restored, routes.PlacedArtifact)
        assert restored == folded
        assert restored.arrangement == folded.arrangement
        assert restored.arrangement_version == folded.arrangement_version
    assert folded != unfolded, "artifact identity must include its placement, not only the three tensor values"
    with pytest.raises(TypeError):
        hash(folded)


def test_zero_m_cannot_bypass_the_compiled_arrangement_predicate(tmp_path):
    """Exercise the real torch/dlsym seam: an empty output is not permission to reinterpret unsupported bytes."""
    root = pathlib.Path(__file__).parents[1]
    if not routes.has_op("gguf_dense_fully_quantized_for_arrangement"):
        pytest.skip("the local extension is not built; setup.py build_ext enables this dlsym contract test")
    fake = root / "dev" / "fold_derivation" / "l140_fake_ppu_backend.cpp"
    common = [
        os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-shared", "-fPIC",
        f"-I{root / 'quactlize' / 'include'}", "-DL140_BACKEND_MARKER=140",
    ]
    q3, q4, predicate_only = (
        tmp_path / "fmt3.so", tmp_path / "fmt0.so", tmp_path / "fmt3_predicate_only.so")
    for output, defs in (
        (q3, ["-DL140_ACCEPT_QTYPE=11", "-DL140_ACCEPT_BITS=2", "-DL140_ACCEPT_TILE_K=64",
              "-DL140_ACCEPT_HIGH_BITS=1"]),
        (q4, ["-DL140_ACCEPT_QTYPE=12", "-DL140_ACCEPT_BITS=4", "-DL140_ACCEPT_TILE_K=64",
              "-DL140_ACCEPT_HIGH_BITS=0"]),
    ):
        built = subprocess.run(common + defs + [str(fake), "-o", str(output)], capture_output=True, text=True)
        assert built.returncode == 0, built.stdout + built.stderr
    built = subprocess.run(
        common + ["-DL140_ACCEPT_QTYPE=11", "-DL140_ACCEPT_BITS=2", "-DL140_ACCEPT_TILE_K=64",
                  "-DL140_ACCEPT_HIGH_BITS=1", "-DL140_OMIT_ARRANGEMENT_READER=1",
                  str(fake), "-o", str(predicate_only)],
        capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    code = r''' 
import torch
from quactlize import formats, routes

def artifact(qtype, tk, k):
    a = formats.placed_arrangement(qtype, tk)
    low = torch.zeros((1, 256, k * a.bits // 8), dtype=torch.uint8)
    high = (torch.zeros((1, 256, k * a.high_bits // 8), dtype=torch.uint8)
            if a.high_bits else torch.empty((0,), dtype=torch.uint8))
    units = torch.zeros((k // (512 if int(qtype) == 11 else 256), 256,
                         28 if int(qtype) == 11 else 16), dtype=torch.uint8)
    return routes.PlacedArtifact((low, high, units), a)

empty = torch.zeros((0, 512), dtype=torch.float16)
got = routes.matmul_fully_quantized_dense(empty, artifact(formats.QuantType.Q3_K, 64, 512), formats.QuantType.Q3_K)
assert tuple(got.shape) == (0, 256)
try:
    routes.matmul_fully_quantized_dense(empty, artifact(formats.QuantType.Q4_K, 32, 512), formats.QuantType.Q4_K)
except RuntimeError as exc:
    assert "no compatible compiled reader" in str(exc), exc
else:
    raise AssertionError("zero-M unsupported F2 artifact bypassed the compiled reader predicate")
print("zero-M supported=PASS unsupported-F2=EXPECTED_RED")
'''
    env = dict(os.environ, QUACTLIZE_PPU_LIB_FMT3=str(q3), QUACTLIZE_PPU_LIB_FMT0=str(q4))
    run = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr

    # Fresh process: load_format caches one result per format.  A library which exports only the predicate must not
    # turn zero-M into a vacuous capability claim; the actual reader symbol is part of the same contract.
    missing_reader = r'''
import torch
from quactlize import formats, routes
a = formats.placed_arrangement(formats.QuantType.Q3_K, 64)
artifact = routes.PlacedArtifact((
    torch.zeros((1, 256, 128), dtype=torch.uint8),
    torch.zeros((1, 256, 64), dtype=torch.uint8),
    torch.zeros((1, 256, 28), dtype=torch.uint8)), a)
try:
    routes.matmul_fully_quantized_dense(
        torch.zeros((0, 512), dtype=torch.float16), artifact, formats.QuantType.Q3_K)
except RuntimeError as exc:
    assert "lacks the arrangement-aware ABI" in str(exc), exc
else:
    raise AssertionError("predicate-only library claimed a zero-M arrangement reader")
print("zero-M predicate-only-without-reader=EXPECTED_RED")
'''
    env = dict(os.environ, QUACTLIZE_PPU_LIB_FMT3=str(predicate_only))
    run = subprocess.run(
        [sys.executable, "-c", missing_reader], cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr

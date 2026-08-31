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


def test_q4_kpack4_v2_producer_and_dense_reader_forward_the_exact_byte_map(monkeypatch):
    calls = []
    arrangement = formats.q4_kpack4_arrangement()

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            if name.startswith("gguf_prepare"):
                return _planes(arrangement)
            return torch.empty((1, 2), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    blocks = torch.zeros((2, 144), dtype=torch.uint8)
    artifact = routes.prepare_fully_quantized_dense(
        blocks, 2, 256, formats.QuantType.Q4_K)
    assert artifact.arrangement == arrangement
    assert artifact.arrangement_version == routes.PLACED_ARTIFACT_VERSION_V2
    wire = (routes.PLACED_ARTIFACT_VERSION_V2, *arrangement)
    assert calls[0][0] == "gguf_prepare_fully_quantized_dense_for_arrangement_v2"
    assert calls[0][1][-9:] == wire

    routes.matmul_fully_quantized_dense(
        torch.zeros((1, 256), dtype=torch.float16), artifact, formats.QuantType.Q4_K)
    assert calls[1][0] == "gguf_dense_fully_quantized_for_arrangement_v2"
    assert calls[1][1][-9:] == wire


@pytest.mark.parametrize("qtype,low_bits,high_bits,transport_k,group_size", [
    (formats.QuantType.Q2_K, 2, 0, 128, 16),
    (formats.QuantType.Q3_K, 2, 1, 256, 16),
    (formats.QuantType.Q5_K, 4, 1, 256, 32),
    (formats.QuantType.Q6_K, 4, 2, 128, 16),
])
def test_kquant_kpack_v2_dense_and_grouped_forward_one_exact_descriptor(
        monkeypatch, qtype, low_bits, high_bits, transport_k, group_size):
    """The new byte map is one dense/grouped ABI, not four route-local guesses."""
    calls = []
    arrangement = formats.kquant_kpack_arrangement(qtype)
    assert arrangement == formats.PlacedArrangementV2(
        formats.PLACED_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1,
        low_bits, high_bits, 0, transport_k, group_size, 0,
        formats.KQUANT_KPACK_MAPPING_ID)

    dense_planes = _planes(arrangement)
    grouped_planes = (
        torch.zeros((2, 2, 256 * low_bits // 8), dtype=torch.uint8),
        (torch.zeros((2, 2, 256 * high_bits // 8), dtype=torch.uint8)
         if high_bits else torch.empty((0,), dtype=torch.uint8)),
        torch.zeros((2, 1, 2, 16), dtype=torch.uint8),
    )

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            if name == "gguf_prepare_fully_quantized_dense_for_arrangement_v2":
                return dense_planes
            if name == "gguf_prepare_fully_quantized_grouped_for_arrangement_v2":
                return grouped_planes
            return torch.empty((1, 2), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    block_bytes = formats.BLOCKS[qtype].block_bytes
    dense = routes.prepare_fully_quantized_dense(
        torch.zeros((2, block_bytes), dtype=torch.uint8), 2, 256, qtype,
        layout="kquant-kpack")
    wire = (routes.PLACED_ARTIFACT_VERSION_V2, *arrangement)
    assert dense.arrangement == arrangement
    assert calls[-1][0] == "gguf_prepare_fully_quantized_dense_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    routes.matmul_fully_quantized_dense(
        torch.zeros((1, 256), dtype=torch.float16), dense, qtype)
    assert calls[-1][0] == "gguf_dense_fully_quantized_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    grouped = routes.prepare_fully_quantized_grouped(
        torch.zeros((4, block_bytes), dtype=torch.uint8), 2, 256, qtype, 2,
        layout="kquant-kpack")
    assert grouped.arrangement == arrangement
    assert calls[-1][0] == "gguf_prepare_fully_quantized_grouped_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    routes.matmul_fully_quantized_grouped(
        torch.zeros((1, 256), dtype=torch.float16), grouped, qtype,
        torch.tensor([1, 0], dtype=torch.int32))
    assert calls[-1][0] == "gguf_grouped_fully_quantized_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire


@pytest.mark.parametrize("qtype", [
    formats.QuantType.Q2_K, formats.QuantType.Q3_K,
    formats.QuantType.Q5_K, formats.QuantType.Q6_K,
])
def test_kquant_kpack_rejects_mapping_qtype_and_plane_drift_before_dispatch(
        monkeypatch, qtype):
    called = []
    monkeypatch.setattr(
        routes, "_op", lambda name: lambda *args: called.append((name, args)))
    arrangement = formats.kquant_kpack_arrangement(qtype)
    a = torch.zeros((1, 256), dtype=torch.float16)

    mutated = arrangement._replace(mapping_id=arrangement.mapping_id ^ 1)
    with pytest.raises(ValueError, match="noncanonical k-quant K-pack descriptor"):
        routes.matmul_fully_quantized_dense(
            a, routes.PlacedArtifact(_planes(mutated), mutated), qtype)

    other = next(q for q in (
        formats.QuantType.Q2_K, formats.QuantType.Q3_K,
        formats.QuantType.Q5_K, formats.QuantType.Q6_K) if q != qtype)
    with pytest.raises(ValueError, match="artifact code planes|requires K-pack descriptor"):
        routes.matmul_fully_quantized_dense(
            a, routes.PlacedArtifact(_planes(arrangement), arrangement), other)

    low, high, units = _planes(arrangement)
    if arrangement.high_bits:
        bad_planes = (low, torch.empty((0,), dtype=torch.uint8), units)
        message = "high plane"
    else:
        bad_planes = (low, torch.ones((1,), dtype=torch.uint8), units)
        message = "single-plane"
    with pytest.raises(ValueError, match=message):
        routes.matmul_fully_quantized_dense(
            a, routes.PlacedArtifact(bad_planes, arrangement), qtype)
    assert called == []


def test_canonical_offline_layout_policy_keeps_non_q4_xplane(monkeypatch):
    expected = {
        formats.QuantType.Q2_K: "xplane",
        formats.QuantType.Q3_K: "xplane",
        formats.QuantType.Q4_K: "q4-kpack4",
        formats.QuantType.Q5_K: "xplane",
        formats.QuantType.Q6_K: "xplane",
    }
    assert {q: formats.canonical_fully_quantized_layout(q) for q in expected} == expected
    assert formats.archived_fully_quantized_layouts(formats.QuantType.Q4_K) == \
        frozenset({"xplane"})
    assert all(not formats.archived_fully_quantized_layouts(q)
               for q in expected if q != formats.QuantType.Q4_K)

    calls = []

    def fake_op(name):
        def call(*args):
            qtype = formats.QuantType(args[3])
            arrangement = (formats.q4_kpack4_arrangement()
                           if qtype == formats.QuantType.Q4_K
                           else formats.placed_arrangement(qtype))
            calls.append((qtype, name))
            return _planes(arrangement)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    for qtype, layout in expected.items():
        blocks = torch.zeros((2, formats.BLOCKS[qtype].block_bytes),
                             dtype=torch.uint8)
        artifact = routes.prepare_fully_quantized_dense(blocks, 2, 256, qtype)
        if layout == "q4-kpack4":
            assert artifact.arrangement == formats.q4_kpack4_arrangement()
            assert calls[-1][1] == \
                "gguf_prepare_fully_quantized_dense_for_arrangement_v2"
        else:
            assert artifact.arrangement == formats.placed_arrangement(qtype)
            assert calls[-1][1] == "gguf_prepare_fully_quantized_dense"


def test_q4_kpack4_rejects_mutated_identity_and_unimplemented_bc_before_dispatch(monkeypatch):
    called = []
    monkeypatch.setattr(routes, "_op", lambda name: lambda *args: called.append((name, args)))
    canonical = formats.q4_kpack4_arrangement()
    bad = canonical._replace(mapping_id=canonical.mapping_id ^ 1)
    bad_artifact = routes.PlacedArtifact(_planes(bad), bad)
    a = torch.zeros((1, 256), dtype=torch.float16)
    with pytest.raises(ValueError, match="noncanonical Q4 K-pack4 descriptor"):
        routes.matmul_fully_quantized_dense(a, bad_artifact, formats.QuantType.Q4_K)
    assert called == []

    artifact = routes.PlacedArtifact(_planes(canonical), canonical)
    with pytest.raises(NotImplementedError, match="no CUDA-core/BC reader"):
        routes.matmul_bc_gemv(a, artifact, formats.QuantType.Q4_K)
    assert called == []


def test_q4_kpack4_inverse_uses_the_v2_recovery_op(monkeypatch):
    calls = []
    arrangement = formats.q4_kpack4_arrangement()
    artifact = routes.PlacedArtifact(_planes(arrangement), arrangement)

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            if name == "gguf_packed_scale_prepass":
                return (torch.ones((1, 8, 2), dtype=torch.float16),) * 2
            return torch.empty((1, 2, 256), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    routes.dequantize_fully_quantized(artifact, formats.QuantType.Q4_K)
    assert calls[-1][0] == "gguf_dense_artifact_dequantize_for_arrangement_v2"
    assert calls[-1][1][-9:] == (routes.PLACED_ARTIFACT_VERSION_V2, *arrangement)


def test_q4_kpack4_grouped_producer_reader_and_inverse_keep_one_descriptor(monkeypatch):
    calls = []
    arrangement = formats.q4_kpack4_arrangement()

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            if name.startswith("gguf_prepare"):
                return _planes(arrangement)
            if name == "gguf_packed_scale_prepass":
                return (torch.ones((1, 8, 2), dtype=torch.float16),) * 2
            return torch.empty((1, 2), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    artifact = routes.prepare_fully_quantized_grouped(
        torch.zeros((2, 144), dtype=torch.uint8), 2, 256,
        formats.QuantType.Q4_K, 1)
    wire = (routes.PLACED_ARTIFACT_VERSION_V2, *arrangement)
    assert isinstance(artifact, routes.PlacedArtifact)
    assert artifact.arrangement == arrangement
    assert calls[-1][0] == "gguf_prepare_fully_quantized_grouped_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    routes.matmul_fully_quantized_grouped(
        torch.zeros((1, 256), dtype=torch.float16), artifact,
        formats.QuantType.Q4_K, torch.tensor([1], dtype=torch.int32))
    assert calls[-1][0] == "gguf_grouped_fully_quantized_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    routes.dequantize_fully_quantized(
        artifact, formats.QuantType.Q4_K, grouped=True)
    assert calls[-1][0] == "gguf_grouped_artifact_dequantize_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    bad = arrangement._replace(mapping_id=arrangement.mapping_id ^ 1)
    with pytest.raises(ValueError, match="noncanonical Q4 K-pack4 descriptor"):
        routes.matmul_fully_quantized_grouped(
            torch.zeros((1, 256), dtype=torch.float16),
            routes.PlacedArtifact(_planes(bad), bad), formats.QuantType.Q4_K,
            torch.tensor([1], dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="no grouped CUDA-core/BC reader"):
        routes.matmul_bc_gemv_moe(
            torch.zeros((1, 256), dtype=torch.float16), artifact,
            formats.QuantType.Q4_K, 1, torch.tensor([1], dtype=torch.int32))


def test_legacy_grouped_xplane_reader_loads_the_qtype_selected_binary(tmp_path):
    """Xplane's tuple ABI does not make its packed-scale kernel format-agnostic.

    Q2/Q3/Q5/Q6 still use the legacy grouped wire, but each reader is compiled into its own PPU_PACKED_FORMAT
    image.  Give the default image a failing grouped marker and FMT2 a successful one: this reaches the real
    torch/dlopen seam and proves Q2 is dispatched by qtype rather than accidentally into the default Q4 image.
    """
    root = pathlib.Path(__file__).parents[1]
    if not routes.has_op("gguf_grouped_fully_quantized"):
        pytest.skip("the local extension is not built; setup.py build_ext enables this dlsym contract test")
    fake = root / "dev" / "fold_derivation" / "l140_fake_ppu_backend.cpp"

    def build(name, marker):
        output = tmp_path / name
        built = subprocess.run([
            os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-shared", "-fPIC",
            f"-I{root / 'quactlize' / 'include'}", f"-DL140_BACKEND_MARKER={marker}",
            "-DL140_GROUPED_LEGACY=1", str(fake), "-o", str(output),
        ], capture_output=True, text=True)
        assert built.returncode == 0, built.stdout + built.stderr
        return output

    default = build("default.so", 140)
    q2 = build("fmt2.so", 0)
    code = r'''
import torch
from quactlize import formats, routes

artifact = (
    torch.zeros((1, 256, 128), dtype=torch.uint8),
    torch.empty((0,), dtype=torch.uint8),
    torch.zeros((1, 2, 256, 20), dtype=torch.uint8),
)
out = routes.matmul_fully_quantized_grouped(
    torch.zeros((1, 512), dtype=torch.float16), artifact,
    formats.QuantType.Q2_K, torch.tensor([1], dtype=torch.int32))
assert tuple(out.shape) == (1, 256)
print("legacy-grouped-xplane qtype-selected-format=PASS")
'''
    env = dict(os.environ, QUACTLIZE_PPU_LIB=str(default), QUACTLIZE_PPU_LIB_FMT2=str(q2))
    run = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "qtype-selected-format=PASS" in run.stdout


def test_q4_kpack4_scalefirst_view_reuses_the_v2_bytes_and_hoisted_metadata(monkeypatch):
    calls = []
    arrangement = formats.q4_kpack4_arrangement()
    artifact = routes.PlacedArtifact(_planes(arrangement), arrangement)
    scale_zero = (torch.ones((1, 8, 2), dtype=torch.float16),) * 2

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            return torch.empty((64, 2), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    routes.matmul_scale_first_dense(
        torch.zeros((64, 256), dtype=torch.float16), artifact,
        formats.QuantType.Q4_K, scale_zero=scale_zero)
    assert [name for name, _ in calls] == ["gguf_dense_scale_first_for_arrangement_v2"]
    assert calls[0][1][-9:] == (routes.PLACED_ARTIFACT_VERSION_V2, *arrangement)


def test_kpack4_dense_dispatches_decode_and_prefill_without_hidden_workspace_build(monkeypatch):
    calls = []
    arrangement = formats.q4_kpack4_arrangement()
    artifact = routes.PlacedArtifact(_planes(arrangement), arrangement)

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            return torch.empty((args[0].shape[0], 2), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    routes.matmul_q4_kpack4_dense(
        torch.zeros((8, 256), dtype=torch.float16), artifact)
    assert calls[-1][0] == "gguf_dense_fully_quantized_for_arrangement_v2"

    before = len(calls)
    with pytest.raises(ValueError, match="hoisted scale_workspace"):
        routes.matmul_q4_kpack4_dense(
            torch.zeros((64, 256), dtype=torch.float16), artifact)
    assert len(calls) == before

    workspace = (torch.ones((1, 8, 2), dtype=torch.float16),) * 2
    routes.matmul_q4_kpack4_dense(
        torch.zeros((64, 256), dtype=torch.float16), artifact,
        scale_workspace=workspace)
    assert calls[-1][0] == "gguf_dense_scale_first_for_arrangement_v2"


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


def test_manifest_roundtrip_restores_q4_kpack4_v2_and_rejects_mapping_drift(tmp_path):
    arrangement = formats.q4_kpack4_arrangement()
    artifact = routes.PlacedArtifact(_planes(arrangement), arrangement)
    tensor_dir = tmp_path / "weight"
    tensor_dir.mkdir()
    pack_gguf._write(tensor_dir, *artifact)
    record = {
        "name": "blk.0.weight", "dir": "weight", "ggml_type": int(formats.QuantType.Q4_K),
        "arrangement_version": routes.PLACED_ARTIFACT_VERSION_V2,
        "arrangement": arrangement._asdict(),
    }
    restored = pack_gguf.restore_artifact(tmp_path, record)
    assert restored == artifact
    assert restored.arrangement == arrangement

    planted = dict(record, arrangement=dict(record["arrangement"], mapping_id=arrangement.mapping_id ^ 1))
    with pytest.raises(ValueError, match="noncanonical Q4 K-pack4 descriptor"):
        pack_gguf.restore_artifact(tmp_path, planted)


@pytest.mark.parametrize("qtype", [
    formats.QuantType.Q2_K, formats.QuantType.Q3_K,
    formats.QuantType.Q5_K, formats.QuantType.Q6_K,
])
def test_manifest_roundtrip_restores_kquant_kpack_v2_and_rejects_identity_drift(
        tmp_path, qtype):
    arrangement = formats.kquant_kpack_arrangement(qtype)
    artifact = routes.PlacedArtifact(_planes(arrangement), arrangement)
    tensor_dir = tmp_path / "weight"
    tensor_dir.mkdir()
    pack_gguf._write(tensor_dir, *artifact)
    record = {
        "name": "blk.0.weight", "dir": "weight", "ggml_type": int(qtype),
        "arrangement_version": routes.PLACED_ARTIFACT_VERSION_V2,
        "arrangement": arrangement._asdict(),
    }
    restored = pack_gguf.restore_artifact(tmp_path, record)
    assert restored == artifact
    assert restored.arrangement == arrangement

    planted = dict(
        record,
        arrangement=dict(
            record["arrangement"],
            transport_tile_k=arrangement.transport_tile_k // 2))
    with pytest.raises(ValueError, match="noncanonical k-quant K-pack descriptor"):
        pack_gguf.restore_artifact(tmp_path, planted)


def test_whole_model_packer_all_kpack_policy_covers_every_plane_and_grouped_route():
    supported = {int(formats.QuantType.Q3_K), int(formats.QuantType.Q4_K)}
    q4 = int(formats.QuantType.Q4_K)
    q3 = int(formats.QuantType.Q3_K)

    # The unchanged canonical policy still refuses to erase a non-Q4 Xplane
    # descriptor on rank-three weights.
    assert pack_gguf._packability(q4, 2, supported) == (True, "dense", None)
    assert pack_gguf._packability(q4, 3, supported) == (True, "grouped", None)
    ok, route, why = pack_gguf._packability(q3, 3, supported)
    assert not ok and route is None and "descriptor-aware reader" in why

    expected = {
        formats.QuantType.Q2_K: ("kquant-kpack", 8, 0),
        formats.QuantType.Q3_K: ("kquant-kpack", 8, 16),
        formats.QuantType.Q4_K: ("q4-kpack4", 4, 0),
        formats.QuantType.Q5_K: ("kquant-kpack", 4, 16),
        formats.QuantType.Q6_K: ("kquant-kpack", 4, 8),
    }
    for qtype, (layout, low_pack, high_pack) in expected.items():
        assert pack_gguf._target_layout(qtype, "all-kpack") == layout
        low_bits, high_bits = formats.placed_code_planes(qtype)
        assert 16 // low_bits == low_pack
        assert (16 // high_bits if high_bits else 0) == high_pack
        assert pack_gguf._packability(
            int(qtype), 3, {int(qtype)}, "all-kpack") == \
            (True, "grouped", None)

    with pytest.raises(ValueError, match="unknown layout policy"):
        pack_gguf._target_layout(q3, "guess")

    assert pack_gguf._tensor_geometry((5120, 8192)) == (8192, 5120, None)
    assert pack_gguf._tensor_geometry((5120, 8192, 64)) == (8192, 5120, 64)
    with pytest.raises(ValueError, match="multiples of 256"):
        pack_gguf._tensor_geometry((5119, 8192, 64))
    with pytest.raises(ValueError, match="multiples of 256"):
        pack_gguf._tensor_geometry((5120, 8191, 64))

    assert pack_gguf._grouped_role_authority("blk.12.ffn_down_exps.weight") == (True, None)
    ok, why = pack_gguf._grouped_role_authority("blk.12.unknown_rank3.weight")
    assert not ok and "no grouped role authority" in why


def test_whole_model_all_kpack_conversion_dispatches_dense_and_grouped_exactly_once():
    calls = []

    class FakeRoutes:
        @staticmethod
        def prepare_fully_quantized_dense(blocks, n, k, qtype, **kwargs):
            calls.append(("dense", blocks, n, k, int(qtype), kwargs))
            return "dense-artifact"

        @staticmethod
        def prepare_fully_quantized_grouped(
                blocks, n, k, qtype, experts, **kwargs):
            calls.append(("grouped", blocks, n, k, int(qtype), experts, kwargs))
            return "grouped-artifact"

    blocks = object()
    for qtype in formats.PLACED_CODE_PLANES:
        want = ("q4-kpack4" if qtype == formats.QuantType.Q4_K
                else "kquant-kpack")
        layout, artifact = pack_gguf._prepare_artifact(
            FakeRoutes, blocks, 256, 512, int(qtype), None, "dense",
            "all-kpack")
        assert (layout, artifact) == (want, "dense-artifact")
        assert calls[-1] == (
            "dense", blocks, 256, 512, int(qtype), {"layout": want})

        layout, artifact = pack_gguf._prepare_artifact(
            FakeRoutes, blocks, 256, 512, int(qtype), 4, "grouped",
            "all-kpack")
        assert (layout, artifact) == (want, "grouped-artifact")
        assert calls[-1] == (
            "grouped", blocks, 256, 512, int(qtype), 4,
            {"layout": want})

    with pytest.raises(ValueError, match="expert extent"):
        pack_gguf._prepare_artifact(
            FakeRoutes, blocks, 256, 512, 10, None, "grouped",
            "all-kpack")


def test_whole_model_grouped_kpack_dispatch_preserves_expert_major_rows(monkeypatch):
    calls = []

    def fake_op(name):
        def invoke(*args):
            calls.append((name, args))
            return _planes(formats.q4_kpack4_arrangement())
        return invoke

    monkeypatch.setattr(routes, "_op", fake_op)
    n, k, experts = pack_gguf._tensor_geometry((512, 256, 4))
    blocks = torch.zeros((experts * n * (k // 256), 144), dtype=torch.uint8)
    artifact = routes.prepare_fully_quantized_grouped(
        blocks, n, k, formats.QuantType.Q4_K, experts, layout="q4-kpack4")
    assert artifact.arrangement == formats.q4_kpack4_arrangement()
    assert calls[-1][0] == "gguf_prepare_fully_quantized_grouped_for_arrangement_v2"
    assert tuple(calls[-1][1][0].shape) == (experts * n * (k // 256), 144)
    assert calls[-1][1][1:5] == (n, k, int(formats.QuantType.Q4_K), experts)


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


def test_zero_m_kpack4_v2_reaches_the_exact_dlsym_predicate(tmp_path):
    """The real torch/dlsym seam must preserve all nine v2 fields even when no kernel launch occurs."""
    root = pathlib.Path(__file__).parents[1]
    if not routes.has_op("gguf_dense_fully_quantized_for_arrangement_v2"):
        pytest.skip("the local extension is not built; setup.py build_ext enables this dlsym contract test")
    fake = root / "dev" / "fold_derivation" / "l140_fake_ppu_backend.cpp"
    output = tmp_path / "fmt0_v2.so"
    built = subprocess.run([
        os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-shared", "-fPIC",
        f"-I{root / 'quactlize' / 'include'}", "-DL140_BACKEND_MARKER=140", "-DL140_ACCEPT_V2=1",
        str(fake), "-o", str(output)], capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    code = r'''
import torch
from quactlize import formats, routes

arr = formats.q4_kpack4_arrangement()
artifact = routes.PlacedArtifact((
    torch.zeros((1, 256, 256), dtype=torch.uint8),
    torch.empty((0,), dtype=torch.uint8),
    torch.zeros((2, 256, 16), dtype=torch.uint8)), arr)
got = routes.matmul_fully_quantized_dense(
    torch.zeros((0, 512), dtype=torch.float16), artifact, formats.QuantType.Q4_K)
assert tuple(got.shape) == (0, 256)

grouped = routes.PlacedArtifact((
    torch.zeros((1, 256, 256), dtype=torch.uint8),
    torch.empty((0,), dtype=torch.uint8),
    torch.zeros((1, 2, 256, 16), dtype=torch.uint8)), arr)
grouped_out = routes.matmul_fully_quantized_grouped(
    torch.zeros((1, 512), dtype=torch.float16), grouped,
    formats.QuantType.Q4_K, torch.tensor([1], dtype=torch.int32))
assert tuple(grouped_out.shape) == (1, 256)
print("zero-M K-pack4-v2 exact-predicate=PASS grouped-v2-dlsym=PASS")
'''
    env = dict(os.environ, QUACTLIZE_PPU_LIB_FMT0=str(output))
    run = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr

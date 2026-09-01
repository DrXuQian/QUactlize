"""Host-only contract tests for the versioned fully-quantized artifact descriptor.

These deliberately replace the torch ops with recording callables. They test the Python ABI seam -- descriptor
ownership, validation and forwarding -- without pretending a host mock proves a PPU kernel accepts an arrangement.
The device-side descriptor x tactic predicate has its own compiled controls.
"""
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
from quactlize import pack_gguf


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
    artifact = routes.prepare_fully_quantized_dense(
        blocks, 256, 256, formats.QuantType.Q2_K,
        tile_k=64, layout="xplane")
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
                return _planes(arrangement, n=256)
            return torch.empty((1, 2), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    blocks = torch.zeros((256, 144), dtype=torch.uint8)
    artifact = routes.prepare_fully_quantized_dense(
        blocks, 256, 256, formats.QuantType.Q4_K)
    assert artifact.arrangement == arrangement
    assert artifact.arrangement_version == routes.PLACED_ARTIFACT_VERSION_V2
    wire = (routes.PLACED_ARTIFACT_VERSION_V2, *arrangement)
    assert calls[0][0] == "gguf_prepare_fully_quantized_dense_for_arrangement_v2"
    assert calls[0][1][-9:] == wire

    routes.matmul_fully_quantized_dense(
        torch.zeros((1, 256), dtype=torch.float16), artifact, formats.QuantType.Q4_K)
    assert calls[1][0] == "gguf_dense_fully_quantized_for_arrangement_v2"
    assert calls[1][1][-9:] == wire


def test_q4_n16k64_direct_layout3_descriptor_is_explicit_and_nondefault():
    arrangement = formats.q4_n16k64_direct_arrangement()
    assert arrangement == formats.PlacedArrangementV2(
        formats.PLACED_LAYOUT_Q4_N16K64_DIRECT_V1,
        4, 0, 0, 64, 32, 0,
        formats.Q4_N16K64_DIRECT_MAPPING_ID)
    arrangement.validate()

    # Registering layout 3 must not change automatic checkpoint production.
    assert formats.canonical_fully_quantized_layout(formats.QuantType.Q4_K) == \
        "q4-kpack4"
    assert formats.q4_kpack4_arrangement().layout == \
        formats.PLACED_LAYOUT_Q4_KPACK4_TRANSPOSE_V1

    for field, value in (
            ("mapping_id", arrangement.mapping_id ^ 1),
            ("transport_tile_k", 128),
            ("group_size", 64),
            ("artifact_tile_k", 64),
            ("high_bits", 1)):
        with pytest.raises(
                ValueError, match="noncanonical Q4 N16xK64 direct descriptor"):
            arrangement._replace(**{field: value}).validate()


def test_q4_n16k64_direct_public_producers_forward_layout3_but_compute_rejects(
        monkeypatch):
    """Layout 3 is producible/recoverable without becoming a shipping reader by accident."""
    calls = []
    direct = formats.q4_n16k64_direct_arrangement()

    def fake_op(name):
        def call(*args):
            calls.append((name, args))
            if name == "gguf_prepare_fully_quantized_grouped_for_arrangement_v2":
                low, high, units = _planes(direct, n=256)
                return low.repeat(2, 1, 1), high, units.repeat(2, 1, 1)
            if name.startswith("gguf_prepare_fully_quantized_dense"):
                return _planes(direct, n=256)
            if name == "gguf_packed_scale_prepass":
                return (torch.ones((1, 8, 256), dtype=torch.float16),) * 2
            if name == "gguf_dense_artifact_dequantize_for_arrangement_v2":
                return torch.empty((1, 256, 256), dtype=torch.float16)
            raise AssertionError(f"unexpected op {name}")
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    blocks = torch.zeros((256, formats.BLOCKS[formats.QuantType.Q4_K].block_bytes), dtype=torch.uint8)
    dense = routes.prepare_fully_quantized_dense(
        blocks, 256, 256, formats.QuantType.Q4_K,
        layout="q4-n16k64-direct")
    wire = (routes.PLACED_ARTIFACT_VERSION_V2, *direct)
    assert dense.arrangement == direct
    assert dense.arrangement_version == routes.PLACED_ARTIFACT_VERSION_V2
    assert calls[-1][0] == "gguf_prepare_fully_quantized_dense_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    grouped = routes.prepare_fully_quantized_grouped(
        blocks.repeat(2, 1), 256, 256, formats.QuantType.Q4_K, 2,
        layout="q4-n16k64-direct")
    assert grouped.arrangement == direct
    assert calls[-1][0] == "gguf_prepare_fully_quantized_grouped_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    # Inverse support is part of an offline ABI and does not imply that a
    # production GEMM reader has been routed.
    routes.dequantize_fully_quantized(dense, formats.QuantType.Q4_K)
    assert calls[-1][0] == "gguf_dense_artifact_dequantize_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    before = len(calls)
    with pytest.raises(ValueError, match="shipping compute requires K-pack descriptor"):
        routes.matmul_fully_quantized_dense(
            torch.zeros((1, 256), dtype=torch.float16), dense,
            formats.QuantType.Q4_K)
    with pytest.raises(ValueError, match="shipping compute requires K-pack descriptor"):
        routes.matmul_fully_quantized_grouped(
            torch.zeros((1, 256), dtype=torch.float16), grouped,
            formats.QuantType.Q4_K, torch.tensor([1, 0], dtype=torch.int32))
    with pytest.raises(ValueError, match="shipping compute requires K-pack descriptor"):
        routes.matmul_scale_first_dense(
            torch.zeros((64, 256), dtype=torch.float16), dense,
            formats.QuantType.Q4_K,
            scale_zero=(torch.ones((1, 8, 256), dtype=torch.float16),) * 2)
    with pytest.raises(ValueError, match="shipping compute requires K-pack descriptor"):
        routes.prepare_q4_kpack4_scale_workspace(
            dense, formats.QuantType.Q4_K)
    with pytest.raises(ValueError, match="shipping compute requires K-pack descriptor"):
        routes.matmul_q4_kpack4_dense(
            torch.zeros((1, 256), dtype=torch.float16), dense,
            formats.QuantType.Q4_K)
    assert len(calls) == before

    # The map is N16-atomic, but the public GGUF producer's established tensor
    # boundary is N%256.  Both failures occur before an arrangement-v2
    # producer can see the request.
    q5 = formats.QuantType.Q5_K
    with pytest.raises(ValueError, match="defined only for Q4_K"):
        routes.prepare_fully_quantized_dense(
            torch.zeros((256, formats.BLOCKS[q5].block_bytes), dtype=torch.uint8),
            256, 256, q5, layout="q4-n16k64-direct")
    with pytest.raises(ValueError, match="defined only for Q4_K"):
        routes.prepare_fully_quantized_grouped(
            torch.zeros((512, formats.BLOCKS[q5].block_bytes), dtype=torch.uint8),
            256, 256, q5, 2, layout="q4-n16k64-direct")
    with pytest.raises(ValueError, match="N multiple of 256"):
        routes.prepare_fully_quantized_dense(
            torch.zeros((255, formats.BLOCKS[formats.QuantType.Q4_K].block_bytes), dtype=torch.uint8),
            255, 256, formats.QuantType.Q4_K,
            layout="q4-n16k64-direct")
    with pytest.raises(ValueError, match="unknown fully-quantized dense layout"):
        routes.prepare_fully_quantized_dense(
            blocks, 256, 256, formats.QuantType.Q4_K,
            layout="q4-n16k64")
    assert len(calls) == before

    # Explicit layout 3 must not perturb the automatic Q4 factory.
    auto = routes.prepare_fully_quantized_dense(
        blocks, 256, 256, formats.QuantType.Q4_K)
    assert auto.arrangement == formats.q4_kpack4_arrangement()
    assert auto.arrangement.layout == formats.PLACED_LAYOUT_Q4_KPACK4_TRANSPOSE_V1


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

    n = 256
    k = 512 if qtype in (formats.QuantType.Q3_K, formats.QuantType.Q6_K) else 256
    dense_planes = _planes(arrangement, n=n, k=k)
    grouped_planes = (
        torch.zeros((2, n, k * low_bits // 8), dtype=torch.uint8),
        (torch.zeros((2, n, k * high_bits // 8), dtype=torch.uint8)
         if high_bits else torch.empty((0,), dtype=torch.uint8)),
        torch.zeros((2, 1, n, 16), dtype=torch.uint8),
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
        torch.zeros((n * (k // 256), block_bytes), dtype=torch.uint8), n, k, qtype,
        layout="kquant-kpack")
    wire = (routes.PLACED_ARTIFACT_VERSION_V2, *arrangement)
    assert dense.arrangement == arrangement
    assert calls[-1][0] == "gguf_prepare_fully_quantized_dense_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    routes.matmul_fully_quantized_dense(
        torch.zeros((1, k), dtype=torch.float16), dense, qtype)
    assert calls[-1][0] == "gguf_dense_fully_quantized_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    grouped = routes.prepare_fully_quantized_grouped(
        torch.zeros((2 * n * (k // 256), block_bytes), dtype=torch.uint8),
        n, k, qtype, 2,
        layout="kquant-kpack")
    assert grouped.arrangement == arrangement
    assert calls[-1][0] == "gguf_prepare_fully_quantized_grouped_for_arrangement_v2"
    assert calls[-1][1][-9:] == wire

    routes.matmul_fully_quantized_grouped(
        torch.zeros((1, k), dtype=torch.float16), grouped, qtype,
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


def test_canonical_offline_layout_policy_uses_kpack_for_every_qtype(monkeypatch):
    expected = {
        formats.QuantType.Q2_K: "kquant-kpack",
        formats.QuantType.Q3_K: "kquant-kpack",
        formats.QuantType.Q4_K: "q4-kpack4",
        formats.QuantType.Q5_K: "kquant-kpack",
        formats.QuantType.Q6_K: "kquant-kpack",
    }
    assert {q: formats.canonical_fully_quantized_layout(q) for q in expected} == expected
    assert all(formats.archived_fully_quantized_layouts(q) == frozenset({"xplane"})
               for q in expected)

    calls = []

    def fake_op(name):
        def call(*args):
            qtype = formats.QuantType(args[3])
            arrangement = (formats.q4_kpack4_arrangement()
                           if qtype == formats.QuantType.Q4_K
                           else formats.kquant_kpack_arrangement(qtype))
            calls.append((qtype, name))
            return _planes(arrangement, n=int(args[1]), k=int(args[2]))
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    for qtype, layout in expected.items():
        k = 512 if qtype in (formats.QuantType.Q3_K, formats.QuantType.Q6_K) else 256
        blocks = torch.zeros((256 * (k // 256), formats.BLOCKS[qtype].block_bytes),
                             dtype=torch.uint8)
        artifact = routes.prepare_fully_quantized_dense(blocks, 256, k, qtype)
        if layout == "q4-kpack4":
            assert artifact.arrangement == formats.q4_kpack4_arrangement()
        else:
            assert artifact.arrangement == formats.kquant_kpack_arrangement(qtype)
        assert calls[-1][1] == \
            "gguf_prepare_fully_quantized_dense_for_arrangement_v2"


@pytest.mark.parametrize("qtype,k_quantum", [
    (formats.QuantType.Q2_K, 256),
    (formats.QuantType.Q3_K, 512),
    (formats.QuantType.Q4_K, 256),
    (formats.QuantType.Q5_K, 256),
    (formats.QuantType.Q6_K, 512),
])
def test_resident_geometry_is_shared_and_rejects_tails_before_dispatch(
        monkeypatch, qtype, k_quantum):
    formats.validate_fully_quantized_resident_geometry(
        qtype, 256, k_quantum)

    called = []
    monkeypatch.setattr(
        routes, "_op", lambda name: lambda *args: called.append((name, args)))
    block_bytes = formats.BLOCKS[qtype].block_bytes
    with pytest.raises(ValueError, match="N multiple of 256"):
        routes.prepare_fully_quantized_dense(
            torch.empty((0, block_bytes), dtype=torch.uint8),
            255, k_quantum, qtype)
    with pytest.raises(ValueError, match=f"K multiple of {k_quantum}"):
        routes.prepare_fully_quantized_grouped(
            torch.empty((0, block_bytes), dtype=torch.uint8),
            256, k_quantum // 2, qtype, 2)
    assert called == []


def test_automatic_producer_rejects_tile_k_instead_of_falling_back_to_xplane(
        monkeypatch):
    called = []
    monkeypatch.setattr(
        routes, "_op", lambda name: lambda *args: called.append((name, args)))
    qtype = formats.QuantType.Q2_K
    blocks = torch.zeros((256, formats.BLOCKS[qtype].block_bytes), dtype=torch.uint8)
    with pytest.raises(ValueError, match="explicit Xplane compatibility setting"):
        routes.prepare_fully_quantized_dense(
            blocks, 256, 256, qtype, tile_k=64)
    assert called == []


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
                return _planes(arrangement, n=256)
            if name == "gguf_packed_scale_prepass":
                return (torch.ones((1, 8, 2), dtype=torch.float16),) * 2
            return torch.empty((1, 2), dtype=torch.float16)
        return call

    monkeypatch.setattr(routes, "_op", fake_op)
    artifact = routes.prepare_fully_quantized_grouped(
        torch.zeros((256, 144), dtype=torch.uint8), 256, 256,
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


def test_grouped_shipping_route_rejects_descriptorless_xplane_before_device(
        monkeypatch):
    artifact = (
        torch.zeros((1, 256, 128), dtype=torch.uint8),
        torch.empty((0,), dtype=torch.uint8),
        torch.zeros((1, 2, 256, 20), dtype=torch.uint8),
    )
    calls = []
    monkeypatch.setattr(
        routes, "_op", lambda name: lambda *args: calls.append((name, args)))
    with pytest.raises(TypeError, match="expected a PlacedArtifact"):
        routes.matmul_fully_quantized_grouped(
            torch.zeros((1, 512), dtype=torch.float16), artifact,
            formats.QuantType.Q2_K, torch.tensor([1], dtype=torch.int32))
    assert calls == []


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
    with pytest.raises(ValueError, match="accepted only by the M>=64"):
        routes.matmul_q4_kpack4_dense(
            torch.zeros((63, 256), dtype=torch.float16), artifact,
            scale_workspace=(torch.ones((1, 8, 2), dtype=torch.float16),) * 2)
    assert len(calls) == before

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


@pytest.mark.parametrize("qtype", [
    formats.QuantType.Q2_K, formats.QuantType.Q3_K,
    formats.QuantType.Q4_K, formats.QuantType.Q5_K,
    formats.QuantType.Q6_K,
])
@pytest.mark.parametrize("m", [1, 8, 9, 63, 64, 2048])
def test_generic_kpack_dense_dispatch_matrix(monkeypatch, qtype, m):
    arrangement = (formats.q4_kpack4_arrangement()
                   if qtype == formats.QuantType.Q4_K
                   else formats.kquant_kpack_arrangement(qtype))
    k = 512 if qtype in (formats.QuantType.Q3_K,
                         formats.QuantType.Q6_K) else 256
    artifact = routes.PlacedArtifact(_planes(arrangement, k=k), arrangement)
    calls = []

    def fake_op(name):
        def invoke(*args):
            calls.append((name, args))
            return torch.empty((args[0].shape[0], 2), dtype=torch.float16)
        return invoke

    monkeypatch.setattr(routes, "_op", fake_op)
    workspace = ((torch.ones((1, 8, 2), dtype=torch.float16),) * 2
                 if qtype == formats.QuantType.Q4_K and
                 m >= routes.KPACK4_SCALEFIRST_MIN_ROWS else None)
    routes.matmul_kpack_dense(
        torch.zeros((m, k), dtype=torch.float16), artifact, qtype,
        scale_workspace=workspace)
    want = ("gguf_dense_scale_first_for_arrangement_v2"
            if qtype == formats.QuantType.Q4_K and
            m >= routes.KPACK4_SCALEFIRST_MIN_ROWS
            else "gguf_dense_fully_quantized_for_arrangement_v2")
    assert [name for name, _args in calls] == [want]


@pytest.mark.parametrize("qtype", [
    formats.QuantType.Q2_K, formats.QuantType.Q3_K,
    formats.QuantType.Q4_K, formats.QuantType.Q5_K,
    formats.QuantType.Q6_K,
])
def test_shipping_grouped_route_accepts_each_exact_kpack_descriptor(
        monkeypatch, qtype):
    arrangement = (formats.q4_kpack4_arrangement()
                   if qtype == formats.QuantType.Q4_K
                   else formats.kquant_kpack_arrangement(qtype))
    k = 512 if qtype in (formats.QuantType.Q3_K,
                         formats.QuantType.Q6_K) else 256
    artifact = routes.PlacedArtifact(_planes(arrangement, k=k), arrangement)
    calls = []
    monkeypatch.setattr(
        routes, "_op", lambda name: lambda *args: calls.append((name, args)))
    routes.matmul_fully_quantized_grouped(
        torch.zeros((1, k), dtype=torch.float16), artifact, qtype,
        torch.tensor([1], dtype=torch.int32))
    assert len(calls) == 1
    assert calls[0][0] == "gguf_grouped_fully_quantized_for_arrangement_v2"
    assert calls[0][1][-9:] == (
        routes.PLACED_ARTIFACT_VERSION_V2, *arrangement)


def test_shipping_fq_routes_reject_nonshipping_descriptors_before_device(
        monkeypatch):
    q2 = formats.QuantType.Q2_K
    xplane_v1 = _artifact(q2, 128)
    xplane = formats.placed_arrangement(q2, 128)
    xplane_v2 = formats.PlacedArrangementV2(
        formats.PLACED_LAYOUT_XPLANE_V1, xplane.bits,
        xplane.high_bits, xplane.tile_k, 0, 0, 0, 0)
    direct = formats.q4_n16k64_direct_arrangement()
    wrong_mapping = formats.kquant_kpack_arrangement(q2)._replace(
        mapping_id=formats.KQUANT_KPACK_MAPPING_ID ^ 1)
    cases = (
        (q2, xplane_v1),
        (q2, routes.PlacedArtifact(_planes(xplane_v2), xplane_v2)),
        (formats.QuantType.Q4_K,
         routes.PlacedArtifact(_planes(direct), direct)),
        (q2, routes.PlacedArtifact(_planes(wrong_mapping), wrong_mapping)),
    )
    calls = []
    monkeypatch.setattr(
        routes, "_op", lambda name: lambda *args: calls.append((name, args)))
    for qtype, artifact in cases:
        for grouped in (False, True):
            with pytest.raises((TypeError, ValueError)):
                if grouped:
                    routes.matmul_fully_quantized_grouped(
                        torch.zeros((1, 256), dtype=torch.float16), artifact,
                        qtype, torch.tensor([1], dtype=torch.int32))
                else:
                    routes.matmul_fully_quantized_dense(
                        torch.zeros((1, 256), dtype=torch.float16), artifact,
                        qtype)
    assert calls == []


@pytest.mark.parametrize("qtype", [
    formats.QuantType.Q2_K, formats.QuantType.Q3_K,
    formats.QuantType.Q5_K, formats.QuantType.Q6_K,
])
def test_non_q4_kpack_rejects_scalefirst_and_q4_only_routes_before_device(
        monkeypatch, qtype):
    arrangement = formats.kquant_kpack_arrangement(qtype)
    artifact = routes.PlacedArtifact(_planes(arrangement), arrangement)
    workspace = (torch.ones((1, 8, 2), dtype=torch.float16),) * 2
    calls = []
    monkeypatch.setattr(
        routes, "_op", lambda name: lambda *args: calls.append((name, args)))

    with pytest.raises(NotImplementedError, match="no ScaleFirst reader"):
        routes.matmul_scale_first_dense(
            torch.zeros((64, 256), dtype=torch.float16), artifact, qtype,
            scale_zero=workspace)
    with pytest.raises(NotImplementedError, match="Q4_K-only"):
        routes.prepare_q4_kpack4_scale_workspace(artifact, qtype)
    with pytest.raises(ValueError, match="accepts only Q4_K"):
        routes.matmul_q4_kpack4_dense(
            torch.zeros((1, 256), dtype=torch.float16), artifact, qtype)
    with pytest.raises(ValueError, match="does not accept a ScaleFirst workspace"):
        routes.matmul_kpack_dense(
            torch.zeros((64, 256), dtype=torch.float16), artifact, qtype,
            scale_workspace=workspace)
    assert calls == []


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


def test_folded_xplane_descriptor_remains_bc_only_and_cannot_enter_shipping_fq(monkeypatch):
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
    with pytest.raises(ValueError, match="version-2 K-pack artifact"):
        routes.matmul_fully_quantized_dense(a, artifact, formats.QuantType.Q3_K)
    assert calls == []
    routes.matmul_bc_gemv(a, artifact, formats.QuantType.Q3_K)
    want_tail = (int(formats.QuantType.Q3_K), routes.PLACED_ARTIFACT_VERSION, 2, 64, 1)
    assert calls[0][0] == "gguf_gemv_bc_for_arrangement"
    assert calls[0][1][-5:] == want_tail


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


def test_f2_xplane_artifact_is_rejected_before_a_shipping_reader(monkeypatch):
    """The shipping route refuses Xplane before a compiled reader can reinterpret it."""
    artifact = _artifact(formats.QuantType.Q3_K, 64)
    called = []
    monkeypatch.setattr(
        routes, "_op", lambda name: lambda *args: called.append((name, args)))
    with pytest.raises(ValueError, match="version-2 K-pack artifact"):
        routes.matmul_fully_quantized_dense(
            torch.zeros((1, 256), dtype=torch.float16), artifact, formats.QuantType.Q3_K)
    assert called == []


@pytest.mark.parametrize("reader", ["bc_moe", "grouped_tensor"])
def test_grouped_readers_reject_a_descriptor_their_legacy_ops_cannot_carry(monkeypatch, reader):
    called = []
    monkeypatch.setattr(routes, "_op", lambda name: lambda *args: called.append((name, args)))
    artifact = _artifact(formats.QuantType.Q3_K, 64)
    rows = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError, match="grouped.*descriptor|descriptor.*grouped|version-2 K-pack"):
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


def test_manifest_roundtrip_restores_q4_n16k64_layout3_without_aliasing_layout1(
        tmp_path):
    arrangement = formats.q4_n16k64_direct_arrangement()
    artifact = routes.PlacedArtifact(_planes(arrangement, n=16), arrangement)
    tensor_dir = tmp_path / "weight"
    tensor_dir.mkdir()
    pack_gguf._write(tensor_dir, *artifact)
    record = {
        "name": "blk.0.weight", "dir": "weight",
        "ggml_type": int(formats.QuantType.Q4_K),
        "arrangement_version": routes.PLACED_ARTIFACT_VERSION_V2,
        "arrangement": arrangement._asdict(),
    }
    restored = pack_gguf.restore_artifact(tmp_path, record)
    assert restored == artifact
    assert restored.arrangement == arrangement
    assert restored.arrangement != formats.q4_kpack4_arrangement()

    planted = dict(
        record,
        arrangement=dict(
            record["arrangement"], mapping_id=arrangement.mapping_id ^ 1))
    with pytest.raises(ValueError, match="noncanonical Q4 N16xK64 direct descriptor"):
        pack_gguf.restore_artifact(tmp_path, planted)

    wrong_qtype = dict(record, ggml_type=int(formats.QuantType.Q5_K))
    with pytest.raises(ValueError, match="layout 3 is Q4_K-only"):
        pack_gguf.restore_artifact(tmp_path, wrong_qtype)


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


def test_whole_model_packer_canonical_policy_covers_every_plane_and_grouped_route():
    supported = {int(formats.QuantType.Q3_K), int(formats.QuantType.Q4_K)}
    q4 = int(formats.QuantType.Q4_K)
    q3 = int(formats.QuantType.Q3_K)

    assert pack_gguf._packability(q4, 2, supported) == (True, "dense", None)
    assert pack_gguf._packability(q4, 3, supported) == (True, "grouped", None)
    assert pack_gguf._packability(q3, 3, supported) == (True, "grouped", None)

    expected = {
        formats.QuantType.Q2_K: ("kquant-kpack", 8, 0),
        formats.QuantType.Q3_K: ("kquant-kpack", 8, 16),
        formats.QuantType.Q4_K: ("q4-kpack4", 4, 0),
        formats.QuantType.Q5_K: ("kquant-kpack", 4, 16),
        formats.QuantType.Q6_K: ("kquant-kpack", 4, 8),
    }
    for qtype, (layout, low_pack, high_pack) in expected.items():
        assert pack_gguf._target_layout(qtype) == layout
        low_bits, high_bits = formats.placed_code_planes(qtype)
        assert 16 // low_bits == low_pack
        assert (16 // high_bits if high_bits else 0) == high_pack
        assert pack_gguf._packability(int(qtype), 3, {int(qtype)}) == \
            (True, "grouped", None)

    assert pack_gguf._tensor_geometry((512, 256), q3) == (256, 512, None)
    assert pack_gguf._tensor_geometry((256, 256, 64), q4) == (256, 256, 64)
    with pytest.raises(ValueError, match="N multiple of 256"):
        pack_gguf._tensor_geometry((512, 255, 64), q3)
    with pytest.raises(ValueError, match="Q3_K.*K multiple of 512"):
        pack_gguf._tensor_geometry((256, 256, 64), q3)

    assert pack_gguf._route_role_authority(
        "blk.12.ffn_down_exps.weight", 3, "grouped") == (True, None)
    ok, why = pack_gguf._route_role_authority(
        "blk.12.unknown_rank3.weight", 3, "grouped")
    assert not ok and "no grouped role authority" in why


def test_whole_model_canonical_conversion_dispatches_dense_and_grouped_exactly_once():
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
            FakeRoutes, blocks, 256, 512, int(qtype), None, "dense")
        assert (layout, artifact) == (want, "dense-artifact")
        assert calls[-1] == (
            "dense", blocks, 256, 512, int(qtype), {"layout": want})

        layout, artifact = pack_gguf._prepare_artifact(
            FakeRoutes, blocks, 256, 512, int(qtype), 4, "grouped")
        assert (layout, artifact) == (want, "grouped-artifact")
        assert calls[-1] == (
            "grouped", blocks, 256, 512, int(qtype), 4,
            {"layout": want})

    with pytest.raises(ValueError, match="expert extent"):
        pack_gguf._prepare_artifact(
            FakeRoutes, blocks, 256, 512, 10, None, "grouped")


def test_whole_model_grouped_kpack_dispatch_preserves_expert_major_rows(monkeypatch):
    calls = []

    def fake_op(name):
        def invoke(*args):
            calls.append((name, args))
            return _planes(formats.q4_kpack4_arrangement())
        return invoke

    monkeypatch.setattr(routes, "_op", fake_op)
    n, k, experts = pack_gguf._tensor_geometry(
        (512, 256, 4), formats.QuantType.Q4_K)
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


def test_zero_m_cannot_bypass_the_compiled_v2_arrangement_predicate(tmp_path):
    """An empty output still needs an exact compiled K-pack predicate and reader symbol."""
    root = pathlib.Path(__file__).parents[1]
    if not routes.has_op("gguf_dense_fully_quantized_for_arrangement_v2"):
        pytest.skip("the local extension is not built; setup.py build_ext enables this dlsym contract test")
    fake = root / "dev" / "fold_derivation" / "l140_fake_ppu_backend.cpp"
    common = [
        os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-shared", "-fPIC",
        f"-I{root / 'quactlize' / 'include'}", "-DL140_BACKEND_MARKER=140",
        "-DL140_ACCEPT_V2=1",
    ]
    wrong_layout = tmp_path / "fmt0_wrong_layout.so"
    predicate_only = tmp_path / "fmt0_predicate_only.so"
    for output, extra in (
        (wrong_layout, ["-DL140_ACCEPT_V2_LAYOUT=3"]),
        (predicate_only, ["-DL140_OMIT_ARRANGEMENT_READER=1"]),
    ):
        built = subprocess.run(
            common + extra + [str(fake), "-o", str(output)],
            capture_output=True, text=True)
        assert built.returncode == 0, built.stdout + built.stderr

    code = r'''
import os
import torch
from quactlize import formats, routes

EXPECTED = os.environ["EXPECTED"]
arr = formats.q4_kpack4_arrangement()
artifact = routes.PlacedArtifact((
    torch.zeros((1, 256, 256), dtype=torch.uint8),
    torch.empty((0,), dtype=torch.uint8),
    torch.zeros((2, 256, 16), dtype=torch.uint8)), arr)
try:
    routes.matmul_fully_quantized_dense(
        torch.zeros((0, 512), dtype=torch.float16), artifact,
        formats.QuantType.Q4_K)
except RuntimeError as exc:
    assert EXPECTED in str(exc), exc
else:
    raise AssertionError("zero-M bypassed the compiled v2 descriptor contract")
'''
    for library, expected in (
        (wrong_layout, "no compatible compiled reader"),
        (predicate_only, "lacks the physical-layout-aware v2 ABI"),
    ):
        run = subprocess.run(
            [sys.executable, "-c", code], cwd=root,
            env=dict(os.environ, QUACTLIZE_PPU_LIB_FMT0=str(library),
                     EXPECTED=expected), capture_output=True, text=True)
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

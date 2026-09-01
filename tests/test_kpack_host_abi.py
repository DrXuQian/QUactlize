"""Raw GGUF <-> canonical K-pack coverage for the loader-facing C ABI."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from quactlize import formats, ppu_bundle, routes
from reference import gguf_kpack


class ArrangementV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int32),
        ("layout", ctypes.c_int32),
        ("bits", ctypes.c_int32),
        ("high_bits", ctypes.c_int32),
        ("artifact_tile_k", ctypes.c_int32),
        ("transport_tile_k", ctypes.c_int32),
        ("group_size", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("mapping_id", ctypes.c_uint64),
    ]


U8P = ctypes.POINTER(ctypes.c_uint8)
ARRP = ctypes.POINTER(ArrangementV2)


def _library(qtype):
    root = os.environ.get("QUACTLIZE_PPU_BUNDLE")
    if not root:
        pytest.skip("QUACTLIZE_PPU_BUNDLE is required for the compiled host ABI test")
    role = next(
        role for role in ppu_bundle.LIBRARY_ROLES
        if role.qtype == int(qtype) and role.packed_format is not None)
    path = Path(root) / role.filename
    if not path.is_file():
        pytest.skip(f"format-selected PPU library is absent: {path}")
    library = ctypes.CDLL(str(path))
    identity = library.quactlize_ppu_build_packed_format_v1
    identity.argtypes = []
    identity.restype = ctypes.c_int32
    assert identity() == role.packed_format, (
        f"{path} reports packed format {identity()}, expected {role.packed_format}")
    prepare = library.quactlize_ppu_prepare_fully_quantized_for_arrangement_v2
    prepare.argtypes = [U8P, U8P, U8P, U8P, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ARRP]
    prepare.restype = ctypes.c_int
    recover = library.quactlize_ppu_recover_fully_quantized_for_arrangement_v2
    recover.argtypes = [U8P, U8P, U8P, U8P, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ARRP]
    recover.restype = ctypes.c_int
    return prepare, recover


def _role_library(role):
    root = os.environ.get("QUACTLIZE_PPU_BUNDLE")
    if not root:
        pytest.skip("QUACTLIZE_PPU_BUNDLE is required for the compiled host ABI test")
    path = Path(root) / role.filename
    if not path.is_file():
        pytest.skip(f"PPU library is absent: {path}")
    return ctypes.CDLL(str(path))


def _u8_pointer(array):
    return array.ctypes.data_as(U8P) if array.size else None


def _guarded_u8(size):
    storage = np.full(size + 32, 0xA5, dtype=np.uint8)
    return storage, storage[16:16 + size]


def _assert_guards(storage):
    assert np.all(storage[:16] == 0xA5)
    assert np.all(storage[-16:] == 0xA5)


def _arrangement(value):
    return ArrangementV2(routes.PLACED_ARTIFACT_VERSION_V2, *value)


def test_arrangement_v2_ctypes_layout_matches_the_public_c_abi():
    assert ctypes.sizeof(ArrangementV2) == 40
    assert ArrangementV2.version.offset == 0
    assert ArrangementV2.reserved.offset == 28
    assert ArrangementV2.mapping_id.offset == 32


@pytest.mark.parametrize("role", ppu_bundle.LIBRARY_ROLES,
                         ids=lambda role: role.role)
def test_format_selected_canonical_arrangement_query_fails_closed(role):
    library = _role_library(role)
    query = library.quactlize_ppu_canonical_arrangement_v2
    query.argtypes = [ctypes.c_int, ARRP]
    query.restype = ctypes.c_int

    def cleared(value):
        return bytes(value) == bytes(ctypes.sizeof(ArrangementV2))

    # A null destination never has an address to clear, but still reports the
    # public malformed-output status rather than selecting a descriptor.
    assert query(int(role.qtype), None) == 23

    # The canonical-arrangement ABI owns the five fully-quantized K formats,
    # not every quantization block layout known by formats.py. A recognized
    # K-quant owned by another FMT is rc=29; a non-K qtype is unknown here and
    # correctly returns rc=22.
    known = tuple(sorted(formats.FUSED_NATIVE_SCALE, key=int))
    if role.packed_format is None:
        for qtype in known:
            out = ArrangementV2(*([0x5A5A5A5A] * 8), 0x5A5A5A5A5A5A5A5A)
            assert query(int(qtype), ctypes.byref(out)) == 29
            assert cleared(out)
        return

    expected = (formats.q4_kpack4_arrangement()
                if role.qtype == int(formats.QuantType.Q4_K)
                else formats.kquant_kpack_arrangement(formats.QuantType(role.qtype)))
    out = ArrangementV2(*([0x5A5A5A5A] * 8), 0x5A5A5A5A5A5A5A5A)
    assert query(int(role.qtype), ctypes.byref(out)) == 0
    assert tuple(getattr(out, field) for field, _ctype in ArrangementV2._fields_) == (
        routes.PLACED_ARTIFACT_VERSION_V2, *expected)

    for qtype in known:
        if int(qtype) == role.qtype:
            continue
        out = ArrangementV2(*([0x5A5A5A5A] * 8), 0x5A5A5A5A5A5A5A5A)
        assert query(int(qtype), ctypes.byref(out)) == 29
        assert cleared(out)

    for qtype in (formats.QuantType.Q4_0, 99):
        out = ArrangementV2(*([0x5A5A5A5A] * 8), 0x5A5A5A5A5A5A5A5A)
        assert query(int(qtype), ctypes.byref(out)) == 22
        assert cleared(out)


def test_unknown_qtype_error_does_not_depend_on_high_pointer():
    prepare, recover = _library(formats.QuantType.Q4_K)
    byte = np.zeros(1, dtype=np.uint8)
    arrangement = _arrangement(formats.q4_kpack4_arrangement())
    for high in (None, _u8_pointer(byte)):
        assert prepare(_u8_pointer(byte), _u8_pointer(byte), high, _u8_pointer(byte),
                       256, 256, 1, 99, ctypes.byref(arrangement)) == 22
        assert recover(_u8_pointer(byte), high, _u8_pointer(byte), _u8_pointer(byte),
                       256, 256, 1, 99, ctypes.byref(arrangement)) == 22


def test_complete_host_abi_rejects_size_overflow_without_touching_outputs():
    prepare, recover = _library(formats.QuantType.Q4_K)
    canary = np.full(32, 0xA5, dtype=np.uint8)
    before = canary.copy()
    arrangement = _arrangement(formats.q4_kpack4_arrangement())
    huge_multiple_of_256 = 2_147_483_392
    assert prepare(_u8_pointer(canary), _u8_pointer(canary), None, _u8_pointer(canary),
                   huge_multiple_of_256, huge_multiple_of_256, 2, 12,
                   ctypes.byref(arrangement)) == 26
    assert np.array_equal(canary, before)
    assert recover(_u8_pointer(canary), None, _u8_pointer(canary), _u8_pointer(canary),
                   huge_multiple_of_256, huge_multiple_of_256, 2, 12,
                   ctypes.byref(arrangement)) == 26
    assert np.array_equal(canary, before)


@pytest.mark.parametrize(("qtype", "k", "experts", "grouped"), [
    pytest.param(qtype, 512, 2, True, id=f"{qtype.name}-grouped")
    for qtype in (
        formats.QuantType.Q2_K,
        formats.QuantType.Q3_K,
        formats.QuantType.Q4_K,
        formats.QuantType.Q5_K,
        formats.QuantType.Q6_K,
    )
] + [
    pytest.param(qtype, 512, 1, False, id=f"{qtype.name}-dense")
    for qtype in (
        formats.QuantType.Q2_K,
        formats.QuantType.Q3_K,
        formats.QuantType.Q4_K,
        formats.QuantType.Q5_K,
        formats.QuantType.Q6_K,
    )
] + [
    pytest.param(formats.QuantType.Q3_K, 1024, 2, True, id="Q3_K-grouped-two-units"),
    pytest.param(formats.QuantType.Q6_K, 1024, 2, True, id="Q6_K-grouped-two-units"),
])
def test_complete_arrangement_v2_host_abi_matches_reference_and_round_trips(
        qtype, k, experts, grouped):
    prepare, recover = _library(qtype)
    n = 256
    block = formats.BLOCKS[qtype]
    rows = experts * n * (k // block.weights)
    raw = ((np.arange(rows * block.block_bytes, dtype=np.uint32) * 37 + int(qtype)) & 0xff)
    raw = raw.astype(np.uint8).reshape(rows, block.block_bytes)
    reference = (gguf_kpack.prepare_grouped(
        torch.from_numpy(raw.copy()), n, k, qtype.name, experts)
        if grouped else gguf_kpack.prepare_dense(
            torch.from_numpy(raw.copy()), n, k, qtype.name))

    low_storage, low = _guarded_u8(reference.low.numel())
    high_storage, high = _guarded_u8(reference.high.numel())
    unit_storage, units = _guarded_u8(reference.units.numel())
    canonical = (formats.q4_kpack4_arrangement()
                 if qtype == formats.QuantType.Q4_K
                 else formats.kquant_kpack_arrangement(qtype))
    arrangement = _arrangement(canonical)
    rc = prepare(_u8_pointer(raw), _u8_pointer(low), _u8_pointer(high), _u8_pointer(units),
                 n, k, experts, int(qtype), ctypes.byref(arrangement))
    assert rc == 0
    assert low.tobytes() == reference.low.numpy().tobytes()
    assert high.tobytes() == reference.high.numpy().tobytes()
    assert units.tobytes() == reference.units.numpy().tobytes()
    for storage in (low_storage, high_storage, unit_storage):
        _assert_guards(storage)

    restored_storage, restored = _guarded_u8(raw.size)
    rc = recover(_u8_pointer(low), _u8_pointer(high), _u8_pointer(units), _u8_pointer(restored),
                 n, k, experts, int(qtype), ctypes.byref(arrangement))
    assert rc == 0
    assert np.array_equal(restored.reshape(raw.shape), raw)
    _assert_guards(restored_storage)

    wrong = ArrangementV2(*[getattr(arrangement, field) for field, _ctype in ArrangementV2._fields_])
    wrong.mapping_id ^= 1
    assert prepare(_u8_pointer(raw), _u8_pointer(low), _u8_pointer(high), _u8_pointer(units),
                   n, k, experts, int(qtype), ctypes.byref(wrong)) != 0
    assert prepare(_u8_pointer(raw), _u8_pointer(low), _u8_pointer(high), _u8_pointer(units),
                   n, k, experts, int(qtype), None) == 23

    unexpected_high = np.empty(1, dtype=np.uint8) if reference.high.numel() == 0 else np.empty(0, dtype=np.uint8)
    assert prepare(_u8_pointer(raw), _u8_pointer(low), _u8_pointer(unexpected_high), _u8_pointer(units),
                   n, k, experts, int(qtype), ctypes.byref(arrangement)) == 21
    assert recover(_u8_pointer(low), _u8_pointer(unexpected_high), _u8_pointer(units), _u8_pointer(restored),
                   n, k, experts, int(qtype), ctypes.byref(arrangement)) == 21


def test_format_selected_library_rejects_another_role_without_writing():
    prepare, recover = _library(formats.QuantType.Q4_K)
    canary = np.full(64, 0xA5, dtype=np.uint8)
    before = canary.copy()
    arrangement = _arrangement(formats.kquant_kpack_arrangement(formats.QuantType.Q2_K))

    assert prepare(_u8_pointer(canary), _u8_pointer(canary), None, _u8_pointer(canary),
                   256, 256, 1, int(formats.QuantType.Q2_K),
                   ctypes.byref(arrangement)) == 29
    assert np.array_equal(canary, before)
    assert recover(_u8_pointer(canary), None, _u8_pointer(canary), _u8_pointer(canary),
                   256, 256, 1, int(formats.QuantType.Q2_K),
                   ctypes.byref(arrangement)) == 29
    assert np.array_equal(canary, before)


@pytest.mark.parametrize("qtype", [formats.QuantType.Q3_K, formats.QuantType.Q6_K])
def test_paired_metadata_geometry_fails_before_alias_checks_or_writes(qtype):
    prepare, recover = _library(qtype)
    canary = np.full(64, 0xA5, dtype=np.uint8)
    before = canary.copy()
    arrangement = _arrangement(formats.kquant_kpack_arrangement(qtype))

    assert prepare(_u8_pointer(canary), _u8_pointer(canary), _u8_pointer(canary),
                   _u8_pointer(canary), 256, 256, 1, int(qtype),
                   ctypes.byref(arrangement)) == 24
    assert np.array_equal(canary, before)
    assert recover(_u8_pointer(canary), _u8_pointer(canary), _u8_pointer(canary),
                   _u8_pointer(canary), 256, 256, 1, int(qtype),
                   ctypes.byref(arrangement)) == 24
    assert np.array_equal(canary, before)


def test_complete_host_abi_rejects_partial_tensor_aliases_without_writing():
    qtype = formats.QuantType.Q4_K
    prepare, recover = _library(qtype)
    n, k, experts = 256, 256, 1
    block = formats.BLOCKS[qtype]
    raw_bytes = experts * n * (k // block.weights) * block.block_bytes
    low_bytes = experts * n * k * 4 // 8
    unit_bytes = experts * n * (k // block.weights) * block.scale_meta_bytes
    arrangement = _arrangement(formats.q4_kpack4_arrangement())

    shared = np.full(raw_bytes + 64, 0xA5, dtype=np.uint8)
    shared_before = shared.copy()
    blocks = shared[16:16 + raw_bytes]
    low = shared[24:24 + low_bytes]
    units = np.full(unit_bytes, 0xA5, dtype=np.uint8)
    units_before = units.copy()
    assert prepare(_u8_pointer(blocks), _u8_pointer(low), None, _u8_pointer(units),
                   n, k, experts, int(qtype), ctypes.byref(arrangement)) == 30
    assert np.array_equal(shared, shared_before)
    assert np.array_equal(units, units_before)

    low_input = shared[16:16 + low_bytes]
    recovered = shared[24:24 + raw_bytes]
    assert recover(_u8_pointer(low_input), None, _u8_pointer(units), _u8_pointer(recovered),
                   n, k, experts, int(qtype), ctypes.byref(arrangement)) == 30
    assert np.array_equal(shared, shared_before)
    assert np.array_equal(units, units_before)


def test_arrangement_descriptor_is_snapshotted_before_grouped_output_writes():
    qtype = formats.QuantType.Q4_K
    prepare, _recover = _library(qtype)
    n, k, experts = 256, 512, 2
    block = formats.BLOCKS[qtype]
    rows = experts * n * (k // block.weights)
    raw = ((np.arange(rows * block.block_bytes, dtype=np.uint32) * 53 + 7) & 0xff)
    raw = raw.astype(np.uint8).reshape(rows, block.block_bytes)
    reference = gguf_kpack.prepare_grouped(
        torch.from_numpy(raw.copy()), n, k, qtype.name, experts)

    low_storage, low = _guarded_u8(reference.low.numel())
    unit_storage, units = _guarded_u8(reference.units.numel())
    descriptor = ArrangementV2.from_buffer(low)
    canonical = _arrangement(formats.q4_kpack4_arrangement())
    ctypes.memmove(ctypes.addressof(descriptor), ctypes.addressof(canonical),
                   ctypes.sizeof(ArrangementV2))

    assert prepare(_u8_pointer(raw), _u8_pointer(low), None, _u8_pointer(units),
                   n, k, experts, int(qtype), ctypes.byref(descriptor)) == 0
    assert low.tobytes() == reference.low.numpy().tobytes()
    assert units.tobytes() == reference.units.numpy().tobytes()
    _assert_guards(low_storage)
    _assert_guards(unit_storage)

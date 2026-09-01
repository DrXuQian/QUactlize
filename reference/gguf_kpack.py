#!/usr/bin/env python3
"""Readable PyTorch reference for GGUF K-quant -> canonical K-pack bytes.

This module is a format reference, not a production converter.  It deliberately
uses scalar Python loops so every source byte, destination word and axis order is
visible to a reader porting the mapping to another implementation.

Input blocks use the official GGUF order::

    dense:   [N * (K / 256), raw_block_bytes]
    grouped: [E * N * (K / 256), raw_block_bytes]

The grouped input is expert-major.  The returned resident artifact contains:

    low:   [E, N, K * low_bits / 8]
    high:  [E, N, K * high_bits / 8], or an empty [0] tensor
    units: dense   [K-unit, N, unit_bytes]
           grouped [E, K-unit, N, unit_bytes]

Although ``low`` and ``high`` retain convenient tensor shapes, their underlying
bytes are physical little-endian b16 tensors ``[K / Pack, N]`` where
``Pack = 16 / plane_bits``.  Q5_K's one-bit high plane has the separately
specified transpose used by its reader.

The block mapping requires only PyTorch.  The optional whole-file commands also
use the official ``gguf`` Python package; neither path loads a quactlize
extension, PPU SDK, CuTe or device.  ``recover_raw_blocks`` is included solely
to make byte-exact round-trip tests possible.

Typical use::

    artifact = prepare_dense(raw_blocks, n=5120, k=8192, qtype="Q4_K")
    low, high, units = artifact.low, artifact.high, artifact.units
    assert torch.equal(recover_raw_blocks(artifact), raw_blocks)

Whole-file reference use::

    python reference/gguf_kpack.py pack model.gguf model.kpack-reference.gguf
    python reference/gguf_kpack.py verify model.kpack-reference.gguf --source model.gguf

The output is an augmented verification GGUF: it retains every source tensor
and metadata field, then appends standard I8 carrier tensors plus a versioned
manifest.  It is deliberately self-verifying and is not a stock llama.cpp
runtime model until a consumer implements those companion tensor names. Use
``quactlize-pack-gguf`` for the source-bound persistent runtime sidecar; this
augmented file remains the portable mapping and inverse oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch


Q2_K = 10
Q3_K = 11
Q4_K = 12
Q5_K = 13
Q6_K = 14

ARRANGEMENT_VERSION = 2
LAYOUT_Q4_KPACK4 = 1
LAYOUT_KQUANT_KPACK = 2
Q4_KPACK4_MAPPING_ID = 0x51344B5034540001
KQUANT_KPACK_MAPPING_ID = 0x514B504B54000001

REFERENCE_GGUF_SCHEMA = "quactlize.kquant-kpack.reference-gguf"
REFERENCE_GGUF_VERSION = 1
REFERENCE_METADATA_PREFIX = "quactlize.kpack.reference."
REFERENCE_SCHEMA_KEY = REFERENCE_METADATA_PREFIX + "schema"
REFERENCE_VERSION_KEY = REFERENCE_METADATA_PREFIX + "version"
REFERENCE_MANIFEST_KEY = REFERENCE_METADATA_PREFIX + "manifest"
REFERENCE_CARRIER_PREFIX = "__qkpack_ref."


@dataclass(frozen=True)
class FormatSpec:
    name: str
    qtype: int
    raw_bytes: int
    low_bits: int
    high_bits: int
    group_size: int
    groups: int
    scale_bits: int
    min_bits: int
    scale_offset: int
    d_offset: int
    dmin_offset: int

    @property
    def has_min(self) -> bool:
        return self.min_bits != 0

    @property
    def header_bytes(self) -> int:
        return 4 if self.has_min else 2

    @property
    def sb_bytes(self) -> int:
        return self.header_bytes + self.groups * (self.scale_bits + self.min_bits) // 8

    @property
    def superblocks_per_unit(self) -> int:
        # Q3_K and Q6_K have 14/18-byte superblock metadata.  Pair two
        # superblocks of the same N column to obtain a 4-byte-copyable unit.
        return 1 if self.sb_bytes % 4 == 0 else 2

    @property
    def unit_bytes(self) -> int:
        return self.superblocks_per_unit * self.sb_bytes

    @property
    def transport_k(self) -> int:
        packs = [16 // self.low_bits]
        if self.high_bits:
            packs.append(16 // self.high_bits)
        return 16 * max(packs)


SPECS: Dict[int, FormatSpec] = {
    Q2_K: FormatSpec("Q2_K", Q2_K, 84, 2, 0, 16, 16, 4, 4, 0, 80, 82),
    Q3_K: FormatSpec("Q3_K", Q3_K, 110, 2, 1, 16, 16, 6, 0, 96, 108, -1),
    Q4_K: FormatSpec("Q4_K", Q4_K, 144, 4, 0, 32, 8, 6, 6, 4, 0, 2),
    Q5_K: FormatSpec("Q5_K", Q5_K, 176, 4, 1, 32, 8, 6, 6, 4, 0, 2),
    Q6_K: FormatSpec("Q6_K", Q6_K, 210, 4, 2, 16, 16, 8, 0, 192, 208, -1),
}
_NAMES = {spec.name: spec for spec in SPECS.values()}


@dataclass(frozen=True)
class ArrangementV2:
    version: int
    layout: int
    bits: int
    high_bits: int
    artifact_tile_k: int
    transport_tile_k: int
    group_size: int
    reserved: int
    mapping_id: int


@dataclass(frozen=True)
class KPackArtifact:
    low: torch.Tensor
    high: torch.Tensor
    units: torch.Tensor
    arrangement: ArrangementV2
    qtype: int
    n: int
    k: int
    experts: int
    grouped: bool


def _spec(qtype: Union[int, str]) -> FormatSpec:
    if isinstance(qtype, str):
        try:
            return _NAMES[qtype.upper()]
        except KeyError as exc:
            raise ValueError(f"unsupported GGUF K-quant {qtype!r}") from exc
    try:
        return SPECS[int(qtype)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported GGUF K-quant {qtype!r}; expected 10..14") from exc


def canonical_arrangement(qtype: Union[int, str]) -> ArrangementV2:
    """Return the exact serialized identity for the bytes this module writes."""
    spec = _spec(qtype)
    if spec.qtype == Q4_K:
        return ArrangementV2(
            ARRANGEMENT_VERSION, LAYOUT_Q4_KPACK4,
            4, 0, 0, 64, 32, 0, Q4_KPACK4_MAPPING_ID,
        )
    return ArrangementV2(
        ARRANGEMENT_VERSION, LAYOUT_KQUANT_KPACK,
        spec.low_bits, spec.high_bits, 0, spec.transport_k,
        spec.group_size, 0, KQUANT_KPACK_MAPPING_ID,
    )


def _validate_geometry(spec: FormatSpec, n: int, k: int) -> Tuple[int, int]:
    n, k = int(n), int(k)
    if n <= 0 or n % 256:
        raise ValueError(f"resident K-pack requires positive N divisible by 256; got N={n}")
    k_quantum = 512 if spec.superblocks_per_unit == 2 else 256
    if k <= 0 or k % k_quantum:
        raise ValueError(
            f"{spec.name} resident K-pack requires positive K divisible by {k_quantum}; got K={k}"
        )
    return n, k


def _as_flat_block_bytes(
    blocks: torch.Tensor, spec: FormatSpec, n: int, k: int, experts: int
) -> bytes:
    if not isinstance(blocks, torch.Tensor):
        raise TypeError("blocks must be a torch.Tensor")
    if blocks.device.type != "cpu" or blocks.dtype != torch.uint8:
        raise TypeError("blocks must be a CPU torch.uint8 tensor")
    if blocks.ndim not in (2, 3) or blocks.shape[-1] != spec.raw_bytes:
        raise ValueError(
            f"{spec.name} blocks must end in raw block size {spec.raw_bytes}; got {tuple(blocks.shape)}"
        )
    rows = experts * n * (k // 256)
    expected_2d = (rows, spec.raw_bytes)
    expected_3d = (experts, n * (k // 256), spec.raw_bytes)
    if tuple(blocks.shape) not in (expected_2d, expected_3d):
        raise ValueError(
            f"expected {spec.name} blocks shaped {expected_2d} or {expected_3d} "
            f"for E={experts}, N={n}, K={k}; got {tuple(blocks.shape)}"
        )
    # This is intentionally a readability reference.  Avoid depending on NumPy
    # or tensor storage internals at the cost of a host-side copy.
    return bytes(blocks.contiguous().view(-1).tolist())


def _tensor_from_bytes(data: bytearray, shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.frombuffer(data, dtype=torch.uint8).clone().reshape(shape)


def _word_get(data: Union[bytes, bytearray], index: int) -> int:
    byte = 2 * index
    return data[byte] | (data[byte + 1] << 8)


def _word_put(data: bytearray, index: int, word: int) -> None:
    byte = 2 * index
    data[byte] = word & 0xFF
    data[byte + 1] = (word >> 8) & 0xFF


def _placed_word_slot(logical_k: int, bits: int) -> Tuple[int, int]:
    pack = 16 // bits
    logical_k_per_delivery = 8 * pack
    physical_kg = (logical_k // logical_k_per_delivery) * 8 + logical_k % 8
    slot = (logical_k % logical_k_per_delivery) // 8
    return physical_kg, slot


def _placed_put(
    data: bytearray,
    *,
    expert: int,
    logical_n: int,
    logical_k: int,
    n_extent: int,
    k_extent: int,
    bits: int,
    value: int,
    q5_high: bool = False,
) -> None:
    if q5_high:
        # Q5 high-plane bit transpose:
        #   physical N bit 3    <- logical K bit 7
        #   physical KG bit 3   <- logical K bit 6
        #   b16 word slot bit 3 <- logical N bit 3
        physical_n = (logical_n & ~15) | (logical_n & 7) | (((logical_k >> 7) & 1) << 3)
        physical_kg = (logical_k // 256) * 16 | (((logical_k >> 6) & 1) << 3) | (logical_k & 7)
        slot = (((logical_n >> 3) & 1) << 3) | ((logical_k >> 3) & 7)
        words_per_expert = k_extent // 16 * n_extent
    else:
        physical_n = logical_n
        physical_kg, slot = _placed_word_slot(logical_k, bits)
        words_per_expert = k_extent // (16 // bits) * n_extent
    index = expert * words_per_expert + physical_kg * n_extent + physical_n
    shift = bits * slot
    mask = ((1 << bits) - 1) << shift
    word = (_word_get(data, index) & ~mask) | ((value << shift) & mask)
    _word_put(data, index, word)


def _placed_get(
    data: bytes,
    *,
    expert: int,
    logical_n: int,
    logical_k: int,
    n_extent: int,
    k_extent: int,
    bits: int,
    q5_high: bool = False,
) -> int:
    if q5_high:
        physical_n = (logical_n & ~15) | (logical_n & 7) | (((logical_k >> 7) & 1) << 3)
        physical_kg = (logical_k // 256) * 16 | (((logical_k >> 6) & 1) << 3) | (logical_k & 7)
        slot = (((logical_n >> 3) & 1) << 3) | ((logical_k >> 3) & 7)
        words_per_expert = k_extent // 16 * n_extent
    else:
        physical_n = logical_n
        physical_kg, slot = _placed_word_slot(logical_k, bits)
        words_per_expert = k_extent // (16 // bits) * n_extent
    index = expert * words_per_expert + physical_kg * n_extent + physical_n
    return (_word_get(data, index) >> (bits * slot)) & ((1 << bits) - 1)


def _raw_code_planes(raw: bytes, base: int, spec: FormatSpec, i: int) -> Tuple[int, int]:
    """Decode one GGUF code directly into its unsigned low/high plane values."""
    group = i // spec.group_size
    j = i % spec.group_size

    if spec.qtype in (Q2_K, Q3_K):
        lo_offset = 16 if spec.qtype == Q2_K else 32
        lo_byte = base + lo_offset + j + 16 * (group & 1) + 32 * (group // 8)
        lo_shift = 2 * ((group // 2) & 3)
        low = (raw[lo_byte] >> lo_shift) & 3
        if spec.qtype == Q2_K:
            return low, 0
        hi_byte = base + j + 16 * (group & 1)
        hi_shift = ((group // 2) & 3) + 4 * (group // 8)
        return low, (raw[hi_byte] >> hi_shift) & 1

    if spec.qtype in (Q4_K, Q5_K):
        lo_offset = 16 if spec.qtype == Q4_K else 48
        lo_byte = base + lo_offset + j + 32 * (group // 2)
        low = (raw[lo_byte] >> (4 * (group & 1))) & 0xF
        if spec.qtype == Q4_K:
            return low, 0
        return low, (raw[base + 16 + j] >> group) & 1

    # Q6_K: the stored 4+2 planes are already the offset-binary code q+32.
    p = group & 1
    q = (group >> 1) & 1
    n = (group >> 2) & 1
    h = group >> 3
    low_byte = base + j + 16 * p + 32 * q + 64 * h
    high_byte = base + 128 + j + 16 * p + 32 * h
    return (raw[low_byte] >> (4 * n)) & 0xF, (raw[high_byte] >> (2 * q + 4 * n)) & 3


def _put_field(raw: bytearray, byte: int, shift: int, width: int, value: int) -> None:
    mask = ((1 << width) - 1) << shift
    raw[byte] = (raw[byte] & ~mask) | ((value << shift) & mask)


def _raw_code_put(
    raw: bytearray, base: int, spec: FormatSpec, i: int, low: int, high: int
) -> None:
    group = i // spec.group_size
    j = i % spec.group_size
    if spec.qtype in (Q2_K, Q3_K):
        lo_offset = 16 if spec.qtype == Q2_K else 32
        _put_field(
            raw,
            base + lo_offset + j + 16 * (group & 1) + 32 * (group // 8),
            2 * ((group // 2) & 3),
            2,
            low,
        )
        if spec.qtype == Q3_K:
            _put_field(
                raw,
                base + j + 16 * (group & 1),
                ((group // 2) & 3) + 4 * (group // 8),
                1,
                high,
            )
        return
    if spec.qtype in (Q4_K, Q5_K):
        lo_offset = 16 if spec.qtype == Q4_K else 48
        _put_field(raw, base + lo_offset + j + 32 * (group // 2), 4 * (group & 1), 4, low)
        if spec.qtype == Q5_K:
            _put_field(raw, base + 16 + j, group, 1, high)
        return
    p = group & 1
    q = (group >> 1) & 1
    n = (group >> 2) & 1
    h = group >> 3
    _put_field(raw, base + j + 16 * p + 32 * q + 64 * h, 4 * n, 4, low)
    _put_field(raw, base + 128 + j + 16 * p + 32 * h, 2 * q + 4 * n, 2, high)


def _metadata_codes(raw: bytes, base: int, spec: FormatSpec, group: int) -> Tuple[int, int]:
    p = base + spec.scale_offset
    if spec.qtype == Q2_K:
        value = raw[p + group]
        return value & 0xF, value >> 4
    if spec.qtype == Q3_K:
        scale = (raw[p + group % 8] >> (4 * (group // 8))) & 0xF
        scale |= ((raw[p + 8 + group % 4] >> (2 * (group // 4))) & 3) << 4
        return scale, 0
    if spec.qtype in (Q4_K, Q5_K):
        t, h = group & 3, group >> 2
        scale = (raw[p + t + 8 * h] & 0xF) | (((raw[p + t] >> (4 + 2 * h)) & 3) << 4)
        minimum = ((raw[p + 4 + t + 4 * h] >> (4 * h)) & 0xF)
        minimum |= (((raw[p + 4 + t] >> (4 + 2 * h)) & 3) << 4)
        return scale, minimum
    return raw[p + group], 0


def _metadata_put(
    raw: bytearray, base: int, spec: FormatSpec, group: int, scale: int, minimum: int
) -> None:
    p = base + spec.scale_offset
    if spec.qtype == Q2_K:
        _put_field(raw, p + group, 0, 4, scale)
        _put_field(raw, p + group, 4, 4, minimum)
        return
    if spec.qtype == Q3_K:
        _put_field(raw, p + group % 8, 4 * (group // 8), 4, scale)
        _put_field(raw, p + 8 + group % 4, 2 * (group // 4), 2, scale >> 4)
        return
    if spec.qtype in (Q4_K, Q5_K):
        t, h = group & 3, group >> 2
        _put_field(raw, p + t + 8 * h, 0, 4, scale)
        _put_field(raw, p + t, 4 + 2 * h, 2, scale >> 4)
        _put_field(raw, p + 4 + t + 4 * h, 4 * h, 4, minimum)
        _put_field(raw, p + 4 + t, 4 + 2 * h, 2, minimum >> 4)
        return
    raw[p + group] = scale & 0xFF


def _unit_bit(spec: FormatSpec, group: int, which: int) -> int:
    run_groups = spec.groups // 2 if spec.has_min else spec.groups
    run_bits = run_groups * (spec.scale_bits + spec.min_bits)
    bits = spec.min_bits if which else spec.scale_bits
    field = (group % run_groups) * bits + (group // run_groups) * run_bits
    if which:
        field += run_groups * spec.scale_bits
    return spec.header_bytes * 8 + field


def _unit_put(data: bytearray, base: int, spec: FormatSpec, group: int, which: int, value: int) -> None:
    bits = spec.min_bits if which else spec.scale_bits
    bit = _unit_bit(spec, group, which)
    byte, shift = base + bit // 8, bit & 7
    shifted = value << shift
    data[byte] |= shifted & 0xFF
    if shift + bits > 8:
        data[byte + 1] |= (shifted >> 8) & 0xFF


def _unit_get(data: bytes, base: int, spec: FormatSpec, group: int, which: int) -> int:
    bits = spec.min_bits if which else spec.scale_bits
    bit = _unit_bit(spec, group, which)
    byte, shift = base + bit // 8, bit & 7
    word = data[byte]
    if shift + bits > 8:
        word |= data[byte + 1] << 8
    return (word >> shift) & ((1 << bits) - 1)


def _prepare(
    blocks: torch.Tensor,
    n: int,
    k: int,
    qtype: Union[int, str],
    *,
    experts: int,
    grouped: bool,
) -> KPackArtifact:
    spec = _spec(qtype)
    n, k = _validate_geometry(spec, n, k)
    experts = int(experts)
    if experts <= 0 or (not grouped and experts != 1):
        raise ValueError("dense requires experts=1; grouped requires a positive expert count")
    raw = _as_flat_block_bytes(blocks, spec, n, k, experts)
    superblocks = k // 256
    num_units = superblocks // spec.superblocks_per_unit

    low = bytearray(experts * n * k * spec.low_bits // 8)
    high = bytearray(experts * n * k * spec.high_bits // 8)
    units = bytearray(experts * num_units * n * spec.unit_bytes)

    for expert in range(experts):
        for logical_n in range(n):
            for sb in range(superblocks):
                block_index = (expert * n + logical_n) * superblocks + sb
                block_base = block_index * spec.raw_bytes
                logical_k_base = sb * 256
                for i in range(256):
                    lo, hi = _raw_code_planes(raw, block_base, spec, i)
                    _placed_put(
                        low,
                        expert=expert,
                        logical_n=logical_n,
                        logical_k=logical_k_base + i,
                        n_extent=n,
                        k_extent=k,
                        bits=spec.low_bits,
                        value=lo,
                    )
                    if spec.high_bits:
                        _placed_put(
                            high,
                            expert=expert,
                            logical_n=logical_n,
                            logical_k=logical_k_base + i,
                            n_extent=n,
                            k_extent=k,
                            bits=spec.high_bits,
                            value=hi,
                            q5_high=spec.qtype == Q5_K,
                        )

                unit = sb // spec.superblocks_per_unit
                unit_sb = sb % spec.superblocks_per_unit
                unit_base = (
                    ((expert * num_units + unit) * n + logical_n) * spec.unit_bytes
                    + unit_sb * spec.sb_bytes
                )
                units[unit_base : unit_base + 2] = raw[
                    block_base + spec.d_offset : block_base + spec.d_offset + 2
                ]
                if spec.has_min:
                    units[unit_base + 2 : unit_base + 4] = raw[
                        block_base + spec.dmin_offset : block_base + spec.dmin_offset + 2
                    ]
                for group in range(spec.groups):
                    scale, minimum = _metadata_codes(raw, block_base, spec, group)
                    _unit_put(units, unit_base, spec, group, 0, scale)
                    if spec.has_min:
                        _unit_put(units, unit_base, spec, group, 1, minimum)

    low_tensor = _tensor_from_bytes(low, (experts, n, k * spec.low_bits // 8))
    high_tensor = (
        _tensor_from_bytes(high, (experts, n, k * spec.high_bits // 8))
        if spec.high_bits
        else torch.empty((0,), dtype=torch.uint8)
    )
    unit_shape = (
        (experts, num_units, n, spec.unit_bytes)
        if grouped
        else (num_units, n, spec.unit_bytes)
    )
    return KPackArtifact(
        low_tensor,
        high_tensor,
        _tensor_from_bytes(units, unit_shape),
        canonical_arrangement(spec.qtype),
        spec.qtype,
        n,
        k,
        experts,
        grouped,
    )


def prepare_dense(
    blocks: torch.Tensor, n: int, k: int, qtype: Union[int, str]
) -> KPackArtifact:
    """Convert one official GGUF dense tensor to canonical resident bytes."""
    return _prepare(blocks, n, k, qtype, experts=1, grouped=False)


def prepare_grouped(
    blocks: torch.Tensor,
    n: int,
    k: int,
    qtype: Union[int, str],
    experts: int,
) -> KPackArtifact:
    """Convert expert-major official GGUF blocks to canonical grouped bytes."""
    return _prepare(blocks, n, k, qtype, experts=experts, grouped=True)


def _flat_tensor_bytes(tensor: torch.Tensor, name: str) -> bytes:
    if tensor.device.type != "cpu" or tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must be a CPU torch.uint8 tensor")
    return bytes(tensor.contiguous().view(-1).tolist())


def recover_raw_blocks(artifact: KPackArtifact) -> torch.Tensor:
    """Invert a reference artifact back to official GGUF block bytes."""
    if not isinstance(artifact, KPackArtifact):
        raise TypeError("artifact must be a KPackArtifact")
    spec = _spec(artifact.qtype)
    if artifact.arrangement != canonical_arrangement(spec.qtype):
        raise ValueError("artifact descriptor is not the canonical mapping for its qtype")
    n, k = _validate_geometry(spec, artifact.n, artifact.k)
    experts = artifact.experts
    expected_low = (experts, n, k * spec.low_bits // 8)
    expected_high = ((experts, n, k * spec.high_bits // 8)
                     if spec.high_bits else (0,))
    num_units = (k // 256) // spec.superblocks_per_unit
    expected_units = ((experts, num_units, n, spec.unit_bytes)
                      if artifact.grouped else (num_units, n, spec.unit_bytes))
    if tuple(artifact.low.shape) != expected_low:
        raise ValueError(f"low carrier must be {expected_low}; got {tuple(artifact.low.shape)}")
    if tuple(artifact.high.shape) != expected_high:
        raise ValueError(f"high carrier must be {expected_high}; got {tuple(artifact.high.shape)}")
    if tuple(artifact.units.shape) != expected_units:
        raise ValueError(f"units carrier must be {expected_units}; got {tuple(artifact.units.shape)}")
    low = _flat_tensor_bytes(artifact.low, "low")
    high = _flat_tensor_bytes(artifact.high, "high") if spec.high_bits else b""
    units = _flat_tensor_bytes(artifact.units, "units")
    superblocks = k // 256
    raw = bytearray(experts * n * superblocks * spec.raw_bytes)

    for expert in range(experts):
        for logical_n in range(n):
            for sb in range(superblocks):
                block_index = (expert * n + logical_n) * superblocks + sb
                block_base = block_index * spec.raw_bytes
                logical_k_base = sb * 256
                for i in range(256):
                    lo = _placed_get(
                        low,
                        expert=expert,
                        logical_n=logical_n,
                        logical_k=logical_k_base + i,
                        n_extent=n,
                        k_extent=k,
                        bits=spec.low_bits,
                    )
                    hi = (
                        _placed_get(
                            high,
                            expert=expert,
                            logical_n=logical_n,
                            logical_k=logical_k_base + i,
                            n_extent=n,
                            k_extent=k,
                            bits=spec.high_bits,
                            q5_high=spec.qtype == Q5_K,
                        )
                        if spec.high_bits
                        else 0
                    )
                    _raw_code_put(raw, block_base, spec, i, lo, hi)

                unit = sb // spec.superblocks_per_unit
                unit_sb = sb % spec.superblocks_per_unit
                unit_base = (
                    ((expert * num_units + unit) * n + logical_n) * spec.unit_bytes
                    + unit_sb * spec.sb_bytes
                )
                raw[block_base + spec.d_offset : block_base + spec.d_offset + 2] = units[
                    unit_base : unit_base + 2
                ]
                if spec.has_min:
                    raw[block_base + spec.dmin_offset : block_base + spec.dmin_offset + 2] = units[
                        unit_base + 2 : unit_base + 4
                    ]
                for group in range(spec.groups):
                    scale = _unit_get(units, unit_base, spec, group, 0)
                    minimum = _unit_get(units, unit_base, spec, group, 1) if spec.has_min else 0
                    _metadata_put(raw, block_base, spec, group, scale, minimum)

    rows = experts * n * superblocks
    return _tensor_from_bytes(raw, (rows, spec.raw_bytes))


def _gguf_modules():
    try:
        import numpy as np
        from gguf import (
            GGMLQuantizationType,
            GGUFEndian,
            GGUFReader,
            GGUFValueType,
            GGUFWriter,
        )
    except ImportError as exc:
        raise RuntimeError(
            "GGUF file I/O needs the official gguf package; install this repository's "
            "packer extra or run `pip install gguf`"
        ) from exc
    return np, GGMLQuantizationType, GGUFEndian, GGUFReader, GGUFValueType, GGUFWriter


def _numpy_bytes(array) -> bytes:
    """Return the physical C-order bytes of a GGUF reader/writer array."""
    np, *_ = _gguf_modules()
    return np.ascontiguousarray(array).view(np.uint8).reshape(-1).tobytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _field_fingerprint(field) -> Tuple[Tuple[int, ...], Tuple[bytes, ...]]:
    """A byte-exact metadata fingerprint, excluding only the field offset."""
    return (
        tuple(int(value_type) for value_type in field.types),
        tuple(part.tobytes() for part in field.parts),
    )


def _field_sha256(field) -> str:
    digest = hashlib.sha256()
    for part in field.parts:
        raw = part.tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _real_metadata(reader) -> Dict[str, object]:
    # Reader-internal header counts have GGUF.* names. Tensor-info entries have
    # no value types. Everything else is serialized model metadata.
    return {
        name: field
        for name, field in reader.fields.items()
        if field.types and not name.startswith("GGUF.")
    }


def _source_architecture(reader) -> str:
    field = reader.get_field("general.architecture")
    if field is None or not field.types:
        raise ValueError("input GGUF has no general.architecture string")
    architecture = field.contents()
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("input GGUF general.architecture is not a nonempty string")
    return architecture


def _copy_source_metadata(reader, writer, GGUFValueType) -> None:
    split_keys = {"split.no", "split.count", "split.tensors.count"}
    metadata = _real_metadata(reader)
    collision = sorted(name for name in metadata if name.startswith(REFERENCE_METADATA_PREFIX))
    if collision:
        raise ValueError(
            "input already owns the K-pack reference metadata namespace: " + ", ".join(collision)
        )
    present_split = sorted(split_keys & set(metadata))
    if present_split:
        raise ValueError(
            "the standalone reference writes one GGUF and does not accept a split shard; "
            f"found {present_split}"
        )
    for name, field in metadata.items():
        if name == "general.architecture":
            # GGUFWriter writes this from its constructor.
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        value = field.contents()
        if value_type == GGUFValueType.ARRAY:
            if len(field.types) != 2:
                raise ValueError(f"metadata {name!r} uses a nested array the reference cannot rewrite")
            if not value:
                raise ValueError(
                    f"metadata {name!r} is an empty array; the official GGUFWriter cannot preserve its subtype"
                )
        writer.add_key_value(name, value, value_type, sub_type=sub_type)


def _source_tensor_record(index: int, tensor) -> dict:
    raw = _numpy_bytes(tensor.data)
    return {
        "index": index,
        "name": tensor.name,
        "tensor_type": int(tensor.tensor_type),
        "type_name": tensor.tensor_type.name,
        "shape": [int(x) for x in tensor.shape],
        "nbytes": len(raw),
        "sha256": _sha256(raw),
    }


def _source_metadata_records(reader) -> list:
    return [
        {
            "index": index,
            "name": name,
            "types": [int(value_type) for value_type in field.types],
            "sha256": _field_sha256(field),
        }
        for index, (name, field) in enumerate(_real_metadata(reader).items())
    ]


def _tensor_geometry(tensor, spec: FormatSpec) -> Tuple[int, int, int, bool]:
    shape = tuple(int(x) for x in tensor.shape)
    if len(shape) not in (2, 3):
        raise ValueError(
            f"{tensor.name}: {spec.name} tensor must have GGUF fast-first shape [K,N] or [K,N,E]; "
            f"got {shape}"
        )
    k, n = shape[:2]
    n, k = _validate_geometry(spec, n, k)
    experts = shape[2] if len(shape) == 3 else 1
    if experts <= 0:
        raise ValueError(f"{tensor.name}: expert extent must be positive; got {experts}")
    return n, k, experts, len(shape) == 3


def _reader_tensor_blocks(tensor, spec: FormatSpec, n: int, k: int, experts: int) -> torch.Tensor:
    raw = _numpy_bytes(tensor.data)
    rows = experts * n * (k // 256)
    expected = rows * spec.raw_bytes
    if len(raw) != expected:
        raise ValueError(
            f"{tensor.name}: GGUF payload has {len(raw)} bytes, expected {expected} "
            f"for E={experts}, N={n}, K={k}, {spec.name}"
        )
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(rows, spec.raw_bytes)


def _carrier_name(index: int, component: str) -> str:
    if component not in ("low", "high", "units"):
        raise ValueError(f"unknown K-pack carrier component {component!r}")
    return f"{REFERENCE_CARRIER_PREFIX}{index:06d}.{component}"


def _add_carrier(writer, name: str, tensor: torch.Tensor) -> dict:
    np, *_ = _gguf_modules()
    if tensor.device.type != "cpu" or tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        raise TypeError(f"carrier {name} must be a contiguous CPU torch.uint8 tensor")
    # GGUF has no custom K-pack ggml_type. I8 is the standard one-byte carrier;
    # viewing uint8 as int8 changes no bits and makes the file spec-valid.
    data = tensor.numpy().view(np.int8)
    writer.add_tensor(name, data)
    raw = data.view(np.uint8).reshape(-1).tobytes()
    return {"name": name, "shape": list(tensor.shape), "sha256": _sha256(raw)}


def _select_reference_tensors(reader, requested: Optional[Sequence[str]]) -> list:
    tensors = list(reader.tensors)
    names = {tensor.name for tensor in tensors}
    collisions = sorted(name for name in names if name.startswith(REFERENCE_CARRIER_PREFIX))
    if collisions:
        raise ValueError(
            "input already owns the K-pack reference tensor namespace: " + ", ".join(collisions[:8])
        )
    if requested is not None:
        requested_set = set(requested)
        if len(requested_set) != len(requested):
            raise ValueError("--tensor names must not be repeated")
        missing = sorted(requested_set - names)
        if missing:
            raise ValueError("requested tensor(s) are absent: " + ", ".join(missing))
    else:
        requested_set = None

    selected = []
    for tensor in tensors:
        chosen = requested_set is None or tensor.name in requested_set
        if not chosen:
            continue
        qtype = int(tensor.tensor_type)
        if qtype not in SPECS:
            if requested_set is not None:
                raise ValueError(
                    f"requested tensor {tensor.name!r} has unsupported type {tensor.tensor_type.name}"
                )
            continue
        if len(tensor.shape) not in (2, 3):
            if requested_set is not None:
                raise ValueError(f"requested tensor {tensor.name!r} has unsupported rank {len(tensor.shape)}")
            continue
        # Validate before creating the output writer so bad geometry publishes
        # neither a final file nor a misleading partial header.
        _tensor_geometry(tensor, SPECS[qtype])
        selected.append(tensor)
    if not selected:
        raise ValueError("input/selection contains no convertible Q2_K/Q3_K/Q4_K/Q5_K/Q6_K tensor")
    return selected


def _write_reference_gguf(writer) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()


def _load_reference_manifest(reader, GGUFValueType) -> dict:
    reserved = {
        name for name in _real_metadata(reader)
        if name.startswith(REFERENCE_METADATA_PREFIX)
    }
    expected_reserved = {REFERENCE_SCHEMA_KEY, REFERENCE_VERSION_KEY, REFERENCE_MANIFEST_KEY}
    if reserved != expected_reserved:
        raise ValueError(
            "K-pack reference metadata namespace disagrees with the schema: "
            f"missing={sorted(expected_reserved - reserved)} extra={sorted(reserved - expected_reserved)}"
        )
    expected_fields = {
        REFERENCE_SCHEMA_KEY: ([GGUFValueType.STRING], REFERENCE_GGUF_SCHEMA),
        REFERENCE_VERSION_KEY: ([GGUFValueType.UINT32], REFERENCE_GGUF_VERSION),
    }
    for name, (types, value) in expected_fields.items():
        field = reader.get_field(name)
        if field is None or field.types != types or field.contents() != value:
            raise ValueError(f"{name} is missing or invalid")
    manifest_field = reader.get_field(REFERENCE_MANIFEST_KEY)
    if manifest_field is None or manifest_field.types != [GGUFValueType.STRING]:
        raise ValueError(f"{REFERENCE_MANIFEST_KEY} is missing or is not a string")
    try:
        manifest = json.loads(manifest_field.contents())
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("K-pack reference manifest is not valid JSON") from exc
    fields = {"schema", "schema_version", "source_metadata", "source_tensors", "tensors"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError(f"K-pack reference manifest must contain exactly {sorted(fields)}")
    if (manifest["schema"] != REFERENCE_GGUF_SCHEMA or
            not isinstance(manifest["schema_version"], int) or
            isinstance(manifest["schema_version"], bool) or
            manifest["schema_version"] != REFERENCE_GGUF_VERSION):
        raise ValueError("K-pack reference manifest schema/version is unsupported")
    if not isinstance(manifest["source_tensors"], list) or not manifest["source_tensors"]:
        raise ValueError("K-pack reference manifest has no source tensor inventory")
    if not isinstance(manifest["source_metadata"], list) or not manifest["source_metadata"]:
        raise ValueError("K-pack reference manifest has no source metadata inventory")
    if not isinstance(manifest["tensors"], list) or not manifest["tensors"]:
        raise ValueError("K-pack reference manifest has no converted tensors")
    return manifest


def _require_int(value, field: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _verify_source_inventory(reader, records: list) -> Tuple[dict, set]:
    expected_fields = {"index", "name", "tensor_type", "type_name", "shape", "nbytes", "sha256"}
    actual = {tensor.name: tensor for tensor in reader.tensors}
    source_names = set()
    by_index = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(f"source_tensors[{position}] has the wrong fields")
        if record["index"] != position or isinstance(record["index"], bool):
            raise ValueError("source tensor inventory is not in canonical input order")
        name = record["name"]
        if not isinstance(name, str) or not name or name.startswith(REFERENCE_CARRIER_PREFIX):
            raise ValueError(f"source_tensors[{position}] has an invalid name")
        if name in source_names or name not in actual:
            raise ValueError(f"source tensor inventory contains a duplicate or missing tensor {name!r}")
        tensor = actual[name]
        if (not isinstance(record["tensor_type"], int) or isinstance(record["tensor_type"], bool) or
                record["tensor_type"] != int(tensor.tensor_type) or
                record["type_name"] != tensor.tensor_type.name):
            raise ValueError(f"source tensor {name}: type identity changed")
        if record["shape"] != [int(x) for x in tensor.shape]:
            raise ValueError(f"source tensor {name}: GGUF shape changed")
        raw = _numpy_bytes(tensor.data)
        if (not isinstance(record["nbytes"], int) or isinstance(record["nbytes"], bool) or
                record["nbytes"] != len(raw) or
                not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or
                record["sha256"] != _sha256(raw)):
            raise ValueError(f"source tensor {name}: byte payload changed")
        source_names.add(name)
        by_index[position] = tensor
    return by_index, source_names


def _verify_metadata_inventory(reader, records: list) -> None:
    expected_fields = {"index", "name", "types", "sha256"}
    metadata = {
        name: field for name, field in _real_metadata(reader).items()
        if not name.startswith(REFERENCE_METADATA_PREFIX)
    }
    if len(records) != len(metadata):
        raise ValueError("source metadata inventory length changed")
    seen = set()
    for position, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != expected_fields or
                record["index"] != position or isinstance(record["index"], bool)):
            raise ValueError(f"source_metadata[{position}] has a noncanonical identity")
        name = record["name"]
        if not isinstance(name, str) or not name or name in seen or name not in metadata:
            raise ValueError(f"source metadata inventory contains a duplicate or missing field {name!r}")
        field = metadata[name]
        types = [int(value_type) for value_type in field.types]
        if (record["types"] != types or not isinstance(record["sha256"], str) or
                len(record["sha256"]) != 64 or record["sha256"] != _field_sha256(field)):
            raise ValueError(f"source metadata {name!r} changed")
        seen.add(name)
    if seen != set(metadata):
        raise ValueError("source metadata inventory keys changed")


def _read_carrier(reader_tensors: dict, record, expected_name: str, expected_shape: list, role: str):
    _, GGMLQuantizationType, *_ = _gguf_modules()
    fields = {"name", "shape", "sha256"}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError(f"{role} carrier record has the wrong fields")
    if (record["name"] != expected_name or record["shape"] != expected_shape or
            not isinstance(record["sha256"], str) or len(record["sha256"]) != 64):
        raise ValueError(f"{role} carrier identity/shape is not canonical")
    tensor = reader_tensors.get(expected_name)
    if tensor is None or tensor.tensor_type != GGMLQuantizationType.I8:
        raise ValueError(f"{role} carrier {expected_name} is missing or is not GGML I8")
    if list(tensor.data.shape) != expected_shape:
        raise ValueError(f"{role} carrier {expected_name} GGUF shape disagrees with its manifest")
    raw = _numpy_bytes(tensor.data)
    if record["sha256"] != _sha256(raw):
        raise ValueError(f"{role} carrier {expected_name} checksum mismatch")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(expected_shape)


def _verify_converted(reader, records: list, source_by_index: dict) -> set:
    fields = {
        "index", "source_index", "source_name", "qtype", "n", "k", "experts", "grouped",
        "arrangement", "carriers",
    }
    arrangement_fields = set(ArrangementV2.__dataclass_fields__)
    reader_tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected_carriers = set()
    seen_sources = set()
    for position, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != fields or
                record["index"] != position or isinstance(record["index"], bool)):
            raise ValueError(f"converted tensor record {position} has a noncanonical identity")
        source_index = _require_int(record["source_index"], f"tensors[{position}].source_index")
        source = source_by_index.get(source_index)
        if source is None or source.name != record["source_name"] or source_index in seen_sources:
            raise ValueError(f"converted tensor record {position} has an invalid source binding")
        seen_sources.add(source_index)
        qtype = _require_int(record["qtype"], f"tensors[{position}].qtype")
        if qtype not in SPECS or qtype != int(source.tensor_type):
            raise ValueError(f"converted tensor {source.name}: qtype disagrees with the source")
        spec = SPECS[qtype]
        n, k, experts, grouped = _tensor_geometry(source, spec)
        if (not isinstance(record["grouped"], bool) or
                not isinstance(record["n"], int) or isinstance(record["n"], bool) or
                not isinstance(record["k"], int) or isinstance(record["k"], bool) or
                not isinstance(record["experts"], int) or isinstance(record["experts"], bool) or
                (record["n"], record["k"], record["experts"], record["grouped"]) !=
                (n, k, experts, grouped)):
            raise ValueError(f"converted tensor {source.name}: geometry disagrees with the source")
        raw_arrangement = record["arrangement"]
        if not isinstance(raw_arrangement, dict) or set(raw_arrangement) != arrangement_fields:
            raise ValueError(f"converted tensor {source.name}: arrangement fields are invalid")
        try:
            arrangement = ArrangementV2(**{
                name: _require_int(raw_arrangement[name], f"arrangement.{name}")
                for name in ArrangementV2.__dataclass_fields__
            })
        except TypeError as exc:
            raise ValueError(f"converted tensor {source.name}: arrangement is invalid") from exc
        if arrangement != canonical_arrangement(qtype):
            raise ValueError(f"converted tensor {source.name}: arrangement is not canonical")
        carriers = record["carriers"]
        if not isinstance(carriers, dict) or set(carriers) != {"low", "high", "units"}:
            raise ValueError(f"converted tensor {source.name}: carriers must name low/high/units exactly")

        low_shape = [experts, n, k * spec.low_bits // 8]
        high_shape = [experts, n, k * spec.high_bits // 8] if spec.high_bits else [0]
        num_units = (k // 256) // spec.superblocks_per_unit
        units_shape = ([experts, num_units, n, spec.unit_bytes]
                       if grouped else [num_units, n, spec.unit_bytes])
        low_name = _carrier_name(position, "low")
        units_name = _carrier_name(position, "units")
        low = _read_carrier(reader_tensors, carriers["low"], low_name, low_shape, "low")
        units = _read_carrier(reader_tensors, carriers["units"], units_name, units_shape, "units")
        expected_carriers.update((low_name, units_name))
        if spec.high_bits:
            high_name = _carrier_name(position, "high")
            high = _read_carrier(reader_tensors, carriers["high"], high_name, high_shape, "high")
            expected_carriers.add(high_name)
        else:
            if carriers["high"] is not None:
                raise ValueError(f"converted tensor {source.name}: format has no high plane")
            high = torch.empty((0,), dtype=torch.uint8)

        artifact = KPackArtifact(
            low, high, units, arrangement, qtype, n, k, experts, grouped,
        )
        restored = recover_raw_blocks(artifact)
        source_blocks = _reader_tensor_blocks(source, spec, n, k, experts)
        if not torch.equal(restored, source_blocks):
            bad = int((restored != source_blocks).sum())
            raise ValueError(
                f"converted tensor {source.name}: recovered official GGUF blocks differ in {bad} bytes"
            )
    return expected_carriers


def _compare_source_file(source_path: pathlib.Path, output_reader) -> None:
    _, _, GGUFEndian, GGUFReader, _, _ = _gguf_modules()
    source = GGUFReader(str(source_path))
    if source.endianess != GGUFEndian.LITTLE:
        raise ValueError("source GGUF is not little-endian")
    source_meta = _real_metadata(source)
    output_meta = {
        name: field for name, field in _real_metadata(output_reader).items()
        if not name.startswith(REFERENCE_METADATA_PREFIX)
    }
    if set(source_meta) != set(output_meta):
        raise ValueError("source metadata keys were not preserved")
    for name in source_meta:
        if _field_fingerprint(source_meta[name]) != _field_fingerprint(output_meta[name]):
            raise ValueError(f"source metadata {name!r} was not preserved byte-exactly")
    output_tensors = {tensor.name: tensor for tensor in output_reader.tensors}
    for source_tensor in source.tensors:
        output_tensor = output_tensors.get(source_tensor.name)
        if output_tensor is None:
            raise ValueError(f"source tensor {source_tensor.name!r} was not preserved")
        if (source_tensor.tensor_type != output_tensor.tensor_type or
                list(source_tensor.shape) != list(output_tensor.shape) or
                _numpy_bytes(source_tensor.data) != _numpy_bytes(output_tensor.data)):
            raise ValueError(f"source tensor {source_tensor.name!r} was not preserved byte-exactly")


def verify_reference_gguf(path: Union[str, pathlib.Path], source_path=None) -> dict:
    """Verify one augmented reference GGUF and invert every K-pack artifact."""
    _, _, GGUFEndian, GGUFReader, GGUFValueType, _ = _gguf_modules()
    path = pathlib.Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"reference GGUF must be a regular file: {path}")
    reader = GGUFReader(str(path))
    if reader.endianess != GGUFEndian.LITTLE:
        raise ValueError("K-pack b16 carriers require a little-endian GGUF")
    manifest = _load_reference_manifest(reader, GGUFValueType)
    _verify_metadata_inventory(reader, manifest["source_metadata"])
    source_by_index, source_names = _verify_source_inventory(reader, manifest["source_tensors"])
    expected_carriers = _verify_converted(reader, manifest["tensors"], source_by_index)
    actual_names = {tensor.name for tensor in reader.tensors}
    if actual_names != source_names | expected_carriers:
        raise ValueError(
            "GGUF tensor inventory disagrees with the reference manifest: "
            f"missing={sorted((source_names | expected_carriers) - actual_names)} "
            f"extra={sorted(actual_names - (source_names | expected_carriers))}"
        )
    if source_path is not None:
        _compare_source_file(pathlib.Path(source_path), reader)
    return manifest


def pack_reference_gguf(
    source_path: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    *,
    tensor_names: Optional[Sequence[str]] = None,
) -> dict:
    """Write a self-verifying GGUF containing source tensors and K-pack companions."""
    np, _, GGUFEndian, GGUFReader, GGUFValueType, GGUFWriter = _gguf_modules()
    source_path = pathlib.Path(source_path)
    output_path = pathlib.Path(output_path)
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError(f"input GGUF must be a regular file: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("input and output GGUF paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output {output_path}")
    partial = output_path.with_name(output_path.name + f".partial.{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(f"refusing to reuse partial output {partial}")

    reader = GGUFReader(str(source_path))
    if reader.endianess != GGUFEndian.LITTLE:
        raise ValueError("K-pack b16 carriers require a little-endian input GGUF")
    selected = _select_reference_tensors(reader, tensor_names)
    selected_names = {tensor.name for tensor in selected}
    architecture = _source_architecture(reader)
    writer = GGUFWriter(
        str(partial), architecture, use_temp_file=True, endianess=GGUFEndian.LITTLE,
    )
    writer.data_alignment = int(reader.alignment)
    try:
        _copy_source_metadata(reader, writer, GGUFValueType)
        source_records = []
        source_indexes = {}
        for index, tensor in enumerate(reader.tensors):
            source_indexes[tensor.name] = index
            source_records.append(_source_tensor_record(index, tensor))
            # This is the copy path used by the official gguf metadata editor:
            # byte-shaped quantized arrays are converted back to logical shape
            # by raw_dtype, while their payload is copied without dequantizing.
            writer.add_tensor(
                tensor.name,
                tensor.data,
                raw_shape=tensor.data.shape,
                raw_dtype=tensor.tensor_type,
                tensor_endianess=reader.endianess,
            )

        converted = []
        for tensor in reader.tensors:
            if tensor.name not in selected_names:
                continue
            qtype = int(tensor.tensor_type)
            spec = SPECS[qtype]
            n, k, experts, grouped = _tensor_geometry(tensor, spec)
            blocks = _reader_tensor_blocks(tensor, spec, n, k, experts)
            artifact = (
                prepare_grouped(blocks, n, k, qtype, experts)
                if grouped else prepare_dense(blocks, n, k, qtype)
            )
            index = len(converted)
            carriers = {
                "low": _add_carrier(writer, _carrier_name(index, "low"), artifact.low),
                "high": (_add_carrier(writer, _carrier_name(index, "high"), artifact.high)
                         if artifact.high.numel() else None),
                "units": _add_carrier(writer, _carrier_name(index, "units"), artifact.units),
            }
            converted.append({
                "index": index,
                "source_index": source_indexes[tensor.name],
                "source_name": tensor.name,
                "qtype": qtype,
                "n": n,
                "k": k,
                "experts": experts,
                "grouped": grouped,
                "arrangement": asdict(artifact.arrangement),
                "carriers": carriers,
            })

        manifest = {
            "schema": REFERENCE_GGUF_SCHEMA,
            "schema_version": REFERENCE_GGUF_VERSION,
            "source_metadata": _source_metadata_records(reader),
            "source_tensors": source_records,
            "tensors": converted,
        }
        writer.add_string(REFERENCE_SCHEMA_KEY, REFERENCE_GGUF_SCHEMA)
        writer.add_uint32(REFERENCE_VERSION_KEY, REFERENCE_GGUF_VERSION)
        writer.add_string(
            REFERENCE_MANIFEST_KEY,
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        )
        _write_reference_gguf(writer)
    finally:
        writer.close()

    # A file becomes the named output only after the official reader can reopen
    # it, every inverse is byte-exact, and the original file copy is unchanged.
    verified = verify_reference_gguf(partial, source_path=source_path)
    try:
        os.link(partial, output_path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite output created during conversion: {output_path}") from exc
    partial.unlink()
    return verified


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read a GGUF, append canonical K-pack reference tensors, and verify byte-exact inversion."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="write a self-verifying augmented GGUF")
    pack.add_argument("input", help="source GGUF")
    pack.add_argument("output", help="new augmented GGUF; must not already exist")
    pack.add_argument(
        "--tensor", action="append", dest="tensors",
        help="convert exactly this tensor name; repeat for more than one (default: all supported rank-2/3 tensors)",
    )
    verify = commands.add_parser("verify", help="verify an augmented GGUF and every inverse")
    verify.add_argument("gguf", help="augmented reference GGUF")
    verify.add_argument(
        "--source", help="also compare all original tensors and metadata with this source GGUF",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            manifest = pack_reference_gguf(args.input, args.output, tensor_names=args.tensors)
            print(
                f"KPACK_REFERENCE_GGUF PASS action=pack converted={len(manifest['tensors'])} "
                f"source_tensors={len(manifest['source_tensors'])} output={args.output}"
            )
        else:
            manifest = verify_reference_gguf(args.gguf, source_path=args.source)
            print(
                f"KPACK_REFERENCE_GGUF PASS action=verify converted={len(manifest['tensors'])} "
                f"source_tensors={len(manifest['source_tensors'])} input={args.gguf}"
            )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"KPACK_REFERENCE_GGUF FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = [
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "ArrangementV2", "KPackArtifact", "SPECS",
    "canonical_arrangement", "prepare_dense", "prepare_grouped",
    "recover_raw_blocks", "pack_reference_gguf", "verify_reference_gguf", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

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

Only PyTorch is required.  No quactlize extension, PPU SDK, CuTe or device is
loaded.  ``recover_raw_blocks`` is included solely to make byte-exact round-trip
tests possible.

Typical use::

    artifact = prepare_dense(raw_blocks, n=5120, k=8192, qtype="Q4_K")
    low, high, units = artifact.low, artifact.high, artifact.units
    assert torch.equal(recover_raw_blocks(artifact), raw_blocks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

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


__all__ = [
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "ArrangementV2", "KPackArtifact", "SPECS",
    "canonical_arrangement", "prepare_dense", "prepare_grouped",
    "recover_raw_blocks",
]

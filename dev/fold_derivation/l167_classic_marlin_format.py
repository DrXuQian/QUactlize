#!/usr/bin/env python3
"""Host oracle for the classic Marlin int4-B and gs128 scale formats.

This deliberately derives the same bytes through two independent source
descriptions:

* the direct lane/provider formula in test_marlin_classic_group.cu; and
* the reshape/permutation formula in awesome-cute's marlin.py.

Unique integer provider ids are compared before they are truncated to int4.
That keeps an asymmetric fixture from hiding a legal-but-wrong permutation.
The actual int4 bytes and both inverse maps are checked separately.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_CLASSIC = Path("/root/marlin_ppu/test_marlin_classic_group.cu")
DEFAULT_AWESOME = Path(
    "/root/marlin_ppu/ref/awesome-cute/gemm/marlin_gemm/marlin.py"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def bind_source(path: Path, needles: Sequence[str]) -> str:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    require(not missing, f"{path}: source anchor drifted; missing {missing!r}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classic_source_hash(path: Path) -> str:
    return bind_source(
        path,
        (
            "int n = nblock * 16 + lane / 4, kb = ktile * 16 + (lane % 4) * 2",
            "q |= Bu(n, kb)         << 0;  q |= Bu(n, kb + 1)     << 16;",
            "q |= Bu(n + 8, kb)     << 8;  q |= Bu(n + 8, kb + 1) << 24;",
            "hB[idx * 4 + (nblock % 4)] = q;",
            "marlin_classic_ppu::marlin_permute_scales",
        ),
    )


def awesome_source_hash(path: Path) -> str:
    return bind_source(
        path,
        (
            "perm = perm.reshape((-1, 8))[:, interleave].ravel()",
            "scale_perm.extend([i + 8 * j for j in range(8)])",
            "w = w.permute((0, 2, 1, 3))",
            "res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)",
            "q |= res[:, i::8] << 4 * i",
        ),
    )


def coord(k: int, n: int, n_size: int) -> int:
    return k * n_size + n


def classic_provider_slots(
    k_size: int, n_size: int, *, wrong_provider_pitch: bool = False
) -> tuple[list[int], int]:
    """Return the logical (k,n) provider for every packed nibble slot.

    The planted negative treats ``idx`` as the uint32 word index and adds the
    provider directly, omitting the shipping ``idx * 4`` provider pitch.  It
    is the code-vs-word-unit failure that an int4-only fixture can conceal.
    """

    require(k_size % 16 == 0, "K must be divisible by 16")
    require(n_size % 64 == 0, "N must be divisible by 64")
    providers = [-1] * (k_size * n_size)
    collisions = 0

    for ktile in range(k_size // 16):
        for nblock in range(n_size // 16):
            for lane in range(32):
                n = nblock * 16 + lane // 4
                kb = ktile * 16 + (lane % 4) * 2
                idx = (n_size // 2) * ktile + (nblock // 4) * 32 + lane
                provider = nblock % 4
                word = idx + provider if wrong_provider_pitch else idx * 4 + provider
                entries = (
                    (0, kb, n),
                    (4, kb + 1, n),
                    (1, kb + 8, n),
                    (5, kb + 9, n),
                    (2, kb, n + 8),
                    (6, kb + 1, n + 8),
                    (3, kb + 8, n + 8),
                    (7, kb + 9, n + 8),
                )
                for nibble, k, column in entries:
                    slot = word * 8 + nibble
                    require(slot < len(providers), "planted map escaped the allocation")
                    if providers[slot] != -1:
                        collisions += 1
                    providers[slot] = coord(k, column, n_size)

    return providers, collisions


def awesome_weight_permutation() -> list[int]:
    perm: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        column = i // 4
        for block in (0, 1):
            for row in (
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ):
                perm1.append(16 * row + column + 8 * block)
        for j in range(4):
            perm.extend(value + 256 * j for value in perm1)

    interleave = (0, 2, 4, 6, 1, 3, 5, 7)
    perm = [
        perm[base + lane]
        for base in range(0, len(perm), 8)
        for lane in interleave
    ]
    require(sorted(perm) == list(range(1024)), "weight map is not a permutation")
    return perm


def awesome_provider_slots(k_size: int, n_size: int) -> list[int]:
    require(k_size % 16 == 0, "K must be divisible by 16")
    require(n_size % 64 == 0, "N must be divisible by 64")
    perm = awesome_weight_permutation()
    slots: list[int] = []
    for ktile in range(k_size // 16):
        # w.reshape(K/16,16,N/16,16).permute(0,2,1,3)
        row = [
            coord(ktile * 16 + ki, nblock * 16 + ni, n_size)
            for nblock in range(n_size // 16)
            for ki in range(16)
            for ni in range(16)
        ]
        require(len(row) % len(perm) == 0, "row cannot be tiled by _perm")
        for base in range(0, len(row), len(perm)):
            chunk = row[base : base + len(perm)]
            slots.extend(chunk[source] for source in perm)
    return slots


def asymmetric_int4(k: int, n: int) -> int:
    # Deliberately not invariant under K/N transpose, 8/16-lane exchange, or
    # any single affine stride.  Provider ids above remain the collision-free
    # primary oracle; this is the actual-byte fixture.
    return (
        7 * k
        + 11 * n
        + 3 * (k // 4)
        + 5 * (n // 8)
        + ((k & 15) * (n & 7))
    ) & 0xF


def logical_codes(k_size: int, n_size: int) -> list[int]:
    return [asymmetric_int4(k, n) for k in range(k_size) for n in range(n_size)]


def pack_from_providers(codes: Sequence[int], providers: Sequence[int]) -> list[int]:
    require(len(providers) % 8 == 0, "provider count must be a whole uint32")
    require(all(provider >= 0 for provider in providers), "provider map has holes")
    words: list[int] = []
    for base in range(0, len(providers), 8):
        word = 0
        for nibble in range(8):
            word |= (codes[providers[base + nibble]] & 0xF) << (4 * nibble)
        words.append(word)
    return words


def unpack_classic(words: Sequence[int], k_size: int, n_size: int) -> list[int]:
    providers, collisions = classic_provider_slots(k_size, n_size)
    require(collisions == 0, "classic inverse saw a provider collision")
    out = [-1] * (k_size * n_size)
    visits = [0] * len(out)
    for slot, provider in enumerate(providers):
        out[provider] = (words[slot // 8] >> (4 * (slot % 8))) & 0xF
        visits[provider] += 1
    require(all(visit == 1 for visit in visits), "classic inverse is not bijective")
    return out


def unpack_awesome(words: Sequence[int], k_size: int, n_size: int) -> list[int]:
    perm = awesome_weight_permutation()
    packed = [
        (word >> (4 * nibble)) & 0xF for word in words for nibble in range(8)
    ]
    out = [-1] * (k_size * n_size)
    row_size = n_size * 16
    require(len(packed) == (k_size // 16) * row_size, "packed shape mismatch")
    for ktile in range(k_size // 16):
        post = packed[ktile * row_size : (ktile + 1) * row_size]
        pre: list[int] = []
        for base in range(0, row_size, len(perm)):
            chunk = post[base : base + len(perm)]
            recovered = [-1] * len(perm)
            for destination, source in enumerate(perm):
                recovered[source] = chunk[destination]
            require(all(value >= 0 for value in recovered), "inverse perm has a hole")
            pre.extend(recovered)

        offset = 0
        for nblock in range(n_size // 16):
            for ki in range(16):
                for ni in range(16):
                    out[coord(ktile * 16 + ki, nblock * 16 + ni, n_size)] = pre[
                        offset
                    ]
                    offset += 1
    require(all(value >= 0 for value in out), "awesome inverse has a hole")
    return out


def scale_permutation() -> list[int]:
    perm = [i + 8 * j for i in range(8) for j in range(8)]
    require(sorted(perm) == list(range(64)), "scale map is not a permutation")
    return perm


def classic_scale_permute(plain: Sequence[int], groups: int, n_size: int) -> list[int]:
    require(n_size % 64 == 0, "N must be divisible by 64")
    require(len(plain) == groups * n_size, "plain scale shape mismatch")
    out = [-1] * len(plain)
    for group in range(groups):
        for c0 in range(0, n_size, 64):
            for i in range(8):
                for j in range(8):
                    out[group * n_size + c0 + 8 * i + j] = plain[
                        group * n_size + c0 + i + 8 * j
                    ]
    return out


def awesome_scale_permute(plain: Sequence[int]) -> list[int]:
    perm = scale_permutation()
    require(len(plain) % len(perm) == 0, "scale array is not 64-aligned")
    return [
        plain[base + source]
        for base in range(0, len(plain), len(perm))
        for source in perm
    ]


def scale_unpermute(packed: Sequence[int]) -> list[int]:
    perm = scale_permutation()
    out: list[int] = []
    for base in range(0, len(packed), len(perm)):
        chunk = packed[base : base + len(perm)]
        recovered = [-1] * len(perm)
        for destination, source in enumerate(perm):
            recovered[source] = chunk[destination]
        require(all(value >= 0 for value in recovered), "scale inverse has a hole")
        out.extend(recovered)
    return out


def count_mismatches(lhs: Sequence[int], rhs: Sequence[int]) -> int:
    require(len(lhs) == len(rhs), "mismatch operands have different sizes")
    return sum(a != b for a, b in zip(lhs, rhs))


def words_to_bytes(words: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<I", word & 0xFFFFFFFF) for word in words)


def scales_to_bytes(scales: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<H", scale & 0xFFFF) for scale in scales)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classic-source", type=Path, default=DEFAULT_CLASSIC)
    parser.add_argument("--awesome-source", type=Path, default=DEFAULT_AWESOME)
    args = parser.parse_args()

    classic_hash = classic_source_hash(args.classic_source)
    awesome_hash = awesome_source_hash(args.awesome_source)

    k_size, n_size, group_size = 512, 256, 128
    groups = k_size // group_size
    codes = logical_codes(k_size, n_size)

    direct_providers, direct_collisions = classic_provider_slots(k_size, n_size)
    perm_providers = awesome_provider_slots(k_size, n_size)
    require(direct_collisions == 0, "shipping direct map collided")
    provider_mismatches = count_mismatches(direct_providers, perm_providers)
    require(provider_mismatches == 0, "the two independent B maps disagree")
    require(sorted(direct_providers) == list(range(k_size * n_size)), "B map not bijective")

    direct_words = pack_from_providers(codes, direct_providers)
    perm_words = pack_from_providers(codes, perm_providers)
    require(direct_words == perm_words, "packed B words disagree")
    classic_roundtrip = count_mismatches(unpack_classic(direct_words, k_size, n_size), codes)
    awesome_roundtrip = count_mismatches(unpack_awesome(direct_words, k_size, n_size), codes)
    require(classic_roundtrip == 0, "classic B roundtrip failed")
    require(awesome_roundtrip == 0, "awesome B roundtrip failed")
    packed_bytes = words_to_bytes(direct_words)

    # Unique tags preserve every (group,column) coordinate exactly; these are
    # representation tags, not floating-point values.
    plain_scales = [group * n_size + column + 1 for group in range(groups) for column in range(n_size)]
    classic_scales = classic_scale_permute(plain_scales, groups, n_size)
    awesome_scales = awesome_scale_permute(plain_scales)
    scale_mismatches = count_mismatches(classic_scales, awesome_scales)
    scale_roundtrip = count_mismatches(scale_unpermute(classic_scales), plain_scales)
    require(scale_mismatches == 0, "the two scale maps disagree")
    require(scale_roundtrip == 0, "scale roundtrip failed")
    scale_bytes = scales_to_bytes(classic_scales)

    wrong_providers, wrong_collisions = classic_provider_slots(
        k_size, n_size, wrong_provider_pitch=True
    )
    wrong_holes = sum(provider < 0 for provider in wrong_providers)
    wrong_mismatches = count_mismatches(wrong_providers, direct_providers)
    require(wrong_collisions > 0, "planted code-vs-word pitch error did not collide")
    require(wrong_holes > 0, "planted code-vs-word pitch error left no holes")
    require(wrong_mismatches > 0, "planted code-vs-word pitch error was not detected")

    # Identity is a plausible but wrong gs128 scale assumption.  The unique
    # coordinate tags make every non-fixed point visible.
    identity_scale_mismatches = count_mismatches(plain_scales, classic_scales)
    require(identity_scale_mismatches > 0, "planted identity scale map stayed green")

    print(
        "[L167][anchors] "
        f"classic={classic_hash[:16]} awesome={awesome_hash[:16]}"
    )
    print(
        "[L167][B-provider] "
        f"K={k_size} N={n_size} unique={len(direct_providers)} "
        f"direct-vs-permutation={provider_mismatches} bijective=1"
    )
    print(
        "[L167][B-bytes] "
        f"words={len(direct_words)} bytes={len(packed_bytes)} "
        f"classic-roundtrip={classic_roundtrip} awesome-roundtrip={awesome_roundtrip} "
        f"sha256={hashlib.sha256(packed_bytes).hexdigest()}"
    )
    print(
        "[L167][gs128-scale] "
        f"groups={groups} entries={len(classic_scales)} "
        f"direct-vs-permutation={scale_mismatches} roundtrip={scale_roundtrip} "
        f"sha256={hashlib.sha256(scale_bytes).hexdigest()}"
    )
    print(
        "[L167][negative-code-vs-word-pitch] "
        f"provider-mismatches={wrong_mismatches} collisions={wrong_collisions} "
        f"holes={wrong_holes} EXPECTED_RED"
    )
    print(
        "[L167][negative-scale-identity] "
        f"mismatches={identity_scale_mismatches} EXPECTED_RED"
    )
    print(
        "[L167] PASS: independent classic/direct and Awesome-CuTe/permutation "
        "anchors agree; asymmetric provider, byte, inverse, and negative controls proved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

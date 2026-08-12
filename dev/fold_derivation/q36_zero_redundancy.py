#!/usr/bin/env python3
"""Constructive proof that Q3_K/Q6_K's external fp16 zero is derived from scale.

This deliberately reads the artifacts returned by the production torch ops.  It
does not inspect the implementation's constants and call that evidence: each
external zero plane is reconstructed from the *returned scale plane alone* and
compared as uint16 bits.  The official ``gguf`` package anchors the raw record
shape and signed-code weight semantics independently of quactlize.

The fully-quantized packed producer is a different proposition: it already
returns only ``(low, high, unit)`` and therefore stores no external fp16 zero
plane.  Its diagnostic decoder is passed a correction explicitly.  That arm
proves the unit has enough scale information to derive the requested logical
zero; it does *not* prove that a particular device collective selected the same
``kPackedZMul``.

The dense and fully-quantized producers require ``QUACTLIZE_PPU_LIB``.  The
pytest gate builds the repository's CUDA stand-in locally; this script itself
fails closed when the supplied library lacks either producer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

import gguf
from gguf.constants import GGMLQuantizationType as GT
from gguf.constants import GGML_QUANT_SIZES

import quactlize


@dataclass(frozen=True)
class FormatResult:
    name: str
    scale_first_elements: int
    dense_elements: int
    packed_decode_elements: int
    scale_first_bad: int
    dense_bad: int
    packed_decode_bad: int
    official_max_block_relative_error: float
    wrong_bias_witnesses: int
    packed_perturbation_witnesses: int
    targeted_dense_actual_vs_staged_bad: int
    targeted_dense_rounding_witnesses: int


_FORMATS = {
    "Q3_K": (GT.Q3_K, 11, 110, (108, 110), 4, -4),
    "Q6_K": (GT.Q6_K, 14, 210, (208, 210), 32, -24),
}


def _half_bits(a: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().contiguous().numpy()
    return np.asarray(a, dtype=np.float16).view(np.uint16)


def _half_mul(scale: np.ndarray, factor: int) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return (scale.astype(np.float32) * np.float32(factor)).astype(np.float16)


def _half_add_from_positive_zero(scale: np.ndarray, factor: int) -> np.ndarray:
    """Packed diagnostic decoder's exact expression, including signed zero."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (
            np.zeros(scale.shape, dtype=np.float32)
            + np.float32(factor) * scale.astype(np.float32)
        ).astype(np.float16)


def _dense_rebuild(name: str, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (shipping reconstruction, tempting-but-wrong direct form).

    Q6's dense producer performs two fp16 roundings: first the scale-first
    ``-32*S`` plane, then ``half(float(z0) + 8*float(S))`` for the int4
    converter's bias.  Collapsing it to ``half(-24*S)`` changes real bits, so
    retaining the two stages is part of the proposition being proved.
    """
    if name == "Q3_K":
        z = _half_mul(scale, -4)
        return z, z.copy()  # no second rounding stage in this path
    z0 = _half_mul(scale, -32)
    with np.errstate(over="ignore", invalid="ignore"):
        staged = (z0.astype(np.float32) + np.float32(8) * scale.astype(np.float32)).astype(np.float16)
    direct = _half_mul(scale, -24)
    return staged, direct


def _raw_blocks(name: str, count: int) -> np.ndarray:
    gt, _qtype, block_bytes, hdr, _bias, _packed = _FORMATS[name]
    assert GGML_QUANT_SIZES[gt] == (256, block_bytes)
    rng = np.random.default_rng(0x203600 + int(gt))
    raw = rng.integers(0, 256, size=(count, block_bytes), dtype=np.uint8)
    # Normal, finite fp16 d values.  The main bitwise witness and the official
    # anchor intentionally contain no overflow-control block: a very large
    # official value must not hide errors in unrelated blocks.
    d = (rng.random(count) * np.float32(0.1) + np.float32(0.001)).astype(np.float16)
    raw[:, hdr[0] : hdr[1]] = d.view(np.uint8).reshape(count, 2)
    return raw


def _q6_rounding_control_blocks(count: int) -> np.ndarray:
    """Separate legal fixture that distinguishes staged and direct Q6 zero.

    This control never enters the main 8,192-element bitwise witness or the
    official-GGUF anchor.  Its first block has S=20*125=2500: ``half(-32*S)``
    overflows before the later ``+8*S``, while direct ``half(-24*S)`` remains
    finite.  The purpose is to prove that the two-stage expression is observed,
    not to suggest that checkpoint scales usually approach this boundary.
    """
    raw = _raw_blocks("Q6_K", count)
    _gt, _qtype, _block_bytes, hdr, _bias, _packed = _FORMATS["Q6_K"]
    # Make every non-target group exact under both expressions: S=1 gives
    # half(half(-32) + 8) == half(-24).  Thus the witness count cannot be
    # supplied accidentally by the random part of this separate fixture.
    raw[:, 192:208] = np.uint8(1)
    raw[:, hdr[0] : hdr[1]] = np.asarray([1.0], dtype=np.float16).view(np.uint8)
    raw[0, 192] = np.uint8(125)
    raw[0, hdr[0] : hdr[1]] = np.asarray([20.0], dtype=np.float16).view(np.uint8)
    return raw


def _official_anchor(name: str, raw: np.ndarray) -> float:
    """Anchor both record structure and the absence of an independent min.

    These size identities are from the installed official GGUF package.  Every
    byte is exhausted by codes, scale integers and one fp16 ``d``; unlike
    Q2/Q4/Q5 there is no ``dmin`` or min field left to encode a zero.
    """
    gt, qtype, block_bytes, _hdr, _bias, _packed = _FORMATS[name]
    expected = 32 + 64 + 12 + 2 if name == "Q3_K" else 128 + 64 + 16 + 2
    assert block_bytes == expected
    assert GGML_QUANT_SIZES[gt] == (256, expected)

    blocks = torch.from_numpy(raw)
    codes, scale, raw_zero = quactlize.gguf_unpack(blocks, qtype)
    # The raw signed-code representation needs no affine zero at all.  The
    # production accessor currently returns -0 (the generic ``-dmin*0`` shape),
    # so mask only the sign bit: either signed zero has zero information, while
    # any nonzero magnitude would disprove the claim.
    assert np.count_nonzero(_half_bits(raw_zero) & np.uint16(0x7FFF)) == 0
    groups = int(scale.shape[1])
    gsz = 256 // groups
    reconstructed = (
        codes.numpy().astype(np.float64).reshape(-1, groups, gsz)
        * scale.numpy().astype(np.float64)[:, :, None]
    ).reshape(-1, 256)
    official = gguf.quants.dequantize(raw.reshape(-1), gt).reshape(-1, 256).astype(np.float64)
    assert np.isfinite(official).all()
    # Normalise independently per official GGUF block.  A single high-amplitude
    # block must not lower the reported error of every other block, which is
    # exactly what one global max denominator would do.
    abs_error = np.abs(reconstructed - official)
    block_denominator = np.maximum(np.max(np.abs(official), axis=1), 1e-30)
    block_relative_error = np.max(abs_error, axis=1) / block_denominator
    rel = float(np.max(block_relative_error))
    # Both paths use the same fp16 d stored by GGUF.  The producer rounds each
    # group scale to fp16 while official gguf retains float32, hence a small
    # fp16-scale difference is expected; a missing independent min is O(1).
    assert rel < 1.0e-3, (name, rel)
    return rel


def _q6_dense_rounding_control(n: int, k: int, qtype: int) -> tuple[int, int]:
    """Run the actual dense producer on a fixture reserved for this control."""
    blocks = torch.from_numpy(_q6_rounding_control_blocks(n * (k // 256)))
    _low, _high, scale_t, zero_t = quactlize.gguf_prepare_dense(blocks, n, k, qtype)
    scale = scale_t.numpy()
    zero = zero_t.numpy()
    staged, direct = _dense_rebuild("Q6_K", scale)
    actual_vs_staged_bad = int(np.count_nonzero(_half_bits(zero) != _half_bits(staged)))
    witnesses = int(np.count_nonzero(_half_bits(staged) != _half_bits(direct)))
    return actual_vs_staged_bad, witnesses


def prove_one(name: str) -> FormatResult:
    gt, qtype, _block_bytes, _hdr, bias, packed_zmul = _FORMATS[name]
    del gt
    n, k = 256, 512
    raw = _raw_blocks(name, n * (k // 256))
    blocks = torch.from_numpy(raw)
    official_rel = _official_anchor(name, raw)

    # 1. Scale-first GEMV: the actual producer's zero is exactly -bias*scale.
    _low, _high, sf_scale_t, sf_zero_t = quactlize.gguf_prepare_gemv(blocks, n, k, qtype)
    sf_scale = sf_scale_t.numpy()
    sf_zero = sf_zero_t.numpy()
    sf_rebuilt = _half_mul(sf_scale, -bias)
    sf_bad = int(np.count_nonzero(_half_bits(sf_zero) != _half_bits(sf_rebuilt)))

    wrong_bias = -32 if name == "Q3_K" else -4
    wrong = _half_mul(sf_scale, wrong_bias)
    wrong_bias_witnesses = int(np.count_nonzero(_half_bits(sf_zero) != _half_bits(wrong)))

    # 2. Placed dense: same returned scale, with Q6's exact two-rounding path.
    _dlow, _dhigh, dense_scale_t, dense_zero_t = quactlize.gguf_prepare_dense(blocks, n, k, qtype)
    dense_scale = dense_scale_t.numpy()
    dense_zero = dense_zero_t.numpy()
    dense_rebuilt, _dense_direct = _dense_rebuild(name, dense_scale)
    dense_bad = int(np.count_nonzero(_half_bits(dense_zero) != _half_bits(dense_rebuilt)))

    # 3. The actual fully-quantized producer returns exactly low/high/unit: it
    # already has no external fp16 zero plane.  The diagnostic decoder accepts
    # a caller-selected correction; after that correction is supplied, logical
    # zero is bitwise derived from the decoded scale.  This does not inspect or
    # validate a device collective's independent kPackedZMul selection.
    _flow, _fhigh, units = quactlize.gguf_prepare_fully_quantized_dense(blocks, n, k, qtype)
    packed_scale_t, packed_zero_t = quactlize.gguf_unit_decode(units, qtype, packed_zmul)
    packed_scale = packed_scale_t.numpy()
    packed_zero = packed_zero_t.numpy()
    packed_rebuilt = _half_add_from_positive_zero(packed_scale, packed_zmul)
    packed_decode_bad = int(np.count_nonzero(_half_bits(packed_zero) != _half_bits(packed_rebuilt)))

    # A one-ULP injection must be visible in the packed *diagnostic decode*.
    # This control says nothing about the two external fp16 producer arms.
    perturbed = _half_bits(packed_rebuilt).copy()
    finite_nonzero = np.flatnonzero(np.isfinite(packed_rebuilt.reshape(-1)) & (packed_rebuilt.reshape(-1) != 0))
    assert finite_nonzero.size
    perturbed.reshape(-1)[finite_nonzero[0]] ^= np.uint16(1)
    packed_perturbation_witnesses = int(np.count_nonzero(perturbed != _half_bits(packed_zero)))

    targeted_dense_actual_vs_staged_bad, targeted_dense_rounding_witnesses = (
        _q6_dense_rounding_control(n, k, qtype) if name == "Q6_K" else (0, 0)
    )

    assert sf_bad == dense_bad == packed_decode_bad == 0
    assert wrong_bias_witnesses > 0
    assert packed_perturbation_witnesses > 0
    if name == "Q6_K":
        assert targeted_dense_actual_vs_staged_bad == 0
        assert targeted_dense_rounding_witnesses > 0
    else:
        assert targeted_dense_rounding_witnesses == 0
    return FormatResult(
        name=name,
        scale_first_elements=int(sf_zero.size),
        dense_elements=int(dense_zero.size),
        packed_decode_elements=int(packed_zero.size),
        scale_first_bad=sf_bad,
        dense_bad=dense_bad,
        packed_decode_bad=packed_decode_bad,
        official_max_block_relative_error=official_rel,
        wrong_bias_witnesses=wrong_bias_witnesses,
        packed_perturbation_witnesses=packed_perturbation_witnesses,
        targeted_dense_actual_vs_staged_bad=targeted_dense_actual_vs_staged_bad,
        targeted_dense_rounding_witnesses=targeted_dense_rounding_witnesses,
    )


def main() -> int:
    if not __debug__:
        raise SystemExit(
            "L139 FAIL: Python assertions are disabled; this proof must not run under -O/PYTHONOPTIMIZE because "
            "its structural and negative controls are assertions")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = [prove_one(name) for name in _FORMATS]
    payload = {
        "backend": quactlize.gguf_backend(),
        "claim": (
            "external-fp16-zero-is-structurally-derived; "
            "packed-unit-already-has-no-external-zero; physical-plane-removal-not-implemented"
        ),
        "scope": {
            "structural_formulas": True,
            "bitwise_witness_elements_per_arm": 8192,
            "sample_is_not_exhaustive": True,
            "packed_collective_kPackedZMul_proved": False,
        },
        "formats": [asdict(r) for r in results],
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(payload["claim"])
        for r in results:
            print(
                f"{r.name}: external sf/dense bad={r.scale_first_bad}/{r.dense_bad}; "
                f"packed diagnostic bad={r.packed_decode_bad}; "
                f"elements sf/dense/packed={r.scale_first_elements}/{r.dense_elements}/"
                f"{r.packed_decode_elements}; "
                f"official_max_block_rel={r.official_max_block_relative_error:.3e}; "
                f"negative witnesses bias/packed-perturb/targeted-round={r.wrong_bias_witnesses}/"
                f"{r.packed_perturbation_witnesses}/{r.targeted_dense_rounding_witnesses}; "
                f"targeted actual-vs-staged bad={r.targeted_dense_actual_vs_staged_bad}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

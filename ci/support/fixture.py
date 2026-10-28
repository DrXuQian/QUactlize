"""Product correctness-test support; no performance campaign."""
import hashlib
import numpy as np
from reference import gguf_kpack as ref
from ci.support.weights import pack_plane, prepare_expert
def bf16_bits(values):
    values = np.asarray(values, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("BF16 fixture input must be finite")
    bits = values.view("<u4")
    shift = np.uint32(16)
    return ((bits + np.uint32(0x7fff) + ((bits >> shift) & np.uint32(1))) >> shift).astype("<u2")


def bf16_float(bits):
    # NumPy 1.x can widen a uint32 scalar shifted by a Python int to int64.
    # Keep the operation explicitly 32-bit for scalar and array bit views.
    return (np.asarray(bits, dtype="<u2").astype("<u4") << np.uint32(16)).view("<f4")


def round_compute(values, compute):
    if compute == "bf16":
        return bf16_float(bf16_bits(values))
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(values, dtype="<f2").astype("<f4")


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def raw_weight(q, n, k, seed):
    """Nonzero, sign-varying codes and dyadic scales: exact in both 16-bit types.

    Official GGUF dequantization is the golden. Dyadic group scales make the
    two-step TC reconstruction equal to that golden, so the gate does not
    accidentally compare BF16 TC with an unrounded FP32 weight expression.
    The format setter is only a fixture producer; the official decoder does
    not consume any placed plane or metadata produced by prepare_expert.
    """
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    block, width = (32, 34) if q == 8 else (256, ref.SPECS[q].raw_bytes)
    raw = np.random.default_rng(seed).integers(0, 256, (n, k // block, width), dtype="u1")
    tags = (np.arange(n)[:, None] * 3 + np.arange(k // block)[None, :] + seed) % 4
    d = np.ldexp(np.ones(tags.shape, dtype="f4"), -(tags + 8)).astype("<f2")
    if q == 8:
        raw[..., :2] = d.view("u1").reshape(n, -1, 2)
    else:
        spec = ref.SPECS[q]
        template = bytearray(width)
        for g in range(spec.groups):
            ref._metadata_put(template, 0, spec, g, 33 if q == 11 else 1, 1)
        metadata_size = 16 if q in (10, 14) else 12
        raw[..., spec.scale_offset:spec.scale_offset + metadata_size] = np.frombuffer(
            template, dtype="u1")[spec.scale_offset:spec.scale_offset + metadata_size]
        raw[..., spec.d_offset:spec.d_offset + 2] = d.view("u1").reshape(n, -1, 2)
        if spec.has_min:
            raw[..., spec.dmin_offset:spec.dmin_offset + 2] = (d / np.float16(2)).astype("<f2").view("u1").reshape(n, -1, 2)
    golden = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(n, k).astype("f4")
    if (not np.isfinite(golden).all() or not np.any(golden) or
            not np.array_equal(golden, round_compute(golden, "bf16")) or
            not np.array_equal(golden, round_compute(golden, "f16"))):
        raise ValueError("fixture is not a nonzero common exact-weight oracle")
    return raw, golden


def planes(raw, q):
    n, sb, _ = raw.shape
    if q != 8:
        return prepare_expert(raw, q, n, sb * 256)
    codes = raw[..., 2:].reshape(n, sb * 32)
    scale = raw[..., :2].copy().view("<f2").reshape(n, sb).T.copy()
    return dict(low=pack_plane(codes ^ 128, 8), high=np.empty(0, "u1"), units=scale,
                scale=scale, zero=np.empty(0, "u1"))


def compare(got, gold, denom, tolerance=0.005):
    got, gold, denom = (np.asarray(x, dtype="f8") for x in (got, gold, denom))
    bad_finite = ~np.isfinite(got)
    errors = np.abs(got - gold) / np.maximum(denom, 1e-30)
    error = float(np.max(errors))
    bad = bad_finite | (errors >= tolerance)
    indices = np.flatnonzero(bad)
    result = dict(cells=got.size, bad=int(bad.sum()), nonfinite=int(bad_finite.sum()),
                  relative_l1_error=error, tolerance=tolerance,
                  output_sha256=digest(got), golden_sha256=digest(gold), first=None)
    if indices.size:
        at = int(indices[0])
        result["first"] = dict(index=at, got=float(got.flat[at]), want=float(gold.flat[at]))
    if not np.isfinite(gold).all() or not np.isfinite(denom).all() or not np.any(gold):
        raise ValueError("nonfinite or all-zero oracle")
    if np.any(bad) or not np.isfinite(error):
        raise ValueError("typed oracle mismatch: " + str(result))
    return result

"""Bounded rounding check for the paired-N4 *synthetic large-BF16* replay.

Not a replacement GEMM tolerance. Only exceptions to the existing output
check are examined, and only saturated positive gates are certifiable: their
SwiGLU is the exact F32 product of two BF16 values. Enumerate the discrete
BF16 bins reachable within an F32 dot forward-error bound; do not accept an
arbitrary value in a continuous error interval. Unsupported cases fail closed.
"""

import numpy as np

from dev.bf16_compute.fixture import bf16_bits, bf16_float, round_compute


def _key(bits):
    bits = int(bits)
    return 0xFFFF - bits if bits & 0x8000 else bits + 0x8000


def _value(key):
    return bf16_float(0xFFFF - key if key < 0x8000 else key - 0x8000)


def projection_bins(a, weight, split):
    """Independent original GGUF weights, never placed/consumer coordinates.

    BF16 products are exact in F32 in this fixture. For any F32 accumulation
    tree, gamma_(K+S) * sum(abs(a*w)) bounds rounding, including S-way reduction.
    The much smaller F64 oracle bound is added too. No under/overflow or
    non-BF16 operands are admitted. A bound spanning >8 bins is inconclusive.
    """
    a, weight = [np.asarray(x, dtype="f4") for x in (a, weight)]
    if a.ndim != 1 or weight.shape != a.shape or split not in (1, 2, 4, 8):
        raise ValueError("rounding certificate: invalid dot geometry")
    for x in (a, weight):
        if not np.isfinite(x).all() or not np.array_equal(x, round_compute(x, "bf16")):
            raise ValueError("rounding certificate: operands must be finite exact BF16")
    products = a.astype("f8") * weight.astype("f8")
    tiny, limit = np.finfo("f4").tiny, np.finfo("f4").max
    absolute = np.abs(products)
    l1 = float(absolute.sum())
    if (
        not 0 < l1 < limit / 2
        or np.any((absolute != 0) & (absolute < tiny))
        or not np.array_equal(products, products.astype("f4").astype("f8"))
    ):
        raise ValueError(
            "rounding certificate: non-exact products or exceptional range"
        )
    length = len(a) + split
    nu = length * 2.0**-24
    if nu >= 0.01:
        raise ValueError("rounding certificate: unsupported accumulation length")
    oracle_nu = (len(a) + 1) * 2.0**-53
    exact = float(products.sum())
    oracle_gamma = oracle_nu / (1 - oracle_nu)
    bound = (nu / (1 - nu) + oracle_gamma) * l1 / (1 - oracle_gamma)
    # Outward F32 endpoints also cover conversion of the independent F64 sum.
    ends = np.array(
        [
            np.nextafter(np.float32(exact - bound), np.float32(-np.inf)),
            np.nextafter(np.float32(exact + bound), np.float32(np.inf)),
        ],
        dtype="f4",
    )
    lower, upper = map(_key, bf16_bits(ends))
    if not 0 <= upper - lower < 8:
        raise ValueError("rounding certificate: dot bound spans too many BF16 bins")
    values = np.array([_value(key) for key in range(lower, upper + 1)], dtype="f4")
    if not np.isfinite(values).all():
        raise ValueError("rounding certificate: nonfinite rounded projection")
    return dict(dot=exact, absolute_sum=l1, f32_bound=bound, bf16=values.tolist())


def certify_output(got, gold, activation, weights, owners, a_rows, split):
    """Return evidence, or reject. Original 0.005 global error stays recorded.

    Caller must separately check buffer guards, and enable this only for the
    TC large-BF16/F32-storage/F32-output rounded-projection replay. Normal cases
    and the diagnostic retain their original checker.
    """
    got, gold = [np.asarray(x, dtype="f4") for x in (got, gold)]
    if (
        got.ndim != 2
        or gold.shape != got.shape
        or not all(np.isfinite(x).all() for x in (got, gold))
    ):
        raise ValueError("rounding certificate: nonfinite or malformed output")
    scale = float(np.max(np.abs(gold)))
    if scale <= 0:
        raise ValueError("rounding certificate: degenerate output oracle")
    difference = np.abs(got.astype("f8") - gold)
    points = np.argwhere(difference >= 0.005 * scale)
    if not len(points):
        raise ValueError("rounding certificate: no original threshold failure")
    activation = round_compute(activation, "bf16")
    records = []
    for row, col in points:
        a = activation[int(a_rows[row])]
        projections = [
            projection_bins(a, w[int(owners[row]), col], split) for w in weights
        ]
        g, u = [np.array(p["bf16"], dtype="f4") for p in projections]
        # exp(-32) is far below half an F32 ULP at 1. Hence the shipping
        # expression 1.f + expf(-g) is exactly 1.f; no exp tolerance is added.
        if np.any(g < 32):
            raise ValueError("rounding certificate: gate is not positively saturated")
        permitted = (g[:, None] * u[None, :]).reshape(-1)
        if not np.isfinite(permitted).all():
            raise ValueError("rounding certificate: nonfinite permitted product")
        bits = np.asarray(got[row, col], dtype="f4").view("u4")
        if not np.any(bits == permitted.view("u4")):
            raise ValueError(
                "rounding certificate: output is not an exact permitted BF16 product"
            )
        records.append(
            dict(
                row=int(row),
                column=int(col),
                expert=int(owners[row]),
                input_row=int(a_rows[row]),
                got=float(got[row, col]),
                original_want=float(gold[row, col]),
                projections=projections,
                permitted_outputs=permitted.tolist(),
            )
        )
    return dict(
        status="CERTIFIED_DISCRETE_ROUNDING",
        original_verdict="FAIL",
        original_error=float(difference.max() / scale),
        original_threshold=0.005,
        method="GAMMA_K_PLUS_S_F32_BOUND_SATURATED_EXACT_BF16_PRODUCT",
        exceptions=records,
    )

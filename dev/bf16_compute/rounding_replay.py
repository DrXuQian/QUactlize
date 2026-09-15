"""CPU replay of the frozen Q4 V3/C4/W4/P4/S1 BF16 down-projection failure.

This is an arithmetic-order diagnostic, never the device gate's golden.
Weights come from the independent GGUF decoder. No K-pack address map is used.
"""
import json

import numpy as np

from dev.bf16_compute.cases import weights, source
from dev.bf16_compute.fixture import compare, digest, round_compute
from dev.bf16_compute.moe_case import swiglu


def replay():
    expanded = np.repeat(source(8, 512, 720), 8, axis=0)
    owners = ((np.arange(8)[None, :] * 17 + np.arange(8)[:, None] * 11) % 256).astype("i4").ravel()
    projections = []
    for role in (0, 1):
        w = weights(12, 512, 512, 256, 910 + role * 37)
        value, _ = w.dot(expanded, owners, "bf16", output_compute=True)
        projections.append(value)
    middle = swiglu(*projections)
    w = weights(12, 512, 512, 256, 984)
    decoded = np.stack([w.weight(e) for e in owners])
    groups = []
    for group in range(16):
        # The gate fixture uses dyadic d and dmin=d/2, scale=min=1.
        seed = 984 + (owners % 17) * 31 + 12 * 1009
        tags = (np.arange(512)[None, :] * 3 + group // 8 + seed[:, None]) % 4
        scale = np.ldexp(np.ones((64, 512), "f4"), -(tags + 8)).astype("f4")
        codes = decoded[:, :, group*32:(group+1)*32] / scale[:, :, None] + np.float32(.5)
        if not np.array_equal(codes, np.rint(codes)):
            raise ValueError("fixture no longer has exact Q4 codes")
        dot, a_sum = np.zeros((64, 512), "f4"), np.zeros(64, "f4")
        for half in range(2):
            for slot in range(4):
                k = slot * 8 + half * 4
                values = middle[:, group*32+k:group*32+k+4]
                a_sum = a_sum + ((values[:, 0] + values[:, 1]) + (values[:, 2] + values[:, 3]))
                for residue in range(4):
                    # BF16 times a 4-bit integer is exact in F64; one F32
                    # rounding reproduces the kernel's FP32 FMA for this fixture.
                    dot = (values[:, residue, None].astype("f8") *
                           codes[:, :, k+residue].astype("f8") + dot).astype("f4")
        zero_term = (-scale * np.float32(.5)) * a_sum[:, None]
        groups.append((scale.astype("f8") * dot + zero_term).astype("f4"))
    partial = np.stack(groups)
    # C4 scatter-reduce pairs adjacent K workers, then folds the two active
    # warps. The remaining two warps contribute exact zero for K512.
    while len(partial) > 1:
        partial = partial[::2] + partial[1::2]
    native_f32 = partial[0]
    actual = round_compute(native_f32, "bf16").astype("f8")
    exact, denom = w.dot(middle, owners, "bf16")
    separately_rounded = round_compute(exact, "bf16").astype("f8")
    legacy_error = np.abs(actual - separately_rounded) / denom
    bad = np.flatnonzero(legacy_error >= .005)
    proof = compare(actual, exact, denom)
    negatives = {}
    for name, wrong in (("zero", np.zeros_like(actual)), ("column_swap", actual[:, ::-1])):
        try:
            compare(wrong, exact, denom)
        except ValueError:
            negatives[name] = "RED"
        else:
            raise ValueError("wrong output accepted: " + name)
    return dict(input_sha256=digest(middle), rounded_output_sha256=digest(actual),
        separately_rounded_golden_sha256=digest(separately_rounded),
        legacy_bad=bad.tolist(), legacy_error=float(legacy_error.max()),
        native_order_f32_error=float((np.abs(native_f32-exact)/denom).max()),
        unrounded_reference_proof=proof, negatives=negatives,
        first=dict(index=int(bad[0]), exact=float(exact.flat[bad[0]]),
                   native_f32=float(native_f32.flat[bad[0]]),
                   expected_bf16=float(separately_rounded.flat[bad[0]]),
                   native_bf16=float(actual.flat[bad[0]])),
        scope="HOST_ARITHMETIC_REPLAY_BOUND_TO_RETURNED_PPU_BF16_OUTPUT_NOT_RAW_F32_DEVICE_CAPTURE")


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))

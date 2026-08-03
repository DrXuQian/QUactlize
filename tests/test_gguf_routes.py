"""THE ROUTES AS PATHS, not as pieces: each one produces a full (m, n) product, compared against one oracle.

WHAT THIS ADDS OVER test_gguf_golden.py. That file checks the DECODE -- scales, codes, one block's dot product,
one block's fp16 expansion -- and is thorough about it. What it cannot check is the ASSEMBLY: whether n*(k/256)
blocks reshape into the (n, k) weight the GEMM wants, whether the activation lines up with the k axis, whether a
grouped route's per-expert slice lands on the right expert. Each of those is an ordering mistake, and each one
produces a plausible number rather than an error.

THE FIXTURE IS BUILT BACKWARDS, and it has to be. The official gguf package has NO k-quant QUANTISER -- only
`dequantize` -- so there is no way to ask it for the bytes of a given weight. Instead: synthesise the bytes, give
the fp16 headers sane values (random bytes make d and dmin NaN, and np.abs(nan) > tol is False, so every block
"passes"), and let the official dequantiser define what those bytes mean. Random code bytes are then a feature:
they cover the full code space rather than whatever a quantiser would have chosen.

ASYMMETRIC IN THREE WAYS, each hiding a different mistake:
  * n != k, so a transposed reshape is a shape error rather than a wrong answer
  * the activation varies along k, so summing the right values in the wrong order is visible
  * each expert's weight differs, so a grouped route reading expert 0 for everyone still passes a shape check
A previous fixture in this project had every row identical and measured nothing but its own axis confusion.
"""
import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf", reason="the official gguf package is the oracle for every route")
from gguf.constants import GGMLQuantizationType as GT
from gguf.constants import GGML_QUANT_SIZES

import quactlize
from quactlize import routes

#  name    ggml type   fp16 header ranges (d, dmin)   qtype int
FORMATS = [
    ("Q2_K", GT.Q2_K, [(80, 82), (82, 84)], 10),
    ("Q3_K", GT.Q3_K, [(108, 110)],         11),
    ("Q4_K", GT.Q4_K, [(0, 2), (2, 4)],     12),
    ("Q5_K", GT.Q5_K, [(0, 2), (2, 4)],     13),
    ("Q6_K", GT.Q6_K, [(208, 210)],         14),
]

N, K = 24, 512          # n != k, and k = 2 superblocks per row so the per-row block loop actually iterates
BPR = K // 256


def _raw_blocks(gt, hdr, n_blocks, rng):
    """Synthetic GGUF blocks: full random code space, fp16 headers small/positive/normal so the golden is finite."""
    _, type_size = GGML_QUANT_SIZES[gt]
    raw = rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8)
    for lo, hi in hdr:
        v = (rng.random(n_blocks) * 0.1 + 0.001).astype(np.float16)
        raw[:, lo:hi] = v.view(np.uint8).reshape(n_blocks, 2)
    return raw


def _oracle(raw, gt, n, k):
    """What the OFFICIAL dequantiser says those bytes mean, as the (n, k) weight."""
    w = gguf.quants.dequantize(raw.reshape(-1), gt).reshape(raw.shape[0], 256).astype(np.float64)
    assert np.isfinite(w).all(), "the fixture is degenerate: a non-finite golden makes every comparison vacuous"
    return w.reshape(n, k)


def _fixture(gt, hdr, seed, n=N, k=K):
    rng = np.random.default_rng(seed)
    raw = _raw_blocks(gt, hdr, n * (k // 256), rng)
    return raw, _oracle(raw, gt, n, k)


def _worst_rel(got, ref):
    """Relative to each row's largest reference magnitude, so cancellation in one entry cannot inflate the ratio."""
    scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-9)
    return float((np.abs(got - ref) / scale).max())


# fp16 weight (2^-11 relative) and fp16 activation, accumulated over k=512 against an fp64 reference. MEASURED
# worst case across all five formats and m in (1, 7, 64): 1.05e-3. 5e-3 leaves headroom for the RNG without
# ceasing to discriminate -- a wrong element ORDER on this fixture lands at O(1).
TOL = 5e-3

# This route test resolves to the CPU arm deliberately: it remains the clean fp32-accumulation witness over the
# production fp16 activation and keeps its measured 1.43e-7 / 1e-6 gate. The direct gguf_cuda_probe test separately
# runs the shipping fp16 accumulator and asserts conditioned error at the fixed 2^-11 floor.
TOL_NATIVE = 1e-6


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_dequantised_weight_is_the_oracles(name, gt, hdr, qtype):
    """The materialisation alone, so a GEMM failure and a reshape failure are distinguishable."""
    raw, ref = _fixture(gt, hdr, seed=qtype)
    got = routes.dequantize_weight(torch.from_numpy(raw), N, K, qtype).numpy().astype(np.float64)
    assert got.shape == ref.shape, f"{name}: materialised {got.shape}, oracle is {ref.shape}"
    rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
    assert rel < 1e-3, f"{name}: materialised weight is not the oracle's, worst {rel:.3e}"


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_dequant_first_dense_matches_oracle(name, gt, hdr, qtype):
    """DEQUANT_FIRST/dense end to end: raw blocks -> fp16 weight -> torch's cuBLAS -> (m, n)."""
    raw, w_ref = _fixture(gt, hdr, seed=qtype)
    for m in (1, 7, 64):
        rng = np.random.default_rng(1000 + m)
        a = (rng.standard_normal((m, K)).astype(np.float32) * 0.5)
        ref = a.astype(np.float64) @ w_ref.T
        got = routes.matmul_dequant_first(torch.from_numpy(a), torch.from_numpy(raw), N, K, qtype) \
                    .numpy().astype(np.float64)
        assert got.shape == (m, N), f"{name}: route returned {got.shape}, expected {(m, N)}"
        rel = _worst_rel(got, ref)
        assert rel < TOL, f"{name} m={m}: dequant_first disagrees with the oracle, worst {rel:.3e}"


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_reshape_orientation_is_guarded(name, gt, hdr, qtype):
    """(n, k) read as (k, n) is a permutation rather than an error whenever n == k.

    Both guards are exercised, because the swapped case trips the CHEAPER one first and would otherwise leave the
    block-count check unproven: k=24 is rejected for not being a superblock multiple before the count is ever
    computed. The second case has a legal k and the wrong number of blocks, which is the mistake that actually
    survives review -- a weight cut from the wrong tensor."""
    raw, _ = _fixture(gt, hdr, seed=qtype)
    with pytest.raises(ValueError, match="not a multiple of the k-quant superblock"):
        routes.dequantize_weight(torch.from_numpy(raw), K, N, qtype)      # n and k swapped

    with pytest.raises(ValueError, match="blocks should be"):
        routes.dequantize_weight(torch.from_numpy(raw), N * 2, K, qtype)  # legal k, half the blocks needed


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_grouped_route_reads_the_right_expert(name, gt, hdr, qtype):
    """DEQUANT_FIRST/grouped: each expert gets DIFFERENT bytes, so reading expert 0 for everyone fails."""
    experts, rows = 3, [5, 0, 11]              # a zero-row expert, because that is the skip branch
    raws, refs = [], []
    for e in range(experts):
        r, w = _fixture(gt, hdr, seed=qtype * 10 + e)
        raws.append(r)
        refs.append(w)
    raw = np.concatenate(raws, axis=0)

    rng = np.random.default_rng(7)
    a = (rng.standard_normal((sum(rows), K)).astype(np.float32) * 0.5)
    starts = np.cumsum([0] + rows[:-1])
    ref = np.concatenate([a[s:s + r].astype(np.float64) @ refs[e].T
                          for e, (s, r) in enumerate(zip(starts, rows)) if r], axis=0)

    got = routes.matmul_dequant_first_grouped(
        torch.from_numpy(a), torch.from_numpy(raw), N, K, qtype,
        experts, torch.tensor(rows, dtype=torch.int64)).numpy().astype(np.float64)

    rel = _worst_rel(got, ref)
    assert rel < TOL, f"{name}: grouped route disagrees with the oracle, worst {rel:.3e}"


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_native_gemv_matches_the_oracle(name, gt, hdr, qtype):
    """FULLY_QUANTIZED/GEMV against the oracle directly, not against another of our routes.

    This is the comparison schemes.py could not make before -- not because either decode was unproven, but because
    neither route could be CALLED from Python. This CPU-resolved assembly witness accumulates the production fp16
    activation in fp32; the direct CUDA golden separately exercises the fp16 accumulator."""
    raw, w_ref = _fixture(gt, hdr, seed=qtype)
    rng = np.random.default_rng(99)
    a = (rng.standard_normal((1, K)).astype(np.float32) * 0.5)
    # The native route's production activation contract is fp16; only its CPU oracle's ACCUMULATOR remains fp32.
    ref = a.astype(np.float16).astype(np.float64) @ w_ref.T

    got = routes.matmul_native_gemv(torch.from_numpy(a), torch.from_numpy(raw), N, K, qtype) \
                .numpy().astype(np.float64)
    assert got.shape == (1, N)
    rel = _worst_rel(got, ref)
    assert rel < TOL_NATIVE, f"{name}: native CPU witness disagrees with the oracle, worst {rel:.3e}"


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_the_two_routes_agree_on_identical_bytes(name, gt, hdr, qtype):
    """Two routes, one number, one input. The difference between them is exactly dequant_first's fp16 rounding."""
    raw, _ = _fixture(gt, hdr, seed=qtype)
    rng = np.random.default_rng(1234)
    a = (rng.standard_normal((1, K)).astype(np.float32) * 0.5)
    g = routes.matmul_native_gemv(torch.from_numpy(a), torch.from_numpy(raw), N, K, qtype).numpy().astype(np.float64)
    d = routes.matmul_dequant_first(torch.from_numpy(a), torch.from_numpy(raw), N, K, qtype).numpy().astype(np.float64)
    rel = _worst_rel(g, d)
    assert rel < TOL, f"{name}: the two routes disagree, worst {rel:.3e}"


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_dequant_first_on_cuda_is_the_same_answer(name, gt, hdr, qtype):
    """THE GEMM HALF, ON THE DEVICE. Every other test here runs `a @ w.T` on CPU tensors, where it is not cuBLAS at
    all -- so the route's whole performance argument ("the GEMM itself is cuBLAS-grade") went unexercised.

    The materialisation stays on the host because gguf_dequantize is a CPU op behind a dlopen seam; the weight is
    then moved, which is also the realistic deployment when the device library is absent. What this establishes is
    that the assembled weight is correct in the layout cuBLAS reads, which a CPU matmul with different blocking
    could in principle hide."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    raw, w_ref = _fixture(gt, hdr, seed=qtype)
    rng = np.random.default_rng(555)
    a = (rng.standard_normal((64, K)).astype(np.float32) * 0.5)
    ref = a.astype(np.float64) @ w_ref.T

    w = routes.dequantize_weight(torch.from_numpy(raw), N, K, qtype).cuda()
    got = routes.matmul_dequant_first(torch.from_numpy(a).cuda(), None, N, K, qtype, weight=w)
    got = got.cpu().numpy().astype(np.float64)

    rel = _worst_rel(got, ref)
    assert rel < TOL, f"{name}: dequant_first on CUDA disagrees with the oracle, worst {rel:.3e}"


def test_grouped_rejects_a_row_count_that_does_not_sum():
    """The FAILURE path of the grouped route's guard, because a guard only ever seen passing is untested."""
    rng = np.random.default_rng(1)
    raw = np.concatenate([_raw_blocks(GT.Q4_K, [(0, 2), (2, 4)], N * BPR, rng) for _ in range(2)], axis=0)
    a = torch.zeros((5, K), dtype=torch.float32)
    with pytest.raises(ValueError, match="sums to"):
        routes.matmul_dequant_first_grouped(a, torch.from_numpy(raw), N, K, 12, 2,
                                            torch.tensor([3, 3], dtype=torch.int64))


def test_gemv_refuses_a_batch():
    """The GEMV band is m=1 by definition; a silent loop over rows would make the route look usable at m=64."""
    rng = np.random.default_rng(2)
    raw = _raw_blocks(GT.Q4_K, [(0, 2), (2, 4)], N * BPR, rng)
    with pytest.raises(ValueError, match="m=1"):
        routes.matmul_native_gemv(torch.zeros((4, K), dtype=torch.float32), torch.from_numpy(raw), N, K, 12)

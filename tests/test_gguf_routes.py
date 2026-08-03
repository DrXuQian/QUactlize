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
import os

import numpy as np
import pytest
import torch
from pathlib import Path

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


@pytest.fixture(scope="module")
def ppu_backend_cuda(tmp_path_factory):
    """Production CUDA-core backend plus the host-only dense layout seam, both built by plain nvcc.

    They need different include precedence: the backend intentionally uses stock CUTLASS for CUDA, while xplane is
    an actlize host template and uses the SDK stubs. Separate objects make that boundary explicit, then one .so is
    linked exactly like the hgcc build does.

    A REAL LIBRARY WINS OVER THE STAND-IN, and forgetting that broke two tests on ppu001. This fixture is a
    STAND-IN: on an NVIDIA development machine there is no libquactlize_ppu.so, so it builds one with nvcc against
    STOCK NVIDIA CUTLASS. On the box there IS one -- build.sh produced it and QUACTLIZE_PPU_LIB points at it -- and
    the skip condition did not notice, because the PPU box has nvcc and reports torch.cuda.is_available(). So it
    compiled PPU device code (ppu_backend.cu -> gemv_wformat.hpp -> cute/tensor.hpp) against NVIDIA's cute instead
    of actlize's, and died in the include chain:

        AssertionError: In file included from .../quactlize/include/gemv_lowbit/gemv_wformat.hpp:35,

    Two tests errored for a build that should never have been attempted there. Taking the provided library is both
    the correct behaviour and strictly less work.

    The check is is_file() rather than "is the variable set", because run_batch's CPU-reference pass deliberately
    sets QUACTLIZE_PPU_LIB to a nonexistent path -- that must still fall through to the stand-in, or the pass would
    lose its device tests entirely for a reason unrelated to what it is testing.
    """
    import shutil, subprocess
    provided = os.environ.get("QUACTLIZE_PPU_LIB", "")
    if provided and Path(provided).is_file():
        return Path(provided)
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("no device library provided and the CUDA stand-in needs nvcc and a CUDA device")
    root = Path(__file__).resolve().parent.parent
    major, minor = torch.cuda.get_device_capability()
    tmp = tmp_path_factory.mktemp("ppu_backend")
    out, backend_o, layout_o = tmp / "libquactlize_ppu.so", tmp / "backend.o", tmp / "dense_layout.o"
    common = ["nvcc", "-std=c++17", "-O3", f"-arch=sm_{major}{minor}", "--expt-relaxed-constexpr",
              "-Xcompiler=-fPIC", f"-I{root / 'quactlize' / 'include'}"]
    commands = [
      common + ["-dc", f"-I{root / 'third_party' / 'cutlass' / 'include'}",
                f"-I{root / 'third_party' / 'actlize' / 'include'}", "-o", str(backend_o),
                str(root / "quactlize" / "csrc" / "device" / "ppu_backend.cu")],
      common + ["-c", f"-I{root / 'dev' / 'fold_derivation' / 'stub_inc'}",
                f"-I{root / 'third_party' / 'actlize' / 'include'}",
                f"-I{root / 'third_party' / 'cutlass' / 'include'}", "-o", str(layout_o),
                str(root / "quactlize" / "csrc" / "device" / "ppu_dense_layout.cu")],
      ["nvcc", "-shared", f"-arch=sm_{major}{minor}", "-o", str(out), str(backend_o), str(layout_o)],
    ]
    for cmd in commands:
        built = subprocess.run(cmd, capture_output=True, text=True)
        assert built.returncode == 0, built.stdout + built.stderr
    return out


def test_device_decode_routes_match_official_oracle_and_reject_planted_faults(ppu_backend_cuda):
    """All 20 decode cases: native dense/MoE plus scale-first dense/MoE, five formats each.

    The subprocess is required because the dlopen decision is intentionally cached once per process. Every fixture
    is asymmetric (n != k, activation varying along k, different expert bytes, ragged rows with an empty expert).
    Each path first observes its oracle reject a planted addressing/code-plane fault, then accepts the real launch.
    """
    import os, subprocess, sys, textwrap
    root = Path(__file__).resolve().parent.parent
    code = textwrap.dedent(r'''
        import numpy as np, torch, gguf, quactlize
        from gguf.constants import GGMLQuantizationType as GT, GGML_QUANT_SIZES
        from quactlize import routes

        formats = [
          ("Q2_K",GT.Q2_K,[(80,82),(82,84)],10), ("Q3_K",GT.Q3_K,[(108,110)],11),
          ("Q4_K",GT.Q4_K,[(0,2),(2,4)],12), ("Q5_K",GT.Q5_K,[(0,2),(2,4)],13),
          ("Q6_K",GT.Q6_K,[(208,210)],14)]
        assert quactlize.gguf_backend().startswith("ppu"), quactlize.gguf_backend()
        floor = 2.0 ** -11
        summaries = []
        for name, gt, hdr, qtype in formats:
            rng = np.random.default_rng(900 + qtype)
            experts, n, k = 4, 24, 2048             # n != k; eight superblocks exercise K assembly
            rows = np.array([2, 0, 3, 1], np.int64) # empty expert + genuinely ragged gathered rows
            offsets = np.cumsum(np.r_[0, rows])
            total = int(rows.sum())
            raw = rng.integers(0, 256, (experts, n * (k // 256), GGML_QUANT_SIZES[gt][1]), np.uint8)
            for e in range(experts):
                for lo, hi in hdr:
                    v = (rng.random(n * (k // 256)) * .03 + .002 + e * .006).astype(np.float16)
                    raw[e, :, lo:hi] = v.view(np.uint8).reshape(-1, 2)
            a = (rng.standard_normal((total, k)) * .2).astype(np.float16)

            ref = np.empty((total, n), np.float64)
            sumabs = np.empty_like(ref)
            for e, count in enumerate(rows):
                if not count: continue
                w = gguf.quants.dequantize(raw[e].reshape(-1), gt).reshape(n, k).astype(np.float64)
                for r in range(int(offsets[e]), int(offsets[e+1])):
                    terms = w * a[r].astype(np.float64)[None, :]
                    ref[r] = terms.sum(1); sumabs[r] = np.abs(terms).sum(1)
            denom = np.maximum(sumabs, np.finfo(np.float64).tiny)
            err = lambda got, rr=ref, dd=denom: float(np.max(np.abs(got.astype(np.float64)-rr) / dd))

            # FULLY_QUANTIZED/GEMV_MOE and exact expert-base negative control.
            bad_raw = raw.copy(); bad_raw[-1] = raw[0]
            bad_native = routes.matmul_native_gemv_moe(torch.from_numpy(a), torch.from_numpy(bad_raw),
                n, k, qtype, experts, torch.from_numpy(rows)).numpy()
            assert err(bad_native) > floor, f"{name}: native oracle missed planted expert-0 reuse"
            native = routes.matmul_native_gemv_moe(torch.from_numpy(a), torch.from_numpy(raw),
                n, k, qtype, experts, torch.from_numpy(rows)).numpy()
            native_err = err(native); assert native_err < floor, (name, "native MoE", native_err)

            # FULLY_QUANTIZED/GEMV uses the same production .so but a distinct dense launch. Copying row zero
            # over the last row is a well-formed addressing fault, not malformed input the wrapper could reject.
            dense_ref, dense_den = ref[:1], denom[:1]
            dense_err = lambda got: float(np.max(np.abs(got.astype(np.float64)-dense_ref) / dense_den))
            native_dense_raw = raw[0].copy()
            bad_native_dense_raw = native_dense_raw.copy(); bad_native_dense_raw[-1] = native_dense_raw[0]
            planted_native_dense = routes.matmul_native_gemv(torch.from_numpy(a[:1]),
                torch.from_numpy(bad_native_dense_raw), n, k, qtype).numpy()

            artifact = routes.prepare_scale_first(torch.from_numpy(raw), n, k, qtype)
            dense_artifact = tuple(x[:1].contiguous() if x.ndim >= 3 else x for x in artifact)

            assert dense_err(planted_native_dense) > floor, \
                f"{name}: native dense oracle missed planted row-0 reuse"
            native_dense = routes.matmul_native_gemv(torch.from_numpy(a[:1]),
                torch.from_numpy(native_dense_raw), n, k, qtype).numpy()
            native_dense_err = dense_err(native_dense)
            assert native_dense_err < floor, (name, "native dense", native_dense_err)

            # SCALE_FIRST/GEMV and a whole low-code-plane fault.
            bad_dense = list(dense_artifact); bad_dense[0] = torch.zeros_like(bad_dense[0])
            planted_dense = routes.matmul_scale_first_gemv(torch.from_numpy(a[:1]), tuple(bad_dense), qtype).numpy()
            assert dense_err(planted_dense) > floor, f"{name}: dense oracle missed planted code-plane fault"
            dense = routes.matmul_scale_first_gemv(torch.from_numpy(a[:1]), dense_artifact, qtype).numpy()
            sf_dense_err = dense_err(dense); assert sf_dense_err < floor, (name, "scale dense", sf_dense_err)

            # SCALE_FIRST/GEMV_MOE and the same expert-base fault across every resident plane.
            bad_artifact = [x.clone() for x in artifact]
            for x in bad_artifact:
                if x.ndim >= 3: x[-1].copy_(x[0])
            planted_moe = routes.matmul_scale_first_gemv_moe(torch.from_numpy(a), tuple(bad_artifact), qtype,
                torch.from_numpy(rows)).numpy()
            assert err(planted_moe) > floor, f"{name}: scale MoE oracle missed planted expert-0 reuse"
            scale_moe = routes.matmul_scale_first_gemv_moe(torch.from_numpy(a), artifact, qtype,
                torch.from_numpy(rows)).numpy()
            sf_moe_err = err(scale_moe); assert sf_moe_err < floor, (name, "scale MoE", sf_moe_err)
            summaries.append(f"{name}: native={native_dense_err:.2e}/{native_err:.2e} "
                             f"scale={sf_dense_err:.2e}/{sf_moe_err:.2e}")
        print("; ".join(summaries))
    ''')
    env = dict(os.environ, QUACTLIZE_PPU_LIB=str(ppu_backend_cuda))
    run = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_scale_first_artifact_has_full_and_scale_inverses(name, gt, hdr, qtype):
    """The GEMV artifact is independently readable without asking GEMV to consume it.

    dequant-all is compared elementwise with official gguf; dequant-scale is compared with the consumer-ready affine
    planes constructed from the raw-block accessor. Zeroing the stored code plane and perturbing one scale
    are both observed failures before accepting the real artifact.
    """
    raw, official = _fixture(gt, hdr, seed=7300 + qtype)
    blocks = torch.from_numpy(raw)
    artifact = routes.prepare_scale_first(blocks, N, K, qtype)
    got = routes.dequantize_scale_first(artifact, qtype).numpy()[0].astype(np.float64)
    rel = np.max(np.abs(got - official)) / max(np.max(np.abs(official)), 1e-9)
    assert rel < 1e-3, f"{name}: artifact dequant-all disagrees with official gguf by {rel:.3e}"

    codes, raw_scale, raw_zero = quactlize.gguf_unpack(blocks, qtype)
    del codes
    groups = raw_scale.shape[1]
    expect_scale = raw_scale.view(N, BPR, groups).view(N, -1)
    bias = 4 if qtype == 11 else 32 if qtype == 14 else 0
    stored_zero = (raw_zero.float() - bias * raw_scale.float()).half()
    expect_zero = stored_zero.view(N, BPR, groups).view(N, -1)
    got_scale, got_zero = routes.dequantize_scale_first_scales(artifact, qtype)
    assert torch.equal(got_scale[0], expect_scale), f"{name}: dequant-scale transposed the scale plane incorrectly"
    assert torch.equal(got_zero[0], expect_zero), f"{name}: dequant-scale read the stored zero plane incorrectly"

    bad_codes = list(artifact)
    bad_codes[0] = torch.zeros_like(bad_codes[0])
    planted = routes.dequantize_scale_first(tuple(bad_codes), qtype).numpy()[0].astype(np.float64)
    planted_rel = np.max(np.abs(planted - official)) / max(np.max(np.abs(official)), 1e-9)
    assert planted_rel > 1e-3, f"{name}: dequant-all missed a zeroed low-code plane"

    bad_scale = list(artifact)
    bad_scale[2] = bad_scale[2].clone()
    bad_scale[2][-1, -1, -1] += .5
    planted_scale, _ = routes.dequantize_scale_first_scales(tuple(bad_scale), qtype)
    assert not torch.equal(planted_scale, got_scale), f"{name}: dequant-scale missed a planted scale fault"


def test_dense_artifact_affine_chain_matches_official_oracle_and_rejects_faults(ppu_backend_cuda):
    """All five dense artifacts have full and scale inverses, against official gguf.

    The xplane producer/inverse pair is mutual placement evidence; official dequant-all independently checks the
    recovered values, and the raw affine accessor checks dequant-scale. This is deliberately not launcher validation:
    the fpA kernel cannot run on the NVIDIA oracle host, so the cell remains IMPLEMENTED.
    """
    import os, subprocess, sys, textwrap
    root = Path(__file__).resolve().parent.parent
    code = textwrap.dedent(r'''
        import sys, numpy as np, torch, gguf, quactlize
        sys.path.insert(0, "tests")
        from test_gguf_routes import FORMATS, _raw_blocks

        n, k, m = 256, 512, 7
        rng = np.random.default_rng(81173)
        kk = np.arange(k, dtype=np.float32)
        a = np.stack([np.sin(kk * (0.007 + r * 0.0013)) + (kk % (11 + r)) * 0.021
                      for r in range(m)]).astype(np.float32)

        def conditioned(got, ref, den):
            return float(np.max(np.abs(got - ref) / np.maximum(den, np.finfo(np.float32).tiny)))

        rows = []
        for name, gt, hdr, qt in FORMATS:
            raw = _raw_blocks(gt, hdr, n * (k // 256), rng)
            blocks = torch.from_numpy(raw)
            dense = quactlize.gguf_prepare_dense(blocks, n, k, qt)
            represented = quactlize.gguf_dense_artifact_dequantize(*dense, qt).numpy()[0].astype(np.float32)
            official = gguf.quants.dequantize(raw.reshape(-1), gt).reshape(n, k).astype(np.float32)
            terms = a[:, None, :] * official[None, :, :]
            got, ref, den = a @ represented.T, terms.sum(2), np.abs(terms).sum(2)
            err = conditioned(got, ref, den)
            assert err < 2e-3, (name, err)

            codes, raw_scale, raw_zero = quactlize.gguf_unpack(blocks, qt)
            del codes
            groups = raw_scale.shape[1]
            expect_scale = raw_scale.view(n, k // 256, groups).view(n, -1)
            bias = 4 if qt == 11 else 32 if qt == 14 else 0
            stored_zero = (raw_zero.float() - bias * raw_scale.float()).half()
            if qt in (12, 13, 14):
                stored_zero = (stored_zero.float() + 8 * raw_scale.float()).half()
            expect_zero = stored_zero.view(n, k // 256, groups).view(n, -1)
            got_scale, got_zero = quactlize.gguf_dense_artifact_dequantize_scale(dense[2], dense[3], qt)
            assert torch.equal(got_scale[0], expect_scale), (name, "scale inverse")
            assert torch.equal(got_zero[0], expect_zero), (name, "zero inverse")

            fault = list(dense); fault[0] = torch.zeros_like(fault[0])
            bad_weight = quactlize.gguf_dense_artifact_dequantize(*fault, qt).numpy()[0].astype(np.float32)
            bad = a @ bad_weight.T
            ferr = conditioned(bad, ref, den)
            assert ferr > 2e-3, (name, ferr)

            scale_fault = list(dense); scale_fault[2] = scale_fault[2].clone()
            scale_fault[2][0, -1, -1] += .5
            planted_scale, _ = quactlize.gguf_dense_artifact_dequantize_scale(
                scale_fault[2], scale_fault[3], qt)
            assert not torch.equal(planted_scale, got_scale), (name, "scale fault was invisible")
            rows.append(f"{name}: err={err:.3e}, planted={ferr:.3e}")
        print("\n".join(rows))
    ''')
    env = os.environ.copy()
    env["QUACTLIZE_PPU_LIB"] = str(ppu_backend_cuda)
    run = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout.strip())


@pytest.fixture(scope="module")
def ppu_backend_dense():
    """An actual hgcc-built library, supplied by the device tier; never substitute the nvcc GEMV fixture."""
    import ctypes, os
    path = os.environ.get("QUACTLIZE_PPU_LIB")
    if not path:
        pytest.skip("dense end-to-end oracle needs QUACTLIZE_PPU_LIB from an actual PPU build")
    path = Path(path)
    assert path.is_file(), f"QUACTLIZE_PPU_LIB does not exist: {path}"
    lib = ctypes.CDLL(str(path))
    for symbol in ("quactlize_ppu_prepare_dense", "quactlize_ppu_recover_dense",
                   "quactlize_ppu_dense_lowbit"):
        assert getattr(lib, symbol, None) is not None, f"device library is missing dense symbol {symbol}"
    assert quactlize.gguf_backend().startswith("ppu"), quactlize.gguf_backend()
    return path


# THE PACKED-DENSE ROUTE'S NAMES, in one place. The op is being written by codex right now (INBOX 012); until it
# lands these resolve to nothing and the test below skips with a reason that says so. Changing a name here is the
# whole of wiring this oracle up -- deliberately, so the moment the signature is posted this is a one-line edit
# rather than an hour of test writing under a deadline.
# THE SINGLE-PLANE FORMATS. Q3/Q5/Q6 are two-plane and their packed-scale path does not exist yet -- the Builder
# ALL FIVE, and the order they arrived in says what the two axes actually were. Q4_K single-plane scu16x1;
# Q2_K single-plane with a 20 B unit staged as five 4 B copies; Q5_K two-plane weight with Q4's SAME scale unit,
# which is why it was a reachability lift rather than a new scale design; Q3_K and Q6_K two-plane AND paired --
# their units span two superblocks (scu28x2, scu36x2), so the units tensor is indexed by K/512, not K/256.
#
# Q6 keeps its box-validated TK128 weight placement and lets scu36x2 span four tiles, rather than moving the
# weight to TK256 to suit the scale. That direction is not arbitrary: TK256 is the tactic whose high-plane map
# produced conditioned error 8.76e-1 earlier today. The scale channel adapts to the weight, not the reverse.
FQ_IMPLEMENTED = list(FORMATS)          # all five, once Q3/Q6 landed


FQ_DENSE_PRODUCER = "prepare_fully_quantized_dense"   # host-side, the analogue of prepare_scale_first_dense
FQ_DENSE_ROUTE    = "matmul_fully_quantized_dense"    # the launch


FQ_GROUPED_ROUTE = "matmul_fully_quantized_grouped"   # the grouped launch; producer is FQ_DENSE_PRODUCER's analogue
FQ_GROUPED_PRODUCER = "prepare_fully_quantized_grouped"


@pytest.mark.fully_quantized_dense
@pytest.mark.parametrize("name,gt,hdr,qtype", FQ_IMPLEMENTED)
def test_fully_quantized_grouped_matches_dequant_first_and_rejects_fault(name, gt, hdr, qtype, ppu_backend_dense):
    """FULLY_QUANTIZED/GROUPED -- the MoE GEMM -- against the same independent arm as the dense cell.

    WHY THIS EXISTS EVEN THOUGH THE CELL ALREADY READS VALIDATED for Q4_K. Its note says what its evidence
    actually is: "ONLY test_q4k_packed_gemm's rowC exercises the packed decoder -- rowA and rowB are fp16-path
    controls". One row of one C++ test, at one group size, behind PPU_PACKED_SCALE=1. That was enough to believe
    the decoder works and it is not enough to believe the CELL works: it says nothing about the assembly -- whether
    n*(k/256) blocks reshape into the weight the GEMM wants, whether a per-expert slice lands on the right expert,
    whether a ragged row count is handled. Those are ordering mistakes and each produces a plausible number.

    RAGGED ROWS WITH AN EMPTY EXPERT are in the fixture on purpose. An expert with zero rows is the case that a
    cumulative-offset bug reads straight past, and it cannot be reached by any uniform shape.

    The planted fault reuses expert 0's slice for every expert -- the specific mistake this fixture is built to
    catch, and the one a shape check cannot see.
    """
    if not (routes.has_op("gguf_prepare_fully_quantized_grouped")
            and routes.has_op("gguf_grouped_fully_quantized")):
        pytest.skip("the grouped packed ops are not in this build yet (INBOX 013).")
    _require_packed_format(qtype, name)
    prepare, launch = routes.prepare_fully_quantized_grouped, routes.matmul_fully_quantized_grouped

    experts, n, k = 4, 256, 512
    rows = np.array([2, 0, 3, 1], dtype=np.int32)      # an EMPTY expert, and no two alike
    rng = np.random.default_rng(29000 + qtype)
    raw = _raw_blocks(gt, hdr, experts * n * (k // 256), rng)
    blocks = torch.from_numpy(raw)
    artifact = prepare(blocks, n, k, qtype, experts)
    official = gguf.quants.dequantize(raw.reshape(-1), gt).reshape(experts, n, k).astype(np.float64)
    bound, observed = 5e-3, []

    m = int(rows.sum())
    a = (rng.standard_normal((m, k)) * .2).astype(np.float16)
    at, rt = torch.from_numpy(a), torch.from_numpy(rows)
    independent = routes.matmul_dequant_first_grouped(
        at, blocks, n, k, qtype, experts, rt).numpy().astype(np.float64)
    # per-row denominator against that row's own expert, so an empty expert contributes nothing rather than a zero
    per_row = np.repeat(np.arange(experts), rows)
    denom = np.stack([np.abs(a.astype(np.float64)[i][None, :] * official[per_row[i]]).sum(1) for i in range(m)])
    conditioned = lambda out: float(np.max(np.abs(out.astype(np.float64) - independent) /
                                           np.maximum(denom, np.finfo(np.float64).tiny)))

    fault = [x.clone() for x in artifact] if isinstance(artifact, (tuple, list)) else [artifact.clone()]
    for x in fault:                                     # every expert reads expert 0 -- the ordering mistake itself
        if x.ndim >= 3 and x.shape[0] == experts:
            for e in range(1, experts):
                x[e].copy_(x[0])
    planted_in = tuple(fault) if isinstance(artifact, (tuple, list)) else fault[0]
    planted_err = conditioned(launch(at, planted_in, qtype, rt).numpy())
    assert planted_err > bound, (
        f"{name}: the grouped oracle MISSED every expert reading expert 0 ({planted_err:.3e}). "
        f"Until this fails, a pass below is not evidence about per-expert addressing.")

    err = conditioned(launch(at, artifact, qtype, rt).numpy())
    assert err < bound, f"{name}: packed grouped disagrees with dequant-first ({err:.3e})"
    observed.append(f"rows={list(rows)} err={err:.3e} planted={planted_err:.3e}")
    print(f"{name} packed grouped vs dequant-first: " + "; ".join(observed))


# THE DEVICE LIBRARY IS BUILT PER FORMAT, and that is deliberate: a PPU_PACKED_FORMAT=2 binary intentionally
# cannot run the Q4 packed launch (codex, 070). So "which formats can this run" is a property of the LIBRARY, not
# of the test file, and a test that parametrised over every implemented format would report a build's scope as a
# set of failures. run_batch builds once per format and sets QUACTLIZE_PACKED_FORMAT to match; the default is
# Q4_K's ggml type, which is what a library built with no PPU_PACKED_FORMAT contains.
def _packed_format_under_test() -> int:
    return int(os.environ.get("QUACTLIZE_PACKED_FORMAT", "12"))


def _require_packed_format(qtype: int, name: str):
    want = _packed_format_under_test()
    if qtype != want:
        pytest.skip(f"this library is built for packed format {want}, not {name} ({qtype}). A format-specific "
                    f"binary cannot run another format's packed launch -- run_batch builds one per format. "
                    f"Set QUACTLIZE_PACKED_FORMAT={qtype} against a matching build to run this case.")


@pytest.mark.fully_quantized_dense
@pytest.mark.parametrize("name,gt,hdr,qtype", FQ_IMPLEMENTED)
def test_fully_quantized_dense_matches_dequant_first_and_rejects_fault(name, gt, hdr, qtype, ppu_backend_dense):
    """FULLY_QUANTIZED/DENSE against an INDEPENDENT arm, from the cell's first day rather than after the fact.

    scale_first/dense ran green for a long time against test_fpA_kquant_dense, which compares two fpA tactics --
    shared constants move both sides together, so a wrong shared constant survives. That is why that cell sat at
    IMPLEMENTED however green it looked, and it is the specific history this test exists to not repeat.

    The independent arm is matmul_dequant_first fed the OFFICIAL package's materialisation: it shares neither
    quactlize's block decoder nor its artifact importer with the packed path. M=7 and M=65 are both genuine dense
    shapes and both are tile tails, so a tail-handling mistake cannot hide behind a round M.

    THE PLANTED FAULT RUNS FIRST. A pass means nothing until the same comparison has been observed to fail on a
    deliberately corrupted artifact -- and the corruption is in the PACKED SCALE UNIT, not only the code plane,
    because the unit is what this cell adds over scale_first and a code-plane fault would not exercise it.
    """
    # THE SKIP TESTS THE OP, NOT THE ROUTE. The route functions exist as soon as I write them, so keying on them
    # would make this test fail on every build that predates the op -- an absence reported as a defect. The op's
    # presence is the real question, and "built and wrong" then stays a failure rather than becoming a skip.
    if not (routes.has_op("gguf_prepare_fully_quantized_dense") and routes.has_op("gguf_dense_fully_quantized")):
        pytest.skip("the packed-dense ops are not in this build yet (INBOX 012). When they land, run_batch's "
                    "required-not-to-skip list must be updated in the same change, or a skip on a device box "
                    "reads as a pass.")
    _require_packed_format(qtype, name)
    prepare, launch = routes.prepare_fully_quantized_dense, routes.matmul_fully_quantized_dense

    n, k = 256, 512
    rng = np.random.default_rng(23000 + qtype)
    raw = _raw_blocks(gt, hdr, n * (k // 256), rng)
    blocks = torch.from_numpy(raw)
    artifact = prepare(blocks, n, k, qtype)
    official = gguf.quants.dequantize(raw.reshape(-1), gt).reshape(n, k).astype(np.float64)
    official_fp16 = torch.from_numpy(official.astype(np.float16))
    bound, observed = 5e-3, []

    for m in (7, 65):
        a = (rng.standard_normal((m, k)) * .2).astype(np.float16)
        at = torch.from_numpy(a)
        independent = routes.matmul_dequant_first(
            at, blocks, n, k, qtype, weight=official_fp16).numpy().astype(np.float64)
        denom = np.abs(a.astype(np.float64)[:, None, :] * official[None, :, :]).sum(2)
        conditioned = lambda out: float(np.max(np.abs(out.astype(np.float64) - independent) /
                                               np.maximum(denom, np.finfo(np.float64).tiny)))

        # CORRUPT THE SCALE UNIT, which is artifact[-1] BY CONTRACT. Two earlier versions of this got it wrong in
        # different ways: an elimination heuristic on element count picks LOW (both tensors divide by the group
        # count), and a hardcoded index 1 was right only while the tuple was (low, units) -- it lands on HIGH the
        # moment a two-plane format arrives. The contract puts units last precisely so one expression serves every
        # format, and a fault on the code plane is one the scale-first cell already catches, so getting this wrong
        # reports the packed unit as covered without ever touching it.
        fault = [x.clone() for x in artifact]
        fault[-1] = torch.zeros_like(fault[-1])

        planted = launch(at, tuple(fault), qtype).numpy()
        planted_err = conditioned(planted)
        assert planted_err > bound, (
            f"{name} m={m}: the packed-dense oracle MISSED a zeroed SCALE UNIT ({planted_err:.3e}). The unit is "
            f"what this cell adds over scale_first; until this fails, a pass below is not evidence about it.")

        got = launch(at, artifact, qtype).numpy()
        err = conditioned(got)
        assert err < bound, f"{name} m={m}: packed dense disagrees with dequant-first ({err:.3e})"
        observed.append(f"m={m} err={err:.3e} planted={planted_err:.3e}")
    print(f"{name} packed dense vs dequant-first: " + "; ".join(observed))


def _f_group_size(qtype):
    """gs from the format table rather than a literal -- Q2/Q3/Q6 are 16 and Q4/Q5 are 32, and hardcoding 32
    would put the planted-fault target on the wrong tensor for three of the five formats."""
    from quactlize import formats as _fm
    return _fm.BLOCKS[_fm.QuantType(qtype)].group_size


@pytest.mark.parametrize("name,gt,hdr,qtype", FORMATS)
def test_scale_first_dense_route_matches_dequant_first_and_rejects_fault(
        name, gt, hdr, qtype, ppu_backend_dense):
    """Independent PPU GEMM oracle: identical raw bytes through dense fpA and dequantise-then-library-GEMM.

    M=7 and M=65 are both genuinely dense and both are tile tails. The low-plane fault is a well-formed resident
    artifact, so the oracle must numerically reject it before the real launch can count as evidence. This test is
    skipped locally by construction: a plain-nvcc GEMV library cannot stand in for the hgcc fpA kernel under test.
    """
    n, k = 256, 512
    rng = np.random.default_rng(19000 + qtype)
    raw = _raw_blocks(gt, hdr, n * (k // 256), rng)
    blocks = torch.from_numpy(raw)
    artifact = routes.prepare_scale_first_dense(blocks, n, k, qtype)
    official = gguf.quants.dequantize(raw.reshape(-1), gt).reshape(n, k).astype(np.float64)
    official_fp16 = torch.from_numpy(official.astype(np.float16))
    bound, observed = 5e-3, []

    for m in (7, 65):
        a = (rng.standard_normal((m, k)) * .2).astype(np.float16)
        at = torch.from_numpy(a)
        # Supply the OFFICIAL package's materialisation explicitly. This still exercises the dequant-first library
        # GEMM route, while sharing neither quactlize's block decoder nor its artifact importer with fpA.
        independent = routes.matmul_dequant_first(
            at, blocks, n, k, qtype, weight=official_fp16).numpy().astype(np.float64)
        denom = np.abs(a.astype(np.float64)[:, None, :] * official[None, :, :]).sum(2)
        conditioned = lambda out: float(np.max(np.abs(out.astype(np.float64) - independent) /
                                                np.maximum(denom, np.finfo(np.float64).tiny)))

        fault = list(artifact)
        fault[0] = torch.zeros_like(fault[0])
        planted = routes.matmul_scale_first_dense(at, tuple(fault), qtype).numpy()
        planted_err = conditioned(planted)
        assert planted_err > bound, f"{name} m={m}: dense oracle missed a zeroed low-code plane ({planted_err:.3e})"

        got = routes.matmul_scale_first_dense(at, artifact, qtype).numpy()
        err = conditioned(got)
        assert err < bound, f"{name} m={m}: fpA dense disagrees with dequant-first ({err:.3e})"
        observed.append(f"m={m} err={err:.3e} planted={planted_err:.3e}")
    print(f"{name} dense PPU vs dequant-first: " + "; ".join(observed))


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
@pytest.mark.cpu_reference
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

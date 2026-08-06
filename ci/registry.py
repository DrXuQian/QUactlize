"""WHAT EACH HARNESS ACTUALLY VALIDATES -- declared here, and CHECKED against the source.

This file exists because the claim "format X is validated" was wrong twice in one week, in both directions:

  * test_q65_bconcat_real.cu is named _real and is SYNTHETIC. Its own comment says so, and I still put Q5_K and Q6_K
    in a "validated against real weights" column because of the filename.
  * five harnesses printed MISMATCH and returned 0, so anything that read the exit status saw a pass.

A coverage claim that lives in someone's head, or in a filename, is not a claim anyone can act on. So each entry below
states what a harness reads and what it compares against, and check() verifies the declaration against the file --
a harness declared to read a real fixture must actually open one, and one declared synthetic must not claim otherwise
in its name. The registry FAILS when the code and the declaration disagree, in either direction.

Oracle kinds, weakest to strongest:

  self       the result is compared against another run of the same kernel (a different tile, L=1, ...). Catches
             plumbing, structurally blind to a wrong shared constant -- both sides move together.
  synthetic  an independent CPU golden over generated inputs. A real oracle for the arithmetic; says nothing about
             whether an importer reads a real checkpoint correctly, or about real value distributions.
  real       an independent CPU golden over bytes extracted from an actual model file. The only kind that validates
             the whole chain from checkpoint to output.
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# WHICH PATH A HARNESS EXERCISES. Evidence is per (format, path), not per format: a harness that validates Q4_K on
# the native-scale GEMM says nothing about Q4_K through the GEMV kernel or the dequantise-then-dense fallback. The
# first version of the cross-check collapsed evidence to "this format has some harness" and then approved every
# capability set containing it, which is how a set can look validated while nothing has ever run it.
#
# Path names match quactlize.formats' capability sets. "scale_decode" and "probe" are not GEMM paths -- they are
# harnesses for a component -- and are named so they cannot be mistaken for evidence of a path.
GEMV_P, NATIVE_P, FP16_P, DENSE_P = "gemv", "fused_native_scale", "fused_fp16_scale", "dequant_then_dense"
# Keep the historical coarse name above: older harness reports and generated build targets use it. Matrix evidence
# uses the four launch-specific names below, so a dense result can no longer approve a gathered expert kernel (or
# an fp16-plane result approve the native-block decoder) just because all four are colloquially called GEMV.
SCALE_GEMV_P = "scale_first_gemv"
SCALE_GEMV_MOE_P = "scale_first_gemv_moe"
NATIVE_GEMV_P = "native_gemv"
NATIVE_GEMV_MOE_P = "native_gemv_moe"
SCALE_DENSE_P = "scale_first_dense"
# Component paths: these prove that derived bytes are independently readable, but deliberately do not approve a
# launch cell. In particular the dense artifact inverse is not evidence that fpA's output is independently right.
SCALE_ARTIFACT_P = "scale_first_artifact"
SCALE_DENSE_ARTIFACT_P = "scale_first_dense_artifact"

HARNESS_PATHS = {
    "test_moe_grouped_verify": [FP16_P], "test_moe_grouped_real": [FP16_P], "test_lowbit_grouped": [FP16_P],
    "test_q3_concat_real": [FP16_P], "test_q3_bconcat_real": [FP16_P], "test_q65_bconcat_real": [FP16_P],
    "test_w2a16_real": [FP16_P], "test_q4k_packed_gemm": [NATIVE_P, FP16_P], "test_q4k_native_scale": ["scale_decode"],
    "test_ppu_f16x2_probe": ["probe"], "test_w1a16_grouped": [FP16_P], "test_w2a16_grouped": [FP16_P],
    "test_w1a16_diag": [FP16_P], "test_w2a16_diag": [FP16_P], "test_w4a16_diag": [FP16_P],
    "test_fpA_intB_ppu": [FP16_P, SCALE_DENSE_P],
    "test_fpA_kquant_dense": [SCALE_DENSE_P],
    "test_gemv_lowbit": [GEMV_P, SCALE_GEMV_P, SCALE_GEMV_MOE_P],
    "test_moe_gemm_ppu": [FP16_P],
    "test_moe_grouped_ppu": [FP16_P],
    # THE FIRST EVIDENCE FOR dequant_then_dense. Until this existed DENSE_P appeared in the path vocabulary and in
    # no harness at all, which is the honest state of a route whose host side was never wired -- and check_against_
    # formats() said nothing, because formats.DEQUANT_THEN_DENSE was empty too. Two empty sets agree.
    # NATIVE_P joined when the packed dense and grouped GEMMs gained oracles in this same file --
    # test_fully_quantized_dense/grouped_matches_dequant_first_and_rejects_fault, each comparing against
    # matmul_dequant_first through official gguf semantics, with a planted SCALE-UNIT fault required to fail
    # first. Declaring it is what lets nine newly VALIDATED cells rest on something rather than on nothing.
    "test_gguf_routes": [DENSE_P, SCALE_GEMV_P, SCALE_GEMV_MOE_P, NATIVE_GEMV_P, NATIVE_GEMV_MOE_P,
                         SCALE_ARTIFACT_P, SCALE_DENSE_ARTIFACT_P, NATIVE_P],
}

# harness -> (formats it covers, oracle kind, fixture it must read or None, note)
HARNESSES = {
    "test_moe_grouped_verify":  (["int4"],            "self",      None,
        "grouped vs the same kernel at L=1; plumbing only, blind to a wrong dequant constant"),
    "test_moe_grouped_real":    (["gptq-int4-sym"],   "real",      None,
        "real GPTQ weights through the grouped path; reads the fixture path from argv"),
    "test_lowbit_grouped":      (["int4", "int2", "int1"], "synthetic", None,
        "CPU golden over the full code range"),
    "test_q3_concat_real":      (["gguf-q3k"],        "real",      "real_q3k_concat.bin",
        "A-concat: two GEMMs summed, same golden as the B-concat but 2x the mma"),
    "test_q3_bconcat_real":     (["gguf-q3k"],        "real",      "real_q3k_concat.bin",
        "ONE GEMM with two B bit planes, vs the native Q3_K golden"),
    "test_q65_bconcat_real":    (["gguf-q5k", "gguf-q6k"], "synthetic", None,
        "SYNTHETIC despite the name -- full code range with a CPU golden, no importer exists for Q5_K/Q6_K"),
    "test_w2a16_real":          (["gguf-q2k"],        "real",      "real_q2k_ffn_gate_L0.bin",
        "int2 on a real 3B ffn_gate slice at gs=16"),
    "test_q4k_packed_gemm":     (["gguf-q4k"],        "real",      "q4k_packed.bin",
        "rowA/rowB run Q4_K on fp16 scale planes and rowC on the native 16 B unit, so it is evidence for BOTH paths; "
        "rowC is the only Scale_TileK==8 row and the only one that has been intermittent"),
    "test_q4k_native_scale":    (["gguf-q4k"],        "real",      "q4k_packed.bin",
        "device decode vs a host reference, no GEMM"),
    "test_ppu_f16x2_probe":     (["gguf-q4k"],        "synthetic", None,
        "the two f16x2 asm ops against the scalar ops they replace"),
    "test_w1a16_grouped":       (["int1"],            "synthetic", None, ""),
    "test_w2a16_grouped":       (["int2"],            "synthetic", None, ""),
    "test_w1a16_diag":          (["int1"],            "synthetic", None, "diagonal probe"),
    "test_w2a16_diag":          (["int2"],            "synthetic", None, "diagonal probe"),
    "test_w4a16_diag":          (["int4"],            "synthetic", None, "diagonal probe"),
    "test_fpA_intB_ppu":        (["int4"],            "self",      None, "dense launcher sweep"),
    "test_fpA_kquant_dense":    (["gguf-q2k", "gguf-q3k", "gguf-q4k", "gguf-q5k", "gguf-q6k"], "self", None,
        "two fpA dense configurations over each resident k-quant artifact; catches launcher plumbing but shares constants"),
    # Also declares gptq-int4-sym, and the reason is representational rather than a second harness: this sweep
    # instantiates a scale-only quant op with fp16 per-group scales over packed int4, which IS the GPTQ symmetric
    # representation -- same packing, same scale dtype, zero folded into the code range. What it does NOT cover is a
    # real GPTQ checkpoint reaching the GEMV, so the oracle stays synthetic.
    "test_gemv_lowbit":         (["int4", "int2", "int1", "gptq-int4-sym"], "synthetic", None, "CUDA-core GEMV"),
    "test_moe_gemm_ppu":        (["int4"],            "self",      None, ""),
    "test_moe_grouped_ppu":     (["int4"],            "self",      None, ""),
    # SYNTHETIC, and the reason is a property of the oracle rather than a shortcut: the official gguf package has no
    # k-quant QUANTISER, only dequantize(), so there is no way to ask it for the bytes of a given weight. The bytes
    # are synthesised and the official dequantiser defines what they mean -- an independent implementation of the
    # spec over generated inputs, which is exactly this file's definition of synthetic. Real-checkpoint bytes reach
    # only Q4_K, through test_q4k_packed_gemm, and that is a different path.
    "test_gguf_routes":         (["gguf-q2k", "gguf-q3k", "gguf-q4k", "gguf-q5k", "gguf-q6k"], "synthetic", None,
        "official gguf oracle over raw-block dense fallback, production native/scale-first GEMV dense/MoE, and "
        "dequant-all/dequant-scale inverses for both derived scale-first formats; asymmetric n/k, ragged experts, "
        "and a path-specific planted fault before every positive comparison"),
}

# Formats we intend to ship, and what each one needs before it can be called supported.
FORMATS = ["int4", "int2", "int1", "gguf-q2k", "gguf-q3k", "gguf-q4k", "gguf-q5k", "gguf-q6k",
           "gptq-int4-sym", "awq-int4"]


# quactlize.formats names the formats a PATH can run; this file names the formats a HARNESS has evidence for. They
# are different axes and both are needed, but they must not contradict: a format the dispatcher would route to a
# kernel, with nothing anywhere validating that kernel on it, is a capability claim with no evidence behind it.
# Only the QuantTypes map -- "int4"/"int2"/"int1" here are bit widths of the synthetic path, not storage formats.
FORMAT_TO_QUANT_TYPE = {
    "gguf-q2k": "Q2_K", "gguf-q3k": "Q3_K", "gguf-q4k": "Q4_K", "gguf-q5k": "Q5_K", "gguf-q6k": "Q6_K",
    "gptq-int4-sym": "GPTQ_INT4_SYM", "awq-int4": "AWQ_INT4",
}


def coverage_by_path():
    """(format, path) -> the strongest oracle any harness gives that pair."""
    rank = {"self": 0, "synthetic": 1, "real": 2}
    best = {}
    for name, (fmts, oracle, _fx, _n) in HARNESSES.items():
        for path in HARNESS_PATHS.get(name, []):
            for f in fmts:
                key = (f, path)
                if rank[oracle] > best.get(key, (None, -1))[1]:
                    best[key] = (f"{name} ({oracle})", rank[oracle])
    return {k: v[0] for k, v in best.items()}


def check_against_formats():
    """Every (format, path) a capability set claims must have at least a synthetic oracle. Returns a list of
    problems, EACH ONE A CAPABILITY CLAIM WITH NOTHING BEHIND IT -- not a test failure, a claim to withdraw."""
    try:
        sys.path.insert(0, str(ROOT))
        from quactlize import formats
    except Exception as e:                                  # formats.py is pure python; an import failure IS a problem
        return [f"cannot import quactlize.formats to cross-check capability claims: {e}"]

    have = coverage_by_path()
    quant_to_format = {v: k for k, v in FORMAT_TO_QUANT_TYPE.items()}
    bad = []
    for set_name, path in (("GEMV", SCALE_GEMV_P), ("FUSED_NATIVE_SCALE", NATIVE_P),
                           ("FUSED_FP16_SCALE", FP16_P), ("DEQUANT_THEN_DENSE", DENSE_P)):
        for q in sorted(getattr(formats, set_name), key=lambda x: x.name):
            fmt = quant_to_format.get(q.name)
            if fmt is None:
                bad.append(f"formats.{set_name} contains {q.name}, which no registry format maps to")
            elif not have.get((fmt, path)):
                bad.append(f"formats.{set_name} claims {q.name}, but no harness runs {fmt!r} through {path!r}")
    return bad


def check():
    """Verify every declaration against the source. Returns a list of problems."""
    bad = []
    tests = ROOT / "tests"
    for name, (fmts, oracle, fixture, _note) in HARNESSES.items():
        # A harness is evidence whatever language it is in. dequant_then_dense is exercised from Python, because its
        # GEMM is a library call and its dequantiser is already a torch op -- writing it in CUDA would have added a
        # toolchain dependency to the one route that does not need one.
        src = next((tests / f"{name}{ext}" for ext in (".cu", ".py") if (tests / f"{name}{ext}").exists()), None)
        if src is None:
            bad.append(f"{name}: declared but neither tests/{name}.cu nor tests/{name}.py exists")
            continue
        text = src.read_text(errors="ignore")
        comment = "#" if src.suffix == ".py" else "//"
        code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(comment))

        if oracle == "real":
            if fixture and fixture not in text:
                bad.append(f"{name}: declared oracle=real on {fixture}, but the source never names that file")
            # Any of the three ways this tree opens a fixture. Written as a list because the first version checked
            # only fopen and reported test_moe_grouped_real -- which uses std::ifstream -- as a problem. A check that
            # is too narrow costs the same as one that is too loose: both make the report untrustworthy.
            if not any(k in code for k in ("fopen", "ifstream", ".load(")):
                bad.append(f"{name}: declared oracle=real but the code opens no file")
        else:
            # A name that claims real while the declaration says otherwise is the exact trap that cost this project a
            # wrong coverage table. Allowed, but it must be stated in the note so nobody reads the filename instead.
            if name.endswith("_real") and "SYNTHETIC" not in _note.upper():
                bad.append(f"{name}: name ends in _real but oracle={oracle}; say SYNTHETIC in the note")

        for f in fmts:
            if f not in FORMATS:
                bad.append(f"{name}: unknown format {f!r}")

        # A harness whose verdict is only text cannot be gated on. This was true of five of them.
        if "MISMATCH" in code and "return bad" not in code and "return fail" not in code and "g_fail" not in code:
            bad.append(f"{name}: prints MISMATCH but the exit status may not carry it")

    # THE SWEEP IS .cu ONLY, and that is a KNOWN HOLE rather than an oversight. It enforces "every CUDA harness is
    # declared"; the Python tests are not enumerated, so one could be added as evidence for a path and never appear
    # here. Not closed in the same change that first needed it, because turning it on demands declaring every
    # existing tests/*.py -- test_gguf_golden, test_formats, test_layouts -- and a sweep that goes red the moment it
    # is switched on gets switched off. Declared as a gap so it is a task, not a surprise.
    for name in sorted(p.stem for p in tests.glob("test_*.cu")):
        if name not in HARNESSES:
            bad.append(f"tests/{name}.cu exists but is not declared in the registry")
    bad += check_against_formats()
    return bad


def coverage():
    """format -> the strongest oracle any harness gives it."""
    rank = {"self": 0, "synthetic": 1, "real": 2}
    best = {f: (None, -1) for f in FORMATS}
    for name, (fmts, oracle, _fx, _n) in HARNESSES.items():
        for f in fmts:
            if f in best and rank[oracle] > best[f][1]:
                best[f] = (f"{name} ({oracle})", rank[oracle])
    return {f: v[0] for f, v in best.items()}


if __name__ == "__main__":
    problems = check()
    print("== registry: declarations against the source ==")
    for p in problems:
        print(f"  PROBLEM  {p}")
    print(f"  {len(problems)} problem(s)")
    print("\n== coverage: the strongest oracle each format has ==")
    for f, who in coverage().items():
        print(f"  {f:<16} {who or 'NOTHING'}")
    sys.exit(1 if problems else 0)

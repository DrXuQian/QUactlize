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
        "the native 16 B scale unit end to end; rowC is the only Scale_TileK==8 row"),
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
    "test_gemv_lowbit":         (["int4", "int2", "int1"], "synthetic", None, "CUDA-core GEMV"),
    "test_moe_gemm_ppu":        (["int4"],            "self",      None, ""),
    "test_moe_grouped_ppu":     (["int4"],            "self",      None, ""),
}

# Formats we intend to ship, and what each one needs before it can be called supported.
FORMATS = ["int4", "int2", "int1", "gguf-q2k", "gguf-q3k", "gguf-q4k", "gguf-q5k", "gguf-q6k",
           "gptq-int4-sym", "awq-int4"]


def check():
    """Verify every declaration against the source. Returns a list of problems."""
    bad = []
    tests = ROOT / "tests"
    for name, (fmts, oracle, fixture, _note) in HARNESSES.items():
        src = tests / f"{name}.cu"
        if not src.exists():
            bad.append(f"{name}: declared but tests/{name}.cu does not exist")
            continue
        text = src.read_text(errors="ignore")
        code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))

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

    for name in sorted(p.stem for p in tests.glob("test_*.cu")):
        if name not in HARNESSES:
            bad.append(f"tests/{name}.cu exists but is not declared in the registry")
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

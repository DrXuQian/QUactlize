#!/usr/bin/env python3
"""WHICH GGUF TYPES DO WE ACTUALLY HANDLE -- measured against ggml's own enum, not against our own list.

WHY THE UNIVERSE COMES FROM ggml.h. quactlize/formats.py can only report on the types it enumerates, so the nine
IQ types and the ternary/fp4 ones do not appear as "unsupported" -- they do not appear at all. A table that omits
a row reads as "these are the formats", when what it means is "these are the formats we thought about". The user
found this by asking; nothing in the repo would have. So the list of what EXISTS is parsed from ggml.h and the
list of what we HANDLE is read from formats.py, and anything in the first and not the second is printed.

    python3 tools/coverage.py [--ggml PATH/ggml.h]

Exit 0 always: this reports, it does not gate. The gate is `python3 ci/local_gates.py -k coverage`, which fails
when a type appears in ggml.h that this file has never classified -- so a new upstream format shows up as a
question rather than as silence.
"""
import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT_GGML = "/root/llama.cpp/ggml/include/ggml.h"

# WHY A ROW IS NOT SUPPORTED, and whether the pipeline could take it. These are CLASSIFICATIONS, not excuses:
# each says what the format IS, because "unsupported" alone does not tell you whether it is a week or a rewrite.
NOTE = {
    "legacy":   ("affine, one fp16 scale per 32 weights, no super-block. STRUCTURALLY EASIER than the k-quants "
                 "we do support -- there is no two-level scale to unpack. Enumerated in formats.py with "
                 "paths: NONE, i.e. the block geometry is known and no kernel reads it."),
    "kquant":   ("supported: bit-planes, packed native scale units, dense + grouped + GEMV."),
    "iq-lut":   ("4-bit codes through a 16-entry non-linear LUT. The CODE plane is exactly our int4 plane; what "
                 "is new is a table lookup in the converter. The nearest thing to free of everything missing."),
    "iq-grid":  ("CODEBOOK formats: the stored code INDEXES a grid of 8 packed values. There is no integer code "
                 "to bit-plane decompose, so CODE_PLANE does not apply and neither does the fold machinery. "
                 "This is a different converter, not a new entry in a table."),
    "ternary":  ("1.6 / 2 bits per weight over {-1,0,1}. Not affine; needs its own converter."),
    "fp4":      ("e2m1 codes with a block exponent (MXFP4) or fp8 scale (NVFP4). Float codes, not integer -- "
                 "the dequant is a bit-pattern reinterpretation rather than a multiply-add."),
    "repack":   ("CPU-side repacked variants of Q4_0 / IQ4_NL for aarch64 SIMD. Not a distinct numeric format "
                 "and not something a GPU backend consumes."),
    "activation": ("intermediate types used for QUANTISED ACTIVATIONS inside dot products, not weight storage. "
                   "No checkpoint stores a weight in these."),
    "notquant": ("not a quantised type."),
}

FAMILY = {
    "F32": "notquant", "F16": "notquant", "BF16": "notquant", "F64": "notquant",
    "I8": "notquant", "I16": "notquant", "I32": "notquant", "I64": "notquant",
    "Q4_0": "legacy", "Q4_1": "legacy", "Q5_0": "legacy", "Q5_1": "legacy", "Q8_0": "legacy",
    "Q4_2": "notquant", "Q4_3": "notquant",          # removed from ggml long ago, still in the enum
    "Q8_1": "activation", "Q8_K": "activation",
    "Q2_K": "kquant", "Q3_K": "kquant", "Q4_K": "kquant", "Q5_K": "kquant", "Q6_K": "kquant",
    "IQ4_NL": "iq-lut", "IQ4_XS": "iq-lut",
    "IQ1_S": "iq-grid", "IQ1_M": "iq-grid", "IQ2_XXS": "iq-grid", "IQ2_XS": "iq-grid",
    "IQ2_S": "iq-grid", "IQ3_XXS": "iq-grid", "IQ3_S": "iq-grid",
    "TQ1_0": "ternary", "TQ2_0": "ternary",
    "MXFP4": "fp4", "NVFP4": "fp4",
    "Q1_0": "ternary",
    "Q4_0_4_4": "repack", "Q4_0_4_8": "repack", "Q4_0_8_8": "repack",
    "IQ4_NL_4_4": "repack", "IQ4_NL_4_8": "repack", "IQ4_NL_8_8": "repack",
}

ORDER = ["kquant", "legacy", "iq-lut", "iq-grid", "fp4", "ternary", "activation", "repack", "notquant"]


def ggml_types(path: pathlib.Path):
    """-> [(name, value)] in enum order. Parsed, not transcribed: a hand-copied list is the thing this file
    exists to replace."""
    text = path.read_text()
    out = []
    for m in re.finditer(r"GGML_TYPE_([A-Z0-9_]+)\s*=\s*(\d+)", text):
        if m.group(1) == "COUNT":
            continue
        out.append((m.group(1), int(m.group(2))))
    return out


def classify(ggml_path=DEFAULT_GGML):
    """-> (rows, unclassified). A type present in ggml.h but absent from FAMILY is returned separately; it is
    the only thing here that can FAIL, and it fails loudly because silence about a new format is the defect."""
    from quactlize import formats as F
    known = {q.name: q for q in F.QuantType}
    rows, unknown = [], []
    for name, val in ggml_types(pathlib.Path(ggml_path)):
        fam = FAMILY.get(name)
        if fam is None:
            unknown.append((name, val))
            continue
        q = known.get(name)
        state = ("SUPPORTED" if fam == "kquant"
                 else "enumerated, no kernel" if q is not None
                 else "-" if fam in ("notquant", "repack", "activation")
                 else "ABSENT")
        rows.append((val, name, fam, state))
    return rows, unknown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ggml", default=DEFAULT_GGML)
    a = ap.parse_args()
    if not pathlib.Path(a.ggml).is_file():
        print(f"cannot read {a.ggml} -- point --ggml at a llama.cpp checkout's ggml/include/ggml.h")
        return 2

    rows, unknown = classify(a.ggml)
    by_fam = {}
    for val, name, fam, state in rows:
        by_fam.setdefault(fam, []).append((val, name, state))

    weight_fams = ("kquant", "legacy", "iq-lut", "iq-grid", "fp4", "ternary")
    done = len(by_fam.get("kquant", []))
    total = sum(len(by_fam.get(f, [])) for f in weight_fams)
    print(f"== GGUF weight-storage coverage: {done} of {total} ==\n")

    for fam in ORDER:
        items = by_fam.get(fam)
        if not items:
            continue
        names = " ".join(n for _, n, _ in items)
        head = f"{fam:<10} ({len(items):>2})"
        print(f"  {head}  {names}")
        print(f"              {NOTE[fam]}\n")

    if unknown:
        print("  !! IN ggml.h AND NEVER CLASSIFIED HERE -- a new upstream format arrived and nothing said so:")
        for name, val in unknown:
            print(f"       GGML_TYPE_{name} = {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

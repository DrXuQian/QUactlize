"""WHICH REORDER A STORED WEIGHT HAS BEEN THROUGH -- named so the name says it.

THE PROBLEM THIS SOLVES. The preprocessing chain calls three different things "interleave": the column-tile
interleave that makes one 128-byte cache line hold one k-tile, the AIU's 256-row column interleave, and a
permutation of elements inside a single 32-bit word. They act on different axes at different granularities for
different consumers, and a tensor that has been through one of them is byte-identical in dtype and shape to one that
has been through another. That is not hypothetical: a request for the AIU interleave silently returned the ordinary
layout, twice over, and nothing about the result could have revealed it.

THE NAMING RULE. A layout's name is the ORDERED JOIN OF ITS STEP TOKENS. Not an opaque label:

    mmarow_tr_cl4_cvtword_bias              the plain mixed-GEMM arrangement
    mmarow_tr_cl4_aiu256_cvtword_bias       the same, plus the AIU column interleave
    mmarow_tr_cl4                           W4A8: neither the bias nor the word permutation (one gate covers both)

Three things follow for free, and each of them is a bug this project actually had:

  * A layout that gains, loses or reorders a step gets a DIFFERENT NAME. No version counter to remember to bump. A
    weight reordered before ColumnsInterleaved was corrected from 256 to 4 was byte-count-, dtype- and
    shape-identical to a correct one; under this rule it would be named cl256 and refused on sight.
  * A STEP'S PARAMETER IS IN THE NAME. cl4 and cl2 are different layouts because they are, and the number is the
    one that mattered.
  * A name that cannot be built is a layout that does not exist. There is no way to write down "AIU interleave on
    int8 weights" and have it look plausible.

STEP TOKENS ARE <consumer><granularity/parameter>. The consumer is the part that matters when something goes wrong,
because it says what breaks: mma (the tensor-core fragment's lane map), mem (the memory system -- cache lines, or
shared-memory banks), aiu (the bulk copy engine), cvt (the dequant converter's fixed emission order).

RUNTIME TRANSFORMS ARE NOT LAYOUTS AND ARE NOT NAMED HERE. The scale buffer's bank swizzle and the AIU's hardware
write/read pairing change addresses during a kernel; they never touch a stored byte. Mixing them into this
vocabulary is how "swizzle" came to mean both "a permutation baked into the file" and "an XOR on a shared-memory
address", which are not the same kind of thing and do not have the same failure mode.
"""
from typing import Dict, List, NamedTuple, Optional, Tuple


class Step(NamedTuple):
    """One reorder in the chain.

    token     what it contributes to a layout name, parameter included
    consumer  what requires it, and therefore what breaks if it is wrong or missing
    unit      the granularity it moves
    upstream  the function that implements it, so the name can be traced to code
    bytes_neutral  whether it only moves bytes. `bias` does not -- it changes values -- which is why it is a
                   separate step rather than part of the word permutation it currently shares a function with.
    """
    token: str
    consumer: str
    unit: str
    upstream: str
    bytes_neutral: bool
    note: str = ""


def mma_row(rows_per_tile: int) -> Step:
    return Step(f"mmarow{'' if rows_per_tile == 32 else rows_per_tile}", "mma", "row",
                "permute_B_rows_for_mixed_gemm", True,
                f"reorders rows within a {rows_per_tile}-row MMA tile to match the fragment's lane map. The "
                f"is_int8_mma variant's index formula has range [0,32) and only fits a 32-row tile")


def axis_transpose() -> Step:
    return Step("tr", "layout", "element", "subbyte_transpose", True,
                "row-major to column-major, sub-byte aware. NOTE: the op returns the INPUT's shape, so for a "
                "non-square matrix the returned metadata contradicts the contents")


def mem_cacheline_col_tile(interleave: int, tile_rows: int = 64) -> Step:
    return Step(f"cl{interleave}", "mem", "column tile", "interleave_column_major_tensor", True,
                f"groups {interleave} column tiles of {tile_rows} rows so one 128-byte cache line holds one k-tile. "
                f"interleave = ElementsPerCacheLine / ThreadblockK: 2 for 8-bit, 4 for 4-bit, 8 for 2-bit")


def aiu_col_tile(tile_rows: int = 256) -> Step:
    return Step(f"aiu{tile_rows}", "aiu", "column tile", "interleave_column_major_tensor_aiu", True,
                f"the PPU's column interleave at a {tile_rows}-row tile. Needs 4-bit weights with k and n both "
                f"multiples of {tile_rows}; it is the IDENTITY when k == {tile_rows} (one tile)")


def aiu_n_fold(factor: int) -> Step:
    return Step(f"foldn{factor}", "aiu", "element group", "fold_traits / nfold", True,
                f"regroups {factor} N-slices contiguously so a transfer reaches the AIU's 32-byte minimum")


def cvt_word_permute() -> Step:
    # Always accompanied by `bias`: one function, one condition. Two tokens because they are different KINDS of
    # transform -- this one moves bytes, bias changes values -- and byte-neutrality depends on telling them apart.
    return Step("cvtword", "cvt", "word (32-bit)", "add_bias_and_interleave_*_inplace (step 2)", True,
                "reorders elements inside each 32-bit word to [7 5 3 1 6 4 2 0] so the converter's fixed emission "
                "order needs no shifts in the main loop")


def code_bias(amount: int) -> Step:
    return Step("bias", "cvt", "element", "add_bias_and_interleave_*_inplace (step 1)", False,
                f"adds {amount} to每 code so the signed range becomes unsigned. NOT a permutation -- it changes "
                f"values -- which is why it is its own step even though it shares a function with cvtword")


def cvt_plane_fold(planes: int) -> Step:
    return Step(f"plane{planes}", "cvt", "element", "xplane_offline", True,
                f"places the folded high plane of a {planes}-plane B-concat so the converter's fixed emission order "
                f"delivers both planes to the right lanes")


class Layout(NamedTuple):
    steps: Tuple[Step, ...]
    alias: str
    applies_to: str

    @property
    def name(self) -> str:
        """The canonical name: the ordered join of the step tokens. THIS is what travels with the tensor."""
        return "_".join(s.token for s in self.steps) if self.steps else "logical"

    @property
    def bytes_neutral(self) -> bool:
        return all(s.bytes_neutral for s in self.steps)

    def explain(self) -> str:
        lines = [f"{self.name}   ({self.alias}; {self.applies_to})"]
        for i, s in enumerate(self.steps, 1):
            keep = "" if s.bytes_neutral else "   [CHANGES VALUES, not a permutation]"
            lines.append(f"  {i}. {s.token:<9} {s.consumer:<5} per {s.unit:<13} {s.upstream}{keep}")
            lines.append(f"     {s.note}")
        return "\n".join(lines)


# THE LAYOUTS THAT EXIST. Not every combination of steps -- only arrangements something in this tree produces or
# consumes. An alias is a short handle for a name nobody wants to type; the NAME is what gets stored.
LAYOUTS: Dict[str, Layout] = {}


def _add(steps, alias, applies_to) -> Layout:
    lay = Layout(tuple(steps), alias, applies_to)
    LAYOUTS[lay.name] = lay
    return lay


LOGICAL = _add([], "logical", "any quant type")

MIXED_GEMM_INT4 = _add(
    [mma_row(32), axis_transpose(), mem_cacheline_col_tile(4), cvt_word_permute(), code_bias(8)],
    "mixed_gemm", "packed int4")

MIXED_GEMM_AIU_INT4 = _add(
    [mma_row(32), axis_transpose(), mem_cacheline_col_tile(4), aiu_col_tile(256), cvt_word_permute(), code_bias(8)],
    "mixed_gemm_aiu", "packed int4, k and n multiples of 256")

# W4A8: a different row permutation, and NEITHER the bias NOR the word permutation. Both of those come from one
# function behind one condition (`if (!is_int8_mma)`), so they are always both present or both absent -- there is no
# arrangement with one and not the other. This entry read mmarow_tr_cl4_cvtword until the gate was read: bias and
# cvtword being separate STEPS does not make them separately selectable, and assuming it did produced a name that
# promised a permutation the chain never runs. The cross-check missed it because the assertion had been written from
# the same wrong belief as the table.
W4A8_INT4 = _add(
    [mma_row(32), axis_transpose(), mem_cacheline_col_tile(4)],
    "w4a8", "packed int4 feeding an int8-activation MMA")

MIXED_GEMM_INT8 = _add(
    [mma_row(16), axis_transpose(), mem_cacheline_col_tile(2), cvt_word_permute(), code_bias(128)],
    "mixed_gemm_int8", "int8")

MIXED_GEMM_INT2 = _add(
    [mma_row(32), axis_transpose(), mem_cacheline_col_tile(8), cvt_word_permute(), code_bias(2)],
    "mixed_gemm_int2", "packed int2")


ALIASES: Dict[str, str] = {lay.alias: name for name, lay in LAYOUTS.items()}


def resolve(name_or_alias: str) -> Layout:
    """A layout by canonical name or by alias. Raises on anything else -- in particular on a name that LOOKS like a
    layout but was never registered, which is what a weight reordered by a different version of this code has."""
    if name_or_alias in LAYOUTS:
        return LAYOUTS[name_or_alias]
    if name_or_alias in ALIASES:
        return LAYOUTS[ALIASES[name_or_alias]]
    raise KeyError(
        f"{name_or_alias!r} is not a registered layout. Registered: "
        + ", ".join(f"{n} ({l.alias})" for n, l in LAYOUTS.items())
        + ". A name that parses but is not registered usually means the weight was reordered by a different "
          "version of the preprocessing -- the steps changed, so the name changed, which is the point.")


def check_compatible(stored: str, required: str) -> Optional[str]:
    """None if a weight stored in `stored` can be fed to a kernel wanting `required`; otherwise why not.

    The message names the first differing step rather than saying the layouts differ, because 'which reorder is
    missing' is the question, and the answer used to require reading two functions."""
    s, r = resolve(stored), resolve(required)
    if s.name == r.name:
        return None
    st, rt = [x.token for x in s.steps], [x.token for x in r.steps]
    for i in range(max(len(st), len(rt))):
        a, b = (st[i] if i < len(st) else None), (rt[i] if i < len(rt) else None)
        if a != b:
            if a is None:
                return f"stored {stored!r} is missing step {i + 1}, {b!r}, that {required!r} needs"
            if b is None:
                return f"stored {stored!r} has an extra step {i + 1}, {a!r}, that {required!r} does not want"
            return f"step {i + 1} differs: stored has {a!r}, {required!r} wants {b!r}"
    return None


def report() -> str:
    out = ["== quactlize weight layouts ==",
           "   the NAME is the ordered join of the step tokens: a layout that gains or loses a step is a",
           "   different name, so a stale reorder cannot be mistaken for a current one",
           ""]
    for lay in LAYOUTS.values():
        out.append(lay.explain())
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())

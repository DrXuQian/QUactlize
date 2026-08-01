"""THE LAYOUT VOCABULARY, AND THE TWO COPIES OF IT KEPT IN AGREEMENT.

The names live twice: quactlize/layouts.py builds them from step objects, and weight_layout.h carries them as a
literal table the C++ reads. Both are wanted -- python composes, C++ looks up -- but a vocabulary that exists twice
drifts, and a drifted layout name is the exact failure this vocabulary was introduced to prevent: two arrangements
that are byte-count-, dtype- and shape-identical and answer to the same name.

So the C++ table is PARSED here and compared entry by entry against the python registry. This is the check that
makes the naming scheme load-bearing rather than decorative.

Needs no device. The op tests need the built extension and skip without it.
"""
import re
from pathlib import Path

import pytest

from quactlize import layouts as L

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "quactlize" / "csrc" / "preprocess" / "weight_layout.h"


def cpp_registry():
    """The rows of detail::registry() in weight_layout.h, as (name, alias, is_int8_mma, use_aiu, bits, multiple)."""
    text = HEADER.read_text()
    body = text[text.index("kTable = {"):text.index("};", text.index("kTable = {"))]
    rows = []
    for m in re.finditer(r'\{"([^"]+)",\s*"([^"]+)",\s*(true|false),\s*(true|false),\s*(-?\d+),\s*(-?\d+)\}', body):
        rows.append((m.group(1), m.group(2), m.group(3) == "true", m.group(4) == "true",
                     int(m.group(5)), int(m.group(6))))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# the two copies agree
# ---------------------------------------------------------------------------------------------------------------

def test_the_cpp_table_and_the_python_registry_have_the_same_names_in_the_same_order():
    """Order too, not just the set: the table is meant to be diffable against layouts.py by eye, and a reordering
    would break that without breaking anything a set comparison could see."""
    assert [r[0] for r in cpp_registry()] == list(L.LAYOUTS)


def test_the_two_copies_agree_on_aliases():
    assert {r[0]: r[1] for r in cpp_registry()} == {n: lay.alias for n, lay in L.LAYOUTS.items()}


@pytest.mark.parametrize("row", cpp_registry(), ids=lambda r: r[0])
def test_each_cpp_row_matches_the_steps_python_composed(row):
    """The C++ flags are a SUMMARY of the python step list, so they can be derived from it and compared. This is
    where a drift actually shows: adding aiu256 to a python layout without setting use_aiu in the table would leave
    the name promising a step the chain never runs -- which is precisely the silent AIU fallback, back again under
    a name that claims otherwise."""
    name, _alias, is_int8_mma, use_aiu, bits, multiple = row
    lay = L.LAYOUTS[name]
    tokens = [s.token for s in lay.steps]

    # aiu256 in the name <=> the AIU interleave runs, and it is the step that needs the 256 multiple
    assert use_aiu == ("aiu256" in tokens)
    assert (multiple == 256) == ("aiu256" in tokens), "only the AIU step imposes a shape multiple"

    # bias and cvtword come from ONE function behind ONE condition, so they are always both present or both absent.
    # Asserted as its own fact rather than folded into the is_int8_mma check: the earlier version tested
    # is_int8_mma == ("bias" not in tokens), which a table listing cvtword without bias satisfies -- and that is
    # exactly the wrong table that got written. A check derived from the same belief as the thing it checks cannot
    # fail; this one is derived from the gate in cutlass_preprocessors.cpp instead.
    assert ("bias" in tokens) == ("cvtword" in tokens), "bias and cvtword share one gate; a layout has both or neither"
    assert is_int8_mma == (name != "logical" and "cvtword" not in tokens)

    # the element width is carried by the cache-line step's parameter: cl4 for 4-bit, cl2 for 8-bit, cl8 for 2-bit
    cl = next((t for t in tokens if t.startswith("cl")), None)
    if cl is None:
        assert bits == 0, "only the logical layout has no cache-line step"
    else:
        assert {"cl2": 8, "cl4": 4, "cl8": 2}[cl] == bits


# ---------------------------------------------------------------------------------------------------------------
# the naming rule itself
# ---------------------------------------------------------------------------------------------------------------

def test_a_name_is_the_ordered_join_of_its_step_tokens():
    assert L.MIXED_GEMM_AIU_INT4.name == "mmarow_tr_cl4_aiu256_cvtword_bias"
    assert L.LOGICAL.name == "logical"


def test_changing_a_step_parameter_changes_the_name():
    """THE PROPERTY THE WHOLE SCHEME EXISTS FOR. A weight reordered before ColumnsInterleaved was corrected from 256
    to 4 was byte-count-, dtype- and shape-identical to a correct one, and there was no way to tell them apart.
    Under this rule it is named cl256, which is not registered, so loading it fails instead of computing nonsense."""
    stale = L.Layout((L.mma_row(32), L.axis_transpose(), L.mem_cacheline_col_tile(256),
                      L.aiu_col_tile(256), L.cvt_word_permute(), L.code_bias(8)), "stale", "the pre-fix reorder")
    assert stale.name == "mmarow_tr_cl256_aiu256_cvtword_bias"
    assert stale.name != L.MIXED_GEMM_AIU_INT4.name
    with pytest.raises(KeyError, match="not a registered layout"):
        L.resolve(stale.name)


def test_adding_a_step_changes_the_name_with_no_version_counter():
    folded = L.Layout(L.MIXED_GEMM_AIU_INT4.steps + (L.aiu_n_fold(2),), "folded", "N-folded")
    assert folded.name == "mmarow_tr_cl4_aiu256_cvtword_bias_foldn2"
    assert folded.name not in L.LAYOUTS


def test_only_the_bias_step_changes_values():
    """Every step but one is a pure permutation, which is what lets the whole chain be byte-neutral. `bias` is the
    exception and is its own step for that reason, though it currently shares a function with cvtword."""
    for lay in L.LAYOUTS.values():
        non_neutral = [s.token for s in lay.steps if not s.bytes_neutral]
        assert non_neutral in ([], ["bias"]), f"{lay.name} changes values in {non_neutral}"
    assert L.W4A8_INT4.bytes_neutral
    assert L.W4A8_INT4.name == "mmarow_tr_cl4"


def test_every_step_names_a_consumer_that_says_what_breaks():
    """The consumer is the useful half of the name: mma means the fragment's lane map, mem the cache line or bank,
    aiu the copy engine, cvt the converter's emission order. An unrecognised one means a step whose failure mode
    nobody wrote down."""
    for lay in L.LAYOUTS.values():
        for s in lay.steps:
            assert s.consumer in ("mma", "mem", "aiu", "cvt", "layout"), f"{s.token}: consumer {s.consumer!r}"
            assert s.upstream, f"{s.token} names no implementation"


# ---------------------------------------------------------------------------------------------------------------
# compatibility messages
# ---------------------------------------------------------------------------------------------------------------

def test_compatible_with_itself():
    for name in L.LAYOUTS:
        assert L.check_compatible(name, name) is None


@pytest.mark.parametrize("stored,required,expect", [
    ("mixed_gemm", "mixed_gemm_aiu", "step 4 differs"),
    ("mixed_gemm", "w4a8", "extra step 4"),
    ("logical", "mixed_gemm", "missing step 1"),
])
def test_incompatibility_names_the_step_not_just_the_layout(stored, required, expect):
    """'which reorder is missing' is the question a mismatch raises, and it used to take reading two functions."""
    msg = L.check_compatible(stored, required)
    assert msg and expect in msg


def test_an_alias_resolves_to_the_same_layout_as_the_canonical_name():
    assert L.resolve("mixed_gemm_aiu") is L.resolve("mmarow_tr_cl4_aiu256_cvtword_bias")


# ---------------------------------------------------------------------------------------------------------------
# the op
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def Q():
    torch = pytest.importorskip("torch")
    so = sorted((ROOT / "quactlize").glob("_C*.so"))
    if not so:
        pytest.skip("extension not built")
    torch.ops.load_library(str(so[0]))
    return torch.ops.quactlize


def test_op_accepts_every_registered_int4_layout(Q):
    import torch
    w = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, (512, 256), dtype=torch.int8))
    for name, lay in L.LAYOUTS.items():
        if lay.applies_to.startswith("packed int4") or name == "logical":
            out = Q.preprocess_weights_to_layout(w, torch.quint4x2, name)
            assert out.numel() == w.numel(), f"{name} is not byte-neutral"


def test_op_refuses_an_unregistered_name(Q):
    import torch
    w = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, (512, 256), dtype=torch.int8))
    with pytest.raises(RuntimeError, match="not a registered weight layout"):
        Q.preprocess_weights_to_layout(w, torch.quint4x2, "mmarow_tr_cl256_aiu256_cvtword_bias")


def test_op_refuses_a_layout_for_a_different_element_width(Q):
    import torch
    w = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, (512, 256), dtype=torch.int8))
    with pytest.raises(RuntimeError, match="8-bit weights"):
        Q.preprocess_weights_to_layout(w, torch.quint4x2, "mixed_gemm_int8")


def test_op_refuses_a_shape_that_would_skip_the_aiu_step(Q):
    """A shape that misses the multiple does not get the arrangement the name promises: the step is skipped
    downstream and the result is a different layout under the same name."""
    import torch
    w = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, (128, 256), dtype=torch.int8))
    with pytest.raises(RuntimeError, match="multiples of 256"):
        Q.preprocess_weights_to_layout(w, torch.quint4x2, "mixed_gemm_aiu")


def test_named_and_boolean_forms_agree(Q):
    """The named entry point is a front end, so it must produce exactly what the flags it resolves to produce. If
    these ever differ, the table in weight_layout.h is describing a chain the chain does not run."""
    import torch
    w = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, (512, 256), dtype=torch.int8))
    for name, mma, aiu in (("mixed_gemm", False, False), ("mixed_gemm_aiu", False, True), ("w4a8", True, False)):
        named = Q.preprocess_weights_to_layout(w, torch.quint4x2, name)
        flags = Q._preprocess_weights_for_mixed_gemm(w, torch.quint4x2, mma, aiu)
        assert torch.equal(named, flags), f"{name} disagrees with ({mma}, {aiu})"

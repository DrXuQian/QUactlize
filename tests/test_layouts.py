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
        return
    assert {"cl2": 8, "cl4": 4, "cl8": 2}[cl] == bits

    # THE MMA TILE HEIGHT IS DETERMINED BY THE ELEMENT WIDTH, so it can be derived and checked rather than trusted:
    # B_ROWS_PER_MMA = 8 * (16 / bits), which is 16 / 32 / 64 for 8 / 4 / 2-bit. The int2 layout was recorded with a
    # 32-row tile, and neither the table nor the name could show it while the height was omitted from the token
    # whenever it took its commonest value.
    mmarow = next(t for t in tokens if t.startswith("mmarow"))
    height = int(mmarow[len("mmarow"):].rstrip("i"))
    assert height == 8 * (16 // bits), f"{name}: {bits}-bit gives a {8 * (16 // bits)}-row MMA tile, token says {height}"

    # and the row-permutation VARIANT is the same bit as is_int8_mma
    assert mmarow.endswith("i") == is_int8_mma


# ---------------------------------------------------------------------------------------------------------------
# the naming rule itself
# ---------------------------------------------------------------------------------------------------------------

def test_a_name_is_the_ordered_join_of_its_step_tokens():
    assert L.MIXED_GEMM_AIU_INT4.name == "mmarow32_tr_cl4_aiu256_cvtword_bias"
    assert L.LOGICAL.name == "logical"


def test_the_two_row_permutation_formulas_get_different_tokens():
    """permute_B_rows_for_mixed_gemm holds two index formulas selected by is_int8_mma, and on a 32-row tile they
    disagree in 24 of 32 positions -- reproduced below so the claim is checked, not asserted. One token naming both
    would be a token that lies; the current table merely happened not to collide, because W4A8 also drops cvtword
    and bias, which is an accident of which OTHER steps differ."""
    imma = [(t % 8) // 4 * 16 + t // 8 * 4 + t % 4 for t in range(32)]
    ldsm = [8 * ((t % 8) // 2) + t % 2 + 2 * (t // 8) for t in range(32)]
    assert sorted(imma) == sorted(ldsm) == list(range(32)), "both must be permutations of the tile"
    assert sum(a != b for a, b in zip(imma, ldsm)) == 24
    assert L.mma_row(32).token != L.mma_row(32, imma=True).token


def test_the_tile_height_is_always_in_the_token():
    """No omission for a common value. A parameter that disappears when it takes its usual value is a parameter
    nobody can check, which is how the int2 layout carried a 32-row tile where the width demands 64."""
    for bits, height in ((8, 16), (4, 32), (2, 64)):
        assert L.mma_row(height).token == f"mmarow{height}"
        assert 8 * (16 // bits) == height


def test_changing_a_step_parameter_changes_the_name():
    """THE PROPERTY THE WHOLE SCHEME EXISTS FOR. A weight reordered before ColumnsInterleaved was corrected from 256
    to 4 was byte-count-, dtype- and shape-identical to a correct one, and there was no way to tell them apart.
    Under this rule it is named cl256, which is not registered, so loading it fails instead of computing nonsense."""
    stale = L.Layout((L.mma_row(32), L.axis_transpose(), L.mem_cacheline_col_tile(256),
                      L.aiu_col_tile(256), L.cvt_word_permute(), L.code_bias(8)), "stale", "the pre-fix reorder")
    assert stale.name == "mmarow32_tr_cl256_aiu256_cvtword_bias"
    assert stale.name != L.MIXED_GEMM_AIU_INT4.name
    with pytest.raises(KeyError, match="not a registered layout"):
        L.resolve(stale.name)


def test_adding_a_step_changes_the_name_with_no_version_counter():
    placed = L.Layout(L.MIXED_GEMM_AIU_INT4.steps + (L.xplane(4, 64, 64, 32, 1),), "placed", "xplane-placed")
    assert placed.name == "mmarow32_tr_cl4_aiu256_cvtword_bias_xp4n64k64wn32f1"
    assert placed.name not in L.LAYOUTS


# ---------------------------------------------------------------------------------------------------------------
# the xplane token, whose parameter set was measured rather than assumed
# ---------------------------------------------------------------------------------------------------------------

XPLANE_PROBE = r"""
#include <cstdio>
#include <vector>
#include "xplane_offline.hpp"
template <int B,int TM,int TN,int TK,int WM,int WN,int F>
void show() {
  auto m = xplane::plane_map<B, TM, TN, TK, WM, WN, F>();
  unsigned h = 2166136261u;
  for (size_t i = 0; i < m.size(); ++i) { h ^= (unsigned)m[i]; h *= 16777619u; }
  printf("%zu %08x\n", m.size(), h);
}
int main() {
  show<1,64,64,256,32,32,1>();   // baseline
  show<1,128,64,256,32,32,1>();  // TM varies
  show<1,64,64,256,32,64,1>();   // WN varies
  show<1,64,128,256,32,32,1>();  // TN varies
  show<1,64,64,256,32,32,2>();   // F varies
  show<1,64,64,128,32,32,2>();   // TK varies (with F=2 to stay legal)
  show<1,64,64,128,64,32,2>();   // WM varies
  show<1,64,64,128,32,64,2>();   // WN varies again, at the other TK
  return 0;
}
"""


@pytest.mark.slow
def test_the_xplane_token_carries_exactly_the_parameters_that_change_the_map():
    """THE MEASUREMENT THE TOKEN RESTS ON, re-run rather than remembered.

    Omitting a parameter that matters is silent -- two arrangements share a name and one of them computes nonsense.
    Including one that does not is loud -- a false rejection someone investigates. So the omissions of TM and WM are
    the risky half of this decision, and they are checked here by instantiating plane_map and comparing hashes.

    Needs nvcc. Skipped, not failed, without it: this pins a claim about the C++ template, and a machine that cannot
    compile it cannot speak to the claim either way."""
    import shutil, subprocess, tempfile
    if not shutil.which("nvcc"):
        pytest.skip("needs nvcc to instantiate xplane::plane_map")
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / "probe.cu", Path(d) / "probe"
        src.write_text(XPLANE_PROBE)
        r = subprocess.run(["nvcc", "-std=c++17", "-x", "cu", "-arch=sm_80", "-w",
                            "-I", str(ROOT / "dev/fold_derivation/stub_inc"),
                            "-I", str(ROOT / "quactlize/include"),
                            "-I", str(ROOT / "third_party/actlize/include"),
                            "-o", str(exe), str(src)], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"xplane probe does not build here: {r.stderr.strip().splitlines()[:1]}")
        out = subprocess.run([str(exe)], capture_output=True, text=True).stdout.split("\n")
    rows = [l for l in out if l.strip()][:8]
    assert len(rows) == 8, f"probe printed {len(rows)} rows, expected 8"
    base, tm, wn, tn, f, tk, wm, wn2 = rows

    # These MUST change the map, so they are in the token
    for label, got in (("TN", tn), ("F", f), ("TK", tk)):
        assert got != base, f"{label} does not change the map -- the token carries a parameter that does nothing"

    # WN IS CONDITIONAL, and that is the interesting case. At (TK=256, F=1) it changes nothing; at (TK=128, F=2) it
    # does. A token cannot be conditional, so a parameter that matters in ANY legal configuration has to be carried
    # in all of them -- carrying it where it is inert costs a false rejection, which is loud, while dropping it where
    # it matters costs two arrangements sharing a name, which is silent. The first draft of this test asserted WN
    # changes the map everywhere; it does not, and the conclusion survives the correction unchanged.
    assert wn == base, "WN now changes the map at (TK=256, F=1) too -- the conditionality noted here has moved"
    assert wn2 != tk, "WN no longer changes the map at (TK=128, F=2) -- the only evidence for carrying it is gone"

    # These must NOT, which is why the token omits them. This is the assertion that would break first if the
    # placement ever grew a dependence on the M axis, and it is the omission that would be silent.
    assert tm == base, "TM changes the map -- it must be added to the xplane token"
    assert wm == tk, "WM changes the map -- it must be added to the xplane token"


def test_xplane_rejects_a_configuration_that_cannot_be_built():
    """F * TK * bits >= 256 is the AIU's 32-byte delivery floor, and it is why int1 exists only at (TK=256, F=1) and
    (TK=128, F=2). A name that violates it describes a placement the template refuses to instantiate, so the name
    should refuse too rather than being written down and failing later at a static_assert."""
    with pytest.raises(ValueError, match="32-byte delivery floor"):
        L.xplane(1, 64, 128, 32, 1)
    assert L.xplane(1, 64, 256, 32, 1).token == "xp1n64k256wn32f1"
    assert L.xplane(1, 64, 128, 32, 2).token == "xp1n64k128wn32f2"


def test_the_xplane_token_distinguishes_configurations_that_differ():
    """One token per (bits, TN, TK, WN, F). The two tokens this replaced -- plane<P> and foldn<F> -- were two names
    for facets of ONE transform, and between them carried two of the five parameters."""
    seen = {L.xplane(*c).token for c in [(1, 64, 256, 32, 1), (1, 64, 128, 32, 2), (1, 64, 128, 64, 2),
                                         (1, 128, 256, 32, 1), (2, 64, 128, 32, 1), (4, 64, 64, 32, 1)]}
    assert len(seen) == 6, "two different placements collided on one token"



def test_only_the_bias_step_changes_values():
    """Every step but one is a pure permutation, which is what lets the whole chain be byte-neutral. `bias` is the
    exception and is its own step for that reason, though it currently shares a function with cvtword."""
    for lay in L.LAYOUTS.values():
        non_neutral = [s.token for s in lay.steps if not s.bytes_neutral]
        assert non_neutral in ([], ["bias"]), f"{lay.name} changes values in {non_neutral}"
    assert L.W4A8_INT4.bytes_neutral
    assert L.W4A8_INT4.name == "mmarow32i_tr_cl4"


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
    # step 1, not a later one: W4A8's row permutation genuinely differs from mixed_gemm's. Before the variant was
    # in the token this reported "extra step 4", pointing at the bias while the real incompatibility was the first
    # transform in the chain -- a diagnosis that would have sent someone to the wrong function.
    ("mixed_gemm", "w4a8", "step 1 differs"),
    ("logical", "mixed_gemm", "missing step 1"),
])
def test_incompatibility_names_the_step_not_just_the_layout(stored, required, expect):
    """'which reorder is missing' is the question a mismatch raises, and it used to take reading two functions."""
    msg = L.check_compatible(stored, required)
    assert msg and expect in msg


def test_an_alias_resolves_to_the_same_layout_as_the_canonical_name():
    assert L.resolve("mixed_gemm_aiu") is L.resolve("mmarow32_tr_cl4_aiu256_cvtword_bias")


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
        Q.preprocess_weights_to_layout(w, torch.quint4x2, "mmarow32_tr_cl256_aiu256_cvtword_bias")


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

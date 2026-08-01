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
    """The C++ registry AS THE COMPILER SEES IT: compile ci/dump_weight_layouts.cpp and read what it prints.

    This replaces a regex over the header, which codex defeated by hand -- a commented-out table with the expected
    rows placed ahead of the real one, then a real row changed. Every regex comparison read the comment and passed
    while the compiled resolver returned the changed value. Comments, #if, macros and line continuations are things
    a compiler resolves and a pattern cannot, so the comparison has to be against the compiler's answer.

    Returns [] if g++ is missing, and the tests that use it skip: a machine that cannot compile the header cannot
    speak to what it compiles to.
    """
    import shutil, subprocess, tempfile
    if not shutil.which("g++"):
        return []
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "dump"
        r = subprocess.run(["g++", "-std=c++17", "-I", str(ROOT / "quactlize/csrc/preprocess"),
                            "-o", str(exe), str(ROOT / "ci/dump_weight_layouts.cpp")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"weight_layout.h does not compile: {r.stderr.strip()[:400]}")
        out = subprocess.run([str(exe)], capture_output=True, text=True)
        assert out.returncode == 0, f"the registry dump failed: {out.stdout}{out.stderr}"
    rows = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        name, alias, mma, aiu, bits, mult = line.split("\t")
        rows.append((name, alias, mma == "1", aiu == "1", int(bits), int(mult)))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# the two copies agree
# ---------------------------------------------------------------------------------------------------------------

def test_the_cpp_table_and_the_python_registry_have_the_same_names_in_the_same_order():
    """Order too, not just the set: the table is meant to be diffable against layouts.py by eye, and a reordering
    would break that without breaking anything a set comparison could see."""
    rows = cpp_registry()
    if not rows:
        pytest.skip("needs g++ to compile the registry dump")
    assert [r[0] for r in rows] == list(L.LAYOUTS)


def test_the_two_copies_agree_on_aliases():
    rows = cpp_registry()
    if not rows:
        pytest.skip("needs g++ to compile the registry dump")
    assert {r[0]: r[1] for r in rows} == {n: lay.alias for n, lay in L.LAYOUTS.items()}


@pytest.mark.parametrize("row", cpp_registry() or [pytest.param(None, marks=pytest.mark.skip(reason="needs g++"))],
                         ids=lambda r: r[0] if r else "no-g++")
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
    assert placed.name == "mmarow32_tr_cl4_aiu256_cvtword_bias_xp4w2k64f1"
    assert placed.name not in L.LAYOUTS


# ---------------------------------------------------------------------------------------------------------------
# the xplane token, whose parameter set was measured rather than assumed
# ---------------------------------------------------------------------------------------------------------------

# Compares the STORED BYTES of place_derived, not the plane_map an earlier version hashed. plane_map is an
# intermediate whose size varies with TN for reasons that never reach the buffer; measuring it concluded the
# placement depended on five parameters when it depends on four, one of them a ratio. A name for a stored
# arrangement has to be derived from stored bytes.
XPLANE_PROBE = r"""
#include <cstdio>
#include <vector>
#include <cstdint>
#include "xplane_offline.hpp"
constexpr int N = 256, K = 256, B = 2;
static std::vector<uint8_t> mk(){ std::vector<uint8_t> q((size_t)N*K); uint32_t s=0x9e3779b9u;
  for(size_t i=0;i<q.size();++i){s^=s<<13;s^=s>>17;s^=s<<5;q[i]=uint8_t(s&3);} return q; }
static std::vector<uint8_t> Q = mk();
template <int TM,int TN,int TK,int WM,int WN,int F>
std::vector<int8_t> place(){ std::vector<int8_t> o((size_t)N*K*B/8);
  xplane::place_derived<B,TM,TN,TK,WM,WN,F>(o.data(), Q, N, K); return o; }
static void d(std::vector<int8_t> const& a, std::vector<int8_t> const& b){
  size_t n=0; for(size_t i=0;i<a.size();++i) n += a[i]!=b[i]; printf("%zu ", n); }
int main(){
  auto base = place<64,64,64,32,32,2>();          // WON = 64/32 = 2
  d(base, place<128,64,64,32,32,2>());            // 0: TM
  d(base, place<64,64,64,64,32,2>());             // 1: WM
  d(base, place<64,128,64,32,64,2>());            // 2: same WON, different TN and WN
  d(base, place<64,128,64,32,32,2>());            // 3: WON 2 -> 4
  d(base, place<64,64,64,32,64,2>());             // 4: WON 2 -> 1
  d(base, place<64,64,128,32,32,2>());            // 5: TK
  d(base, place<64,64,128,32,32,1>());            // 6: TK and F
  return 0;
}
"""


@pytest.mark.slow
def test_the_xplane_token_carries_exactly_what_changes_the_stored_bytes():
    """THE MEASUREMENT THE TOKEN RESTS ON, re-run rather than remembered -- and on the right object this time.

    Two failure directions, and they are not symmetric. Omitting a parameter that matters means two arrangements
    share a name, which is silent and computes nonsense. Carrying one that does not means a false rejection, which
    is loud. Both are wrong; only one is dangerous. So TM and WM are omitted on measurement, TN and WN are folded
    into their ratio on measurement, and this re-runs both."""
    import shutil, subprocess, tempfile
    if not shutil.which("nvcc"):
        pytest.skip("needs nvcc to instantiate xplane::place_derived")
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
        out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
    n = [int(x) for x in out.split()]
    assert len(n) == 7, f"probe printed {len(n)} numbers, expected 7"
    tm, wm, same_won, won4, won1, tk, tk_f = n

    assert tm == 0, "TM reaches the stored bytes -- it must be added to the xplane token"
    assert wm == 0, "WM reaches the stored bytes -- it must be added to the xplane token"
    assert same_won == 0, ("TN and WN reach the stored bytes independently of their ratio -- the token must carry "
                           "them separately again")
    for label, got in (("WON 2->4", won4), ("WON 2->1", won1), ("TK", tk), ("TK+F", tk_f)):
        assert got > 0, f"{label} does not reach the stored bytes -- the token carries a parameter that does nothing"


def test_the_won_ratio_is_what_the_token_carries():
    """Same ratio, same token, however TN and WN are split between them."""
    assert L.xplane(2, 64, 64, 32, 2).token == L.xplane(2, 128, 64, 64, 2).token == "xp2w2k64f2"
    assert L.xplane(2, 128, 64, 32, 2).token == "xp2w4k64f2"
    assert L.xplane(2, 64, 64, 64, 2).token == "xp2w1k64f2"



def test_xplane_rejects_a_configuration_that_cannot_be_built():
    """F * TK * bits >= 256 is the AIU's 32-byte delivery floor, and it is why int1 exists only at (TK=256, F=1) and
    (TK=128, F=2). A name that violates it describes a placement the template refuses to instantiate, so the name
    should refuse too rather than being written down and failing later at a static_assert."""
    with pytest.raises(ValueError, match="32-byte delivery floor"):
        L.xplane(1, 64, 128, 32, 1)
    assert L.xplane(1, 64, 256, 32, 1).token == "xp1w2k256f1"
    assert L.xplane(1, 64, 128, 32, 2).token == "xp1w2k128f2"


XPLANE_HI_PROBE = r"""
#include <cstdio>
#include <vector>
#include <cstdint>
#include "xplane_offline.hpp"
int main() {
  auto a = xplane::tile_map_hi<4,1,64,64,128,32,32,2,1>();
  auto b = xplane::plane_map<1,64,64,128,32,32,2>();
  size_t n = a.size() < b.size() ? a.size() : b.size(), d = 0;
  for (size_t i = 0; i < n; ++i) d += (a[i] != b[i]);
  printf("%zu %zu\n", d, n);
  constexpr int N = 256, K = 256;
  std::vector<uint8_t> q((size_t)N * K);
  uint32_t s = 0x9e3779b9u;
  for (size_t i = 0; i < q.size(); ++i) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; q[i] = uint8_t(s & 1); }
  std::vector<int8_t> own((size_t)N * K / 8), hi(own.size());
  xplane::place_derived<1,64,64,128,32,32,2>(own.data(), q, N, K);
  xplane::place_hi<4,1,64,64,128,32,32,2,1>(hi.data(), q, N, K);
  size_t bd = 0;
  for (size_t i = 0; i < own.size(); ++i) bd += own[i] != hi[i];
  printf("%zu %zu\n", bd, own.size());
  return 0;
}
"""


@pytest.mark.slow
def test_a_high_plane_is_not_a_standalone_plane_at_the_same_parameters():
    """THE COLLISION THAT FORCED xplane_hi INTO EXISTENCE, re-measured rather than remembered.

    At identical (bits=1, TN=64, TK=128, WN=32, F=2), Q5's high plane placed through place_hi<low=4, high=1> and a
    standalone 1-bit placement through place_derived disagree in 6144 of 8192 map entries and 6114 of 8192 stored
    bytes on a non-aliasing probe. Both would have carried the name xp1n64k128wn32f2.

    If this test ever reports zero differences, the two placements have converged and xplane_hi is redundant -- so
    it fails on agreement as well as on a changed count."""
    import shutil, subprocess, tempfile
    if not shutil.which("nvcc"):
        pytest.skip("needs nvcc to instantiate the two placements")
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / "probe.cu", Path(d) / "probe"
        src.write_text(XPLANE_HI_PROBE)
        r = subprocess.run(["nvcc", "-std=c++17", "-x", "cu", "-arch=sm_80", "-w",
                            "-I", str(ROOT / "dev/fold_derivation/stub_inc"),
                            "-I", str(ROOT / "quactlize/include"),
                            "-I", str(ROOT / "third_party/actlize/include"),
                            "-o", str(exe), str(src)], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"probe does not build here: {r.stderr.strip().splitlines()[:1]}")
        lines = [l for l in subprocess.run([str(exe)], capture_output=True, text=True).stdout.split("\n") if l.strip()]
    map_diff, map_n = (int(x) for x in lines[0].split())
    byte_diff, byte_n = (int(x) for x in lines[1].split())
    assert (map_diff, map_n) == (6144, 8192), f"map differences moved: {map_diff}/{map_n}"
    assert (byte_diff, byte_n) == (6114, 8192), f"stored-byte differences moved: {byte_diff}/{byte_n}"
    # and the two tokens must differ, which is the whole point
    assert L.xplane(1, 64, 128, 32, 2).token != L.xplane_hi(4, 1, 64, 128, 32, 2, 1).token


def test_the_xplane_token_distinguishes_configurations_that_differ():
    """One token per (bits, TN, TK, WN, F). The two tokens this replaced -- plane<P> and foldn<F> -- were two names
    for facets of ONE transform, and between them carried two of the five parameters."""
    seen = {L.xplane(*c).token for c in [(1, 64, 256, 32, 1), (1, 64, 128, 32, 2), (1, 64, 128, 64, 2),
                                         (1, 256, 256, 32, 1), (2, 64, 128, 32, 1), (4, 64, 64, 32, 1)]}
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
    """Load the built extension, FAILING if it is older than the sources it was built from.

    A stale .so is the failure mode that wasted a debugging cycle here already: weight_layout.h was edited, setuptools
    does not track headers, and the rebuilt .so carried the previous table -- producing a test failure that read like
    a logic error. setup.py now declares the headers as dependencies, and this is the second line of defence, because
    a test that silently exercises last hour's binary reports on code nobody is running."""
    torch = pytest.importorskip("torch")
    so = sorted((ROOT / "quactlize").glob("_C*.so"))
    if not so:
        pytest.skip("extension not built -- python setup.py build_ext --inplace")
    built = so[0].stat().st_mtime
    newer = [p for p in (ROOT / "quactlize" / "csrc").rglob("*")
             if p.suffix in (".cpp", ".h", ".hpp") and p.stat().st_mtime > built]
    assert not newer, ("the built extension is older than " + ", ".join(p.name for p in newer[:4])
                       + " -- rebuild before trusting these results: python setup.py build_ext --inplace")
    torch.ops.load_library(str(so[0]))
    return torch.ops.quactlize


def test_op_accepts_every_registered_int4_layout(Q):
    """Byte count, VALUE MULTISET, and mutual distinctness -- not just numel.

    Checking numel alone accepts an all-zero output and any wrong permutation, which is most of what could go wrong.
    The multiset check is the same invariant used on the preprocessing itself: a layout transform may move nibbles
    and (where the layout has `bias`) shift them by 8, and may do nothing else. Distinctness is what says the layout
    argument was read at all -- three names returning identical bytes would pass every other assertion here."""
    import torch
    src = torch.randint(-8, 8, (512, 256), dtype=torch.int8)
    w = Q.pack_int8_tensor_to_packed_int4(src)
    plain = torch.sort(src.flatten().to(torch.int64) & 0xF).values
    biased = torch.sort((src.flatten().to(torch.int64) + 8) & 0xF).values

    produced = {}
    for name, lay in L.LAYOUTS.items():
        if not (lay.applies_to.startswith("packed int4") or name == "logical"):
            continue
        out = Q.preprocess_weights_to_layout(w, torch.quint4x2, name)
        assert out.numel() == w.numel() and out.dtype == w.dtype, f"{name} is not byte-neutral"

        b = out.flatten().to(torch.int32) & 0xFF
        nibbles = torch.sort(torch.stack([b & 0xF, (b >> 4) & 0xF], 1).flatten().to(torch.int64)).values
        want = biased if any(st.token == "bias" for st in lay.steps) else plain
        assert torch.equal(nibbles, want), f"{name} lost, duplicated or corrupted a nibble"

        # The WHOLE buffer, not a prefix. A 64-element fingerprint reported mixed_gemm and mixed_gemm_aiu as
        # identical: they differ in 65042 of 65536 bytes, but the first difference is at index 128, so the AIU
        # interleave leaves the opening tile in place. A prefix is not a fingerprint of a permutation.
        key = hash(out.flatten().numpy().tobytes())
        assert key not in produced, f"{name} produced the same bytes as {produced[key]} -- the layout was not read"
        produced[key] = name
    assert len(produced) >= 3, "fewer than three int4 layouts were exercised"


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
    """WIRING ONLY, and worth saying so. Both entry points call the same implementation, so this shows the name
    resolves to the flags the table claims -- it cannot show those flags produce the arrangement the name promises.
    The check that could is a kernel on the box reading the result."""
    import torch
    w = Q.pack_int8_tensor_to_packed_int4(torch.randint(-8, 8, (512, 256), dtype=torch.int8))
    for name, mma, aiu in (("mixed_gemm", False, False), ("mixed_gemm_aiu", False, True), ("w4a8", True, False)):
        named = Q.preprocess_weights_to_layout(w, torch.quint4x2, name)
        flags = Q._preprocess_weights_for_mixed_gemm(w, torch.quint4x2, mma, aiu)
        assert torch.equal(named, flags), f"{name} disagrees with ({mma}, {aiu})"

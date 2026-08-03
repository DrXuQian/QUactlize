"""THE TWO PLACES THAT STATE THE SAME FACTS MUST AGREE, and until this file existed nothing checked that they did.

schemes.py's header claimed formats.py's capability sets were "DERIVED FROM THIS TABLE rather than written beside
it. They were a second place where the same facts lived, and a second place is where drift starts." Every clause of
that was right except the tense: the sets were literals, schemes.capability() had NO CALLERS ANYWHERE, and the
drift the warning predicted duly happened -- a dense-path harness was registered on one side and the other side
still read `frozenset()`.

Derivation is not available: formats.py cannot import schemes.py, because schemes.py imports formats.py for
QuantType. So the literals stay -- they are the readable form, and a reader looking for "which formats does this
path support" should not have to run a function -- and this file is what makes them answerable to the matrix.

WHAT COUNTS AS SUPPORT. A capability set says the DISPATCHER may route a format to that path, so the bar is
IMPLEMENTED (the path runs), not VALIDATED (an independent oracle covers it). Those are different questions and
ci/registry.py answers the second one; conflating them would either bar a working path from being used or let an
unproven one look proven.
"""
import pytest

from quactlize import formats, schemes
from quactlize.schemes import Scheme, Shape, Status

# capability set in formats.py -> the (scheme, shapes) it corresponds to in the matrix.
# Mirrors _CELL_PATH in schemes.py, which maps the same cells to ci/registry.py's path names. Three vocabularies
# for one concept is two too many, and collapsing them is a real task; naming the correspondence in both places is
# the interim that at least fails when they disagree.
SETS = [
    ("GEMV",               Scheme.SCALE_FIRST,     (Shape.GEMV,)),
    ("FUSED_NATIVE_SCALE", Scheme.FULLY_QUANTIZED, (Shape.DENSE, Shape.GROUPED)),
    ("FUSED_FP16_SCALE",   Scheme.SCALE_FIRST,     (Shape.DENSE, Shape.GROUPED)),
    ("DEQUANT_THEN_DENSE", Scheme.DEQUANT_FIRST,   (Shape.DENSE, Shape.GROUPED)),
]


@pytest.mark.parametrize("set_name,scheme,shapes", SETS)
def test_capability_set_matches_the_matrix(set_name, scheme, shapes):
    literal = getattr(formats, set_name)
    derived = schemes.capability(scheme, *shapes, minimum=Status.IMPLEMENTED)
    missing = sorted(f.name for f in derived - literal)
    extra = sorted(f.name for f in literal - derived)
    assert not missing and not extra, (
        f"formats.{set_name} and the schemes matrix disagree.\n"
        f"  the matrix has these at IMPLEMENTED or better, the set omits them: {missing or 'none'}\n"
        f"  the set claims these, the matrix does not reach IMPLEMENTED:       {extra or 'none'}\n"
        f"Fix whichever is wrong -- but a set entry with no matrix cell behind it is a capability claim with "
        f"nothing running under it, which is the direction that ships a bad dispatch.")


def test_capability_has_a_caller_now():
    """The function this file exists to use. It had none, which is why the derivation claim went unchecked for as
    long as it did -- dead code cannot disagree with anything."""
    assert schemes.capability(Scheme.SCALE_FIRST, Shape.GROUPED, minimum=Status.VALIDATED), \
        "scale_first/grouped is the workhorse cell; an empty result means capability() itself is broken"


def test_every_matrix_cell_maps_to_a_registry_path():
    """A VALIDATED cell whose (scheme, shape) has no path name would be approved by DEFAULT, because the lookup
    that should refuse it finds nothing to check and returns no problem. Absence is the dangerous direction."""
    unmapped = sorted(f"{s.value}/{sh.value}" for s in Scheme for sh in Shape
                      if (s, sh) not in schemes._CELL_PATH)
    assert not unmapped, f"these cells map to no registry path, so a VALIDATED claim in them checks nothing: {unmapped}"


def test_no_cell_is_validated_on_borrowed_evidence():
    """The cells where the registry's four path names are coarser than the nine matrix cells.

    Sharing a path name means sharing evidence, and for these two the shared evidence is about a different kernel:
    fully_quantized/gemv would be approved by the fp16-PLANE GEMV harness, and dequant_first/gemv by a GEMM run at
    m in (1, 7, 64). Both are mapped -- unmapped is approved by default -- and this is what refuses them. When one
    of these genuinely becomes validated, split the vocabulary rather than deleting the entry here."""
    guilty = [f"{s.value}/{sh.value}/{f.name}"
              for (s, sh, f), impl in schemes.IMPL.items()
              if impl.status >= Status.VALIDATED and (s, sh) in schemes._CELL_PATH_IS_COARSE]
    assert not guilty, (
        "these cells claim VALIDATED while their only registry evidence is for a different kernel:\n  "
        + "\n  ".join(sorted(guilty))
        + "\nSplit ci/registry.py's path vocabulary before claiming them.")


def test_validated_cells_have_a_harness_for_that_path():
    """The registry cross-check, run as a test rather than only on demand.

    It compared per FORMAT until this was written, which meant a format validated on one cell approved every other
    cell containing it for free -- ten new VALIDATED cells were added and it reported nothing, and so did a planted
    claim on a path with no harness in existence. It compares per (format, path) now."""
    problems = schemes.check_against_registry()
    assert not problems, "VALIDATED cells with no harness behind them:\n  " + "\n  ".join(problems)

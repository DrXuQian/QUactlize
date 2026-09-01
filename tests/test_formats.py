"""THE FORMAT TABLE, CHECKED AGAINST THE FORMATS' PUBLISHED BLOCK SIZES.

quactlize/formats.py computes each format's storage cost from the FIELDS of its block struct rather than recording a
number. That is the right shape -- the same reason ColumnsInterleaved is derived and not written down -- but it moves
the risk rather than removing it: a mistyped field size gives a wrong block total and a plausible-looking growth
figure. So the block totals are checked here against ggml's documented sizes, which are independent of how this file
adds them up.

Needs no torch and no device.
"""
from pathlib import Path

import pytest

from quactlize.formats import (BLOCKS, PACKED_UNITS, DENSE_CROSSOVER_ROWS, DEQUANT_THEN_DENSE, FUSED_FP16_SCALE,
                               FUSED_NATIVE_SCALE, GEMV, QuantType,
                               needs_native_scale, packed_unit_layout, report, select_path, storage_growth)

# block size in bytes, from ggml-common.h. These are the numbers the format is DEFINED by -- a GGUF file's tensor
# size divides by them exactly -- so they are the right thing to check the field sums against.
GGML_BLOCK_SIZES = {
    QuantType.Q4_0: 18, QuantType.Q4_1: 20, QuantType.Q5_0: 22, QuantType.Q5_1: 24, QuantType.Q8_0: 34,
    QuantType.Q2_K: 84, QuantType.Q3_K: 110, QuantType.Q4_K: 144, QuantType.Q5_K: 176, QuantType.Q6_K: 210,
}


@pytest.mark.parametrize("qtype,size", sorted(GGML_BLOCK_SIZES.items()))
def test_block_bytes_match_ggml(qtype, size):
    assert BLOCKS[qtype].block_bytes == size, f"{qtype.name}: field sum disagrees with ggml's block size"


@pytest.mark.parametrize("qtype", sorted(BLOCKS))
def test_scale_meta_is_part_of_the_block(qtype):
    b = BLOCKS[qtype]
    assert 0 < b.scale_meta_bytes < b.block_bytes
    assert b.weights % b.group_size == 0


def test_ggml_type_numbers_are_ggmls():
    """The enum values are ggml_type, so a GGUF file's type field indexes this directly. Getting one wrong would
    silently route a tensor to the wrong format, which no amount of numerical testing downstream would attribute
    back here."""
    assert (QuantType.Q4_0, QuantType.Q4_1, QuantType.Q5_0, QuantType.Q5_1, QuantType.Q8_0) == (2, 3, 6, 7, 8)
    assert (QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K) == (10, 11, 12, 13, 14)


def test_packed_unit_geometry_is_explicit_and_byte_neutral():
    expected = {
        QuantType.Q2_K: (1, 20), QuantType.Q3_K: (2, 28),
        QuantType.Q4_K: (1, 16), QuantType.Q5_K: (1, 16),
        QuantType.Q6_K: (2, 36),
    }
    assert set(PACKED_UNITS) == set(expected)
    for qtype, (superblocks, unit_bytes) in expected.items():
        unit = packed_unit_layout(qtype)
        assert unit.superblocks_per_unit == superblocks
        assert unit.unit_bytes == unit_bytes
        assert unit.bytes_per_superblock == BLOCKS[qtype].scale_meta_bytes
    with pytest.raises(ValueError, match="no packed metadata unit"):
        packed_unit_layout(QuantType.Q8_0)


def test_legacy_formats_cost_nothing_on_the_fp16_scale_path():
    """Their scale meta already IS fp16, so the native-scale question does not arise for them. If any of these ever
    reports a non-zero growth, a field size is wrong."""
    for q in (QuantType.Q4_0, QuantType.Q4_1, QuantType.Q5_0, QuantType.Q5_1, QuantType.Q8_0):
        assert storage_growth(q) == 0.0, f"{q.name} should have fp16 scales already"
        assert not needs_native_scale(q)


@pytest.mark.parametrize("qtype,expected_pct", [
    (QuantType.Q2_K, 52.4), (QuantType.Q3_K, 16.4), (QuantType.Q4_K, 11.1),
    (QuantType.Q5_K, 9.1), (QuantType.Q6_K, 6.7),
])
def test_kquant_growth_is_what_the_format_decision_was_made_on(qtype, expected_pct):
    """These five numbers decided which formats can ship without a native scale channel, so they are pinned. The
    expected values are the ones the decision was taken against; if the computation changes, the decision has to be
    revisited rather than the test updated."""
    assert storage_growth(qtype) * 100 == pytest.approx(expected_pct, abs=0.05)


def test_every_kquant_needs_a_native_channel_and_gptq_does_not():
    """The one-line summary of the whole format plan: GPTQ symmetric can ship today because its scales are already
    fp16; every k-quant grows the weight on the fp16-scale path and therefore cannot."""
    for q in (QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K):
        assert needs_native_scale(q)
    assert not needs_native_scale(QuantType.GPTQ_INT4_SYM)
    assert needs_native_scale(QuantType.AWQ_INT4), "AWQ's zeros are int4; as fp16 they are 4x"


# ---------------------------------------------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------------------------------------------

def test_decode_gemv_capability_and_storage_policy_are_separate():
    """All five k-quants can run the scale-first GEMV, but auto selection does not silently materialise planes.

    A caller explicitly providing a plane workspace can select it; the default Q4_K route remains its native-scale
    implementation. This keeps a product storage policy from being mistaken for missing technical support.
    """
    assert select_path(QuantType.GPTQ_INT4_SYM, 1) == "gemv"
    assert {QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K} <= GEMV
    assert select_path(QuantType.Q4_K, 1) == "fused_native_scale"
    for q in (QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K):
        assert select_path(q, 1, fp16_planes="workspace") == "gemv"


def test_the_middle_band_prefers_native_scale_when_it_exists():
    assert select_path(QuantType.Q4_K, 128) == "fused_native_scale"


def test_storage_admissibility_is_part_of_selection_not_a_separate_check():
    """THE POINT OF THIS FILE IN ONE TEST. Q4_K without its native channel has only plane-consuming paths left, and
    materialising those planes is the increase the constraint forbids. Selection must refuse, not quietly route
    there -- an earlier version returned fused_fp16_scale for a format its own needs_native_scale() called
    inadmissible, which is the file contradicting itself."""
    assert needs_native_scale(QuantType.Q4_K)
    with pytest.raises(NotImplementedError, match="grows the stored weight"):
        select_path(QuantType.Q4_K, 128, native_scale_available=False)


def test_transient_planes_are_allowed_because_a_workspace_is_not_storage():
    """The constraint is on STORED bytes. A prefill pre-pass may build fp16 planes and discard them; the weight on
    disk and in HBM is unchanged. That is what separates 'Q3_K cannot ship' from 'Q3_K cannot prefill'.

    The row counts are BELOW DENSE_CROSSOVER_ROWS on purpose. At 2048 this used to read fused_fp16_scale and now
    reads dequant_then_dense, because populating that capability set activated a branch that had always been there
    -- so the original 2048 would have kept passing while testing a different claim. The claim here is about
    planes_ok, and it needs a row count where the plane-consuming path is the one under test.

    AND IT HAPPENED A SECOND TIME, to this test, for the same reason. Q3_K used to reach fused_fp16_scale here
    because it was not in FUSED_NATIVE_SCALE; the nine ppu001 promotions put all five k-quants in that set, so it
    now routes native -- correctly, since native materialises NOTHING and this test's whole subject is the cost
    of materialising. The old line would have gone on passing only while Q3_K stayed unsupported, which is a
    strange thing for a test to depend on.

    So the case that carries the claim is now the one BELOW: a format whose native channel is unavailable, which
    is the only remaining way to need transient planes. Populating a capability set is a DISPATCH CHANGE, and
    this file has now recorded two of them."""
    assert select_path(QuantType.Q3_K, 256, fp16_planes="workspace") == "fused_native_scale"
    assert select_path(QuantType.Q4_K, 128, native_scale_available=False,
                       fp16_planes="workspace") == "fused_fp16_scale"
    # THE REFUSAL, and its premise moved for the same reason as the line above. This used to be
    # select_path(Q3_K, 2048, fp16_planes="never") -- it raised because Q3_K had no native channel, so "never
    # materialise" left nothing. With all five k-quants native it no longer raises, CORRECTLY: native
    # materialises nothing, so the request is satisfiable. The property is still worth pinning, so it is pinned
    # where it can still be false -- native unavailable, planes forbidden, and no path left.
    with pytest.raises(NotImplementedError):
        select_path(QuantType.Q3_K, 2048, native_scale_available=False, fp16_planes="never")


def test_a_format_whose_scales_are_already_fp16_needs_no_special_pleading():
    """GPTQ symmetric passes fp16_planes='auto' because its scales ARE fp16 planes -- nothing is materialised, so
    nothing grows. This is the mechanism by which one format can ship today and the k-quants cannot."""
    assert select_path(QuantType.GPTQ_INT4_SYM, 128) == "fused_fp16_scale"
    assert storage_growth(QuantType.GPTQ_INT4_SYM) == 0.0


def test_the_dense_fallback_is_populated_by_a_harness_and_not_by_an_edit():
    """The set was EMPTY, and the note said populating it needed a dense-path harness rather than an edit here. That
    is what happened: tests/test_gguf_routes.py runs raw blocks -> fp16 weight -> torch's cuBLAS, dense and
    per-expert, against the official gguf package.

    GPTQ stays out, and the exclusion is the substance of this test. routes.py reads k-quant blocks; the symmetric
    packed forms have no host binding, so nothing can call the route for them. A set that generalises from the five
    formats a harness covers to the six a path could in principle serve is how a claim outruns its evidence -- which
    is exactly what the previous version of this set did before per-(format, path) evidence caught it."""
    assert DEQUANT_THEN_DENSE == frozenset({
        QuantType.Q2_K, QuantType.Q3_K, QuantType.Q4_K, QuantType.Q5_K, QuantType.Q6_K})
    assert QuantType.GPTQ_INT4_SYM not in DEQUANT_THEN_DENSE


def test_populating_a_capability_set_changed_the_ROUTING(  ):
    """POPULATING A SET IS A DISPATCH CHANGE, and it happened silently the first time.

    select_path's second branch -- num_rows >= DENSE_CROSSOVER_ROWS and qtype in DEQUANT_THEN_DENSE -- was written
    to prefer the dense fallback at large M and was inert only because the set was empty. Filling the set activated
    it, and two existing tests changed answer without anyone deciding to change them.

    The routing is what the branch always intended. What is worth pinning is that it now rests on
    DENSE_CROSSOVER_ROWS, whose own comment says the crossover has NOT been swept -- it is one measurement at
    M=2048 (2.1x) turned into a boundary. So this asserts the new behaviour AND that it is a boundary someone chose,
    by checking it flips exactly at that constant rather than at some emergent value."""
    below = select_path(QuantType.Q4_K, DENSE_CROSSOVER_ROWS - 1, fp16_planes="workspace")
    at = select_path(QuantType.Q4_K, DENSE_CROSSOVER_ROWS, fp16_planes="workspace")
    assert below == "fused_native_scale", f"below the crossover Q4_K should stay native, got {below}"
    assert at == "dequant_then_dense", f"at the crossover Q4_K should go dense, got {at}"


def test_a_format_with_no_path_at_all_raises():
    for q in (QuantType.AWQ_INT4, QuantType.GPTQ_INT4_ASYM, QuantType.Q8_0):
        with pytest.raises(NotImplementedError):
            select_path(q, 128)


def test_native_scale_set_is_a_subset_of_the_fp16_one():
    """A format the fused kernel can run natively must also be runnable with prepared fp16 planes -- the native
    channel is an optimisation of the same kernel, so the reverse would mean the fallback cannot serve it."""
    assert FUSED_NATIVE_SCALE <= FUSED_FP16_SCALE


def test_gemv_formats_have_a_batched_path_too():
    """Anything that can decode must also be able to prefill; a model cannot run on the decode path alone."""
    assert GEMV <= (FUSED_FP16_SCALE | DEQUANT_THEN_DENSE)


def test_every_capability_claim_has_evidence_behind_it():
    """The registry cross-check, run from the test suite as well as from the CI tier, because this is the assertion
    that keeps formats.py from becoming a wish list. Each problem it reports is a claim to withdraw."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ci"))
    import registry
    assert registry.check_against_formats() == []


def test_report_names_every_format():
    text = report()
    for q in QuantType:
        assert q.name in text

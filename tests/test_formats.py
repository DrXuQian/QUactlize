"""THE FORMAT TABLE, CHECKED AGAINST THE FORMATS' PUBLISHED BLOCK SIZES.

quactlize/formats.py computes each format's storage cost from the FIELDS of its block struct rather than recording a
number. That is the right shape -- the same reason ColumnsInterleaved is derived and not written down -- but it moves
the risk rather than removing it: a mistyped field size gives a wrong block total and a plausible-looking growth
figure. So the block totals are checked here against ggml's documented sizes, which are independent of how this file
adds them up.

Needs no torch and no device.
"""
import pytest

from quactlize.formats import (BLOCKS, DEQUANT_THEN_DENSE, FUSED_FP16_SCALE, FUSED_NATIVE_SCALE, GEMV, QuantType,
                               needs_native_scale, report, select_path, storage_growth)

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

def test_decode_takes_the_gemv_path_and_prefill_the_dense_one():
    assert select_path(QuantType.Q4_K, 1) == "gemv"
    assert select_path(QuantType.Q4_K, 2048) == "dequant_then_dense"
    assert select_path(QuantType.GPTQ_INT4_SYM, 1) == "gemv"


def test_the_middle_band_prefers_native_scale_when_it_exists():
    assert select_path(QuantType.Q4_K, 128) == "fused_native_scale"
    assert select_path(QuantType.Q4_K, 128, native_scale_available=False) == "fused_fp16_scale"


def test_a_format_without_a_fast_path_falls_back_rather_than_failing():
    """Q3_K has no GEMV kernel. At one row it must still produce an answer, on a slower path -- falling back is a
    real outcome, and it is the difference between 'slow' and 'unsupported'."""
    assert QuantType.Q3_K not in GEMV
    assert select_path(QuantType.Q3_K, 1) == "fused_fp16_scale"


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


def test_report_names_every_format():
    text = report()
    for q in QuantType:
        assert q.name in text

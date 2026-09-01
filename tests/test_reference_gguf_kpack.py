import importlib.util
import hashlib
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "reference" / "gguf_kpack.py"


def _load_reference():
    spec = importlib.util.spec_from_file_location("quactlize_gguf_kpack_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ref = _load_reference()


# Produced independently by the C++ raw-block decoder, canonical placement
# headers and packed-unit producer at develop@725a41f.  These bind the Python
# carrier to exact bytes rather than accepting a mutually consistent
# pack/recover pair with the wrong physical map.
CPP_GOLDEN_SHA256 = {
    10: (
        "17c5bc5ffa2c3796060988b43dcbe6b8ab15f1e0cfff8d1fc203163b511cf834",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "8e74ce7f1c0a324e377025b52d041fe951adb73382476027c962c18f18182b06",
    ),
    11: (
        "25d1cd007ac3b09944ee3fe7907f8a6c28b1648a805b125d72646ee2a4442673",
        "aeb7f677e045094859431309010b14a87131d0cd0227f263472145d0d2b12065",
        "681e897a5308d65fb365262bc53445b179a2ef685d951c430285a8ebb789a580",
    ),
    12: (
        "3d4096663c410cf9fe53640b326327d3bc25af58c52a183a9fa9bd0c35652452",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "1a9546a67e9b7a15374150f1c418d00f98b5aa48f25fe21f0231013e805f338a",
    ),
    13: (
        "53f75fa3c1661bd764addd82ef2c183247a6244645d90e5ebc780f9ed6a41fe2",
        "c59f97a90cac811b5c7f5ac46cadc1f53a3ddd1a580ec8423c5f9d5584a1a4f4",
        "316e801f5b78e98e736a8640c15eaba740256fa9d14ec52578f45862f571d1ab",
    ),
    14: (
        "a38cc40e0b4763215d7799a02ac61ba2f391a537729c04203c9c78275f54b27e",
        "c32f0a1246d31041c5cc6b1f316d7ce445f4f1dd41933ab309709989ed938ea7",
        "b679c272d8da7aec284aa4b9ee41aba5960c2e7861a70e264604ee7e5f8d1285",
    ),
}


def _sha256(tensor):
    return hashlib.sha256(bytes(tensor.reshape(-1).tolist())).hexdigest()


def _raw(spec, *, n=256, k=512, experts=1):
    rows = experts * n * (k // 256)
    values = bytearray((i * 73 + spec.qtype * 19 + 7) & 0xFF
                       for i in range(rows * spec.raw_bytes))
    return torch.frombuffer(values, dtype=torch.uint8).clone().reshape(rows, spec.raw_bytes)


@pytest.mark.parametrize("qtype", [10, 11, 12, 13, 14])
def test_reference_dense_is_byte_exact_and_has_canonical_carriers(qtype):
    spec = ref.SPECS[qtype]
    raw = _raw(spec)
    artifact = ref.prepare_dense(raw, 256, 512, qtype)

    assert artifact.low.shape == (1, 256, 512 * spec.low_bits // 8)
    expected_high = ((1, 256, 512 * spec.high_bits // 8)
                     if spec.high_bits else (0,))
    assert artifact.high.shape == expected_high
    assert artifact.units.shape == (
        2 // spec.superblocks_per_unit, 256, spec.unit_bytes)
    assert artifact.arrangement == ref.canonical_arrangement(qtype)
    assert tuple(map(_sha256, (artifact.low, artifact.high, artifact.units))) == \
        CPP_GOLDEN_SHA256[qtype]
    assert torch.equal(ref.recover_raw_blocks(artifact), raw)


@pytest.mark.parametrize("qtype", [11, 13])
def test_reference_grouped_adds_only_the_expert_major_axis(qtype):
    spec = ref.SPECS[qtype]
    raw = _raw(spec, experts=2)
    grouped = ref.prepare_grouped(raw, 256, 512, qtype, experts=2)
    per_expert_rows = 256 * 2

    assert grouped.units.shape == (
        2, 2 // spec.superblocks_per_unit, 256, spec.unit_bytes)
    for expert in range(2):
        dense = ref.prepare_dense(
            raw[expert * per_expert_rows:(expert + 1) * per_expert_rows],
            256, 512, qtype)
        assert torch.equal(grouped.low[expert], dense.low[0])
        assert torch.equal(grouped.high[expert], dense.high[0])
        assert torch.equal(grouped.units[expert], dense.units)
    assert torch.equal(ref.recover_raw_blocks(grouped), raw)


def test_q4_word_is_little_endian_k8_gather():
    spec = ref.SPECS[12]
    raw = torch.zeros((256, spec.raw_bytes), dtype=torch.uint8)
    # N=0, SB=0, logical K={0,8,16,24}; all are low nibbles of
    # official Q4 bytes 16+K and become slots 0..3 of physical word (kg=0,n=0).
    raw[0, 16 + 0] = 1
    raw[0, 16 + 8] = 2
    raw[0, 16 + 16] = 3
    raw[0, 16 + 24] = 4
    artifact = ref.prepare_dense(raw, 256, 256, 12)
    flat = artifact.low.reshape(-1)
    assert flat[:2].tolist() == [0x21, 0x43]


def test_q5_high_plane_uses_its_special_n_k_bit_transpose():
    spec = ref.SPECS[13]
    raw = torch.zeros((256, spec.raw_bytes), dtype=torch.uint8)
    # logical N=8, K=0, high bit 1.  Q5 maps logical N bit3 into
    # b16 slot bit3 and K bit7 into physical N bit3, so this lands at
    # physical word (kg=0,n=0), bit8 -- bytes [0x00,0x01].
    raw[8, 16] = 1
    artifact = ref.prepare_dense(raw, 256, 256, 13)
    assert artifact.high.reshape(-1)[:2].tolist() == [0x00, 0x01]


def test_q3_q6_pair_metadata_along_k_within_one_n_column():
    for qtype in (11, 14):
        spec = ref.SPECS[qtype]
        raw = torch.zeros((256 * 2, spec.raw_bytes), dtype=torch.uint8)
        # The first two input rows are adjacent K superblocks of N=0.
        raw[0, spec.d_offset:spec.d_offset + 2] = torch.tensor([0x12, 0x34], dtype=torch.uint8)
        raw[1, spec.d_offset:spec.d_offset + 2] = torch.tensor([0x56, 0x78], dtype=torch.uint8)
        artifact = ref.prepare_dense(raw, 256, 512, qtype)
        unit = artifact.units[0, 0]
        assert unit[:2].tolist() == [0x12, 0x34]
        assert unit[spec.sb_bytes:spec.sb_bytes + 2].tolist() == [0x56, 0x78]


def test_reference_fails_closed_on_non_product_geometry_and_unknown_qtype():
    q2 = ref.SPECS[10]
    with pytest.raises(ValueError, match="N divisible by 256"):
        ref.prepare_dense(torch.zeros((1, q2.raw_bytes), dtype=torch.uint8), 1, 256, 10)
    with pytest.raises(ValueError, match="divisible by 512"):
        ref.prepare_dense(torch.zeros((256, 110), dtype=torch.uint8), 256, 256, 11)
    with pytest.raises(ValueError, match="unsupported"):
        ref.canonical_arrangement(99)

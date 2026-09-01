import importlib.util
import pathlib
import shutil
import sys

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "reference" / "gguf_kpack.py"
SPEC = importlib.util.spec_from_file_location("quactlize_gguf_kpack_file_reference", REFERENCE)
ref = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ref
SPEC.loader.exec_module(ref)


def _pattern(shape, seed):
    size = int(np.prod(shape))
    return ((np.arange(size, dtype=np.uint64) * 73 + seed) & 0xFF).astype(np.uint8).reshape(shape)


def _write_source(path: pathlib.Path, *, reserved=False, invalid_geometry=False):
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_name("K-pack reference fixture")
    writer.add_uint32("fixture.scalar", 17)
    writer.add_key_value(
        "fixture.array", [2, 3, 5, 7], gguf.GGUFValueType.ARRAY,
        sub_type=gguf.GGUFValueType.UINT32,
    )
    if reserved:
        writer.add_string(ref.REFERENCE_SCHEMA_KEY, "collision")

    q4_n = 128 if invalid_geometry else 256
    q4 = _pattern((q4_n, 144), 12)
    writer.add_tensor(
        "blk.0.attn_q.weight", q4, raw_shape=q4.shape,
        raw_dtype=gguf.GGMLQuantizationType.Q4_K,
    )
    q3 = _pattern((2, 256, 220), 11)
    writer.add_tensor(
        "blk.1.ffn_gate_exps.weight", q3, raw_shape=q3.shape,
        raw_dtype=gguf.GGMLQuantizationType.Q3_K,
    )
    q8 = _pattern((2, 34), 8)
    writer.add_tensor(
        "token_embd.weight", q8, raw_shape=q8.shape,
        raw_dtype=gguf.GGMLQuantizationType.Q8_0,
    )
    writer.add_tensor("output_norm.weight", np.arange(6, dtype=np.float32).reshape(2, 3))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _tensor_bytes(tensor):
    return np.ascontiguousarray(tensor.data).view(np.uint8).reshape(-1).tobytes()


def _real_metadata(reader):
    return {
        name: field for name, field in reader.fields.items()
        if field.types and not name.startswith("GGUF.")
    }


def test_reference_reads_and_writes_a_self_verifying_augmented_gguf(tmp_path):
    source = tmp_path / "source.gguf"
    output = tmp_path / "packed.gguf"
    _write_source(source)

    manifest = ref.pack_reference_gguf(source, output)
    assert output.is_file()
    assert [record["source_name"] for record in manifest["tensors"]] == [
        "blk.0.attn_q.weight", "blk.1.ffn_gate_exps.weight",
    ]
    assert manifest["tensors"][0]["carriers"]["high"] is None
    assert manifest["tensors"][1]["carriers"]["high"] is not None

    source_reader = gguf.GGUFReader(source)
    output_reader = gguf.GGUFReader(output)
    output_tensors = {tensor.name: tensor for tensor in output_reader.tensors}
    assert len(output_tensors) == 9  # four originals + Q4(low,units) + Q3(low,high,units)
    for tensor in source_reader.tensors:
        copied = output_tensors[tensor.name]
        assert copied.tensor_type == tensor.tensor_type
        assert copied.shape.tolist() == tensor.shape.tolist()
        assert _tensor_bytes(copied) == _tensor_bytes(tensor)
    for record in manifest["tensors"]:
        for carrier in record["carriers"].values():
            if carrier is not None:
                assert output_tensors[carrier["name"]].tensor_type == gguf.GGMLQuantizationType.I8

    source_meta = _real_metadata(source_reader)
    output_meta = _real_metadata(output_reader)
    for name, field in source_meta.items():
        assert output_meta[name].types == field.types
        assert output_meta[name].contents() == field.contents()
    assert output_meta[ref.REFERENCE_SCHEMA_KEY].contents() == ref.REFERENCE_GGUF_SCHEMA
    assert ref.verify_reference_gguf(output, source_path=source) == manifest


def test_reference_file_layer_covers_every_kquant_format(tmp_path):
    source = tmp_path / "all-formats.gguf"
    output = tmp_path / "all-formats.kpack.gguf"
    writer = gguf.GGUFWriter(source, "llama")
    formats = [
        ("Q2_K", gguf.GGMLQuantizationType.Q2_K, 84, 256),
        ("Q3_K", gguf.GGMLQuantizationType.Q3_K, 110, 512),
        ("Q4_K", gguf.GGMLQuantizationType.Q4_K, 144, 256),
        ("Q5_K", gguf.GGMLQuantizationType.Q5_K, 176, 256),
        ("Q6_K", gguf.GGMLQuantizationType.Q6_K, 210, 512),
    ]
    for index, (name, qtype, block_bytes, k) in enumerate(formats):
        raw = _pattern((256, (k // 256) * block_bytes), index + 10)
        writer.add_tensor(
            f"blk.{index}.weight.{name}", raw, raw_shape=raw.shape, raw_dtype=qtype,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    manifest = ref.pack_reference_gguf(source, output)
    assert [record["qtype"] for record in manifest["tensors"]] == [10, 11, 12, 13, 14]
    assert [record["carriers"]["high"] is not None for record in manifest["tensors"]] == [
        False, True, False, True, True,
    ]
    ref.verify_reference_gguf(output, source_path=source)


@pytest.mark.parametrize("plant", ["carrier", "source"])
def test_reference_verifier_rejects_carrier_and_preserved_source_byte_plants(tmp_path, plant):
    source = tmp_path / "source.gguf"
    output = tmp_path / "packed.gguf"
    corrupt = tmp_path / f"{plant}.gguf"
    _write_source(source)
    manifest = ref.pack_reference_gguf(source, output)
    shutil.copyfile(output, corrupt)

    reader = gguf.GGUFReader(corrupt, mode="r+")
    if plant == "carrier":
        name = manifest["tensors"][0]["carriers"]["low"]["name"]
    else:
        name = "token_embd.weight"  # unsupported Q8 passthrough is still inventoried
    payload = {tensor.name: tensor for tensor in reader.tensors}[name].data.view(np.uint8).reshape(-1)
    payload[0] ^= np.uint8(1)
    reader.data.flush()

    with pytest.raises(ValueError, match="checksum mismatch|byte payload changed"):
        ref.verify_reference_gguf(corrupt)


def test_reference_verifier_rejects_a_noncanonical_mapping_id(tmp_path):
    source = tmp_path / "source.gguf"
    output = tmp_path / "packed.gguf"
    corrupt = tmp_path / "wrong-mapping.gguf"
    _write_source(source)
    manifest = ref.pack_reference_gguf(source, output)
    shutil.copyfile(output, corrupt)

    reader = gguf.GGUFReader(corrupt, mode="r+")
    payload = reader.get_field(ref.REFERENCE_MANIFEST_KEY).parts[-1]
    mapping = manifest["tensors"][0]["arrangement"]["mapping_id"]
    old, new = str(mapping).encode(), str(mapping + 1).encode()
    assert len(old) == len(new)
    raw = payload.tobytes()
    offset = raw.index(old)
    payload[offset:offset + len(old)] = np.frombuffer(new, dtype=np.uint8)
    reader.data.flush()

    with pytest.raises(ValueError, match="arrangement is not canonical"):
        ref.verify_reference_gguf(corrupt)


def test_reference_verifier_binds_original_metadata_without_the_source_file(tmp_path):
    source = tmp_path / "source.gguf"
    output = tmp_path / "packed.gguf"
    corrupt = tmp_path / "metadata.gguf"
    _write_source(source)
    ref.pack_reference_gguf(source, output)
    shutil.copyfile(output, corrupt)

    reader = gguf.GGUFReader(corrupt, mode="r+")
    reader.get_field("fixture.scalar").parts[-1][0] = np.uint32(16)
    reader.data.flush()
    with pytest.raises(ValueError, match="source metadata 'fixture.scalar' changed"):
        ref.verify_reference_gguf(corrupt)


def test_reference_tensor_selection_keeps_every_original_and_converts_only_requested(tmp_path):
    source = tmp_path / "source.gguf"
    output = tmp_path / "selected.gguf"
    _write_source(source)
    manifest = ref.pack_reference_gguf(
        source, output, tensor_names=["blk.0.attn_q.weight"],
    )
    assert [record["source_name"] for record in manifest["tensors"]] == ["blk.0.attn_q.weight"]
    assert len(gguf.GGUFReader(output).tensors) == 6  # four originals + low + units
    ref.verify_reference_gguf(output, source_path=source)


def test_reference_file_io_fails_closed_before_publishing(tmp_path):
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    _write_source(source, reserved=True)
    with pytest.raises(ValueError, match="metadata namespace"):
        ref.pack_reference_gguf(source, output)
    assert not output.exists()

    invalid = tmp_path / "invalid.gguf"
    _write_source(invalid, invalid_geometry=True)
    with pytest.raises(ValueError, match="N divisible by 256"):
        ref.pack_reference_gguf(invalid, output)
    assert not output.exists()

    with pytest.raises(ValueError, match="paths must differ"):
        ref.pack_reference_gguf(invalid, invalid)
    output.write_bytes(b"owned")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ref.pack_reference_gguf(invalid, output)
    assert output.read_bytes() == b"owned"


def test_reference_gguf_cli_pack_and_verify(tmp_path, capsys):
    source = tmp_path / "source.gguf"
    output = tmp_path / "packed.gguf"
    _write_source(source)
    assert ref.main([
        "pack", str(source), str(output), "--tensor", "blk.0.attn_q.weight",
    ]) == 0
    assert "action=pack" in capsys.readouterr().out
    assert ref.main(["verify", str(output), "--source", str(source)]) == 0
    assert "action=verify" in capsys.readouterr().out

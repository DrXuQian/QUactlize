"""Strict contracts for the installable production K-pack blob bundle."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

from quactlize import formats, pack_gguf, routes


_RECIPES = {
    formats.QuantType.Q2_K: (2, 0, 1, 20),
    formats.QuantType.Q3_K: (2, 1, 2, 28),
    formats.QuantType.Q4_K: (4, 0, 1, 16),
    formats.QuantType.Q5_K: (4, 1, 1, 16),
    formats.QuantType.Q6_K: (4, 2, 2, 36),
}


class _FakeGGUFEndian:
    LITTLE = "little"
    BIG = "big"


def _fake_artifact(qtype, route, n, k, experts):
    expert_count = experts or 1
    shapes = pack_gguf._canonical_bundle_shapes(qtype, route, expert_count, n, k)
    tensors = []
    for index, name in enumerate(("low", "high", "units")):
        shape = shapes[name]
        numel = 1
        for extent in shape:
            numel *= extent
        value = ((torch.arange(numel, dtype=torch.int64) + index * 17) & 0xff).to(torch.uint8)
        tensors.append(value.reshape(shape))
    arrangement = (formats.q4_kpack4_arrangement()
                   if qtype == formats.QuantType.Q4_K
                   else formats.kquant_kpack_arrangement(qtype))
    return routes.PlacedArtifact(tuple(tensors), arrangement)


def _bundle(tmp_path, qtype=formats.QuantType.Q4_K, route="dense"):
    root = tmp_path / "bundle"
    root.mkdir()
    source = tmp_path / "model.gguf"
    grouped = route == "grouped"
    name = "blk.0.ffn_up_exps.weight" if grouped else "blk.0.attn_q.weight"
    n, k = 256, 512
    experts = 3 if grouped else 1
    raw_size = experts * n * (k // formats.BLOCKS[qtype].weights) * formats.BLOCKS[qtype].block_bytes
    source_payload = ((np.arange(raw_size, dtype=np.uint32) * 29 + int(qtype)) & 0xff).astype(np.uint8)
    source.write_bytes(source_payload.tobytes())
    low_bits, high_bits, sb_per_unit, unit_bytes = _RECIPES[qtype]
    arrangement = (formats.q4_kpack4_arrangement()
                   if qtype == formats.QuantType.Q4_K
                   else formats.kquant_kpack_arrangement(qtype))
    low = torch.zeros((experts, n, k * low_bits // 8), dtype=torch.uint8)
    high = (torch.zeros((experts, n, k * high_bits // 8), dtype=torch.uint8)
            if high_bits else torch.empty((0,), dtype=torch.uint8))
    unit_shape = (k // (256 * sb_per_unit), n, unit_bytes)
    units = torch.zeros(((experts,) + unit_shape) if grouped else unit_shape,
                        dtype=torch.uint8)
    artifact = routes.PlacedArtifact((low, high, units), arrangement)
    weights_path = root / pack_gguf.KPACK_BUNDLE_WEIGHTS
    with weights_path.open("xb") as weights:
        region, spans = pack_gguf._append_bundle_artifact(weights, artifact)
    record = {
        "name": name,
        "ggml_type": int(qtype),
        "type_name": qtype.name,
        "route_class": route,
        "layout_name": formats.canonical_fully_quantized_layout(qtype),
        "plane_packs": {"low": 16 // low_bits,
                        "high": 16 // high_bits if high_bits else 0},
        "rank": 3 if grouped else 2,
        "n": n,
        "k": k,
        "experts": experts if grouped else None,
        "arrangement_version": routes.PLACED_ARTIFACT_VERSION_V2,
        "arrangement": arrangement._asdict(),
        "source_tensor": {
            "index": 0,
            "data_offset": 0,
            "size_bytes": raw_size,
            "sha256": hashlib.sha256(source_payload).hexdigest(),
        },
        "region": region,
        "spans": spans,
    }
    record["source_tensor"]["binding_sha256"] = pack_gguf._source_tensor_binding(record)
    manifest = {
        "schema": pack_gguf.KPACK_BUNDLE_SCHEMA,
        "schema_version": pack_gguf.KPACK_BUNDLE_VERSION,
        "arrangement_version": routes.PLACED_ARTIFACT_VERSION_V2,
        "model": str(source),
        "source": pack_gguf._source_file_identity(source),
        "storage": pack_gguf._bundle_storage_identity(weights_path),
        "selection": {
            "layout_policy": "production-kpack-only",
            "packable_total": 1,
            "packed": 1,
            "skipped": 0,
        },
        "tensors": [record],
        "skipped": [],
    }
    (root / pack_gguf.KPACK_BUNDLE_MANIFEST).write_text(json.dumps(manifest))
    return root, manifest, artifact


def _rewrite(root, manifest):
    (root / pack_gguf.KPACK_BUNDLE_MANIFEST).write_text(json.dumps(manifest))


def _rehash_storage(root, manifest):
    path = root / pack_gguf.KPACK_BUNDLE_WEIGHTS
    manifest["storage"]["size_bytes"] = path.stat().st_size
    manifest["storage"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_kpack_bundle_loads_only_after_complete_schema_and_byte_validation(tmp_path):
    root, manifest, artifact = _bundle(tmp_path)
    loaded = pack_gguf.load_kpack_bundle(root, source=tmp_path / "model.gguf")
    assert loaded.manifest == manifest
    assert list(loaded.artifacts) == ["blk.0.attn_q.weight"]
    assert loaded.artifacts["blk.0.attn_q.weight"] == artifact
    assert set(entry.name for entry in root.iterdir()) == {
        pack_gguf.KPACK_BUNDLE_MANIFEST, pack_gguf.KPACK_BUNDLE_WEIGHTS}


def test_non_dry_run_packer_publishes_one_mixed_dense_grouped_blob(
        tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    specs = [
        ("blk.0.attn_q.weight", formats.QuantType.Q4_K, (512, 256)),
        ("blk.0.ffn_up_exps.weight", formats.QuantType.Q3_K, (512, 256, 2)),
    ]
    tensor_rows = []
    source_bytes = bytearray()
    for tensor_index, (_name, qtype, shape) in enumerate(specs):
        k, n = shape[:2]
        experts = shape[2] if len(shape) == 3 else 1
        block = formats.BLOCKS[qtype]
        rows = experts * n * (k // block.weights)
        data = ((np.arange(rows * block.block_bytes, dtype=np.uint32) + tensor_index * 31) & 0xff)
        data = data.astype(np.uint8).reshape(rows, block.block_bytes)
        tensor_rows.append((len(source_bytes), data))
        source_bytes.extend(data.tobytes())
    model.write_bytes(source_bytes)

    class FakeReader:
        def __init__(self, path):
            assert str(path) == str(model)
            self.endianess = _FakeGGUFEndian.LITTLE
            self.tensors = []
            for (name, qtype, shape), (data_offset, data) in zip(specs, tensor_rows):
                self.tensors.append(types.SimpleNamespace(
                    name=name, tensor_type=qtype, shape=shape, data=data,
                    data_offset=data_offset, n_bytes=data.nbytes))

    def prepare(_routes, _blocks, n, k, qtype, experts, route):
        artifact = _fake_artifact(formats.QuantType(qtype), route, n, k, experts)
        return pack_gguf._target_layout(qtype), artifact

    monkeypatch.setitem(
        sys.modules, "gguf",
        types.SimpleNamespace(GGUFReader=FakeReader, GGUFEndian=_FakeGGUFEndian))
    monkeypatch.setattr(pack_gguf, "_prepare_artifact", prepare)
    import quactlize
    monkeypatch.setattr(quactlize, "gguf_backend_for_qtype", lambda _qtype: "ppu:test")
    output = tmp_path / "model.kpack"
    assert pack_gguf.main([str(model), str(output)]) == 0

    loaded = pack_gguf.load_kpack_bundle(output, source=model)
    assert list(loaded.artifacts) == [specs[0][0], specs[1][0]]
    assert [record["route_class"] for record in loaded.manifest["tensors"]] == ["dense", "grouped"]
    assert [record["ggml_type"] for record in loaded.manifest["tensors"]] == [12, 11]
    assert loaded.manifest["tensors"][1]["region"]["offset_bytes"] == (
        loaded.manifest["tensors"][0]["region"]["size_bytes"])
    assert set(entry.name for entry in output.iterdir()) == {
        pack_gguf.KPACK_BUNDLE_MANIFEST, pack_gguf.KPACK_BUNDLE_WEIGHTS}


def test_packer_failure_never_publishes_a_partial_final_bundle(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    raw = np.zeros((512, formats.BLOCKS[formats.QuantType.Q4_K].block_bytes), dtype=np.uint8)
    model.write_bytes(raw.tobytes() + raw.tobytes())

    class FakeReader:
        def __init__(self, _path):
            self.endianess = _FakeGGUFEndian.LITTLE
            self.tensors = [
                types.SimpleNamespace(
                    name="blk.0.attn_q.weight", tensor_type=formats.QuantType.Q4_K,
                    shape=(512, 256), data=raw, data_offset=0, n_bytes=raw.nbytes),
                types.SimpleNamespace(
                    name="blk.1.attn_q.weight", tensor_type=formats.QuantType.Q4_K,
                    shape=(512, 256), data=raw, data_offset=raw.nbytes, n_bytes=raw.nbytes),
            ]

    calls = 0

    def prepare(_routes, _blocks, n, k, qtype, experts, route):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("planted second-tensor producer failure")
        return pack_gguf._target_layout(qtype), _fake_artifact(
            formats.QuantType(qtype), route, n, k, experts)

    monkeypatch.setitem(
        sys.modules, "gguf",
        types.SimpleNamespace(GGUFReader=FakeReader, GGUFEndian=_FakeGGUFEndian))
    monkeypatch.setattr(pack_gguf, "_prepare_artifact", prepare)
    import quactlize
    monkeypatch.setattr(quactlize, "gguf_backend_for_qtype", lambda _qtype: "ppu:test")
    output = tmp_path / "model.kpack"
    with pytest.raises(RuntimeError, match="planted second-tensor"):
        pack_gguf.main([str(model), str(output)])
    assert not output.exists()
    staging = output.with_name(output.name + f".partial.{os.getpid()}")
    assert staging.is_dir()
    assert (staging / pack_gguf.KPACK_BUNDLE_WEIGHTS).is_file()
    assert not (staging / pack_gguf.KPACK_BUNDLE_MANIFEST).exists()


def test_persistent_bundle_source_binding_is_content_based_and_fail_closed(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    source = tmp_path / "model.gguf"
    relocated = tmp_path / "relocated.gguf"
    relocated.write_bytes(source.read_bytes())
    pack_gguf.load_kpack_bundle(root, source=relocated)
    assert pack_gguf.validate_kpack_bundle_source(manifest, relocated) == manifest["source"]

    source.write_bytes(b"X" * manifest["source"]["size_bytes"])
    with pytest.raises(ValueError, match="K-pack bundle source mismatch"):
        pack_gguf.load_kpack_bundle(root, source=source)


def test_source_identity_is_exact_and_rejects_missing_or_nonregular_inputs(tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"known GGUF bytes")
    assert pack_gguf._source_file_identity(source) == {
        "format": "gguf",
        "size_bytes": 16,
        "sha256": "203ac6a39daa689f08fdd867c5ed59c8e5183340e2118e854ed0908655f4055e",
    }
    with pytest.raises(ValueError, match="cannot open source GGUF"):
        pack_gguf._source_file_identity(tmp_path / "missing.gguf")
    with pytest.raises(ValueError, match="must resolve to a regular file"):
        pack_gguf._source_file_identity(tmp_path)


@pytest.mark.parametrize("plant,match", [
    (lambda source: source.pop("format"), "contain exactly"),
    (lambda source: source.update(extra=1), "contain exactly"),
    (lambda source: source.update(format="raw"), "format must be gguf"),
    (lambda source: source.update(size_bytes=True), "must be a nonnegative integer"),
    (lambda source: source.update(size_bytes=-1), "must be a nonnegative integer"),
    (lambda source: source.update(sha256="A" * 64), "exact lowercase SHA-256"),
])
def test_source_authority_schema_plants_fail_closed(tmp_path, plant, match):
    root, manifest, _ = _bundle(tmp_path)
    plant(manifest["source"])
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match=match):
        pack_gguf.load_kpack_bundle(root)


@pytest.mark.parametrize("field,value", [("size_bytes", 1), ("sha256", "0" * 64)])
def test_source_aware_load_rejects_validly_shaped_manifest_authority_plants(
        tmp_path, field, value):
    root, manifest, _ = _bundle(tmp_path)
    manifest["source"][field] = value
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="K-pack bundle source mismatch"):
        pack_gguf.load_kpack_bundle(root, source=tmp_path / "model.gguf")


@pytest.mark.parametrize("qtype", list(_RECIPES))
@pytest.mark.parametrize("route", ["dense", "grouped"])
def test_all_shipping_qtypes_and_routes_use_the_exact_producer_tensor_abi(
        tmp_path, qtype, route):
    root, manifest, artifact = _bundle(tmp_path, qtype, route)
    loaded = pack_gguf.load_kpack_bundle(root)
    record = manifest["tensors"][0]
    assert [list(t.shape) for t in artifact] == [
        record["spans"]["low"]["shape"],
        record["spans"]["high"]["shape"],
        record["spans"]["units"]["shape"],
    ]
    assert record["region"]["offset_bytes"] % pack_gguf.KPACK_BUNDLE_ALIGNMENT == 0
    assert record["region"]["size_bytes"] % pack_gguf.KPACK_BUNDLE_ALIGNMENT == 0
    assert record["region"]["size_bytes"] == (
        (3 if route == "grouped" else 1) * 256 * 2 * formats.BLOCKS[qtype].block_bytes)
    assert loaded.artifacts[record["name"]] == artifact


@pytest.mark.parametrize("plant,match", [
    (lambda m: m.update(schema_version=999), "unsupported K-pack bundle schema"),
    (lambda m: m["source"].update(format="raw"), "format must be gguf"),
    (lambda m: m["storage"].update(file="arrays.npy"), "storage.file must be weights.bin"),
    (lambda m: m["storage"].update(alignment_bytes=64), "alignment must be 128"),
    (lambda m: m.update(arrangement_version=1), "arrangement version 2"),
    (lambda m: m["selection"].update(layout_policy="x"), "production-kpack-only"),
    (lambda m: m["tensors"][0].update(arrangement_version=1), "arrangement_version=2"),
    (lambda m: m["tensors"][0].update(layout_name="q4-n16k64-direct"), "canonical q4-kpack4"),
    (lambda m: m["tensors"][0]["arrangement"].update(layout=3), "invalid production arrangement"),
    (lambda m: m["tensors"][0].update(name="blk.9.attn_q.weight"), "source tensor binding"),
    (lambda m: m["tensors"][0]["source_tensor"].update(index=True), "must be a nonnegative integer"),
    (lambda m: m["tensors"][0]["source_tensor"].update(binding_sha256="0" * 64), "source tensor binding"),
    (lambda m: m["tensors"][0]["region"].update(offset_bytes=128), "canonical manifest order"),
    (lambda m: m["tensors"][0]["region"].update(size_bytes=129), "128-byte aligned"),
    (lambda m: m["tensors"][0]["spans"]["low"].update(offset_bytes=128), "offset is not canonical"),
    (lambda m: m["tensors"][0]["spans"]["low"].update(shape=[1]), "low shape must be canonical"),
    (lambda m: m["tensors"][0]["spans"]["low"].update(size_bytes=1), "low size must be canonical"),
])
def test_kpack_bundle_manifest_plants_fail_closed(tmp_path, plant, match):
    root, manifest, _ = _bundle(tmp_path)
    planted = copy.deepcopy(manifest)
    plant(planted)
    _rewrite(root, planted)
    with pytest.raises(ValueError, match=match):
        pack_gguf.load_kpack_bundle(root)


@pytest.mark.parametrize("version,match", [
    (1, "v1 is source-unbound; repack"),
    (2, "v2 uses the retired NPY carrier; repack"),
])
def test_retired_bundle_versions_require_repack(tmp_path, version, match):
    root, manifest, _ = _bundle(tmp_path)
    manifest["schema_version"] = version
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match=match):
        pack_gguf.load_kpack_bundle(root)


def test_manifest_duplicate_json_keys_are_rejected(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    text = json.dumps(manifest)
    text = text.replace('"schema_version": 3', '"schema_version": 3, "schema_version": 3', 1)
    (root / pack_gguf.KPACK_BUNDLE_MANIFEST).write_text(text)
    with pytest.raises(ValueError, match="duplicate JSON object key 'schema_version'"):
        pack_gguf.load_kpack_bundle(root)


def test_source_tensor_range_hash_is_checked_independently_of_whole_source_hash(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    source = tmp_path / "model.gguf"
    planted = bytearray(source.read_bytes())
    planted[0] ^= 1
    source.write_bytes(planted)
    manifest["source"] = pack_gguf._source_file_identity(source)
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="K-pack source tensor mismatch"):
        pack_gguf.load_kpack_bundle(root, source=source)


@pytest.mark.parametrize("plant,match", [
    ("duplicate-skipped", "duplicate tensor name in skipped"),
    ("packed-and-skipped", "tensors and skipped inventory overlap"),
])
def test_skipped_inventory_names_are_unambiguous(tmp_path, plant, match):
    root, manifest, _ = _bundle(tmp_path)
    skipped = {"name": "unsupported.weight", "type_name": "F32", "reason": "not K-quant"}
    manifest["skipped"] = [skipped, dict(skipped)] if plant == "duplicate-skipped" else [
        dict(skipped, name=manifest["tensors"][0]["name"])]
    manifest["selection"]["skipped"] = len(manifest["skipped"])
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match=match):
        pack_gguf.load_kpack_bundle(root)


def test_blob_inventory_symlinks_and_byte_corruption_fail_closed(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    extra = root / "unlisted.txt"
    extra.write_text("not part of the bundle")
    with pytest.raises(ValueError, match="extra=.*unlisted"):
        pack_gguf.load_kpack_bundle(root)
    extra.unlink()

    weights = root / pack_gguf.KPACK_BUNDLE_WEIGHTS
    original = weights.read_bytes()
    planted = bytearray(original)
    planted[manifest["tensors"][0]["spans"]["low"]["offset_bytes"]] ^= 1
    weights.write_bytes(planted)
    _rehash_storage(root, manifest)
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="low span checksum mismatch"):
        pack_gguf.load_kpack_bundle(root)

    weights.write_bytes(original)
    _rehash_storage(root, manifest)
    _rewrite(root, manifest)
    target = root / "real-weights.bin"
    weights.rename(target)
    weights.symlink_to(target.name)
    with pytest.raises(ValueError, match="extra=.*real-weights|must be a real regular file"):
        pack_gguf.load_kpack_bundle(root)


@pytest.mark.parametrize("name", [
    pack_gguf.KPACK_BUNDLE_MANIFEST,
    pack_gguf.KPACK_BUNDLE_WEIGHTS,
])
def test_bundle_regular_file_checks_reject_fifos_without_blocking(tmp_path, name):
    root, _manifest, _ = _bundle(tmp_path)
    path = root / name
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="must be a real regular file"):
        pack_gguf.load_kpack_bundle(root)


def test_manifest_symlink_is_rejected_by_the_consuming_open(tmp_path):
    root, _manifest, _ = _bundle(tmp_path)
    manifest = root / pack_gguf.KPACK_BUNDLE_MANIFEST
    external = tmp_path / "external-manifest.json"
    manifest.rename(external)
    manifest.symlink_to(external)
    with pytest.raises(ValueError, match="must be one readable regular file"):
        pack_gguf.load_kpack_bundle(root)


def test_storage_checksum_and_exact_region_coverage_are_independent_guards(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    weights = root / pack_gguf.KPACK_BUNDLE_WEIGHTS
    planted = bytearray(weights.read_bytes())
    planted[0] ^= 1
    weights.write_bytes(planted)
    low_size = manifest["tensors"][0]["spans"]["low"]["size_bytes"]
    manifest["tensors"][0]["spans"]["low"]["sha256"] = hashlib.sha256(
        planted[:low_size]).hexdigest()
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="storage checksum mismatch"):
        pack_gguf.load_kpack_bundle(root)

    _rehash_storage(root, manifest)
    manifest["storage"]["size_bytes"] += pack_gguf.KPACK_BUNDLE_ALIGNMENT
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="do not cover storage.size_bytes"):
        pack_gguf.load_kpack_bundle(root)


@pytest.mark.parametrize("array,new_shape", [
    ("low", [1, 1, 256 * 256]),
    ("units", [1, 2, 256, 16]),
])
def test_same_numel_noncanonical_dense_shapes_are_rejected(tmp_path, array, new_shape):
    root, manifest, _ = _bundle(tmp_path)
    record = manifest["tensors"][0]
    old_numel = 1
    for extent in record["spans"][array]["shape"]:
        old_numel *= extent
    new_numel = 1
    for extent in new_shape:
        new_numel *= extent
    assert new_numel == old_numel
    record["spans"][array]["shape"] = new_shape
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match=f"{array} shape must be canonical"):
        pack_gguf.load_kpack_bundle(root)


def test_partial_limit_schema_and_cli_option_are_rejected(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    manifest["selection"].update(limit=1, held_back_by_limit=1)
    manifest["held_back_by_limit"] = [
        {"name": "blk.1.attn_q.weight", "type_name": "Q4_K", "reason": "held back"}]
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="manifest must contain exactly|selection must contain exactly"):
        pack_gguf.load_kpack_bundle(root)

    with pytest.raises(SystemExit):
        pack_gguf.main(["model.gguf", str(tmp_path / "out"), "--limit", "1"])


def test_bundle_publication_refuses_dangling_final_and_staging_symlinks(tmp_path):
    final = tmp_path / "bundle-out"
    final.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pack_gguf._create_bundle_staging_root(final)
    assert final.is_symlink()

    final.unlink()
    staging = final.with_name(final.name + f".partial.{os.getpid()}")
    staging.symlink_to(tmp_path / "missing-staging-target")
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        pack_gguf._create_bundle_staging_root(final)
    assert staging.is_symlink()


def test_bundle_publication_never_replaces_a_concurrently_created_target(tmp_path):
    final = tmp_path / "bundle-out"
    staging = pack_gguf._create_bundle_staging_root(final)
    (staging / "payload").write_bytes(b"complete")
    final.mkdir()
    (final / "owner").write_text("concurrent publisher")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pack_gguf._publish_bundle_noreplace(staging, final)
    assert (final / "owner").read_text() == "concurrent publisher"
    assert (staging / "payload").read_bytes() == b"complete"


def test_bundle_publication_renames_one_complete_staging_directory(tmp_path):
    final = tmp_path / "bundle-out"
    staging = pack_gguf._create_bundle_staging_root(final)
    (staging / "payload").write_bytes(b"complete")
    pack_gguf._publish_bundle_noreplace(staging, final)
    assert not staging.exists()
    assert (final / "payload").read_bytes() == b"complete"


def test_development_restore_compatibility_does_not_weaken_product_bundle(tmp_path):
    root, manifest, artifact = _bundle(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    directory = pack_gguf._tensor_dir_name(0, "blk.0.attn_q.weight")
    tensor_dir = legacy / directory
    tensor_dir.mkdir()
    direct = formats.q4_n16k64_direct_arrangement()
    pack_gguf._write(tensor_dir, *artifact)
    compatibility = {
        "name": "blk.0.attn_q.weight",
        "dir": directory,
        "ggml_type": int(formats.QuantType.Q4_K),
        "arrangement_version": routes.PLACED_ARTIFACT_VERSION_V2,
        "arrangement": direct._asdict(),
    }
    restored = pack_gguf.restore_artifact(legacy, compatibility)
    assert restored.arrangement == direct

    manifest["tensors"][0]["arrangement"] = direct._asdict()
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="arrangement is not canonical|invalid production arrangement"):
        pack_gguf.load_kpack_bundle(root)

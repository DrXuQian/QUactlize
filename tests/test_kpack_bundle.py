"""Strict contracts for the installable production K-pack directory bundle."""

from __future__ import annotations

import copy
import json
import os

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


def _bundle(tmp_path, qtype=formats.QuantType.Q4_K, route="dense"):
    root = tmp_path / "bundle"
    root.mkdir()
    source = tmp_path / "model.gguf"
    source.write_bytes(b"GGUF source authority fixture\n")
    grouped = route == "grouped"
    name = "blk.0.ffn_up_exps.weight" if grouped else "blk.0.attn_q.weight"
    directory = pack_gguf._tensor_dir_name(0, name)
    tensor_dir = root / directory
    tensor_dir.mkdir()
    n, k = 256, 512
    experts = 3 if grouped else 1
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
    pack_gguf._write(tensor_dir, *artifact)
    record = {
        "name": name,
        "dir": directory,
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
        "shapes": {"low": list(low.shape), "high": list(high.shape), "units": list(units.shape)},
        "sha256": pack_gguf._bundle_file_hashes(tensor_dir),
    }
    manifest = {
        "schema": pack_gguf.KPACK_BUNDLE_SCHEMA,
        "schema_version": pack_gguf.KPACK_BUNDLE_VERSION,
        "arrangement_version": routes.PLACED_ARTIFACT_VERSION_V2,
        "model": str(source),
        "source": pack_gguf._source_file_identity(source),
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


def test_kpack_bundle_loads_only_after_complete_schema_and_byte_validation(tmp_path):
    root, manifest, artifact = _bundle(tmp_path)
    loaded = pack_gguf.load_kpack_bundle(root, source=tmp_path / "model.gguf")
    assert loaded.manifest == manifest
    assert list(loaded.artifacts) == ["blk.0.attn_q.weight"]
    assert loaded.artifacts["blk.0.attn_q.weight"] == artifact


def test_persistent_bundle_source_binding_is_content_based_and_fail_closed(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    source = tmp_path / "model.gguf"
    relocated = tmp_path / "relocated.gguf"
    relocated.write_bytes(source.read_bytes())

    # Moving an identical source is safe; the recorded path is diagnostic only.
    pack_gguf.load_kpack_bundle(root, source=relocated)
    assert pack_gguf.validate_kpack_bundle_source(manifest, relocated) == manifest["source"]

    # Replacing a model at the same path must never reuse the old K-pack bytes.
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
        record["shapes"]["low"], record["shapes"]["high"], record["shapes"]["units"]]
    assert loaded.artifacts[record["name"]] == artifact


def test_tensor_directories_are_collision_safe_and_bound_to_order_and_name():
    assert pack_gguf._tensor_dir_name(0, "a/b") != pack_gguf._tensor_dir_name(0, "a.b")
    assert pack_gguf._tensor_dir_name(0, "same") != pack_gguf._tensor_dir_name(1, "same")
    assert pack_gguf._tensor_dir_name(7, "x").startswith("tensor-000007-")


@pytest.mark.parametrize("plant,match", [
    (lambda m: m.update(schema_version=999), "unsupported K-pack bundle schema"),
    (lambda m: m["source"].update(format="raw"), "format must be gguf"),
    (lambda m: m["source"].update(size_bytes=0), "source.size_bytes must be positive"),
    (lambda m: m["source"].update(sha256="ABC"), "exact lowercase SHA-256"),
    (lambda m: m.update(arrangement_version=1), "arrangement version 2"),
    (lambda m: m["selection"].update(layout_policy="x"), "production-kpack-only"),
    (lambda m: m["tensors"][0].update(dir="weight"), "collision-safe"),
    (lambda m: m["tensors"][0].update(arrangement_version=1), "arrangement_version=2"),
    (lambda m: m["tensors"][0].update(layout_name="q4-n16k64-direct"), "canonical q4-kpack4"),
    (lambda m: m["tensors"][0]["arrangement"].update(layout=3), "invalid production arrangement"),
    (lambda m: m["tensors"][0]["shapes"].update(low=[1]), "low shape must be canonical"),
])
def test_kpack_bundle_manifest_plants_fail_closed(tmp_path, plant, match):
    root, manifest, _ = _bundle(tmp_path)
    planted = copy.deepcopy(manifest)
    plant(planted)
    _rewrite(root, planted)
    with pytest.raises(ValueError, match=match):
        pack_gguf.load_kpack_bundle(root)


def test_source_unbound_v1_bundle_requires_repack(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    manifest["schema_version"] = 1
    manifest.pop("source")
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="v1 is source-unbound; repack"):
        pack_gguf.load_kpack_bundle(root)


def test_kpack_bundle_rejects_unlisted_files_and_non_uint8_arrays(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    (root / "unlisted.txt").write_text("not part of the bundle")
    with pytest.raises(ValueError, match="extra=.*unlisted"):
        pack_gguf.load_kpack_bundle(root)
    (root / "unlisted.txt").unlink()

    record = manifest["tensors"][0]
    np.save(root / record["dir"] / "low.npy", np.zeros(record["shapes"]["low"], dtype=np.float16))
    with pytest.raises(ValueError, match="checksum mismatch"):
        pack_gguf.load_kpack_bundle(root)
    record["sha256"] = pack_gguf._bundle_file_hashes(root / record["dir"])
    _rewrite(root, manifest)
    with pytest.raises(ValueError, match="low dtype must be uint8"):
        pack_gguf.load_kpack_bundle(root)


@pytest.mark.parametrize("array,new_shape", [
    ("low", [1, 1, 256 * 256]),
    ("units", [1, 2, 256, 16]),
])
def test_same_numel_noncanonical_dense_shapes_are_rejected(tmp_path, array, new_shape):
    root, manifest, _ = _bundle(tmp_path)
    record = manifest["tensors"][0]
    assert np.prod(new_shape) == np.prod(record["shapes"][array])
    path = root / record["dir"] / f"{array}.npy"
    value = np.load(path, allow_pickle=False).reshape(new_shape)
    np.save(path, value)
    record["shapes"][array] = new_shape
    record["sha256"] = pack_gguf._bundle_file_hashes(root / record["dir"])
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


def test_development_restore_compatibility_does_not_weaken_product_bundle(tmp_path):
    root, manifest, _ = _bundle(tmp_path)
    record = manifest["tensors"][0]
    direct = formats.q4_n16k64_direct_arrangement()
    compatibility = dict(record, arrangement=direct._asdict())
    restored = pack_gguf.restore_artifact(root, compatibility)
    assert restored.arrangement == direct

    planted = copy.deepcopy(manifest)
    planted["tensors"][0]["arrangement"] = direct._asdict()
    _rewrite(root, planted)
    with pytest.raises(ValueError, match="arrangement is not canonical|invalid production arrangement"):
        pack_gguf.load_kpack_bundle(root)

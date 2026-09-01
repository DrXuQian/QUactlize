import hashlib
import json
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parent.parent
VERIFY = ROOT / "tools" / "verify_prebuilt_ppu_bundle.py"
SOURCE_SHA = "a" * 40
ACTLIZE_SHA = "b" * 40


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir()
    packed = bins / "packed-q4"
    packed.write_bytes(b"packed PPU image\n")
    default = bins / "default"
    default.write_bytes(b"default PPU image\n")
    document = {
        "schema_version": 1,
        "source_sha": SOURCE_SHA,
        "submodules": {"third_party/actlize": ACTLIZE_SHA},
        "sdk": "PPU SDK 2.1.1 a5c56e",
        "compiler": "hgcc 2.1.1",
        "arch": "ppu0010",
        "artifacts": [
            {
                "role": "fully-quantized",
                "qtype": 12,
                "path": "bin/packed-q4",
                "size": packed.stat().st_size,
                "sha256": _sha256(packed),
                "target": "quactlize_ppu",
                "ppu_defs": ["PPU_PACKED_SCALE=1", "PPU_PACKED_FORMAT=0"],
            },
            {
                "role": "scale-first",
                "qtype": None,
                "path": "bin/default",
                "size": default.stat().st_size,
                "sha256": _sha256(default),
                "target": "quactlize_ppu",
                "ppu_defs": [],
            },
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return manifest, document, packed, default


def _run(manifest, *arguments):
    return subprocess.run(
        [sys.executable, str(VERIFY), str(manifest), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def _packed_args():
    return ["--role", "fully-quantized", "--qtype", "12",
            "--target", "quactlize_ppu"]


def test_success_verifies_complete_bundle_and_prints_only_selected_path(tmp_path):
    manifest, _, packed, _ = _write_bundle(tmp_path)
    result = _run(
        manifest,
        *_packed_args(),
        "--expect-source", SOURCE_SHA,
        "--expect-submodule", f"third_party/actlize={ACTLIZE_SHA}",
        "--expect-sdk", "PPU SDK 2.1.1 a5c56e",
        "--expect-compiler", "hgcc 2.1.1",
        "--expect-arch", "ppu0010",
        "--expect-ppu-def", "PPU_PACKED_SCALE=1",
        "--expect-ppu-def", "PPU_PACKED_FORMAT=0",
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == f"{packed.absolute()}\n"


def test_null_qtype_selects_default_artifact(tmp_path):
    manifest, _, _, default = _write_bundle(tmp_path)
    result = _run(
        manifest, "--role", "scale-first", "--qtype", "null",
        "--target", "quactlize_ppu")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{default.absolute()}\n"


@pytest.mark.parametrize("bad_path", ["../escape", "/tmp/escape", "bin//packed-q4",
                                      "bin/./packed-q4", "bin\\packed-q4"])
def test_rejects_noncanonical_or_escaping_paths(tmp_path, bad_path):
    manifest, document, _, _ = _write_bundle(tmp_path)
    document["artifacts"][0]["path"] = bad_path
    manifest.write_text(json.dumps(document), encoding="utf-8")
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "path" in result.stderr
    assert result.stdout == ""


def test_rejects_artifact_symlink_and_symlinked_directory(tmp_path):
    manifest, document, packed, _ = _write_bundle(tmp_path)
    real = tmp_path / "real-payload"
    real.write_bytes(packed.read_bytes())
    packed.unlink()
    packed.symlink_to(real)
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "symbolic link" in result.stderr

    packed.unlink()
    packed.write_bytes(real.read_bytes())
    real_dir = tmp_path / "real-bin"
    real_dir.mkdir()
    moved = real_dir / "packed-q4"
    packed.replace(moved)
    (tmp_path / "bin" / "default").unlink()
    (tmp_path / "bin").rmdir()
    (tmp_path / "bin").symlink_to(real_dir, target_is_directory=True)
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "symbolic link" in result.stderr


def test_rejects_missing_unselected_payload(tmp_path):
    manifest, _, _, default = _write_bundle(tmp_path)
    default.unlink()
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "missing" in result.stderr


def test_rejects_size_mismatch(tmp_path):
    manifest, document, _, _ = _write_bundle(tmp_path)
    document["artifacts"][0]["size"] += 1
    manifest.write_text(json.dumps(document), encoding="utf-8")
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "size differs" in result.stderr


def test_rejects_zero_size_in_manifest(tmp_path):
    manifest, document, _, _ = _write_bundle(tmp_path)
    document["artifacts"][0]["size"] = 0
    manifest.write_text(json.dumps(document), encoding="utf-8")
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_rejects_hash_mismatch(tmp_path):
    manifest, _, packed, _ = _write_bundle(tmp_path)
    tampered = bytearray(packed.read_bytes())
    tampered[0] ^= 1
    packed.write_bytes(tampered)
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "SHA-256 differs" in result.stderr


@pytest.mark.parametrize(
    "arguments,diagnostic",
    [
        (["--expect-source", "c" * 40], "source_sha differs"),
        (["--expect-submodule", f"third_party/actlize={'c' * 40}"],
         "submodule third_party/actlize differs"),
        (["--expect-submodule", f"third_party/missing={'c' * 40}"],
         "submodule third_party/missing differs"),
        (["--expect-sdk", "different SDK"], "sdk differs"),
        (["--expect-compiler", "different compiler"], "compiler differs"),
        (["--expect-arch", "ppu9999"], "arch differs"),
    ],
)
def test_rejects_mismatched_expected_identity(tmp_path, arguments, diagnostic):
    manifest, _, _, _ = _write_bundle(tmp_path)
    result = _run(manifest, *_packed_args(), *arguments)
    assert result.returncode == 2
    assert diagnostic in result.stderr


def test_expected_ppu_definitions_are_exact_and_ordered(tmp_path):
    manifest, _, _, _ = _write_bundle(tmp_path)
    reversed_definitions = _run(
        manifest,
        *_packed_args(),
        "--expect-ppu-def", "PPU_PACKED_FORMAT=0",
        "--expect-ppu-def", "PPU_PACKED_SCALE=1",
    )
    assert reversed_definitions.returncode == 2
    assert "ppu_defs differ" in reversed_definitions.stderr

    missing_definition = _run(
        manifest,
        *_packed_args(),
        "--expect-ppu-def", "PPU_PACKED_SCALE=1",
    )
    assert missing_definition.returncode == 2
    assert "ppu_defs differ" in missing_definition.stderr


def test_rejects_missing_or_ambiguous_selection(tmp_path):
    manifest, document, _, _ = _write_bundle(tmp_path)
    missing = _run(
        manifest, "--role", "missing", "--qtype", "12",
        "--target", "quactlize_ppu")
    assert missing.returncode == 2
    assert "is absent" in missing.stderr

    duplicate = dict(document["artifacts"][0])
    duplicate["path"] = "bin/packed-q4-copy"
    copy = tmp_path / duplicate["path"]
    copy.write_bytes((tmp_path / document["artifacts"][0]["path"]).read_bytes())
    document["artifacts"].append(duplicate)
    manifest.write_text(json.dumps(document), encoding="utf-8")
    ambiguous = _run(manifest, *_packed_args())
    assert ambiguous.returncode == 2
    assert "identity" in ambiguous.stderr


def test_rejects_symlinked_manifest(tmp_path):
    manifest, _, _, _ = _write_bundle(tmp_path)
    link = tmp_path / "manifest-link.json"
    link.symlink_to(manifest)
    result = _run(link, *_packed_args())
    assert result.returncode == 2
    assert "manifest must not be a symbolic link" in result.stderr


def test_rejects_duplicate_json_key(tmp_path):
    manifest, _, _, _ = _write_bundle(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    text = text.replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    manifest.write_text(text, encoding="utf-8")
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "duplicate JSON key" in result.stderr


def test_rejects_unknown_manifest_field(tmp_path):
    manifest, document, _, _ = _write_bundle(tmp_path)
    document["unexpected"] = True
    manifest.write_text(json.dumps(document), encoding="utf-8")
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "wrong top-level fields" in result.stderr


def test_rejects_duplicate_artifact_path(tmp_path):
    manifest, document, _, _ = _write_bundle(tmp_path)
    document["artifacts"][1]["path"] = document["artifacts"][0]["path"]
    manifest.write_text(json.dumps(document), encoding="utf-8")
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "artifact path" in result.stderr


def test_rejects_duplicate_ppu_definition(tmp_path):
    manifest, document, _, _ = _write_bundle(tmp_path)
    document["artifacts"][0]["ppu_defs"].append("PPU_PACKED_SCALE=1")
    manifest.write_text(json.dumps(document), encoding="utf-8")
    result = _run(manifest, *_packed_args())
    assert result.returncode == 2
    assert "ppu_defs contains a duplicate" in result.stderr

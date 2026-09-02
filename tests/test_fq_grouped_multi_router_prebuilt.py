"""Fail-closed tests for the grouped multi-router prebuilt handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools/fq_grouped_multi_router_manifest.py"
SPEC = importlib.util.spec_from_file_location("fq_grouped_manifest", MODULE_PATH)
manifest_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(manifest_module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    source.mkdir()
    bundle.mkdir()
    for relative in manifest_module.SOURCE_PATHS:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n", encoding="utf-8")

    binaries = {}
    for qtype in (10, 11, 12, 13, 14):
        directory = bundle / "bin" / f"q{qtype}"
        directory.mkdir(parents=True)
        binary = directory / "test"
        library = directory / "lib.so"
        binary.write_text(f"binary:{qtype}\n", encoding="utf-8")
        binary.chmod(0o755)
        library.write_text(f"library:{qtype}\n", encoding="utf-8")
        binaries[str(qtype)] = {
            "path": binary.relative_to(bundle).as_posix(),
            "sha256": _sha(binary),
            "library_path": library.relative_to(bundle).as_posix(),
            "library_sha256": _sha(library),
        }

    commit = "a" * 40
    submodule_commit = "b" * 40

    def fake_check_output(args, text=True):
        if args[-2:] == ["rev-parse", "HEAD"]:
            return commit + "\n"
        if args[-2:] == ["status", "--recursive"]:
            return f" {submodule_commit} third_party/test (heads/main)\n"
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    document = {
        "schema": manifest_module.SCHEMA,
        "source_sha": commit,
        "source_files": {
            relative: _sha(source / relative)
            for relative in manifest_module.SOURCE_PATHS
        },
        "submodules": {"third_party/test": submodule_commit},
        "sdk": {
            "release_version": "2.1.1-test",
            "release_sha256": "0" * 64,
            "compiler_sha256": "1" * 64,
            "inspector_sha256": "2" * 64,
        },
        "build": {
            "target": "test_fq_grouped_multi_router_perf",
            "arch": "ppu0010",
            "qtypes": [10, 11, 12, 13, 14],
        },
        "binaries": binaries,
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    return source, bundle, manifest_path


def test_box_accepts_same_release_with_different_receipt_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, bundle, manifest_path = _fixture(tmp_path, monkeypatch)
    release = tmp_path / "release.yaml"
    release.write_text("vendor: box\nversion: 2.1.1-test\n", encoding="utf-8")
    manifest_module.verify(bundle, manifest_path, source, release)


def test_box_rejects_different_release_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, bundle, manifest_path = _fixture(tmp_path, monkeypatch)
    release = tmp_path / "release.yaml"
    release.write_text("version: incompatible\n", encoding="utf-8")
    with pytest.raises(manifest_module.ManifestError, match="release version differs"):
        manifest_module.verify(bundle, manifest_path, source, release)


def test_runner_requires_measured_single_device_and_never_builds():
    runner = (
        ROOT / "tools/run_fq_grouped_multi_router_prebuilt_box.sh"
    ).read_text(encoding="utf-8")
    assert 'probe["device_count"] == 1' in runner
    assert 'probe["status"] in ("measured", "properties-unavailable")' in runner
    assert "build.sh" not in runner
    assert "PPU_BUILD_DIR" not in runner
    subprocess.run(
        ["bash", "-n", str(ROOT / "tools/run_fq_grouped_multi_router_prebuilt_box.sh")],
        check=True,
    )

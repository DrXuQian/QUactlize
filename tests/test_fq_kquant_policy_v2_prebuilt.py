"""Fail-closed tests for the execute-only Q12 policy-v2 bundle."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools/fq_kquant_policy_v2_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("fq_policy_prebuilt", MODULE_PATH)
prebuilt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prebuilt)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"; bundle = tmp_path / "bundle"; sdk = tmp_path / "sdk"
    source.mkdir(); bundle.mkdir(); (sdk / "bin").mkdir(parents=True)
    for name in prebuilt.BUILD_INPUTS:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"authority:{name}\n", encoding="utf-8")
    release = sdk / "release.yaml"; release.write_text("version: test\n")
    compiler = sdk / "bin/hgcc"; compiler.write_text("compiler\n"); compiler.chmod(0o755)
    inspector = sdk / "bin/hgobjdump"; inspector.write_text("inspector\n"); inspector.chmod(0o755)
    binary = bundle / "test_fq_kquant_layout_perf"; binary.write_text("binary\n"); binary.chmod(0o755)
    library = bundle / "libquactlize_ppu.so"; library.write_text("library\n")
    commit = "a" * 40; subs = [{"path": "third_party/test", "commit": "b" * 40}]
    original_run = prebuilt.run
    def fake_run(*args, cwd=prebuilt.ROOT):
        if args[:3] == ("git", "rev-parse", "HEAD"): return commit
        if args[0] == str(compiler) and args[1] == "--version": return "hgcc Release version test"
        return original_run(*args, cwd=cwd)
    monkeypatch.setattr(prebuilt, "run", fake_run)
    monkeypatch.setattr(prebuilt, "submodules", lambda root: subs)
    record = lambda path, relative: {"path": relative, "size": path.stat().st_size,
                                     "sha256": _sha(path)}
    manifest = {
        "schema": prebuilt.SCHEMA,
        "source": {"commit": commit, "submodules": subs,
                   "build_inputs": {name: _sha(source / name)
                                    for name in prebuilt.BUILD_INPUTS}},
        "sdk": {"release": dict(record(release, "release.yaml"), version="test"),
                "compiler": dict(record(compiler, "bin/hgcc"),
                                 version="hgcc Release version test"),
                "inspector": record(inspector, "bin/hgobjdump")},
        "build": {"target": "test_fq_kquant_layout_perf", "qtype": 12,
                  "arch": "ppu0010", "definitions": prebuilt.DEFINITIONS,
                  "build_log_sha256": "c" * 64, "cmake_log_sha256": "d" * 64,
                  "build_make_sha256": "e" * 64},
        "artifacts": {"binary": record(binary, binary.name),
                      "library": record(library, library.name)},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return source, bundle, sdk, manifest


def test_manifest_verifier_accepts_exact_source_sdk_and_artifacts(tmp_path, monkeypatch):
    source, bundle, sdk, manifest = _fixture(tmp_path, monkeypatch)
    assert prebuilt.verify(bundle, source, sdk) == manifest


def test_execution_sdk_may_change_tool_bytes_with_the_same_release(
        tmp_path, monkeypatch):
    source, bundle, sdk, manifest = _fixture(tmp_path, monkeypatch)
    compiler = sdk / "bin/hgcc"
    inspector = sdk / "bin/hgobjdump"
    compiler.write_text("different execution compiler\n")
    inspector.write_text("different execution inspector\n")
    assert prebuilt.verify(
        bundle, source, sdk, execution_sdk_compatible=True) == manifest
    with pytest.raises(prebuilt.ManifestError):
        prebuilt.verify(bundle, source, sdk)

    compiler.unlink()
    inspector.unlink()
    assert prebuilt.verify(
        bundle, source, sdk, execution_sdk_compatible=True) == manifest


def test_execution_sdk_still_requires_the_build_release(tmp_path, monkeypatch):
    source, bundle, sdk, _ = _fixture(tmp_path, monkeypatch)
    (sdk / "release.yaml").write_text("version: incompatible\n")
    with pytest.raises(prebuilt.ManifestError, match="release differs"):
        prebuilt.verify(bundle, source, sdk, execution_sdk_compatible=True)


@pytest.mark.parametrize("plant", [
    "source", "submodule", "input", "sdk", "target", "binary-sha",
    "library-size", "extra-artifact", "extra-top-level",
])
def test_manifest_authority_plants_fail_closed(tmp_path, monkeypatch, plant):
    source, bundle, sdk, manifest = _fixture(tmp_path, monkeypatch)
    broken = copy.deepcopy(manifest)
    if plant == "source": broken["source"]["commit"] = "f" * 40
    elif plant == "submodule": broken["source"]["submodules"][0]["commit"] = "f" * 40
    elif plant == "input": broken["source"]["build_inputs"][prebuilt.BUILD_INPUTS[0]] = "f" * 64
    elif plant == "sdk": broken["sdk"]["inspector"]["sha256"] = "f" * 64
    elif plant == "target": broken["build"]["qtype"] = 11
    elif plant == "binary-sha": broken["artifacts"]["binary"]["sha256"] = "f" * 64
    elif plant == "library-size": broken["artifacts"]["library"]["size"] += 1
    elif plant == "extra-artifact": broken["artifacts"]["other"] = {}
    else: broken["unbound"] = True
    (bundle / "manifest.json").write_text(json.dumps(broken))
    with pytest.raises(prebuilt.ManifestError):
        prebuilt.verify(bundle, source, sdk)


def test_manifest_and_artifact_symlinks_fail_closed(tmp_path, monkeypatch):
    source, bundle, sdk, _ = _fixture(tmp_path, monkeypatch)
    manifest = bundle / "manifest.json"; outside = tmp_path / "manifest.json"
    manifest.rename(outside); manifest.symlink_to(outside)
    with pytest.raises(prebuilt.ManifestError): prebuilt.verify(bundle, source, sdk)


def test_box_runner_is_execute_only_and_builder_has_exact_q12_identity():
    runner = (ROOT / "tools/run_fq_kquant_policy_v2_box.sh").read_text()
    builder = (ROOT / "tools/build_fq_kquant_policy_v2_prebuilt.sh").read_text()
    assert "build.sh" not in runner and "PPU_BUILD_DIR" not in runner and "TARGET=" not in runner
    assert "probe_box_identity.py" in runner and "CUDA_VISIBLE_DEVICES" in runner
    assert "probe['device_count']==1" in runner
    assert "--execution-sdk-compatible" in runner
    assert "TARGET=test_fq_kquant_layout_perf" in builder
    assert "FQ_KQUANT_PERF_QTYPE=12" in builder
    assert "PPU_PACKED_FORMAT=0" in builder and "QUACTLIZE_DENSE_ONLY=12" in builder
    assert "RESUME" in builder and "PPU_BUILD_RESUME" in builder
    assert "write-build-authority" in builder
    assert "quactlize/csrc/fq_kquant_layout_perf.cmake.in" in prebuilt.BUILD_INPUTS
    assert "tools/probe_box_identity.py" in prebuilt.BUILD_INPUTS
    subprocess.run(["bash", "-n", str(ROOT / "tools/build_fq_kquant_policy_v2_prebuilt.sh"),
                    str(ROOT / "tools/run_fq_kquant_policy_v2_box.sh")], check=True)

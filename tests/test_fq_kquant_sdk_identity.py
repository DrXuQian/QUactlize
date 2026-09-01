from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).parents[1]
TOOL = ROOT / "tools" / "fq_kquant_sdk_identity.py"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sdk(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    root = tmp_path / "sdk"
    (root / "bin").mkdir(parents=True)
    runtime = root / "targets" / "x86_64-linux" / "lib"
    runtime.mkdir(parents=True)
    (root / "lib").symlink_to("targets/x86_64-linux/lib", target_is_directory=True)
    release = root / "release.yaml"
    release.write_text("version: test-sdk-1\n")
    for name in ("hgcc", "hgobjdump"):
        path = root / "bin" / name
        path.write_bytes((name + "\n").encode())
        path.chmod(0o755)
    libraries = []
    for name in ("libhggc.so", "libhggcrt.13.0.so", "libhggc_wrapper.so",
                 "libhg_wrapper.so"):
        path = runtime / name
        path.write_bytes((name + "\n").encode())
        path.chmod(0o755)
        libraries.append({
            "path": f"lib/{name}",
            "size": path.stat().st_size,
            "sha256": _sha(path),
        })
    (runtime / "libhggcrt1.so").symlink_to("libhggcrt.13.0.so")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "sdk": {
            "release": "test-sdk-1",
            "receipt": {"size": release.stat().st_size, "sha256": _sha(release)},
            "compiler": {
                "installed_path": "/build/sdk/bin/hgcc", "sha256": _sha(root / "bin/hgcc")},
            "inspector": {
                "installed_path": "/build/sdk/bin/hgobjdump",
                "sha256": _sha(root / "bin/hgobjdump")},
            "runtime_libraries": libraries,
        },
    }))
    return root, manifest


def _run(root: pathlib.Path, manifest: pathlib.Path, output: pathlib.Path, **env):
    run_env = dict(os.environ)
    run_env.pop("ALLOW_UNVERIFIED_SDK", None)
    run_env.update(env)
    return subprocess.run(
        [sys.executable, str(TOOL), "--manifest", str(manifest),
         "--sdk-root", str(root), "--output", str(output)],
        text=True, capture_output=True, check=False, env=run_env)


def test_strict_identity_records_every_actual_sdk_file(tmp_path):
    root, manifest = _sdk(tmp_path)
    output = tmp_path / "identity.json"
    result = _run(root, manifest, output)
    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text())
    assert document["identity_status"] == "VERIFIED"
    assert document["evidence_grade"] == "verified-sdk"
    assert document["mismatches"] == []
    assert document["actual"]["release"]["release"] == "test-sdk-1"
    assert document["actual"]["release"]["size"] > 0
    assert len(document["actual"]["release"]["sha256"]) == 64
    assert document["actual"]["compiler"]["size"] > 0
    assert document["actual"]["inspector"]["size"] > 0
    assert document["actual"]["runtime_directory"] == {
        "logical_path": "lib",
        "link_target": "targets/x86_64-linux/lib",
        "resolved_path": "targets/x86_64-linux/lib",
    }
    assert len(document["actual"]["runtime_libraries"]) == 4
    assert all(row["size"] > 0 and len(row["sha256"]) == 64
               for row in document["actual"]["runtime_libraries"])
    assert document["matches"]["release"] == {
        "value": True, "size": True, "sha256": True}
    assert all(all(fields.values()) for fields in
               document["matches"]["runtime_libraries"].values())


def test_strict_mismatch_is_red_but_explicit_opt_in_is_unverified(tmp_path):
    root, manifest = _sdk(tmp_path)
    (root / "release.yaml").write_text("version: other-sdk\n")
    output = tmp_path / "identity.json"
    strict = _run(root, manifest, output)
    assert strict.returncode == 2
    assert "SDK identity differs" in strict.stderr
    assert not output.exists()

    relaxed = _run(root, manifest, output, ALLOW_UNVERIFIED_SDK="1")
    assert relaxed.returncode == 0, relaxed.stderr
    document = json.loads(output.read_text())
    assert document["identity_status"] == "MISMATCH_ALLOWED"
    assert document["evidence_grade"] == "unverified-sdk"
    assert document["expected"]["release"] == "test-sdk-1"
    assert document["actual"]["release"]["release"] == "other-sdk"
    assert {row["field"] for row in document["mismatches"]} == {
        "release.value", "release.size", "release.sha256"}
    assert document["matches"]["release"] == {
        "value": False, "size": False, "sha256": False}


@pytest.mark.parametrize("value", ["", "yes", "true", "2", " 1"])
def test_opt_in_accepts_only_exact_one(tmp_path, value):
    root, manifest = _sdk(tmp_path)
    output = tmp_path / "identity.json"
    result = _run(root, manifest, output, ALLOW_UNVERIFIED_SDK=value)
    assert result.returncode == 2
    assert "must be exactly 0 or 1" in result.stderr


@pytest.mark.parametrize("plant", ["missing-runtime", "compiler-symlink", "root-symlink"])
def test_opt_in_never_relaxes_sdk_structure(tmp_path, plant):
    root, manifest = _sdk(tmp_path)
    if plant == "missing-runtime":
        (root / "lib/libhggc_wrapper.so").unlink()
        supplied = root
    elif plant == "compiler-symlink":
        compiler = root / "bin/hgcc"
        compiler.unlink()
        compiler.symlink_to("hgobjdump")
        supplied = root
    else:
        supplied = tmp_path / "sdk-link"
        supplied.symlink_to(root, target_is_directory=True)
    result = _run(
        supplied, manifest, tmp_path / "identity.json",
        ALLOW_UNVERIFIED_SDK="1")
    assert result.returncode == 2
    assert not (tmp_path / "identity.json").exists()


def test_runtime_digest_mismatch_is_retained_not_relabelled_pass(tmp_path):
    root, manifest = _sdk(tmp_path)
    runtime = root / "lib/libhggc.so"
    runtime.write_bytes(b"different-runtime\n")
    runtime.chmod(0o755)
    output = tmp_path / "identity.json"
    result = _run(root, manifest, output, ALLOW_UNVERIFIED_SDK="1")
    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text())
    rows = {row["field"]: row for row in document["mismatches"]}
    assert "runtime_libraries[lib/libhggc.so].size" in rows
    assert "runtime_libraries[lib/libhggc.so].sha256" in rows
    actual = next(row for row in document["actual"]["runtime_libraries"]
                  if row["path"] == "lib/libhggc.so")
    assert rows["runtime_libraries[lib/libhggc.so].sha256"]["actual"] == actual["sha256"]
    assert document["identity_status"] == "MISMATCH_ALLOWED"
    assert document["evidence_grade"] == "unverified-sdk"

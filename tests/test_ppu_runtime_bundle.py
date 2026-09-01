import hashlib
import json
import pathlib
import re
import subprocess

import pytest

from quactlize import ppu_bundle


ROOT = pathlib.Path(__file__).parents[1]


def _bundle(tmp_path):
    libraries = []
    for role in ppu_bundle.LIBRARY_ROLES:
        payload = f"{role.role}:{role.packed_format}".encode()
        (tmp_path / role.filename).write_bytes(payload)
        libraries.append({
            "role": role.role,
            "filename": role.filename,
            "packed_scale": role.packed_scale,
            "packed_format": role.packed_format,
            "qtype": role.qtype,
            "dense_only": role.qtype,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "definitions": role.definitions,
        })
    manifest = {
        "schema": ppu_bundle.SCHEMA,
        "schema_version": ppu_bundle.SCHEMA_VERSION,
        "source": {"commit": "a" * 40, "tree_state": "clean", "submodules": []},
        "toolchain": {
            "arch": "ppu0010",
            "sdk_release": ppu_bundle.SDK_RELEASE,
            "sdk_archive_sha256": ppu_bundle.SDK_ARCHIVE_SHA256,
            "hgcc": "test",
        },
        "libraries": libraries,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_runtime_bundle_manifest_binds_all_six_files(tmp_path):
    expected = _bundle(tmp_path)
    assert ppu_bundle.verify_bundle(tmp_path, inspect_binaries=False) == expected


def test_runtime_bundle_requires_loader_facing_arrangement_v2_exports():
    assert {
        "quactlize_ppu_canonical_arrangement_v2",
        "quactlize_ppu_prepare_fully_quantized_for_arrangement_v2",
        "quactlize_ppu_recover_fully_quantized_for_arrangement_v2",
        "quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2",
        "quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2",
        "quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2",
        "quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4",
        "quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v2",
        "quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2",
        "quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_config_valid_for_arrangement_v2",
        "quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2",
    } <= ppu_bundle.REQUIRED_EXPORTS


def test_runtime_bundle_requires_complete_units_producer_exports():
    assert {
        "quactlize_ppu_units_bytes",
        "quactlize_ppu_prepare_units",
        "quactlize_ppu_prepare_units_grouped",
    } <= ppu_bundle.REQUIRED_EXPORTS


def test_runtime_bundle_requires_scalefirst_prepass_exports():
    assert ppu_bundle.SCALEFIRST_PREPASS_REQUIRED_EXPORTS <= ppu_bundle.REQUIRED_EXPORTS


@pytest.mark.parametrize("missing", sorted(ppu_bundle.SCALEFIRST_PREPASS_REQUIRED_EXPORTS))
def test_runtime_bundle_binary_inspection_requires_each_scalefirst_prepass_export(
        tmp_path, monkeypatch, missing):
    _bundle(tmp_path)
    sdk = tmp_path.with_name(tmp_path.name + "-sdk")
    (sdk / "bin").mkdir(parents=True)
    (sdk / "release.yaml").write_text(
        f"version: {ppu_bundle.SDK_RELEASE}\n", encoding="utf-8")
    inspector = sdk / "bin" / "hgobjdump"
    inspector.write_text("#!/bin/sh\n", encoding="utf-8")
    inspector.chmod(0o755)
    exports = ppu_bundle.REQUIRED_EXPORTS - {missing}
    monkeypatch.setattr(
        ppu_bundle, "_run",
        lambda command: "\n".join(
            f"00000000 T {symbol}" for symbol in sorted(exports)))
    with pytest.raises(ppu_bundle.BundleError, match=missing):
        ppu_bundle.verify_bundle(tmp_path, sdk_root=sdk, inspect_binaries=True)


@pytest.mark.parametrize("missing", sorted(ppu_bundle.ANY_M_REQUIRED_EXPORTS))
def test_runtime_bundle_binary_inspection_requires_each_any_m_export(
        tmp_path, monkeypatch, missing):
    _bundle(tmp_path)
    sdk = tmp_path.with_name(tmp_path.name + "-sdk")
    (sdk / "bin").mkdir(parents=True)
    (sdk / "release.yaml").write_text(
        f"version: {ppu_bundle.SDK_RELEASE}\n", encoding="utf-8")
    inspector = sdk / "bin" / "hgobjdump"
    inspector.write_text("#!/bin/sh\n", encoding="utf-8")
    inspector.chmod(0o755)
    exports = ppu_bundle.REQUIRED_EXPORTS - {missing}

    def inspect(command):
        assert command[:3] == ["nm", "-D", "--defined-only"]
        return "\n".join(f"00000000 T {symbol}" for symbol in sorted(exports))

    monkeypatch.setattr(ppu_bundle, "_run", inspect)
    with pytest.raises(ppu_bundle.BundleError, match=missing):
        ppu_bundle.verify_bundle(tmp_path, sdk_root=sdk, inspect_binaries=True)


@pytest.mark.parametrize("plant", [
    "wrong-role", "wrong-format", "wrong-hash", "wrong-definitions",
    "extra-manifest-field", "extra-library-field", "extra-file", "symlink",
])
def test_runtime_bundle_rejects_identity_and_file_plants(tmp_path, plant):
    manifest = _bundle(tmp_path)
    if plant == "wrong-role":
        manifest["libraries"][0]["role"] = "fmt0"
    elif plant == "wrong-format":
        manifest["libraries"][3]["packed_format"] = 4
    elif plant == "wrong-hash":
        manifest["libraries"][2]["sha256"] = "0" * 64
    elif plant == "wrong-definitions":
        manifest["libraries"][2]["definitions"] = ["PPU_PACKED_SCALE=1"]
    elif plant == "extra-manifest-field":
        manifest["unmeasured"] = True
    elif plant == "extra-library-field":
        manifest["libraries"][2]["unmeasured"] = True
    elif plant == "extra-file":
        (tmp_path / "stale.log").write_text("not admitted")
    else:
        path = tmp_path / ppu_bundle.LIBRARY_ROLES[0].filename
        path.unlink()
        path.symlink_to(ppu_bundle.LIBRARY_ROLES[1].filename)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ppu_bundle.BundleError):
        ppu_bundle.verify_bundle(tmp_path, inspect_binaries=False)


def test_bundle_builder_owns_the_exact_six_role_recipe():
    script = (ROOT / "tools" / "build_ppu_runtime_bundle.sh").read_text()
    rows = re.findall(
        r"^(default|fmt[0-4]) ([01]) (-?\d+) (\d+) (libquactlize_ppu(?:_fmt[0-4])?\.so)$",
        script, re.MULTILINE)
    got = [
        (role, filename, int(packed_scale),
         None if int(packed_format) < 0 else int(packed_format), int(qtype))
        for role, packed_scale, packed_format, qtype, filename in rows
    ]
    want = [
        (role.role, role.filename, role.packed_scale, role.packed_format, role.qtype)
        for role in ppu_bundle.LIBRARY_ROLES
    ]
    assert got == want
    assert "PPU_BUILD_RESUME=0" in script
    assert "PPU_SDK_ARCHIVE" in script
    assert "sha256sum \"$sdk_archive\"" in script
    assert "Release version $sdk_release" in script
    assert 'bundle_jobs="${PPU_BUNDLE_JOBS:-1}"' in script
    assert 'setsid env PPU_SDK="$sdk"' in script
    assert "trap 'on_build_signal INT 130' INT" in script
    assert "trap on_build_exit EXIT" in script
    assert "assert_source_state \"$root\" \"$source_sha\" \"$submodules_file\" 'after builds'" in script
    assert "assert_source_state \"$root\" \"$source_sha\" \"$submodules_file\" 'before publish'" in script
    assert 'assert_role_source "$work/$role" "$role" "$source_sha"' in script
    assert 'done <"$roles"' in script
    assert 'python3 -m quactlize.ppu_bundle "$stage" --ppu-sdk "$sdk"' in script
    assert 'host_loader="${PPU_HOST_LOADER:-}"' in script
    assert 'host_python="${PPU_HOST_PYTHON:-}"' in script
    assert 'host_library_path="${PPU_HOST_LIBRARY_PATH:-}"' in script
    assert '"${selected_config_oracle[@]}" "$stage"' in script
    assert "mv -- \"$stage\" \"$out\"" in script
    result = subprocess.run(["bash", "-n", str(ROOT / "tools" / "build_ppu_runtime_bundle.sh")])
    assert result.returncode == 0


def test_bundle_builder_rejects_a_dangling_output_symlink(tmp_path):
    output = tmp_path / "bundle"
    output.symlink_to(tmp_path / "missing")
    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "build_ppu_runtime_bundle.sh"), str(output)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stdout


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _source_builder_call(function, *args):
    command = 'source "$1"; shift; ' + function + ' "$@"'
    return subprocess.run(
        ["bash", "-c", command, "bash", str(ROOT / "tools" / "build_ppu_runtime_bundle.sh"),
         *(str(arg) for arg in args)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def test_bundle_builder_source_authority_rejects_clean_cross_head_drift(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Bundle Test")
    (repo / "tracked").write_text("same tree\n")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-m", "A")
    source_sha = _git(repo, "rev-parse", "HEAD")
    submodules = tmp_path / "submodules.status"
    submodules.write_text("")

    clean = _source_builder_call(
        "assert_source_state", repo, source_sha, submodules, "test-clean",
    )
    assert clean.returncode == 0, clean.stdout

    (repo / "tests").mkdir()
    untracked_input = repo / "tests" / "untracked.cu"
    untracked_input.write_text("build input\n")
    untracked = _source_builder_call(
        "assert_source_state", repo, source_sha, submodules, "test-untracked",
    )
    assert untracked.returncode != 0
    assert "untracked build input" in untracked.stdout
    untracked_input.unlink()

    _git(repo, "commit", "--allow-empty", "-m", "B")
    drift = _source_builder_call(
        "assert_source_state", repo, source_sha, submodules, "test-final",
    )
    assert drift.returncode != 0
    assert "source HEAD changed" in drift.stdout


@pytest.mark.parametrize("plant", ["wrong", "missing", "symlink", "dirty"])
def test_bundle_builder_rejects_role_source_authority_plants(tmp_path, plant):
    build = tmp_path / "fmt2"
    build.mkdir()
    marker = build / ".quactlize-source-head"
    source_sha = "a" * 40
    marker.write_text(source_sha + "\n")
    if plant == "wrong":
        marker.write_text("b" * 40 + "\n")
    elif plant == "missing":
        marker.unlink()
    elif plant == "symlink":
        marker.unlink()
        target = tmp_path / "authority"
        target.write_text(source_sha + "\n")
        marker.symlink_to(target)
    elif plant == "dirty":
        (build / ".quactlize-source-dirty").write_text("M tracked\n")

    result = _source_builder_call("assert_role_source", build, "fmt2", source_sha)
    assert result.returncode != 0
    assert "fmt2" in result.stdout

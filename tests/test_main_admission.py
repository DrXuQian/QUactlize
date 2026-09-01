"""The selective product-main boundary is exact and fail-closed."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ci import check_main_admission as admission


def _candidate(tmp_path):
    admission._write_self_test_candidate(tmp_path)
    return tmp_path


def _manifest(root):
    path = root / admission.ADMISSION_MANIFEST
    return path, json.loads(path.read_text())


def _set_console_scripts(root, scripts):
    rows = [
        "[project]",
        "name='quactlize'",
        "[project.scripts]",
        *(f"{json.dumps(name)}={json.dumps(target)}"
          for name, target in scripts.items()),
        "[tool.setuptools]",
        "packages=['quactlize']",
        "",
    ]
    (root / "pyproject.toml").write_text("\n".join(rows))
    path, value = _manifest(root)
    value["console_scripts"] = scripts
    path.write_text(json.dumps(value))


def test_main_admission_internal_controls():
    admission.self_test()


def test_clean_explicit_candidate_is_admitted(tmp_path):
    root = _candidate(tmp_path)
    assert admission.audit(root) == []


def test_manifest_generator_seeds_exact_reviewable_surface_and_refuses_overwrite(tmp_path):
    root = _candidate(tmp_path)
    manifest_path = root / admission.ADMISSION_MANIFEST
    manifest_path.unlink()
    written = admission.generate_manifest(root)
    assert written == manifest_path
    value = json.loads(written.read_text())
    assert value["files"] == sorted([
        admission.ADMISSION_MANIFEST,
        "pyproject.toml",
        "quactlize/__init__.py",
        "quactlize/formats.py",
    ])
    assert admission.audit(root) == []
    with pytest.raises(ValueError, match="refusing to overwrite"):
        admission.generate_manifest(root)


@pytest.mark.parametrize("token,label", [
    (admission._COLLABORATION_NAMES[0], "collaboration provenance"),
    (admission._COLLABORATION_NAMES[1], "collaboration provenance"),
    (admission._RETIRED_LAYOUT, "retired offline layout"),
    (admission._DIRECT_LAYOUT, "experimental direct layout"),
    (admission._B_CHUNK_SWITCH, "global B delivery switch"),
    (admission._FUSED_SWITCH, "global packed metadata switch"),
    (admission._DEVICE_MACROS[0], "non-PPU architecture guard"),
    ("#include <" + "cuda" + "_runtime.h>", "non-PPU compiler/runtime dependency"),
    (admission._VENDOR_NAME + " platform path", "non-PPU platform-specific source"),
    ("rt_test_" + "fail", "test backend hook"),
])
def test_forbidden_product_sources_are_rejected(tmp_path, token, label):
    root = _candidate(tmp_path)
    (root / "quactlize" / "formats.py").write_text(f"# {token}\n")
    assert any(label in hit for hit in admission.audit(root))


def test_inventory_public_module_and_console_surfaces_are_independent_gates(tmp_path):
    root = _candidate(tmp_path)
    (root / "unlisted.txt").write_text("extra")
    assert any("file inventory mismatch" in hit for hit in admission.audit(root))
    (root / "unlisted.txt").unlink()

    public = root / "quactlize" / "new_public.py"
    public.write_text("VALUE = 1\n")
    path, value = _manifest(root)
    value["files"].append("quactlize/new_public.py")
    value["files"].sort()
    path.write_text(json.dumps(value))
    assert any("public Python module surface" in hit for hit in admission.audit(root))

    public.unlink()
    value["files"].remove("quactlize/new_public.py")
    value["console_scripts"] = {"other": "quactlize.formats:FORMAT"}
    path.write_text(json.dumps(value))
    assert any("console-script surface" in hit for hit in admission.audit(root))


def test_scaffolding_paths_and_symlinks_are_rejected_even_when_declared(tmp_path):
    root = _candidate(tmp_path)
    scaffold = root / "dev" / "probe.py"
    scaffold.parent.mkdir()
    scaffold.write_text("VALUE = 1\n")
    path, value = _manifest(root)
    value["files"].append("dev/probe.py")
    value["files"].sort()
    path.write_text(json.dumps(value))
    assert any("scaffolding path" in hit for hit in admission.audit(root))

    scaffold.unlink()
    scaffold.parent.rmdir()
    value["files"].remove("dev/probe.py")
    link = root / "alias.py"
    link.symlink_to("quactlize/formats.py")
    value["files"].append("alias.py")
    value["files"].sort()
    path.write_text(json.dumps(value))
    assert any("symbolic links" in hit for hit in admission.audit(root))


@pytest.mark.parametrize("directory", [
    ".coord", ".codex", "dev", "scratchpad", "benchmark", "benchmarks",
    "artifact", "artifacts", ".artifacts", "profile", "profiles", "profiler",
    "profilers", ".profiler", "profiling", "acu", "diag", "diags",
    "diagnostic", "diagnostics", ".diagnostics", "probe", "probes",
])
def test_every_forbidden_path_family_is_rejected_when_declared(
        tmp_path, directory):
    root = _candidate(tmp_path)
    planted = root / "nested" / directory / "payload.txt"
    planted.parent.mkdir(parents=True)
    planted.write_text("not product input\n")
    path, value = _manifest(root)
    value["files"].append(planted.relative_to(root).as_posix())
    value["files"].sort()
    path.write_text(json.dumps(value))
    hits = admission.audit(root)
    assert any("scaffolding path" in hit and f"/{directory}/" in hit
               for hit in hits), hits


@pytest.mark.parametrize("directory", [
    "artifact-notes", "profiler_api", "diagnostic-guide", "coordinates",
])
def test_forbidden_path_families_match_exact_components(tmp_path, directory):
    root = _candidate(tmp_path)
    note = root / "docs" / directory / "notes.txt"
    note.parent.mkdir(parents=True)
    note.write_text("product documentation\n")
    path, value = _manifest(root)
    value["files"].append(note.relative_to(root).as_posix())
    value["files"].sort()
    path.write_text(json.dumps(value))
    assert admission.audit(root) == []


@pytest.mark.parametrize("name,target", [
    ("bad command", "quactlize.formats:FORMAT"),
    ("-bad", "quactlize.formats:FORMAT"),
    ("bad-", "quactlize.formats:FORMAT"),
    ("good", "quactlize.formats"),
    ("good", "quactlize.formats:bad-name"),
    ("good", "quactlize.formats:FORMAT:extra"),
])
def test_console_script_name_and_target_syntax_are_fail_closed(
        tmp_path, name, target):
    root = _candidate(tmp_path)
    _set_console_scripts(root, {name: target})
    assert any("console_scripts" in hit or "[project.scripts]" in hit
               for hit in admission.audit(root))


@pytest.mark.parametrize("target,needle", [
    ("quactlize.missing:main", "target module does not exist"),
    ("quactlize.formats:missing", "target object does not exist"),
])
def test_console_script_target_module_and_object_must_exist(
        tmp_path, target, needle):
    root = _candidate(tmp_path)
    _set_console_scripts(root, {"quactlize-tool": target})
    assert any(needle in hit for hit in admission.audit(root))


def test_console_script_target_can_be_a_statically_bound_function(tmp_path):
    root = _candidate(tmp_path)
    (root / "quactlize" / "formats.py").write_text(
        "def main():\n    return 0\n")
    _set_console_scripts(root, {"quactlize-tool": "quactlize.formats:main"})
    assert admission.audit(root) == []


def test_python39_falls_back_to_tomli(monkeypatch):
    real_import = builtins.__import__
    fake_tomli = SimpleNamespace(loads=lambda text: {"parsed": text})

    def planted_import(name, *args, **kwargs):
        if name == "tomllib":
            raise ImportError("Python 3.9 plant")
        if name == "tomli":
            return fake_tomli
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", planted_import)
    parser, backend = admission._load_toml_backend()
    assert backend == "tomli"
    assert parser.loads("payload") == {"parsed": "payload"}


def test_python39_tomli_dependency_is_declared():
    project = admission._read_pyproject(Path("pyproject.toml"))
    assert "tomli>=1.1.0; python_version < '3.11'" in \
        project["project"]["dependencies"]


def test_binary_lfs_pointer_and_unknown_file_types_are_not_source(tmp_path):
    root = _candidate(tmp_path)
    path, value = _manifest(root)
    binary = root / "quactlize" / "kernel.so"
    binary.write_bytes(b"\x7fELF")
    value["files"].append("quactlize/kernel.so")
    value["files"].sort()
    path.write_text(json.dumps(value))
    assert any("binary or unrecognised" in hit for hit in admission.audit(root))

    binary.unlink()
    value["files"].remove("quactlize/kernel.so")
    source = root / "quactlize" / "formats.py"
    source.write_text("version https://git-lfs.github.com/spec/v1\n"
                      "oid sha256:" + "0" * 64 + "\nsize 4\n")
    path.write_text(json.dumps(value))
    assert any("Git LFS pointer" in hit for hit in admission.audit(root))


def test_required_extensionless_license_is_text_scanned_and_admitted(tmp_path):
    root = _candidate(tmp_path)
    license_path = root / "LICENSE"
    license_path.write_text("Apache License 2.0\n")
    path, value = _manifest(root)
    value["files"].append("LICENSE")
    value["files"].sort()
    path.write_text(json.dumps(value))
    assert admission.audit(root) == []

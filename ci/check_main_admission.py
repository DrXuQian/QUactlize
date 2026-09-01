#!/usr/bin/env python3
"""Fail-closed audit for a selectively staged product-main candidate.

The candidate owns an exact file inventory and an exact Python/CLI surface in
``.main-admission.json``.  This checker does not treat the development checkout
as a product candidate: callers must stage the intended files into a separate
tree, write the manifest, and audit that tree.

Seed a reviewable exact manifest, then audit it::

    python ci/check_main_admission.py --root /tmp/quactlize-main --generate-manifest
    python ci/check_main_admission.py --root /tmp/quactlize-main

Generation refuses to overwrite an existing manifest. It records the tree; it
does not waive any content rule, and the generated candidate is audited in the
same invocation.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional


def _load_toml_backend():
    try:
        import tomllib as parser
        return parser, "tomllib"
    except ImportError:  # pragma: no cover - selected on Python 3.9/3.10
        try:
            import tomli as parser
            return parser, "tomli"
        except ImportError:  # pragma: no cover - exercised through an import plant
            return None, None


_toml, _TOML_BACKEND = _load_toml_backend()


ADMISSION_SCHEMA = "quactlize.main-admission"
ADMISSION_VERSION = 1
ADMISSION_MANIFEST = ".main-admission.json"
MANIFEST_FIELDS = {
    "schema", "version", "files", "python_packages", "public_python_modules",
    "console_scripts",
}
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cmake", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".in",
    ".inc", ".inl", ".json", ".md", ".py", ".pyi", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}
EXTENSIONLESS_TEXT = {"LICENSE", "NOTICE", "COPYING", "Makefile", ".gitignore", ".gitmodules"}


def _word(*parts: str) -> str:
    # Keep the checker admissible in the tree it checks: deny-list spellings are
    # assembled rather than embedded in its own source.
    return "".join(parts)


_COLLABORATION_NAMES = (_word("co", "dex"), _word("clau", "de"))
_RETIRED_LAYOUT = _word("x", "plane")
_DIRECT_LAYOUT = _word("n16", "k64")
_B_CHUNK_SWITCH = _word("PPU_B_", "CHUNK")
_FUSED_SWITCH = _word("PPU_PACKED_SCALE_", "FUSED")
_VENDOR_NAME = _word("NVI", "DIA")
_DEVICE_MACROS = (_word("__CUD", "ACC__"), _word("__CUDA_", "ARCH__"))
_FORBIDDEN_TEXT = (
    ("collaboration provenance", re.compile(
        r"\b(?:" + "|".join(map(re.escape, _COLLABORATION_NAMES)) + r")\b", re.I)),
    ("retired offline layout", re.compile(r"\b" + re.escape(_RETIRED_LAYOUT) + r"\b", re.I)),
    ("experimental direct layout", re.compile(
        r"(?:" + re.escape(_DIRECT_LAYOUT) + r"|layout[\s_=-]*3\b)", re.I)),
    ("global B delivery switch", re.compile(r"\b" + re.escape(_B_CHUNK_SWITCH) + r"\b")),
    ("global packed metadata switch", re.compile(r"\b" + re.escape(_FUSED_SWITCH) + r"\b")),
    ("non-PPU architecture guard", re.compile(
        r"\b(?:" + "|".join(map(re.escape, _DEVICE_MACROS)) + r")\b")),
    ("non-PPU compiler/runtime dependency", re.compile(
        r"(?:\b" + _word("nv", "cc") + r"\b|#\s*include\s*[<\"]cuda(?:_runtime|_fp16)?\.h[>\"])", re.I)),
    ("test backend hook", re.compile(
        r"(?:" + _word("rt_test_", "fail") + r"|" + _word("QUACTLIZE_", "FAKE") +
        r"|" + _word("fake", " backend") + r")", re.I)),
)
def _path_family(*names: str) -> set[str]:
    return {variant for name in names for variant in (name, "." + name)}


_FORBIDDEN_PATH_FAMILIES = {
    "development/control": {
        _word("d", "ev"), _word("scratch", "pad"),
        _word(".", "coord"), _word(".", "codex"),
    },
    "benchmark": _path_family(
        _word("bench", "mark"), _word("bench", "marks")),
    "artifact": _path_family(
        _word("arti", "fact"), _word("arti", "facts")),
    "profiler": _path_family(
        _word("pro", "file"), _word("pro", "files"),
        _word("pro", "filer"), _word("pro", "filers"),
        _word("pro", "filing"), _word("a", "cu")),
    "diagnostic": _path_family(
        _word("d", "iag"), _word("d", "iags"),
        _word("diag", "nostic"), _word("diag", "nostics"),
        _word("pro", "be"), _word("pro", "bes")),
}
_CONSOLE_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_PYTHON_NAME = r"[A-Za-z_]\w*"
_CONSOLE_TARGET = re.compile(
    rf"(?P<module>{_PYTHON_NAME}(?:\.{_PYTHON_NAME})*):"
    rf"(?P<object>{_PYTHON_NAME}(?:\.{_PYTHON_NAME})*)\Z")


def audit(root: Path) -> list[str]:
    """Return every admission violation; an empty list is an admissible tree."""
    root = Path(root).resolve()
    if not root.is_dir():
        return [f"candidate root is not a directory: {root}"]
    manifest_path = root / ADMISSION_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return [f"missing regular admission manifest: {ADMISSION_MANIFEST}"]
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid admission manifest: {exc}"]
    hits = _validate_manifest(value)
    if hits:
        return hits

    declared = set(value["files"])
    actual, symlinks = set(), []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlinks.append(rel)
        elif path.is_file():
            actual.add(rel)
    for rel in symlinks:
        hits.append(f"{rel}: symbolic links are not admitted")
    if actual != declared:
        hits.append(
            "file inventory mismatch: "
            f"missing={sorted(declared - actual)} extra={sorted(actual - declared)}")

    for rel in sorted(actual & declared):
        path = root / rel
        parts = [part.casefold() for part in PurePosixPath(rel).parts]
        bad_families = sorted({
            family
            for family, names in _FORBIDDEN_PATH_FAMILIES.items()
            if any(part in names for part in parts[:-1])
        })
        if bad_families or _word("fa", "ke") in parts[-1]:
            detail = ",".join(bad_families) if bad_families else "fake-file"
            hits.append(
                f"{rel}: development/test scaffolding path is not admitted "
                f"(family={detail})")
        if path.suffix.lower() not in SOURCE_SUFFIXES and path.name not in EXTENSIONLESS_TEXT:
            hits.append(f"{rel}: binary or unrecognised product file type is not admitted")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            hits.append(f"{rel}: declared source is not UTF-8")
            continue
        if text.startswith("version https://git-lfs.github.com/spec/v1\n"):
            hits.append(f"{rel}: Git LFS pointer is not product source")
            continue
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            for label, pattern in _FORBIDDEN_TEXT:
                if pattern.search(line):
                    hits.append(f"{rel}:{lineno}: {label}")
            # Vendor copyright notices must remain intact; every other vendor
            # mention is platform-specific implementation provenance.
            if re.search(r"\b" + re.escape(_VENDOR_NAME) + r"\b", line, re.I) and not \
                    re.search(r"Copyright.*" + re.escape(_VENDOR_NAME) + r".*All rights reserved", line, re.I):
                hits.append(f"{rel}:{lineno}: non-PPU platform-specific source")

    hits.extend(_audit_python_surface(root, value))
    return hits


def _validate_manifest(value) -> list[str]:
    if not isinstance(value, dict) or set(value) != MANIFEST_FIELDS:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        return [f"admission manifest fields must be exactly {sorted(MANIFEST_FIELDS)}, got {got}"]
    if value["schema"] != ADMISSION_SCHEMA or value["version"] != ADMISSION_VERSION:
        return [f"unsupported admission schema/version: {value['schema']!r}/{value['version']!r}"]
    hits = []
    for field in ("files", "python_packages", "public_python_modules"):
        items = value[field]
        if (not isinstance(items, list) or not all(isinstance(item, str) and item for item in items) or
                items != sorted(set(items))):
            hits.append(f"{field} must be a sorted list of unique nonempty strings")
    scripts = value["console_scripts"]
    hits.extend(_console_script_syntax_hits(scripts, "console_scripts"))
    if hits:
        return hits
    if ADMISSION_MANIFEST not in value["files"]:
        hits.append(f"files must include {ADMISSION_MANIFEST}")
    for rel in value["files"]:
        pure = PurePosixPath(rel)
        if pure.is_absolute() or rel != pure.as_posix() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
            hits.append(f"invalid inventory path: {rel!r}")
    for package in value["python_packages"]:
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", package):
            hits.append(f"invalid Python package name: {package!r}")
    for module in value["public_python_modules"]:
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            hits.append(f"invalid public Python module name: {module!r}")
    return hits


def _audit_python_surface(root: Path, manifest: dict) -> list[str]:
    hits = []
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return ["pyproject.toml is required to bind the installable public surface"]
    if _toml is None:
        return ["TOML parser unavailable: install tomli on Python 3.9/3.10"]
    try:
        project = _read_pyproject(pyproject)
    except ValueError as exc:
        return [str(exc)]
    packages = project.get("tool", {}).get("setuptools", {}).get("packages")
    observed_packages = (sorted(set(packages))
                         if isinstance(packages, list) and all(isinstance(item, str) for item in packages)
                         else packages)
    if observed_packages != manifest["python_packages"]:
        hits.append(
            f"setuptools package surface differs: manifest={manifest['python_packages']} "
            f"pyproject={observed_packages!r}")
    scripts = project.get("project", {}).get("scripts", {})
    if scripts != manifest["console_scripts"]:
        hits.append(
            f"console-script surface differs: manifest={manifest['console_scripts']} pyproject={scripts!r}")
    script_syntax = _console_script_syntax_hits(scripts, "[project.scripts]")
    hits.extend(script_syntax)
    if not script_syntax:
        hits.extend(_audit_console_script_targets(
            root, scripts, manifest["python_packages"]))

    discovered, module_hits = _discover_public_modules(root, manifest["python_packages"])
    hits.extend(module_hits)
    if discovered != manifest["public_python_modules"]:
        hits.append(
            "public Python module surface differs: "
            f"manifest={manifest['public_python_modules']} discovered={discovered}")
    return hits


def generate_manifest(root: Path) -> Path:
    """Seed the exact candidate inventory and installable surface for review."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"candidate root is not a directory: {root}")
    path = root / ADMISSION_MANIFEST
    if path.exists() or path.is_symlink():
        raise ValueError(
            f"refusing to overwrite {path}; remove it explicitly before regenerating the reviewed boundary")
    symlinks = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_symlink()]
    if symlinks:
        raise ValueError(f"candidate contains symbolic links: {symlinks}")
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise ValueError("pyproject.toml is required before generating an admission manifest")
    project = _read_pyproject(pyproject_path)
    packages = project.get("tool", {}).get("setuptools", {}).get("packages")
    scripts = project.get("project", {}).get("scripts", {})
    if (not isinstance(packages, list) or
            any(not isinstance(package, str) or not package for package in packages)):
        raise ValueError("[tool.setuptools].packages must be a list of package names")
    if not isinstance(scripts, dict):
        raise ValueError("[project.scripts] must be a table")
    script_hits = _console_script_syntax_hits(scripts, "[project.scripts]")
    if not script_hits:
        script_hits = _audit_console_script_targets(root, scripts, packages)
    if script_hits:
        raise ValueError("; ".join(script_hits))
    modules, module_hits = _discover_public_modules(root, packages)
    if module_hits:
        raise ValueError("; ".join(module_hits))
    files = sorted(
        {ADMISSION_MANIFEST} |
        {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()})
    value = {
        "schema": ADMISSION_SCHEMA,
        "version": ADMISSION_VERSION,
        "files": files,
        "python_packages": sorted(set(packages)),
        "public_python_modules": modules,
        "console_scripts": dict(sorted(scripts.items())),
    }
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _read_pyproject(path: Path) -> dict:
    if _toml is None:
        raise ValueError("TOML parser unavailable: install tomli on Python 3.9/3.10")
    try:
        return _toml.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # TOMLDecodeError is implementation-specific before 3.11.
        raise ValueError(f"invalid pyproject.toml: {exc}") from exc


def _console_script_syntax_hits(value, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must map command names to module:object entry points"]
    hits = []
    for name, target in value.items():
        if not isinstance(name, str) or not _CONSOLE_NAME.fullmatch(name):
            hits.append(f"{label} has invalid command name: {name!r}")
        if not isinstance(target, str) or not _CONSOLE_TARGET.fullmatch(target):
            hits.append(
                f"{label} entry {name!r} must use dotted.module:object syntax, "
                f"got {target!r}")
    return hits


def _module_source(root: Path, module: str,
                   packages: list[str]) -> Optional[Path]:
    if not any(module == package or module.startswith(package + ".")
               for package in packages):
        return None
    base = root.joinpath(*module.split("."))
    module_file = base.with_suffix(".py")
    package_file = base / "__init__.py"
    if module_file.is_file() and not module_file.is_symlink():
        return module_file
    if package_file.is_file() and not package_file.is_symlink():
        return package_file
    return None


def _bound_nodes(body: list[ast.stmt]) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}

    def bind_target(target: ast.AST, owner: ast.AST) -> None:
        if isinstance(target, ast.Name):
            result[target.id] = owner
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                bind_target(item, owner)

    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                bind_target(target, node)
        elif isinstance(node, ast.AnnAssign):
            bind_target(node.target, node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name.split(".", 1)[0]
                result[name] = node
    return result


def _object_is_bound(tree: ast.Module, dotted_object: str) -> bool:
    parts = dotted_object.split(".")
    bindings = _bound_nodes(tree.body)
    node = bindings.get(parts[0])
    if node is None:
        return False
    for part in parts[1:]:
        if not isinstance(node, ast.ClassDef):
            return False
        node = _bound_nodes(node.body).get(part)
        if node is None:
            return False
    return True


def _audit_console_script_targets(root: Path, scripts: dict[str, str],
                                  packages: list[str]) -> list[str]:
    hits = []
    for name, target in sorted(scripts.items()):
        match = _CONSOLE_TARGET.fullmatch(target)
        if match is None:  # Syntax is adjudicated by the caller.
            continue
        module, object_name = match.group("module"), match.group("object")
        source = _module_source(root, module, packages)
        if source is None:
            hits.append(
                f"console script {name!r} target module does not exist in "
                f"the admitted packages: {module}")
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"),
                             filename=source.as_posix())
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            hits.append(
                f"console script {name!r} target module cannot be parsed: {exc}")
            continue
        if not _object_is_bound(tree, object_name):
            hits.append(
                f"console script {name!r} target object does not exist: "
                f"{module}:{object_name}")
    return hits


def _discover_public_modules(root: Path, packages: list[str]) -> tuple[list[str], list[str]]:
    discovered, hits = [], []
    for package in packages:
        package_path = root.joinpath(*package.split("."))
        if not (package_path / "__init__.py").is_file():
            hits.append(f"declared package {package} has no __init__.py")
            continue
        for path in sorted(package_path.rglob("*.py")):
            rel_parts = path.relative_to(package_path).parts
            stem_parts = rel_parts[:-1] + (() if rel_parts[-1] == "__init__.py" else (path.stem,))
            if any(part.startswith("_") for part in stem_parts):
                continue
            discovered.append(".".join((package, *stem_parts)))
    return sorted(set(discovered)), hits


def _write_self_test_candidate(root: Path) -> None:
    (root / "quactlize").mkdir()
    (root / "quactlize" / "__init__.py").write_text("VALUE = 1\n")
    (root / "quactlize" / "formats.py").write_text("FORMAT = 1\n")
    (root / "pyproject.toml").write_text(
        "[project]\nname='quactlize'\n[project.scripts]\n"
        "quactlize-pack-gguf='quactlize.formats:FORMAT'\n"
        "[tool.setuptools]\npackages=['quactlize']\n")
    files = [ADMISSION_MANIFEST, "pyproject.toml", "quactlize/__init__.py", "quactlize/formats.py"]
    manifest = {
        "schema": ADMISSION_SCHEMA,
        "version": ADMISSION_VERSION,
        "files": files,
        "python_packages": ["quactlize"],
        "public_python_modules": ["quactlize", "quactlize.formats"],
        "console_scripts": {"quactlize-pack-gguf": "quactlize.formats:FORMAT"},
    }
    (root / ADMISSION_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="main-admission-") as tmp:
        root = Path(tmp)
        _write_self_test_candidate(root)
        hits = audit(root)
        if hits:
            raise RuntimeError(f"clean planted candidate failed admission: {hits}")
        target = root / "quactlize" / "formats.py"
        for label, token in (
                ("retired", _RETIRED_LAYOUT),
                ("delivery", _B_CHUNK_SWITCH),
                ("metadata", _FUSED_SWITCH),
                ("architecture", _DEVICE_MACROS[0])):
            target.write_text(f"# {token}\n")
            if not audit(root):
                raise RuntimeError(f"{label} negative plant was admitted")
        target.write_text("FORMAT = 1\n")
        extra = root / "quactlize" / "new_public.py"
        extra.write_text("VALUE = 1\n")
        manifest_path = root / ADMISSION_MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].append("quactlize/new_public.py")
        manifest["files"].sort()
        manifest_path.write_text(json.dumps(manifest))
        if not any("public Python module surface" in hit for hit in audit(root)):
            raise RuntimeError("public-surface negative plant was not adjudicated")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="separately staged product-main candidate tree")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--generate-manifest", action="store_true",
                        help="seed the exact inventory/public surface; refuses overwrite, then audits")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("[main-admission:self-test] PASS")
        if args.root is None:
            return 0
    if args.root is None:
        parser.error("--root is required unless --self-test is used alone")
    if args.generate_manifest:
        try:
            path = generate_manifest(args.root)
        except ValueError as exc:
            print(f"[main-admission:manifest] FAIL {exc}")
            return 1
        print(f"[main-admission:manifest] WROTE {path}; review this boundary before committing it")
    hits = audit(args.root)
    if hits:
        print(f"[main-admission] FAIL root={args.root}")
        for hit in hits:
            print(f"  {hit}")
        return 1
    print(f"[main-admission] PASS root={args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

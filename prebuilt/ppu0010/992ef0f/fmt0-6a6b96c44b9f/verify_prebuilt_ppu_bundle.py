#!/usr/bin/env python3
"""Verify and select one artifact from a prebuilt PPU binary bundle.

The verifier is intentionally independent of quactlize and third-party Python
packages so that a PPU box can run it before loading any shipped binary.  A
successful invocation writes only the selected artifact's absolute path to
stdout.  Every payload in the manifest is checked before that path is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
from typing import Any, Sequence


SCHEMA_VERSION = 1
TOP_LEVEL_FIELDS = {
    "schema_version",
    "source_sha",
    "submodules",
    "sdk",
    "compiler",
    "arch",
    "artifacts",
}
ARTIFACT_FIELDS = {
    "role",
    "qtype",
    "path",
    "size",
    "sha256",
    "target",
    "ppu_defs",
}
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class BundleError(ValueError):
    """The manifest or one of its payloads is not admissible."""


def _one_line(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError(f"{field} must be a non-empty string")
    if any(mark in value for mark in ("\0", "\n", "\r")):
        raise BundleError(f"{field} must be one line")
    return value


def _git_sha(value: Any, field: str) -> str:
    value = _one_line(value, field)
    if not GIT_SHA_RE.fullmatch(value):
        raise BundleError(f"{field} must be a lowercase full Git SHA")
    return value


def _relative_path(value: Any, field: str) -> pathlib.PurePosixPath:
    raw = _one_line(value, field)
    if "\\" in raw:
        raise BundleError(f"{field} must use portable '/' separators")
    path = pathlib.PurePosixPath(raw)
    if (path.is_absolute() or not path.parts or path == pathlib.PurePosixPath(".") or
            any(part in {"", ".", ".."} for part in path.parts) or
            path.as_posix() != raw):
        raise BundleError(f"{field} must be a normalized relative path")
    return path


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_manifest(path: pathlib.Path) -> object:
    if path.is_symlink():
        raise BundleError("manifest must not be a symbolic link")
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source, object_pairs_hook=_unique_object)
    except BundleError:
        raise
    except FileNotFoundError as error:
        raise BundleError(f"manifest is missing: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read manifest {path}: {error}") from error


def _validate_artifact(value: object, index: int) -> dict[str, Any]:
    field = f"artifacts[{index}]"
    if not isinstance(value, dict) or set(value) != ARTIFACT_FIELDS:
        raise BundleError(f"{field} has the wrong fields")

    role = _one_line(value["role"], f"{field}.role")
    qtype = value["qtype"]
    if qtype is not None and (isinstance(qtype, bool) or
                              not isinstance(qtype, int) or qtype < 0):
        raise BundleError(f"{field}.qtype must be a non-negative integer or null")
    path = _relative_path(value["path"], f"{field}.path")
    size = value["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise BundleError(f"{field}.size must be a positive integer")
    digest = _one_line(value["sha256"], f"{field}.sha256")
    if not SHA256_RE.fullmatch(digest):
        raise BundleError(f"{field}.sha256 must be a lowercase SHA-256 digest")
    target = _one_line(value["target"], f"{field}.target")

    definitions = value["ppu_defs"]
    if not isinstance(definitions, list):
        raise BundleError(f"{field}.ppu_defs must be a list")
    checked_definitions = [
        _one_line(item, f"{field}.ppu_defs[{definition_index}]")
        for definition_index, item in enumerate(definitions)
    ]
    if len(set(checked_definitions)) != len(checked_definitions):
        raise BundleError(f"{field}.ppu_defs contains a duplicate")

    return {
        "role": role,
        "qtype": qtype,
        "path": path,
        "size": size,
        "sha256": digest,
        "target": target,
        "ppu_defs": checked_definitions,
    }


def validate_manifest(document: object) -> dict[str, Any]:
    """Validate the v1 schema without touching artifact payloads."""
    if not isinstance(document, dict) or set(document) != TOP_LEVEL_FIELDS:
        raise BundleError("manifest has the wrong top-level fields")
    if (isinstance(document["schema_version"], bool) or
            document["schema_version"] != SCHEMA_VERSION):
        raise BundleError(
            f"unsupported schema_version {document['schema_version']!r}")

    source_sha = _git_sha(document["source_sha"], "source_sha")
    submodules = document["submodules"]
    if not isinstance(submodules, dict):
        raise BundleError("submodules must be an object mapping paths to Git SHAs")
    checked_submodules: dict[str, str] = {}
    for name, revision in submodules.items():
        normalized = _relative_path(name, f"submodules[{name!r}]").as_posix()
        checked_submodules[normalized] = _git_sha(
            revision, f"submodules[{name!r}]")

    sdk = _one_line(document["sdk"], "sdk")
    compiler = _one_line(document["compiler"], "compiler")
    arch = _one_line(document["arch"], "arch")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise BundleError("artifacts must be a non-empty list")
    checked_artifacts = [
        _validate_artifact(artifact, index)
        for index, artifact in enumerate(artifacts)
    ]

    identities: set[tuple[str, int | None, str]] = set()
    paths: set[pathlib.PurePosixPath] = set()
    for artifact in checked_artifacts:
        identity = (artifact["role"], artifact["qtype"], artifact["target"])
        if identity in identities:
            raise BundleError(
                "artifact identity (role, qtype, target) is not unique")
        identities.add(identity)
        if artifact["path"] in paths:
            raise BundleError(f"artifact path {artifact['path']} is not unique")
        paths.add(artifact["path"])

    return {
        "schema_version": SCHEMA_VERSION,
        "source_sha": source_sha,
        "submodules": checked_submodules,
        "sdk": sdk,
        "compiler": compiler,
        "arch": arch,
        "artifacts": checked_artifacts,
    }


def _check_no_symlink(root: pathlib.Path, relative: pathlib.PurePosixPath) -> pathlib.Path:
    if root.is_symlink():
        raise BundleError("bundle directory must not be a symbolic link")
    candidate = root.joinpath(*relative.parts)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise BundleError(f"artifact path contains a symbolic link: {relative}")
    return candidate


def _verify_payload(root: pathlib.Path, artifact: dict[str, Any]) -> pathlib.Path:
    relative = artifact["path"]
    candidate = _check_no_symlink(root, relative)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except FileNotFoundError as error:
        raise BundleError(f"artifact is missing: {relative}") from error
    except OSError as error:
        raise BundleError(f"cannot open artifact {relative}: {error}") from error

    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise BundleError(f"artifact is not a regular file: {relative}")
        if opened.st_size != artifact["size"]:
            raise BundleError(
                f"artifact size differs for {relative}: "
                f"expected {artifact['size']}, got {opened.st_size}")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact["sha256"]:
            raise BundleError(f"artifact SHA-256 differs for {relative}")

        try:
            after = candidate.lstat()
        except FileNotFoundError as error:
            raise BundleError(f"artifact disappeared while verifying: {relative}") from error
        if (stat.S_ISLNK(after.st_mode) or
                (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)):
            raise BundleError(f"artifact changed while verifying: {relative}")
    finally:
        os.close(descriptor)
    return candidate


def _parse_expected_submodules(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, revision = value.partition("=")
        if not separator:
            raise BundleError("--expect-submodule must be PATH=SHA")
        normalized = _relative_path(name, "--expect-submodule path").as_posix()
        checked_revision = _git_sha(revision, "--expect-submodule SHA")
        if normalized in result:
            raise BundleError(f"duplicate --expect-submodule for {normalized}")
        result[normalized] = checked_revision
    return result


def verify_bundle(manifest_path: pathlib.Path, *, role: str,
                  qtype: int | None, target: str,
                  expect_source: str | None = None,
                  expect_submodules: Sequence[str] = (),
                  expect_sdk: str | None = None,
                  expect_compiler: str | None = None,
                  expect_arch: str | None = None,
                  expect_ppu_defs: Sequence[str] | None = None) -> pathlib.Path:
    """Verify the complete bundle and return one uniquely selected payload."""
    manifest_path = pathlib.Path(os.path.abspath(manifest_path))
    document = validate_manifest(_read_manifest(manifest_path))

    if expect_source is not None:
        expected = _git_sha(expect_source, "--expect-source")
        if document["source_sha"] != expected:
            raise BundleError("source_sha differs from --expect-source")
    for name, revision in _parse_expected_submodules(expect_submodules).items():
        if document["submodules"].get(name) != revision:
            raise BundleError(f"submodule {name} differs from --expect-submodule")
    for option, field, expected in (
            ("--expect-sdk", "sdk", expect_sdk),
            ("--expect-compiler", "compiler", expect_compiler),
            ("--expect-arch", "arch", expect_arch)):
        if expected is not None:
            checked = _one_line(expected, option)
            if document[field] != checked:
                raise BundleError(f"{field} differs from {option}")

    checked_role = _one_line(role, "--role")
    checked_target = _one_line(target, "--target")
    selected = [
        artifact for artifact in document["artifacts"]
        if artifact["role"] == checked_role and artifact["qtype"] == qtype and
        artifact["target"] == checked_target
    ]
    if not selected:
        raise BundleError("requested (role, qtype, target) is absent")
    if len(selected) != 1:
        raise BundleError("requested (role, qtype, target) is ambiguous")
    if expect_ppu_defs is not None:
        checked_definitions = [
            _one_line(value, "--expect-ppu-def") for value in expect_ppu_defs
        ]
        if len(set(checked_definitions)) != len(checked_definitions):
            raise BundleError("--expect-ppu-def contains a duplicate")
        if selected[0]["ppu_defs"] != checked_definitions:
            raise BundleError("ppu_defs differ from --expect-ppu-def")

    bundle_root = manifest_path.parent
    verified: dict[pathlib.PurePosixPath, pathlib.Path] = {}
    for artifact in document["artifacts"]:
        verified[artifact["path"]] = _verify_payload(bundle_root, artifact)
    return pathlib.Path(os.path.abspath(verified[selected[0]["path"]]))


def _qtype(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    try:
        result = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "qtype must be a non-negative integer or 'null'") from error
    if result < 0:
        raise argparse.ArgumentTypeError(
            "qtype must be a non-negative integer or 'null'")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--qtype", required=True, type=_qtype)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expect-source")
    parser.add_argument("--expect-submodule", action="append", default=[])
    parser.add_argument("--expect-sdk")
    parser.add_argument("--expect-compiler")
    parser.add_argument("--expect-arch")
    parser.add_argument("--expect-ppu-def", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selected = verify_bundle(
            args.manifest,
            role=args.role,
            qtype=args.qtype,
            target=args.target,
            expect_source=args.expect_source,
            expect_submodules=args.expect_submodule,
            expect_sdk=args.expect_sdk,
            expect_compiler=args.expect_compiler,
            expect_arch=args.expect_arch,
            expect_ppu_defs=args.expect_ppu_def,
        )
    except BundleError as error:
        print(f"prebuilt PPU bundle rejected: {error}", file=sys.stderr)
        return 2
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

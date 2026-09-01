#!/usr/bin/env python3
"""Capture the SDK identity used by a prebuilt FQ K-quant execution.

The build manifest remains the authority for the SDK that produced the
payload.  ``ALLOW_UNVERIFIED_SDK=1`` permits execution with a different SDK,
but it never changes that authority: every actual file identity is retained
and the resulting evidence is explicitly downgraded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import tempfile
from typing import Any, Sequence


SCHEMA = "quactlize.fq-kquant-runtime-sdk-identity.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class SDKIdentityError(ValueError):
    """The installed SDK is unusable or violates the selected policy."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SDKIdentityError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _one_line(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SDKIdentityError(f"{field} must be a nonempty string")
    if any(mark in value for mark in ("\0", "\n", "\r")):
        raise SDKIdentityError(f"{field} must be one line")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SDKIdentityError(f"{field} must be a positive integer")
    return value


def _sha256_value(value: Any, field: str) -> str:
    value = _one_line(value, field)
    if not SHA256_RE.fullmatch(value):
        raise SDKIdentityError(f"{field} must be lowercase SHA-256")
    return value


def _relative_path(value: Any, field: str) -> pathlib.PurePosixPath:
    raw = _one_line(value, field)
    path = pathlib.PurePosixPath(raw)
    if ("\\" in raw or path.is_absolute() or path.as_posix() != raw or
            any(part in {"", ".", ".."} for part in path.parts)):
        raise SDKIdentityError(f"{field} must be a normalized relative path")
    return path


def _digest_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _regular_identity(root: pathlib.Path, relative: pathlib.PurePosixPath, *,
                      executable: bool, field: str,
                      record_path: pathlib.PurePosixPath | None = None) -> dict[str, Any]:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SDKIdentityError(f"{field} must not contain a symbolic link: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(current, flags)
    except OSError as error:
        raise SDKIdentityError(f"cannot open {field} {relative}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SDKIdentityError(f"{field} is not a regular file: {relative}")
        if executable and not before.st_mode & stat.S_IXUSR:
            raise SDKIdentityError(f"{field} is not executable: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = _digest_stream(stream)
        after = current.lstat()
        if (stat.S_ISLNK(after.st_mode) or
                (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)):
            raise SDKIdentityError(f"{field} changed while reading: {relative}")
        return {
            "path": (record_path or relative).as_posix(),
            "size": before.st_size,
            "sha256": digest,
        }
    finally:
        os.close(descriptor)


def _sdk_manifest(document: object) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("sdk"), dict):
        raise SDKIdentityError("manifest.sdk is missing")
    sdk = document["sdk"]
    required = {
        "release", "receipt", "compiler", "inspector", "runtime_libraries",
    }
    if not required <= set(sdk):
        raise SDKIdentityError("manifest.sdk is incomplete")
    release = _one_line(sdk["release"], "manifest.sdk.release")

    receipt = sdk["receipt"]
    if not isinstance(receipt, dict):
        raise SDKIdentityError("manifest.sdk.receipt must be an object")
    expected_receipt = {
        "size": _positive_int(receipt.get("size"), "manifest.sdk.receipt.size"),
        "sha256": _sha256_value(
            receipt.get("sha256"), "manifest.sdk.receipt.sha256"),
    }

    tools: dict[str, dict[str, str]] = {}
    for name in ("compiler", "inspector"):
        row = sdk[name]
        if not isinstance(row, dict):
            raise SDKIdentityError(f"manifest.sdk.{name} must be an object")
        installed = pathlib.PurePosixPath(
            _one_line(row.get("installed_path"), f"manifest.sdk.{name}.installed_path"))
        if not installed.name or installed.name in {".", ".."}:
            raise SDKIdentityError(f"manifest.sdk.{name}.installed_path is malformed")
        tools[name] = {
            "path": f"bin/{installed.name}",
            "sha256": _sha256_value(
                row.get("sha256"), f"manifest.sdk.{name}.sha256"),
        }

    runtime = sdk["runtime_libraries"]
    if not isinstance(runtime, list) or not runtime:
        raise SDKIdentityError("manifest.sdk.runtime_libraries must be nonempty")
    runtime_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(runtime):
        if not isinstance(row, dict):
            raise SDKIdentityError(
                f"manifest.sdk.runtime_libraries[{index}] must be an object")
        path = _relative_path(
            row.get("path"), f"manifest.sdk.runtime_libraries[{index}].path")
        if not path.parts or path.parts[0] != "lib" or path.as_posix() in seen:
            raise SDKIdentityError("manifest SDK runtime library paths are invalid")
        seen.add(path.as_posix())
        runtime_rows.append({
            "path": path.as_posix(),
            "size": _positive_int(
                row.get("size"), f"manifest.sdk.runtime_libraries[{index}].size"),
            "sha256": _sha256_value(
                row.get("sha256"),
                f"manifest.sdk.runtime_libraries[{index}].sha256"),
        })
    return {
        "release": release,
        "receipt": expected_receipt,
        "compiler": tools["compiler"],
        "inspector": tools["inspector"],
        "runtime_libraries": runtime_rows,
    }


def _read_manifest(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SDKIdentityError("manifest must be a real regular file")
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SDKIdentityError(f"cannot read manifest: {error}") from error
    return _sdk_manifest(document)


def _release_value(path: pathlib.Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SDKIdentityError(f"cannot read SDK release receipt: {error}") from error
    values = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*version\s*:\s*([^\s#][^#]*?)\s*(?:#.*)?", line)
        if match:
            values.append(match.group(1).strip().strip("'\""))
    if len(values) != 1 or not values[0] or any(
            mark in values[0] for mark in ("\0", "\n", "\r")):
        raise SDKIdentityError("SDK release.yaml must contain one concrete version")
    return values[0]


def _mismatch(field: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {"field": field, "expected": expected, "actual": actual}


def inspect_sdk(manifest_path: pathlib.Path, sdk_root: pathlib.Path, *,
                allow_unverified: bool = False) -> dict[str, Any]:
    """Return the complete actual SDK identity or reject it.

    ``allow_unverified`` relaxes identity equality only.  Structural failures
    (missing files, symlinks, non-executable tools, or an unusable runtime
    directory) remain fatal.
    """
    expected = _read_manifest(pathlib.Path(manifest_path))
    supplied_root = pathlib.Path(os.path.abspath(sdk_root))
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise SDKIdentityError("PPU_SDK must be a real directory, not a symlink")
    root = pathlib.Path(os.path.realpath(supplied_root))
    if root != supplied_root:
        raise SDKIdentityError("PPU_SDK path must not traverse symbolic links")
    bin_directory = root / "bin"
    if bin_directory.is_symlink() or not bin_directory.is_dir():
        raise SDKIdentityError(f"SDK directory must be real: {bin_directory}")
    logical_runtime = root / "lib"
    runtime_link_target: str | None = None
    if logical_runtime.is_symlink():
        runtime_link_target = os.readlink(logical_runtime)
        target = pathlib.PurePosixPath(runtime_link_target)
        if not runtime_link_target or target.is_absolute() or ".." in target.parts:
            raise SDKIdentityError("SDK lib symlink must be relative and remain inside PPU_SDK")
    elif not logical_runtime.is_dir():
        raise SDKIdentityError(f"SDK runtime directory is missing: {logical_runtime}")
    runtime_directory = pathlib.Path(os.path.realpath(logical_runtime))
    try:
        runtime_relative = runtime_directory.relative_to(root)
    except ValueError as error:
        raise SDKIdentityError("SDK lib directory resolves outside PPU_SDK") from error
    if runtime_directory.is_symlink() or not runtime_directory.is_dir():
        raise SDKIdentityError("resolved SDK runtime directory must be a real directory")

    release = _regular_identity(
        root, pathlib.PurePosixPath("release.yaml"), executable=False,
        field="SDK release receipt")
    release["release"] = _release_value(root / "release.yaml")
    compiler = _regular_identity(
        root, pathlib.PurePosixPath(expected["compiler"]["path"]), executable=True,
        field="SDK compiler")
    inspector = _regular_identity(
        root, pathlib.PurePosixPath(expected["inspector"]["path"]), executable=True,
        field="SDK inspector")
    runtime = [
        _regular_identity(
            runtime_directory,
            pathlib.PurePosixPath(row["path"]).relative_to("lib"), executable=True,
            field=f"SDK runtime library {row['path']}",
            record_path=pathlib.PurePosixPath(row["path"]),
        )
        for row in expected["runtime_libraries"]
    ]

    alias_path = runtime_directory / "libhggcrt1.so"
    if not alias_path.is_symlink():
        raise SDKIdentityError("SDK runtime alias lib/libhggcrt1.so must be a symlink")
    alias_target = os.readlink(alias_path)
    if (not alias_target or pathlib.PurePosixPath(alias_target).is_absolute() or
            len(pathlib.PurePosixPath(alias_target).parts) != 1 or
            alias_target in {".", ".."}):
        raise SDKIdentityError("SDK runtime alias target is not a local library name")
    resolved_alias = alias_path.parent / alias_target
    if resolved_alias.is_symlink() or not resolved_alias.is_file():
        raise SDKIdentityError("SDK runtime alias does not resolve to a real library")

    actual = {
        "root": str(root),
        "runtime_directory": {
            "logical_path": "lib",
            "link_target": runtime_link_target,
            "resolved_path": runtime_relative.as_posix(),
        },
        "release": release,
        "compiler": compiler,
        "inspector": inspector,
        "runtime_libraries": runtime,
        "runtime_alias": {
            "path": "lib/libhggcrt1.so",
            "target": alias_target,
        },
    }
    mismatches: list[dict[str, Any]] = []
    for field, wanted, got in (
            ("release.value", expected["release"], release["release"]),
            ("release.size", expected["receipt"]["size"], release["size"]),
            ("release.sha256", expected["receipt"]["sha256"], release["sha256"]),
            ("compiler.sha256", expected["compiler"]["sha256"], compiler["sha256"]),
            ("inspector.sha256", expected["inspector"]["sha256"], inspector["sha256"]),
            ("runtime_alias.target", "libhggcrt.13.0.so", alias_target)):
        if wanted != got:
            mismatches.append(_mismatch(field, wanted, got))
    actual_runtime = {row["path"]: row for row in runtime}
    for row in expected["runtime_libraries"]:
        got = actual_runtime[row["path"]]
        for name in ("size", "sha256"):
            if row[name] != got[name]:
                mismatches.append(_mismatch(
                    f"runtime_libraries[{row['path']}].{name}", row[name], got[name]))

    if mismatches and not allow_unverified:
        fields = ", ".join(row["field"] for row in mismatches)
        raise SDKIdentityError(f"SDK identity differs: {fields}")
    unverified = bool(mismatches)
    mismatch_fields = {row["field"] for row in mismatches}
    matches = {
        "release": {
            name: f"release.{name}" not in mismatch_fields
            for name in ("value", "size", "sha256")
        },
        "compiler": {
            "sha256": "compiler.sha256" not in mismatch_fields,
        },
        "inspector": {
            "sha256": "inspector.sha256" not in mismatch_fields,
        },
        "runtime_libraries": {
            row["path"]: {
                name: f"runtime_libraries[{row['path']}].{name}" not in mismatch_fields
                for name in ("size", "sha256")
            }
            for row in expected["runtime_libraries"]
        },
        "runtime_alias": {
            "target": "runtime_alias.target" not in mismatch_fields,
        },
    }
    return {
        "schema": SCHEMA,
        "policy": "allow-unverified-sdk" if allow_unverified else "strict",
        "identity_status": "MISMATCH_ALLOWED" if unverified else "VERIFIED",
        "evidence_grade": "unverified-sdk" if unverified else "verified-sdk",
        "expected": expected,
        "actual": actual,
        "matches": matches,
        "mismatches": mismatches,
    }


def _atomic_json(path: pathlib.Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SDKIdentityError("output must not be a symbolic link")
    data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".current",
            delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _allow_unverified_from_environment() -> bool:
    value = os.environ.get("ALLOW_UNVERIFIED_SDK", "0")
    if value not in {"0", "1"}:
        raise SDKIdentityError("ALLOW_UNVERIFIED_SDK must be exactly 0 or 1")
    return value == "1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--sdk-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = inspect_sdk(
            args.manifest, args.sdk_root,
            allow_unverified=_allow_unverified_from_environment())
        _atomic_json(args.output, document)
    except SDKIdentityError as error:
        print(f"FQ K-quant SDK identity rejected: {error}", file=os.sys.stderr)
        return 2
    print(
        "[fq-kquant-sdk] "
        f"{document['identity_status']} evidence_grade={document['evidence_grade']} "
        f"release={document['actual']['release']['release']} "
        f"root={document['actual']['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

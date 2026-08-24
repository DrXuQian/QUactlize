#!/usr/bin/env python3
"""Archive the completed Q4_K ScaleFirst M=64/2048/4096 evidence.

The source bundle is verified against its published ``bundle.json`` first.
Raw logs, plans, policies, generated manifests, model projections and results
are copied.  Build binaries are deliberately omitted but remain named and
hash-bound in ``archive.json`` so the archive is compact and honest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any


BUNDLE_SCHEMA = "quactlize.scalefirst_q4k_real_shapes_bundle.v1"
ARCHIVE_SCHEMA = "quactlize.scalefirst_q4k_prefill_archive.v1"
PREFILL_M = {64, 2048, 4096}


class ArchiveError(ValueError):
    pass


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def should_copy(relative: pathlib.PurePosixPath) -> bool:
    if relative.parts[0] == "build":
        return False
    if relative.parts[0] == "generated":
        return relative.name == "manifest.json"
    return True


def archive(source: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise ArchiveError("source bundle must be a real directory")
    if output.exists():
        raise ArchiveError(f"refusing existing archive {output}")
    bundle_path = source / "bundle.json"
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise ArchiveError("source lacks a regular bundle.json")
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ArchiveError("source is not a completed Q4_K real-shape bundle")
    members = bundle.get("files")
    if not isinstance(members, dict) or not members:
        raise ArchiveError("source bundle member census is empty")
    plan_relative = "plan.json"
    if plan_relative not in members:
        raise ArchiveError("source bundle does not bind plan.json")
    plan_path = source / plan_relative
    if digest(plan_path) != members[plan_relative]:
        raise ArchiveError("source plan hash differs")
    plan = json.loads(plan_path.read_text())
    shapes = plan.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ArchiveError("source plan has no shapes")
    observed_m = {int(shape["m"]) for shape in shapes}
    if observed_m != PREFILL_M:
        raise ArchiveError(
            f"source is not the exact M=64/2048/4096 prefill scope: {sorted(observed_m)}")

    output.mkdir(parents=True)
    copied: dict[str, str] = {}
    omitted: dict[str, dict[str, str]] = {}
    for name, expected in sorted(members.items()):
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArchiveError(f"unsafe source member {name!r}")
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ArchiveError(f"source member is absent/non-regular: {name}")
        actual = digest(path)
        if actual != expected:
            raise ArchiveError(f"source member hash differs: {name}")
        if should_copy(relative):
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if digest(destination) != expected:
                raise ArchiveError(f"archive copy hash differs: {name}")
            copied[name] = expected
        else:
            omitted[name] = {
                "sha256": expected,
                "reason": "BUILD_BINARY_OR_REGENERABLE_GENERATED_SOURCE",
            }
    shutil.copy2(bundle_path, output / "source-bundle.json")
    source_bundle_sha = digest(bundle_path)
    copied["source-bundle.json"] = source_bundle_sha
    document = {
        "schema": ARCHIVE_SCHEMA,
        "scope": {
            "format": "Q4_K", "route": "ScaleFirst prefill",
            "M": sorted(PREFILL_M),
            "decode_results_included": False,
        },
        "source": {
            "path_at_archive_time": str(source),
            "bundle_sha256": source_bundle_sha,
            "git_sha": bundle.get("git_sha"),
            "bound_member_count": len(members),
        },
        "copied_file_count": len(copied),
        "copied_files": copied,
        "omitted_bound_files": omitted,
        "rebuild_note": "timing/results authority retained; binaries intentionally omitted",
    }
    atomic_json(output / "archive.json", document)
    return document


def verify(output: pathlib.Path) -> dict[str, Any]:
    output = output.resolve(strict=True)
    manifest = output / "archive.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ArchiveError("archive.json is absent/non-regular")
    document = json.loads(manifest.read_text())
    if document.get("schema") != ARCHIVE_SCHEMA or \
            document.get("scope", {}).get("M") != sorted(PREFILL_M):
        raise ArchiveError("archive schema/scope differs")
    files = document.get("copied_files")
    if not isinstance(files, dict) or len(files) != document.get("copied_file_count"):
        raise ArchiveError("archive copied-file denominator differs")
    for name, expected in files.items():
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArchiveError(f"unsafe archive member {name!r}")
        path = output / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ArchiveError(f"archive member differs: {name}")
    return document


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        source, output = root / "source", root / "archive"
        (source / "build/a64").mkdir(parents=True)
        (source / "results").mkdir()
        plan = {"shapes": [{"m": m} for m in sorted(PREFILL_M)]}
        (source / "plan.json").write_text(json.dumps(plan))
        (source / "results/summary.json").write_text("{}")
        (source / "build/a64/binary").write_bytes(b"binary")
        names = ("plan.json", "results/summary.json", "build/a64/binary")
        bundle = {"schema": BUNDLE_SCHEMA, "git_sha": "a" * 40,
                  "files": {name: digest(source / name) for name in names}}
        (source / "bundle.json").write_text(json.dumps(bundle))
        result = archive(source, output)
        if (output / "build/a64/binary").exists() or \
                not (output / "results/summary.json").is_file() or \
                "build/a64/binary" not in result["omitted_bound_files"]:
            raise AssertionError("archive include/exclude contract differs")
        verify(output)
        planted = json.loads(json.dumps(plan))
        planted["shapes"].pop()
        (source / "plan.json").write_text(json.dumps(planted))
        bundle["files"]["plan.json"] = digest(source / "plan.json")
        (source / "bundle.json").write_text(json.dumps(bundle))
        try:
            archive(source, root / "bad")
        except ArchiveError:
            pass
        else:
            raise AssertionError("missing prefill M stayed green")
    print("[q4k-prefill-archive:self-test] PASS: exact M scope, full source hash "
          "verification, compact evidence copy, and missing-M negative")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verify", type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        elif args.verify is not None:
            result = verify(args.verify)
            print(f"[q4k-prefill-archive] VERIFIED copied={result['copied_file_count']} "
                  f"archive={args.verify}")
        else:
            if args.source is None or args.output is None:
                raise ArchiveError("--source and --output are required")
            result = archive(args.source, args.output)
            print(f"[q4k-prefill-archive] PASS copied={result['copied_file_count']} "
                  f"omitted={len(result['omitted_bound_files'])} output={args.output}")
        return 0
    except (ArchiveError, AssertionError, KeyError, OSError, TypeError,
            json.JSONDecodeError) as error:
        print(f"[q4k-prefill-archive] FAIL: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

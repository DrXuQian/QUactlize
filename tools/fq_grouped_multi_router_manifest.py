#!/usr/bin/env python3
"""Strict manifest verifier for grouped multi-router prebuilt bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

SCHEMA = "quactlize.fq-grouped-multi-router-prebuilt.v1"
SOURCE_PATHS = {
    "build.sh",
    "CMakeLists.txt",
    "quactlize/csrc/CMakeLists.txt.in",
    "quactlize/csrc/fq_kquant_layout_perf.cmake.in",
    "quactlize/csrc/fq_grouped_multi_router_perf.cmake.in",
    "benchmarks/test_fq_kquant_layout_perf.cu",
    "benchmarks/test_fq_grouped_multi_router_perf.cu",
    "quactlize/csrc/device/ppu_dense_backend.cu",
    "quactlize/include/ppu_grouped_configs.inc",
    "quactlize/include/ppu_q4_kpack4_shipping_policy.hpp",
    "tools/fq_grouped_multi_router.py",
    "tools/plan_fq_grouped_multi_router.py",
    "tools/analyze_fq_grouped_multi_router.py",
    "tools/fq_grouped_multi_router_manifest.py",
    "tools/build_fq_grouped_multi_router_bundle.sh",
    "tools/run_fq_grouped_multi_router_prebuilt_box.sh",
    "tools/probe_box_identity.py",
    "ci/check_fq_grouped_multi_router.py",
}


class ManifestError(ValueError):
    pass


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def load(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ManifestError("manifest root is not an object")
    return value


def shape(value: dict) -> None:
    if set(value) != {
        "schema",
        "source_sha",
        "source_files",
        "submodules",
        "sdk",
        "build",
        "binaries",
    }:
        raise ManifestError("manifest top-level denominator differs")
    if value["schema"] != SCHEMA or not re.fullmatch(
        r"[0-9a-f]{40}", value["source_sha"]
    ):
        raise ManifestError("schema/source identity differs")
    if set(value["source_files"]) != SOURCE_PATHS:
        raise ManifestError("source-file denominator differs")
    if any(not re.fullmatch(r"[0-9a-f]{64}", item)
           for item in value["source_files"].values()):
        raise ManifestError("source-file digest is malformed")
    if not isinstance(value["submodules"], dict) or not value["submodules"] or \
       any(not isinstance(path, str) or not path or path.startswith("/") or
           not re.fullmatch(r"[0-9a-f]{40}", commit)
           for path, commit in value["submodules"].items()):
        raise ManifestError("submodule authority is malformed")
    if set(value["sdk"]) != {"release_sha256", "compiler_sha256", "inspector_sha256"}:
        raise ManifestError("SDK denominator differs")
    if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in value["sdk"].values()):
        raise ManifestError("SDK digest differs")
    if value["build"] != {
        "target": "test_fq_grouped_multi_router_perf",
        "arch": "ppu0010",
        "qtypes": [10, 11, 12, 13, 14],
    }:
        raise ManifestError("build denominator differs")
    if set(value["binaries"]) != {"10", "11", "12", "13", "14"}:
        raise ManifestError("binary denominator differs")
    for row in value["binaries"].values():
        if set(row) != {"path", "sha256", "library_path", "library_sha256"}:
            raise ManifestError("binary row denominator differs")
        if any(not isinstance(row[name], str) or not row[name] or
               row[name].startswith("/") or
               ".." in pathlib.PurePosixPath(row[name]).parts
               for name in ("path", "library_path")):
            raise ManifestError("binary payload path is malformed")
        if any(not re.fullmatch(r"[0-9a-f]{64}", row[name])
               for name in ("sha256", "library_sha256")):
            raise ManifestError("binary payload digest is malformed")


def verify(
    bundle: pathlib.Path,
    manifest_path: pathlib.Path,
    source_root: pathlib.Path,
    release: pathlib.Path,
) -> None:
    value = load(manifest_path)
    shape(value)
    head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if value["source_sha"] != head:
        raise ManifestError("manifest source SHA differs from checkout")
    submodules = {}
    for line in subprocess.check_output(
        ["git", "-C", str(source_root), "submodule", "status", "--recursive"], text=True
    ).splitlines():
        if not line.startswith(" "):
            raise ManifestError("recursive submodule is not clean/exact")
        fields = line.strip().split()
        submodules[fields[1]] = fields[0]
    if value["submodules"] != submodules:
        raise ManifestError("submodule map differs")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(release) != value["sdk"]["release_sha256"]:
        raise ManifestError("box SDK release differs from build release")
    for path, expected in value["source_files"].items():
        source = (source_root / path).resolve()
        if (
            source_root not in source.parents
            or not source.is_file()
            or digest(source) != expected
        ):
            raise ManifestError(f"source input differs: {path}")
    for row in value["binaries"].values():
        for path_key, digest_key, executable in (
            ("path", "sha256", True),
            ("library_path", "library_sha256", False),
        ):
            raw = bundle / row[path_key]
            resolved = raw.resolve()
            if bundle not in resolved.parents or not raw.is_file() or raw.is_symlink():
                raise ManifestError(f"payload path invalid: {raw}")
            if executable and not os.access(raw, os.X_OK):
                raise ManifestError(f"payload is not executable: {raw}")
            if (
                not re.fullmatch(r"[0-9a-f]{64}", row[digest_key])
                or digest(resolved) != row[digest_key]
            ):
                raise ManifestError(f"payload digest differs: {raw}")


def self_test() -> None:
    digest = "0" * 64
    valid = {
        "schema": SCHEMA,
        "source_sha": "0" * 40,
        "source_files": {path: digest for path in SOURCE_PATHS},
        "submodules": {"third_party/example": "0" * 40},
        "sdk": {
            key: digest
            for key in ("release_sha256", "compiler_sha256", "inspector_sha256")
        },
        "build": {
            "target": "test_fq_grouped_multi_router_perf",
            "arch": "ppu0010",
            "qtypes": [10, 11, 12, 13, 14],
        },
        "binaries": {
            str(q): {
                "path": f"bin/q{q}/test",
                "sha256": digest,
                "library_path": f"bin/q{q}/lib.so",
                "library_sha256": digest,
            }
            for q in (10, 11, 12, 13, 14)
        },
    }
    shape(valid)
    sdk_extra = dict(valid["sdk"], extra=digest)
    row_extra = dict(valid["binaries"]["10"], extra=1)
    binaries_extra = dict(valid["binaries"], **{"10": row_extra})
    plants = [
        dict(valid, extra=1),
        dict(valid, binaries={"10": valid["binaries"]["10"]}),
        dict(valid, sdk=sdk_extra),
        dict(valid, binaries=binaries_extra),
    ]
    for plant in plants:
        try:
            shape(plant)
        except ManifestError:
            pass
        else:
            raise AssertionError("manifest denominator negative stayed green")
    try:
        json.loads('{"schema":1,"schema":2}', object_pairs_hook=no_duplicates)
    except ManifestError:
        pass
    else:
        raise AssertionError("duplicate-key negative stayed green")
    print(
        "[fq-grouped-multi-router-manifest:self-test] PASS exact top/nested denominator; extra/duplicate RED"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    check = sub.add_parser("verify")
    check.add_argument("--bundle", type=pathlib.Path, required=True)
    check.add_argument("--manifest", type=pathlib.Path, required=True)
    check.add_argument("--source-root", type=pathlib.Path, required=True)
    check.add_argument("--release", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            verify(
                args.bundle.resolve(),
                args.manifest,
                args.source_root.resolve(),
                args.release,
            )
        return 0
    except (
        AssertionError,
        ManifestError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"[fq-grouped-multi-router-manifest] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

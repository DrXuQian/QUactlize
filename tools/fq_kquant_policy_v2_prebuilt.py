#!/usr/bin/env python3
"""Create and verify the source-bound Q12 policy-v2 prebuilt manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "quactlize.fq-kquant-policy-v2-prebuilt.v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
DEFINITIONS = ["PPU_PACKED_SCALE=1", "PPU_PACKED_FORMAT=0",
               "QUACTLIZE_DENSE_ONLY=12"]
BUILD_INPUTS = (
    "build.sh", "CMakeLists.txt", "quactlize/csrc/CMakeLists.txt.in",
    "benchmarks/test_fq_kquant_layout_perf.cu",
    "quactlize/csrc/device/ppu_backend.cu",
    "quactlize/csrc/device/ppu_dense_backend.cu",
    "quactlize/include/ppu_dense_configs.inc",
    "quactlize/include/ppu_grouped_configs.inc",
    "quactlize/include/ppu_q4_kpack4_shipping_policy.hpp",
    "quactlize/include/kquant_kpack_offline.hpp",
    "tools/plan_fq_kquant_policy_v2.py",
    "tools/analyze_fq_kquant_policy_v2.py",
    "tools/fq_kquant_policy_v2_prebuilt.py",
    "tools/build_fq_kquant_policy_v2_prebuilt.sh",
    "tools/run_fq_kquant_kpack_perf_box.sh",
    "tools/run_fq_kquant_policy_v2_box.sh",
)


class ManifestError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path, executable: bool = False) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ManifestError(f"not a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ManifestError(f"not executable: {path}")


def release_version(path: Path) -> str:
    regular(path)
    versions = [line.split(":", 1)[1].strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("version:")]
    if len(versions) != 1 or not versions[0]:
        raise ManifestError(f"SDK release receipt is malformed: {path}")
    return versions[0]


def run(*args: str, cwd: Path = ROOT) -> str:
    completed = subprocess.run(args, cwd=cwd, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise ManifestError(f"command failed ({completed.returncode}): {' '.join(args)}")
    return completed.stdout.rstrip()


def submodules(root: Path) -> list[dict[str, str]]:
    rows = []
    text = run("git", "submodule", "status", "--recursive", cwd=root)
    for line in text.splitlines():
        match = re.fullmatch(r" ([0-9a-f]{40}) ([^ ]+)(?: .*)?", line)
        if not match:
            raise ManifestError(f"submodule is absent/dirty/malformed: {line!r}")
        if run("git", "status", "--porcelain", "--untracked-files=all",
               cwd=root / match.group(2)):
            raise ManifestError(f"submodule worktree is dirty: {match.group(2)}")
        rows.append({"path": match.group(2), "commit": match.group(1)})
    if not rows:
        raise ManifestError("recursive submodule authority is empty")
    return rows


def file_record(path: Path, relative: str, executable: bool = False) -> dict[str, object]:
    regular(path, executable)
    return {"path": relative, "size": path.stat().st_size, "sha256": sha256(path)}


def _unique_json(path: Path) -> object:
    def hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ManifestError(f"duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)


def create(bundle: Path, build: Path, sdk: Path, build_log: Path,
           cmake_log: Path, build_make: Path) -> dict[str, object]:
    bundle = bundle.resolve(); build = build.resolve(); sdk = sdk.resolve()
    binary_src = next(build.rglob("test_fq_kquant_layout_perf"), None)
    library_src = next(build.rglob("libquactlize_ppu.so"), None)
    if binary_src is None or library_src is None:
        raise ManifestError("exact Q12 binary/library are missing")
    regular(binary_src, True); regular(library_src)
    regular(build_log); regular(cmake_log); regular(build_make)
    release = sdk / "release.yaml"; compiler = sdk / "bin/hgcc"
    inspector = sdk / "bin/hgobjdump"
    regular(release); regular(compiler, True); regular(inspector, True)
    source = run("git", "rev-parse", "HEAD")
    if not HEX40.fullmatch(source): raise ManifestError("source HEAD is malformed")
    if run("git", "status", "--porcelain", "--", *BUILD_INPUTS):
        raise ManifestError("build inputs are dirty or untracked")
    inputs = {path: sha256(ROOT / path) for path in BUILD_INPUTS}
    text = build_log.read_text(encoding="utf-8", errors="replace")
    make = build_make.read_text(encoding="utf-8", errors="replace")
    cmake = cmake_log.read_text(encoding="utf-8", errors="replace")
    required = ["[build.sh] FQ_KQUANT_PERF_QTYPE=12", *DEFINITIONS]
    if any(mark not in text + make + cmake for mark in required):
        raise ManifestError("build/CMake/target identity is incomplete")
    if "-DFQ_KQUANT_PERF_QTYPE=12" not in make or \
       "FullyQuantized K-quant layout perf: qtype=12 carrier=production-C-ABI" not in cmake:
        raise ManifestError("Q12 target identity did not reach CMake/build.make")
    bundle.mkdir(parents=True, exist_ok=False)
    binary = bundle / "test_fq_kquant_layout_perf"
    library = bundle / "libquactlize_ppu.so"
    binary.write_bytes(binary_src.read_bytes()); binary.chmod(0o755)
    library.write_bytes(library_src.read_bytes()); library.chmod(0o644)
    manifest = {
        "schema": SCHEMA,
        "source": {"commit": source, "submodules": submodules(ROOT),
                   "build_inputs": inputs},
        "sdk": {
            "release": dict(file_record(release, "release.yaml"),
                            version=release_version(release)),
            "compiler": dict(file_record(compiler, "bin/hgcc", True),
                             version=run(str(compiler), "--version").splitlines()[0]),
            "inspector": file_record(inspector, "bin/hgobjdump", True),
        },
        "build": {"target": "test_fq_kquant_layout_perf", "qtype": 12,
                  "arch": "ppu0010", "definitions": DEFINITIONS,
                  "build_log_sha256": sha256(build_log),
                  "cmake_log_sha256": sha256(cmake_log),
                  "build_make_sha256": sha256(build_make)},
        "artifacts": {
            "binary": file_record(binary, binary.name, True),
            "library": file_record(library, library.name),
        },
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify(bundle, ROOT, sdk)
    return manifest


def verify(bundle: Path, source_root: Path, sdk: Path,
           *, execution_sdk_compatible: bool = False) -> dict[str, object]:
    bundle = bundle.resolve(); source_root = source_root.resolve(); sdk = sdk.resolve()
    manifest_path = bundle / "manifest.json"; regular(manifest_path)
    value = _unique_json(manifest_path)
    if not isinstance(value, dict) or set(value) != {"schema", "source", "sdk", "build", "artifacts"}:
        raise ManifestError("manifest top-level shape differs")
    if value["schema"] != SCHEMA: raise ManifestError("manifest schema differs")
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {"commit", "submodules", "build_inputs"}:
        raise ManifestError("source authority shape differs")
    if source.get("commit") != run("git", "rev-parse", "HEAD", cwd=source_root):
        raise ManifestError("source commit differs from runner checkout")
    if source.get("submodules") != submodules(source_root):
        raise ManifestError("submodule authority differs")
    inputs = source.get("build_inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(BUILD_INPUTS):
        raise ManifestError("build-input denominator differs")
    for name, digest in inputs.items():
        if not HEX64.fullmatch(digest) or sha256(source_root / name) != digest:
            raise ManifestError(f"build input differs: {name}")
    build = value["build"]
    if not isinstance(build, dict): raise ManifestError("build authority is not an object")
    if build != {**build, "target": "test_fq_kquant_layout_perf", "qtype": 12,
                 "arch": "ppu0010", "definitions": DEFINITIONS}:
        raise ManifestError("build target identity differs")
    if set(build) != {"target", "qtype", "arch", "definitions", "build_log_sha256",
                      "cmake_log_sha256", "build_make_sha256"} or \
       any(not HEX64.fullmatch(build[name]) for name in
           ("build_log_sha256", "cmake_log_sha256", "build_make_sha256")):
        raise ManifestError("build evidence shape differs")
    sdk_rows = value["sdk"]
    if not isinstance(sdk_rows, dict) or set(sdk_rows) != {"release", "compiler", "inspector"}:
        raise ManifestError("SDK authority shape differs")
    sdk_specs = (("release", "release.yaml", False, True),
                 ("compiler", "bin/hgcc", True, True),
                 ("inspector", "bin/hgobjdump", True, False))
    for name, relative, executable, has_version in sdk_specs:
        row = sdk_rows.get(name)
        expected_fields = {"path", "size", "sha256"} | ({"version"} if has_version else set())
        if not isinstance(row, dict) or set(row) != expected_fields or \
           row.get("path") != relative or \
           not isinstance(row.get("size"), int) or row["size"] <= 0 or \
           not isinstance(row.get("sha256"), str) or not HEX64.fullmatch(row["sha256"]):
            raise ManifestError(f"SDK {name} manifest identity is malformed")
    release = sdk / "release.yaml"
    compiler = sdk / "bin/hgcc"
    inspector = sdk / "bin/hgobjdump"
    if release_version(release) != sdk_rows["release"].get("version"):
        raise ManifestError("execution SDK release differs from the build SDK")
    regular(inspector, True)
    if not execution_sdk_compatible:
        for name, path, executable in (("release", release, False),
                                       ("compiler", compiler, True),
                                       ("inspector", inspector, True)):
            regular(path, executable)
            row = sdk_rows[name]
            if row["size"] != path.stat().st_size or row["sha256"] != sha256(path):
                raise ManifestError(f"SDK {name} identity differs")
        compiler_version = run(str(compiler), "--version").splitlines()[0]
        if sdk_rows["compiler"].get("version") != compiler_version or \
           "stub" in compiler_version.lower():
            raise ManifestError("SDK compiler version differs or is a stub")
    if not isinstance(value["artifacts"], dict) or set(value["artifacts"]) != {"binary", "library"}:
        raise ManifestError("artifact denominator differs")
    for name, expected, executable in (("binary", "test_fq_kquant_layout_perf", True),
                                       ("library", "libquactlize_ppu.so", False)):
        row = value["artifacts"][name]
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"} or \
           row.get("path") != expected:
            raise ManifestError(f"{name} path differs")
        path = bundle / expected; regular(path, executable)
        if row.get("size") != path.stat().st_size or row.get("sha256") != sha256(path):
            raise ManifestError(f"{name} bytes differ")
    return value


def self_test() -> None:
    assert SCHEMA.endswith(".v1") and len(BUILD_INPUTS) == len(set(BUILD_INPUTS))
    base = {"schema": SCHEMA, "source": {}, "sdk": {}, "build": {}, "artifacts": {}}
    plants = []
    for field in ("schema", "source", "sdk", "build", "artifacts"):
        broken = copy.deepcopy(base); broken.pop(field); plants.append(broken)
    assert all(set(row) != {"schema", "source", "sdk", "build", "artifacts"}
               for row in plants)
    print("[fq-kquant-policy-v2-prebuilt:self-test] PASS five manifest-shape plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    check = commands.add_parser("verify")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--source-root", type=Path, default=ROOT)
    check.add_argument("--sdk", type=Path, required=True)
    check.add_argument("--execution-sdk-compatible", action="store_true",
                       help="require the recorded SDK release, not byte-identical build tools")
    make = commands.add_parser("create")
    for name in ("bundle", "build", "sdk", "build-log", "cmake-log", "build-make"):
        make.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test": self_test()
        elif args.command == "verify":
            verify(args.bundle, args.source_root, args.sdk,
                   execution_sdk_compatible=args.execution_sdk_compatible)
        else: create(args.bundle, args.build, args.sdk, args.build_log,
                     args.cmake_log, args.build_make)
        return 0
    except (ManifestError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"[fq-kquant-policy-v2-prebuilt] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

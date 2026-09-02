#!/usr/bin/env python3
"""Seal and verify the portable FullyQuantized Split-K reducer binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import plan_fq_splitk_reducer_lookup as reducer_plan


SCHEMA = "quactlize.fq-splitk-reducer-prebuilt.v1"
BUILD_AUTHORITY_SCHEMA = f"{SCHEMA}.build-authority.v1"
TARGET = "test_fq_splitk_reducer_lookup"
ARCH = "ppu0010"
BUILD_INPUTS = (
    "CMakeLists.txt",
    "build.sh",
    "benchmarks/test_fq_splitk_reducer_lookup.cu",
    "quactlize/csrc/CMakeLists.txt.in",
    "quactlize/include/actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp",
    "quactlize/include/dense_splitk_parallel_ppu.cuh",
    "tools/analyze_fq_splitk_reducer_lookup.py",
    "tools/box_identity_schema.py",
    "tools/box_identity_probe.cpp",
    "tools/build_fq_splitk_reducer_lookup_prebuilt.sh",
    "tools/fq_splitk_reducer_prebuilt.py",
    "tools/fully_quantized_kpack_discovery_matrix.py",
    "tools/plan_fq_kpack_route_optimal.py",
    "tools/plan_fq_splitk_reducer_lookup.py",
    "tools/probe_box_identity.py",
    "tools/kpack_global_build_preflight.py",
    "tools/run_fq_splitk_reducer_lookup_box.sh",
)
SDK_RUNTIME_FILES = (
    "lib/libhggc_wrapper.so",
    "lib/libhggc.so",
    "lib/libhg_wrapper.so",
    "lib/libhggcrt.13.0.so",
)
BUNDLE_FILES = (
    "test_fq_splitk_reducer_lookup",
    "box_identity_probe",
    "build-authority.json",
    "global-preflight.json",
    "reducer-plan.json",
    "fq_splitk_reducer_lookup_cases.inc",
    "ppu-elf-list.txt",
    "ppu-isa.txt",
    "build.log",
    "cmake.log",
    "build.make",
    "identity-probe-build.log",
)
HEX = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
HEX64 = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


def regular(path: Path, *, executable: bool = False) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ManifestError(f"cannot inspect {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ManifestError(f"not a plain regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ManifestError(f"not executable: {path}")


def run(argv: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        detail = "\n".join(result.stdout.splitlines()[-30:])
        raise ManifestError(
            f"command failed ({result.returncode}): {' '.join(argv)}\n{detail}")
    return result.stdout.rstrip()


def git(root: Path, *arguments: str) -> str:
    return run(["git", "-C", str(root), *arguments], cwd=root)


def submodules(root: Path) -> list[dict[str, str]]:
    rows = []
    for line in git(root, "submodule", "status", "--recursive").splitlines():
        match = re.fullmatch(r" ([0-9a-f]{40,64}) ([^ ]+)(?: .*)?", line)
        if not match:
            raise ManifestError(f"submodule status is absent/dirty: {line!r}")
        checkout = root / match.group(2)
        current = git(checkout, "rev-parse", "HEAD")
        if current != match.group(1) or git(
                checkout, "status", "--porcelain", "--untracked-files=all"):
            raise ManifestError(f"submodule checkout differs: {match.group(2)}")
        rows.append({"path": match.group(2), "commit": current})
    if not rows:
        raise ManifestError("recursive submodule authority is empty")
    return rows


def release_receipt(sdk: Path) -> tuple[Path, str]:
    yaml = sdk / "release.yaml"
    version_file = sdk / "VERSION.txt"
    if yaml.is_file() and not yaml.is_symlink():
        lines = [line.split(":", 1)[1].strip()
                 for line in yaml.read_text(encoding="utf-8").splitlines()
                 if line.startswith("version:")]
        if len(lines) != 1 or not lines[0]:
            raise ManifestError("SDK release.yaml has no unique version")
        return yaml, lines[0]
    if version_file.is_file() and not version_file.is_symlink():
        version = version_file.read_text(encoding="utf-8").strip()
        if not version or "\n" in version:
            raise ManifestError("SDK VERSION.txt is malformed")
        return version_file, version
    raise ManifestError("SDK has neither a plain release.yaml nor VERSION.txt")


def file_record(path: Path, relative: str, *, executable: bool = False) -> dict[str, Any]:
    regular(path, executable=executable)
    return {"path": relative, "size": path.stat().st_size,
            "sha256": sha256_file(path)}


def sdk_authority(sdk: Path) -> dict[str, Any]:
    sdk = sdk.resolve(strict=True)
    release, version = release_receipt(sdk)
    compiler = sdk / "bin/hgcc"
    inspector = sdk / "bin/hgobjdump"
    regular(compiler, executable=True)
    regular(inspector, executable=True)
    compiler_version = run([str(compiler), "--version"], cwd=sdk).splitlines()[0]
    if "stub" in compiler_version.lower():
        raise ManifestError("stub SDK compiler is forbidden")
    runtime = []
    for relative in SDK_RUNTIME_FILES:
        runtime.append(file_record(sdk / relative, relative, executable=True))
    return {
        "version": version,
        "release": file_record(release, str(release.relative_to(sdk))),
        "compiler": dict(file_record(compiler, "bin/hgcc", executable=True),
                         version=compiler_version),
        "inspector": file_record(inspector, "bin/hgobjdump", executable=True),
        "runtime": runtime,
    }


def source_authority(source_root: Path, *, require_clean: bool) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    commit = git(source_root, "rev-parse", "HEAD")
    tree = git(source_root, "rev-parse", "HEAD^{tree}")
    if not HEX.fullmatch(commit) or not HEX.fullmatch(tree):
        raise ManifestError("Git commit/tree authority is malformed")
    if require_clean:
        if git(source_root, "diff", "--name-only", "HEAD", "--", *BUILD_INPUTS):
            raise ManifestError("tracked reducer build inputs are dirty")
        untracked = git(source_root, "ls-files", "--others", "--exclude-standard",
                        "--", *BUILD_INPUTS)
        if untracked:
            raise ManifestError("untracked reducer build inputs are not committed")
    records = {}
    for relative in BUILD_INPUTS:
        path = source_root / relative
        regular(path, executable=relative.endswith(".sh"))
        records[relative] = {"size": path.stat().st_size,
                             "sha256": sha256_file(path)}
    return {"commit": commit, "tree": tree,
            "submodules": submodules(source_root), "build_inputs": records}


def build_authority(source_root: Path, sdk: Path) -> dict[str, Any]:
    return {
        "schema": BUILD_AUTHORITY_SCHEMA,
        "source": source_authority(source_root, require_clean=True),
        "sdk": sdk_authority(sdk),
        "build": {
            "target": TARGET, "arch": ARCH,
            "cmake_enable": "FQ_SPLITK_REDUCER_LOOKUP_ENABLE=ON",
            "plan_sha256": reducer_plan.digest(reducer_plan.materialize()),
        },
    }


def _write_new(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise ManifestError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_build_authority(output: Path, source_root: Path, sdk: Path) -> None:
    _write_new(output, build_authority(source_root, sdk))


def read_unique_json(path: Path) -> Any:
    regular(path)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot parse {path}: {error}") from error


def verify_build_authority(path: Path, source_root: Path, sdk: Path) -> dict[str, Any]:
    value = read_unique_json(path)
    expected = build_authority(source_root, sdk)
    if value != expected:
        raise ManifestError("build authority differs from current exact source/SDK")
    return value


def _copy(source: Path, destination: Path, *, executable: bool = False) -> dict[str, Any]:
    regular(source, executable=executable)
    if destination.exists() or destination.is_symlink():
        raise ManifestError(f"publish destination exists: {destination}")
    shutil.copyfile(source, destination)
    destination.chmod(0o755 if executable else 0o644)
    if sha256_file(source) != sha256_file(destination):
        raise ManifestError(f"published bytes differ: {destination.name}")
    return file_record(destination, destination.name, executable=executable)


def _validate_build_evidence(build_log: Path, cmake_log: Path,
                             build_make: Path, generated_plan: Path,
                             generated_include: Path) -> dict[str, Any]:
    for path in (build_log, cmake_log, build_make, generated_plan, generated_include):
        regular(path)
    plan = read_unique_json(generated_plan)
    reducer_plan.validate_plan(plan)
    expected_include = reducer_plan.cpp_include(plan)
    if generated_include.read_bytes() != expected_include:
        raise ManifestError("generated reducer include differs from canonical plan")
    cmake = cmake_log.read_text(encoding="utf-8", errors="replace")
    make = build_make.read_text(encoding="utf-8", errors="replace")
    log = build_log.read_text(encoding="utf-8", errors="replace")
    if "FQ Split-K reducer lookup: cases=1035" not in cmake or \
            "test_fq_splitk_reducer_lookup" not in make or \
            "fq_splitk_reducer_lookup_cases.inc" not in generated_include.name or \
            "repository-global checks reused from" not in log or \
            ("built:" not in log and "Built target test_fq_splitk_reducer_lookup" not in log):
        raise ManifestError("build/CMake/target evidence is incomplete")
    return {"plan_sha256": reducer_plan.digest(plan)}


def create(bundle: Path, source_root: Path, sdk: Path, build_authority_path: Path,
           binary: Path, generated_plan: Path, generated_include: Path,
           build_log: Path, cmake_log: Path, build_make: Path,
           global_preflight: Path, identity_probe: Path,
           identity_probe_build_log: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    source_root = source_root.resolve(strict=True)
    sdk = sdk.resolve(strict=True)
    authority = verify_build_authority(build_authority_path, source_root, sdk)
    evidence = _validate_build_evidence(
        build_log, cmake_log, build_make, generated_plan, generated_include)
    regular(binary, executable=True)
    regular(identity_probe, executable=True)
    regular(identity_probe_build_log)
    regular(global_preflight)
    if bundle.exists() or bundle.is_symlink():
        raise ManifestError(f"bundle already exists: {bundle}")
    bundle.mkdir(parents=True)
    try:
        artifacts = {
            "binary": _copy(binary, bundle / TARGET, executable=True),
            "identity_probe": _copy(
                identity_probe, bundle / "box_identity_probe", executable=True),
            "build_authority": _copy(
                build_authority_path, bundle / "build-authority.json"),
            "global_preflight": _copy(
                global_preflight, bundle / "global-preflight.json"),
            "plan": _copy(generated_plan, bundle / "reducer-plan.json"),
            "include": _copy(generated_include,
                             bundle / "fq_splitk_reducer_lookup_cases.inc"),
            "build_log": _copy(build_log, bundle / "build.log"),
            "cmake_log": _copy(cmake_log, bundle / "cmake.log"),
            "build_make": _copy(build_make, bundle / "build.make"),
            "identity_probe_build_log": _copy(
                identity_probe_build_log, bundle / "identity-probe-build.log"),
        }
        inspector = sdk / "bin/hgobjdump"
        elf = run([str(inspector), "--list-elf", str(binary)], cwd=source_root) + "\n"
        isa = run([str(inspector), "--dump-isa", str(binary)], cwd=source_root) + "\n"
        required = (
            "initialize_partials_kernel", "validate_output_kernel",
            "plant_output_fault_kernel", "PpuMixedInputSplitKParallelReductionKernel",
            "PpuMixedInputSplitKParallelM1FastReductionKernelILi2ELi2",
            "PpuMixedInputSplitKParallelM1FastReductionKernelILi2ELi4",
            "PpuMixedInputSplitKParallelM1FastReductionKernelILi2ELi8",
        )
        if any(marker not in elf for marker in required):
            raise ManifestError("linked PPU image lacks the exact reducer/validator kernels")
        for name, text in (("ppu-elf-list.txt", elf), ("ppu-isa.txt", isa)):
            path = bundle / name
            path.write_text(text, encoding="utf-8")
            artifacts["elf_list" if name.startswith("ppu-elf") else "isa"] = \
                file_record(path, name)
        manifest = {
            "schema": SCHEMA,
            "source": authority["source"],
            "sdk": authority["sdk"],
            "build": {
                **authority["build"],
                "build_authority_sha256": sha256_file(build_authority_path),
                "plan_sha256": evidence["plan_sha256"],
                "measurement": {
                    "rounds": 3, "warmups": 3, "samples": 11,
                    "schedule_seed_schema": reducer_plan.SCHEDULE_SEED_SCHEMA,
                    "round_seeds": [f"0x{seed:016x}" for seed in reducer_plan.ROUND_SEEDS],
                    "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
                    "top_n": None, "point_estimate_pruning": False,
                },
            },
            "artifacts": artifacts,
        }
        _write_new(bundle / "manifest.json", manifest)
        verify(bundle, source_root, sdk)
        for path in bundle.iterdir():
            path.chmod(path.stat().st_mode & ~0o222)
        return manifest
    except Exception:
        # The caller publishes through a private stage. Preserve it for
        # inspection; never silently replace a partially attested bundle.
        raise


def _record_matches(row: Any, path: Path, expected_path: str,
                    *, executable: bool = False) -> None:
    if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"} or \
            row.get("path") != expected_path or \
            isinstance(row.get("size"), bool) or not isinstance(row.get("size"), int) or \
            row["size"] <= 0 or not isinstance(row.get("sha256"), str) or \
            not HEX64.fullmatch(row["sha256"]):
        raise ManifestError(f"artifact record is malformed: {expected_path}")
    regular(path, executable=executable)
    if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
        raise ManifestError(f"artifact bytes differ: {expected_path}")


def verify(bundle: Path, source_root: Path, sdk: Path) -> dict[str, Any]:
    bundle = bundle.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    sdk = sdk.resolve(strict=True)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ManifestError("bundle is not a plain directory")
    names = {path.name for path in bundle.iterdir()}
    if names != {"manifest.json", *BUNDLE_FILES}:
        raise ManifestError(f"bundle file denominator differs: {sorted(names)}")
    value = read_unique_json(bundle / "manifest.json")
    if not isinstance(value, dict) or set(value) != {
            "schema", "source", "sdk", "build", "artifacts"} or \
            value.get("schema") != SCHEMA:
        raise ManifestError("prebuilt manifest top-level shape differs")
    live_source = source_authority(source_root, require_clean=False)
    if value["source"] != live_source:
        raise ManifestError("bundle source authority differs from runner checkout")
    live_sdk = sdk_authority(sdk)
    if value["sdk"] != live_sdk:
        raise ManifestError("bundle SDK authority differs from execution SDK")
    build = value["build"]
    expected_keys = {
        "target", "arch", "cmake_enable", "plan_sha256",
        "build_authority_sha256", "measurement"}
    if not isinstance(build, dict) or set(build) != expected_keys or \
            build.get("target") != TARGET or build.get("arch") != ARCH or \
            build.get("cmake_enable") != "FQ_SPLITK_REDUCER_LOOKUP_ENABLE=ON" or \
            not HEX64.fullmatch(str(build.get("plan_sha256", ""))) or \
            not HEX64.fullmatch(str(build.get("build_authority_sha256", ""))):
        raise ManifestError("bundle build identity differs")
    expected_measurement = {
        "rounds": 3, "warmups": 3, "samples": 11,
        "schedule_seed_schema": reducer_plan.SCHEDULE_SEED_SCHEMA,
        "round_seeds": [f"0x{seed:016x}" for seed in reducer_plan.ROUND_SEEDS],
        "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
        "top_n": None, "point_estimate_pruning": False,
    }
    if build["measurement"] != expected_measurement:
        raise ManifestError("bundle measurement contract differs")
    artifact_keys = {
        "binary", "identity_probe", "build_authority", "plan", "include",
        "elf_list", "isa", "global_preflight", "build_log", "cmake_log",
        "build_make", "identity_probe_build_log"}
    artifacts = value["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != artifact_keys:
        raise ManifestError("bundle artifact denominator differs")
    specs = {
        "binary": (TARGET, True),
        "identity_probe": ("box_identity_probe", True),
        "build_authority": ("build-authority.json", False),
        "global_preflight": ("global-preflight.json", False),
        "plan": ("reducer-plan.json", False),
        "include": ("fq_splitk_reducer_lookup_cases.inc", False),
        "elf_list": ("ppu-elf-list.txt", False),
        "isa": ("ppu-isa.txt", False),
        "build_log": ("build.log", False),
        "cmake_log": ("cmake.log", False),
        "build_make": ("build.make", False),
        "identity_probe_build_log": ("identity-probe-build.log", False),
    }
    for key, (name, executable) in specs.items():
        _record_matches(artifacts[key], bundle / name, name, executable=executable)
    if artifacts["build_authority"]["sha256"] != build["build_authority_sha256"]:
        raise ManifestError("copied build authority digest differs")
    recorded_authority = read_unique_json(bundle / "build-authority.json")
    if recorded_authority != {
            "schema": BUILD_AUTHORITY_SCHEMA,
            "source": value["source"], "sdk": value["sdk"],
            "build": {key: build[key] for key in (
                "target", "arch", "cmake_enable", "plan_sha256")}}:
        raise ManifestError("copied build authority content differs")
    plan = read_unique_json(bundle / "reducer-plan.json")
    reducer_plan.validate_plan(plan)
    if reducer_plan.digest(plan) != build["plan_sha256"] or \
            reducer_plan.cpp_include(plan) != \
            (bundle / "fq_splitk_reducer_lookup_cases.inc").read_bytes():
        raise ManifestError("bundle plan/include/build binding differs")
    _validate_build_evidence(
        bundle / "build.log", bundle / "cmake.log", bundle / "build.make",
        bundle / "reducer-plan.json", bundle / "fq_splitk_reducer_lookup_cases.inc")
    elf = (bundle / "ppu-elf-list.txt").read_text(encoding="utf-8")
    if "PpuMixedInputSplitKParallelM1FastReductionKernelILi2ELi8" not in elf or \
            "validate_output_kernel" not in elf:
        raise ManifestError("recorded PPU ELF evidence differs")
    return value


def self_test() -> None:
    if len(BUILD_INPUTS) != len(set(BUILD_INPUTS)) or \
            len(BUNDLE_FILES) != len(set(BUNDLE_FILES)):
        raise AssertionError("authority denominators contain duplicates")
    plan = reducer_plan.materialize()
    if len(plan["cases"]) != 1035 or len(reducer_plan.ROUND_SEEDS) != 3:
        raise AssertionError("reducer plan contract drifted")
    print(
        "[fq-splitk-reducer-prebuilt:self-test] PASS source=HEAD+tree+inputs "
        "sdk=compiler+inspector+runtime plan=1035 rounds=3 portable=1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    authority = commands.add_parser("write-build-authority")
    authority.add_argument("--output", type=Path, required=True)
    authority.add_argument("--source-root", type=Path, default=ROOT)
    authority.add_argument("--sdk", type=Path, required=True)
    verify_authority = commands.add_parser("verify-build-authority")
    verify_authority.add_argument("--file", type=Path, required=True)
    verify_authority.add_argument("--source-root", type=Path, default=ROOT)
    verify_authority.add_argument("--sdk", type=Path, required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--bundle", type=Path, required=True)
    create_parser.add_argument("--source-root", type=Path, default=ROOT)
    create_parser.add_argument("--sdk", type=Path, required=True)
    create_parser.add_argument("--build-authority", type=Path, required=True)
    for name in ("binary", "generated-plan", "generated-include", "build-log",
                 "cmake-log", "build-make", "global-preflight",
                 "identity-probe", "identity-probe-build-log"):
        create_parser.add_argument(f"--{name}", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--source-root", type=Path, default=ROOT)
    verify_parser.add_argument("--sdk", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "write-build-authority":
            write_build_authority(args.output, args.source_root, args.sdk)
        elif args.command == "verify-build-authority":
            verify_build_authority(args.file, args.source_root, args.sdk)
        elif args.command == "create":
            create(args.bundle, args.source_root, args.sdk, args.build_authority,
                   args.binary, args.generated_plan, args.generated_include,
                   args.build_log, args.cmake_log, args.build_make,
                   args.global_preflight, args.identity_probe,
                   args.identity_probe_build_log)
        else:
            verify(args.bundle, args.source_root, args.sdk)
        return 0
    except (ManifestError, reducer_plan.PlanError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"[fq-splitk-reducer-prebuilt] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

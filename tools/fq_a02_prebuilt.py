#!/usr/bin/env python3
"""Seal/verify A02 prebuilt artifacts and the referenced A01 product result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "quactlize.fq-a02-typed-diagnostics-prebuilt"
SCHEMA_VERSION = 1
A01_SCHEMA = "quactlize.prebuilt-six-library-box-gate"
A01_SCHEMA_VERSION = 2
HEX64 = re.compile(r"[0-9a-f]{64}")
PRODUCT = ["fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0"]
NONPRODUCT = [
    "fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1",
    "fq_tc_q11_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap0",
    "fq_tc_q11_a64_tm8_tn64_tk256_wm8_wn16_s2_bc1_ap0",
]
FORMAT_NAMES = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
LIBRARY_ROLES = {"default", "fmt0", "fmt1", "fmt2", "fmt3", "fmt4"}
FORMAT_IDENTITIES = {
    "Q2_K": (10, "fmt2", 2),
    "Q3_K": (11, "fmt3", 3),
    "Q4_K": (12, "fmt0", 0),
    "Q5_K": (13, "fmt1", 1),
    "Q6_K": (14, "fmt4", 4),
}
Q4_PRODUCT_POLICY = {
    "dense": {"packed_a_rows": 0, "bchunk": 0,
              "metadata_publication": "InterleavedHalf2"},
    "grouped": {"packed_a_rows": 0, "bchunk": 0,
                "metadata_publication": "SeparateHalfPlanes"},
}
Q4_PER_FORMAT_POLICY = {
    **Q4_PRODUCT_POLICY,
    "authority": "manifest-bound runtime source plus exact selected-config ABI",
}
INPUTS = (
    "build.sh",
    "quactlize/csrc/CMakeLists.txt.in",
    "quactlize/csrc/fq_internal_sweep.cmake.in",
    "benchmarks/test_fully_quantized_internal_sweep.cu",
    "benchmarks/fully_quantized_splitk_producer_bench.hpp",
    "benchmarks/fully_quantized_splitk_producer_unit.inc",
    "tools/fully_quantized_internal_matrix.py",
    "tools/gen_fully_quantized_splitk_producer_units.py",
    "tools/select_fq_a02_typed_diagnostics.py",
    "tools/check_fq_a02_typed_diagnostics.py",
    "tools/fq_a02_prebuilt.py",
    "tools/build_fq_a02_prebuilt.sh",
    "tools/run_fq_a02_prebuilt_box.sh",
)


class Error(ValueError):
    pass


def strict_json(path: Path) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Error(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        raise Error(f"malformed JSON: {path}: {exc}") from exc


def require_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise Error(f"{label} keys differ")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(path: Path, executable: bool = False) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise Error(f"not a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise Error(f"not executable: {path}")


def run(*args: str, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise Error(f"command failed ({completed.returncode}): {' '.join(args)}")
    return completed.stdout.rstrip()


def submodules(root: Path) -> list[dict[str, str]]:
    result = []
    for line in run("git", "submodule", "status", "--recursive", cwd=root).splitlines():
        match = re.fullmatch(r" ([0-9a-f]{40}) ([^ ]+)(?: .*)?", line)
        if not match:
            raise Error(f"submodule identity differs: {line!r}")
        if run("git", "status", "--porcelain", "--untracked-files=all",
               cwd=root / match.group(2)):
            raise Error(f"submodule worktree is dirty: {match.group(2)}")
        result.append({"commit": match.group(1), "path": match.group(2)})
    if not result:
        raise Error("submodule authority is empty")
    return result


def sdk_release(sdk: Path) -> str:
    release = sdk / "release.yaml"
    regular(release)
    versions = [line.split(":", 1)[1].strip()
                for line in release.read_text(encoding="utf-8").splitlines()
                if line.startswith("version:")]
    if len(versions) != 1 or not versions[0]:
        raise Error("SDK release is malformed")
    return versions[0]


def record(path: Path, name: str, executable: bool = False) -> dict[str, object]:
    regular(path, executable)
    return {"path": name, "size": path.stat().st_size, "sha256": sha256(path)}


def validate_classification(value: object) -> None:
    expected = {"product_shipping": PRODUCT,
                "typed_diagnostic_nonproduct": NONPRODUCT}
    if value != expected:
        raise Error("classification differs")


def current_build_authority(root: Path, sdk: Path) -> dict[str, object]:
    root, sdk = root.resolve(), sdk.resolve()
    compiler, inspector = sdk / "bin/hgcc", sdk / "bin/hgobjdump"
    regular(compiler, True)
    regular(inspector, True)
    return {
        "schema": "quactlize.fq-a02-build-authority.v1",
        "source": {
            "commit": run("git", "rev-parse", "HEAD", cwd=root),
            "submodules": submodules(root),
            "inputs": {name: sha256(root / name) for name in INPUTS},
        },
        "sdk": {
            "release": sdk_release(sdk),
            "compiler_sha256": sha256(compiler),
            "inspector_sha256": sha256(inspector),
        },
        "build": {
            "arch": "ppu0010",
            "q4_target": "test_fully_quantized_internal_sweep",
            "q3_target": "test_fq_a02_q3_bchunk_aggregate",
        },
    }


def write_build_authority(path: Path, root: Path, sdk: Path) -> None:
    if path.exists() or path.is_symlink():
        raise Error("refusing existing build authority")
    path.write_text(
        json.dumps(current_build_authority(root, sdk), indent=2,
                   sort_keys=True) + "\n", encoding="utf-8")


def verify_build_authority(path: Path, root: Path, sdk: Path) -> None:
    regular(path)
    recorded = require_keys(
        strict_json(path), {"schema", "source", "sdk", "build"},
        "build authority")
    if recorded != current_build_authority(root, sdk):
        raise Error("build authority differs from source/submodules/inputs/SDK")


def create(bundle: Path, stage: Path, sdk: Path, logs: list[Path],
           cmake_logs: list[Path], build_makes: list[Path]) -> dict[str, object]:
    bundle, stage, sdk = bundle.resolve(), stage.resolve(), sdk.resolve()
    if run("git", "status", "--porcelain", "--", *INPUTS):
        raise Error("A02 authority is dirty or untracked")
    compiler, inspector = sdk / "bin/hgcc", sdk / "bin/hgobjdump"
    regular(compiler, True); regular(inspector, True)
    evidence_sets = ((logs, "build logs"), (cmake_logs, "CMake logs"),
                     (build_makes, "build.make files"))
    for paths, label in evidence_sets:
        if len(paths) != 2:
            raise Error(f"exactly two {label} are required")
        for path in paths:
            regular(path)
    bundle.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, object]] = {}
    specs = (
        ("q4_binary", "test_fully_quantized_internal_sweep", True),
        ("q3_binary", "test_fq_a02_q3_bchunk_aggregate", True),
        ("q4_isa", "q4.isa.txt", False),
        ("q3_isa", "q3.isa.txt", False),
    )
    for key, name, executable in specs:
        source, target = stage / name, bundle / name
        regular(source, executable)
        if source.stat().st_size <= 0:
            raise Error(f"empty artifact: {name}")
        target.write_bytes(source.read_bytes())
        target.chmod(0o755 if executable else 0o644)
        artifacts[key] = record(target, name, executable)
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "commit": run("git", "rev-parse", "HEAD"),
            "submodules": submodules(ROOT),
            "inputs": {name: sha256(ROOT / name) for name in INPUTS},
        },
        "sdk": {
            "release": sdk_release(sdk),
            "runtime_abi": ["hg_wrapper", "hggc_wrapper", "hggcrt1", "hggc"],
            "build_compiler_sha256": sha256(compiler),
            "build_inspector_sha256": sha256(inspector),
        },
        "build": {
            "targets": ["test_fully_quantized_internal_sweep",
                        "test_fq_a02_q3_bchunk_aggregate"],
            "q4": "q12/A64/bc0/AP0+AP1",
            "q3": "q11/A64/bc0+effective-bc1/AP0",
            "logs": {log.name: sha256(log) for log in logs},
            "cmake_logs": {f"{index}-{path.name}": sha256(path)
                           for index, path in enumerate(cmake_logs)},
            "build_makes": {f"{index}-{path.name}": sha256(path)
                            for index, path in enumerate(build_makes)},
        },
        "artifacts": artifacts,
        "classification": {
            "product_shipping": PRODUCT,
            "typed_diagnostic_nonproduct": NONPRODUCT,
        },
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify(bundle, ROOT, sdk)
    return manifest


def verify(bundle: Path, root: Path, sdk: Path) -> dict[str, object]:
    bundle, root, sdk = bundle.resolve(), root.resolve(), sdk.resolve()
    manifest_path = bundle / "manifest.json"
    regular(manifest_path)
    manifest = require_keys(strict_json(manifest_path), {
        "schema", "schema_version", "source", "sdk", "build", "artifacts",
        "classification"}, "manifest")
    if manifest["schema"] != SCHEMA or manifest["schema_version"] != SCHEMA_VERSION:
        raise Error("manifest schema/version differs")
    source = require_keys(manifest["source"], {"commit", "submodules", "inputs"}, "source")
    if source["commit"] != run("git", "rev-parse", "HEAD", cwd=root):
        raise Error("source commit differs")
    if source["submodules"] != submodules(root):
        raise Error("submodules differ")
    inputs = require_keys(source["inputs"], set(INPUTS), "source.inputs")
    for name, digest in inputs.items():
        if not isinstance(digest, str) or not HEX64.fullmatch(digest) or sha256(root / name) != digest:
            raise Error(f"source input differs: {name}")
    sdk_row = require_keys(manifest["sdk"], {
        "release", "runtime_abi", "build_compiler_sha256",
        "build_inspector_sha256"}, "sdk")
    if sdk_row["release"] != sdk_release(sdk):
        raise Error("SDK release differs")
    if sdk_row["runtime_abi"] != ["hg_wrapper", "hggc_wrapper", "hggcrt1", "hggc"]:
        raise Error("SDK runtime ABI differs")
    for name in ("build_compiler_sha256", "build_inspector_sha256"):
        if not isinstance(sdk_row[name], str) or not HEX64.fullmatch(sdk_row[name]):
            raise Error(f"build-time SDK digest malformed: {name}")
    # Compatibility is release/runtime based. Build compiler/inspector bytes
    # are provenance only: the execute-only box neither requires nor compares
    # those tools. The runner's hggc one-device probe proves runtime presence.
    build = require_keys(manifest["build"], {
        "targets", "q4", "q3", "logs", "cmake_logs", "build_makes"},
        "build")
    if build["targets"] != ["test_fully_quantized_internal_sweep",
                            "test_fq_a02_q3_bchunk_aggregate"] or \
       build["q4"] != "q12/A64/bc0/AP0+AP1" or \
       build["q3"] != "q11/A64/bc0+effective-bc1/AP0":
        raise Error("build identity differs")
    logs = build["logs"]
    if not isinstance(logs, dict) or set(logs) != {"q4-build.log", "q3-build.log"} or \
       any(not isinstance(value, str) or not HEX64.fullmatch(value)
           for value in logs.values()):
        raise Error("build-log authority differs")
    for name in ("cmake_logs", "build_makes"):
        evidence = build[name]
        if not isinstance(evidence, dict) or set(evidence) != {
                "0-cmake.log" if name == "cmake_logs" else "0-build.make",
                "1-cmake.log" if name == "cmake_logs" else "1-build.make"} or \
           any(not isinstance(value, str) or not HEX64.fullmatch(value)
               for value in evidence.values()):
            raise Error(f"{name} authority differs")
    validate_classification(manifest["classification"])
    expected_artifacts = {
        "q4_binary": ("test_fully_quantized_internal_sweep", True),
        "q3_binary": ("test_fq_a02_q3_bchunk_aggregate", True),
        "q4_isa": ("q4.isa.txt", False),
        "q3_isa": ("q3.isa.txt", False),
    }
    artifacts = require_keys(manifest["artifacts"], set(expected_artifacts), "artifacts")
    for key, (name, executable) in expected_artifacts.items():
        path = bundle / name
        regular(path, executable)
        expected = {"path": name, "size": path.stat().st_size, "sha256": sha256(path)}
        if path.stat().st_size <= 0 or artifacts[key] != expected:
            raise Error(f"artifact differs: {key}")
    return manifest


def validate_a01_result(path: Path, minimum_q4_repeats: int = 1024) -> dict[str, object]:
    regular(path)
    result = require_keys(strict_json(path), {
        "schema", "schema_version", "status", "execution", "bundle", "python",
        "formats", "coverage"}, "A01 result")
    if (result["schema"], result["schema_version"], result["status"]) != \
       (A01_SCHEMA, A01_SCHEMA_VERSION, "PASS"):
        raise Error("A01 schema/version/status differs")
    execution = require_keys(result["execution"], {
        "device_library_builds", "host_compilations", "runner",
        "library_load_mode"}, "A01 execution")
    if execution["device_library_builds"] != 0 or execution["host_compilations"] != 0:
        raise Error("A01 was not execute-only")
    if execution["runner"] != "python-ctypes" or \
       execution["library_load_mode"] != "six DSOs, RTLD_LOCAL, one process":
        raise Error("A01 execution identity differs")
    bundle = require_keys(result["bundle"], {
        "manifest_sha256", "source", "sdk", "default_library_identity",
        "libraries"}, "A01 bundle")
    source = require_keys(bundle["source"], {
        "bundle_source_commit", "checkout_head", "runner_sha256"},
        "A01 bundle.source")
    if any(not isinstance(source[name], str) or not re.fullmatch(r"[0-9a-f]{40}", source[name])
           for name in ("bundle_source_commit", "checkout_head")) or \
       not isinstance(source["runner_sha256"], str) or \
       not HEX64.fullmatch(source["runner_sha256"]):
        raise Error("A01 source authority is malformed")
    sdk = require_keys(bundle["sdk"], {
        "root", "release", "archive_sha256", "release_receipt_sha256",
        "hgobjdump_sha256", "runtime_path", "runtime_sha256", "device"},
        "A01 bundle.sdk")
    for name in ("archive_sha256", "release_receipt_sha256", "hgobjdump_sha256",
                 "runtime_sha256"):
        if not isinstance(sdk[name], str) or not HEX64.fullmatch(sdk[name]):
            raise Error(f"A01 SDK digest is malformed: {name}")
    default_identity = require_keys(bundle["default_library_identity"], {
        "path", "packed_format", "any_m"}, "A01 default library")
    if default_identity["packed_format"] != -1 or \
       default_identity["any_m"] != "REJECTS_ALL_CANONICAL_DESCRIPTORS":
        raise Error("A01 default library identity differs")
    require_keys(result["python"], {"numpy", "gguf"}, "A01 python")
    libraries = bundle["libraries"]
    if not isinstance(libraries, list) or len(libraries) != 6:
        raise Error("A01 six-library denominator differs")
    roles = set()
    for row in libraries:
        item = require_keys(row, {"role", "filename", "size", "sha256"}, "A01 library")
        roles.add(item["role"])
        if not isinstance(item["size"], int) or item["size"] <= 0 or \
           not isinstance(item["sha256"], str) or not HEX64.fullmatch(item["sha256"]):
            raise Error("A01 library identity malformed")
    if roles != LIBRARY_ROLES:
        raise Error("A01 library roles differ")
    formats = require_keys(result["formats"], FORMAT_NAMES, "A01 formats")
    coverage = require_keys(result["coverage"], {
        "dense_exact_shape", "grouped_shape", "empty_expert_rows",
        "null_and_explicit_launches", "bad_mapping_workspace_queries",
        "zeroed_scale_unit_fault", "grouped_expert0_rebind_fault",
        "grouped_units_only_metadata_fault", "q4_correctness_repeats",
        "q4_product_policy", "q4_product_policy_source", "numeric_reference",
        "host_prepare_recover_scope"}, "A01 coverage")
    format_keys = {"qtype", "role", "packed_format", "arrangement", "any_m",
                   "selected_config", "shipping_policy", "host_prepare_recover",
                   "frozen_host_artifact", "bad_mapping", "dense", "grouped"}
    grouped_keys = {"shape", "rows_per_expert", "null_config_error",
                    "explicit_config_error", "expert0_rebind_error",
                    "units_only_metadata_error", "raw_bit_stability",
                    "workspace_bytes", "launch_status"}
    dense_keys = {"shape", "null_config_error", "explicit_config_error",
                  "zeroed_scale_unit_error", "raw_bit_stability",
                  "workspace_bytes", "launch_status"}
    stability_keys = {"phase", "launches", "raw_bits_stable", "raw_sha256",
                      "first_launch_status", "last_launch_status"}
    for name, row in formats.items():
        item = require_keys(row, format_keys, f"A01 format {name}")
        qtype, role, packed_format = FORMAT_IDENTITIES[name]
        if (item["qtype"], item["role"], item["packed_format"]) != \
           (qtype, role, packed_format):
            raise Error(f"A01 format identity differs: {name}")
        if item["host_prepare_recover"] != "BYTE_EXACT":
            raise Error(f"A01 host prepare/recover differs: {name}")
        if name == "Q4_K":
            if item["shipping_policy"] != Q4_PER_FORMAT_POLICY:
                raise Error("A01 Q4 per-format product policy differs")
        elif item["shipping_policy"] is not None:
            raise Error(f"A01 non-Q4 shipping policy is unexpected: {name}")
        dense = require_keys(item["dense"], dense_keys, f"A01 dense {name}")
        grouped = require_keys(item["grouped"], grouped_keys, f"A01 grouped {name}")
        if grouped["rows_per_expert"] != [2, 0, 3, 1]:
            raise Error(f"A01 empty-expert coverage differs: {name}")
        expected_launches = coverage["q4_correctness_repeats"] \
            if name == "Q4_K" else 1
        for route, stability in (("dense", dense["raw_bit_stability"]),
                                 ("grouped", grouped["raw_bit_stability"])):
            evidence = require_keys(
                stability, stability_keys, f"A01 {route} stability {name}")
            if evidence["raw_bits_stable"] is not True or \
               evidence["launches"] != expected_launches or \
               not isinstance(evidence["raw_sha256"], str) or \
               not HEX64.fullmatch(evidence["raw_sha256"]):
                raise Error(f"A01 {route} raw-bit stability differs: {name}")
    if coverage["empty_expert_rows"] != [2, 0, 3, 1] or \
       set(coverage["null_and_explicit_launches"]) != FORMAT_NAMES:
        raise Error("A01 grouped/five-format coverage differs")
    repeats = coverage["q4_correctness_repeats"]
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < minimum_q4_repeats:
        raise Error("A01 Q4 correctness repeats are insufficient")
    if coverage["q4_product_policy"] != Q4_PRODUCT_POLICY:
        raise Error("A01 Q4 product policy differs")
    summary = {
        "schema": "quactlize.fq-a02-a01-reference.v1",
        "result_sha256": sha256(path),
        "status": "PASS",
        "libraries": 6,
        "formats": 5,
        "empty_expert_rows": [2, 0, 3, 1],
        "q4_correctness_repeats": repeats,
        "q4_product_policy": Q4_PRODUCT_POLICY,
    }
    return summary


def self_test() -> None:
    if len(INPUTS) != len(set(INPUTS)):
        raise AssertionError("duplicate A02 build input")
    validate_classification({"product_shipping": PRODUCT,
                             "typed_diagnostic_nonproduct": NONPRODUCT})
    plants = (
        {"product_shipping": [*PRODUCT, NONPRODUCT[0]],
         "typed_diagnostic_nonproduct": NONPRODUCT[1:]},
        {"product_shipping": PRODUCT,
         "typed_diagnostic_nonproduct": NONPRODUCT[:-1]},
    )
    for planted in plants:
        try: validate_classification(planted)
        except Error: pass
        else: raise AssertionError("classification plant stayed green")
    print("[fq-a02-prebuilt:self-test] PASS strict authority; promotion/drop plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    authority_create = commands.add_parser("create-build-authority")
    authority_create.add_argument("--output", type=Path, required=True)
    authority_create.add_argument("--sdk", type=Path, required=True)
    authority_create.add_argument("--source-root", type=Path, default=ROOT)
    authority_verify = commands.add_parser("verify-build-authority")
    authority_verify.add_argument("--file", type=Path, required=True)
    authority_verify.add_argument("--sdk", type=Path, required=True)
    authority_verify.add_argument("--source-root", type=Path, default=ROOT)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--sdk", type=Path, required=True)
    verify_parser.add_argument("--source-root", type=Path, default=ROOT)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--bundle", type=Path, required=True)
    create_parser.add_argument("--stage", type=Path, required=True)
    create_parser.add_argument("--sdk", type=Path, required=True)
    create_parser.add_argument("--logs", type=Path, nargs="+", required=True)
    create_parser.add_argument("--cmake-logs", type=Path, nargs="+", required=True)
    create_parser.add_argument("--build-makes", type=Path, nargs="+", required=True)
    a01_parser = commands.add_parser("verify-a01")
    a01_parser.add_argument("--result", type=Path, required=True)
    a01_parser.add_argument("--summary", type=Path, required=True)
    a01_parser.add_argument("--minimum-q4-repeats", type=int, default=1024)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "create-build-authority":
            write_build_authority(args.output, args.source_root, args.sdk)
        elif args.command == "verify-build-authority":
            verify_build_authority(args.file, args.source_root, args.sdk)
        elif args.command == "verify":
            verify(args.bundle, args.source_root, args.sdk)
        elif args.command == "create":
            create(args.bundle, args.stage, args.sdk, args.logs,
                   args.cmake_logs, args.build_makes)
        else:
            summary = validate_a01_result(args.result, args.minimum_q4_repeats)
            if args.summary.exists() or args.summary.is_symlink():
                raise Error("refusing existing A01 summary")
            args.summary.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except (Error, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"[fq-a02-prebuilt] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Strictly verify the prebuilt FQ K-quant execution bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import sys
from typing import Any


SCHEMA = "quactlize.fq-kquant-prebuilt-execution-bundle.v1"
SOURCE = "2b513637fc3d315077b14ab81784ff1fb21e1bb7"
SOURCE_TREE = "261ca1bcd5911d76d4c7c9d91dafd970f24a3b42"
SUBMODULES = {
    "third_party/actlize": "8d46b758c8931807df840a6ed87d272d74a8fdf4",
    "third_party/cutlass": "f94ec46f4f63f96003d6cfdf2014731e7672c281",
}
FORMATS = {
    10: ("Q2_K", 2, ("dense", "grouped")),
    11: ("Q3_K", 3, ("dense", "grouped")),
    12: ("Q4_K", 0, ("grouped",)),
    13: ("Q5_K", 1, ("dense", "grouped")),
    14: ("Q6_K", 4, ("dense", "grouped")),
}
PLAN_PROFILES = {
    "layout-ab": ("plans/plan-layout-ab.json", 77, 24),
    "heuristic": ("plans/plan-heuristic.json", 143, 52),
}
AUTHORITY_MODES = {
    "benchmarks/test_fq_kquant_layout_perf.cu": "0644",
    "benchmarks/moe_router_fixture.hpp": "0644",
    "benchmarks/workloads.py": "0644",
    "build.sh": "0755",
    "quactlize/gguf_roles.py": "0644",
    "quactlize/csrc/device/ppu_dense_backend.cu": "0644",
    "quactlize/include/ppu_format_config.inc": "0644",
    "quactlize/include/kquant_kpack_offline.hpp": "0644",
    "quactlize/include/ppu_dense_configs.inc": "0644",
    "quactlize/include/ppu_grouped_configs.inc": "0644",
    "tools/analyze_fq_kquant_kpack_perf.py": "0755",
    "tools/box_identity_schema.py": "0644",
    "tools/fit_fq_kquant_config_heuristic.py": "0644",
    "tools/gguf_internal_shape_inventory.py": "0644",
    "tools/plan_fq_kquant_kpack_perf.py": "0755",
    "tools/run_fq_kquant_kpack_perf_box.sh": "0755",
}
SDK_RUNTIME = {
    "lib/libhggc.so": (37126360, "4acb6f71da458fbef346db163e5c04a1bdc341c8c560158412ec2c9618c1525a"),
    "lib/libhggcrt.13.0.so": (6490712, "f4765821e374712a5d9a21cb7276101067ff7da56f9e3c55366ffe88c1997e9f"),
    "lib/libhggc_wrapper.so": (5573896, "71c32cb41191458503234324360fcd3f1fa890dd5a082d465bb07328630c775e"),
    "lib/libhg_wrapper.so": (7623832, "fd93f23bfb05dfee0ffa48fd07d3c0cf6d8ab37d266e12f841ee056291affcd5"),
}
TOP_FIELDS = {
    "arch", "authorities", "build", "bundle_tools", "execution", "pairs", "plans",
    "runtime", "schema", "schema_version", "sdk", "source",
}
FILE_FIELDS = {"mode", "path", "sha256", "size"}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class BundleError(ValueError):
    pass


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def one_line(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError(f"{field} must be a nonempty string")
    if any(mark in value for mark in ("\0", "\n", "\r")):
        raise BundleError(f"{field} must be one line")
    return value


def exact_fields(value: Any, fields: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise BundleError(f"{field} has the wrong fields")
    return value


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BundleError(f"{field} must be a positive integer")
    return value


def relative_path(value: Any, field: str) -> pathlib.PurePosixPath:
    raw = one_line(value, field)
    if "\\" in raw:
        raise BundleError(f"{field} must use '/' separators")
    path = pathlib.PurePosixPath(raw)
    if (path.is_absolute() or path.as_posix() != raw or
            any(part in {"", ".", ".."} for part in path.parts)):
        raise BundleError(f"{field} must be a normalized relative path")
    return path


def file_record(value: Any, field: str) -> dict[str, Any]:
    row = exact_fields(value, FILE_FIELDS, field)
    path = relative_path(row["path"], f"{field}.path")
    size = positive_int(row["size"], f"{field}.size")
    digest = one_line(row["sha256"], f"{field}.sha256")
    if not SHA256_RE.fullmatch(digest):
        raise BundleError(f"{field}.sha256 is not lowercase SHA-256")
    mode = one_line(row["mode"], f"{field}.mode")
    if mode not in {"0644", "0755"}:
        raise BundleError(f"{field}.mode is not admitted")
    return {"path": path, "size": size, "sha256": digest, "mode": mode}


def safe_payload(root: pathlib.Path, record: dict[str, Any], field: str) -> None:
    relative = record["path"]
    current = root
    if current.is_symlink() or not current.is_dir():
        raise BundleError(f"{field} root is not a real directory")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise BundleError(f"{field} contains a symlink: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(current, flags)
    except OSError as error:
        raise BundleError(f"cannot open {field} {relative}: {error}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BundleError(f"{field} is not regular: {relative}")
        if before.st_size != record["size"]:
            raise BundleError(f"{field} size differs: {relative}")
        if stat.S_IMODE(before.st_mode) != int(record["mode"], 8):
            raise BundleError(f"{field} mode differs: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["sha256"]:
            raise BundleError(f"{field} SHA-256 differs: {relative}")
        after = current.lstat()
        if (stat.S_ISLNK(after.st_mode) or
                (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)):
            raise BundleError(f"{field} changed while verifying: {relative}")
    finally:
        os.close(descriptor)


def read_manifest(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BundleError("manifest must be a real regular file")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read manifest: {error}") from error
    return exact_fields(value, TOP_FIELDS, "manifest")


def verify_manifest_sidecar(path: pathlib.Path) -> str:
    sidecar = path.with_name("manifest.sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise BundleError("manifest SHA-256 sidecar must be a real regular file")
    try:
        text = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise BundleError(f"cannot read manifest SHA-256 sidecar: {error}") from error
    match = re.fullmatch(r"([0-9a-f]{64})  manifest\.json\n", text)
    if match is None:
        raise BundleError("manifest SHA-256 sidecar is malformed")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != match.group(1):
        raise BundleError("manifest SHA-256 sidecar differs")
    expected_directory = f"fq-kquant-layout-perf-{digest[:12]}"
    if path.parent.name != expected_directory:
        raise BundleError("bundle directory does not carry the manifest digest prefix")
    return digest


def validate_manifest(value: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (value["schema"] != SCHEMA or isinstance(value["schema_version"], bool) or
            value["schema_version"] != 1):
        raise BundleError("unsupported manifest schema")
    if value["arch"] != "ppu0010":
        raise BundleError("architecture is not ppu0010")

    source = exact_fields(value["source"],
                          {"commit", "submodules", "tracked_clean", "tree"},
                          "source")
    if source["commit"] != SOURCE or not GIT_SHA_RE.fullmatch(source["commit"]):
        raise BundleError("source commit differs")
    if source["tree"] != SOURCE_TREE:
        raise BundleError("source tree differs")
    if source["tracked_clean"] is not True or source["submodules"] != SUBMODULES:
        raise BundleError("source cleanliness or submodules differ")

    sdk = exact_fields(value["sdk"],
                       {"archive", "compiler", "inspector", "receipt", "release",
                        "runtime_libraries"},
                       "sdk")
    if sdk["release"] != "2.1.1-a5c56e":
        raise BundleError("SDK release differs")
    archive = exact_fields(sdk["archive"],
                           {"authority_path", "filename", "sha256", "size",
                            "verified_this_build"},
                           "sdk.archive")
    if archive["filename"] != "PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz":
        raise BundleError("SDK archive filename differs")
    if archive["sha256"] != "63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd":
        raise BundleError("SDK archive digest differs")
    if (archive["authority_path"] !=
            "/root/ppu-sdk-cache/PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz" or
            archive["size"] != 1875743407 or
            archive["verified_this_build"] is not True):
        raise BundleError("SDK archive build-time authority differs")
    for name in ("compiler", "inspector"):
        tool = exact_fields(sdk[name], {"identity", "installed_path", "sha256"},
                            f"sdk.{name}")
        one_line(tool["identity"], f"sdk.{name}.identity")
        one_line(tool["installed_path"], f"sdk.{name}.installed_path")
        if not SHA256_RE.fullmatch(one_line(tool["sha256"], f"sdk.{name}.sha256")):
            raise BundleError(f"sdk.{name}.sha256 is malformed")
    receipt = file_record(sdk["receipt"], "sdk.receipt")
    expected_tools = {
        "compiler": {
            "identity": "hgcc(HGGC C/C++ Compiler) | Release version 2.1.1-a5c56e | Built on Jul 25 2026 at 09:15:42",
            "installed_path": "/root/ppu-sdk/2.1.1/bin/hgcc",
            "sha256": "fa62c590c67411c23fa4028f15fa562b39ce0cf830830d038a1ec04c59d8c76e",
        },
        "inspector": {
            "identity": "LLVM version 13.0.1 | ppu version: 2.1.1-a5c56e-",
            "installed_path": "/root/ppu-sdk/2.1.1/bin/hgobjdump",
            "sha256": "4176b21eb5b26e9f73acb5e17667a01fbb2808987b303d95cd8b31078221d16c",
        },
    }
    if sdk["compiler"] != expected_tools["compiler"] or sdk["inspector"] != expected_tools["inspector"]:
        raise BundleError("installed tool identity differs")
    if (receipt["path"].as_posix(), receipt["size"], receipt["sha256"], receipt["mode"]) != (
            "evidence/sdk-release.yaml", 22,
            "7ba29b9d2e768c3edd1e165bceaf90e27880248f2327d44852de37833d0a2cf7",
            "0644"):
        raise BundleError("SDK release receipt differs")
    runtime_libraries = sdk["runtime_libraries"]
    if not isinstance(runtime_libraries, list) or len(runtime_libraries) != len(SDK_RUNTIME):
        raise BundleError("SDK runtime library denominator differs")
    checked_runtime = [file_record(row, f"sdk.runtime_libraries[{index}]")
                       for index, row in enumerate(runtime_libraries)]
    if {row["path"].as_posix() for row in checked_runtime} != set(SDK_RUNTIME):
        raise BundleError("SDK runtime library paths differ")
    for row in checked_runtime:
        expected_size, expected_digest = SDK_RUNTIME[row["path"].as_posix()]
        if (row["size"], row["sha256"], row["mode"]) != (
                expected_size, expected_digest, "0755"):
            raise BundleError("SDK runtime library identity differs")

    build = exact_fields(value["build"],
                         {"common_env", "host_env_unset", "host_link_option",
                          "host_link_provenance_commit", "jobs",
                          "source_contains_host_link_option", "target"},
                         "build")
    if (build["target"] != "test_fq_kquant_layout_perf" or
            isinstance(build["jobs"], bool) or build["jobs"] != 4):
        raise BundleError("build target or jobs differs")
    if build["common_env"] != {"PPU_ARCHS": "ppu0010"}:
        raise BundleError("build common environment differs")
    if build["host_env_unset"] != [
            "CC", "CXX", "CMAKE_GENERATOR", "CMAKE_TOOLCHAIN_FILE",
            "PPU_EXTRA_DEFS"]:
        raise BundleError("build host environment reset differs")
    if build["host_link_option"] != "-Wl,--allow-shlib-undefined":
        raise BundleError("host compatibility link option differs")
    if (build["host_link_provenance_commit"] !=
            "44ca067ca75941ea0c0b861ead324b6bb1cb6881" or
            build["source_contains_host_link_option"] is not False):
        raise BundleError("host compatibility link provenance differs")

    bundle_tools = value["bundle_tools"]
    if not isinstance(bundle_tools, list) or len(bundle_tools) != 3:
        raise BundleError("bundle tool denominator differs")
    tool_records = [file_record(row, f"bundle_tools[{index}]")
                    for index, row in enumerate(bundle_tools)]
    if {row["path"].as_posix() for row in tool_records} != {
            "fq-kquant-sdk-identity.py", "run-prebuilt.sh", "verify-bundle.py"}:
        raise BundleError("bundle tool paths differ")
    if any(row["mode"] != "0755" for row in tool_records):
        raise BundleError("bundle tool mode differs")

    plans = value["plans"]
    if not isinstance(plans, list) or len(plans) != 2:
        raise BundleError("plans must contain exactly two profiles")
    plan_records: list[dict[str, Any]] = []
    seen_profiles: set[str] = set()
    for index, raw in enumerate(plans):
        row = exact_fields(raw, {"dense", "file", "grouped", "profile"},
                           f"plans[{index}]")
        profile = one_line(row["profile"], f"plans[{index}].profile")
        if profile not in PLAN_PROFILES or profile in seen_profiles:
            raise BundleError("plan profile set differs")
        expected_path, dense, grouped = PLAN_PROFILES[profile]
        if row["dense"] != dense or row["grouped"] != grouped:
            raise BundleError(f"{profile} denominator differs")
        record = file_record(row["file"], f"plans[{index}].file")
        if record["path"].as_posix() != expected_path or record["mode"] != "0644":
            raise BundleError(f"{profile} plan path or mode differs")
        plan_records.append(record)
        seen_profiles.add(profile)
    if seen_profiles != set(PLAN_PROFILES):
        raise BundleError("plan profile set is incomplete")

    authorities = value["authorities"]
    if not isinstance(authorities, list) or len(authorities) != len(AUTHORITY_MODES):
        raise BundleError("authority file denominator differs")
    authority_records = [file_record(row, f"authorities[{index}]")
                         for index, row in enumerate(authorities)]
    if {row["path"].as_posix() for row in authority_records} != set(AUTHORITY_MODES):
        raise BundleError("authority file paths differ")
    for row in authority_records:
        if row["mode"] != AUTHORITY_MODES[row["path"].as_posix()]:
            raise BundleError("authority file mode differs")

    pairs = value["pairs"]
    if not isinstance(pairs, list) or len(pairs) != len(FORMATS):
        raise BundleError("binary/library pair denominator differs")
    bundle_records: list[dict[str, Any]] = [receipt, *tool_records, *plan_records]
    seen_qtypes: set[int] = set()
    for index, raw in enumerate(pairs):
        row = exact_fields(raw,
                           {"binary", "evidence", "format", "library", "packed_format",
                            "ppu_defs", "qtype"},
                           f"pairs[{index}]")
        qtype = row["qtype"]
        if isinstance(qtype, bool) or qtype not in FORMATS or qtype in seen_qtypes:
            raise BundleError("qtype pair set differs")
        name, packed_format, _ = FORMATS[qtype]
        if (row["format"] != name or isinstance(row["packed_format"], bool) or
                row["packed_format"] != packed_format):
            raise BundleError(f"qtype {qtype} format mapping differs")
        expected_defs = [
            "PPU_PACKED_SCALE=1",
            f"PPU_PACKED_FORMAT={packed_format}",
            f"QUACTLIZE_DENSE_ONLY={qtype}",
            f"FQ_KQUANT_PERF_QTYPE={qtype}",
        ]
        if row["ppu_defs"] != expected_defs:
            raise BundleError(f"qtype {qtype} definitions differ")
        binary = file_record(row["binary"], f"pairs[{index}].binary")
        library = file_record(row["library"], f"pairs[{index}].library")
        if binary["path"].as_posix() != f"q{qtype}/test_fq_kquant_layout_perf":
            raise BundleError(f"qtype {qtype} executable path differs")
        if library["path"].as_posix() != f"q{qtype}/libquactlize_ppu.so":
            raise BundleError(f"qtype {qtype} library path differs")
        if binary["mode"] != "0755" or library["mode"] != "0755":
            raise BundleError(f"qtype {qtype} payload mode differs")
        evidence = exact_fields(row["evidence"],
                                {"binary_device_images", "build_log", "cmake_log",
                                 "initial_link_exit", "library_device_images",
                                 "link_command"},
                                f"pairs[{index}].evidence")
        if (positive_int(evidence["binary_device_images"], "binary_device_images") < 1 or
                positive_int(evidence["library_device_images"], "library_device_images") < 1):
            raise BundleError("embedded device image count differs")
        build_log = file_record(evidence["build_log"], f"pairs[{index}].build_log")
        cmake_log = file_record(evidence["cmake_log"], f"pairs[{index}].cmake_log")
        link_command = file_record(evidence["link_command"],
                                   f"pairs[{index}].link_command")
        if isinstance(evidence["initial_link_exit"], bool) or evidence["initial_link_exit"] != 2:
            raise BundleError(f"qtype {qtype} initial link exit differs")
        if build_log["path"].as_posix() != f"evidence/build-q{qtype}.log":
            raise BundleError(f"qtype {qtype} build log path differs")
        if cmake_log["path"].as_posix() != f"evidence/cmake-q{qtype}.log":
            raise BundleError(f"qtype {qtype} CMake log path differs")
        if link_command["path"].as_posix() != f"evidence/link-q{qtype}.txt":
            raise BundleError(f"qtype {qtype} link command path differs")
        bundle_records.extend((binary, library, build_log, cmake_log, link_command))
        seen_qtypes.add(qtype)
    if seen_qtypes != set(FORMATS):
        raise BundleError("qtype pair set is incomplete")

    execution = exact_fields(
        value["execution"],
        {"all_configs", "default_profile", "heuristic", "iterations", "qtype_operators",
         "rounds", "threshold_pct", "warmups"},
        "execution")
    if (execution["default_profile"] != "heuristic" or
            isinstance(execution["all_configs"], bool) or execution["all_configs"] != 1):
        raise BundleError("execution default is not heuristic/all-configs")
    controls = (execution["iterations"], execution["warmups"], execution["rounds"])
    if any(isinstance(item, bool) or not isinstance(item, int) for item in controls):
        raise BundleError("execution integer controls are malformed")
    threshold = execution["threshold_pct"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
            not math.isfinite(float(threshold)) or
            (*controls, float(threshold)) != (11, 3, 3, 3.0)):
        raise BundleError("execution measurement defaults differ")
    expected_operators = {str(q): list(FORMATS[q][2]) for q in FORMATS}
    if execution["qtype_operators"] != expected_operators:
        raise BundleError("execution operator scope differs")
    if execution["heuristic"] != {
            "max_leaves": 8, "min_leaf_families": 1, "min_leaf_rows": 2}:
        raise BundleError("heuristic fit defaults differ")

    runtime = exact_fields(value["runtime"],
                           {"build_host", "execution_floor", "loader_preflight"},
                           "runtime")
    if runtime["build_host"] != {
            "distribution": "ubuntu", "execution_verified": False,
            "glibc": "2.35", "glibcxx_max": "GLIBCXX_3.4.30",
            "version_id": "22.04"}:
        raise BundleError("build host disclosure differs")
    if runtime["execution_floor"] != {
            "distribution": "ubuntu", "glibc_min": "2.38",
            "glibcxx_min": "GLIBCXX_3.4.32", "version_id": "24.04"}:
        raise BundleError("execution runtime floor differs")
    if runtime["loader_preflight"] != {
            "build_host_status": "blocked-by-runtime-floor",
            "target_requirement": "all five ldd closures: no missing/version errors",
            "verified_on_target": False}:
        raise BundleError("loader preflight contract differs")

    paths = [record["path"] for record in bundle_records]
    if len(paths) != len(set(paths)):
        raise BundleError("bundle file path is duplicated")
    return bundle_records, authority_records


def validate_runner_contract(runner: str, sdk_helper: str) -> None:
    runner_required = (
        'fq-kquant-sdk-identity.py',
        'quactlize.fq-kquant-prebuilt-runtime-preflight.v2',
        'runtime-preflight.sha256',
        'cmp -s -- "$runtime_preflight_current" "$runtime_preflight"',
        'resume runtime preflight differs (host, SDK root, policy, or actual identity)',
        'quactlize.fq-kquant-prebuilt-result-authority.v2',
        '"evidence_grade": grade',
        '"sdk_mismatch_count": mismatch_count',
    )
    missing = [item for item in runner_required if item not in runner]
    if missing:
        raise BundleError(f"runner relaxed-SDK contract is incomplete: {missing[0]}")
    compare = runner.index(
        'cmp -s -- "$runtime_preflight_current" "$runtime_preflight"')
    execution = runner.index('for qtype in 10 11 12 13 14; do', compare)
    if compare >= execution:
        raise BundleError("runner binds runtime preflight after benchmark execution")
    if 'quactlize.fq-kquant-prebuilt-runtime-preflight.v1' in runner:
        raise BundleError("runner retains the obsolete runtime preflight schema")
    helper_required = (
        'ALLOW_UNVERIFIED_SDK',
        '"identity_status": "MISMATCH_ALLOWED" if unverified else "VERIFIED"',
        '"evidence_grade": "unverified-sdk" if unverified else "verified-sdk"',
        '"expected": expected',
        '"actual": actual',
        '"matches": matches',
        '"mismatches": mismatches',
    )
    missing = [item for item in helper_required if item not in sdk_helper]
    if missing:
        raise BundleError(f"SDK helper evidence contract is incomplete: {missing[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--repo-root", required=True, type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        manifest = pathlib.Path(os.path.abspath(args.manifest))
        repo_root = pathlib.Path(os.path.abspath(args.repo_root))
        verify_manifest_sidecar(manifest)
        value = read_manifest(manifest)
        bundle_records, authority_records = validate_manifest(value)
        for record in bundle_records:
            safe_payload(manifest.parent, record, "bundle payload")
        for record in authority_records:
            safe_payload(repo_root, record, "source authority")
        runner_text = (manifest.parent / "run-prebuilt.sh").read_text(encoding="utf-8")
        sdk_helper_text = (manifest.parent / "fq-kquant-sdk-identity.py").read_text(
            encoding="utf-8")
        validate_runner_contract(runner_text, sdk_helper_text)
        if args.self_test:
            plants = []
            broken = copy.deepcopy(value)
            broken["pairs"][0]["packed_format"] = 4
            plants.append(lambda broken=broken: validate_manifest(broken))
            broken = copy.deepcopy(value)
            broken["runtime"]["execution_floor"]["glibc_min"] = "2.35"
            plants.append(lambda broken=broken: validate_manifest(broken))
            broken_record = dict(bundle_records[0], sha256="0" * 64)
            plants.append(lambda: safe_payload(manifest.parent, broken_record,
                                                "planted payload"))
            plants.append(lambda: unique_object([("duplicate", 1), ("duplicate", 2)]))
            plants.append(lambda: validate_runner_contract(
                runner_text.replace(
                    'cmp -s -- "$runtime_preflight_current" "$runtime_preflight"',
                    'true # planted missing runtime-preflight comparison', 1),
                sdk_helper_text))
            for plant in plants:
                try:
                    plant()
                except BundleError:
                    pass
                else:
                    raise BundleError("verifier negative plant stayed green")
    except BundleError as error:
        print(f"prebuilt FQ K-quant bundle rejected: {error}", file=sys.stderr)
        return 2
    print(f"[prebuilt-fq-kquant] VERIFIED source={SOURCE} pairs=5 payloads=31")
    if args.self_test:
        print("[prebuilt-fq-kquant] SELF_TEST PASS plants=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

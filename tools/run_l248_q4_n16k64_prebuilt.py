#!/usr/bin/env python3
"""Verify and execute one source-bound prebuilt L248 raw-bit probe.

This runner never configures, compiles, links, or selects a product route.  It
accepts exactly one manifest-owned executable, verifies its source/submodule,
SDK, runtime, payload, and ISA authority, requires one visible PPU device, and
then executes the positive and wrong-oracle arms.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence


SCHEMA = "quactlize.l248-q4-n16k64-prebuilt"
SCHEMA_VERSION = 1
REQUIRED_ANCESTOR = "481491936a61e54f3c340fb60fed4953ac26f4b9"
SDK_RELEASE = "2.1.1-a5c56e"
SDK_ARCHIVE_SHA256 = (
    "63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd"
)
RUNTIME_SHA256 = (
    "71c32cb41191458503234324360fcd3f1fa890dd5a082d465bb07328630c775e"
)
TARGET = "test_q4_n16k64_delivery_rawbit"
ARCH = "ppu0010"
PPU_DEFS = [
    "PPU_PACKED_SCALE=1",
    "PPU_PACKED_FORMAT=0",
    "QUACTLIZE_DENSE_ONLY=12",
]
LAYOUT = 3
MAPPING_ID = 0x51344E3136440001
WORDS = 2048
STAGE_BYTES = 8192
KERNEL_SYMBOL_TOKEN = "q4_n16k64_delivery_rawbit_kernel"
ARTIFACT_PATH = "bin/test_q4_n16k64_delivery_rawbit"
RUNNER_PATH = "run_l248_q4_n16k64_prebuilt.py"
AUTHORITY_PATH = "authority.sha256"


class GateError(RuntimeError):
    """A required source, payload, device, or result condition did not hold."""


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: pathlib.Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read strict JSON {path}: {exc}") from exc


def _write_json(path: pathlib.Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except OSError as exc:
        raise GateError(f"cannot publish JSON {path}: {exc}") from exc


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GateError(f"{label} must be an object")
    got = set(value)
    if got != expected:
        raise GateError(
            f"{label} fields differ: got={sorted(got)} expected={sorted(expected)}")
    return value


def _hex(value: object, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
            rf"[0-9a-f]{{{length}}}", value):
        raise GateError(f"{label} must be {length} lowercase hexadecimal digits")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateError(f"{label} must be an integer >= {minimum}")
    return value


def _canonical_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise GateError(f"{label} is not a canonical relative POSIX path")
    pure = pathlib.PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise GateError(f"{label} is not a canonical relative POSIX path: {value!r}")
    if pure.as_posix() != value:
        raise GateError(f"{label} is not canonical: {value!r}")
    return value


def _bundle_file(bundle: pathlib.Path, relative: str) -> pathlib.Path:
    relative = _canonical_relative(relative, "payload path")
    candidate = bundle.joinpath(*pathlib.PurePosixPath(relative).parts)
    current = bundle
    for part in pathlib.PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise GateError(f"payload path contains a symbolic link: {relative}")
    if not candidate.is_file():
        raise GateError(f"payload file is missing or not regular: {relative}")
    try:
        candidate.resolve().relative_to(bundle.resolve())
    except ValueError as exc:
        raise GateError(f"payload path escapes bundle: {relative}") from exc
    return candidate


def _manifest_entry(entry: object, label: str) -> Mapping[str, object]:
    item = _exact_keys(entry, {"path", "size", "sha256"}, label)
    _canonical_relative(item["path"], f"{label}.path")
    _integer(item["size"], f"{label}.size", 1)
    _hex(item["sha256"], 64, f"{label}.sha256")
    return item


def _validate_manifest(document: object) -> Mapping[str, object]:
    manifest = _exact_keys(
        document,
        {"schema", "schema_version", "source", "sdk", "target",
         "artifact", "evidence", "runner"},
        "manifest")
    if manifest["schema"] != SCHEMA or manifest["schema_version"] != SCHEMA_VERSION:
        raise GateError(
            f"manifest schema differs: {manifest.get('schema')!r}/"
            f"{manifest.get('schema_version')!r}")

    source = _exact_keys(
        manifest["source"],
        {"commit", "tree", "required_ancestor", "submodules", "inputs"},
        "manifest.source")
    _hex(source["commit"], 40, "manifest.source.commit")
    _hex(source["tree"], 40, "manifest.source.tree")
    if source["required_ancestor"] != REQUIRED_ANCESTOR:
        raise GateError("manifest source ancestor is not the committed L249 gate")
    if not isinstance(source["submodules"], list) or not source["submodules"]:
        raise GateError("manifest.source.submodules must be a nonempty list")
    seen_submodules: set[str] = set()
    for index, raw in enumerate(source["submodules"]):
        item = _exact_keys(raw, {"path", "commit", "tree"},
                           f"manifest.source.submodules[{index}]")
        path = _canonical_relative(item["path"], "submodule path")
        if path in seen_submodules:
            raise GateError(f"duplicate submodule authority: {path}")
        seen_submodules.add(path)
        _hex(item["commit"], 40, f"submodule {path} commit")
        _hex(item["tree"], 40, f"submodule {path} tree")
    if seen_submodules != {"third_party/actlize", "third_party/cutlass"}:
        raise GateError(f"submodule authority set differs: {sorted(seen_submodules)}")
    if not isinstance(source["inputs"], list) or not source["inputs"]:
        raise GateError("manifest.source.inputs must be a nonempty list")
    seen_inputs: set[str] = set()
    for index, raw in enumerate(source["inputs"]):
        item = _exact_keys(raw, {"path", "blob", "sha256"},
                           f"manifest.source.inputs[{index}]")
        path = _canonical_relative(item["path"], "source input path")
        if path in seen_inputs:
            raise GateError(f"duplicate source input authority: {path}")
        seen_inputs.add(path)
        _hex(item["blob"], 40, f"source input {path} blob")
        _hex(item["sha256"], 64, f"source input {path} sha256")

    sdk = _exact_keys(
        manifest["sdk"],
        {"release", "archive_sha256", "compiler_identity", "files"},
        "manifest.sdk")
    if sdk["release"] != SDK_RELEASE or sdk["archive_sha256"] != SDK_ARCHIVE_SHA256:
        raise GateError("manifest SDK release/archive is not admitted")
    if not isinstance(sdk["compiler_identity"], str) or SDK_RELEASE not in sdk["compiler_identity"]:
        raise GateError("manifest compiler identity does not bind the SDK release")
    files = _exact_keys(
        sdk["files"],
        {"bin/hgcc", "bin/hgobjdump", "lib/libhggc_wrapper.so"},
        "manifest.sdk.files")
    for relative, digest in files.items():
        _canonical_relative(relative, "SDK file path")
        _hex(digest, 64, f"SDK file {relative} sha256")
    if files["lib/libhggc_wrapper.so"] != RUNTIME_SHA256:
        raise GateError("manifest runtime digest is not the admitted PPU runtime")

    target = _exact_keys(
        manifest["target"],
        {"name", "arch", "ppu_defs", "layout", "mapping_id", "words",
         "stage_bytes"},
        "manifest.target")
    expected_target = {
        "name": TARGET,
        "arch": ARCH,
        "ppu_defs": PPU_DEFS,
        "layout": LAYOUT,
        "mapping_id": MAPPING_ID,
        "words": WORDS,
        "stage_bytes": STAGE_BYTES,
    }
    if dict(target) != expected_target:
        raise GateError(f"manifest target contract differs: {dict(target)!r}")

    artifact = _exact_keys(
        manifest["artifact"],
        {"path", "size", "sha256", "kernel_count", "symbol", "isa_counts"},
        "manifest.artifact")
    if artifact["path"] != ARTIFACT_PATH:
        raise GateError("manifest artifact path differs")
    _integer(artifact["size"], "manifest.artifact.size", 1)
    _hex(artifact["sha256"], 64, "manifest.artifact.sha256")
    if artifact["kernel_count"] != 1:
        raise GateError("manifest must contain exactly one PPU kernel")
    if not isinstance(artifact["symbol"], str) or \
            KERNEL_SYMBOL_TOKEN not in artifact["symbol"]:
        raise GateError("manifest artifact symbol is not the L248 raw-bit kernel")
    counts = _exact_keys(
        artifact["isa_counts"],
        {"aiu_plain_b32", "aiu_all", "universal_tsm_b32x4", "tsm_load_all",
         "commit", "wait", "barrier", "swizzle"},
        "manifest.artifact.isa_counts")
    if dict(counts) != {
            "aiu_plain_b32": 4, "aiu_all": 4,
            "universal_tsm_b32x4": 16, "tsm_load_all": 16,
            "commit": 1, "wait": 1, "barrier": 1, "swizzle": 0}:
        raise GateError(f"manifest ISA contract differs: {dict(counts)!r}")

    if not isinstance(manifest["evidence"], list) or not manifest["evidence"]:
        raise GateError("manifest.evidence must be a nonempty list")
    seen_evidence: set[str] = set()
    for index, raw in enumerate(manifest["evidence"]):
        item = _manifest_entry(raw, f"manifest.evidence[{index}]")
        path = str(item["path"])
        if path in seen_evidence:
            raise GateError(f"duplicate evidence path: {path}")
        seen_evidence.add(path)

    runner = _manifest_entry(manifest["runner"], "manifest.runner")
    if runner["path"] != RUNNER_PATH:
        raise GateError("manifest runner path differs")
    return manifest


def _verify_authority(
        bundle: pathlib.Path,
        expected: Mapping[str, tuple[int, str]],
        ) -> dict[str, str]:
    authority = _bundle_file(bundle, AUTHORITY_PATH)
    entries: dict[str, str] = {}
    try:
        lines = authority.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateError(f"cannot read payload authority: {exc}") from exc
    if not lines:
        raise GateError("payload authority is empty")
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise GateError(f"malformed payload authority line: {line!r}")
        digest, relative = match.groups()
        relative = _canonical_relative(relative, "authority path")
        if relative in entries:
            raise GateError(f"duplicate authority path: {relative}")
        entries[relative] = digest
    if set(entries) != set(expected):
        raise GateError(
            f"authority file set differs: got={sorted(entries)} "
            f"expected={sorted(expected)}")

    actual_files: set[str] = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise GateError(f"payload contains a symbolic link: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(bundle).as_posix())
    wanted_files = set(expected) | {AUTHORITY_PATH}
    if actual_files != wanted_files:
        raise GateError(
            f"payload regular-file set differs: got={sorted(actual_files)} "
            f"expected={sorted(wanted_files)}")

    verified: dict[str, str] = {}
    for relative in sorted(expected):
        path = _bundle_file(bundle, relative)
        wanted_size, wanted_digest = expected[relative]
        if path.stat().st_size != wanted_size:
            raise GateError(
                f"payload size differs for {relative}: got={path.stat().st_size} "
                f"expected={wanted_size}")
        got = _sha256(path)
        if got != wanted_digest or entries[relative] != wanted_digest:
            raise GateError(
                f"payload SHA-256 differs for {relative}: got={got} "
                f"manifest={wanted_digest} authority={entries[relative]}")
        verified[relative] = got
    return verified


def _load_bundle(bundle: pathlib.Path, require_invoked_runner: bool = True):
    if bundle.is_symlink() or not bundle.is_dir():
        raise GateError(f"bundle must be a regular directory, not a symlink: {bundle}")
    bundle = bundle.resolve()
    manifest_path = _bundle_file(bundle, "manifest.json")
    manifest = _validate_manifest(_read_json(manifest_path))

    expected: dict[str, tuple[int, str]] = {
        "manifest.json": (manifest_path.stat().st_size, _sha256(manifest_path)),
    }
    artifact = manifest["artifact"]
    expected[str(artifact["path"])] = (
        int(artifact["size"]), str(artifact["sha256"]))
    runner = manifest["runner"]
    expected[str(runner["path"])] = (int(runner["size"]), str(runner["sha256"]))
    for item in manifest["evidence"]:
        expected[str(item["path"])] = (int(item["size"]), str(item["sha256"]))
    verified = _verify_authority(bundle, expected)

    runner_path = _bundle_file(bundle, RUNNER_PATH)
    if require_invoked_runner and pathlib.Path(__file__).resolve() != runner_path.resolve():
        raise GateError(
            "execute the manifest-owned runner from inside the payload, not another checkout")
    binary = _bundle_file(bundle, ARTIFACT_PATH)
    if not os.access(binary, os.X_OK):
        raise GateError("manifest-owned L248 artifact is not executable")
    return bundle, manifest, binary, verified


def _run_git(source_tree: pathlib.Path, arguments: Sequence[str], binary=False):
    command = ["git", "-C", str(source_tree), *arguments]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True,
            text=not binary, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"git source-authority command failed: {exc}") from exc
    if result.returncode != 0:
        error = result.stderr if not binary else result.stderr.decode(
            "utf-8", errors="replace")
        raise GateError(
            f"git source-authority command failed rc={result.returncode}: "
            f"{error.strip()}")
    return result.stdout


def _verify_source_authority(
        source_tree: pathlib.Path, source: Mapping[str, object]) -> dict[str, object]:
    if source_tree.is_symlink() or not source_tree.is_dir():
        raise GateError(f"source tree must be a regular directory: {source_tree}")
    source_tree = source_tree.resolve()
    commit = str(source["commit"])
    _run_git(source_tree, ["cat-file", "-e", f"{commit}^{{commit}}"])
    tree = str(_run_git(source_tree, ["rev-parse", f"{commit}^{{tree}}"])).strip()
    if tree != source["tree"]:
        raise GateError(f"source tree object differs: got={tree} expected={source['tree']}")
    try:
        ancestor = subprocess.run(
            ["git", "-C", str(source_tree), "merge-base", "--is-ancestor",
             str(source["required_ancestor"]), commit],
            check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"cannot prove committed L249 ancestry: {exc}") from exc
    if ancestor.returncode != 0:
        raise GateError("bundle source is not descended from the committed L249 gate")

    for item in source["inputs"]:
        path = str(item["path"])
        blob = str(_run_git(source_tree, ["rev-parse", f"{commit}:{path}"])).strip()
        if blob != item["blob"]:
            raise GateError(
                f"source input blob differs for {path}: got={blob} expected={item['blob']}")
        contents = _run_git(source_tree, ["show", f"{commit}:{path}"], binary=True)
        got = _bytes_sha256(contents)
        if got != item["sha256"]:
            raise GateError(
                f"source input content differs for {path}: got={got} "
                f"expected={item['sha256']}")

    checked_submodules: dict[str, str] = {}
    for item in source["submodules"]:
        path = str(item["path"])
        listing = str(_run_git(source_tree, ["ls-tree", commit, "--", path])).strip()
        match = re.fullmatch(r"160000 commit ([0-9a-f]{40})\t(.+)", listing)
        if not match or match.group(2) != path or match.group(1) != item["commit"]:
            raise GateError(f"source gitlink differs for {path}: {listing!r}")
        submodule = source_tree / path
        if submodule.is_symlink() or not submodule.is_dir():
            raise GateError(f"source submodule is missing or symlinked: {path}")
        _run_git(submodule, ["cat-file", "-e", f"{item['commit']}^{{commit}}"])
        sub_tree = str(_run_git(
            submodule, ["rev-parse", f"{item['commit']}^{{tree}}"])).strip()
        if sub_tree != item["tree"]:
            raise GateError(f"submodule tree differs for {path}")
        checked_submodules[path] = str(item["commit"])
    head = str(_run_git(source_tree, ["rev-parse", "HEAD"])).strip()
    return {
        "bundle_source_commit": commit,
        "bundle_source_tree": tree,
        "checkout_head": head,
        "submodules": checked_submodules,
        "inputs": len(source["inputs"]),
    }


def _release(path: pathlib.Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateError(f"cannot read SDK release receipt: {exc}") from exc
    versions = [line.split(":", 1)[1].strip()
                for line in lines if line.startswith("version:")]
    if len(versions) != 1 or not versions[0]:
        raise GateError("SDK release.yaml must contain exactly one version")
    return versions[0]


def _assert_sdk_runtime_compatibility(
        release: str, runtime_digest: str,
        sdk: Mapping[str, object]) -> dict[str, object]:
    """Admit same-release box SDKs without requiring build-tool byte identity."""
    if release != SDK_RELEASE or release != sdk["release"]:
        raise GateError(f"PPU SDK release differs: got={release} expected={SDK_RELEASE}")
    recorded_runtime = sdk["files"]["lib/libhggc_wrapper.so"]
    if recorded_runtime != RUNTIME_SHA256:
        raise GateError("manifest runtime digest is not the admitted PPU runtime")
    if runtime_digest != recorded_runtime:
        raise GateError(
            f"PPU SDK runtime digest differs: got={runtime_digest} "
            f"expected={recorded_runtime}")
    return {
        "release": release,
        "runtime_sha256": runtime_digest,
        "runtime_abi": "hggcGetDeviceCount+hggcGetDevice+L248 executable",
        "build_tool_hashes_recorded": {
            "bin/hgcc": sdk["files"]["bin/hgcc"],
            "bin/hgobjdump": sdk["files"]["bin/hgobjdump"],
        },
        "build_tool_byte_identity_required_on_box": False,
    }


def _validate_sdk(sdk_root: pathlib.Path, sdk: Mapping[str, object]):
    if sdk_root.is_symlink() or not sdk_root.is_dir():
        raise GateError(f"PPU SDK must be a regular directory: {sdk_root}")
    sdk_root = sdk_root.resolve()
    receipt = sdk_root / "release.yaml"
    if receipt.is_symlink() or not receipt.is_file():
        raise GateError("PPU SDK release.yaml is missing or symlinked")
    release = _release(receipt)
    runtime = sdk_root / "lib" / "libhggc_wrapper.so"
    if runtime.is_symlink() or not runtime.is_file():
        raise GateError("PPU SDK runtime is missing or symlinked")
    compatibility = _assert_sdk_runtime_compatibility(
        release, _sha256(runtime), sdk)
    return sdk_root, compatibility


def _visible_device_ordinal(environment: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environment is None else environment
    value = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+", value):
        raise GateError(
            "CUDA_VISIBLE_DEVICES must name exactly one numeric device ordinal")
    return value


def _runtime_device_identity_from_calls(
        get_device_count, get_device,
        environment: Mapping[str, str] | None = None) -> dict[str, object]:
    visible = _visible_device_ordinal(environment)
    count = ctypes.c_int(-1)
    rc = int(get_device_count(ctypes.byref(count)))
    if rc != 0:
        raise GateError(f"hggcGetDeviceCount failed rc={rc}")
    if count.value != 1:
        raise GateError(
            f"hggcGetDeviceCount returned {count.value}; exactly one visible device is required")
    current = ctypes.c_int(-1)
    rc = int(get_device(ctypes.byref(current)))
    if rc != 0:
        raise GateError(f"hggcGetDevice failed rc={rc}")
    if current.value != 0:
        raise GateError(
            f"hggcGetDevice returned logical ordinal {current.value}; expected 0")
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "hggc_device_count": count.value,
        "hggc_current_device": current.value,
    }


def _runtime_device_identity(
        runtime_path: pathlib.Path,
        environment: Mapping[str, str] | None = None) -> dict[str, object]:
    # Reject unset or multi-device visibility before even loading the runtime.
    _visible_device_ordinal(environment)
    try:
        runtime = ctypes.CDLL(
            str(runtime_path), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    except OSError as exc:
        raise GateError(f"cannot load admitted PPU runtime {runtime_path}: {exc}") from exc
    try:
        get_count = runtime.hggcGetDeviceCount
        get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_count.restype = ctypes.c_int
        get_device = runtime.hggcGetDevice
        get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_device.restype = ctypes.c_int
    except AttributeError as exc:
        raise GateError(f"PPU runtime lacks one-device query exports: {exc}") from exc
    return _runtime_device_identity_from_calls(
        get_count, get_device, environment=environment)


PROBE_RE = re.compile(
    r"^FQ_Q4_N16K64_DELIVERY_RAWBIT verdict=(?P<verdict>PASS|FAIL) "
    r"layout=(?P<layout>[0-9]+) mapping_id=0x(?P<mapping>[0-9a-f]{16}) "
    r"words=(?P<words>[0-9]+) raw_bad=(?P<raw_bad>[0-9]+) "
    r"sentinel=(?P<sentinel>[0-9]+) "
    r"source_hash=0x(?P<source_hash>[0-9a-f]{16}) "
    r"want_hash=0x(?P<want_hash>[0-9a-f]{16}) "
    r"got_hash=0x(?P<got_hash>[0-9a-f]{16}) "
    r"first=\[index:(?P<first>-?[0-9]+),kblock:(?P<kblock>-?[0-9]+),"
    r"ncohort:(?P<ncohort>-?[0-9]+),lane:(?P<lane>-?[0-9]+),"
    r"vreg:(?P<vreg>-?[0-9]+),want:0x(?P<first_want>[0-9a-f]{8}),"
    r"got:0x(?P<first_got>[0-9a-f]{8})\] "
    r"launch=\[before:(?P<before>[0-9]+),immediate:(?P<immediate>[0-9]+),"
    r"sync:(?P<sync>[0-9]+),copy:(?P<copy>[0-9]+)\] "
    r"plant=(?P<plant>none|wrong-oracle)$")


def _parse_probe(stdout: str) -> dict[str, object]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise GateError(f"L248 probe emitted {len(lines)} nonempty lines, expected 1")
    match = PROBE_RE.fullmatch(lines[0])
    if not match:
        raise GateError(f"L248 probe verdict is malformed: {lines[0]!r}")
    result: dict[str, object] = match.groupdict()
    for key in ("layout", "words", "raw_bad", "sentinel", "first", "kblock",
                "ncohort", "lane", "vreg", "before", "immediate", "sync", "copy"):
        result[key] = int(str(result[key]))
    result["mapping_id"] = int(str(result.pop("mapping")), 16)
    return result


def _assert_positive(returncode: int, result: Mapping[str, object]) -> None:
    if returncode != 0 or result["verdict"] != "PASS":
        raise GateError(f"positive L248 arm failed rc={returncode}: {dict(result)!r}")
    expected = {
        "layout": LAYOUT, "mapping_id": MAPPING_ID, "words": WORDS,
        "raw_bad": 0, "sentinel": 0, "first": -1, "kblock": -1,
        "ncohort": -1, "lane": -1, "vreg": -1,
        "before": 0, "immediate": 0, "sync": 0, "copy": 0,
        "plant": "none",
    }
    for key, wanted in expected.items():
        if result[key] != wanted:
            raise GateError(
                f"positive L248 field {key} differs: got={result[key]!r} expected={wanted!r}")
    hashes = {result["source_hash"], result["want_hash"], result["got_hash"]}
    if len(hashes) != 1 or hashes == {"0000000000000000"}:
        raise GateError("positive L248 hashes are not one identical nonzero fingerprint")


def _assert_negative(
        returncode: int, result: Mapping[str, object],
        positive: Mapping[str, object]) -> None:
    if returncode != 1 or result["verdict"] != "FAIL":
        raise GateError(
            f"wrong-oracle L248 arm did not return exact RED rc=1: "
            f"rc={returncode} result={dict(result)!r}")
    expected = {
        "layout": LAYOUT, "mapping_id": MAPPING_ID, "words": WORDS,
        "raw_bad": 1, "sentinel": 0, "first": 0, "kblock": 0,
        "ncohort": 0, "lane": 0, "vreg": 0,
        "before": 0, "immediate": 0, "sync": 0, "copy": 0,
        "plant": "wrong-oracle",
    }
    for key, wanted in expected.items():
        if result[key] != wanted:
            raise GateError(
                f"wrong-oracle L248 field {key} differs: "
                f"got={result[key]!r} expected={wanted!r}")
    if result["source_hash"] != positive["source_hash"] or \
            result["got_hash"] != positive["got_hash"] or \
            result["want_hash"] == result["got_hash"]:
        raise GateError("wrong-oracle RED did not isolate only the independent golden")


def _execute_binary(
        binary: pathlib.Path, arguments: Sequence[str], environment: Mapping[str, str],
        timeout_seconds: int):
    try:
        return subprocess.run(
            [str(binary), *arguments], check=False, capture_output=True, text=True,
            env=dict(environment), timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"cannot execute manifest-owned L248 binary: {exc}") from exc


def _create_output(path: pathlib.Path) -> pathlib.Path:
    if not path.is_absolute():
        raise GateError("--output must be absolute")
    if path.exists() or path.is_symlink():
        raise GateError(f"refusing to overwrite device evidence output: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise GateError(f"device evidence parent is missing or symlinked: {parent}")
    try:
        path.mkdir(mode=0o755)
    except OSError as exc:
        raise GateError(f"cannot create device evidence output {path}: {exc}") from exc
    return path


def _execution_log(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"returncode={result.returncode}\n"
        "stdout-begin\n" + result.stdout +
        ("" if result.stdout.endswith("\n") else "\n") +
        "stdout-end\n"
        "stderr-begin\n" + result.stderr +
        ("" if result.stderr.endswith("\n") or not result.stderr else "\n") +
        "stderr-end\n")


def _result_authority(output: pathlib.Path, files: Sequence[str]) -> None:
    lines = []
    for relative in sorted(files):
        path = output / relative
        lines.append(f"{_sha256(path)}  {relative}\n")
    (output / "authority.sha256").write_text("".join(lines), encoding="utf-8")


def run_bundle(
        bundle: pathlib.Path, sdk_root: pathlib.Path, source_tree: pathlib.Path,
        output: pathlib.Path | None, verify_only: bool, timeout_seconds: int,
        expected_source_sha: str, expected_binary_sha256: str,
        expected_manifest_sha256: str):
    bundle, manifest, binary, payload_authority = _load_bundle(bundle)
    got_manifest_sha256 = _sha256(bundle / "manifest.json")
    if manifest["source"]["commit"] != expected_source_sha:
        raise GateError(
            f"external source authority differs: manifest={manifest['source']['commit']} "
            f"expected={expected_source_sha}")
    if manifest["artifact"]["sha256"] != expected_binary_sha256:
        raise GateError(
            f"external binary authority differs: manifest={manifest['artifact']['sha256']} "
            f"expected={expected_binary_sha256}")
    if got_manifest_sha256 != expected_manifest_sha256:
        raise GateError(
            f"external manifest authority differs: got={got_manifest_sha256} "
            f"expected={expected_manifest_sha256}")
    source_authority = _verify_source_authority(source_tree, manifest["source"])
    sdk_root, sdk_authority = _validate_sdk(sdk_root, manifest["sdk"])
    if verify_only:
        print(
            "L248_Q4_N16K64_PREBUILT_VERIFY PASS "
            f"source_sha={manifest['source']['commit']} "
            f"binary_sha256={manifest['artifact']['sha256']} "
            "kernels=1 aiu_plain_b32=4 universal_tsm_b32x4=16 "
            "fresh_device_execution=0",
            flush=True)
        return {
            "status": "PASS", "mode": "VERIFY_ONLY",
            "source_authority": source_authority,
            "sdk_authority": sdk_authority,
            "payload_files": len(payload_authority),
        }
    if output is None:
        raise GateError("--output is required for device execution")
    if timeout_seconds <= 0:
        raise GateError("--timeout-seconds must be positive")

    runtime_path = sdk_root / "lib" / "libhggc_wrapper.so"
    device = _runtime_device_identity(runtime_path)
    output = _create_output(output)
    environment = dict(os.environ)
    existing_preload = environment.get("LD_PRELOAD", "")
    if existing_preload and pathlib.Path(existing_preload).resolve() != runtime_path.resolve():
        raise GateError("refusing execution with an unrelated LD_PRELOAD")
    environment["LD_PRELOAD"] = str(runtime_path)
    old_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = str(sdk_root / "lib") + (
        ":" + old_library_path if old_library_path else "")

    positive_run = _execute_binary(binary, [], environment, timeout_seconds)
    positive = _parse_probe(positive_run.stdout)
    _assert_positive(positive_run.returncode, positive)
    negative_run = _execute_binary(
        binary, ["--plant-wrong-oracle"], environment, timeout_seconds)
    negative = _parse_probe(negative_run.stdout)
    _assert_negative(negative_run.returncode, negative, positive)

    (output / "positive.log").write_text(
        _execution_log(positive_run), encoding="utf-8")
    (output / "wrong-oracle.log").write_text(
        _execution_log(negative_run), encoding="utf-8")
    shutil.copyfile(bundle / "manifest.json", output / "payload-manifest.json")
    result = {
        "schema": "quactlize.l248-q4-n16k64-device-result",
        "schema_version": 1,
        "status": "PASS",
        "source": source_authority,
        "sdk": {
            **sdk_authority,
        },
        "payload": {
            "manifest_sha256": got_manifest_sha256,
            "binary_sha256": manifest["artifact"]["sha256"],
            "verified_files": payload_authority,
        },
        "device": device,
        "positive": positive,
        "wrong_oracle": negative,
        "execution": {
            "compiler_invocations": 0,
            "device_binary_builds": 0,
            "fresh_device_execution": 1,
            "positive_launches": 1,
            "negative_launches": 1,
        },
    }
    _write_json(output / "result.json", result)
    _result_authority(
        output,
        ["payload-manifest.json", "positive.log", "result.json",
         "wrong-oracle.log"])
    print(
        "L248_Q4_N16K64_PREBUILT_DEVICE PASS "
        f"source_sha={manifest['source']['commit']} "
        f"binary_sha256={manifest['artifact']['sha256']} "
        f"visible_device={device['CUDA_VISIBLE_DEVICES']} "
        "mapping_id=0x51344e3136440001 words=2048 raw_bad=0 sentinel=0 "
        "launch=0/0/0/0 reds=1 fresh_build=0",
        flush=True)
    return result


def _sample_probe_lines() -> tuple[str, str]:
    positive = (
        "FQ_Q4_N16K64_DELIVERY_RAWBIT verdict=PASS layout=3 "
        "mapping_id=0x51344e3136440001 words=2048 raw_bad=0 sentinel=0 "
        "source_hash=0x1111111111111111 want_hash=0x1111111111111111 "
        "got_hash=0x1111111111111111 "
        "first=[index:-1,kblock:-1,ncohort:-1,lane:-1,vreg:-1,"
        "want:0x00000000,got:0x00000000] "
        "launch=[before:0,immediate:0,sync:0,copy:0] plant=none\n")
    negative = (
        "FQ_Q4_N16K64_DELIVERY_RAWBIT verdict=FAIL layout=3 "
        "mapping_id=0x51344e3136440001 words=2048 raw_bad=1 sentinel=0 "
        "source_hash=0x1111111111111111 want_hash=0x2222222222222222 "
        "got_hash=0x1111111111111111 "
        "first=[index:0,kblock:0,ncohort:0,lane:0,vreg:0,"
        "want:0x12345678,got:0x87654321] "
        "launch=[before:0,immediate:0,sync:0,copy:0] plant=wrong-oracle\n")
    return positive, negative


def self_test() -> None:
    base = pathlib.Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    if not base.is_dir():
        raise GateError(f"self-test temporary directory is missing: {base}")
    with tempfile.TemporaryDirectory(prefix="l248-prebuilt-selftest-", dir=base) as raw:
        bundle = pathlib.Path(raw) / "payload"
        (bundle / "bin").mkdir(parents=True)
        files = {
            "manifest.json": b"manifest\n",
            "runner.py": b"runner\n",
            "bin/probe": b"probe\n",
        }
        expected: dict[str, tuple[int, str]] = {}
        for relative, contents in files.items():
            path = bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            expected[relative] = (len(contents), _bytes_sha256(contents))
        (bundle / AUTHORITY_PATH).write_text(
            "".join(
                f"{expected[path][1]}  {path}\n" for path in sorted(expected)),
            encoding="utf-8")
        _verify_authority(bundle, expected)
        (bundle / "bin/probe").write_bytes(b"tampered\n")
        try:
            _verify_authority(bundle, expected)
        except GateError:
            pass
        else:
            raise GateError("self-test authority tamper did not turn red")

    try:
        json.loads('{"x": 1, "x": 2}', object_pairs_hook=_strict_pairs)
    except GateError:
        pass
    else:
        raise GateError("self-test duplicate JSON key did not turn red")
    if _visible_device_ordinal({"CUDA_VISIBLE_DEVICES": "7"}) != "7":
        raise GateError("self-test one-device positive failed")
    for bad in ("", "0,1", "GPU-0", "-1"):
        try:
            _visible_device_ordinal({"CUDA_VISIBLE_DEVICES": bad})
        except GateError:
            continue
        raise GateError(f"self-test one-device plant escaped: {bad!r}")

    def get_count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0

    def get_device(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 0
        return 0

    identity = _runtime_device_identity_from_calls(
        get_count, get_device, {"CUDA_VISIBLE_DEVICES": "3"})
    if identity["hggc_device_count"] != 1 or identity["hggc_current_device"] != 0:
        raise GateError("self-test runtime one-device identity failed")
    sdk_manifest = {
        "release": SDK_RELEASE,
        "files": {
            # These are deliberately unrelated build-tool bytes.  The box
            # execution contract records them but admits any tools from the
            # same release because it never invokes them.
            "bin/hgcc": "a" * 64,
            "bin/hgobjdump": "b" * 64,
            "lib/libhggc_wrapper.so": RUNTIME_SHA256,
        },
    }
    compatibility = _assert_sdk_runtime_compatibility(
        SDK_RELEASE, RUNTIME_SHA256, sdk_manifest)
    if compatibility["build_tool_byte_identity_required_on_box"] is not False:
        raise GateError("self-test same-release build-tool compatibility failed")
    for release, runtime_digest in (
            ("2.1.1-wrong", RUNTIME_SHA256),
            (SDK_RELEASE, "0" * 64)):
        try:
            _assert_sdk_runtime_compatibility(
                release, runtime_digest, sdk_manifest)
        except GateError:
            continue
        raise GateError("self-test SDK release/runtime plant escaped")
    positive_text, negative_text = _sample_probe_lines()
    positive = _parse_probe(positive_text)
    negative = _parse_probe(negative_text)
    _assert_positive(0, positive)
    _assert_negative(1, negative, positive)
    print(
        "L248_Q4_N16K64_PREBUILT_SELF_TEST PASS "
        "authority=PASS duplicate_json=RED one_device=PASS/RED "
        "sdk_same_release_tools=PASS sdk_release_runtime=RED/RED "
        "positive=PASS wrong_oracle=RED reds=6",
        flush=True)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer: {value!r}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer: {value!r}")
    return result


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?", type=pathlib.Path)
    parser.add_argument("--ppu-sdk", type=pathlib.Path)
    parser.add_argument("--source-tree", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--expect-source-sha")
    parser.add_argument("--expect-binary-sha256")
    parser.add_argument("--expect-manifest-sha256")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--timeout-seconds", type=_positive_int, default=120)
    args = parser.parse_args(argv)
    if args.self_test:
        if any(value is not None for value in
               (args.bundle, args.ppu_sdk, args.source_tree, args.output,
                args.expect_source_sha, args.expect_binary_sha256,
                args.expect_manifest_sha256)) or args.verify_only:
            parser.error("--self-test accepts no bundle, SDK, source, output, or verify mode")
        return args
    if args.bundle is None or args.ppu_sdk is None or args.source_tree is None:
        parser.error("bundle, --ppu-sdk, and --source-tree are required")
    for name, length in (("expect_source_sha", 40),
                         ("expect_binary_sha256", 64),
                         ("expect_manifest_sha256", 64)):
        value = getattr(args, name)
        if value is None or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            parser.error(f"--{name.replace('_', '-')} requires {length} lowercase hex digits")
    if not args.verify_only and args.output is None:
        parser.error("--output is required unless --verify-only is selected")
    if args.verify_only and args.output is not None:
        parser.error("--verify-only does not create --output")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.self_test:
            self_test()
        else:
            run_bundle(
                args.bundle, args.ppu_sdk, args.source_tree, args.output,
                args.verify_only, args.timeout_seconds,
                args.expect_source_sha, args.expect_binary_sha256,
                args.expect_manifest_sha256)
    except GateError as exc:
        print(f"[l248-prebuilt] FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

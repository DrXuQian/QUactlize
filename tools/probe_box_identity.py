#!/usr/bin/env python3
"""Resolve one fail-closed PPU box identity and read fields from it.

Production use has two commands::

    python3 tools/probe_box_identity.py resolve --output identity.json
    python3 tools/probe_box_identity.py get --file identity.json \
        --field device_model --part value

The runtime device count is an atomic observation.  A successful observation
of zero or multiple visible devices is never repaired by an operator string:
the operator must instead make exactly one device visible and rerun.  Manual
``QUACTLIZE_BOX_*`` values are fallback evidence only when a probe or one of
its fields is genuinely unavailable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import NoReturn


ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import box_identity_schema as identity_schema

CPP_SOURCE = ROOT / "tools" / "box_identity_probe.cpp"
SCHEMA = identity_schema.SCHEMA
FIELDS = identity_schema.FIELDS
ENV_FIELDS = {
    "device_model": "QUACTLIZE_BOX_DEVICE_MODEL",
    "pci_identity": "QUACTLIZE_BOX_PCI_IDENTITY",
    "driver_version": "QUACTLIZE_BOX_DRIVER_VERSION",
    "sdk_compiler_identity": "QUACTLIZE_BOX_SDK_COMPILER_IDENTITY",
}
SOURCES = identity_schema.SOURCES
REJECTED_VALUES = identity_schema.REJECTED_VALUES
PCI_RE = identity_schema.PCI_RE


class ProbeError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise SystemExit(f"box identity probe: {message}")


def _one_line(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProbeError(f"{field} is empty")
    value = value.strip()
    if value.lower() in REJECTED_VALUES:
        raise ProbeError(f"{field} is not a concrete identity: {value!r}")
    if any(mark in value for mark in ("\n", "\r", "\0")):
        raise ProbeError(f"{field} must be one line")
    return value


def _sdk_root(environ: dict[str, str]) -> tuple[Path | None, str]:
    for variable in ("PPU_SDK", "PPU_HOME", "PPU_SDK_SITE_DEFAULT"):
        value = environ.get(variable, "").strip()
        if value:
            return Path(value), variable
    return None, ""


def _first_nonempty_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _probe_sdk_compiler(environ: dict[str, str]) -> dict[str, object]:
    evidence: dict[str, object] = {
        "status": "unavailable",
        "reason": "",
        "sdk_root_authority": "",
        "sdk_root": "",
        "compiler_path": "",
        "version_first_line": "",
        "identity_value": "",
    }
    root, authority = _sdk_root(environ)
    if root is None:
        evidence["reason"] = "sdk-root-not-set"
        return evidence
    evidence["sdk_root_authority"] = authority
    evidence["sdk_root"] = str(root.resolve(strict=False))
    compiler = root / "bin" / "hgcc"
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        evidence["reason"] = "hgcc-not-executable"
        return evidence
    compiler = compiler.resolve()
    evidence["compiler_path"] = str(compiler)
    completed = subprocess.run(
        [str(compiler), "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    version = _first_nonempty_line(completed.stdout)
    if completed.returncode != 0 or not version:
        evidence["reason"] = "hgcc-version-query-failed"
        return evidence
    value = f"{compiler} :: {version}"
    evidence.update(status="measured", version_first_line=version,
                    identity_value=value)
    return evidence


def _probe_compile_command(sdk_root: Path, output: Path,
                           environ: dict[str, str]) -> list[str]:
    cxx_words = shlex.split(environ.get("CXX", "c++"))
    if not cxx_words:
        raise ProbeError("CXX expands to an empty command")
    command = cxx_words + ["-std=c++17", "-DSWITCH_TO_HGGCRT"]
    # Match third_party/actlize/cmake/PPUToolchain.cmake exactly.  Searching
    # every targets/* tree could silently compile this probe against a
    # different target ABI than the binary whose provenance it records.
    include_dirs = [sdk_root / "include",
                    sdk_root / "targets" / "x86_64-linux" / "include"]
    for directory in include_dirs:
        if directory.is_dir():
            command += ["-I", str(directory)]
    lib_dir = sdk_root / "lib"
    command += [
        str(CPP_SOURCE), "-o", str(output), "-L", str(lib_dir),
        f"-Wl,-rpath,{lib_dir}", "-Wl,--no-as-needed", "-lhg_wrapper",
        "-lhggc_wrapper", "-lhggcrt1", "-lhggc", "-ldl",
    ]
    return command


def _decode_hex(value: str, field: str) -> str:
    if len(value) % 2 or not re.fullmatch(r"[0-9a-f]*", value):
        raise ProbeError(f"malformed {field} hex from runtime probe")
    try:
        return bytes.fromhex(value).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProbeError(f"malformed {field} from runtime probe: {exc}") from exc


def _parse_runtime_wire(stdout: str) -> dict[str, object]:
    lines = stdout.splitlines()
    if not lines or lines[0] != "QZ_HGGC_DEVICE_PROBE_V1":
        raise ProbeError("runtime probe did not emit its versioned header")
    count: int | None = None
    candidates: list[dict[str, object]] = []
    errors: list[dict[str, int]] = []
    driver_version = ""
    driver_method = "unavailable"
    pci_method = "unavailable"
    for line in lines[1:]:
        words = line.split("\t")
        if words[0] == "count" and len(words) == 2 and count is None:
            try:
                count = int(words[1])
            except ValueError as exc:
                raise ProbeError("runtime probe emitted a non-integer count") from exc
            if count < 0:
                raise ProbeError("runtime probe emitted a negative count")
        elif words[0] == "device" and len(words) == 7:
            try:
                ordinal, major, minor, compute_units = map(
                    int, (words[1], words[3], words[4], words[5]))
            except ValueError as exc:
                raise ProbeError("runtime probe emitted a non-integer property") from exc
            pci = "" if words[6] == "-" else words[6].lower()
            if pci and not PCI_RE.fullmatch(pci):
                raise ProbeError(f"runtime probe emitted malformed PCI identity {pci!r}")
            candidates.append({
                "ordinal": ordinal,
                "name": _decode_hex(words[2], "device name"),
                "compute_capability": f"{major}.{minor}",
                "compute_units": compute_units,
                "pci_identity": pci,
            })
        elif words[0] == "device_error" and len(words) == 3:
            try:
                errors.append({"ordinal": int(words[1]), "status": int(words[2])})
            except ValueError as exc:
                raise ProbeError("runtime probe emitted a malformed device error") from exc
        elif words[0] == "pci_method" and len(words) == 2 and pci_method == "unavailable":
            pci_method = "unavailable" if words[1] == "-" else words[1]
        elif words[0] == "driver" and len(words) == 3 and not driver_version:
            driver_version = "" if words[1] == "-" else words[1]
            driver_method = "unavailable" if words[2] == "-" else words[2]
        else:
            raise ProbeError(f"runtime probe emitted an unknown record: {line!r}")
    if count is None:
        raise ProbeError("runtime probe omitted the device count")
    ordinals = [item["ordinal"] for item in candidates]
    error_ordinals = [item["ordinal"] for item in errors]
    if sorted(ordinals + error_ordinals) != list(range(count)):
        raise ProbeError("runtime probe did not account for every reported ordinal")
    return {
        "status": "measured" if not errors else "properties-unavailable",
        "method": "hggcGetDeviceCount+hggcGetDeviceProperties",
        "reason": "" if not errors else "one-or-more-properties-unavailable",
        "device_count": count,
        "candidates": candidates,
        "property_errors": errors,
        "runtime_driver_version": driver_version,
        "selected_ordinal": None,
        "pci_measurement": pci_method,
        "driver_measurement": driver_method,
    }


def _run_runtime_probe(environ: dict[str, str]) -> dict[str, object]:
    unavailable: dict[str, object] = {
        "status": "unavailable",
        "method": "hggcGetDeviceCount+hggcGetDeviceProperties",
        "reason": "",
        "device_count": None,
        "candidates": [],
        "property_errors": [],
        "runtime_driver_version": "",
        "selected_ordinal": None,
        "pci_measurement": "unavailable",
        "driver_measurement": "unavailable",
    }
    if not CPP_SOURCE.is_file():
        raise ProbeError(f"runtime probe source is missing: {CPP_SOURCE}")
    root, _ = _sdk_root(environ)
    if root is None or not root.is_dir():
        unavailable["reason"] = "sdk-root-unavailable"
        return unavailable
    runtime_headers = [root / "include" / "hggc_runtime.h",
                       root / "targets" / "x86_64-linux" / "include" /
                       "hggc_runtime.h"]
    if not any(path.is_file() for path in runtime_headers):
        unavailable["reason"] = "hggc-runtime-header-unavailable"
        return unavailable
    cxx_words = shlex.split(environ.get("CXX", "c++"))
    if not cxx_words or shutil.which(cxx_words[0]) is None:
        unavailable["reason"] = "host-cxx-unavailable"
        return unavailable
    with tempfile.TemporaryDirectory(prefix="quactlize-box-identity-") as temp:
        binary = Path(temp) / "box_identity_probe"
        try:
            command = _probe_compile_command(root.resolve(), binary, environ)
        except ProbeError as exc:
            unavailable["reason"] = str(exc)
            return unavailable
        built = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
        if built.returncode != 0 or not binary.is_file():
            unavailable["reason"] = "runtime-probe-build-failed"
            return unavailable
        ran = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=False)
        if ran.returncode != 0:
            unavailable["reason"] = "runtime-query-failed"
            return unavailable
    try:
        return _parse_runtime_wire(ran.stdout)
    except ProbeError as exc:
        raise ProbeError(f"invalid runtime probe output: {exc}") from exc


def _sysfs_driver_version(pci_identity: str) -> tuple[str, str]:
    if not pci_identity or not PCI_RE.fullmatch(pci_identity):
        return "", "unavailable"
    device = Path("/sys/bus/pci/devices") / pci_identity
    paths = (device / "driver" / "module" / "version",
             device / "driver" / "module" / "srcversion")
    for path in paths:
        try:
            value = _first_nonempty_line(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if value:
            # srcversion is a concrete loaded-module identity, not a semantic
            # release number.  Keep that distinction in the value as well as
            # the evidence method so it cannot masquerade as a version string.
            if path.name == "srcversion":
                value = f"srcversion:{value}"
            return value, f"sysfs:{path.name}"
    return "", "unavailable"


def _candidate_label(candidate: dict[str, object]) -> str:
    name = candidate.get("name") or "<empty-name>"
    pci = candidate.get("pci_identity") or "<pci-unavailable>"
    return (f"ordinal={candidate.get('ordinal')} name={name!r} pci={pci} "
            f"cu={candidate.get('compute_units')}")


def _fallback(field: str, environ: dict[str, str]) -> dict[str, str]:
    variable = ENV_FIELDS[field]
    value = environ.get(variable, "")
    try:
        value = _one_line(value, variable)
    except ProbeError as exc:
        raise ProbeError(
            f"{field} could not be measured and {variable} is not a usable "
            f"operator fallback: {exc}") from exc
    return {"value": value, "source": "operator"}


def _resolve_from_observations(runtime: dict[str, object],
                               sdk: dict[str, object],
                               environ: dict[str, str]) -> dict[str, object]:
    runtime = copy.deepcopy(runtime)
    sdk = copy.deepcopy(sdk)
    candidates = runtime.get("candidates", [])
    if not isinstance(candidates, list):
        raise ProbeError("runtime candidates are malformed")
    status = runtime.get("status")
    selected: dict[str, object] | None = None
    if status in {"measured", "properties-unavailable"}:
        count = runtime.get("device_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ProbeError("successful runtime observation has no integer count")
        if count == 0:
            raise ProbeError("hggc runtime reports 0 visible devices; candidates: <none>")
        if count != 1:
            labels_by_ordinal = {
                item.get("ordinal"): _candidate_label(item) for item in candidates}
            for error in runtime.get("property_errors", []):
                if isinstance(error, dict):
                    labels_by_ordinal[error.get("ordinal")] = (
                        f"ordinal={error.get('ordinal')} <properties unavailable> "
                        f"status={error.get('status')}")
            labels = "; ".join(labels_by_ordinal[key]
                               for key in sorted(labels_by_ordinal)
                               if isinstance(key, int))
            if not labels:
                labels = "<properties unavailable>"
            raise ProbeError(
                f"hggc runtime reports {count} visible devices; candidates: {labels}. "
                "Set CUDA_VISIBLE_DEVICES to exactly one ordinal and rerun; operator "
                "identity strings do not disambiguate devices.")
        if len(candidates) == 1 and not runtime.get("property_errors"):
            selected = candidates[0]
    elif status != "unavailable":
        raise ProbeError(f"unknown runtime probe status {status!r}")

    measured: dict[str, str] = {}
    # Driver identity is a process-wide runtime query, independent of whether
    # hggcGetDeviceProperties succeeded for the one visible ordinal.  A
    # per-device property failure must not demote this already measured field
    # to a lower-grade operator assertion.
    driver = runtime.get("runtime_driver_version")
    if (runtime.get("device_count") == 1 and isinstance(driver, str) and
            driver.strip()):
        measured["driver_version"] = driver.strip()
    if selected is not None:
        if isinstance(selected.get("name"), str) and selected["name"].strip():
            measured["device_model"] = selected["name"].strip()
        if isinstance(selected.get("pci_identity"), str) and selected["pci_identity"].strip():
            measured["pci_identity"] = selected["pci_identity"].strip().lower()
        if "driver_version" not in measured and "pci_identity" in measured:
            driver, method = _sysfs_driver_version(measured["pci_identity"])
            if driver:
                measured["driver_version"] = driver
                runtime["runtime_driver_version"] = driver
                runtime["driver_measurement"] = method
            else:
                runtime["driver_measurement"] = "unavailable"
        runtime["selected_ordinal"] = selected["ordinal"]
        if "pci_identity" not in measured:
            runtime["pci_measurement"] = "unavailable"

    if sdk.get("status") == "measured":
        value = sdk.get("identity_value")
        if isinstance(value, str) and value.strip():
            measured["sdk_compiler_identity"] = value.strip()

    identity: dict[str, dict[str, str]] = {}
    for field in FIELDS:
        if field in measured:
            identity[field] = {
                "value": _one_line(measured[field], field),
                "source": "measured",
            }
        else:
            identity[field] = _fallback(field, environ)
    return {
        "schema": SCHEMA,
        "identity": identity,
        "device_probe": runtime,
        "sdk_compiler_probe": sdk,
    }


def _validate_document(document: object) -> dict[str, object]:
    if not isinstance(document, dict) or set(document) != {
            "schema", "identity", "device_probe", "sdk_compiler_probe"}:
        raise ProbeError("identity probe JSON has the wrong top-level keys")
    if document["schema"] != SCHEMA:
        raise ProbeError(f"identity probe JSON has unknown schema {document['schema']!r}")
    identity = document["identity"]
    if not isinstance(identity, dict) or set(identity) != set(FIELDS):
        raise ProbeError("identity probe JSON has the wrong identity fields")
    for field in FIELDS:
        entry = identity[field]
        if not isinstance(entry, dict) or set(entry) != {"value", "source"}:
            raise ProbeError(f"identity field {field} has the wrong shape")
        _one_line(entry["value"], field)
        if entry["source"] not in SOURCES:
            raise ProbeError(f"identity field {field} has invalid source {entry['source']!r}")
    if not isinstance(document["device_probe"], dict):
        raise ProbeError("device_probe must be an object")
    device_probe = document["device_probe"]
    if set(device_probe) != {
            "status", "method", "reason", "device_count", "candidates",
            "property_errors", "runtime_driver_version", "selected_ordinal",
            "pci_measurement", "driver_measurement"}:
        raise ProbeError("device_probe has the wrong evidence keys")
    if device_probe["status"] not in {
            "measured", "properties-unavailable", "unavailable"}:
        raise ProbeError("device_probe has an invalid status")
    if not all(isinstance(device_probe[key], str) for key in (
            "method", "reason", "runtime_driver_version", "pci_measurement",
            "driver_measurement")):
        raise ProbeError("device_probe string evidence is malformed")
    if (device_probe["device_count"] is not None and
            (isinstance(device_probe["device_count"], bool) or
             not isinstance(device_probe["device_count"], int) or
             device_probe["device_count"] < 0)):
        raise ProbeError("device_probe device_count is malformed")
    if (device_probe["selected_ordinal"] is not None and
            (isinstance(device_probe["selected_ordinal"], bool) or
             not isinstance(device_probe["selected_ordinal"], int) or
             device_probe["selected_ordinal"] < 0)):
        raise ProbeError("device_probe selected_ordinal is malformed")
    if not isinstance(device_probe["candidates"], list):
        raise ProbeError("device_probe candidates must be a list")
    for candidate in device_probe["candidates"]:
        if not isinstance(candidate, dict) or set(candidate) != {
                "ordinal", "name", "compute_capability", "compute_units",
                "pci_identity"}:
            raise ProbeError("device_probe candidate has the wrong shape")
        if (isinstance(candidate["ordinal"], bool) or
                not isinstance(candidate["ordinal"], int) or
                isinstance(candidate["compute_units"], bool) or
                not isinstance(candidate["compute_units"], int)):
            raise ProbeError("device_probe candidate integers are malformed")
        if not all(isinstance(candidate[key], str) for key in (
                "name", "compute_capability", "pci_identity")):
            raise ProbeError("device_probe candidate strings are malformed")
    if not isinstance(device_probe["property_errors"], list):
        raise ProbeError("device_probe property_errors must be a list")
    for error in device_probe["property_errors"]:
        if (not isinstance(error, dict) or set(error) != {"ordinal", "status"} or
                any(isinstance(error[key], bool) or not isinstance(error[key], int)
                    for key in ("ordinal", "status"))):
            raise ProbeError("device_probe property error is malformed")
    if not isinstance(document["sdk_compiler_probe"], dict):
        raise ProbeError("sdk_compiler_probe must be an object")
    sdk_probe = document["sdk_compiler_probe"]
    if set(sdk_probe) != {
            "status", "reason", "sdk_root_authority", "sdk_root",
            "compiler_path", "version_first_line", "identity_value"}:
        raise ProbeError("sdk_compiler_probe has the wrong evidence keys")
    if sdk_probe["status"] not in {"measured", "unavailable"}:
        raise ProbeError("sdk_compiler_probe has an invalid status")
    if not all(isinstance(sdk_probe[key], str) for key in sdk_probe if key != "status"):
        raise ProbeError("sdk_compiler_probe string evidence is malformed")
    try:
        identity_schema.validate(document)
    except identity_schema.IdentityProbeError as exc:
        raise ProbeError(str(exc)) from exc
    return document


def _canonical_json_bytes(document: dict[str, object]) -> bytes:
    """The SHA authority: sorted compact UTF-8 JSON, with no trailing LF."""
    _validate_document(document)
    return identity_schema.canonical_bytes(document)


def _canonical_output_bytes(document: dict[str, object]) -> bytes:
    """Human-friendly file form; its JSON payload hashes via _canonical_json_bytes."""
    return _canonical_json_bytes(document) + b"\n"


def _atomic_write(path: Path, data: bytes) -> None:
    if not data:
        raise ProbeError("refusing to write an empty identity probe")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resolve_command(args: argparse.Namespace) -> int:
    environ = dict(os.environ)
    sdk = _probe_sdk_compiler(environ)
    runtime = _run_runtime_probe(environ)
    try:
        document = _resolve_from_observations(runtime, sdk, environ)
        _atomic_write(Path(args.output), _canonical_output_bytes(document))
    except ProbeError as exc:
        fail(str(exc))
    return 0


def get_command(args: argparse.Namespace) -> int:
    try:
        document = json.loads(Path(args.file).read_text(encoding="utf-8"))
        document = _validate_document(document)
    except (OSError, json.JSONDecodeError, ProbeError) as exc:
        fail(f"cannot read validated identity probe {args.file}: {exc}")
    print(document["identity"][args.field][args.part])
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--output", required=True)
    resolve.set_defaults(func=resolve_command)
    get = subparsers.add_parser("get")
    get.add_argument("--file", required=True)
    get.add_argument("--field", choices=FIELDS, required=True)
    get.add_argument("--part", choices=("value", "source"), required=True)
    get.set_defaults(func=get_command)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as exc:
        fail(str(exc))

#!/usr/bin/env python3
"""Pure-local falsification controls for tools/probe_box_identity.py."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "probe_box_identity", ROOT / "tools" / "probe_box_identity.py")
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
WRITER_SPEC = importlib.util.spec_from_file_location(
    "write_box_run_provenance", ROOT / "tools" / "write_box_run_provenance.py")
assert WRITER_SPEC and WRITER_SPEC.loader
WRITER = importlib.util.module_from_spec(WRITER_SPEC)
WRITER_SPEC.loader.exec_module(WRITER)


ENV = {
    "QUACTLIZE_BOX_DEVICE_MODEL": "operator-model",
    "QUACTLIZE_BOX_PCI_IDENTITY": "0000:01:00.0",
    "QUACTLIZE_BOX_DRIVER_VERSION": "operator-driver",
    "QUACTLIZE_BOX_SDK_COMPILER_IDENTITY": "operator-sdk",
}


def runtime(candidates, driver="12010"):
    return {
        "status": "measured",
        "method": "fixture",
        "reason": "",
        "device_count": len(candidates),
        "candidates": candidates,
        "property_errors": [],
        "runtime_driver_version": driver,
        "selected_ordinal": None,
        "pci_measurement": "fixture" if any(
            row.get("pci_identity") for row in candidates) else "unavailable",
        "driver_measurement": "fixture" if driver else "unavailable",
    }


def candidate(ordinal, name, pci):
    return {"ordinal": ordinal, "name": name, "compute_capability": "10.0",
            "compute_units": 72, "pci_identity": pci}


def expect_red(label, operation, needle):
    try:
        operation()
    except PROBE.ProbeError as exc:
        assert needle in str(exc), (label, str(exc))
        return
    raise AssertionError(f"{label}: unexpectedly green")


def compiled_runtime_probe():
    """Compile the real C++ probe against a controllable runtime seam."""
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        (temp / "hggc_runtime.h").write_text(r'''#pragma once
#include <cstddef>
typedef int hggcError_t;
enum { hggcSuccess = 0 };
struct hggcDeviceProp {
  char name[64]; int major; int minor; int multiProcessorCount;
};
extern "C" hggcError_t hggcGetDeviceCount(int*);
extern "C" hggcError_t hggcGetDeviceProperties(hggcDeviceProp*, int);
extern "C" const char* hggcGetErrorName(hggcError_t);
extern "C" const char* hggcGetErrorString(hggcError_t);
''')
        (temp / "mock_runtime.cpp").write_text(r'''#include "hggc_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" hggcError_t hggcGetDeviceCount(int* n) {
  *n = std::atoi(std::getenv("QZ_MOCK_COUNT") ?: "1"); return 0;
}
extern "C" hggcError_t hggcGetDeviceProperties(hggcDeviceProp* p, int i) {
  std::snprintf(p->name, sizeof(p->name), "Fixture-PPU-%d", i);
  p->major=10; p->minor=0; p->multiProcessorCount=72+i; return 0;
}
extern "C" const char* hggcGetErrorName(hggcError_t) { return "fixture"; }
extern "C" const char* hggcGetErrorString(hggcError_t) { return "fixture"; }
extern "C" int hggcDeviceGetPCIBusId(char* out, int size, int device) {
  std::snprintf(out, size, "0000:%02x:00.0", device + 2); return 0;
}
extern "C" int hggcDriverGetVersion(int* version) { *version=12010; return 0; }
''')
        binary = temp / "probe"
        built = subprocess.run([
            os.environ.get("CXX", "c++"), "-std=c++17", "-rdynamic",
            "-I", str(temp), str(ROOT / "tools" / "box_identity_probe.cpp"),
            str(temp / "mock_runtime.cpp"), "-ldl", "-o", str(binary),
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert built.returncode == 0, built.stdout
        one = subprocess.run([str(binary)], env={**os.environ, "QZ_MOCK_COUNT": "1"},
                             text=True, stdout=subprocess.PIPE, check=True)
        parsed_one = PROBE._parse_runtime_wire(one.stdout)
        assert parsed_one["device_count"] == 1
        assert parsed_one["candidates"][0]["name"] == "Fixture-PPU-0"
        assert parsed_one["candidates"][0]["pci_identity"] == "0000:02:00.0"
        assert parsed_one["pci_measurement"] == "hggcDeviceGetPCIBusId"
        assert parsed_one["runtime_driver_version"] == "12010"
        assert parsed_one["driver_measurement"] == "hggcDriverGetVersion"
        two = subprocess.run([str(binary)], env={**os.environ, "QZ_MOCK_COUNT": "2"},
                             text=True, stdout=subprocess.PIPE, check=True)
        parsed_two = PROBE._parse_runtime_wire(two.stdout)
        assert [row["ordinal"] for row in parsed_two["candidates"]] == [0, 1]
        return parsed_one, parsed_two


def main():
    sdk = {"status": "measured", "reason": "", "sdk_root_authority": "PPU_SDK",
           "sdk_root": "/sdk", "compiler_path": "/sdk/bin/hgcc",
           "version_first_line": "fixture",
           "identity_value": "/sdk/bin/hgcc :: fixture"}
    one = runtime([candidate(0, "PPU-ZW810", "0000:02:00.0")])
    document = PROBE._resolve_from_observations(one, sdk, dict(ENV))
    assert document["identity"] == {
        "device_model": {"value": "PPU-ZW810", "source": "measured"},
        "pci_identity": {"value": "0000:02:00.0", "source": "measured"},
        "driver_version": {"value": "12010", "source": "measured"},
        "sdk_compiler_identity": {
            "value": "/sdk/bin/hgcc :: fixture", "source": "measured"},
    }
    # Operator strings are fallback, not assertions capable of overriding a
    # contradictory measured value.
    assert document["identity"]["device_model"]["value"] != ENV[
        "QUACTLIZE_BOX_DEVICE_MODEL"]

    two = runtime([candidate(0, "PPU-A", "0000:02:00.0"),
                   candidate(1, "PPU-B", "0000:03:00.0")])
    expect_red("multiple visible devices", lambda: PROBE._resolve_from_observations(
        two, sdk, dict(ENV)), "PPU-A")
    expect_red("zero visible devices", lambda: PROBE._resolve_from_observations(
        runtime([]), sdk, dict(ENV)), "0 visible devices")

    partial_properties = runtime([], driver="12010")
    partial_properties.update(
        status="properties-unavailable", device_count=1,
        property_errors=[{"ordinal": 0, "status": 17}])
    partial_doc = PROBE._resolve_from_observations(
        partial_properties, sdk, dict(ENV))
    assert partial_doc["identity"]["device_model"]["source"] == "operator"
    assert partial_doc["identity"]["pci_identity"]["source"] == "operator"
    assert partial_doc["identity"]["driver_version"] == {
        "value": "12010", "source": "measured"}

    # A successful single-device probe may have an unavailable individual
    # field. Only that field falls back; an empty value can never enter JSON.
    sparse = runtime([candidate(0, "PPU-ZW810", "")], driver="")
    sparse_doc = PROBE._resolve_from_observations(sparse, sdk, dict(ENV))
    assert sparse_doc["identity"]["device_model"]["source"] == "measured"
    assert sparse_doc["identity"]["pci_identity"] == {
        "value": "0000:01:00.0", "source": "operator"}
    assert sparse_doc["identity"]["driver_version"]["source"] == "operator"
    no_pci_env = dict(ENV)
    no_pci_env["QUACTLIZE_BOX_PCI_IDENTITY"] = ""
    expect_red("empty auto and operator value", lambda: PROBE._resolve_from_observations(
        sparse, sdk, no_pci_env), "is empty")

    compiled_one, compiled_two = compiled_runtime_probe()
    compiled_doc = PROBE._resolve_from_observations(compiled_one, sdk, dict(ENV))
    assert compiled_doc["identity"]["pci_identity"] == {
        "value": "0000:02:00.0", "source": "measured"}
    expect_red("compiled two-device runtime", lambda: PROBE._resolve_from_observations(
        compiled_two, sdk, dict(ENV)), "Fixture-PPU-1")

    canonical = PROBE._canonical_json_bytes(document)
    assert not canonical.endswith(b"\n")
    assert canonical == PROBE._canonical_json_bytes(json.loads(canonical))
    non_ascii = json.loads(canonical)
    non_ascii["identity"]["device_model"]["value"] = "PPU-测试"
    non_ascii["device_probe"]["candidates"][0]["name"] = "PPU-测试"
    non_ascii_canonical = PROBE._canonical_json_bytes(non_ascii)
    assert b"\\u6d4b\\u8bd5" in non_ascii_canonical
    output = PROBE._canonical_output_bytes(document)
    assert output == canonical + b"\n"
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "identity.json"
        PROBE._atomic_write(path, output)
        loaded = PROBE._validate_document(json.loads(path.read_text()))
        assert loaded["identity"]["device_model"]["source"] == "measured"
        unicode_path = Path(temp) / "unicode-identity.json"
        PROBE._atomic_write(
            unicode_path, PROBE._canonical_output_bytes(non_ascii))
        _, _, writer_digest = WRITER.read_identity_probe(unicode_path)
        assert writer_digest == hashlib.sha256(non_ascii_canonical).hexdigest(), (
            "probe producer and provenance writer disagree on canonical bytes")

    print("box identity probe: PASS (compiled runtime, unique measured, 0/multi red, empty fallback, canonical schema)")


if __name__ == "__main__":
    main()

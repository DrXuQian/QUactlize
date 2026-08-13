#!/usr/bin/env python3
"""One semantic authority for the box identity probe artifact.

The probe, provenance writer, and offline adjudicator all consume this module.
That is deliberate: a source label is evidence only when it agrees with the
measured fields below, not merely because three readers accept the same JSON
shape or digest.
"""

from __future__ import annotations

import json
import re
from typing import Any


SCHEMA = "quactlize-box-identity-probe-v1"
FIELDS = (
    "device_model",
    "pci_identity",
    "driver_version",
    "sdk_compiler_identity",
)
SOURCES = {"measured", "operator"}
REJECTED_VALUES = {"unknown", "unset", "n/a", "na", "none"}
PCI_RE = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\Z")


class IdentityProbeError(ValueError):
    pass


def concrete(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdentityProbeError(f"{field} is empty")
    value = value.strip()
    if value.lower() in REJECTED_VALUES:
        raise IdentityProbeError(
            f"{field} is not a concrete identity: {value!r}")
    if any(mark in value for mark in ("\n", "\r", "\0")):
        raise IdentityProbeError(f"{field} must be one line")
    return value


def _optional_concrete(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    value = value.strip()
    if value.lower() in REJECTED_VALUES or any(
            mark in value for mark in ("\n", "\r", "\0")):
        return ""
    return value


def validate(document: object) -> dict[str, object]:
    if not isinstance(document, dict) or set(document) != {
            "schema", "identity", "device_probe", "sdk_compiler_probe"}:
        raise IdentityProbeError("identity probe JSON has the wrong top-level keys")
    if document["schema"] != SCHEMA:
        raise IdentityProbeError(
            f"identity probe JSON has unknown schema {document['schema']!r}")

    identity = document["identity"]
    if not isinstance(identity, dict) or set(identity) != set(FIELDS):
        raise IdentityProbeError("identity probe JSON has the wrong identity fields")
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for field in FIELDS:
        entry = identity[field]
        if not isinstance(entry, dict) or set(entry) != {"value", "source"}:
            raise IdentityProbeError(f"identity field {field} has the wrong shape")
        values[field] = concrete(entry["value"], field)
        if entry["source"] not in SOURCES:
            raise IdentityProbeError(
                f"identity field {field} has invalid source {entry['source']!r}")
        sources[field] = entry["source"]

    device = document["device_probe"]
    device_keys = {
        "status", "method", "reason", "device_count", "candidates",
        "property_errors", "runtime_driver_version", "selected_ordinal",
        "pci_measurement", "driver_measurement",
    }
    if not isinstance(device, dict) or set(device) != device_keys:
        raise IdentityProbeError("device_probe has the wrong evidence keys")
    if device["status"] not in {
            "measured", "properties-unavailable", "unavailable"}:
        raise IdentityProbeError("device_probe has an invalid status")
    if not all(isinstance(device[key], str) for key in (
            "method", "reason", "runtime_driver_version", "pci_measurement",
            "driver_measurement")):
        raise IdentityProbeError("device_probe string evidence is malformed")
    for key in ("device_count", "selected_ordinal"):
        item = device[key]
        if item is not None and (isinstance(item, bool) or
                                 not isinstance(item, int) or item < 0):
            raise IdentityProbeError(f"device_probe {key} is malformed")
    if not isinstance(device["candidates"], list):
        raise IdentityProbeError("device_probe candidates must be a list")
    for candidate in device["candidates"]:
        if not isinstance(candidate, dict) or set(candidate) != {
                "ordinal", "name", "compute_capability", "compute_units",
                "pci_identity"}:
            raise IdentityProbeError("device_probe candidate has the wrong shape")
        if (isinstance(candidate["ordinal"], bool) or
                not isinstance(candidate["ordinal"], int) or
                candidate["ordinal"] < 0 or
                isinstance(candidate["compute_units"], bool) or
                not isinstance(candidate["compute_units"], int) or
                candidate["compute_units"] < 0 or
                any(not isinstance(candidate[key], str) for key in (
                    "name", "compute_capability", "pci_identity"))):
            raise IdentityProbeError("device_probe candidate values are malformed")
        pci = candidate["pci_identity"]
        if pci and not PCI_RE.fullmatch(pci):
            raise IdentityProbeError("device_probe candidate PCI identity is malformed")
    if not isinstance(device["property_errors"], list):
        raise IdentityProbeError("device_probe property_errors must be a list")
    for error in device["property_errors"]:
        if (not isinstance(error, dict) or set(error) != {"ordinal", "status"} or
                any(isinstance(error[key], bool) or not isinstance(error[key], int)
                    for key in ("ordinal", "status"))):
            raise IdentityProbeError("device_probe property error is malformed")

    status = device["status"]
    candidates = device["candidates"]
    property_errors = device["property_errors"]
    if status == "unavailable":
        if (device["device_count"] is not None or
                device["selected_ordinal"] is not None or candidates or
                property_errors):
            raise IdentityProbeError(
                "unavailable device probe contains a guessed device selection")
    elif status == "measured":
        if (device["device_count"] != 1 or len(candidates) != 1 or
                property_errors or device["selected_ordinal"] !=
                candidates[0]["ordinal"]):
            raise IdentityProbeError(
                "measured device probe is not one uniquely selected device")
    else:
        if (device["device_count"] != 1 or candidates or
                len(property_errors) != 1 or
                property_errors[0]["ordinal"] != 0 or
                device["selected_ordinal"] is not None):
            raise IdentityProbeError(
                "property-unavailable device probe is not one failed ordinal")

    expected_measured: dict[str, str] = {}
    if status == "measured":
        candidate = candidates[0]
        name = _optional_concrete(candidate["name"])
        if name:
            expected_measured["device_model"] = name
        pci = _optional_concrete(candidate["pci_identity"])
        if pci:
            expected_measured["pci_identity"] = pci.lower()
            if device["pci_measurement"] == "unavailable":
                raise IdentityProbeError(
                    "measured PCI identity has no measurement method")
    driver = _optional_concrete(device["runtime_driver_version"])
    if device["device_count"] == 1 and driver:
        expected_measured["driver_version"] = driver
        if device["driver_measurement"] == "unavailable":
            raise IdentityProbeError(
                "measured driver identity has no measurement method")

    sdk = document["sdk_compiler_probe"]
    sdk_keys = {
        "status", "reason", "sdk_root_authority", "sdk_root",
        "compiler_path", "version_first_line", "identity_value",
    }
    if not isinstance(sdk, dict) or set(sdk) != sdk_keys:
        raise IdentityProbeError("sdk_compiler_probe has the wrong evidence keys")
    if sdk["status"] not in {"measured", "unavailable"}:
        raise IdentityProbeError("sdk_compiler_probe has an invalid status")
    if not all(isinstance(sdk[key], str) for key in sdk if key != "status"):
        raise IdentityProbeError("sdk_compiler_probe string evidence is malformed")
    if sdk["status"] == "measured":
        for key in ("sdk_root_authority", "sdk_root", "compiler_path",
                    "version_first_line", "identity_value"):
            concrete(sdk[key], f"sdk_compiler_probe.{key}")
        expected_measured["sdk_compiler_identity"] = sdk["identity_value"].strip()
    elif sdk["identity_value"]:
        raise IdentityProbeError(
            "unavailable SDK compiler probe contains a measured identity value")

    for field in FIELDS:
        expected = expected_measured.get(field)
        if expected is None:
            if sources[field] != "operator":
                raise IdentityProbeError(
                    f"identity field {field} claims measured without evidence")
        else:
            if sources[field] != "measured":
                raise IdentityProbeError(
                    f"identity field {field} uses operator despite measured evidence")
            if values[field] != expected:
                raise IdentityProbeError(
                    f"identity field {field} differs from measured evidence")
    return document


def values_and_sources(document: object) -> tuple[dict[str, str], dict[str, str]]:
    value = validate(document)
    identity = value["identity"]
    return (
        {field: identity[field]["value"].strip() for field in FIELDS},
        {field: identity[field]["source"] for field in FIELDS},
    )


def canonical_bytes(document: object) -> bytes:
    value = validate(document)
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")

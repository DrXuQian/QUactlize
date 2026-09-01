#!/usr/bin/env python3
"""Verify the host-only K-pack selected-config ABI in a runtime bundle.

This oracle creates no PPU context and launches no kernel.  It first binds the
six manifest-owned libraries, requires the two selected-config exports in each
one, and then exercises the FMT2/Q2 library at five policy boundaries:

* an exact measured dense point;
* an unmeasured dense point falling back to the compiled shipping default;
* an explicit config overriding the measured point;
* an unknown explicit config failing closed and clearing its output;
* the grouped compiled default (grouped measured routing is not public yet).

The exact measured row is deliberately fixed.  A regenerated policy that
changes it must update this admission oracle in the same reviewed change.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quactlize import ppu_bundle  # noqa: E402


class ArrangementV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int32),
        ("layout", ctypes.c_int32),
        ("bits", ctypes.c_int32),
        ("high_bits", ctypes.c_int32),
        ("artifact_tile_k", ctypes.c_int32),
        ("transport_tile_k", ctypes.c_int32),
        ("group_size", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("mapping_id", ctypes.c_uint64),
    ]


class ConfigV3(ctypes.Structure):
    _fields_ = [
        ("enable_cuda_kernel", ctypes.c_bool),
        ("name", ctypes.c_char_p),
        ("tile_m", ctypes.c_int32),
        ("tile_n", ctypes.c_int32),
        ("tactic_tile_k", ctypes.c_int32),
        ("artifact_tile_k", ctypes.c_int32),
        ("warp_m", ctypes.c_int32),
        ("warp_n", ctypes.c_int32),
        ("stages", ctypes.c_int32),
    ]


class ConfigV4(ctypes.Structure):
    _fields_ = ConfigV3._fields_ + [("split_k_slices", ctypes.c_int32)]


ARRP = ctypes.POINTER(ArrangementV2)


@dataclass(frozen=True)
class ExpectedConfig:
    name: str
    tile_m: int
    tile_n: int
    tactic_tile_k: int
    artifact_tile_k: int
    warp_m: int
    warp_n: int
    stages: int
    split_k_slices: int | None = None


EXACT_DENSE = ExpectedConfig(
    "32x32:16x16:s3", 32, 32, 256, 0, 16, 16, 3, 1)
COMPILED_DECODE_DEFAULT = ExpectedConfig(
    "8x128:8x32:s3", 8, 128, 256, 0, 8, 32, 3, 1)
GROUPED_DEFAULT = ExpectedConfig(
    "16x128:16x16:s2", 16, 128, 256, 0, 16, 16, 2)


class OracleError(RuntimeError):
    pass


def _require_layouts() -> None:
    expected = {
        "ArrangementV2": (ArrangementV2, 40, {"mapping_id": 32}),
        "ConfigV3": (ConfigV3, 48, {"name": 8, "stages": 40}),
        "ConfigV4": (ConfigV4, 48, {"name": 8, "split_k_slices": 44}),
    }
    for name, (ctype, size, offsets) in expected.items():
        if ctypes.sizeof(ctype) != size:
            raise OracleError(
                f"{name} ctypes size is {ctypes.sizeof(ctype)}, expected {size}")
        for field, offset in offsets.items():
            got = getattr(ctype, field).offset
            if got != offset:
                raise OracleError(
                    f"{name}.{field} offset is {got}, expected {offset}")


def _symbol(library: ctypes.CDLL, name: str):
    try:
        return getattr(library, name)
    except AttributeError as exc:
        raise OracleError(
            f"{pathlib.Path(library._name).name} lacks {name}; this is a "
            "legacy runtime bundle, not a selected-config ABI bundle") from exc


def _load_libraries(bundle: pathlib.Path) -> dict[str, ctypes.CDLL]:
    manifest = ppu_bundle.verify_bundle(bundle, inspect_binaries=False)
    entries = {entry["role"]: entry for entry in manifest["libraries"]}
    libraries: dict[str, ctypes.CDLL] = {}
    for role in ppu_bundle.LIBRARY_ROLES:
        path = bundle / entries[role.role]["filename"]
        inspected = subprocess.run(
            ["nm", "-D", "--defined-only", str(path)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if inspected.returncode:
            raise OracleError(
                f"cannot inspect exports in {path}: {inspected.stdout.strip()}")
        exports = {
            line.split()[-1] for line in inspected.stdout.splitlines()
            if line.split()
        }
        missing = sorted(ppu_bundle.SELECTED_CONFIG_REQUIRED_EXPORTS - exports)
        if missing:
            raise OracleError(
                f"{role.filename} lacks {missing}; this is a legacy runtime "
                "bundle, not a selected-config ABI bundle")
        try:
            library = ctypes.CDLL(
                str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        except OSError as exc:
            raise OracleError(f"cannot load {path}: {exc}") from exc
        identity = _symbol(library, "quactlize_ppu_build_packed_format_v1")
        identity.argtypes = []
        identity.restype = ctypes.c_int32
        expected_identity = -1 if role.packed_format is None else role.packed_format
        got_identity = int(identity())
        if got_identity != expected_identity:
            raise OracleError(
                f"{role.filename} reports packed format {got_identity}, "
                f"expected {expected_identity}")
        for required in ppu_bundle.SELECTED_CONFIG_REQUIRED_EXPORTS:
            _symbol(library, required)
        libraries[role.role] = library
    return libraries


def _config_values(value: ConfigV3 | ConfigV4) -> tuple:
    if value.name is None:
        raise OracleError("successful selected-config query returned a null name")
    result = (
        bool(value.enable_cuda_kernel), value.name.decode("utf-8"),
        int(value.tile_m), int(value.tile_n), int(value.tactic_tile_k),
        int(value.artifact_tile_k), int(value.warp_m), int(value.warp_n),
        int(value.stages),
    )
    if isinstance(value, ConfigV4):
        result += (int(value.split_k_slices),)
    return result


def _expect_config(
        label: str, value: ConfigV3 | ConfigV4, expected: ExpectedConfig) -> str:
    want = (
        False, expected.name, expected.tile_m, expected.tile_n,
        expected.tactic_tile_k, expected.artifact_tile_k,
        expected.warp_m, expected.warp_n, expected.stages,
    )
    if isinstance(value, ConfigV4):
        want += (expected.split_k_slices,)
    got = _config_values(value)
    if got != want:
        raise OracleError(f"{label} differs: got={got!r} expected={want!r}")
    return expected.name


def verify_selected_config(bundle: pathlib.Path) -> dict[str, str]:
    bundle = bundle.resolve()
    _require_layouts()
    libraries = _load_libraries(bundle)
    library = libraries["fmt2"]

    canonical = _symbol(library, "quactlize_ppu_canonical_arrangement_v2")
    canonical.argtypes = [ctypes.c_int, ARRP]
    canonical.restype = ctypes.c_int
    arrangement = ArrangementV2()
    rc = int(canonical(10, ctypes.byref(arrangement)))
    expected_arrangement = (2, 2, 2, 0, 0, 128, 16, 0,
                            0x514B504B54000001)
    got_arrangement = tuple(
        int(getattr(arrangement, name)) for name, _ in ArrangementV2._fields_)
    if rc != 0 or got_arrangement != expected_arrangement:
        raise OracleError(
            f"FMT2 canonical Q2 arrangement differs: rc={rc} "
            f"got={got_arrangement!r} expected={expected_arrangement!r}")

    dense = _symbol(
        library,
        "quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2")
    dense.argtypes = [
        ctypes.POINTER(ConfigV4), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ARRP, ctypes.c_char_p,
    ]
    dense.restype = ctypes.c_int32

    def dense_query(m: int, requested: bytes | None) -> tuple[int, ConfigV4]:
        out = ConfigV4()
        rc_ = int(dense(ctypes.byref(out), m, 256, 3072, 16, 10,
                        ctypes.byref(arrangement), requested))
        return rc_, out

    rc, exact = dense_query(1, None)
    if rc != 1:
        raise OracleError(f"exact measured dense query returned {rc}, expected 1")
    exact_name = _expect_config("exact measured dense", exact, EXACT_DENSE)

    # M=3 is deliberately absent from the measured dynamic-value census while
    # every other field matches the exact family above.
    rc, fallback = dense_query(3, None)
    if rc != 1:
        raise OracleError(f"unmeasured dense fallback returned {rc}, expected 1")
    fallback_name = _expect_config(
        "unmeasured dense compiled default", fallback, COMPILED_DECODE_DEFAULT)

    rc, explicit = dense_query(1, COMPILED_DECODE_DEFAULT.name.encode("ascii"))
    if rc != 1:
        raise OracleError(f"explicit dense override returned {rc}, expected 1")
    explicit_name = _expect_config(
        "explicit dense override", explicit, COMPILED_DECODE_DEFAULT)

    stale = ConfigV4()
    ctypes.memset(ctypes.byref(stale), 0x5A, ctypes.sizeof(stale))
    rc = int(dense(ctypes.byref(stale), 1, 256, 3072, 16, 10,
                   ctypes.byref(arrangement), b"stale-config"))
    if rc != 0 or bytes(stale) != bytes(ctypes.sizeof(stale)):
        raise OracleError(
            f"stale dense config did not fail closed: rc={rc} "
            f"cleared={bytes(stale) == bytes(ctypes.sizeof(stale))}")

    grouped = _symbol(
        library,
        "quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2")
    grouped.argtypes = [
        ctypes.POINTER(ConfigV3), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ARRP,
        ctypes.c_char_p,
    ]
    grouped.restype = ctypes.c_int32
    grouped_out = ConfigV3()
    rc = int(grouped(
        ctypes.byref(grouped_out), 8, 512, 2048, 16, 256, 1, 10,
        ctypes.byref(arrangement), None))
    if rc != 1:
        raise OracleError(f"grouped compiled-default query returned {rc}, expected 1")
    grouped_name = _expect_config(
        "grouped compiled default", grouped_out, GROUPED_DEFAULT)

    return {
        "dense_exact": exact_name,
        "dense_unmeasured": fallback_name,
        "dense_explicit": explicit_name,
        "dense_stale": "FAIL_CLOSED",
        "grouped_default": grouped_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=pathlib.Path,
                        help="six-library PPU runtime bundle root")
    args = parser.parse_args(argv)
    try:
        result = verify_selected_config(args.bundle)
    except (OracleError, ppu_bundle.BundleError) as exc:
        print(f"[kquant-selected-config] FAIL: {exc}")
        return 1
    print(
        "[kquant-selected-config] PASS "
        + " ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

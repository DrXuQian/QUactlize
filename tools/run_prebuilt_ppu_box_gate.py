#!/usr/bin/env python3
"""Execute the canonical K-pack gate using only a prebuilt six-library bundle.

Nothing in this program compiles or links code.  The box consumes the six
manifest-owned shared libraries, the installed SDK runtime, NumPy, and the
official GGUF decoder.  Host artifact placement/recovery is called through
ctypes; device storage/copies/synchronization use the SDK runtime; and dense
and grouped kernels use the public arrangement-v2 workspace/device ABI.

Admission is fail-closed: the bundle manifest, hashes, exports, embedded PPU
images, source identity, selected-config ABI, exact testcase shapes, expected
fault sensitivity, and evidence output must all succeed. Set
``CUDA_VISIBLE_DEVICES`` to one numeric ordinal; the runtime must report one
visible device.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
PPU_BUNDLE_PATH = ROOT / "quactlize" / "ppu_bundle.py"
PPU_BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "quactlize_box_gate_ppu_bundle", PPU_BUNDLE_PATH)
if PPU_BUNDLE_SPEC is None or PPU_BUNDLE_SPEC.loader is None:
    raise RuntimeError(f"cannot load bundle verifier: {PPU_BUNDLE_PATH}")
ppu_bundle = importlib.util.module_from_spec(PPU_BUNDLE_SPEC)
sys.modules[PPU_BUNDLE_SPEC.name] = ppu_bundle
PPU_BUNDLE_SPEC.loader.exec_module(ppu_bundle)


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
VOIDP = ctypes.c_void_p

SCHEMA = "quactlize.prebuilt-six-library-box-gate"
SCHEMA_VERSION = 1
GGUF_ORACLE_VERSION = "0.19.0"
HGG_RUNTIME_SHA256 = "71c32cb41191458503234324360fcd3f1fa890dd5a082d465bb07328630c775e"
CORRECTNESS_BOUND = 5e-3
DENSE_SHAPE = (1, 1024, 5120)  # M, N, K: exact measured family for non-Q4.
GROUPED_ROWS = (2, 0, 3, 1)
GROUPED_SHAPE = (sum(GROUPED_ROWS), 256, 512, len(GROUPED_ROWS))
KPACK_MAPPING_ID = 0x514B504B54000001
Q4_KPACK4_MAPPING_ID = 0x51344B5034540001
GROUPED_DEFAULT_NAME = "16x128:16x16:s2"
SMALL_SQUARE_NAME = "32x32:16x16:s3"
DECODE_DEFAULT_NAME = "8x128:8x32:s3"
Q4_DECODE_NAME = "kpack4:8x32x256:8x16:s3:S4"
HOST_ABI_FIXTURE_SHAPE = (256, 512)  # N, K
# Independent reference/gguf_kpack.py output for raw[i]=(37*i+qtype)&255.
# These hashes lock the offline bytes; prepare+recover alone would only prove
# that one library's forward and inverse agree with each other.
HOST_ABI_FIXTURE_SHA256 = {
    10: (
        "2ae811801c20502591c9a68f411190223d2b0e8317b9febb1dc1f7d4083a3731",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "0a3acd38c51d950fcf89fdbc4845dddfbfd4efb6abe4f300ee3569343309126a",
    ),
    11: (
        "a2ffb980687863562385f0b1c7232bc8e41d98bd0637d96c247a2b3c161c1e05",
        "d6978526fc7704d38cc431f12de3d7ba966c25391f634639b26c16655ed13320",
        "d4064a45dfab853c7459c76318329d523ba746351b96c602f66eb457dd5bd262",
    ),
    12: (
        "af69675165134cef8bab24398601d08f6c77744795a2b10ad4aaa3e60ff7a2e4",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "1695701593a89b3558729f42cd70e045aa8910c56f2518e0d61c0841c7de96ef",
    ),
    13: (
        "40c5cc522bd0fdf58cf3fb36eee097e28a43de5270ff533b6896f119d6523873",
        "898c2770946a42c75aa197a0e31e05f3d1f0d7ce565aba3828cd247dc23693fc",
        "2e07adec9e1999507dddd7aa2d8f471dcf005db4371009c96eb0b9f263d1441a",
    ),
    14: (
        "1d3c7ca2c05727e71f72bde83bd84c01c2357ae9b1e37d482076aa8090bdbe39",
        "bae8ee4e9c64969efb1e7b3c247d25ad959cf125d6beac92717e76a18c2fdb55",
        "0ab490350c0166003ed526ca9d6ef940f00c6d75dd7473ea91bb2a381d6cdd3d",
    ),
}

SOURCE_INPUTS = (
    "quactlize",
    "third_party/actlize",
    "third_party/cutlass",
)
RUNNER_INPUT = "tools/run_prebuilt_ppu_box_gate.py"


class GateError(RuntimeError):
    """A required box-admission condition did not hold."""


@dataclass(frozen=True)
class FormatSpec:
    name: str
    qtype: int
    role: str
    packed_format: int
    bits: int
    high_bits: int
    group_size: int
    block_bytes: int
    header_ranges: tuple[tuple[int, int], ...]
    layout: int
    transport_tile_k: int
    mapping_id: int
    expected_dense_config: str

    @property
    def expected_arrangement(self) -> tuple[int, ...]:
        return (
            2, self.layout, self.bits, self.high_bits, 0,
            self.transport_tile_k, self.group_size, 0, self.mapping_id,
        )

    @property
    def tactic_tile_k(self) -> int:
        return 128 if self.qtype == 14 else 256

    @property
    def expected_dense_record(self) -> tuple[object, ...]:
        if self.qtype == 12:
            return (False, Q4_DECODE_NAME, 8, 32, 256, 0, 8, 16, 3, 4)
        return (
            False, SMALL_SQUARE_NAME, 32, 32, self.tactic_tile_k, 0,
            16, 16, 3, 1,
        )

    @property
    def expected_grouped_record(self) -> tuple[object, ...]:
        return (
            False, GROUPED_DEFAULT_NAME, 16, 128, self.tactic_tile_k, 0,
            16, 16, 2,
        )

    @property
    def expected_decode_default_record(self) -> tuple[object, ...]:
        return (
            False, DECODE_DEFAULT_NAME, 8, 128, self.tactic_tile_k, 0,
            8, 32, 3, 1,
        )


FORMATS = (
    FormatSpec("Q2_K", 10, "fmt2", 2, 2, 0, 16, 84,
               ((80, 82), (82, 84)), 2, 128, KPACK_MAPPING_ID,
               SMALL_SQUARE_NAME),
    FormatSpec("Q3_K", 11, "fmt3", 3, 2, 1, 16, 110,
               ((108, 110),), 2, 256, KPACK_MAPPING_ID,
               SMALL_SQUARE_NAME),
    FormatSpec("Q4_K", 12, "fmt0", 0, 4, 0, 32, 144,
               ((0, 2), (2, 4)), 1, 64, Q4_KPACK4_MAPPING_ID,
               Q4_DECODE_NAME),
    FormatSpec("Q5_K", 13, "fmt1", 1, 4, 1, 32, 176,
               ((0, 2), (2, 4)), 2, 256, KPACK_MAPPING_ID,
               SMALL_SQUARE_NAME),
    FormatSpec("Q6_K", 14, "fmt4", 4, 4, 2, 16, 210,
               ((208, 210),), 2, 128, KPACK_MAPPING_ID,
               SMALL_SQUARE_NAME),
)


@dataclass(frozen=True)
class BoundLibrary:
    path: pathlib.Path
    handle: ctypes.CDLL
    identity: object
    canonical: object
    dense_any_m: object
    grouped_any_m: object
    units_bytes: object
    prepare: object
    recover: object
    dense_selected: object
    grouped_selected: object
    dense_workspace: object
    grouped_workspace: object
    dense_device: object
    grouped_device: object


def _bind(library: ctypes.CDLL, name: str, argtypes: list[object], restype: object):
    try:
        function = getattr(library, name)
    except AttributeError as exc:
        raise GateError(
            f"{pathlib.Path(library._name).name} lacks required ABI {name}") from exc
    function.argtypes = argtypes
    function.restype = restype
    return function


def _bind_format_library(path: pathlib.Path) -> BoundLibrary:
    try:
        handle = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    except OSError as exc:
        raise GateError(f"cannot load format library {path}: {exc}") from exc
    i32 = ctypes.c_int
    i64 = ctypes.c_int64
    return BoundLibrary(
        path=path,
        handle=handle,
        identity=_bind(handle, "quactlize_ppu_build_packed_format_v1", [], i32),
        canonical=_bind(
            handle, "quactlize_ppu_canonical_arrangement_v2",
            [i32, ARRP], i32),
        dense_any_m=_bind(
            handle,
            "quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2",
            [i32, i32, i32, ARRP], i32),
        grouped_any_m=_bind(
            handle,
            "quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2",
            [i32, i32, i32, i32, ARRP], i32),
        units_bytes=_bind(
            handle, "quactlize_ppu_units_bytes", [i32, i32, i32], i64),
        prepare=_bind(
            handle, "quactlize_ppu_prepare_fully_quantized_for_arrangement_v2",
            [VOIDP, VOIDP, VOIDP, VOIDP, i32, i32, i32, i32, ARRP], i32),
        recover=_bind(
            handle, "quactlize_ppu_recover_fully_quantized_for_arrangement_v2",
            [VOIDP, VOIDP, VOIDP, VOIDP, i32, i32, i32, i32, ARRP], i32),
        dense_selected=_bind(
            handle,
            "quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2",
            [ctypes.POINTER(ConfigV4), i32, i32, i32, i32, i32, ARRP,
             ctypes.c_char_p], i32),
        grouped_selected=_bind(
            handle,
            "quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2",
            [ctypes.POINTER(ConfigV3), i32, i32, i32, i32, i32, i32, i32,
             ARRP, ctypes.c_char_p], i32),
        dense_workspace=_bind(
            handle,
            "quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2",
            [i32, i32, i32, i32, ARRP], i64),
        grouped_workspace=_bind(
            handle,
            "quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2",
            [i32, i32, i32, i32, i32, i32, ARRP], i64),
        dense_device=_bind(
            handle,
            "quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2",
            [VOIDP, VOIDP, VOIDP, VOIDP, VOIDP,
             i32, i32, i32, i32, VOIDP, i64, VOIDP, ctypes.c_char_p, ARRP],
            i32),
        grouped_device=_bind(
            handle,
            "quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2",
            [VOIDP, VOIDP, VOIDP, VOIDP, VOIDP, VOIDP,
             i32, i32, i32, i32, i32, i32,
             VOIDP, i64, VOIDP, ctypes.c_char_p, ARRP],
            i32),
    )


def _assert_default_library_identity(path: pathlib.Path) -> dict[str, object]:
    try:
        handle = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    except OSError as exc:
        raise GateError(f"cannot load default library {path}: {exc}") from exc
    identity = _bind(
        handle, "quactlize_ppu_build_packed_format_v1", [], ctypes.c_int)
    got = int(identity())
    if got != -1:
        raise GateError(f"default library reports packed format {got}, expected -1")
    return {"path": str(path), "packed_format": got}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(_contiguous(value).tobytes()).hexdigest()


def _write_json(path: pathlib.Path, value: object) -> None:
    try:
        payload = json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise GateError(f"gate evidence is not strict JSON: {exc}") from exc
    temporary: pathlib.Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent)
        temporary = pathlib.Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        # A hard link publishes the complete inode only if the final name is
        # absent. Unlike replace/rename, it cannot silently overwrite a prior
        # authority record if the output directory changes concurrently.
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise GateError(f"cannot write gate evidence {path}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _run_git(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"cannot inspect source authority with git: {exc}") from exc


def _assert_source_authority(manifest: Mapping[str, object]) -> dict[str, str]:
    head_process = _run_git(["rev-parse", "HEAD"])
    if head_process.returncode:
        raise GateError(f"cannot resolve source HEAD: {head_process.stdout.strip()}")
    head = head_process.stdout.strip()
    source = manifest.get("source")
    recorded = source.get("commit") if isinstance(source, dict) else None
    if not isinstance(recorded, str):
        raise GateError("bundle manifest has no source commit")
    exists = _run_git(["cat-file", "-e", f"{recorded}^{{commit}}"])
    if exists.returncode:
        raise GateError(
            f"bundle source commit is absent from this checkout: {recorded}")

    diff = _run_git([
        "diff", "--quiet", "--ignore-submodules=none", recorded, "HEAD", "--",
        *SOURCE_INPUTS,
    ])
    if diff.returncode == 1:
        raise GateError(
            "runtime implementation differs between bundle source and checkout HEAD")
    if diff.returncode:
        raise GateError(f"cannot inspect tracked gate inputs: {diff.stdout.strip()}")

    tracked_runner = _run_git(["ls-files", "--error-unmatch", RUNNER_INPUT])
    if tracked_runner.returncode or tracked_runner.stdout.strip() != RUNNER_INPUT:
        raise GateError("box gate runner must be a tracked file in checkout HEAD")

    worktree_inputs = (*SOURCE_INPUTS, RUNNER_INPUT)
    for arguments, description in (
            (["diff", "--quiet", "--ignore-submodules=none", "HEAD", "--",
              *worktree_inputs], "unstaged runtime/gate inputs"),
            (["diff", "--cached", "--quiet", "--ignore-submodules=none",
              "HEAD", "--", *worktree_inputs], "staged runtime/gate inputs")):
        dirty = _run_git(arguments)
        if dirty.returncode == 1:
            raise GateError(f"checkout has {description}")
        if dirty.returncode:
            raise GateError(
                f"cannot inspect {description}: {dirty.stdout.strip()}")
    untracked = _run_git([
        "ls-files", "--others", "--exclude-standard", "--", *worktree_inputs])
    if untracked.returncode:
        raise GateError(
            f"cannot inspect untracked runtime inputs: {untracked.stdout.strip()}")
    if untracked.stdout.strip():
        raise GateError(
            f"checkout has untracked runtime inputs: {untracked.stdout.strip()}")

    submodule_process = _run_git(["submodule", "status", "--recursive"])
    if submodule_process.returncode:
        raise GateError(
            f"cannot inspect recursive submodules: {submodule_process.stdout.strip()}")
    actual: dict[str, str] = {}
    for line in submodule_process.stdout.splitlines():
        if not line:
            continue
        state, fields = line[0], line[1:].split()
        if state != " " or len(fields) < 2:
            raise GateError(f"submodule is not at its recorded commit: {line!r}")
        actual[fields[1]] = fields[0]
    expected = {
        str(item["path"]): str(item["commit"])
        for item in source.get("submodules", [])
    }
    if actual != expected:
        raise GateError(
            f"submodules differ from bundle: actual={actual!r} expected={expected!r}")
    for path in sorted(expected):
        submodule_dirty = _run_git([
            "-C", path, "status", "--porcelain=v1", "--untracked-files=all"])
        if submodule_dirty.returncode:
            raise GateError(
                f"cannot inspect submodule worktree {path}: "
                f"{submodule_dirty.stdout.strip()}")
        if submodule_dirty.stdout.strip():
            raise GateError(
                f"submodule worktree is dirty: {path}: "
                f"{submodule_dirty.stdout.strip()}")
    return {
        "bundle_source_commit": recorded,
        "checkout_head": head,
        "runner_sha256": _sha256(pathlib.Path(__file__).resolve()),
    }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"numpy": np.__version__}
    try:
        versions["gguf"] = importlib.metadata.version("gguf")
    except importlib.metadata.PackageNotFoundError as exc:
        raise GateError("official GGUF Python package is required") from exc
    if versions["gguf"] != GGUF_ORACLE_VERSION:
        raise GateError(
            f"official GGUF oracle version differs: got={versions['gguf']} "
            f"expected={GGUF_ORACLE_VERSION}")
    try:
        import gguf
        from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
        if not callable(gguf.quants.dequantize):
            raise AttributeError("gguf.quants.dequantize")
        for spec in FORMATS:
            qtype = getattr(GGMLQuantizationType, spec.name)
            if GGML_QUANT_SIZES[qtype][1] != spec.block_bytes:
                raise GateError(f"official GGUF block size differs for {spec.name}")
    except (ImportError, AttributeError, KeyError) as exc:
        raise GateError(f"official GGUF oracle API is incomplete: {exc}") from exc
    return versions


def _visible_device_ordinal(environment: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environment is None else environment
    value = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+", value):
        raise GateError(
            "CUDA_VISIBLE_DEVICES must name exactly one numeric device ordinal")
    return value


def _require_runtime_digest(path: pathlib.Path) -> str:
    got = _sha256(path)
    if got != HGG_RUNTIME_SHA256:
        raise GateError(
            f"SDK libhggc_wrapper digest differs: got={got} "
            f"expected={HGG_RUNTIME_SHA256}")
    return got


def _runtime_device_identity(
        get_device_count, get_device, check,
        environment: Mapping[str, str] | None = None) -> dict[str, object]:
    visible = _visible_device_ordinal(environment)
    count = ctypes.c_int(-1)
    check("hggcGetDeviceCount", int(get_device_count(ctypes.byref(count))))
    if count.value != 1:
        raise GateError(
            f"hggcGetDeviceCount returned {count.value}; exactly one visible device is required")
    current = ctypes.c_int(-1)
    check("hggcGetDevice", int(get_device(ctypes.byref(current))))
    if current.value < 0 or current.value >= count.value:
        raise GateError(
            f"hggcGetDevice returned {current.value} outside visible count {count.value}")
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "hggc_device_count": count.value,
        "hggc_current_device": current.value,
    }


class HggcRuntime:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self, sdk_root: pathlib.Path):
        runtime_path = (sdk_root / "lib" / "libhggc_wrapper.so").resolve()
        try:
            runtime_path.relative_to(sdk_root.resolve())
        except ValueError as exc:
            raise GateError(f"SDK runtime escapes --ppu-sdk: {runtime_path}") from exc
        if not runtime_path.is_file():
            raise GateError(f"SDK runtime is missing: {runtime_path}")
        self.sha256 = _require_runtime_digest(runtime_path)
        try:
            handle = ctypes.CDLL(
                str(runtime_path), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        except OSError as exc:
            raise GateError(f"cannot load SDK runtime {runtime_path}: {exc}") from exc
        self.path = runtime_path
        self.handle = handle
        self.malloc = _bind(
            handle, "hggcMalloc",
            [ctypes.POINTER(VOIDP), ctypes.c_size_t], ctypes.c_int)
        self.free = _bind(handle, "hggcFree", [VOIDP], ctypes.c_int)
        self.memcpy = _bind(
            handle, "hggcMemcpy", [VOIDP, VOIDP, ctypes.c_size_t, ctypes.c_int],
            ctypes.c_int)
        self.synchronize = _bind(handle, "hggcDeviceSynchronize", [], ctypes.c_int)
        self.get_device_count = _bind(
            handle, "hggcGetDeviceCount", [ctypes.POINTER(ctypes.c_int)], ctypes.c_int)
        self.get_device = _bind(
            handle, "hggcGetDevice", [ctypes.POINTER(ctypes.c_int)], ctypes.c_int)
        self.get_last_error = _bind(handle, "hggcGetLastError", [], ctypes.c_int)
        self.error_string = _bind(
            handle, "hggcGetErrorString", [ctypes.c_int], ctypes.c_char_p)
        self.device_identity = _runtime_device_identity(
            self.get_device_count, self.get_device, self.check)

    def _error(self, code: int) -> str:
        try:
            value = self.error_string(code)
            return value.decode("utf-8", errors="replace") if value else "unknown"
        except Exception:
            return "unknown"

    def check(self, operation: str, code: int) -> None:
        if code != 0:
            raise GateError(f"{operation} failed rc={code}: {self._error(code)}")


def _checked_device_launch(runtime: HggcRuntime, label: str, enqueue) -> dict[str, int]:
    before = int(runtime.get_last_error())
    device_call = int(enqueue())
    immediate = int(runtime.get_last_error())
    synchronize = int(runtime.synchronize())
    deferred = int(runtime.get_last_error())
    result = {
        "before": before,
        "device_call": device_call,
        "immediate": immediate,
        "synchronize": synchronize,
        "deferred": deferred,
    }
    if any(result.values()):
        diagnostics = {
            key: (value, runtime._error(value))
            for key, value in result.items() if value
        }
        raise GateError(f"{label} launch/runtime status differs: {diagnostics!r}")
    return result


@dataclass(frozen=True)
class DeviceBuffer:
    pointer: ctypes.c_void_p
    size: int


class DeviceArena:
    def __init__(self, runtime: HggcRuntime):
        self.runtime = runtime
        self.allocations: list[DeviceBuffer] = []

    def __enter__(self) -> "DeviceArena":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        failures: list[str] = []
        for allocation in reversed(self.allocations):
            code = int(self.runtime.free(allocation.pointer))
            if code:
                failures.append(
                    f"hggcFree({allocation.size}) rc={code}: {self.runtime._error(code)}")
        self.allocations.clear()
        if failures and exc_type is None:
            raise GateError("; ".join(failures))
        return False

    def allocate(self, size: int) -> DeviceBuffer:
        if size <= 0:
            raise GateError(f"device allocation size must be positive, got {size}")
        pointer = VOIDP()
        self.runtime.check(
            f"hggcMalloc({size})",
            int(self.runtime.malloc(ctypes.byref(pointer), size)))
        if not pointer.value:
            raise GateError(f"hggcMalloc({size}) returned a null pointer")
        allocation = DeviceBuffer(pointer, size)
        self.allocations.append(allocation)
        return allocation

    def upload(self, host: np.ndarray) -> DeviceBuffer:
        host = _contiguous(host)
        allocation = self.allocate(host.nbytes)
        self.copy_to_device(allocation, host)
        return allocation

    def copy_to_device(self, allocation: DeviceBuffer, host: np.ndarray) -> None:
        host = _contiguous(host)
        if host.nbytes != allocation.size:
            raise GateError(
                f"H2D size differs: host={host.nbytes} device={allocation.size}")
        self.runtime.check(
            f"hggcMemcpy(H2D,{host.nbytes})",
            int(self.runtime.memcpy(
                allocation.pointer, VOIDP(host.ctypes.data), host.nbytes,
                self.runtime.HOST_TO_DEVICE)))

    def download(self, allocation: DeviceBuffer, host: np.ndarray) -> None:
        host = _contiguous(host)
        if host.nbytes != allocation.size:
            raise GateError(
                f"D2H size differs: host={host.nbytes} device={allocation.size}")
        self.runtime.check(
            f"hggcMemcpy(D2H,{host.nbytes})",
            int(self.runtime.memcpy(
                VOIDP(host.ctypes.data), allocation.pointer, host.nbytes,
                self.runtime.DEVICE_TO_HOST)))


def _contiguous(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise GateError("ctypes tensor inputs must be C-contiguous NumPy arrays")
    return value


def _host_pointer(value: np.ndarray | None) -> ctypes.c_void_p | None:
    if value is None or value.size == 0:
        return None
    return VOIDP(_contiguous(value).ctypes.data)


def _arrangement_tuple(value: ArrangementV2) -> tuple[int, ...]:
    return tuple(int(getattr(value, field)) for field, _ctype in value._fields_)


def _copy_arrangement(value: ArrangementV2) -> ArrangementV2:
    return ArrangementV2.from_buffer_copy(bytes(value))


def _config_tuple(value: ConfigV3 | ConfigV4) -> tuple[object, ...]:
    if value.name is None:
        raise GateError("successful selected-config query returned a null name")
    result: tuple[object, ...] = (
        bool(value.enable_cuda_kernel), value.name.decode("utf-8"),
        int(value.tile_m), int(value.tile_n), int(value.tactic_tile_k),
        int(value.artifact_tile_k), int(value.warp_m), int(value.warp_n),
        int(value.stages),
    )
    if isinstance(value, ConfigV4):
        result += (int(value.split_k_slices),)
    return result


def _require_ctypes_layouts() -> None:
    expected = (
        ("ArrangementV2", ArrangementV2, 40, {"mapping_id": 32}),
        ("ConfigV3", ConfigV3, 48, {"name": 8, "stages": 40}),
        ("ConfigV4", ConfigV4, 48, {"name": 8, "split_k_slices": 44}),
    )
    for name, record, size, offsets in expected:
        if ctypes.sizeof(record) != size:
            raise GateError(
                f"{name} ctypes size={ctypes.sizeof(record)}, expected {size}")
        for field, offset in offsets.items():
            if getattr(record, field).offset != offset:
                raise GateError(
                    f"{name}.{field} offset={getattr(record, field).offset}, "
                    f"expected {offset}")


def _query_dense_config(
        library: BoundLibrary, spec: FormatSpec, arrangement: ArrangementV2,
        requested: bytes | None,
        shape: tuple[int, int, int] = DENSE_SHAPE) -> ConfigV4:
    m, n, k = shape
    output = ConfigV4()
    rc = int(library.dense_selected(
        ctypes.byref(output), m, n, k, spec.group_size, spec.qtype,
        ctypes.byref(arrangement), requested))
    if rc != 1:
        raise GateError(
            f"{spec.name} dense selected-config rc={rc} requested={requested!r}")
    return output


def _query_grouped_config(
        library: BoundLibrary, spec: FormatSpec, arrangement: ArrangementV2,
        requested: bytes | None) -> ConfigV3:
    total, n, k, experts = GROUPED_SHAPE
    output = ConfigV3()
    rc = int(library.grouped_selected(
        ctypes.byref(output), total, n, k, spec.group_size, experts,
        max(GROUPED_ROWS), spec.qtype, ctypes.byref(arrangement), requested))
    if rc != 1:
        raise GateError(
            f"{spec.name} grouped selected-config rc={rc} requested={requested!r}")
    return output


def _assert_selected_config_contract(
        library: BoundLibrary, spec: FormatSpec,
        arrangement: ArrangementV2) -> tuple[bytes, bytes, dict[str, object]]:
    automatic = _query_dense_config(library, spec, arrangement, None)
    automatic_tuple = _config_tuple(automatic)
    automatic_name = str(automatic_tuple[1])
    if automatic_tuple != spec.expected_dense_record:
        raise GateError(
            f"{spec.name} null dense selector differs: got={automatic_tuple!r} "
            f"expected={spec.expected_dense_record!r}")
    explicit = _query_dense_config(
        library, spec, arrangement, automatic_name.encode("ascii"))
    if _config_tuple(explicit) != automatic_tuple:
        raise GateError(f"{spec.name} explicit dense selector differs from named row")

    unmeasured_tuple: tuple[object, ...] | None = None
    override_tuple: tuple[object, ...] | None = None
    if spec.layout == 2:
        unmeasured = _query_dense_config(
            library, spec, arrangement, None, (3, DENSE_SHAPE[1], DENSE_SHAPE[2]))
        unmeasured_tuple = _config_tuple(unmeasured)
        if unmeasured_tuple != spec.expected_decode_default_record:
            raise GateError(
                f"{spec.name} exact-key miss did not retain compiled default: "
                f"got={unmeasured_tuple!r} "
                f"expected={spec.expected_decode_default_record!r}")
        override = _query_dense_config(
            library, spec, arrangement, DECODE_DEFAULT_NAME.encode("ascii"))
        override_tuple = _config_tuple(override)
        if override_tuple != spec.expected_decode_default_record:
            raise GateError(
                f"{spec.name} explicit override did not outrank measured selection: "
                f"got={override_tuple!r}")

    stale = ConfigV4()
    ctypes.memset(ctypes.byref(stale), 0x5A, ctypes.sizeof(stale))
    m, n, k = DENSE_SHAPE
    stale_rc = int(library.dense_selected(
        ctypes.byref(stale), m, n, k, spec.group_size, spec.qtype,
        ctypes.byref(arrangement), b"stale-config"))
    if stale_rc != 0 or bytes(stale) != bytes(ctypes.sizeof(stale)):
        raise GateError(f"{spec.name} stale dense config did not fail and clear output")

    grouped = _query_grouped_config(library, spec, arrangement, None)
    grouped_tuple = _config_tuple(grouped)
    grouped_name = str(grouped_tuple[1])
    if grouped_tuple != spec.expected_grouped_record:
        raise GateError(
            f"{spec.name} grouped default differs: got={grouped_tuple!r} "
            f"expected={spec.expected_grouped_record!r}")
    grouped_explicit = _query_grouped_config(
        library, spec, arrangement, grouped_name.encode("ascii"))
    if _config_tuple(grouped_explicit) != grouped_tuple:
        raise GateError(f"{spec.name} explicit grouped selector differs from named row")

    return (
        automatic_name.encode("ascii"), grouped_name.encode("ascii"),
        {"dense": automatic_tuple, "grouped": grouped_tuple,
         "dense_unmeasured": unmeasured_tuple,
         "dense_explicit_override": override_tuple,
         "stale": "FAIL_CLOSED"},
    )


def _assert_any_m_contract(
        library: BoundLibrary, spec: FormatSpec,
        arrangement: ArrangementV2) -> dict[str, dict[str, int]]:
    _m, dense_n, dense_k = DENSE_SHAPE
    dense_admitted = int(library.dense_any_m(
        dense_n, dense_k, spec.qtype, ctypes.byref(arrangement)))
    if dense_admitted != 1:
        raise GateError(
            f"{spec.name} dense any-M admission returned {dense_admitted}, "
            "expected 1")

    _total, n, k, experts = GROUPED_SHAPE
    grouped_admitted = int(library.grouped_any_m(
        n, k, experts, spec.qtype, ctypes.byref(arrangement)))
    if grouped_admitted != 1:
        raise GateError(
            f"{spec.name} grouped any-M admission returned {grouped_admitted}, "
            "expected 1")

    bad = _copy_arrangement(arrangement)
    bad.mapping_id ^= 1
    dense_bad_mapping = int(library.dense_any_m(
        dense_n, dense_k, spec.qtype, ctypes.byref(bad)))
    dense_null_arrangement = int(library.dense_any_m(
        dense_n, dense_k, spec.qtype, None))
    grouped_bad_mapping = int(library.grouped_any_m(
        n, k, experts, spec.qtype, ctypes.byref(bad)))
    grouped_null_arrangement = int(library.grouped_any_m(
        n, k, experts, spec.qtype, None))
    foreign_qtype = 10 if spec.qtype != 10 else 11
    dense_foreign_format = int(library.dense_any_m(
        dense_n, dense_k, foreign_qtype, ctypes.byref(arrangement)))
    grouped_foreign_format = int(library.grouped_any_m(
        n, k, experts, foreign_qtype, ctypes.byref(arrangement)))
    failures = (
        dense_bad_mapping, dense_null_arrangement, dense_foreign_format,
        grouped_bad_mapping, grouped_null_arrangement, grouped_foreign_format,
    )
    if any(failures):
        raise GateError(
            f"{spec.name} any-M admission did not fail closed: "
            f"dense=[{dense_bad_mapping},{dense_null_arrangement},"
            f"{dense_foreign_format}] grouped=[{grouped_bad_mapping},"
            f"{grouped_null_arrangement},{grouped_foreign_format}]")
    return {
        "dense": {
            "valid": dense_admitted,
            "bad_mapping": dense_bad_mapping,
            "null_arrangement": dense_null_arrangement,
            "foreign_format": dense_foreign_format,
        },
        "grouped": {
            "valid": grouped_admitted,
            "bad_mapping": grouped_bad_mapping,
            "null_arrangement": grouped_null_arrangement,
            "foreign_format": grouped_foreign_format,
        },
    }


def _raw_blocks(
        spec: FormatSpec, n: int, k: int, experts: int, seed: int) -> np.ndarray:
    block_rows = experts * n * (k // 256)
    rng = np.random.default_rng(seed)
    raw = rng.integers(
        0, 256, size=(block_rows, spec.block_bytes), dtype=np.uint8)
    for low, high in spec.header_ranges:
        values = (rng.random(block_rows) * 0.1 + 0.001).astype(np.float16)
        raw[:, low:high] = values.view(np.uint8).reshape(block_rows, 2)
    return np.ascontiguousarray(raw)


def _artifact_sizes(
        library: BoundLibrary, spec: FormatSpec,
        n: int, k: int, experts: int) -> tuple[int, int, int]:
    low = experts * n * k * spec.bits // 8
    high = experts * n * k * spec.high_bits // 8
    units_one = int(library.units_bytes(n, k, spec.qtype))
    if units_one <= 0:
        raise GateError(
            f"{spec.name} units size query rejected n={n} k={k}: {units_one}")
    return low, high, experts * units_one


def _prepare_artifact(
        library: BoundLibrary, spec: FormatSpec, arrangement: ArrangementV2,
        raw: np.ndarray, n: int, k: int, experts: int,
        *, round_trip: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_size, high_size, units_size = _artifact_sizes(
        library, spec, n, k, experts)
    low = np.empty(low_size, dtype=np.uint8)
    high = np.empty(high_size, dtype=np.uint8)
    units = np.empty(units_size, dtype=np.uint8)
    rc = int(library.prepare(
        _host_pointer(raw), _host_pointer(low), _host_pointer(high),
        _host_pointer(units), n, k, experts, spec.qtype,
        ctypes.byref(arrangement)))
    if rc != 0:
        raise GateError(f"{spec.name} host artifact prepare failed rc={rc}")
    if round_trip:
        recovered = np.empty_like(raw)
        rc = int(library.recover(
            _host_pointer(low), _host_pointer(high), _host_pointer(units),
            _host_pointer(recovered), n, k, experts, spec.qtype,
            ctypes.byref(arrangement)))
        if rc != 0 or not np.array_equal(recovered, raw):
            raise GateError(
                f"{spec.name} host artifact inverse is not byte-exact rc={rc}")
    return low, high, units


def _assert_frozen_host_artifact(
        library: BoundLibrary, spec: FormatSpec,
        arrangement: ArrangementV2) -> dict[str, object]:
    n, k = HOST_ABI_FIXTURE_SHAPE
    rows = n * (k // 256)
    raw = ((np.arange(rows * spec.block_bytes, dtype=np.uint64) * 37 +
            spec.qtype) & 0xFF).astype(np.uint8).reshape(rows, spec.block_bytes)
    low, high, units = _prepare_artifact(
        library, spec, arrangement, np.ascontiguousarray(raw), n, k, 1)
    got = tuple(_array_sha256(value) for value in (low, high, units))
    expected = HOST_ABI_FIXTURE_SHA256[spec.qtype]
    if got != expected:
        raise GateError(
            f"{spec.name} offline artifact ABI drifted: got={got!r} "
            f"expected={expected!r}")
    return {
        "shape": [n, k],
        "raw_formula": "(37*i+qtype)&255",
        "low_sha256": got[0],
        "high_sha256": got[1],
        "units_sha256": got[2],
        "authority": "frozen independent reference/gguf_kpack.py",
    }


def _official_weights(
        spec: FormatSpec, raw: np.ndarray, n: int, k: int,
        experts: int) -> np.ndarray:
    import gguf
    from gguf.constants import GGMLQuantizationType
    qtype = getattr(GGMLQuantizationType, spec.name)
    weights = gguf.quants.dequantize(raw.reshape(-1), qtype)
    weights = np.asarray(weights, dtype=np.float64).reshape(experts, n, k)
    if not np.isfinite(weights).all():
        raise GateError(f"{spec.name} official GGUF oracle produced non-finite weights")
    return weights


def _conditioned_error(
        got: np.ndarray, reference: np.ndarray,
        denominator: np.ndarray) -> float:
    got64 = np.asarray(got, dtype=np.float64)
    if not np.isfinite(got64).all():
        return float("inf")
    floor = np.finfo(np.float64).tiny
    return float(np.max(np.abs(got64 - reference) / np.maximum(denominator, floor)))


def _error_evidence(value: float) -> float | str:
    return value if math.isfinite(value) else "NONFINITE"


def _dense_launches(
        runtime: HggcRuntime, library: BoundLibrary, spec: FormatSpec,
        arrangement: ArrangementV2, low: np.ndarray, high: np.ndarray,
        units: np.ndarray, activation: np.ndarray, explicit_name: bytes,
        workspace_bytes: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    m, n, k = DENSE_SHAPE
    poison = np.full((m, n), np.nan, dtype=np.float16)
    zero_workspace = np.zeros(workspace_bytes, dtype=np.uint8)
    launch_status: list[dict[str, object]] = []
    with DeviceArena(runtime) as arena:
        device_act = arena.upload(activation)
        device_low = arena.upload(low)
        device_high = arena.upload(high) if high.size else None
        device_units = arena.upload(units)
        device_out = arena.upload(poison)
        device_workspace = arena.upload(zero_workspace)

        def launch(config: bytes | None, phase: str) -> np.ndarray:
            arena.copy_to_device(device_out, poison)
            arena.copy_to_device(device_workspace, zero_workspace)
            status = _checked_device_launch(
                runtime, f"{spec.name} dense {phase}",
                lambda: library.dense_device(
                    device_act.pointer, device_low.pointer,
                    device_high.pointer if device_high else None,
                    device_units.pointer, device_out.pointer,
                    m, n, k, spec.qtype, device_workspace.pointer,
                    workspace_bytes, None, config, ctypes.byref(arrangement)))
            launch_status.append({"phase": phase, **status})
            result = np.empty_like(poison)
            arena.download(device_out, result)
            return result

        automatic = launch(None, "null-config")
        explicit = launch(explicit_name, "same-row-explicit-config")
        arena.copy_to_device(device_units, np.zeros_like(units))
        planted = launch(explicit_name, "zeroed-scale-unit-plant")
    return automatic, explicit, planted, launch_status


def _grouped_launches(
        runtime: HggcRuntime, library: BoundLibrary, spec: FormatSpec,
        arrangement: ArrangementV2, low: np.ndarray, high: np.ndarray,
        units: np.ndarray, activation: np.ndarray, explicit_name: bytes,
        workspace_bytes: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    total, n, k, experts = GROUPED_SHAPE
    offsets = np.concatenate((
        np.zeros(1, dtype=np.int32), np.cumsum(GROUPED_ROWS, dtype=np.int32)))
    poison = np.full((total, n), np.nan, dtype=np.float16)
    zero_workspace = np.zeros(workspace_bytes, dtype=np.uint8)
    planted_low = _plant_grouped_expert_zero(low, experts)
    planted_high = _plant_grouped_expert_zero(high, experts)
    planted_units = _plant_grouped_expert_zero(units, experts)
    launch_status: list[dict[str, object]] = []
    with DeviceArena(runtime) as arena:
        device_act = arena.upload(activation)
        device_low = arena.upload(low)
        device_high = arena.upload(high) if high.size else None
        device_units = arena.upload(units)
        device_offsets = arena.upload(offsets)
        device_out = arena.upload(poison)
        device_workspace = arena.upload(zero_workspace)

        def launch(config: bytes | None, phase: str) -> np.ndarray:
            arena.copy_to_device(device_out, poison)
            arena.copy_to_device(device_workspace, zero_workspace)
            status = _checked_device_launch(
                runtime, f"{spec.name} grouped {phase}",
                lambda: library.grouped_device(
                    device_act.pointer, device_low.pointer,
                    device_high.pointer if device_high else None,
                    device_units.pointer, device_offsets.pointer,
                    device_out.pointer, total, n, k, experts,
                    max(GROUPED_ROWS), spec.qtype, device_workspace.pointer,
                    workspace_bytes, None, config, ctypes.byref(arrangement)))
            launch_status.append({"phase": phase, **status})
            result = np.empty_like(poison)
            arena.download(device_out, result)
            return result

        automatic = launch(None, "null-config")
        explicit = launch(explicit_name, "same-row-explicit-config")
        arena.copy_to_device(device_low, planted_low)
        if device_high:
            arena.copy_to_device(device_high, planted_high)
        arena.copy_to_device(device_units, planted_units)
        planted = launch(explicit_name, "expert0-rebind-plant")
    return automatic, explicit, planted, launch_status


def _plant_grouped_expert_zero(value: np.ndarray, experts: int) -> np.ndarray:
    if value.size == 0:
        return value.copy()
    if value.size % experts:
        raise GateError(
            f"grouped artifact byte count {value.size} is not divisible by {experts}")
    planted = value.reshape(experts, -1).copy()
    for expert, count in enumerate(GROUPED_ROWS):
        if expert and count:
            planted[expert] = planted[0]
    return np.ascontiguousarray(planted.reshape(-1))


def _dense_reference(
        activation: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    activation64 = activation.astype(np.float64)
    weight = weights[0]
    return activation64 @ weight.T, np.abs(activation64) @ np.abs(weight).T


def _grouped_reference(
        activation: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.empty((activation.shape[0], weights.shape[1]), dtype=np.float64)
    denominator = np.empty_like(reference)
    start = 0
    for expert, count in enumerate(GROUPED_ROWS):
        if count:
            rows = activation[start:start + count].astype(np.float64)
            reference[start:start + count] = rows @ weights[expert].T
            denominator[start:start + count] = np.abs(rows) @ np.abs(weights[expert]).T
        start += count
    return reference, denominator


def _mapping_red(
        library: BoundLibrary, spec: FormatSpec,
        arrangement: ArrangementV2) -> dict[str, int]:
    bad = _copy_arrangement(arrangement)
    bad.mapping_id ^= 1
    m, dense_n, dense_k = DENSE_SHAPE
    dense = int(library.dense_workspace(
        m, dense_n, dense_k, spec.qtype, ctypes.byref(bad)))
    total, grouped_n, grouped_k, experts = GROUPED_SHAPE
    grouped = int(library.grouped_workspace(
        total, max(GROUPED_ROWS), grouped_n, grouped_k, experts, spec.qtype,
        ctypes.byref(bad)))
    if dense != -1 or grouped != -1:
        raise GateError(
            f"{spec.name} bad mapping was not rejected: dense={dense} grouped={grouped}")
    return {"dense_workspace": dense, "grouped_workspace": grouped}


def _run_format_gate(
        runtime: HggcRuntime, library: BoundLibrary,
        spec: FormatSpec) -> dict[str, object]:
    identity = int(library.identity())
    if identity != spec.packed_format:
        raise GateError(
            f"{library.path.name} identity={identity}, expected FMT{spec.packed_format}")
    arrangement = ArrangementV2()
    rc = int(library.canonical(spec.qtype, ctypes.byref(arrangement)))
    got_arrangement = _arrangement_tuple(arrangement)
    if rc != 0 or got_arrangement != spec.expected_arrangement:
        raise GateError(
            f"{spec.name} canonical arrangement differs rc={rc}: "
            f"got={got_arrangement!r} expected={spec.expected_arrangement!r}")

    any_m = _assert_any_m_contract(
        library, spec, arrangement)
    dense_config, grouped_config, selected = _assert_selected_config_contract(
        library, spec, arrangement)
    bad_mapping = _mapping_red(library, spec, arrangement)
    frozen_artifact = _assert_frozen_host_artifact(
        library, spec, arrangement)

    m, n, k = DENSE_SHAPE
    dense_raw = _raw_blocks(spec, n, k, 1, 51000 + spec.qtype)
    dense_low, dense_high, dense_units = _prepare_artifact(
        library, spec, arrangement, dense_raw, n, k, 1)
    dense_weights = _official_weights(spec, dense_raw, n, k, 1)
    dense_rng = np.random.default_rng(52000 + spec.qtype)
    dense_activation = np.ascontiguousarray(
        (dense_rng.standard_normal((m, k)) * 0.2).astype(np.float16))
    dense_reference, dense_denominator = _dense_reference(
        dense_activation, dense_weights)
    dense_workspace = int(library.dense_workspace(
        m, n, k, spec.qtype, ctypes.byref(arrangement)))
    if dense_workspace <= 0:
        raise GateError(f"{spec.name} dense workspace query returned {dense_workspace}")
    dense_auto, dense_explicit, dense_plant, dense_launch_status = _dense_launches(
        runtime, library, spec, arrangement, dense_low, dense_high,
        dense_units, dense_activation, dense_config, dense_workspace)
    dense_auto_error = _conditioned_error(
        dense_auto, dense_reference, dense_denominator)
    dense_explicit_error = _conditioned_error(
        dense_explicit, dense_reference, dense_denominator)
    dense_plant_error = _conditioned_error(
        dense_plant, dense_reference, dense_denominator)
    if not np.array_equal(
            dense_auto.view(np.uint16), dense_explicit.view(np.uint16)):
        raise GateError(
            f"{spec.name} null/explicit dense launches selected different raw output")
    if dense_auto_error >= CORRECTNESS_BOUND or dense_explicit_error >= CORRECTNESS_BOUND:
        raise GateError(
            f"{spec.name} dense numerical mismatch: null={dense_auto_error:.3e} "
            f"explicit={dense_explicit_error:.3e}")
    if dense_plant_error <= CORRECTNESS_BOUND:
        raise GateError(
            f"{spec.name} dense oracle missed zeroed packed scale units: "
            f"error={dense_plant_error:.3e}")

    total, grouped_n, grouped_k, experts = GROUPED_SHAPE
    grouped_raw = _raw_blocks(
        spec, grouped_n, grouped_k, experts, 53000 + spec.qtype)
    grouped_low, grouped_high, grouped_units = _prepare_artifact(
        library, spec, arrangement, grouped_raw, grouped_n, grouped_k,
        experts)
    grouped_weights = _official_weights(
        spec, grouped_raw, grouped_n, grouped_k, experts)
    grouped_rng = np.random.default_rng(54000 + spec.qtype)
    grouped_activation = np.ascontiguousarray(
        (grouped_rng.standard_normal((total, grouped_k)) * 0.2).astype(np.float16))
    grouped_reference, grouped_denominator = _grouped_reference(
        grouped_activation, grouped_weights)
    grouped_workspace = int(library.grouped_workspace(
        total, max(GROUPED_ROWS), grouped_n, grouped_k, experts, spec.qtype,
        ctypes.byref(arrangement)))
    if grouped_workspace <= 0:
        raise GateError(
            f"{spec.name} grouped workspace query returned {grouped_workspace}")
    grouped_auto, grouped_explicit, grouped_plant, grouped_launch_status = _grouped_launches(
        runtime, library, spec, arrangement, grouped_low, grouped_high,
        grouped_units, grouped_activation, grouped_config, grouped_workspace)
    grouped_auto_error = _conditioned_error(
        grouped_auto, grouped_reference, grouped_denominator)
    grouped_explicit_error = _conditioned_error(
        grouped_explicit, grouped_reference, grouped_denominator)
    grouped_plant_error = _conditioned_error(
        grouped_plant, grouped_reference, grouped_denominator)
    if not np.array_equal(
            grouped_auto.view(np.uint16), grouped_explicit.view(np.uint16)):
        raise GateError(
            f"{spec.name} null/explicit grouped launches selected different raw output")
    if (grouped_auto_error >= CORRECTNESS_BOUND or
            grouped_explicit_error >= CORRECTNESS_BOUND):
        raise GateError(
            f"{spec.name} grouped numerical mismatch: null={grouped_auto_error:.3e} "
            f"explicit={grouped_explicit_error:.3e}")
    if grouped_plant_error <= CORRECTNESS_BOUND:
        raise GateError(
            f"{spec.name} grouped oracle missed expert0 artifact rebinding: "
            f"error={grouped_plant_error:.3e}")

    print(
        f"[prebuilt-box-gate] format={spec.name} "
        f"dense_null={dense_auto_error:.3e} "
        f"dense_explicit={dense_explicit_error:.3e} "
        f"unit_plant={dense_plant_error:.3e} "
        f"grouped_null={grouped_auto_error:.3e} "
        f"grouped_explicit={grouped_explicit_error:.3e} "
        f"grouped_rebind={grouped_plant_error:.3e}",
        flush=True)
    return {
        "qtype": spec.qtype,
        "role": spec.role,
        "packed_format": spec.packed_format,
        "arrangement": list(got_arrangement),
        "any_m": any_m,
        "selected_config": selected,
        "host_prepare_recover": "BYTE_EXACT",
        "frozen_host_artifact": frozen_artifact,
        "bad_mapping": bad_mapping,
        "dense": {
            "shape": [m, n, k],
            "null_config_error": dense_auto_error,
            "explicit_config_error": dense_explicit_error,
            "zeroed_scale_unit_error": _error_evidence(dense_plant_error),
            "workspace_bytes": dense_workspace,
            "launch_status": dense_launch_status,
        },
        "grouped": {
            "shape": [total, grouped_n, grouped_k, experts],
            "rows_per_expert": list(GROUPED_ROWS),
            "null_config_error": grouped_auto_error,
            "explicit_config_error": grouped_explicit_error,
            "expert0_rebind_error": _error_evidence(grouped_plant_error),
            "workspace_bytes": grouped_workspace,
            "launch_status": grouped_launch_status,
        },
    }


def _create_output(path: pathlib.Path) -> pathlib.Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise GateError(f"refusing to overwrite gate output: {path}")
    if not path.parent.is_dir():
        raise GateError(f"gate output parent does not exist: {path.parent}")
    try:
        path.mkdir()
    except OSError as exc:
        raise GateError(f"cannot create gate output {path}: {exc}") from exc
    return path


def _format_library_paths(
        bundle: pathlib.Path, manifest: Mapping[str, object]) -> dict[str, pathlib.Path]:
    paths = {
        str(entry["role"]): bundle / str(entry["filename"])
        for entry in manifest["libraries"]
    }
    expected = {role.role for role in ppu_bundle.LIBRARY_ROLES}
    if set(paths) != expected:
        raise GateError(f"verified manifest roles differ: {sorted(paths)}")
    return paths


def _sdk_evidence(
        sdk_root: pathlib.Path, runtime: HggcRuntime,
        manifest: Mapping[str, object]) -> dict[str, object]:
    release = sdk_root / "release.yaml"
    hgobjdump = sdk_root / "bin" / "hgobjdump"
    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict):
        raise GateError("verified bundle lost its toolchain record")
    return {
        "root": str(sdk_root),
        "release": toolchain.get("sdk_release"),
        "archive_sha256": toolchain.get("sdk_archive_sha256"),
        "release_receipt_sha256": _sha256(release),
        "hgobjdump_sha256": _sha256(hgobjdump),
        "runtime_path": str(runtime.path),
        "runtime_sha256": _sha256(runtime.path),
        "device": runtime.device_identity,
    }


def run_gate(
        bundle: pathlib.Path, sdk_root: pathlib.Path,
        output: pathlib.Path) -> dict[str, object]:
    bundle = bundle.resolve()
    sdk_root = sdk_root.resolve()
    output = _create_output(output)

    print("[prebuilt-box-gate] phase=bundle-manifest-elf", flush=True)
    _require_ctypes_layouts()
    manifest = ppu_bundle.verify_bundle(
        bundle, sdk_root=sdk_root, inspect_binaries=True)
    source_authority = _assert_source_authority(manifest)
    versions = _dependency_versions()
    paths = _format_library_paths(bundle, manifest)
    runtime = HggcRuntime(sdk_root)
    sdk_evidence = _sdk_evidence(sdk_root, runtime, manifest)
    default_identity = _assert_default_library_identity(paths["default"])

    format_results: dict[str, object] = {}
    for spec in FORMATS:
        print(f"[prebuilt-box-gate] phase=numeric format={spec.name}", flush=True)
        library = _bind_format_library(paths[spec.role])
        format_results[spec.name] = _run_format_gate(runtime, library, spec)

    bundle_record = {
        "manifest_sha256": _sha256(bundle / "manifest.json"),
        "source": source_authority,
        "sdk": sdk_evidence,
        "default_library_identity": default_identity,
        "libraries": [
            {
                "role": entry["role"],
                "filename": entry["filename"],
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
            for entry in manifest["libraries"]
        ],
    }
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "execution": {
            "device_library_builds": 0,
            "host_compilations": 0,
            "runner": "python-ctypes",
            "library_load_mode": "six DSOs, RTLD_LOCAL, one process",
        },
        "bundle": bundle_record,
        "python": versions,
        "formats": format_results,
        "coverage": {
            "dense_exact_shape": list(DENSE_SHAPE),
            "grouped_shape": list(GROUPED_SHAPE),
            "empty_expert_rows": list(GROUPED_ROWS),
            "null_and_explicit_launches": list(format_results),
            "bad_mapping_workspace_queries": "EXPECTED_MINUS_ONE",
            "zeroed_scale_unit_fault": "EXPECTED_NUMERIC_RED",
            "grouped_expert0_rebind_fault": "EXPECTED_NUMERIC_RED",
            "numeric_reference": "official gguf 0.19.0 dequantize plus NumPy float64",
            "host_prepare_recover_scope": "self-inverse plus independent frozen artifact hashes",
        },
    }
    _write_json(output / "bundle.json", bundle_record)
    _write_json(output / "result.json", result)
    print(
        "[prebuilt-box-gate] PASS "
        f"source={source_authority['bundle_source_commit']} "
        f"checkout={source_authority['checkout_head']} "
        f"libraries=6 formats=5 output={output}",
        flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: CUDA_VISIBLE_DEVICES=0 python3 "
            "tools/run_prebuilt_ppu_box_gate.py BUNDLE --ppu-sdk SDK "
            "--output /workspace/ppu-box-gate"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=pathlib.Path,
                        help="prebuilt six-library runtime bundle root")
    parser.add_argument("--ppu-sdk", required=True, type=pathlib.Path,
                        help="admitted SDK root with hgobjdump and libhggc_wrapper")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="new directory for bundle.json and result.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    try:
        run_gate(args.bundle, args.ppu_sdk, output)
    except (GateError, ppu_bundle.BundleError) as exc:
        if output.is_dir():
            try:
                with (output / "FAIL.txt").open("x", encoding="utf-8") as destination:
                    destination.write(str(exc) + "\n")
                    destination.flush()
                    os.fsync(destination.fileno())
            except OSError:
                pass
        print(f"[prebuilt-box-gate] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

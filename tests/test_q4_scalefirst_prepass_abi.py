"""Public ABI checks for the Q4 K-pack4 ScaleFirst metadata prepass."""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from quactlize import ppu_bundle


ROOT = Path(__file__).parents[1]
HEADER = ROOT / "quactlize/include/quactlize_ppu_device.h"
SOURCE = ROOT / "quactlize/csrc/device/ppu_backend.cu"


class ArrangementV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int32), ("layout", ctypes.c_int32),
        ("bits", ctypes.c_int32), ("high_bits", ctypes.c_int32),
        ("artifact_tile_k", ctypes.c_int32),
        ("transport_tile_k", ctypes.c_int32),
        ("group_size", ctypes.c_int32), ("reserved", ctypes.c_int32),
        ("mapping_id", ctypes.c_uint64),
    ]


def canonical_q4_kpack4():
    return ArrangementV2(2, 1, 4, 0, 0, 64, 32, 0, 0x51344B5034540001)


def test_public_header_exposes_only_the_fixed_versioned_prepass_contract():
    header = HEADER.read_text(encoding="utf-8")
    assert "QUACTLIZE_PPU_Q4_KPACK4_SCALEFIRST_PLANE_ALIGNMENT_V1 16" in header
    assert "quactlize_ppu_q4_kpack4_scalefirst_metadata_plane_bytes_for_arrangement_v2" in header
    assert "quactlize_ppu_q4_kpack4_scalefirst_prepass_dev_for_arrangement_v2" in header
    assert "quactlize_ppu_prepass_unit" not in header
    for contract in ("pairwise", "Extra bytes", "default stream", "event", "enqueued"):
        assert contract in header


@pytest.mark.parametrize("language,standard,compiler_names", [
    ("c", "c11", ("cc", "gcc", "clang")),
    ("c++", "c++17", ("c++", "g++", "clang++")),
])
def test_public_prepass_declarations_are_valid_c_abi(
        language, standard, compiler_names):
    compiler = next((shutil.which(name) for name in compiler_names
                     if shutil.which(name)), None)
    if compiler is None:
        pytest.skip(f"a host {language} compiler is required")
    source = r'''
#include "quactlize_ppu_device.h"
static int64_t (*query_fn)(
    int, int, quactlize_ppu_placed_arrangement_v2 const*) =
    &quactlize_ppu_q4_kpack4_scalefirst_metadata_plane_bytes_for_arrangement_v2;
static int (*launch_fn)(
    uint8_t const*, int64_t, uint16_t*, int64_t, uint16_t*, int64_t,
    int, int, void*, quactlize_ppu_placed_arrangement_v2 const*) =
    &quactlize_ppu_q4_kpack4_scalefirst_prepass_dev_for_arrangement_v2;
int main(void) { return query_fn == 0 || launch_fn == 0; }
'''
    result = subprocess.run(
        [compiler, f"-std={standard}", "-Wall", "-Wextra", "-Werror",
         "-fsyntax-only", "-I", str(ROOT / "quactlize/include"),
         "-x", language, "-"], input=source, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout


def _assert_device_prepass_source_contract(source):
    start = source.index(
        'extern "C" int\nquactlize_ppu_q4_kpack4_scalefirst_prepass_dev_for_arrangement_v2')
    end = source.index('\nextern "C" int quactlize_ppu_gemv_lowbit', start)
    body = source[start:end]
    assert "prepass_unit_kernel<KType::Q4_K, 8>" in body
    assert "sizes.grid, 256, 0, s" in body
    assert "rt_check_launch" in body
    assert "device_span" in body
    for pair in ("spans_overlap(u0, u1, s0, s1)",
                 "spans_overlap(u0, u1, z0, z1)",
                 "spans_overlap(s0, s1, z0, z1)"):
        assert pair in body
    assert "#if !QUACTLIZE_Q4_SCALEFIRST_PREPASS_AVAILABLE" in body
    assert "static_assert(" in source and "power of two" in source
    assert "DevBuf" not in body
    assert "rt_sync" not in body
    assert "from_host" not in body


def test_device_prepass_is_async_fixed_q4_zmul8_and_checks_spans():
    _assert_device_prepass_source_contract(SOURCE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("old,new", [
    ("prepass_unit_kernel<KType::Q4_K, 8>",
     "prepass_unit_kernel<KType::Q4_K, 0>"),
    ("#if !QUACTLIZE_Q4_SCALEFIRST_PREPASS_AVAILABLE", "#if 0"),
    ("spans_overlap(s0, s1, z0, z1)", "false"),
    ("rt_check_launch", "rt_sync"),
])
def test_source_contract_rejects_planted_regressions(old, new):
    planted = SOURCE.read_text(encoding="utf-8").replace(old, new)
    with pytest.raises(AssertionError):
        _assert_device_prepass_source_contract(planted)


@pytest.mark.parametrize("role", ppu_bundle.LIBRARY_ROLES, ids=lambda role: role.role)
def test_compiled_bundle_query_and_ctypes_signature_fail_closed_by_role(role):
    root = os.environ.get("QUACTLIZE_PPU_BUNDLE")
    if not root:
        pytest.skip("QUACTLIZE_PPU_BUNDLE is required for the compiled ABI test")
    library = ctypes.CDLL(str(Path(root) / role.filename))
    query = library.quactlize_ppu_q4_kpack4_scalefirst_metadata_plane_bytes_for_arrangement_v2
    query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ArrangementV2)]
    query.restype = ctypes.c_int64
    launch = library.quactlize_ppu_q4_kpack4_scalefirst_prepass_dev_for_arrangement_v2
    launch.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                       ctypes.POINTER(ArrangementV2)]
    launch.restype = ctypes.c_int
    arrangement = canonical_q4_kpack4()
    expected = 256 * 256 // 16 if role.role == "default" else -1
    assert query(256, 256, ctypes.byref(arrangement)) == expected
    assert query(256, 256, None) == -1
    for field, _ in ArrangementV2._fields_:
        mutated = canonical_q4_kpack4()
        setattr(mutated, field, getattr(mutated, field) ^ 1)
        assert query(256, 256, ctypes.byref(mutated)) == -1, field
    for n, k in ((0, 256), (256, 0), (-256, 256), (256, -256),
                 (255, 256), (256, 255), (512, 256),
                 (2147483392, 2147483392)):
        result = query(n, k, ctypes.byref(canonical_q4_kpack4()))
        if role.role == "default" and (n, k) == (512, 256):
            assert result == n * k // 16
        else:
            assert result == -1

    cap = 256 * 256 // 16
    args = (ctypes.c_void_p(0x1000), cap, ctypes.c_void_p(0x3000), cap,
            ctypes.c_void_p(0x5000), cap, 256, 256, None,
            ctypes.byref(canonical_q4_kpack4()))
    if role.role != "default":
        assert launch(*args) == 34
        return

    assert launch(None, cap, ctypes.c_void_p(0x3000), cap,
                  ctypes.c_void_p(0x5000), cap, 256, 256, None,
                  ctypes.byref(canonical_q4_kpack4())) == 30
    assert launch(ctypes.c_void_p(0x1000), cap - 1, ctypes.c_void_p(0x3000), cap,
                  ctypes.c_void_p(0x5000), cap, 256, 256, None,
                  ctypes.byref(canonical_q4_kpack4())) == 37
    assert launch(ctypes.c_void_p(0x1001), cap, ctypes.c_void_p(0x3000), cap,
                  ctypes.c_void_p(0x5000), cap, 256, 256, None,
                  ctypes.byref(canonical_q4_kpack4())) == 30
    assert launch(ctypes.c_void_p(0x1000), cap, ctypes.c_void_p(0x1800), cap,
                  ctypes.c_void_p(0x5000), cap, 256, 256, None,
                  ctypes.byref(canonical_q4_kpack4())) == 30
    assert launch(ctypes.c_void_p(0xFFFFFFFFFFFFFFF0), cap,
                  ctypes.c_void_p(0x3000), cap, ctypes.c_void_p(0x5000), cap,
                  256, 256, None, ctypes.byref(canonical_q4_kpack4())) == 30
    assert launch(ctypes.c_void_p(0x1000), -1, ctypes.c_void_p(0x3000), cap,
                  ctypes.c_void_p(0x5000), cap, 256, 256, None,
                  ctypes.byref(canonical_q4_kpack4())) == 30
    assert launch(ctypes.c_void_p(0x1000), cap, ctypes.c_void_p(0x3000), cap,
                  ctypes.c_void_p(0x5000), cap, 256, 256, None, None) == 30

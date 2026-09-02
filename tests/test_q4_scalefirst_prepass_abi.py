"""Public ABI checks for the five-format K-pack ScaleFirst metadata prepass."""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from quactlize import formats
from quactlize import ppu_bundle


ROOT = Path(__file__).parents[1]
HEADER = ROOT / "quactlize/include/quactlize_ppu_device.h"
SOURCE = ROOT / "quactlize/csrc/device/ppu_backend.cu"
PACKED_UNIT = ROOT / "quactlize/include/gguf_packed_unit.hpp"


class ArrangementV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int32), ("layout", ctypes.c_int32),
        ("bits", ctypes.c_int32), ("high_bits", ctypes.c_int32),
        ("artifact_tile_k", ctypes.c_int32),
        ("transport_tile_k", ctypes.c_int32),
        ("group_size", ctypes.c_int32), ("reserved", ctypes.c_int32),
        ("mapping_id", ctypes.c_uint64),
    ]


FORMAT_SPECS = {
    10: {"bits": 2, "high": 0, "transport": 128, "group": 16,
         "sb_per_unit": 1, "unit_bytes": 20, "zmul": 0},
    11: {"bits": 2, "high": 1, "transport": 256, "group": 16,
         "sb_per_unit": 2, "unit_bytes": 28, "zmul": -4},
    12: {"bits": 4, "high": 0, "transport": 64, "group": 32,
         "sb_per_unit": 1, "unit_bytes": 16, "zmul": 8},
    13: {"bits": 4, "high": 1, "transport": 256, "group": 32,
         "sb_per_unit": 1, "unit_bytes": 16, "zmul": 8},
    14: {"bits": 4, "high": 2, "transport": 128, "group": 16,
         "sb_per_unit": 2, "unit_bytes": 36, "zmul": -24},
}


def canonical_arrangement(qtype):
    spec = FORMAT_SPECS[qtype]
    q4 = qtype == 12
    return ArrangementV2(
        2, 1 if q4 else 2, spec["bits"], spec["high"], 0,
        spec["transport"], spec["group"], 0,
        0x51344B5034540001 if q4 else 0x514B504B54000001)


def canonical_q4_kpack4():
    return canonical_arrangement(12)


def test_all_format_contract_matches_the_independent_python_registry():
    for qtype, spec in FORMAT_SPECS.items():
        q = formats.QuantType(qtype)
        arrangement = (formats.q4_kpack4_arrangement() if qtype == 12
                       else formats.kquant_kpack_arrangement(q))
        unit = formats.packed_unit_layout(q)
        assert (arrangement.bits, arrangement.high_bits,
                arrangement.transport_tile_k, arrangement.group_size) == (
                    spec["bits"], spec["high"], spec["transport"],
                    spec["group"])
        assert (unit.superblocks_per_unit, unit.unit_bytes) == (
            spec["sb_per_unit"], spec["unit_bytes"])
        assert formats.placed_code_zmul(q) == spec["zmul"]


def test_public_header_exposes_generic_and_compatible_q4_contracts():
    header = HEADER.read_text(encoding="utf-8")
    assert "QUACTLIZE_PPU_KQUANT_SCALEFIRST_PLANE_ALIGNMENT_V1 16" in header
    assert "quactlize_ppu_kquant_scalefirst_metadata_plane_bytes_for_arrangement_v2" in header
    assert "quactlize_ppu_kquant_scalefirst_prepass_dev_for_arrangement_v2" in header
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
static int64_t (*generic_query_fn)(
    int, int, int, int, quactlize_ppu_placed_arrangement_v2 const*) =
    &quactlize_ppu_kquant_scalefirst_metadata_plane_bytes_for_arrangement_v2;
static int (*launch_fn)(
    uint8_t const*, int64_t, uint16_t*, int64_t, uint16_t*, int64_t,
    int, int, void*, quactlize_ppu_placed_arrangement_v2 const*) =
    &quactlize_ppu_q4_kpack4_scalefirst_prepass_dev_for_arrangement_v2;
static int (*generic_launch_fn)(
    uint8_t const*, int64_t, uint16_t*, int64_t, uint16_t*, int64_t,
    int, int, int, int, void*, quactlize_ppu_placed_arrangement_v2 const*) =
    &quactlize_ppu_kquant_scalefirst_prepass_dev_for_arrangement_v2;
int main(void) {
  return query_fn == 0 || launch_fn == 0 ||
         generic_query_fn == 0 || generic_launch_fn == 0;
}
'''
    result = subprocess.run(
        [compiler, f"-std={standard}", "-Wall", "-Wextra", "-Werror",
         "-fsyntax-only", "-I", str(ROOT / "quactlize/include"),
         "-x", language, "-"], input=source, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout


def _assert_device_prepass_source_contract(source, packed_unit):
    helper_start = source.index(
        "template <KType T>\nint launch_kquant_scalefirst_prepass")
    helper_end = source.index(
        "\n\ntemplate <ppu_gemv::WFormat", helper_start)
    body = source[helper_start:helper_end]
    for qtype in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"):
        assert f"launch_kquant_scalefirst_prepass<KType::{qtype}>" in body
    assert "kCanonicalPlacedZMul<T>" in body
    assert "prepass_unit_kernel<T, kZMul>" in body
    assert "sizes.grid, 256, 0, stream" in body
    assert "prepass_unit_grid_size<T>" in source
    assert "rt_check_launch" in body
    assert "device_span" in body
    for pair in ("spans_overlap(u0, u1, s0, s1)",
                 "spans_overlap(u0, u1, z0, z1)",
                 "spans_overlap(s0, s1, z0, z1)"):
        assert pair in body
    assert "if constexpr (!Available)" in body
    assert "kKQuantScaleFirstPrepassAvailable" in source
    assert "kQ4ScaleFirstPrepassAvailable" in source
    assert "static_assert(" in source and "power of two" in source
    assert "DevBuf" not in body
    assert "rt_sync" not in body
    assert "from_host" not in body
    for name, value in (("Q2_K", 0), ("Q3_K", -4), ("Q4_K", 8),
                        ("Q5_K", 8), ("Q6_K", -24)):
        assert (f"CanonicalPlacedZMul<KType::{name}> "
                f"{{ static constexpr int value = {value}; }}") in packed_unit

    wrapper_start = source.index(
        'extern "C" int\nquactlize_ppu_q4_kpack4_scalefirst_prepass_dev_for_arrangement_v2')
    wrapper_end = source.index(
        "\nextern \"C\" int quactlize_ppu_gemv_lowbit", wrapper_start)
    wrapper = source[wrapper_start:wrapper_end]
    assert "n, k, 1, 12, stream, arrangement" in wrapper


def test_device_prepass_is_async_all_format_fixed_zmul_and_checks_spans():
    _assert_device_prepass_source_contract(
        SOURCE.read_text(encoding="utf-8"),
        PACKED_UNIT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("target,old,new", [
    ("packed", "CanonicalPlacedZMul<KType::Q6_K> { static constexpr int value = -24; }",
     "CanonicalPlacedZMul<KType::Q6_K> { static constexpr int value = 0; }"),
    ("source", "prepass_unit_grid_size<T>", "prepass_grid_size"),
    ("source", "spans_overlap(s0, s1, z0, z1)", "false"),
    ("source", "rt_check_launch", "rt_sync"),
    ("source", "n, k, 1, 12, stream, arrangement",
     "n, k, 1, 10, stream, arrangement"),
])
def test_source_contract_rejects_planted_regressions(target, old, new):
    source = SOURCE.read_text(encoding="utf-8")
    packed = PACKED_UNIT.read_text(encoding="utf-8")
    assert old in (source if target == "source" else packed)
    if target == "source":
        source = source.replace(old, new)
    else:
        packed = packed.replace(old, new)
    with pytest.raises(AssertionError):
        _assert_device_prepass_source_contract(source, packed)


def _plane_bytes(qtype, n, k, experts):
    return experts * n * (k // FORMAT_SPECS[qtype]["group"]) * 2


def _units_bytes(qtype, n, k, experts):
    spec = FORMAT_SPECS[qtype]
    return (experts * n * (k // (256 * spec["sb_per_unit"])) *
            spec["unit_bytes"])


@pytest.mark.parametrize("role", ppu_bundle.LIBRARY_ROLES,
                         ids=lambda role: f"generic-{role.role}")
def test_compiled_bundle_generic_query_is_all_format_default_only(role):
    root = os.environ.get("QUACTLIZE_PPU_BUNDLE")
    if not root:
        pytest.skip("QUACTLIZE_PPU_BUNDLE is required for the compiled ABI test")
    library = ctypes.CDLL(str(Path(root) / role.filename))
    query = library.quactlize_ppu_kquant_scalefirst_metadata_plane_bytes_for_arrangement_v2
    query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.POINTER(ArrangementV2)]
    query.restype = ctypes.c_int64
    launch = library.quactlize_ppu_kquant_scalefirst_prepass_dev_for_arrangement_v2
    launch.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ctypes.POINTER(ArrangementV2)]
    launch.restype = ctypes.c_int

    n, k, experts = 256, 512, 3
    for qtype in FORMAT_SPECS:
        arrangement = canonical_arrangement(qtype)
        plane = _plane_bytes(qtype, n, k, experts)
        units = _units_bytes(qtype, n, k, experts)
        expected = plane if role.role == "default" else -1
        assert query(n, k, experts, qtype,
                     ctypes.byref(arrangement)) == expected
        assert query(n, k, experts, qtype, None) == -1

        other = 10 if qtype != 10 else 11
        assert query(n, k, experts, other,
                     ctypes.byref(arrangement)) == -1
        for field, _ in ArrangementV2._fields_:
            mutated = canonical_arrangement(qtype)
            setattr(mutated, field, getattr(mutated, field) ^ 1)
            assert query(n, k, experts, qtype,
                         ctypes.byref(mutated)) == -1, (qtype, field)

        valid_args = (
            ctypes.c_void_p(0x100000), units,
            ctypes.c_void_p(0x400000), plane,
            ctypes.c_void_p(0x800000), plane,
            n, k, experts, qtype, None, ctypes.byref(arrangement))
        if role.role != "default":
            assert launch(*valid_args) == 34
            wrong = canonical_arrangement(qtype)
            wrong.mapping_id ^= 1
            wrong_args = valid_args[:-1] + (ctypes.byref(wrong),)
            assert launch(*wrong_args) == 38
            continue

        assert launch(None, units, ctypes.c_void_p(0x400000), plane,
                      ctypes.c_void_p(0x800000), plane,
                      n, k, experts, qtype, None,
                      ctypes.byref(arrangement)) == 30
        assert launch(ctypes.c_void_p(0x100000), units - 1,
                      ctypes.c_void_p(0x400000), plane,
                      ctypes.c_void_p(0x800000), plane,
                      n, k, experts, qtype, None,
                      ctypes.byref(arrangement)) == 37
        assert launch(ctypes.c_void_p(0x100001), units,
                      ctypes.c_void_p(0x400000), plane,
                      ctypes.c_void_p(0x800000), plane,
                      n, k, experts, qtype, None,
                      ctypes.byref(arrangement)) == 30
        assert launch(ctypes.c_void_p(0x100000), units,
                      ctypes.c_void_p(0x100010), plane,
                      ctypes.c_void_p(0x800000), plane,
                      n, k, experts, qtype, None,
                      ctypes.byref(arrangement)) == 30
        assert launch(ctypes.c_void_p(0x100000), -1,
                      ctypes.c_void_p(0x400000), plane,
                      ctypes.c_void_p(0x800000), plane,
                      n, k, experts, qtype, None,
                      ctypes.byref(arrangement)) == 30
        assert launch(ctypes.c_void_p(0x100000), units,
                      ctypes.c_void_p(0x400000), plane,
                      ctypes.c_void_p(0x800000), plane,
                      n, k, experts, qtype, None, None) == 30

    for qtype in (11, 14):
        arrangement = canonical_arrangement(qtype)
        assert query(256, 256, 1, qtype,
                     ctypes.byref(arrangement)) == -1
    assert query(2147483392, 2147483392, 2147483391, 10,
                 ctypes.byref(canonical_arrangement(10))) == -1


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

"""Host and source checks, not PPU numerical/performance admission."""
import ctypes as C
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dev.gemv_cuda.build import q4_large_static_source
from dev.gemv_ppu.astage_source import CASES, ROOT, STAGE, dispatch, restore_body, source, stage_body, static_body
from dev.gemv_ppu.run_astage import check_exact, parse_result
from dev.gemv_ppu.astage_source import validation
from quactlize.execution.native import Call, Config, Sizes, Arrangement, arrangement


@pytest.fixture(scope="module")
def bodies():
    original = (ROOT / "quactlize/execution/gemv.cu").read_text()
    return q4_large_static_source(original), source(original)


@pytest.mark.parametrize("name", ["kpack_q4_small_static", "kpack_q4_large_static"])
def test_only_a_staging_and_address_change_in_kernel_body(bodies, name):
    before, after = bodies
    old = static_body(before, name)[2]
    new = static_body(after, name + "_astage")[2].replace(name + "_astage", name)
    assert restore_body(new) == old
    for token in ("aligned_unit(", "uint64_t word=", "fmaf(", "__hfma2(", "__shfl_xor_sync("):
        assert new.count(token) == old.count(token)
    assert new.count("__syncthreads();") == old.count("__syncthreads();") + 1
    assert "aligned_activation<Type>(c.a," not in new
    assert new.count("aligned_activation<Type>(staged_a,") == 1


@pytest.mark.parametrize("k,warps", sorted({(k, r[1]) for _, k, _, r, _ in CASES}))
def test_actual_stage_formula_has_one_writer_and_exact_fp16_bits(k, warps):
    # Each source vector is opaque 128-bit data: signed zero, tiny and NaN
    # encodings are transported, never converted by the staging operation.
    assert "pass * Warps * 32 + tid" in STAGE and "if (i < K / 8)" in STAGE
    assert "i < K / 8" in STAGE and "float4 const*>(c.a)[i]" in STAGE
    tags = np.arange(k, dtype=np.uint16) ^ np.uint16(0x9e37)
    staged = np.empty_like(tags)
    owners = np.zeros(k // 8, dtype=np.int32)
    for tid in range(warps * 32):
        for turn in range((k // 8 + warps * 32 - 1) // (warps * 32)):
            i = turn * warps * 32 + tid
            if i < k // 8:
                owners[i] += 1
                staged[8*i:8*i+8] = tags[8*i:8*i+8]
    assert np.all(owners == 1) and staged.tobytes() == tags.tobytes()
    wrong = np.roll(staged, 8)
    assert not np.array_equal(wrong, tags)  # wrong K coordinate is observable
    owners[0] -= 1
    assert not np.all(owners == 1)  # missing writer
    owners[0] += 2
    assert not np.all(owners == 1)  # duplicate writer


def test_changed_source_seams_reject_instead_of_silently_patching(bodies):
    body = static_body(bodies[0], "kpack_q4_small_static")[2]
    with pytest.raises(ValueError): stage_body(body.replace("    auto low=", "    auto weights="))
    with pytest.raises(ValueError): stage_body(body.replace("aligned_activation<Type>(c.a,", "different_a("))
    with pytest.raises(ValueError): restore_body(stage_body(body).replace("if (i < K / 8)", "if (i < K / 8 - 1)"))


def test_fixed_recipes_keep_every_historical_winner_and_s1(bodies):
    assert len(CASES) == 12 and len({(n, k) for n, k, *_ in CASES}) == 6
    assert all(r[-1] == x[-1] == 1 for _, _, _, r, x in CASES)
    assert len({r[:2] for _, _, _, r, _ in CASES}) == 5
    candidate = bodies[1]
    assert candidate.count("size_t(c.k)*sizeof(__half),stream>>>(c);") == 6
    assert "return launch<4,8,true>(c,f.split);" in candidate
    assert "return launch<32,8,true>(c,f.split);" not in candidate
    gate = dispatch((ROOT / "quactlize/execution/dispatch.cpp").read_text())
    for token in ("c->mode!=QKG_DENSE", "c->rows!=1", "c->experts!=1", "c->input_type!=QKG_F16",
                  "f->split!=1", "uintptr_t(c->a)&15", "uintptr_t(c->low)&7", "uintptr_t(c->units)&15"):
        assert gate.count(token) == 2
    for n, k, _, (c, w, _), _ in CASES:
        assert f"c->n=={n} && c->k=={k} && f->columns=={c} && f->warps=={w}" in gate


def test_compiled_host_admission_before_any_device_call(tmp_path):
    (tmp_path / "dispatch.cpp").write_text(dispatch((ROOT / "quactlize/execution/dispatch.cpp").read_text()))
    (tmp_path / "validation.hpp").write_text(validation((ROOT / "quactlize/execution/validation.hpp").read_text()))
    (tmp_path / "link_only.cpp").write_text('''#include "api.h"
extern "C" int qkg_launch_12(qkg_call_v1 const&, qkg_config_v1 const&) { return 99; }
extern "C" int qkg_pair_launch_12(qkg_call_v1 const&, qkg_config_v1 const&) { return 99; }
''')
    library = tmp_path / "host-admission.so"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", f"-I{tmp_path}",
                    f"-I{ROOT / 'quactlize/execution'}", f"-I{ROOT / 'quactlize/include'}",
                    str(tmp_path / "dispatch.cpp"), str(tmp_path / "link_only.cpp"), "-o", str(library)],
                   capture_output=True, text=True, check=True)
    lib = C.CDLL(str(library))
    query = lib.quactlize_kpack_gemv_pair_query_v1
    query.argtypes = [C.POINTER(Call), C.POINTER(Config), C.POINTER(Arrangement), C.POINTER(Sizes)]
    query.restype = C.c_int
    for n, k, _, recipe, _ in CASES:
        call = Call(version=1, size=C.sizeof(Call), qtype=12, n=n, k=k, experts=1, rows=1,
                    mode=0, input_type=0, channels=1, topk=1, a_row_stride=k, a_token_stride=k,
                    out_row_stride=n)
        cfg, arr, out = Config(*recipe), arrangement(12), Sizes()
        assert query(C.byref(call), C.byref(cfg), C.byref(arr), C.byref(out)) == 0
        assert out.workspace_bytes == 0 and out.low_bytes == n*k//2
        for field, bad in (("rows", 2), ("mode", 1), ("input_type", 1), ("experts", 2),
                           ("qtype", 13), ("n", n+16), ("a", 4), ("low", 2), ("units", 4)):
            original = getattr(call, field)
            setattr(call, field, bad)
            assert query(C.byref(call), C.byref(cfg), C.byref(arr), C.byref(out)) != 0
            setattr(call, field, original)
        cfg.split = 2
        assert query(C.byref(call), C.byref(cfg), C.byref(arr), C.byref(out)) != 0


def test_raw_compare_rejects_small_error_that_loose_oracle_could_accept():
    a = np.array([0.0, 1., -2., .5], dtype="<f4")
    assert len(check_exact(a.tobytes(), a.tobytes())) == 64
    b = a.copy(); b.view("<u4")[1] += 1
    with pytest.raises(ValueError, match="bad=1/4"): check_exact(a.tobytes(), b.tobytes())
    for bad in (b"", a.tobytes()[:-1], np.array([0., 1., np.nan, .5], dtype="<f4").tobytes()):
        with pytest.raises(ValueError): check_exact(a.tobytes(), bad)


def record():
    return dict(status="PASS", arm="new", variant="shared-a", recipe=[4, 8, 1], shape=[1, 5120, 8192],
                mode="warm", zero_code_negative="PASS", error=.00003, samples_us=[10., 11., 12.], median_us=11.,
                copies=1, weight_bytes=5120*8192*9//16, calls_per_graph=32,
                device=dict(l2_bytes=64*1024**2), output_type="F32", raw_control="EXACT_FP32",
                raw_output_sha256="a"*64, zero_a_check="PASS")


def parse(row):
    return parse_result("Q4_PPU_CELL " + json.dumps(row), "shared-a", [4, 8, 1], [1, 5120, 8192], "warm", 3)


def test_complete_receipt_requires_independent_and_exact_oracles():
    assert parse(record())["median_us"] == 11.


@pytest.mark.parametrize("key,value", [
    ("raw_control", "SKIP"), ("zero_a_check", "SKIP"), ("raw_output_sha256", ""),
    ("variant", "baseline"), ("output_type", "F16"), ("zero_code_negative", "SKIP"),
    ("error", float("nan")), ("samples_us", [1., 2.]), ("recipe", [4, 8, 2]),
])
def test_incomplete_or_dirty_receipt_fails_closed(key, value):
    with pytest.raises(ValueError): parse(record() | {key: value})


def test_box_script_preserves_shell_and_uses_prebuilt_no_jit():
    path = ROOT / "tools/run_q4_astage_ppu_box.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "\n(\n" in text and text.rstrip().endswith(")")
    assert "build_astage.py --" not in text and "JIT=NONE sweep=NONE" in text
    assert 'source "$SDK/envsetup.sh"' in text and '"$RUN.results.tgz"' in text


def test_published_addon_when_present():
    from dev.gemv_ppu.build_astage import verify
    path = ROOT / "prebuilt/ppu0010/q4-astage-v1"
    if not (path / "manifest.json").exists():
        pytest.skip("not yet packaged")
    m = verify(path, ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    assert m["device_validated"] is False and m["production_changed"] is False

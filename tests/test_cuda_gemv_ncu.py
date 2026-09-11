"""Host controls for the standalone profiler and N2 source experiment."""

import json
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from dev.gemv_cuda.build import n2_schedule, replace_dot
from dev.gemv_cuda.check_inputs import cases
from dev.gemv_cuda.export_standalone import export
from dev.gemv_cuda.run_standalone import parse
from reference.gguf_kpack import canonical_arrangement

ROOT = Path(__file__).resolve().parents[1]


def test_n2_uses_actual_offline_address_maps(tmp_path):
    exe = tmp_path / "maps"
    subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(ROOT / "quactlize/include"),
                    str(ROOT / "dev/gemv_cuda/n2_layout_check.cpp"), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert result.stdout.count("bad=0 adjacent_aligned=PASS") == 7
    assert "format=Q5-high cells=16384 bad=0 adjacent_aligned=PASS legacy_duplicate_high_bad=8192" in result.stdout


def test_n2_source_change_is_pair_only_and_keeps_b16_contract():
    old = (ROOT / "quactlize/execution/gemv.cu").read_text()
    changed = n2_schedule(replace_dot(old, "n2_dot.cuh"))
    begin, end = "    } else {\n    for (int g =", "    __shared__ float partial"
    assert changed[changed.index(begin):changed.index(end)] == old[old.index(begin):old.index(end)]
    assert "(Pair ? 2 : 1)" in changed
    assert "second+=partial[Warps*32+w*Columns+threadIdx.x]" in changed
    assert "if ((uintptr_t(ptr)&3)==0)" in changed
    assert "uint32_t(ptr[0])|(uint32_t(ptr[1])<<16)" in changed
    assert "col+1]=__int_as_float(0x7fc00000)" in changed
    assert "N bit 3" in changed
    with pytest.raises(ValueError, match="seam changed"):
        n2_schedule(changed)


def test_multi_token_is_not_one_larger_topk():
    profiles = list(cases())
    assert {(t, c) for mode, t, c in profiles if mode == 2} == {
        (t, c) for t in (1, 2, 3, 4) for c in (1, 8)}
    assert (1, 6, 1) in profiles
    assert {(m, t, c) for m, t, c in profiles if m == 0} == {(0, t, 1) for t in (1, 2, 4)}


@pytest.mark.parametrize("q", (8, 10, 11, 12, 13, 14))
def test_binary_fixture_descriptor_and_payload_are_preserved(tmp_path, q):
    arrays = dict(raw=np.arange(32, dtype="u1"), low=np.array([1, 7, 19, 37], dtype="<u2"),
                  high=np.empty(0, dtype="u1"), units=np.array([9, 2, 6, 5], dtype="u1"),
                  a=np.array([[.5, -.25]], dtype="f4"), ids=np.array([1, 3], dtype="i4"),
                  golden=np.array([[.25], [-1]], dtype="f8"), denom=np.ones((2, 1), dtype="f8"))
    src, dest = tmp_path / "fixture.npz", tmp_path / "fixture.bin"
    np.savez(src, q=q, n=256, k=512, experts=4, channels=1, **arrays)
    receipt = export(src, dest)
    header = struct.unpack("<Q8i8iQ8Q", dest.read_bytes()[:144])
    assert header[:9] == (0x3146584D5647514B, 1, q, 256, 512, 4, 2, 1, 0)
    if q != 8:
        a = canonical_arrangement(q)
        assert header[9:18] == tuple(getattr(a, name) for name in (
            "version", "layout", "bits", "high_bits", "artifact_tile_k", "transport_tile_k",
            "group_size", "reserved", "mapping_id"))
    else:
        assert header[9:18] == (2, 4, 8, 0, 0, 32, 32, 0, 0x51384B5032540001)
    assert dest.read_bytes()[144:] == b"".join(x.tobytes() for x in arrays.values())
    assert receipt["selected_ids"] == [1, 3]
    with pytest.raises(FileExistsError):
        export(src, dest)


@pytest.mark.parametrize("ids", ([0, 0], [-1, 1], [0, 4]))
def test_binary_fixture_does_not_hide_expert_aliases(tmp_path, ids):
    source = tmp_path / "bad.npz"
    np.savez(source, q=12, n=256, k=512, experts=4, channels=1, ids=np.array(ids))
    with pytest.raises(ValueError, match="distinct in-range"):
        export(source, tmp_path / "bad.bin")


def good_line():
    return "GEMV_STANDALONE q=12 kind=pair config=16-4-8 error=0.0001 status=PASS median_us=2.000000 samples=" + json.dumps([2.] * 15)


def test_samples_recompute_reported_median():
    assert parse(good_line())["samples_us"] == [2.] * 15


@pytest.mark.parametrize("plant", ("nan", "negative", "missing", "wrong-median", "duplicate", "bad-error", "bad-status"))
def test_invalid_timing_or_numeric_receipt_rejected(plant):
    text = good_line()
    if plant == "nan":
        text = text.replace("[2.0", "[NaN", 1)
    if plant == "negative":
        text = text.replace("[2.0", "[-2.0", 1)
    if plant == "missing":
        text = text.replace(", 2.0]", "]")
    if plant == "wrong-median":
        text = text.replace("median_us=2", "median_us=3")
    if plant == "duplicate":
        text += "\n" + text
    if plant == "bad-error":
        text = text.replace("error=0.0001", "error=nan")
    if plant == "bad-status":
        text = text.replace("status=PASS", "status=FAIL")
    with pytest.raises(ValueError):
        parse(text)

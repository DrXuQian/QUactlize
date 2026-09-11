"""The Q4 instruction experiment cannot change format, dispatch or dot ownership."""
from pathlib import Path
import json

import pytest

from dev.gemv_cuda.build import n2_schedule, q4_n2_source, replace_dot
from dev.gemv_cuda.compare_q4_native import parse_timing

ROOT = Path(__file__).resolve().parents[1]


def test_q4_native_preserves_n2_launch_and_reducer():
    old = (ROOT / "quactlize/execution/gemv.cu").read_text()
    baseline = n2_schedule(replace_dot(old, "n2_dot.cuh"))
    candidate = q4_n2_source(old)
    marker = "template<int Columns, int Warps, bool Pair = false>"
    assert candidate[candidate.index(marker):] == baseline[baseline.index(marker):]
    assert "#if QKG_QTYPE != 12" in candidate
    assert "generic_n2_pair_dot<Reader>" in candidate
    assert "R::LowMap::word_index(col, g*32+r, c.n)" in candidate
    assert "g += split*workers" in candidate
    assert "a_base + g*32" in candidate


def test_q4_native_keeps_half_boundary_and_weak_alignment():
    source = (ROOT / "dev/gemv_cuda/q4_native.cuh").read_text()
    assert "uintptr_t(p) & 15" in source
    assert "p[4*i+3]" in source
    assert "uintptr_t(p) & 3" in source
    assert "__float2half_rn(__half2float(zero) + 8.f * __half2float(scale))" in source
    assert "__hfma2(codes<Slot>(words[first+i]), scale, zero)" in source
    assert "x[i] = fmaf(av[i], w.x, x[i])" in source
    assert "y[i] = fmaf(av[i], w.y, y[i])" in source
    assert "__float2half_rn(v.x)" in source


def test_math_gate_uses_canonical_host_metadata_not_a_second_copy():
    source = (ROOT / "dev/gemv_cuda/q4_native_check.cu").read_text()
    assert "packed_unit::unit_group<KType::Q4_K,8>(p, g)" in source
    assert "packed_unit::put_code<KType::Q4_K>" in source
    assert "{0, 1, 2, 4, 8, 15}" in source
    assert "cudaGetLastError" in source


@pytest.mark.parametrize("plant", (None, "arm", "recipe", "mode", "shape", "samples", "median", "nan", "error"))
def test_q4_timing_cannot_mix_shapes_recipes_or_bad_samples(plant):
    line = ("Q4_LAYOUT_PROFILE arm=kpack config=16-4-8 shape=1x4096x2048 sm=48 L2_bytes=50331648 "
            "copies=24 mode=rotating purpose=timing error=0.0001 median_us=4.0 status=PASS samples="
            + json.dumps([4.0]*15))
    replacements = dict(arm=("arm=kpack", "arm=xplane"), recipe=("config=16-4-8", "config=16-4-1"),
                        mode=("mode=rotating", "mode=warm"), shape=("shape=1x4096x2048", "shape=1x4096x4096"),
                        samples=(", 4.0]", "]"), median=("median_us=4.0", "median_us=3.0"),
                        nan=("[4.0", "[NaN"), error=("error=0.0001", "error=NaN"))
    if plant:
        line = line.replace(*replacements[plant])
        with pytest.raises(ValueError):
            parse_timing(line, "kpack", [16,4,8], "rotating", [1,4096,2048])
    else:
        assert parse_timing(line, "kpack", [16,4,8], "rotating", [1,4096,2048])["samples_us"] == [4.]*15

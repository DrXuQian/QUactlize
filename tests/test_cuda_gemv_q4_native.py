"""The Q4 instruction experiment cannot change format, dispatch or dot ownership."""
from pathlib import Path
import json

import pytest

from dev.gemv_cuda.build import (n2_schedule, q4_n2_source, q4_wide_source, q4_wide_validation,
                                q4_grid_source, q4_aligned_source, replace_dot)
from dev.gemv_cuda.build import q4_n4_source, q4_tree_source, q4_small_source, q4_warp_source, q4_coop_source, q4_static_source
from dev.gemv_cuda.build import q4_balanced_source
from dev.gemv_cuda.compare_q4_native import parse_timing
from dev.gemv_cuda.build_xplane_accum import fp32_source

ROOT = Path(__file__).resolve().parents[1]


def test_xplane_fp32_control_only_changes_dot_and_intragroup_reduction():
    old=(ROOT / "quactlize/include/gguf_bc_q4_gemv.hpp").read_text()
    new=fp32_source(old)
    assert "half2& d0" not in new
    assert "float2& d0" in new
    assert "__half22float2(w)" in new and "fmaf(wf.x,af.x,sum.x)" in new
    # Neither the launch, global addressing nor the shared A staging changes.
    for seam in ("template <int CTA_N, int WARPS_N, int WARPS_K = 1>\ninline void launch",
                 "  unsigned const tid = threadIdx.x;", "    uint4 metadata[CTA_N], quant[CTA_N];"):
        if "inline void launch" in seam:
            assert old[old.index(seam):] == new[new.index(seam):]
    a="  unsigned const tid = threadIdx.x;"
    b="      half2 d0 = __float2half2_rn(0.0f);"
    assert old[old.index(a):old.index(b)] == new[new.index(a):new.index("      float2 d0 = make_float2(0.f,0.f);")]
    assert old[old.index("#pragma unroll\n  for (int ii = 0; ii < CTA_N; ++ii) {\n    float v"):] == new[new.index("#pragma unroll\n  for (int ii = 0; ii < CTA_N; ++ii) {\n    float v"):]


def test_wide_n2_keeps_existing_code_and_only_adds_q4_pair_launches():
    old = (ROOT / "quactlize/execution/gemv.cu").read_text()
    baseline, candidate = q4_n2_source(old), q4_wide_source(old)
    prefix, tail = candidate.split("#if QKG_QTYPE == 12\n", 1)
    extra, suffix = tail.split("#endif\n", 1)
    assert prefix + suffix == baseline
    for c in (4, 8):
        for w in (2, 4, 8):
            assert f"launch<{c},{w},true>" in extra
    validation = (ROOT / "quactlize/execution/validation.hpp").read_text()
    changed = q4_wide_validation(validation)
    assert "pair && c.qtype == 12 && (f.columns == 4 || f.columns == 8)" in changed
    assert "(f.columns != 16 && f.columns != 32) ||" in validation


def test_grid_preserves_dot_and_reducer_and_owns_all_axes():
    old=(ROOT/"quactlize/execution/gemv.cu").read_text()
    native,grid=q4_wide_source(old),q4_grid_source(old)
    a="__global__ void kpack_gemv_reduce"
    b="template<int Columns, int Warps, bool Pair = false> int launch"
    assert native[native.index(a):native.index(b)]==grid[grid.index(a):grid.index(b)]
    assert "tile=blockIdx.x, partition=blockIdx.y, row=blockIdx.z" in grid
    assert "dim3 const grid(c.n/(Columns*(Pair ? 2 : 1)),split,c.rows)" in grid
    assert "c.rows>65535" in grid
    assert "int const outer = int(blockIdx.x) / tiles" not in grid


def test_alignment_specialization_has_guarded_fallback_and_same_dot_order():
    old=(ROOT/"quactlize/execution/gemv.cu").read_text()
    source=q4_aligned_source(old)
    assert "!(uintptr_t(low)&3) && !(uintptr_t(units)&15)" in source
    assert "!((uintptr_t(c.a)+2*a_base)&3)" in source
    assert "!((uintptr_t(c.a)+4*a_base)&15)" in source
    assert "return fallback_q4_pair_dot<Reader>" in source
    assert "aligned_dot_slot<3,Type>" in source
    assert "g += split*workers" in source


def test_n4_guards_vector_alignment_without_narrowing_n2_fallback():
    old=(ROOT/"quactlize/execution/gemv.cu").read_text()
    source=q4_n4_source(old)
    assert "!(uintptr_t(c.low)&7) && !(uintptr_t(c.units)&15)" in source
    assert "!(c.a_row_stride&1)" in source and "!(c.a_row_stride&3)" in source
    assert "dim3 const n4_grid(c.n/(Columns*4),split,c.rows)" in source
    assert "kpack_gemv<Columns,Warps,Pair><<<grid,Warps*32,0,stream>>>" in source
    for col in (1,2,4,8):
        assert f"launch<{col},2,true>" in source
    assert "R::LowMap::word_index(col,g*32+r,c.n)" in source
    assert "g+=split*Workers" in source


def test_n4_tree_only_changes_fp32_reduction_and_adds_w16_dispatch():
    old=(ROOT/"quactlize/execution/gemv.cu").read_text()
    base,tree=q4_n4_source(old),q4_tree_source(old)
    begin="__global__ void kpack_q4_n4"
    end="    constexpr int T=Warps*32;"
    assert base[base.index(begin):base.index(end,base.index(begin))] == tree[
        tree.index(begin):tree.index("    float v0=(x[0]+x[1])",tree.index(begin))]
    assert "for(int distance=Columns;distance<32;distance*=2)" in tree
    assert "for(int w=0;w<Warps;++w)" in tree
    for c in (1,2,4,8,16,32):
        assert f"launch<{c},16,true>" in tree
    # Each output column's subgroup stays closed under every XOR step.
    for columns in (1,2,4,8,16,32):
        for lane in range(32):
            owners={lane}
            distance=columns
            while distance<32:
                owners |= {x^distance for x in owners}
                distance*=2
            assert owners==set(range(lane%columns,32,columns))


def test_standalone_w16_is_kpack_only():
    from dev.gemv_cuda.tune_q4_standalone import recipes
    rows=recipes((1,2,4,8,16,32),(2,4,8,16))
    assert len(rows)==108 and len(set(rows))==108
    assert all(w!=16 for arm,c,w,s in rows if arm=="xplane")
    assert len([r for r in rows if r[0]=="kpack" and r[2]==16])==24


def test_small_n_tree_has_two_columns_and_b32_alignment():
    source=q4_small_source((ROOT/"quactlize/execution/gemv.cu").read_text())
    assert "kpack_q4_n4" not in source
    assert "kpack_q4_n2_tree<Columns,Warps,0>" in source
    assert "n4_grid(c.n/(Columns*2),split,c.rows)" in source
    assert "!(uintptr_t(c.low)&3)" in source
    assert "col+3" not in source
    assert "words[r]=aligned_word(low+R::LowMap::word_index(col,g*32+r,c.n))" in source
    assert "kpack_gemv<Columns,Warps,Pair><<<grid,Warps*32,0,stream>>>" in source


def test_warp_owned_output_covers_n_once_without_a_cta_barrier():
    source=q4_warp_source((ROOT/"quactlize/execution/gemv.cu").read_text())
    begin=source.index("__global__ void kpack_q4_n2_warp")
    end=source.index("\n#endif\ntemplate<int Columns",begin)
    assert "__shared__" not in source[begin:end]
    assert "__syncthreads" not in source[begin:end]
    for n in (256,512,1024):
        for columns in (1,2,4,8,16,32):
            for warps in (2,4,8,16):
                tile=2*columns*warps
                owners=[]
                for block in range((n+tile-1)//tile):
                    for w in range(warps):
                        lanes=[(block*warps*columns+w*columns+t%columns)*2 for t in range(32)]
                        assert all(c<n for c in lanes) or all(c>=n for c in lanes)
                        owners.extend(c+i for c in lanes[:columns] if c<n for i in range(2))
                assert sorted(owners)==list(range(n))


def test_cooperative_epilogue_keeps_dot_and_sums_every_warp_once():
    old=(ROOT/"quactlize/execution/gemv.cu").read_text()
    tree,coop=q4_tree_source(old),q4_coop_source(old)
    seam="    constexpr int T=Warps*Columns;"
    expected=tree[:tree.index(seam)].replace("kpack_q4_n4","kpack_q4_n4_coop")
    assert coop.startswith(expected)
    assert "__shared__ float4 partial[Warps*Columns]" in coop
    for c in (1,2,4,8,16,32):
        for w in (2,4,8,16):
            for col in range(c):
                entries=[v for lane in range(col,32,c) for v in range(lane//c,w,32//c)]
                assert sorted(entries)==list(range(w))


def test_small_static_is_bounded_to_f16_dense_m1_s1_and_preserves_fallback():
    source=q4_static_source((ROOT/"quactlize/execution/gemv.cu").read_text())
    assert "c.input_type==0 && c.mode==QKG_DENSE && c.rows==1 && split==1" in source
    assert "c.n==512 && c.k==2048" in source and "c.n==1024 && c.k==5120" in source
    assert "kpack_q4_n4_coop<Columns,Warps,0>" in source
    assert source.index("if (Pair && aligned_b && aligned_a)") < source.index("c.n==512 && c.k==2048")
    begin=source.index("__global__ void kpack_q4_small_static")
    end=source.index("\n#endif\ntemplate<int Columns",begin)
    body=source[begin:end]
    assert "c.n" not in body and "c.k" not in body and "expert_for" not in body
    for k in (2048,5120):
        for c in (1,2,4,8,16,32):
            for w in (2,4,8,16):
                workers=w*32//c
                groups=[p*workers+i for p in range((k//32+workers-1)//workers)
                        for i in range(workers) if p*workers+i<k//32]
                assert groups==list(range(k//32))


def test_balanced_warps_are_only_an_extra_s1_q4_domain():
    from dev.gemv_cuda.tune_q4_standalone import recipes
    source=q4_balanced_source((ROOT/"quactlize/execution/gemv.cu").read_text())
    assert "if (f.split==1)" in source
    for w in (5,10):
        assert f"launch<4,{w},true>" in source
    rows=recipes((1,2,4,8,16,32),(2,4,8,16),(5,10))
    assert len(rows)==120 and len(set(rows))==120
    assert all(arm=="kpack" and s==1 for arm,c,w,s in rows if w in (5,10))
    # The K=5120/C4 subjects have no final partially active K-group pass.
    assert all(160%(w*32//4)==0 for w in (5,10))


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

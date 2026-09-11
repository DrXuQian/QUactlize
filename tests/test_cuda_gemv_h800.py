"""CPU proofs for the bounded CUDA experiment, not device admission."""
from pathlib import Path

import numpy as np
import pytest

from dev.gemv_cuda.build_h800_candidates import source

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("columns", [1, 2, 4])
@pytest.mark.parametrize("warps", [2, 4, 5, 8, 10, 16])
@pytest.mark.parametrize("width", [2, 4, 8, 16])
@pytest.mark.parametrize("k", [2048, 4096, 5120, 8192])
def test_wide_lane_and_word_ownership(columns, warps, width, k):
    # Per K group, exactly Columns threads own disjoint vectors of width N.
    workers = warps * 32 // columns
    covered = []
    for tid in range(warps * 32):
        gs = np.arange(tid // columns, k // 32, workers)
        ns = np.arange(tid % columns * width, (tid % columns + 1) * width)
        covered.extend((gs[:, None] * columns * width + ns).reshape(-1))
    assert np.array_equal(np.sort(covered), np.arange(k // 32 * columns * width))
    # The actual b16/nibble map must reconstruct logical K, including the
    # 8-spaced K nibbles and the two 4-residue vector loads.
    logical = np.arange(k)
    word = (logical // 32) * 8 + logical % 8
    nibble = logical % 32 // 8
    recovered = word // 8 * 32 + word % 8 + 8 * nibble
    assert np.array_equal(recovered, logical)
    bad = word // 8 * 32 + word % 8 + nibble  # planted adjacent-K interpretation
    assert np.count_nonzero(bad != logical) == 3 * k // 4


@pytest.mark.parametrize("arm", ["shared-a", "shared-b", "warp-global", "warp-shared", "warp-sb",
                                "n8-global", "n8-shared", "n8-static-global", "n8-static-shared", "n2-global",
                                "ldmatrix-single", "ldmatrix-pipeline"])
def test_candidate_generators_are_dense_f16_s1_only(arm):
    body, recipes = source(arm)
    wrapper = body[body.index('extern "C" int qkg_pair_launch_12'):]
    assert "c.mode!=QKG_DENSE || c.rows!=1 || c.experts!=1 || c.input_type!=QKG_F16 || f.split!=1" in wrapper
    assert "return QKG_INVALID;" in wrapper
    assert recipes and len(set(recipes)) == len(recipes)
    assert "float2 accum" in body if arm.startswith("n8-") else True
    assert "__hfma2(q, scale[p], zero[p])" in body if arm.startswith("n8-") else True


def test_nwide_never_uses_fp16_dot_accumulators():
    body = (ROOT / "dev/gemv_cuda/q4_nwide.cuh").read_text()
    assert "float2 accum[Pairs][2]{}" in body
    assert "fmaf(ax[r], w.x, accum[p][r%2].x)" in body
    assert "fmaf(ax[r], w.y, accum[p][r%2].y)" in body
    assert "__shared__ float2 partial" in body
    assert "reinterpret_cast<uint4 const*>" in body
    assert "__hfma2" in body  # weight affine only
    assert "__hfma2(q, scale[p], zero[p])" in body


@pytest.mark.parametrize("width", [8, 16, 32])
@pytest.mark.parametrize("warps", [2, 4, 5, 8, 10, 16])
@pytest.mark.parametrize("k", [2048, 4096, 5120, 8192])
def test_cooperative_metadata_donors_and_k_coverage(width, warps, k):
    coverage = []
    for warp in range(warps):
        for chunk in range(warp, k // 128, warps):
            lane = np.arange(32)
            group = chunk * 4 + lane // 8
            for pair in range(width // 2):
                for component in (0, 1):
                    donor = (lane & ~7) + 2 * (pair % 4) + component
                    assert np.array_equal(chunk * 4 + donor // 8, group)
                    donor_column = (pair // 4) * 8 + donor % 8
                    assert np.all(donor_column == pair * 2 + component)
            coverage.extend((group[:, None] * 32 + lane[:, None] % 8 + 8 * np.arange(4)).reshape(-1))
    assert np.array_equal(np.sort(coverage), np.arange(k))


def test_unsigned_arm_keeps_input_bytes_but_removes_signed_compensation():
    body, _ = source("meta-global-unsigned")
    assert "return {scale,zero};" in body
    assert "__float2half2_rn(1032.f)" not in body
    assert "__float2half2_rn(1024.f)" in body
    assert "q4_cooperative_metadata" in body


@pytest.mark.parametrize("width,stride", [(8,1),(16,1),(32,1),(8,2),(8,4)])
def test_warp_reduce_scatter_owns_one_output_per_lane(width, stride):
    rng = np.random.default_rng(9216 + width + stride)
    original = rng.integers(-100, 100, (32, width)).astype(np.float32)
    value = original.copy()
    lane = np.arange(32)
    count, step = width, stride
    while count > 1:
        odd = (lane & step) != 0
        for i in range(count // 2):
            keep = np.where(odd, value[:, 2*i+1], value[:, 2*i])
            send = np.where(odd, value[:, 2*i], value[:, 2*i+1])
            value[:, i] = keep + send[lane ^ step]
        count //= 2; step *= 2
    out = value[:, 0]
    while step < 32:
        out = out + out[lane ^ step]
        step *= 2
    for tid in range(32):
        column_group, output_column = tid % stride, (tid // stride) % width
        assert out[tid] == original[column_group::stride, output_column].sum()
    wrong_group = original.sum(axis=0)
    if stride > 1: assert np.any(out != wrong_group[(lane // stride) % width])


@pytest.mark.parametrize("bias", [0, 8])
def test_shared_byte_shift_mantissa_conversion_is_exact(bias):
    word = np.arange(65536, dtype=np.uint32)
    for slot in range(4):
        pos = (slot & 1) * 4
        source = word >> (8 if slot >= 2 else 0)
        bits = ((source & (15 << pos)) | 0x6400).astype(np.uint16)
        value = bits.view(np.float16).astype(np.float32)
        # Both operands are powers of two / exactly representable. Model the
        # single half FMA with full precision followed by one final rounding.
        decoded = (value / (1 << pos) - (1024 >> pos) - bias).astype(np.float16)
        expected = ((word >> (4 * slot)) & 15).astype(np.int16) - bias
        assert np.array_equal(decoded.astype(np.float32), expected)


@pytest.mark.parametrize("tile_n", [8,16])
@pytest.mark.parametrize("tile_k", [256,512,1024])
def test_matrix_copy_swizzle_and_register_ownership(tile_n,tile_k):
    def addr(row,col):
        return row*tile_n+(col^((row&4)*2) if tile_n==16 else col)
    shared = np.full(tile_k//4*tile_n,-1,dtype=np.int64)
    for row in range(tile_k//4):
        for col in range(0,tile_n,8):
            for j in range(8):
                target=addr(row,col)+j
                assert target==addr(row,col+j)
                assert shared[target]==-1
                shared[target]=row*tile_n+col+j
    assert np.array_equal(np.sort(shared),np.arange(shared.size))
    warp_k=128 if tile_n==8 else 64
    owners=[]
    for micro in range(tile_k//warp_k):
        for lane in range(32):
            for v in range(4):
                n=lane//4+(0 if tile_n==8 else (v%2)*8)
                group=micro*(warp_k//32)+(v if tile_n==8 else v//2)
                for half in (0,1):
                    row=group*8+2*(lane%4)+half
                    logical=row*tile_n+n
                    assert shared[addr(row,n)]==logical
                    owners.append(logical)
    assert np.array_equal(np.sort(owners),np.arange(shared.size))
    # Each 8x8 operand uses all 32 banks (four per 16-byte row).
    banks=[(addr(row,0)//2+j)%32 for row in range(8) for j in range(4)]
    assert sorted(banks)==list(range(32))


@pytest.mark.parametrize("arm", ["meta-static-global-rs-fast-bare", "affine8-early-fast-bare"])
def test_private_bare_arguments_do_not_change_public_contract(arm):
    body, _ = source(arm)
    name = "q4_group_affine" if arm.startswith("affine") else "q4_cooperative_metadata"
    assert f"__global__ void {name}(void const* a_ptr,uint8_t const* low_ptr," in body
    assert ">>>(c.a,c.low,c.units,c.output);" in body
    assert 'extern "C" int qkg_pair_launch_12(qkg_call_v1 const& c,qkg_config_v1 const& f)' in body
    assert "c.mode!=QKG_DENSE || c.rows!=1 || c.experts!=1 || c.input_type!=QKG_F16 || f.split!=1" in body


def test_group_affine_has_a_distinct_fp32_weight_contract():
    body, _ = source("affine8-early-fast-bare")
    kernel = body[body.index("__device__ __forceinline__ float2 q4_affine_header"):]
    kernel = kernel[:kernel.index("template<int Columns, int Warps, bool Pair")]
    assert "__half2float(__ushort_as_half(uint16_t(u.x)))*sc" in kernel
    assert "dot[p].x=fmaf(ax[r],v.x,dot[p].x)" in kernel
    assert "total[p].x+=fmaf(s0.x,dot[p].x,s0.y*a_sum)" in kernel
    assert "__hfma2" not in kernel  # exact integer conversion lives in codes<>.
    assert "sum_g (d*s_g * dot(q_g,A_g) - dmin*m_g * sum(A_g)) in FP32" in body


@pytest.mark.parametrize("seed", [611, 1709, 4097])
def test_group_affine_packed_bytes_match_independent_gguf(seed):
    gguf = pytest.importorskip("gguf")
    from tools.kpack_warmup_fixture import prepare_expert
    n, k = 256, 512
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (n * k // 256, 144), dtype=np.uint8)
    for offset in (0, 2):
        header = rng.uniform(0.001, 0.03, len(raw)).astype("<f2")
        raw[:, offset:offset+2] = header.view(np.uint8).reshape(-1, 2)
    placed = prepare_expert(raw, 12, n, k)
    units = placed["units"].reshape(k//256, n, 16).copy().view("<u4").reshape(k//256,n,4)
    words = placed["low"].astype(np.uint32)
    a = rng.normal(0, 0.3, k).astype(np.float16).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    wrong = np.zeros(n, dtype=np.float32)
    for g in range(k//32):
        u = units[g//8].astype(np.uint64)
        run = (u[:,2]>>16) | (u[:,3]<<16) if g&4 else u[:,1] | ((u[:,2]&65535)<<32)
        shift = 6*(g&3)
        sc = ((run>>shift)&63).astype(np.float32)
        mn = ((run>>(24+shift))&63).astype(np.float32)
        d = (u[:,0]&65535).astype("<u2").view("<f2").astype(np.float32)
        dm = (u[:,0]>>16).astype("<u2").view("<f2").astype(np.float32)
        q = np.stack([((words[g*8:g*8+8]>>(slot*4))&15) for slot in range(4)]).reshape(32,n)
        ag = a[g*32:g*32+32]
        dot = (q.astype(np.float32)*ag[:,None]).sum(axis=0,dtype=np.float32)
        out += (d*sc)*dot - (dm*mn)*ag.sum(dtype=np.float32)
        wrong += (d*sc)*dot  # dropped zero/min plane must be detected.
    weight = gguf.quants.dequantize(raw.reshape(-1), gguf.GGMLQuantizationType.Q4_K).reshape(n,k).astype(np.float64)
    gold = weight @ a.astype(np.float64)
    denom = np.abs(weight) @ np.abs(a.astype(np.float64))
    assert np.max(np.abs(out-gold)/denom) < 1e-6
    assert np.max(np.abs(wrong-gold)/denom) > .005


@pytest.mark.parametrize("width", [4,8])
def test_cooperative_affine_group_scatter_never_crosses_scale_groups(width):
    rng=np.random.default_rng(9481+width)
    original=rng.integers(-100,100,(32,width)).astype(np.float32)
    lane=np.arange(32)
    value=original.copy()
    count,step=width,1
    while count>1:
        odd=(lane&step)!=0
        next_value=np.empty((32,count//2),dtype=np.float32)
        for i in range(count//2):
            keep=np.where(odd,value[:,2*i+1],value[:,2*i])
            send=np.where(odd,value[:,2*i],value[:,2*i+1])
            next_value[:,i]=keep+send[lane^step]
        value=next_value
        count//=2;step*=2
    out=value[:,0]
    while step<8:
        out=out+out[lane^step]
        step*=2
    for i in range(32):
        assert out[i]==original[i//8*8:(i//8+1)*8,i%width].sum()
    assert np.any(out!=out[lane^8])  # four different groups must not be mixed before affine.


def test_vector_a_register_transpose_preserves_every_half_bit():
    lane=np.arange(8)
    original=np.arange(32,dtype=np.uint16).reshape(8,2,2)
    def swap(value,bit):
        result=value.copy()
        for r in range(8):
            if r&bit:
                result[r,:,0]=value[r^bit,:,1]
                result[r,:,1]=value[r,:,1]
            else:
                result[r,:,0]=value[r,:,0]
                result[r,:,1]=value[r^bit,:,0]
        return result
    value=swap(original,1)
    send=value[lane,np.where(lane&2,0,1)].copy()
    value[lane,np.where(lane&2,0,1)]=send[lane^2]
    value=swap(value,4)
    got=value.transpose(0,2,1).reshape(8,4)
    assert np.array_equal(got,np.arange(32,dtype=np.uint16).reshape(4,8).T)


def test_shared_a_stage_is_inside_timed_launch():
    body,_=source("affine8-early-fast-bare-as")
    assert "reinterpret_cast<uint4*>(act_stage)[i]=reinterpret_cast<uint4 const*>(a_ptr)[i]" in body
    assert "aligned_activation<0>(act_stage," in body
    assert "size_t(c.k)*2,static_cast<hggcStream_t>" in body


@pytest.mark.parametrize("k", [2048,4096,5120,8192])
def test_cta_k_phase_keeps_all_groups_once(k):
    groups=k//32
    for phase in range(8):
        shifted=np.arange(groups)+phase*(groups//8)
        shifted=np.where(shifted>=groups,shifted-groups,shifted)
        assert np.array_equal(np.sort(shifted),np.arange(groups))
    body,_=source("affine8-early-fast-bare-skew")
    assert body.count("int g=logical_g+(int(blockIdx.x)%8)*(K/32/8)")==1


def test_s2_query_and_full_timed_launch_include_reducer():
    body,recipes=source("affine8-early-fast-bare-s2")
    wrapper=body[body.index('extern "C" int qkg_pair_launch_12'):]
    assert "f.split!=2" in wrapper
    assert "out_ptr[blockIdx.y*N+blockIdx.x*TileN+tid]=sum;" in body
    assert wrapper.count("q4_two_part_reduce<<<")==len(recipes)*6
    assert wrapper.count("static_cast<float*>(c.workspace)")==len(recipes)*6
    assert "col=(blockIdx.x*128+threadIdx.x)*4" in body
    for k in (2048,4096,5120,8192):
        groups=np.concatenate([np.arange(k//64)+part*(k//64) for part in (0,1)])
        assert np.array_equal(groups,np.arange(k//32))


@pytest.mark.parametrize("k", [2048,4096,5120,8192])
@pytest.mark.parametrize("warps", [2,4,5,8,10,16])
def test_two_residue_lane_ownership(k,warps):
    covered=[]
    for warp in range(warps):
        for chunk in range(warp,k//256,warps):
            for lane in range(32):
                g=chunk*8+lane//4
                for residue in (0,1):
                    for slot in range(4):
                        covered.append(g*32+2*(lane%4)+residue+8*slot)
    assert np.array_equal(np.sort(covered),np.arange(k))


def test_u32_unit_decode_matches_the_two_full_48_bit_runs():
    rng=np.random.default_rng(92751)
    u=rng.integers(0,2**32,(16384,4),dtype=np.uint32).astype(np.uint64)
    for group in range(8):
        shift=6*(group&3)
        run=(u[:,2]>>16)|(u[:,3]<<16) if group&4 else u[:,1]|((u[:,2]&65535)<<32)
        scales=((u[:,2]>>16)|(u[:,3]<<16))&0xffffffff if group&4 else u[:,1]
        minima=(u[:,3]>>8) if group&4 else ((u[:,1]>>24)|(u[:,2]<<8))&0xffffffff
        assert np.array_equal((scales>>shift)&63,(run>>shift)&63)
        assert np.array_equal((minima>>shift)&63,(run>>(24+shift))&63)
        assert np.any(((minima>>(shift+1))&63)!=((run>>(24+shift))&63))


def test_explicit_activation_vectors_preserve_all_thirty_two_half_values():
    values=np.arange(32,dtype=np.uint16)
    activation=values.view(np.uint32).reshape(4,4)
    for half in (0,1):
        for slot in range(4):
            bits=activation[slot,half*2:half*2+2]
            got=bits.copy().view(np.uint16)
            assert np.array_equal(got,values[slot*8+half*4:slot*8+half*4+4])
    body,_=source("affine8-early-fast-bare-a8-u32")
    assert "activation[slot]=*reinterpret_cast<uint4 const*>" in body
    assert "q4_unit_codes(u,group)" in body
    assert "(uintptr_t(c.a)&15)" in body
    body,_=source("affine8-early-fast-bare-a4")
    assert "uint2 const packed=*reinterpret_cast<uint2 const*>(p);" in body


def test_full_b_stage_keeps_the_portable_shared_memory_floor():
    body,recipes=source("matrix8-full-fast")
    assert "q4_ldmatrix_v2<8,5120," in body
    assert "q4_ldmatrix_v2<8,8192," not in body
    assert "int const chunks=TileK/TileK;" in body
    for _,warps in recipes:
        for k in (2048,4096,5120):
            assert k//4*8*2+warps*8*4+k*2<=48*1024


def test_direct_integer_codes_still_dot_in_fp32():
    body,_=source("affine8-early-fast-bare-a4-u32-int")
    kernel=body[body.index("__global__ void q4_group_affine"):]
    kernel=kernel[:kernel.index("\n}\n")]
    assert "float((code_word>>(4*slot))&15)" in kernel
    assert "float((code_word>>(16+4*slot))&15)" in kernel
    assert "dot[p].x=fmaf(ax[r],v.x,dot[p].x)" in kernel
    assert "codes<" not in kernel

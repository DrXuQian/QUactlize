"""Host contracts for the isolated H800 multi-row experiments."""
import ast
import ctypes
import json
import struct

import numpy as np
import pytest

from dev.h800_smallm import build_gemv, build_moe, run_gemv
from dev.h800_smallm import select as selection


@pytest.mark.parametrize('family',build_gemv.ARMS)
def test_candidate_preserves_m1_recipe_and_device_row_bases(family):
    text,recipes=build_gemv.kpack_source(family)
    if family=='scalar-small':
        assert 'int const rr=blockIdx.y;' in text
        assert 'q4_scalar_affine<' in text
        assert len(recipes['512x2048'])==12
        assert 'c.rows)' in text and 'c.ids[' in text
        return
    assert ('int const rr=blockIdx.y*RowTile;' if family.startswith('rows') else 'int const rr=blockIdx.y;') in text
    assert 'c.ids[int64_t(token)*c.ids_stride+slot]' in text
    assert 'int64_t(slot%c.channels)*c.a_row_stride' in text
    assert 'float* out_ptr=c.output+int64_t(rr)*c.out_row_stride;' in text
    assert 'int64_t(expert)*N*K/2' in text
    assert 'int64_t(expert)*N*K/16' in text
    if family.endswith('-small'):
        assert set(recipes)=={'512x2048'}
        expected=6 if family.startswith('rows') else 25 if family=='affine2-small' else 22 if family=='affine4-small' else 18
        assert len(recipes['512x2048'])==expected
    for (_,n,k),(arm,recipe) in build_gemv.POLICY.items():
        if not family.endswith(('-small','-medium')) and arm==build_gemv.ARMS[family]:
            assert recipe in recipes[f'{n}x{k}']
            assert len(recipes[f'{n}x{k}'])==18
    assert 'c.rows!=1' not in text
    assert 'if(split!=1) return -1;' in text
    assert 'cudaMemcpy' not in text


@pytest.mark.parametrize('kind',['xplane','reference'])
def test_fp32_controls_index_the_same_device_rows(kind):
    text=build_gemv.control_header(kind)
    wrapper,recipes=build_gemv.control_source(kind)
    assert 'int const rr=blockIdx.x;' in text
    assert 'auto act=static_cast<half const*>(c.a)+abase;' in text
    assert 'auto out=c.output+int64_t(rr)*c.out_row_stride;' in text
    assert 'float2 fp32_dot' in text
    assert len(recipes)==(12 if kind=='xplane' else 60)
    assert 'dim3(c.rows,c.n/' in wrapper
    assert 'c.k%1024' in wrapper
    assert 'p->a_row_stride%8' in wrapper
    assert 'launch_q4k_gemv' not in text  # no stale M1 host launcher


def test_call_fields_are_the_production_definition():
    source=(build_gemv.ROOT/'quactlize/execution/native.py').read_text()
    tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Call')
    namespace={'C':ctypes}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'Call','exec'),namespace)
    assert run_gemv.Call._fields_==namespace['Call']._fields_


def test_moe_candidates_retain_production_oracle_and_fallback():
    text=build_moe.source()
    assert 'router_top8_warp' in text
    assert 'if (publish) const_cast<int32_t*>(io.ids)[slot]=expert;' in text
    assert 'if (threadIdx.x<32 && router.bias)' in text
    assert 'experts==256 && topk==8 && tokens<=4' in text
    assert 'plan.gate.tile_m>=tokens && plan.down.tile_m>=tokens' in text
    assert 'else moe_chain_prepare<Shape,Stride>' in text
    for oracle in ('router_equivalence();','for (int replay=0;replay<7;++replay)',
                   'rounding_red','gate_up_red','m>32','invalid'):
        assert oracle in text
    assert 'router.logits+=int64_t(warp)*256' in text
    assert 'io.ids+=int64_t(warp)*io.ids_stride' in text
    assert 'router.weights+=warp*8' in text


def test_dense_fixture_retains_q4_kpack4_plane_extent(tmp_path):
    n,k=512,2048
    # Nonzero finite metadata; independently decoded by the official package.
    rng=np.random.default_rng(6417)
    raw=rng.integers(0,256,(n,k//256,144),dtype='u1')
    raw[:,:,:4]=np.array([.02,.01],dtype='<f2').view('u1')
    chunks=[raw.tobytes(),bytes(n*k//2),b'',bytes(n*k//16),
            bytes(k*4),bytes(4),bytes(n*8),bytes(n*8)]
    arr=[2,1,4,0,0,256,32,0,0x51344b5034540001]
    header=struct.pack('<Q8i8iQ8Q',0x3146584D5647514B,1,12,n,k,1,1,1,0,
                       *arr,*map(len,chunks))
    path=tmp_path/'fixture.bin';path.write_bytes(header+b''.join(chunks))
    _,low,units,weights,_=run_gemv.read_fixture(path,n,k)
    assert low.shape==(k//4,n)
    assert units.shape==(k//256,n,16)
    assert weights.shape==(n,k) and np.isfinite(weights).all()
    assert np.count_nonzero(weights)>n*k//2
    path.write_bytes(path.read_bytes()+b'\0')
    with pytest.raises(ValueError,match='length'):
        run_gemv.read_fixture(path,n,k)


@pytest.mark.parametrize('sample',[0.,-1.,float('nan'),float('inf')])
def test_selection_rejects_invalid_samples(tmp_path,sample):
    path=tmp_path/'case.json'
    path.write_text(json.dumps(dict(status='PASS',records=[dict(samples_us=[sample],error=0.,median_us=sample)])))
    with pytest.raises(ValueError,match='invalid screen'):
        selection.load_case(path)


def test_selection_retains_best_two_and_other_control_challenges(tmp_path):
    for i,family in enumerate(['small','affine4-small']):
        out=tmp_path/str(i);out.mkdir()
        records=[dict(arm=arm,recipe=[c,4,1],samples_us=[1.+c/10+i/10],
                      median_us=1.+c/10+i/10,error=0.)
                 for arm in ('kpack','xplane','reference') for c in (1,2,4)]
        (out/'cell.json').write_text(json.dumps(dict(status='PASS',n=512,k=2048,
            kpack_implementation=family,delta_pct={'xplane':5-i,'reference':5-i},records=records)))
    selected=selection.select([tmp_path/'0',tmp_path/'1'])['cell']
    assert selected['_implementation']=='affine4-small'
    assert selected['kpack']==[[1,4,1],[2,4,1]]


def test_multirow_candidate_has_cta_tail_and_shared_reduction_barrier():
    text,_=build_gemv.kpack_source('rows2-small')
    assert 'rr+r<c.rows' in text
    assert 'warp_value[RowTile*Warps*Width]' in text
    assert 'out_ptr[int64_t(r)*c.out_row_stride+first+tid]=sum;' in text
    assert 'if(c.mode!=QKG_DENSE) return -1;' in text


def test_vector_gather_retains_scalar_unaligned_heads_and_odd_k_fallback():
    text=build_moe.source()
    assert '(p.k&1)' in text and 'moe_m1_gather(plan,lane,stride); return;' in text
    assert 'int const head=(uintptr_t(gate)&3)?1:0;' in text
    assert 'if (tail<p.k)' in text
    assert '__floats2half2_rn(p.io.a[col],p.io.a[col+1])' in text

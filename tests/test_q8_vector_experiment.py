from dev.gemv_simt.q8_vector_access import pattern
from dev.gemv_simt.q8_vector_build import source
from quactlize.execution.simt_codegen import inventory
import pytest


def test_q8_inventory_keeps_all_incumbents_and_compute_types():
    text=source()
    for c in inventory(8):
        for arm in range(2):
            for compute in range(2):
                assert f'invoke<{arm},{compute},{c.variant},{c.columns},{c.warps},{c.values}>' in text
    assert 'd->call.input_type!=QKG_F32' in text
    assert 'query_v2(*d,*f,&a,sizes)' in text
    assert 'register_reuse_reduce<8>' in text


def test_vector_metadata_footprint_and_minimum_abi_alignment():
    for c in inventory(8):
        aligned=pattern(c,256,512)
        assert aligned['metadata_vectorized']
        assert aligned['candidate_metadata_requests']==1
        v=next(s for s in aligned['streams'] if s['name']=='metadata-vector')
        assert v['width_bytes']==2*c.values
        for lane,address in zip(v['lanes'],v['addresses']):
            group=aligned['lane_groups'][lane]
            column=aligned['lane_columns'][lane]
            assert address==2*(group*256+column)
        weak=pattern(c,256,512,bases=dict(A=0,low=0,high=0,units=2))
        assert not weak['metadata_vectorized']
        assert weak['candidate_metadata_requests']==c.values
        assert aligned['packed_b_live_words_per_thread']['candidate']*2==aligned['packed_b_live_words_per_thread']['baseline']


def test_same_logical_k_order_and_exhaustive_two_code_values():
    baseline=[s*8+h*4+r for h in range(2) for s in range(4) for r in range(4)]
    candidate=[seg*16+s*8+h*4+r for h in range(2) for seg in range(2) for s in range(2) for r in range(4)]
    assert baseline==candidate
    assert sorted(candidate)==list(range(32))
    for slot in (0,1):
        for x in range(256):
            for y in range(256):
                word=(x|(y<<16))<<(8*slot)
                bits=((word>>(8*slot))&0x00ff00ff)|0x64006400
                # Half exponent 0x6400 has an exact unit-quantized mantissa.
                assert (bits&1023)-128==x-128
                assert ((bits>>16)&1023)-128==y-128


def test_missing_sdk_l2_requires_verified_override():
    from dev.gemv_simt.q8_vector_run import l2_identity
    assert l2_identity(dict(l2_bytes=0))['l2_source']=='UNAVAILABLE'
    assert l2_identity(dict(l2_bytes=0),64*1024**2)['l2_bytes']==64*1024**2
    assert l2_identity(dict(l2_bytes=48*1024**2))['l2_source']=='RUNTIME_QUERY'
    with pytest.raises(ValueError):l2_identity(dict(l2_bytes=48*1024**2),64*1024**2)

import copy
import hashlib
import json

import numpy as np
import pytest

from tools.build_kpack_dispatch import plan
from tools.kpack_prefill_measurement import (Weights, families, requests, reducer_cases,
    partial_values, reducer_expected, hashes, validate_gemm, validate_reducer,
    validate_fixture, BOARD, SCOPE)
from tools.kpack_dequant_fixture import fixture
from tools.kpack_bf16_fixture import row_domain, Oracle


def test_selected_closure(tmp_path):
    parents,selected=plan(tmp_path,requests())
    assert len(families())==22 and len(selected)==88 and len(parents)==12
    assert all(r['status']=='SELECTED' and r['split']==1 for r in selected)
    assert len(reducer_cases())==48
    assert len({tuple(w[k] for k in ('compact','m','n','split')) for w in reducer_cases()})==48
    assert {w['split'] for w in reducer_cases()}=={2,4,8}


@pytest.mark.parametrize('q',[12,13])
def test_fixture_matches_existing_bytes_and_activation(q):
    w=dict(q=q,n=256,k=512,experts=2)
    a=Weights(w)
    planes,gold=fixture(q,256,512,2,1)
    assert hashes(planes)==a.identity['fixture_hashes']
    assert hashlib.sha256(gold.tobytes()).hexdigest()==a.identity['bf16_golden_sha256']
    _,sf=fixture(q,256,512,2,0)
    assert hashlib.sha256(sf.tobytes()).hexdigest()==a.identity['sf_golden_sha256']
    bits,coeff=Oracle(gold).activations(7)
    activation,co=a.activation(7)
    assert np.array_equal(coeff,co)
    assert np.array_equal(activation.astype('f4'),(bits.astype('u4')<<16).view('f4'))
    indices=np.array([0,0,0,1,1,1,1])
    expected=np.stack([co[i]@a.sums[e] for i,e in enumerate(indices)]).astype('f2')
    assert a.error(expected,co,indices)<.001
    assert a.error(np.zeros_like(expected),co,indices)>.005
    with pytest.raises(ValueError):a.error(np.full_like(expected,np.nan),co,indices)


def test_official_factorization_is_not_bf16_weight_oracle():
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    w=Weights(dict(q=12,n=256,k=256,experts=1))
    rng=np.random.default_rng(np.random.SeedSequence([935712,12,256,256,0]))
    raw=rng.integers(0,256,(256,144),dtype='u1')
    for off in (0,2):
        h=(rng.random(256)*.02+.005).astype('f2');raw[:,off:off+2]=h.view('u1').reshape(-1,2)
    weights=dequantize(raw.reshape(-1),GGMLQuantizationType.Q4_K).reshape(256,256).astype('f8')
    activation,c=w.activation(7)
    want=activation.astype('f8')@weights.T
    np.testing.assert_allclose(want,c@w.sums[0],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('s',[2,4,8])
def test_reducer_oracle_and_zero_plane_negative(s):
    parts=np.stack([partial_values(0,2048,i) for i in range(s)])
    total=np.zeros(2048,dtype='f4')
    for p in parts:total+=p
    assert np.array_equal(total.astype('f2'),reducer_expected(0,2048,s))
    bad=parts[1:].sum(0,dtype='f4').astype('f2')
    assert np.count_nonzero(bad!=total.astype('f2'))>1000


def gemm_record():
    w=families()[0];rows,_,rh=row_domain(2048,1)
    return dict(status='PASS',workload=w,tokens=2048,route=0,scope=SCOPE,
        sf_prepass_timed=False,reducer_included=True,error=.0001,
        rows=rows.tolist(),routes_sha256=rh,max_rows_bound=2048,
        input_ring_bytes=3*64*1024**2,device={'l2_bytes':64*1024**2},copies=4,calls_per_graph=8,
        samples_us=[1.]*15,median_us=1.,round_medians_us=[1.]*3,
        timing='CAPTURED_COMPLETE_RING_EVENTS_FIRST_USE_EXCLUDED',
        guards='PASS',zero_a='PASS',changed_a_graph='PASS',changed_a_sequence='PASS')


@pytest.mark.parametrize('key,value',[
    ('scope','PRODUCER_ONLY'),('reducer_included',False),('sf_prepass_timed',True),
    ('max_rows_bound',1024),('input_ring_bytes',1024),('calls_per_graph',7),
    ('error',float('nan')),('zero_a','MISSING'),('changed_a_graph','MISSING'),
    ('samples_us',[True]*15),('samples_us',[1.]*14),('median_us',2.),
    ('round_medians_us',[2.]*3),('routes_sha256','stale'),('status','FAIL')])
def test_gemm_rejects_wrong_scope_or_missing_proofs(key,value):
    r=gemm_record();validate_gemm(r,r['workload'],2048,0)
    r[key]=value
    with pytest.raises(ValueError):validate_gemm(r,r['workload'],2048,0)


def test_empty_high_hash():
    assert hashes({'x':np.empty((1,0),dtype='u2')})['x']==hashlib.sha256(b'').hexdigest()


def test_receipt_rejects_self_consistent_wrong_weights():
    row=json.loads(BOARD.read_text())['rows'][0];w=families()[0]
    identity=dict(fixture_hashes={},bf16_golden_sha256=row['golden_sha256'],sf_golden_sha256='scale')
    evidence={'weights':{w['id']:copy.deepcopy(identity)}}
    validate_fixture(w,2048,identity,evidence)
    identity['bf16_golden_sha256']='bad'
    with pytest.raises(ValueError):validate_fixture(w,2048,identity,evidence)


def reducer_record():
    w=reducer_cases()[0];size=w['m']*w['n']*w['split']*4
    copies=max(2,(3*64*1024**2+size-1)//size)
    return dict(status='PASS',workload=w,samples_us=[1.]*15,median_us=1.,round_medians_us=[1.]*3,
        timing='CAPTURED_COMPLETE_RING_EVENTS_FIRST_USE_EXCLUDED',calls_per_graph=copies*2,
        copies=copies,partial_bytes=size,input_ring_bytes=copies*size,device={'l2_bytes':64*1024**2},
        guards='PASS',raw_bad=0,negative_bad=100,producer_timed=False,
        partial_layout='FP32_S_M_N',output_dtype='FP16',addition_order='INCREASING_S',
        scope='REDUCER_ONLY_SYNTHETIC_PARTIALS_ROTATING_NOT_PRODUCER_CONSUMER_CACHE')


@pytest.mark.parametrize('key,value',[
    ('raw_bad',1),('negative_bad',0),('producer_timed',True),('partial_bytes',16),
    ('copies',1),('input_ring_bytes',1),('scope','FULL_OUTPUT'),('output_dtype','BF16'),
    ('addition_order','ARBITRARY'),('guards','SKIPPED')])
def test_reducer_rejects_false_timing(key,value):
    r=reducer_record();validate_reducer(r,r['workload'])
    r[key]=value
    with pytest.raises(ValueError):validate_reducer(r,r['workload'])


def test_shell_preserves_parent_and_python_before_sdk():
    import subprocess
    from pathlib import Path
    shell=Path('tools/run_kpack_prefill_measurement_ppu_box.sh')
    subprocess.run(['bash','-n',str(shell)],check=True)
    result=subprocess.run(['bash','-c',f'bash {shell} invalid; printf "PARENT_ALIVE\\n"'],capture_output=True,text=True)
    assert result.returncode==0 and 'PARENT_ALIVE' in result.stdout
    text=shell.read_text()
    assert text.index('PYTHON=$(command')<text.index('source "$SDK/envsetup.sh"')


def test_shipping_reducer_types_are_reused():
    from pathlib import Path
    src=Path('dev/prefill_components/reducer.cu').read_text()
    assert '__global__' not in src
    assert 'PpuMixedInputSplitKParallelM1FastReduction<2>' in src
    assert 'PpuMixedInputSplitKParallelCompactReduction<2>' in src

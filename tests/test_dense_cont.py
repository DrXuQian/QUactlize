"""Production contract and numerical regression tests."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from ci.support.io import inputs, expected, check_oracle, compare_bits, check_controls
ROOT=Path(__file__).resolve().parents[1]


def test_actual_cpp_eligibility(tmp_path):
    exe=tmp_path/'guard'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ROOT),
                    str(ROOT/'tests/dense_cont_host.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


@pytest.mark.parametrize('storage',[0,1])
@pytest.mark.parametrize('input',[0,1])
def test_device_oracle_vectors_cover_all_words_rounding_and_faults(storage,input):
    data=inputs(storage,input)
    if not input:assert np.array_equal(data,np.arange(65536,dtype='<u2'))
    else:
        assert np.isnan(data).any() and np.isinf(data).any()
        assert np.any((data!=0)&(np.abs(data)<np.finfo('f4').tiny))
        assert np.any(data==65504) and np.any(data==65520)
    want=expected(data,storage,input)
    check_oracle(want,want,storage,input)
    bad=want.copy();bad.view('<u2' if input else '<u4')[0]^=1
    with pytest.raises(ValueError,match='IEEE'):check_oracle(bad,want,storage,input)


@pytest.mark.parametrize('storage,input,error',[(0,0,None),(0,1,None),(1,1,None),
                                               (1,0,None),(1,0,0),(1,0,1)])
def test_nan_baseline_contract_is_only_bf16_widening_without_error(storage,input,error):
    source=np.array([0x7f82,0x7f81],dtype='<u2')
    scalar=np.array([0x7fffffff,0x7fffffff],dtype='<u4').view('<f4')
    vector=np.array([0x7fffffff,0x7f810000],dtype='<u4').view('<f4')
    # Exact observed low/high pair; all other storage/direction/error settings
    # must keep their stricter scalar contract, even for the same diagnostic.
    comparisons={f'{a}-vs-{b}':compare_bits(x,y,1,0,source) for a,b,x,y in
                 ((1,0,vector,scalar),(4,0,vector,scalar),(4,1,vector,vector))}
    for output in (scalar,vector):check_oracle(output,expected(source,1,0),1,0)
    assert comparisons['1-vs-0']['first']==[dict(index=1,input_bits='0x7f81',
        want_bits='0x7fffffff',got_bits='0x7f810000',both_nan=True)]
    if storage==1 and not input and not error:
        assert check_controls(comparisons,storage,input,error)
    else:
        with pytest.raises(ValueError,match='scalar-control'):
            check_controls(comparisons,storage,input,error)


@pytest.mark.parametrize('word',[0,0x80000000,0x7f800000,0x7f810001,0xff810000,0x7fc10000])
def test_bf16_candidate_payload_class_sign_and_quiet_bit_remain_strict(word):
    source=np.array([0x7f81],dtype='<u2')
    scalar=np.array([0x7fffffff],dtype='<u4').view('<f4')
    old=np.array([0x7f810000],dtype='<u4').view('<f4')
    candidate=np.array([word],dtype='<u4').view('<f4')
    comparisons={f'{a}-vs-{b}':compare_bits(x,y,1,0,source) for a,b,x,y in
                 ((1,0,old,scalar),(4,0,candidate,scalar),(4,1,candidate,old))}
    with pytest.raises(ValueError,match='candidate differs'):
        check_controls(comparisons,1,0,None)


@pytest.mark.parametrize('storage',[0,1])
@pytest.mark.parametrize('input',[0,1])
@pytest.mark.parametrize('kind',['finite','zero-sign','infinity-sign','nan-class','nan-payload'])
def test_bit_diagnostics_distinguish_classes_without_relaxing_raw_equality(storage,input,kind):
    words=([0,0x8000,0x3f80,0x7f80,0x7fc1] if storage else [0,0x8000,0x3c00,0x7c00,0x7e01]) if input else [
        0,0x80000000,0x3f800000,0x7f800000,0x7fc00001]
    want=np.array(words,dtype='<u2' if input else '<u4')
    if not input:want=want.view('<f4')
    got=want.copy();raw=got.view('<u2' if input else '<u4')
    sign=0x8000 if input else 0x80000000
    if kind=='finite':raw[2]^=1
    elif kind=='zero-sign':raw[0]^=sign
    elif kind=='infinity-sign':raw[3]^=sign
    elif kind=='nan-class':raw[4]=0
    else:raw[4]^=1
    source=np.zeros(5,dtype='<f4' if input else '<u2')
    d=compare_bits(got,want,storage,input,source)
    assert d['raw_bad']==1
    assert d['nan_encoding_bad']==int(kind=='nan-payload')
    assert d['nan_class_bad']==int(kind=='nan-class')
    assert d['non_nan_bad']==int(kind in ('finite','zero-sign','infinity-sign'))
    assert d['first'][0]['got_bits']!=d['first'][0]['want_bits']
    json.dumps(d,allow_nan=False)

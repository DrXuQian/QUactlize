"""Production contract and numerical regression tests."""
import numpy as np
import pytest
from ci.support.reducer import partials, expected, cases
def test_independent_oracle_detects_changed_addition_order():
    x=partials(3,64,4);correct=expected(x);wrong=expected(x[[0,2,1,3]])
    assert correct[0]==np.float16(1).view('u2') and correct[1]==0
    assert np.any(correct!=wrong)
    assert correct[2]==np.float16(1).view('u2') and correct[5]==0x7c00


@pytest.mark.parametrize('m',[3072,4096,5120,8192])
@pytest.mark.parametrize('width',[2,4,8])
def test_reducer_gate_covers_extended_m_span(m,width):
    point=dict(workload=dict(operation='dense',compute='f16',m=m,n=256),
               cases=[dict(arm='sf',recipe=dict(parameters=[['split',4],['dense_reducer',width]]))])
    assert (m,256,256,4,0,0) in cases([point])


@pytest.mark.parametrize('change',[dict(m=8193),dict(m=8),dict(n=65537),dict(n=0),
    dict(m=9,n=7),dict(compute='bf16'),dict(operation='grouped'),dict(split=3),
    dict(width=3),dict(width=True),dict(arm='fq')])
def test_reducer_gate_keeps_non_m_contracts(change):
    w=dict(operation='dense',compute='f16',m=5120,n=256)
    w.update({k:v for k,v in change.items() if k in w})
    point=dict(workload=w,cases=[dict(arm=change.get('arm','sf'),recipe=dict(parameters=[
        ['split',change.get('split',2)],['dense_reducer',change.get('width',2)]]))])
    with pytest.raises(ValueError,match='M=.*N=.*split=.*width='):cases([point])

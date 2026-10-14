import pytest

from dev.strategy.coverage import execution_gaps, require_executable


def test_factored_domain_and_controls_cannot_be_silently_skipped():
    point=dict(id='p',workload=dict(operation='indexed',m=128,qtype=12),
               candidates=[dict(kind='q4',route='fq'),dict(kind='tc',route='sf')],
               expansion_search=[dict(configs=[0,1])],grid_search={'recipe':{}},
               champion_controls=[dict(kind='tc',route='fq')])
    result=execution_gaps(dict(points=[point]))
    assert result['status']=='BLOCKED'
    assert not result['device_started']
    assert len(result['gaps'])==5
    assert all(points==['p'] for points in result['gaps'].values())
    with pytest.raises(ValueError,match='no device workload or JIT'):
        require_executable(dict(points=[point]))


def test_empty_diagnostic_plan_starts_no_work():
    assert require_executable(dict(points=[]))['status']=='READY'

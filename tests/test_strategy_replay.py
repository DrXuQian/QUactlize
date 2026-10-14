import pytest

from dev.strategy.replay import compare
from quactlize.dispatch.table import Workload
from test_strategy_prescreen import simt


def test_replay_does_not_inject_faster_target_answer():
    w=Workload(12,1024,2048,1);rules=[simt(0)];old=tuple(rules)
    result=compare(w,rules,[(simt(0),20.),(simt(1),10.)])
    assert result['status']=='KNOWN_HISTORICAL_GAP' and result['regret_pct']==100
    assert tuple(rules)==old


def test_replay_missing_measurement_is_not_proven_regression_or_parity():
    w=Workload(12,1024,2048,1)
    assert compare(w,[simt(0)],[])['status']=='NO_COMPATIBLE_HISTORICAL_REFERENCE'
    result=compare(w,[simt(0)],[(simt(1),10.)])
    assert result['status']=='NO_MEASURED_RULE_CANDIDATE' and result['regret_pct'] is None


def test_replay_near_tie_is_not_strict_no_regression():
    w=Workload(12,1024,2048,1)
    result=compare(w,[simt(0)],[(simt(0),10.4),(simt(1),10.)])
    assert result['status']=='WITHIN_5PCT_NOT_STRICT_PARITY'
    assert compare(w,[simt(1)],[(simt(0),10.4),(simt(1),10.)])['status']=='HISTORICAL_BEST_INCLUDED'


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),0.,-1.])
def test_replay_rejects_invalid_costs(invalid):
    with pytest.raises(ValueError,match='historical confirmation'):
        compare(Workload(12,1024,2048,1),[simt()],[(simt(),invalid)])

"""A short final sample uses observed counts; missing whole samples still fail."""
from copy import deepcopy

import pytest

from dev.strategy.asys_components import parse_component_rows
from test_asys_component_coverage import fixture


def correlated():
    records,schedule,rows=fixture();apis=[]
    for i,item in enumerate(schedule):
        cid=i+1
        apis.append(dict(name='hggcGraphLaunch',correlation_id=cid,return_value=0,
                         start=i*100,end=i*100+10,tid=1))
        for j,row in enumerate(rows[i*16:(i+1)*16]):
            row.update(correlation_id=cid,graph_node_id=j+1)
    apis += [dict(name=name,correlation_id=0,return_value=0,start=t,end=t+1,tid=1)
             for name,t in (('hggcStreamSynchronize',1200),('hggcProfilerStop',1210))]
    return records,schedule,rows,apis


def parse(records,schedule,rows,apis):
    return parse_component_rows(records,schedule,rows,{},4,3,
        runtime_apis=apis,allow_terminal_short=True)


def test_terminal_short_uses_fifteen_real_intervals_not_sixteen_or_imputation():
    records,schedule,rows,apis=correlated()
    before=deepcopy((records,schedule,rows,apis))
    result=parse(records,schedule,rows[:-1],apis)['test-component']
    assert result['mean_us']==1 and result['samples_us']==[[1]*3]*4
    tail=result['trace_slices'][-1]
    assert tail['count']==15 and tail['expected_count']==16
    assert tail['durations_ns']==[1000]*15 and tail['first_activity']==176
    proof=result['terminal_activity_exception']
    assert proof['imputed_intervals']==0
    assert proof['expected_total']==192 and proof['observed_total']==191
    assert proof['denominator']=='OBSERVED_GPU_ACTIVITIES'
    assert (records,schedule,rows,apis)==before


@pytest.mark.parametrize('fault', ['two','middle-call','wrong-correlation',
    'duplicate-correlation','missing-api','failed-launch','grid','empty-last-call'])
def test_missing_or_unproven_activity_is_still_rejected(fault):
    records,schedule,rows,apis=correlated()
    if fault=='two':rows=rows[:-2]
    elif fault=='middle-call':rows.pop(15)
    elif fault=='empty-last-call':rows=rows[:-16]
    else:
        rows.pop()
        if fault=='wrong-correlation':rows[-1]['correlation_id']=999
        elif fault=='duplicate-correlation':apis[-3]['correlation_id']=1
        elif fault=='missing-api':apis.pop(0)
        elif fault=='failed-launch':apis[-3]['return_value']=1
        elif fault=='grid':rows[-1]['grid']=[2,1,1]
    with pytest.raises(ValueError):parse(records,schedule,rows,apis)


def test_same_kernel_last_sample_does_not_require_graph_node_or_stop_metadata():
    records,schedule,rows,apis=correlated()
    for row in rows:row.pop('graph_node_id')
    rows.pop(-2)
    result=parse(records,schedule,rows,apis[:-2])['test-component']
    assert result['trace_slices'][-1]['count']==15 and result['mean_us']==1


def test_strict_callers_still_reject_and_complete_captures_are_unchanged():
    records,schedule,rows,apis=correlated()
    with pytest.raises(ValueError,match='coverage incomplete'):
        parse_component_rows(records,schedule,rows[:-1],{},4,3)
    assert parse(records,schedule,rows,apis)==parse_component_rows(records,schedule,rows,{},4,3)

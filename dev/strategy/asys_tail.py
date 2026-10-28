"""Use observed intervals for a single short final sample, without imputation."""
from collections import defaultdict

POLICY='OBSERVED_COUNT_SINGLE_SHORT_FINAL_SAMPLE_V1'


def terminal_loss(records,schedule,rows,apis):
    def require(ok,message):
        if not ok:raise ValueError('terminal Asys activity exception rejected: '+message)
    expected=sum(x['kernels'] for x in schedule)
    require(bool(schedule) and len(rows)==expected-1,'not exactly one missing activity')
    calls=[a for a in apis if a['name'] in ('hggcGraphLaunch','cudaGraphLaunch')]
    require(len(calls)==len(schedule),'graph launch count differs from sample schedule')
    ids=[a['correlation_id'] for a in calls]
    require(all(type(i)==int and i>0 for i in ids) and len(set(ids))==len(ids),
            'missing or duplicate launch correlation')
    require(all(a['return_value']==0 for a in calls),'unsuccessful graph launch')
    grouped=defaultdict(list)
    for row in rows:grouped[row.get('correlation_id')].append(row)
    require(set(grouped)==set(ids),'unassigned or entirely missing graph activity')
    cursor=0
    for index,(item,call) in enumerate(zip(schedule,calls)):
        group=grouped[call['correlation_id']]
        want=item['kernels']-int(index==len(calls)-1)
        require(len(group)==want,'non-terminal missing or extra activity')
        require(rows[cursor:cursor+want]==group,'interleaved graph activity')
        cursor+=want
        # The normal parser below checks kernel identity and positive durations.
    last=schedule[-1];call=calls[-1];count=last['kernels']
    require(count==records[last['component']]['calls_per_graph'] and count>=16,'invalid repeat count')
    return dict(policy=POLICY,component=last['component'],round=last['round'],sample=last['sample'],
        expected_total=expected,observed_total=len(rows),expected_count=count,observed_count=count-1,
        launch_correlation_id=call['correlation_id'],
        denominator='OBSERVED_GPU_ACTIVITIES',imputed_intervals=0,
        interpretation='EXECUTION_COUNT_UNCHANGED_ONE_TRACE_INTERVAL_MISSING')

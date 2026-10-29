"""Locate one missing interval by launch correlation, without imputation."""
from collections import defaultdict

POLICY='OBSERVED_COUNT_SINGLE_SHORT_SAMPLE_V2'


def single_activity_loss(records,schedule,rows,apis):
    def require(ok,message):
        if not ok:raise ValueError('single Asys activity exception rejected: '+message)
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
    cursor=0;short=None
    for index,(item,call) in enumerate(zip(schedule,calls)):
        group=grouped[call['correlation_id']]
        count=item['kernels']
        if len(group)!=count:
            require(short is None and len(group)==count-1,'multiple missing or extra activities')
            require(count==records[item['component']]['calls_per_graph'] and count>=16,
                    'invalid repeat count')
            short=dict(component=item['component'],round=item['round'],sample=item['sample'],
                sample_index=index,expected_count=count,observed_count=len(group),
                launch_correlation_id=call['correlation_id'])
        require(rows[cursor:cursor+len(group)]==group,'interleaved graph activity')
        cursor+=len(group)
        # The normal parser below checks kernel identity and positive durations.
    require(short is not None,'no uniquely identified short sample')
    return dict(short,policy=POLICY,expected_total=expected,observed_total=len(rows),
        denominator='OBSERVED_GPU_ACTIVITIES',imputed_intervals=0,
        interpretation='EXECUTION_COUNT_UNCHANGED_ONE_TRACE_INTERVAL_MISSING')

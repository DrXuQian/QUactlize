"""GPU-activity timing of isolated production stages; never an event fallback."""
import copy
import ctypes as C
import statistics

from dev.strategy.isolated import checked
from dev.strategy.components import statistics_checked
from dev.strategy.smallm_asys import METHOD,mean_decision,validate
from dev.prefill_asys.trace import activities
from quactlize.runtime.compiler import sha


class Collector:
    def __init__(self):self.schedule=[];self.rt=None

    def start(self,rt):
        if self.rt is not None:raise ValueError('capture already active')
        rt.sync()
        for name in ('ProfilerStart','ProfilerStop'):
            fn=getattr(rt.lib,'hggc'+name);fn.argtypes=[];fn.restype=C.c_int
            setattr(self,name,fn)
        checked(self.ProfilerStart(),'Asys profiler start')
        self.rt=rt

    def sample(self,cid,graph,round_no,index):
        if self.rt is None:raise ValueError('sample outside profiler range')
        graph.rt.sync();graph.run();graph.rt.sync()
        self.schedule.append(dict(component=cid,round=round_no,sample=index,kernels=graph.count))
        return None  # No guessed/event/host duration is published by the child.

    def stop(self):
        if self.rt is not None:
            try:
                try:self.rt.sync()
                finally:checked(self.ProfilerStop(),'Asys profiler stop')
            finally:self.rt=None


def parse_rows(captured,rows,provenance):
    if captured.get('status')!='CAPTURED_NOT_TIMED' or captured.get('timing_method')!=METHOD:
        raise ValueError('not an untimed Asys capture receipt')
    result=copy.deepcopy(captured);records=result['component_records']
    rounds,samples=result['protocol']['rounds'],result['protocol']['samples']
    values={k:[[None]*samples for _ in range(rounds)] for k in records}
    if any(a['end']>b['start'] for a,b in zip(rows,rows[1:])):
        raise ValueError('isolated kernel activities overlap or are unordered')
    cursor=0;sequences={k:[] for k in records}
    for item in captured['trace_schedule']:
        cid=item['component'];r=item['round'];s=item['sample'];count=item['kernels']
        record=records[cid];node=record['identity']['node']
        if (count!=record['calls_per_graph'] or count<16 or
                not 0<=r<rounds or not 0<=s<samples or values[cid][r][s] is not None):
            raise ValueError('duplicate/invalid Asys sample schedule')
        block=rows[cursor:cursor+count];cursor+=count
        if len(block)!=count:raise ValueError('Asys kernel activity coverage incomplete')
        for actual in block:
            if (actual['kind']!='kernel' or actual['end']<=actual['start'] or
                    any(actual.get(f)!=node[f] for f in ('symbol','grid','block','shared_bytes'))):
                raise ValueError('Asys executed kernel/launch differs from captured component: '+cid)
        if any(a['end']>b['start'] for a,b in zip(block,block[1:])):
            raise ValueError('isolated kernel activities overlap')
        durations=[(v['end']-v['start'])/1000 for v in block]
        values[cid][r][s]=statistics.fmean(durations)
        sequences[cid].append(dict(round=r,sample=s,first_activity=cursor-count,count=count,
                                   mean_us=values[cid][r][s],min_us=min(durations),max_us=max(durations),
                                   durations_ns=[v['end']-v['start'] for v in block]))
    if cursor!=len(rows):raise ValueError('unaccounted Asys GPU activity; no timing admitted')
    for cid,record in records.items():
        if any(x is None for row in values[cid] for x in row):raise ValueError('missing component sample')
        record.update(statistics_checked(values[cid]),timing_method=METHOD,
            mean_us=statistics.fmean(x for row in values[cid] for x in row),
            latency_provenance=dict(provenance,timing_method=METHOD),trace_slices=sequences[cid])
    costs={aid:[[sum(records[k]['samples_us'][r][i] for k in keys) for i in range(samples)]
                for r in range(rounds)] for aid,keys in result['arm_components'].items()}
    result.update(status='PASS',decision=mean_decision(costs,result['roles']),
                  latency_provenance=dict(provenance,timing_method=METHOD))
    validate(result)
    return result


def parse(captured,database,report,provenance):
    # GraphKernelNodeParams.sharedMemBytes is dynamic, not static+dynamic.
    return parse_rows(captured,activities(database,launch_shared=True),dict(provenance,
        trace_sha256=sha(report),sqlite_sha256=sha(database)))

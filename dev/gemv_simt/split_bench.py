"""Independent partition oracle and one-component-at-a-time Split-K timing."""
from contextlib import ExitStack
import ctypes as C
import numpy as np

from dev.bf16_compute.fixture import compare
from dev.gemv_simt.general_bench import Kernel as BaseKernel, numerical as whole_numerical
from dev.gemv_simt.general_fixture import Weights as BaseWeights
from dev.gemv_simt.general_plan import configs
from dev.smallm_closure.bench import Buffer
from dev.strategy.components import statistics_checked
from dev.strategy.indexed import RuntimeView
from dev.strategy.isolated import Captured
from quactlize.execution.native import SimtCallV2, Arrangement
from quactlize.runtime.native import checked
from tools.run_q6_metadata import raw_equal


class Weights(BaseWeights):
    def __init__(self,q,n,k,experts):
        self.group=32 if q in (12,13) else 16
        self.spots=np.unique(np.linspace(0,n-1,16,dtype=int))
        shape=(min(experts,17),4,k//self.group,len(self.spots))
        self.parts=np.empty(shape,dtype='f8');self.abs_parts=np.empty_like(self.parts)
        def observe(expert,first,gold,categories):
            idx=np.flatnonzero((self.spots>=first)&(self.spots<first+gold.shape[0]))
            if not len(idx):return
            block=gold[self.spots[idx]-first].astype('f8')
            for cat in range(4):
                values=np.where(categories[None,:]==cat,block,0).reshape(len(idx),k//self.group,self.group)
                self.parts[expert,cat][:,idx]=values.sum(axis=2).T
                self.abs_parts[expert,cat][:,idx]=np.abs(values).sum(axis=2).T
        super().__init__(q,n,k,experts,observe=observe)

    def partial_oracle(self,b,recipe,split):
        workers=recipe['warps']*32//recipe['columns']
        owner=(np.arange(self.k//self.group)//workers)%split
        category_positions=[np.flatnonzero(self.categories==c)[0] for c in range(4)]
        if b.w.operation=='indexed':
            row=np.arange(b.rows)//b.w.topk*b.w.channels+np.arange(b.rows)%b.w.topk%b.w.channels
            a=b.data['a'][row]
        else:a=b.data['a']
        coeff=a[:,category_positions].astype('f8')
        gold=np.empty((b.rows,split,len(self.spots)),dtype='f8');denom=np.empty_like(gold)
        for row,expert in enumerate(b.data['owners']):
            e=int(expert)%min(self.experts,17)
            for s in range(split):
                gold[row,s]=coeff[row]@self.parts[e][:,owner==s,:].sum(axis=1)
                denom[row,s]=np.abs(coeff[row])@self.abs_parts[e][:,owner==s,:].sum(axis=1)
        return gold,denom


class Kernel(BaseKernel):
    def __init__(self,b,library,split):
        super().__init__(b,library,entry='split_reader_run')
        self.split=split
        self.fn.argtypes=[C.POINTER(SimtCallV2),C.POINTER(Arrangement),C.c_int,C.c_int,C.c_int,C.c_int]
        if not hasattr(b,'partial'):
            b.partial=Buffer(b.r,b.rows*b.w.n*8*4)
        for d in self.calls:
            d.call.workspace=b.partial.ptr if split>1 else None
            d.call.workspace_bytes=b.partial.size if split>1 else 0

    def launch(self,config,arm,copy=0,stage=0):
        return self.fn(C.byref(self.calls[copy]),C.byref(self.b.arr),config,self.split,arm,stage)

    def partial(self):
        b=self.b;b.sdk.synchronize(b.r.stream)
        count=b.rows*self.split*b.w.n
        values=b.partial.read((-1,))
        if np.any(values[count:].view('<u4')!=0xa5a5a5a5):
            raise ValueError('producer wrote beyond the declared partial region')
        return values[:count].reshape(b.rows,self.split,b.w.n).copy()


def numerical(core,config):
    whole=whole_numerical(core,config)
    if core.split==1:return dict(whole=whole,partials='NOT_APPLICABLE_S1')
    b=core.b;recipe=configs(b.w.qtype)[config];checks=[]
    for repeat in (0,1,2):
        b.update(repeat,large=repeat==2 and b.w.compute=='bf16')
        gold,denom=b.ring.w.partial_oracle(b,recipe,core.split)
        for copy in sorted({0,b.ring.copies-1}):
            values=[]
            for arm in (0,1):
                b.partial.poison();b.out.poison()
                checked(core.launch(config,arm,copy),'partial numeric full call');b.check()
                value=core.partial();checks.append(compare(value[:,:,b.ring.w.spots],gold,denom));values.append(value)
            raw_equal(*values)
    b.update(0);b.partial.poison();b.out.poison()
    # A reducer by itself cannot supply valid GEMM output.
    checked(core.launch(config,1,stage=2),'omitted Split-K producer')
    try:b.check()
    except ValueError as e:
        if not str(e).startswith('typed oracle mismatch:'):raise
    else:raise ValueError('reducer accepted omitted producer')
    # Conversely, a producer by itself must not fill the final output.
    b.out.poison();checked(core.launch(config,1,stage=1),'omitted reducer')
    try:b.check()
    except ValueError as e:
        if not str(e).startswith('typed oracle mismatch:'):raise
    else:raise ValueError('producer accepted omitted reducer')
    return dict(whole=whole,partials=dict(checks=checks,raw_equal=True,
                spots=b.ring.w.spots.tolist(),scope='ALL_PARTITIONS_16_N_SPOTS_OFFICIAL_GGUF',
                omitted_producer='DETECTED',omitted_reducer='DETECTED'))


def timing(core,config):
    b=core.b;rt=RuntimeView(b);captures=[];producers=[];nodes=[]
    recipe=configs(b.w.qtype)[config];tile=recipe['columns']*recipe['values']
    with ExitStack() as cleanup:
        for arm in (0,1):
            b.partial.poison()
            for copy in range(b.ring.copies):checked(core.launch(config,arm,copy),'untimed full-call warmup')
            b.check()
            cap=Captured(rt,[lambda i=i,a=arm:core.launch(config,a,i) for i in range(b.ring.copies)])
            cleanup.callback(cap.close);captures.append(cap)
            expected=1 if core.split==1 else 2
            if len(cap.groups)!=expected:raise ValueError('unexpected producer/reducer inventory')
            row=[cap.descriptions[g[0]] for g in cap.groups]
            grid=[b.rows*core.split*(b.w.n//tile),1,1] if arm==0 else [b.w.n//tile,b.rows,core.split]
            if row[0]['grid']!=grid or row[0]['block']!=[recipe['warps']*32,1,1]:
                raise ValueError('wrong Split-K producer geometry')
            if arm==1 and f'split_candidate_{config}_{b.p["mode"]}(' not in row[0]['name']:
                raise ValueError('candidate did not execute')
            nodes.append(row)
            graph=cap.isolate(0,weight_ring=True,repeats=16)
            cleanup.callback(graph.close);producers.append(graph)
        reducer=None
        if core.split>1:
            if nodes[0][1]!=nodes[1][1]:raise ValueError('supposed shared reducer differs')
            for cap in captures:
                param=cap.params[cap.groups[1][0]]
                if param.func!=captures[0].params[captures[0].groups[1][0]].func:
                    raise ValueError('different reducer function identity')
            reducer=captures[0].isolate(1,weight_ring=False,repeats=16)
            cleanup.callback(reducer.close)
        b.update(1);outputs=[];proofs=[];partial_checks=[]
        for producer in producers:
            b.out.poison();b.partial.poison();producer.run();rt.sync()
            if reducer:
                gold,denom=b.ring.w.partial_oracle(b,recipe,core.split)
                partial_checks.append(compare(core.partial()[:,:,b.ring.w.spots],gold,denom))
                reducer.run();rt.sync()
            proofs.append(b.check());outputs.append(b.out.read((b.rows,b.w.n)).copy())
        raw_equal(*outputs);b.update(0)
        samples=[[],[]];reduction=[]
        for rnd in range(6):
            for arm in ((0,1) if rnd%2==0 else (1,0)):
                samples[arm].append([producers[arm].sample() for _ in range(5)])
                if reducer:reducer.run();rt.sync()
                b.check()
            if reducer:
                reduction.append([reducer.sample() for _ in range(5)]);b.check()
        stats=[statistics_checked(v) for v in samples]
        reduced=statistics_checked(reduction) if reducer else None
        costs=[s['median_us']+(reduced['median_us'] if reduced else 0) for s in stats]
        return dict(status='PASS',producer=stats,reducer=reduced,component_sum_us=costs,
                    delta_pct=100*(costs[1]/costs[0]-1),nodes=nodes,graph_checks=proofs,
                    partial_checks=partial_checks,raw_equal=True,
                    calls_per_graph=[g.count for g in producers],
                    reducer_calls_per_graph=reducer.count if reducer else 0,
                    reducer_measured_once_for_identical_implementation=bool(reducer),
                    timed_kernel_types_per_interval=1,host_launch_cost_added=False,
                    scope='SUM_OF_ISOLATED_COMPONENTS_NOT_MODEL_OR_MEASURED_E2E')


def profile(core,config,arm):
    # ACU --launch-count 1 collects the producer only. The subsequent reducer
    # is still executed for correctness; neither a cold ring nor graphs enter
    # this path. Prior numerical and isolated reducer timing are mandatory.
    b=core.b;checks=[]
    for repeat in (0,1,0):
        b.update(repeat);checked(core.launch(config,arm),'profile warmup');checks.append(b.check())
    start,stop=b.sdk.lib.hggcProfilerStart,b.sdk.lib.hggcProfilerStop
    for fn in (start,stop):fn.argtypes,fn.restype=[],C.c_int
    print('SPLIT_PROFILE_BEGIN weight_copies=1 graphs=0 full_calls=1 collected_producers=1',flush=True)
    checked(start(),'ACU start')
    try:checked(core.launch(config,arm),'single direct split call');b.sdk.synchronize(b.r.stream)
    finally:checked(stop(),'ACU stop')
    checks.append(b.check());b.partial.guard()
    print('SPLIT_PROFILE_END',flush=True)
    return dict(status='PASS',checks=checks,weight_copies=1,graphs=0,full_calls=1,
                collected_producers=1,unprofiled_reducers=int(core.split>1))

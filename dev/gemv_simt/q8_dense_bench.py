"""Reuse the independent oracle; vary only the selected Q8 producer entry."""
from contextlib import ExitStack
from dev.gemv_simt.general_bench import Kernel as GeneralKernel
from dev.gemv_simt.q8_dense_source import geometry,symbol_name
from dev.strategy.components import statistics_checked
from dev.strategy.indexed import RuntimeView
from dev.strategy.isolated import Captured
from quactlize.runtime.native import checked
from tools.run_q6_metadata import raw_equal


class Kernel(GeneralKernel):
    def __init__(self,b,library,candidate):
        super().__init__(b,library)
        self.candidate=candidate

    def launch(self,config,arm,copy=0):
        return super().launch(config,0 if arm==0 else self.candidate,copy)


def timing(core,config,recipe,*,geometry_fn=geometry,symbol_fn=symbol_name):
    b=core.b;rt=RuntimeView(b);nodes=[];graphs=[]
    with ExitStack() as cleanup:
        for arm in (0,1):
            actual=0 if arm==0 else core.candidate
            for copy in range(b.ring.copies):checked(core.launch(config,arm,copy),'excluded Q8 warmup')
            b.check()
            cap=Captured(rt,[lambda i=i,a=arm:core.launch(config,a,i) for i in range(b.ring.copies)])
            cleanup.callback(cap.close)
            grid,block=geometry_fn(b.w,recipe,actual)
            if (len(cap.groups)!=1 or symbol_fn(config,actual)+'(' not in cap.descriptions[0]['name'] or
                    cap.descriptions[0]['grid']!=grid or cap.descriptions[0]['block']!=block):
                raise ValueError('Q8 ablation captured inventory differs')
            nodes.append(cap.descriptions[0])
            graph=cap.isolate(0,weight_ring=True,repeats=16);cleanup.callback(graph.close);graphs.append(graph)
        b.update(1);outputs=[];proofs=[]
        for g in graphs:
            b.out.poison();g.run();rt.sync();proofs.append(b.check());outputs.append(b.out.read((b.rows,b.w.n)).copy())
        raw_equal(*outputs);b.update(0)
        samples=[[],[]]
        for rnd in range(6):
            for arm in ((0,1) if rnd%2==0 else (1,0)):
                samples[arm].append([graphs[arm].sample() for _ in range(5)]);b.check()
        stats=[statistics_checked(s) for s in samples]
        weight_bytes=sum(v.nbytes for v in b.ring.w.planes.values())
        return dict(status='PASS',nodes=nodes,timings=stats,graph_checks=proofs,raw_equal=True,
                    calls_per_graph=[g.count for g in graphs],compulsory_weight_bytes=weight_bytes,
                    modeled_weight_mbu_pct=[weight_bytes/s['median_us']/27000 for s in stats],
                    assumed_peak_gbps=2700,mbu_scope='WEIGHT_BYTES_PER_EVENT_TIME_NOT_ACU_DRAM_BANDWIDTH',
                    delta_pct=100*(stats[1]['median_us']/stats[0]['median_us']-1),
                    scope='Q8_DENSE_S1_ROTATING_EVENT_SCREEN_NOT_CLOCK_CONFIRMED_SELECTION')

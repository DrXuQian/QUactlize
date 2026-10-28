"""Fresh large-M probe pieces: GPU route maps + unchanged general TC/provider.

No decode-only endpoint and no CPU-prepared map in the measured pipeline.
The native caller adapter owns the routing kernel; these are experiment-only
wrappers. Provider/JIT/capture support must pass on Box before a full sweep.
"""
import ctypes as C

import numpy as np

from dev.strategy.asys_components import Collector
from dev.strategy.cold_decode import replay_extracted
from dev.strategy.components import component_key, stage_role
from dev.strategy.prefill_fixture import Context
from dev.strategy.prefill_full import Full
from dev.strategy.prefill_native import TensorCore


def check_maps(w, route, source, destination, offsets):
    """Accept any valid within-expert permutation, not just CPU stable sort."""
    rows = w.m*w.topk
    if (source.shape != (rows,) or destination.shape != (rows,) or
            offsets.shape != (w.experts+1,) or not np.array_equal(offsets, route['offsets']) or
            not np.array_equal(np.sort(destination), np.arange(rows))):
        raise ValueError('GPU routing bounds/permutation differ')
    if (not np.array_equal(route['owners'][destination], route['sorted_owners']) or
            not np.array_equal(source, route['source'][destination])):
        raise ValueError('GPU routing expert/source mapping differs')


class RoutedContext(Context):
    def __init__(self, sdk, ring, w, helpers, native_adapter):
        if (w.operation, w.experts, w.topk) != ('indexed', 256, 8) or not 8 < w.m <= 5120:
            raise ValueError('MoE prefill routing probe requires E256/top8 and M9..5120')
        self.ids = None
        self.adapter = C.CDLL(str(native_adapter), mode=C.RTLD_LOCAL)
        self.route_fn = self.adapter.nc_route_launch
        self.route_fn.argtypes, self.route_fn.restype = [C.c_int]*4+[C.c_void_p]*5, C.c_int
        self.route_error = self.adapter.nc_error
        self.route_error.argtypes, self.route_error.restype = [], C.c_char_p
        super().__init__(sdk, ring, w, helpers)
        from dev.native_components.kernel_nodes import KernelNodes
        self.rt.kernel_nodes = KernelNodes(self.adapter)
        self.ids = self.allocate(self.rows*4)
        self.update(0)

    def launch_route(self):
        w = self.w
        if self.route_fn(w.m,w.experts,w.topk,w.channels,self.ids.ptr,self.source.ptr,
                         self.destination.ptr,self.offsets.ptr,self.r.stream):
            raise ValueError('native GPU row maps: '+self.route_error().decode(errors='replace'))
        return 0

    def validate_routes(self):
        self.rt.sync()
        arrays = [np.frombuffer(self.sdk.download(buf.ptr,buf.size),dtype='<i4')
                  for buf in (self.source,self.destination,self.offsets)]
        check_maps(self.w,self.route,*arrays)
        for buf in (self.ids,self.source,self.destination,self.offsets):
            buf.guard()

    def update(self, repeat, *, zero=False, amplitude=1):
        super().update(repeat,zero=zero,amplitude=amplitude)
        if self.ids is not None:
            self.ids.upload(self.route['owners'])
            # Poison the oracle's host-uploaded maps. A no-op GPU router must
            # fail here rather than borrowing their correct initial contents.
            for buf in (self.source,self.destination,self.offsets):
                buf.poison()
            self.launch_route()
            self.validate_routes()

    def check(self, *, zero=False):
        self.validate_routes()
        return super().check(zero=zero)


class RoutedTC(TensorCore):
    def launch(self, copy=0):
        self.b.launch_route()
        return super().launch(copy)


class RoutedFull(Full):
    def launch(self, copy=0):
        self.b.launch_route()
        return super().launch(copy)


class AsysStages:
    """Isolated component collector for the existing correctness fixtures.

    No event timer/cache fallback. Results remain untimed until the exact raw
    Asys activity inventory is parsed; publication is the pilot parent's job.
    """
    def __init__(self, *, rounds=6, samples=5, known=None):
        if (rounds,samples) not in ((4,3),(6,5)):
            raise ValueError('prefill Asys requires screen4x3 or confirm6x5')
        self.collector = Collector()
        self.rounds,self.samples=rounds,samples
        self.records = dict(known or {})

    def collect(self, capture, w, recipe, authority, source, proof, *, active=None, sf=False):
        split = dict(recipe.parameters).get('split',1) if recipe else 1
        nodes = capture.descriptions[:len(capture.groups)]
        roles = [stage_role(n['name'],split) for n in nodes]
        if ('unclassified' in roles or (recipe is None and roles != ['expansion']) or
                (recipe is not None and ('producer' not in roles or
                 not any('mm_ids_helper' in n['name'] for n in nodes))) or
                (split>1 and 'reducer' not in roles)):
            raise ValueError('MoE prefill captured inventory incomplete: '+repr(list(zip(roles,nodes))))
        replay_extracted(capture,proof)
        graphs = []
        result = []
        new = {}
        try:
            for i,(node,role) in enumerate(zip(nodes,roles)):
                rotating = role in ('producer','expansion')
                cache = ('ROTATING_ACTIVE_READ_SET_AT_LEAST_2P25_L2' if rotating else
                         'REPEATED_RESIDENT_SCRATCH')
                node = dict(node,split=split)
                if sf:
                    node['expert_domain'] = 'ALL'
                elif recipe is None and active is not None:
                    node['active_expert_ids'] = list(map(int,active))
                cid,identity = component_key(w,recipe,node,authority=authority,source=source,cache=cache)
                result.append(dict(id=cid,role=role,median_us=None,spread_pct=None))
                if cid in self.records or cid in new:
                    old=(self.records if cid in self.records else new)[cid]
                    if old['identity']!=identity:raise ValueError('shared prefill component identity changed')
                    continue
                graph = capture.isolate(i,weight_ring=rotating,repeats=16)
                graphs.append((cid,graph))
                new[cid] = dict(identity=identity,component_id=cid,kind=role,
                    calls_per_graph=graph.count,first_use_excluded=True,timed_kernel_types=1,
                    host_launch_cost_added=False,cache=cache)
            # Graph upload, warmup, oracle/readback and allocation precede the
            # only measured range. Every graph repeats ONE kernel type.
            if graphs:
                self.collector.start(capture.rt)
                for r in range(self.rounds):
                    for cid,graph in (graphs if r%2==0 else reversed(graphs)):
                        for s in range(self.samples):
                            self.collector.sample(cid,graph,r,s)
        finally:
            self.collector.stop()
            for _,graph in reversed(graphs):
                graph.close()
        replay_extracted(capture,proof)
        self.records.update(new)
        return result, []  # no event/component cache publication

    def stages(self,capture,b,recipe,authority,source,output,proof,*,helper=False):
        return self.collect(capture,b.w,recipe,authority,source,proof,
                            active=b.route['active'] if helper else None)

    def sf(self,rt,capture,w,recipe,authority,source,output,proof,*,separate_expansion=False):
        if recipe is not None or not separate_expansion:
            raise ValueError('SF requires its own all-expert metadata cold ring')
        return self.collect(capture,w,None,authority,source,proof,sf=True)

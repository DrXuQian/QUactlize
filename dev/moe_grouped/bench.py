"""Real TC -> helper composition. No new GEMM selection or model claim."""
import ctypes as C
from contextlib import ExitStack
import json
import statistics

import numpy as np

from dev.bf16_compute.fixture import Weights,compare
from dev.moe_route.bench import Runtime,Buffer,Graph,bind
from dev.moe_scaffold.bench import Library,Fixture,start_clock
from dev.moe_scaffold.protocol import Case,Value,Plan,decode
from dev.moe_route.protocol import Call as RouteCall,Config,Sizes
from dev.strategy.resources import NativeProbe
from dev.strategy.sf import metadata_bits
from dev.strategy.refresh_clock import observe
from quactlize.decode.native_compute import GroupedDeviceCall,GroupedMetadataCall
from quactlize.runtime.native import Call,Recipe,Resources,checked


class Result(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32),('data',C.c_void_p)]+[
        (n,C.c_int32) for n in ('dtype','partial_count','compute_type','reserved')]+[
        (n,C.c_int64) for n in ('row_stride','partial_stride')]


def value(result):
    if (result.version!=1 or result.size!=C.sizeof(Result) or result.reserved or not result.data or
        result.compute_type not in (0,1) or result.partial_count not in (1,2,4,8) or result.row_stride<=0):
        raise ValueError('grouped output ABI differs')
    partial=result.partial_count>1
    if (partial and (result.dtype!=2 or result.partial_stride<result.row_stride) or
        not partial and (result.dtype!=result.compute_type or result.partial_stride)):
        raise ValueError('grouped output state differs')
    dtype=(1,2,0)[result.dtype]
    return Value(result.data,dtype,0,int(partial),result.partial_count,result.row_stride,
                 result.partial_stride,2 if result.compute_type else 1,0)


class DeviceWeights:
    def __init__(self,rt,q,n,k,e,seed,route):
        self.rt,self.buffers=rt,[]
        self.host=Weights(q,n,k,e,seed=seed,pool_size=3)
        self.planes={}
        try:
            data={name:self.host.planes[name] for name in ('low','high','units')}
            if route=='sf-grouped' and q!=8:
                data['units'],data['zero']=metadata_bits(data['units'],q,n,k,e,True)
            for name,v in data.items():
                b=Buffer(rt,v.nbytes);self.buffers.append(b);b.upload(v.tobytes())
                self.planes[name]=b.ptr if v.size else None
        except BaseException:self.close();raise

    def close(self):
        self.rt.sync()
        for b in self.buffers:b.close()
        self.buffers=[]


class Producer:
    def __init__(self,rt,spec,record,weights,m,split,a,output,offsets,shared=None,algorithm=0):
        self.rt,self.buffers,self.old,self.dynamic=rt,[],C.c_void_p(),C.c_void_p()
        self.owns=shared is None
        p=NativeProbe(spec,record);self.probe=p
        self.prepare=p._bind('grouped_prepare_v4',[C.POINTER(GroupedMetadataCall),C.POINTER(Recipe),C.POINTER(C.c_void_p)])
        self.prepare_dynamic=p._bind('grouped_prepare_dynamic_v1',self.prepare.argtypes)
        self.run=p._bind('grouped_run_dynamic_v1',[C.c_void_p,C.POINTER(GroupedMetadataCall),C.c_int])
        self.describe=p._bind('grouped_output_dynamic_v1',self.run.argtypes+[C.POINTER(Result)])
        self.stage=p._bind('test_grouped_dynamic_stage',self.run.argtypes+[C.c_int])
        self.legacy=p._bind('run_v1',[C.c_void_p,C.c_void_p])
        self.destroy=p._bind('destroy_v1',[C.c_void_p],None)
        h=weights.host
        c=Call(version=1,size=C.sizeof(Call),m=m*8,n=h.n,k=h.k,experts=h.experts,
            group_size=32 if h.q in (8,12,13) else 16,device=p.device['ordinal'],
            compute_units=p.device['compute_units'],mapping_id=p.mapping,stream=rt.stream.value)
        self.call=GroupedMetadataCall(4,C.sizeof(GroupedMetadataCall),
            GroupedDeviceCall(2,C.sizeof(GroupedDeviceCall),c,m,0),1,int(h.q!=8))
        self.recipe=Recipe(1,C.sizeof(Recipe),algorithm,split,1 if algorithm else 0)
        query=p._bind('grouped_query_v4',[C.POINTER(GroupedMetadataCall),C.POINTER(Recipe),C.POINTER(Resources)])
        resources=Resources();checked(query(C.byref(self.call),C.byref(self.recipe),C.byref(resources)),'grouped resource')
        try:
            if shared is None:
                checked(self.prepare_dynamic(C.byref(self.call),C.byref(self.recipe),C.byref(self.dynamic)),'grouped dynamic prepare')
            else:self.dynamic=shared.dynamic
            scratch=Buffer(rt,resources.workspace_bytes);self.buffers.append(scratch)
            c=self.call.device_call.call
            c.a,c.output,c.offsets_device=a,output,offsets
            c.low,c.high,c.metadata=(weights.planes.get(x) for x in ('low','high','units'))
            c.zero=weights.planes.get('zero');c.workspace,c.workspace_bytes=scratch.ptr,scratch.size
            checked(self.prepare(C.byref(self.call),C.byref(self.recipe),C.byref(self.old)),'bound control prepare')
            if self.legacy(self.dynamic,rt.stream)!=2:raise ValueError('dynamic handle accepted legacy run')
            self.weights=weights;self.resources=resources
        except BaseException:self.close();raise

    def output(self,mode):
        r=Result();checked(self.describe(self.dynamic,C.byref(self.call),mode,C.byref(r)),'producer output contract')
        return value(r)

    def launch(self,mode,legacy=False):
        return self.legacy(self.old,self.rt.stream) if legacy else self.run(self.dynamic,C.byref(self.call),mode)

    def numerical_spots(self,offsets):
        c=self.call.device_call.call;w=self.weights.host
        rows=np.unique(np.linspace(0,c.m-1,min(8,c.m),dtype=int))
        cols=np.unique(np.linspace(0,c.n-1,min(16,c.n),dtype=int))
        errors=[]
        for row in rows:
            e=int(np.searchsorted(offsets,int(row),side='right')-1)
            a=decode(np.frombuffer(self.rt.download(c.a+int(row)*c.k*2,c.k*2),'<u2'),2).astype('f8')
            b=w.weight(e)[cols].astype('f8')
            got=decode(np.frombuffer(self.rt.download(c.output+int(row)*c.n*2,c.n*2),'<u2'),2)[cols]
            gold=b@a;denom=np.abs(b)@np.abs(a)
            errors.append(compare(got,gold,denom)['relative_l1_error'])
        return dict(cells=len(rows)*len(cols),max_l1_error=max(errors),oracle='SAMPLED_OFFICIAL_GGUF_FP64_DOT')

    def close(self):
        self.rt.sync()
        if self.old.value:self.destroy(self.old);self.old=C.c_void_p()
        if self.dynamic.value and self.owns:self.destroy(self.dynamic);self.dynamic=C.c_void_p()
        for b in self.buffers:b.close()
        self.buffers=[]


class Chain:
    def __init__(self,rt,lib,job,records,weights,shared=None):
        self.rt,self.lib,self.job=rt,lib,job
        k,m=job['k'],job['m']
        self.b=Fixture(rt,lib,Case(m,columns=k,hidden=512,output=k,padding=0,merged=0,profile=job['profile']))
        self.producers=[]
        try:
            for i,name in enumerate(('gate','up','down')):
                q=13 if i==2 else 12;spec,record=records[str(q)]
                self.producers.append(Producer(rt,spec,record,weights[i],m,
                    job['down_split'] if i==2 else job['gate_split'],
                    self.b.t.output if i==2 else self.b.p.output,self.b.ptr(name),self.b.p.route.offsets,
                    shared.producers[i] if shared else None))
            self.bind(0)
        except BaseException:self.close();raise

    def bind(self,mode):
        self.mode=mode;b=self.b
        b.t.gate=self.producers[0].output(mode);b.t.up=self.producers[1].output(mode)
        b.f.down=self.producers[2].output(mode)
        checked(self.lib.finish_select(C.byref(b.f),0,C.byref(b.finish_plan)),'real finish selection')
        if b.finish_plan.workspace_bytes>b.buffers['selected_scratch'].size:raise ValueError('chain finish scratch short')

    def launch(self,legacy=False,unfused=False):
        b=self.b
        actions=([lambda s=s:self.lib.prepare_unfused(C.byref(b.p),s) for s in range(5)]
                 if unfused else [b.selected_prepare])
        actions += [lambda p=p:p.launch(self.mode,legacy) for p in self.producers[:2]]
        actions += [x[0] for x in b.transition_control()] if unfused else [lambda:self.lib.transition(C.byref(b.t))]
        actions += [lambda:self.producers[2].launch(self.mode,legacy)]
        actions += [x[0] for x in b.finish_control()] if unfused else [b.selected_finish]
        for fn in actions:
            rc=fn()
            if rc:return rc
        return 0

    def snapshots(self):
        self.rt.sync()
        buffers=self.b.buffers
        values=[buffers[n].read() for n in ('activation','finish')]
        if not np.isfinite(np.frombuffer(values[1],'<f4')).all():raise ValueError('nonfinite real-chain output')
        return values

    def update(self,seed):
        # Final descriptors have no partial stride; partial descriptors point
        # into producer-owned scratch. Neither is a synthetic fixture plane.
        self.b.update_weights(self.b.update_inputs(seed))

    def check(self,seed):
        self.update(seed);self.bind(0)
        checked(self.launch(legacy=True,unfused=True),'legacy TC plus decomposed helpers')
        self.b.prepare_check();expected=self.snapshots()
        self.b.reset_prepare()
        checked(self.b.combined_prepare(),'combined prepare with real producer bindings')
        self.b.prepare_check()
        spots=[p.numerical_spots(self.b.maps[2]) for p in self.producers]
        projected=[self.b.buffers[n].read() for n in ('gate','up','down')]
        for n in ('gate','up','down','activation','finish'):self.b.buffers[n].poison()
        checked(self.launch(),'dynamic complete chain')
        if expected!=self.snapshots() or projected!=[self.b.buffers[n].read() for n in ('gate','up','down')]:
            raise ValueError('dynamic complete differs from bound control')
        self.bind(1)
        checked(self.launch(),'dynamic producers plus fused helpers')
        if expected!=self.snapshots():raise ValueError('producer/helper chain differs from ordered complete control')
        for p in self.producers:
            bad=GroupedMetadataCall.from_buffer_copy(p.call);bad.device_call.call.workspace_bytes-=1
            if p.run(p.dynamic,C.byref(bad),1)!=2:raise ValueError('short scratch accepted')
            bad=GroupedMetadataCall.from_buffer_copy(p.call);bad.device_call.call.output=bad.device_call.call.a
            if p.run(p.dynamic,C.byref(bad),1)!=2:raise ValueError('live output alias accepted')
        if expected!=self.snapshots():raise ValueError('rejected producer wrote output')
        for p in self.producers:
            for buf in p.buffers:buf.guard()
        for buf in self.b.buffers.values():buf.guard()
        return spots

    def components(self,mode):
        self.bind(mode);b=self.b;all_groups=b.component_groups()
        chosen={1:'fused',2:'hybrid',3:'unfused',16:'fused',17:'slots4',18:'slots8',19:'unfused'}
        prepare='unfused' if mode==0 else chosen[b.prepare_plan.implementation]
        finish='unfused' if mode==0 else chosen[b.finish_plan.implementation]
        result=[]
        def helpers(group):
            for i,(launch,symbol,grid,block) in enumerate(all_groups[group]):
                result.append((group+'/'+str(i),launch,symbol))
        helpers('prepare/'+prepare)
        for i,name in enumerate(('gate','up','down')):
            if i==2:helpers('transition/'+('unfused' if mode==0 else 'fused'))
            p=self.producers[i]
            needles=['device_metadata<','build_kernel<','device_kernel','reduce_bf16_grouped']
            for stage in range(4 if mode==0 and p.recipe.split>1 else 3):
                result.append((name+'/'+str(stage),lambda p=p,s=stage:p.stage(p.dynamic,C.byref(p.call),mode,s),needles[stage]))
        helpers('finish/'+finish)
        return result

    def close(self):
        self.rt.sync()
        for p in self.producers:p.close()
        self.producers=[];self.b.close()

    def route_controls(self):
        """Same decomposed helpers with every legal optimized row-map arm."""
        b=self.b;lib=self.lib.lib;rows=b.rows
        query=bind(lib,'quactlize_ppu_moe_route_query_v1',[C.POINTER(RouteCall),C.POINTER(Config),C.POINTER(Sizes)])
        stage=bind(lib,'qzr_test_stage_v1',[C.POINTER(RouteCall),C.POINTER(Config),C.c_int])
        register=bind(lib,'qzr_test_register_v1',[C.POINTER(RouteCall)])
        ballot=bind(lib,'qzr_test_ballot_v1',[C.POINTER(RouteCall),C.c_int])
        result={}
        for algorithm,tile in ((1,512),(2,256),(2,512),(2,1024)):
            route=RouteCall.from_buffer_copy(b.p.route);config=Config(1,C.sizeof(Config),algorithm,tile)
            sizes=Sizes();checked(query(C.byref(route),C.byref(config),C.byref(sizes)),'route-control query')
            route.workspace=b.alloc(f'route-control-{algorithm}-{tile}',sizes.workspace_bytes)
            route.workspace_bytes=sizes.workspace_bytes
            name='warp' if algorithm==1 else f'tile{tile}'
            needles=['warp_compact<'] if algorithm==1 else ['histogram_rank<','prefix_chunks(','scatter_maps<']
            result[name]=[(lambda r=route,c=config,s=s:stage(C.byref(r),C.byref(c),s),needle)
                          for s,needle in enumerate(needles)]
        if rows<=256:result['register']=[(lambda:register(C.byref(b.p.route)),'warp_register<')]
        if rows<=512:
            result['ballot']=[(lambda:ballot(C.byref(b.p.route),0),'warp_ballot<')]
            result['scalar']=[(lambda:ballot(C.byref(b.p.route),1),'warp_scalar<')]
        for items in result.values():
            items.extend([(lambda:self.lib.prepare_unfused(C.byref(b.p),3),'prepare_prefix<'),
                          (lambda:self.lib.prepare_unfused(C.byref(b.p),4),'prepare_gather<')])
            for fn,_ in items:checked(fn(),'optimized route-control prepare')
            b.prepare_check()
        return result


def measure(sdk,receipt,records,jobs,smi,clock_mhz):
    output=[];schedule=[]
    with ExitStack() as stack:
        rt=Runtime(sdk);stack.callback(rt.close)
        lib=Library(receipt['candidate']);clock=start_clock(rt,lib.device,smi,clock_mhz)
        for k in sorted({j['k'] for j in jobs}):
            with ExitStack() as family:
                weights=[]
                for q,n,inner,seed in ((12,512,k,131),(12,512,k,719),(13,k,512,953)):
                    w=DeviceWeights(rt,q,n,inner,256,seed,'fq-grouped');family.callback(w.close);weights.append(w)
                for job in (j for j in jobs if j['k']==k):
                    with ExitStack() as point:
                        print('MOE_GROUPED_CHECK '+json.dumps(job),flush=True)
                        b=Chain(rt,lib,job,records,weights);point.callback(b.close)
                        proof=[]
                        for seed in (7,19):proof.append(b.check(seed))
                        b.bind(1);graph=Graph(rt,b.launch,1);point.callback(graph.close)
                        b.update(31);b.bind(0);checked(b.launch(legacy=True,unfused=True),'changed-ID control')
                        expected=b.snapshots();graph.run()
                        if b.snapshots()!=expected:raise ValueError('changed-ID chain graph differs')
                        groups={};graphs={}
                        for mode,arm in ((0,'complete-control'),(1,'producer-selected')):
                            b.bind(mode);checked(b.launch(unfused=mode==0),'stage input initialization')
                            groups[arm]=[]
                            for label,launch,needle in b.components(mode):
                                key=f'{job["id"]}/{arm}/{label}';groups[arm].append(key)
                                g=Graph(rt,launch,16);point.callback(g.close);g.run();rt.sync()
                                graphs[key]=(g,needle)
                        route_groups={}
                        for arm,items in b.route_controls().items():
                            route_groups[arm]=[]
                            for i,(launch,needle) in enumerate(items):
                                key=f'{job["id"]}/route-control/{arm}/{i}';route_groups[arm].append(key)
                                g=Graph(rt,launch,16);point.callback(g.close);g.run();rt.sync();graphs[key]=(g,needle)
                        # Full chains are measured separately from isolated stage
                        # sums; never mix the two metrics in a winner decision.
                        chains={}
                        for mode,arm in ((0,'complete-control'),(1,'producer-selected')):
                            b.bind(mode)
                            g=Graph(rt,lambda u=mode==0:b.launch(legacy=u,unfused=u),1)
                            point.callback(g.close);g.run();rt.sync();chains[arm]=g
                        checked(rt.ProfilerStart(),'chain profiler start')
                        try:
                            for r in range(4):
                                items=list(graphs.items())
                                for key,(g,needle) in items if r%2==0 else reversed(items):
                                    g.run();rt.sync();schedule.append(dict(component=key,kernel=needle,repeats=16,round=r))
                                for arm,g in (list(chains.items()) if r%2==0 else list(chains.items())[::-1]):
                                    g.run();rt.sync();schedule.append(dict(component=job['id']+'/chain/'+arm,kernel=None,repeats=1,round=r))
                        finally:rt.sync();checked(rt.ProfilerStop(),'chain profiler stop')
                        output.append(dict(job=job,components=groups,prepare_alternatives=route_groups,
                            correctness='BITWISE_COMPLETE_AND_FUSED_PASS',
                            sampled_oracle=proof,recipes=[dict(parent=p.probe.spec['parent'],split=p.recipe.split,
                                resources=dict(workspace=p.resources.workspace_bytes,occupancy=p.resources.occupancy))
                                for p in b.producers],scratch_bytes=sum(buf.size for buf in b.b.buffers.values())+
                                    sum(buf.size for p in b.producers for buf in p.buffers)))
                        print('MOE_GROUPED_POINT '+json.dumps(dict(id=job['id'],m=job['m'],k=k,status='PASS')),flush=True)
        clock['after']=observe(smi,lib.device['pci'],clock_mhz,advisory=True)
        return dict(points=output,schedule=schedule,device=lib.device,clock=clock,
            scope='REAL_TC_HELPER_COMPOSITION_EXPLICIT_RECIPES_NOT_PRODUCTION_SELECTION_OR_MODEL',
            baseline='BOUND_V4_COMPLETE_PLUS_SAME_MATH_DECOMPOSED_TILED512_HELPERS',
            pending=['BEST_ROUTE_WHOLE_CHAIN','SELECTED_SIMT_TC_FULL_CHAIN','LLAMA_MODEL_AB'],
            production_changed=False,no_regression_admitted=False)


def parse(capture,rows,apis):
    from collections import defaultdict
    grouped=defaultdict(list);samples=defaultdict(list);geometries={};observed=defaultdict(int);missing=0
    for row in rows:grouped[row['correlation_id']].append(row)
    calls=[a for a in apis if a['name'].startswith(('hggcGraphLaunch','cudaGraphLaunch'))]
    if len(calls)!=len(capture['schedule']):raise ValueError('chain trace launch count differs')
    for call,item in zip(calls,capture['schedule']):
        block=grouped[call['correlation_id']]
        if call['return_value'] or not block:raise ValueError('failed/unobserved chain call')
        key=item['component']
        if item['kernel'] is not None:
            if not 8<=len(block)<=item['repeats']:raise ValueError('isolated stage activity count differs: '+key)
            missing+=item['repeats']-len(block)
            for row in block:
                if row['kind']!='kernel' or item['kernel'] not in row['name'] or row['end']<=row['start']:
                    raise ValueError('unexpected producer/helper activity: '+repr(row))
                identity=(row['name'],row['grid'],row['block'])
                if key in geometries and geometries[key]!=identity:raise ValueError('stage launch changed: '+key)
                geometries[key]=identity
            samples[key].append(statistics.fmean((x['end']-x['start'])/1000 for x in block))
        else:
            if any(x['kind']!='kernel' or x['end']<=x['start'] for x in block):raise ValueError('non-kernel in TC chain')
            # Sum device activities, not host API time. Preserve elapsed span
            # separately so launch/scheduling gaps are visible, not imputed.
            samples[key].append(sum(x['end']-x['start'] for x in block)/1000)
            samples[key+'/span'].append((max(x['end'] for x in block)-min(x['start'] for x in block))/1000)
        observed[key]+=len(block)
    means={key:statistics.fmean(v) for key,v in samples.items()}
    for point in capture['points']:
        point['isolated_component_sum_us']={arm:sum(means[k] for k in ids) for arm,ids in point['components'].items()}
        point['chain_activity_us']={arm:means[point['job']['id']+'/chain/'+arm] for arm in point['components']}
        point['chain_span_us']={arm:means[point['job']['id']+'/chain/'+arm+'/span'] for arm in point['components']}
        if point.get('prepare_alternatives'):
            costs={arm:sum(means[k] for k in keys) for arm,keys in point['prepare_alternatives'].items()}
            best=min(costs,key=costs.get)
            old=sum(means[k] for k in point['components']['complete-control'] if '/prepare/' in k)
            point['optimized_mm_ids_control']=dict(prepare_us=costs,best=best,
                complete_component_sum_us=point['isolated_component_sum_us']['complete-control']-old+costs[best],
                scope='BEST_MEASURED_ROUTE_PLUS_SAME_DECOMPOSED_HELPERS_NOT_NATIVE_OR_WHOLE_CHAIN')
    return dict(capture,status='PASS',component_us=means,samples_us=dict(samples),
        observed_counts=dict(observed),missing_intervals=missing,imputed_intervals=0,launches=geometries,
        timing_method='ASYS_ISOLATED_STAGES_AND_SEPARATE_REAL_CHAIN',
        cache='RESIDENT_WEIGHTS_MECHANISM_AB_NOT_COLD_WINNER_ADMISSION')


def contract_gate(sdk,receipt,records):
    """Every format/split; shared recipes with independent addresses/streams."""
    passed=[]
    with ExitStack() as root:
        rt=Runtime(sdk);root.callback(rt.close);lib=Library(receipt['candidate'])
        for q in (8,10,11,12,13,14):
            spec,record=records[str(q)]
            with ExitStack() as group:
                weights=DeviceWeights(rt,q,256,2048,16,197,spec['parent']['route']);group.callback(weights.close)
                for split in (1,2,4,8):
                    for algorithm in ((0,1) if q==8 else (0,)):
                        with ExitStack() as check:
                            fixtures=[]
                            for i in range(2):
                                stream=Runtime(sdk);check.callback(stream.close)
                                a,out,offsets,tmp=(Buffer(stream,size) for size in (136*2048*2,136*256*2,17*4,136*256*4))
                                for b in (a,out,offsets,tmp):check.callback(b.close)
                                p=Producer(stream,spec,record,weights,17,split,a.ptr,out.ptr,offsets.ptr,
                                           fixtures[0][0] if fixtures else None,algorithm)
                                check.callback(p.close)
                                graph=Graph(stream,lambda p=p:p.launch(1),1);check.callback(graph.close)
                                fixtures.append((p,a,out,offsets,tmp,graph))
                            for seed in (3,13):
                                expected=[]
                                for i,(p,a,out,offsets,tmp,graph) in enumerate(fixtures):
                                    rng=np.random.default_rng(seed+71*i)
                                    from dev.moe_scaffold.protocol import encode
                                    data=rng.uniform(-2,2,(136,2048)).astype('f4');data[0,0]=243383.484
                                    a.upload(encode(data,2).tobytes())
                                    counts=np.full(16,8,np.int32);counts[(np.arange(8)+seed)%16]+=1
                                    if seed==13:counts[:]=0;counts[(np.arange(8)+i)%16]=17
                                    maps=np.r_[0,counts.cumsum()].astype('<i4');offsets.upload(maps.tobytes())
                                    checked(p.launch(0,legacy=True),'bound format control');p.rt.sync()
                                    expected.append(out.read());p.numerical_spots(maps)
                                    out.poison();checked(p.launch(0),'dynamic format complete');p.rt.sync()
                                    if out.read()!=expected[-1]:raise ValueError('format dynamic complete bits differ')
                                    out.poison();graph.run()
                                for (p,a,out,offsets,tmp,graph),want in zip(fixtures,expected):
                                    v=p.output(1)
                                    checked(lib.materialize(C.byref(v),136,256,tmp.ptr,p.rt.stream),'producer descriptor consumer')
                                    p.rt.sync()
                                    got=np.frombuffer(tmp.read(),'<f4')
                                    if got.tobytes()!=decode(np.frombuffer(want,'<u2'),2).tobytes():
                                        raise ValueError('raw partial consumer bits differ from legacy completion')
                                    for b in (a,out,offsets,tmp,*p.buffers):b.guard()
                            passed.append(dict(q=q,split=split,algorithm=algorithm,streams=2,replays=2,status='PASS'))
                            print('MOE_GROUPED_CONTRACT '+json.dumps(passed[-1]),flush=True)
    return dict(status='PASS',cases=passed,scope='ALL_FORMAT_BF16_DYNAMIC_TC_NOT_PERFORMANCE')

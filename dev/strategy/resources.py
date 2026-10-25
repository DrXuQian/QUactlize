"""Offline resource admission, separate from config selection and timing.

Query the exact compiled endpoint/schedule. Never borrow occupancy from the
ordinary kernel for a persistent kernel, or from F16 for BF16. No weights,
allocations, prepare/run or benchmarks are needed for these module queries.
"""
import ctypes as C
from dataclasses import asdict
import json
from pathlib import Path

from dev.strategy.prescreen import parent
from dev.strategy.space import grid_search, grid_variants, resolved_grid, scheduled_key
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest, ROUTES


def module_spec(w, r):
    if r.kind != 'tc':
        raise ValueError('resource probe requires a TC recipe')
    legacy = dict(r.parameters).get('legacy_adapters', False)
    spec = dict(parent=parent(r,w), compute=w.compute,
                endpoint='grouped' if w.operation!='dense' else
                         'native' if w.compute=='f16' and (w.m>8 or legacy) else 'typed')
    if legacy:
        if w.operation!='dense' or w.compute!='f16' or r.route!='fq':
            raise ValueError('legacy flat adapters require dense F16 FQ')
        spec['legacy_adapters'] = True
    return spec


def probe_spec(w, r, algorithm):
    if w.operation=='grouped' or algorithm not in (0,1):
        raise ValueError('resource probe requires dense/indexed and an explicit schedule')
    p=dict(r.parameters)
    spec=dict(module=module_spec(w,r), m=w.m*w.topk, n=w.n, k=w.k, experts=w.experts,
              max_rows=w.m if w.operation=='indexed' else 0, mapping=w.mapping,
              group_size=32 if w.qtype in (8,12,13) else 16,
              algorithm=algorithm, split=p['split'])
    return json.loads(json.dumps(spec))


def inventory(plan):
    modules,queries={},{}
    for point in plan['points']:
        w=Workload(**point['workload'])
        pools=[('rule',point['candidates']),('control',point.get('champion_controls',[]))]
        if point.get('original_recipe'):pools.append(('original',[point['original_recipe']]))
        for role,rows in pools:
            for value in rows:
                r=Recipe.from_dict(value)
                if r.kind!='tc':continue
                spec=module_spec(w,r);mid=digest(spec)
                modules[mid]=spec
                domain=grid_search(w,r) if role=='rule' and not point.get('fixed_schedules') else None
                modes=domain['modes'] if domain else [dict(r.parameters).get('grid_mode',0)]
                for algorithm in sorted({int(mode>=2) for mode in modes}):
                    request=probe_spec(w,r,algorithm);qid=digest(request)
                    row=queries.setdefault(qid,dict(request=request,module=mid,uses=[]))
                    use=dict(point=point['id'],recipe=r.key,role=role)
                    if use not in row['uses']:row['uses'].append(use)
    return dict(modules=modules,queries=queries,counts=dict(modules=len(modules),queries=len(queries)),
                scope='COMPILE_AND_RESOURCE_QUERY_ONLY_NOT_NUMERICS_OR_TIMING')


def compiler(spec, sdk, cache):
    from quactlize.runtime.compiler import Compiler
    from quactlize.decode.compiler import DecodeCompiler
    from quactlize.decode.grouped_compiler import GroupedComputeCompiler
    if spec['endpoint']=='native':
        if spec['compute']!='f16':raise ValueError('native dense requires F16')
        if spec.get('legacy_adapters'):
            # Compile the unchanged shipping casts alongside the experimental
            # control parent. Source text participates in the cache key.
            class LegacyCompiler(Compiler):
                def __init__(self, sdk, cache):
                    super().__init__(sdk, cache)
                    path=Path(__file__).resolve().parents[2]/'quactlize/dense/copy.cu'
                    self.adapters=path.read_text()
                    self.identity['legacy_adapters']=digest(self.adapters)
                    self.input_stats[path]=(path.stat().st_mtime_ns,path.stat().st_size)
                def source(self, parent, key):
                    return super().source(parent,key)+'\n'+self.adapters
            return LegacyCompiler(sdk,cache)
        return Compiler(sdk,cache)
    cls=GroupedComputeCompiler if spec['endpoint']=='grouped' else DecodeCompiler
    return cls(sdk,cache,compute_type=spec['compute'])


def request_abi(spec, device):
    from quactlize.runtime.native import Call
    from quactlize.decode.native_compute import (DenseIO,DenseComputeCall,
                                                GroupedDeviceCall,GroupedMetadataCall)
    if any(type(device.get(k)) is not int or device[k]<minimum
           for k,minimum in (('ordinal',0),('compute_units',1))):
        raise ValueError('positive real device resources required')
    call=Call(version=1,size=C.sizeof(Call),m=spec['m'],n=spec['n'],k=spec['k'],experts=spec['experts'],
              group_size=spec['group_size'],device=device['ordinal'],compute_units=device['compute_units'],
              mapping_id=spec['mapping'])
    compute=int(spec['module']['compute']=='bf16');endpoint=spec['module']['endpoint']
    if endpoint=='native':return call
    if endpoint=='typed':return DenseComputeCall(2,C.sizeof(DenseComputeCall),
                                               DenseIO(1,C.sizeof(DenseIO),call,1,1),compute)
    # v4, not the obsolete v3 request accidentally cast to a v4 pointer.
    metadata=int(compute and spec['module']['parent']['qtype']!=8)
    return GroupedMetadataCall(4,C.sizeof(GroupedMetadataCall),
           GroupedDeviceCall(2,C.sizeof(GroupedDeviceCall),call,spec['max_rows'],0),compute,metadata)


class NativeProbe:
    def __init__(self, spec, record):
        from quactlize.runtime.compiler import sha
        from quactlize.runtime.native import Identity,Resources,Recipe as Launch
        from quactlize.decode.native_compute import ComputeIdentity,ComputeMetadataIdentity
        if sha(record['path'])!=record['sha256'] or record['parent']!=spec['parent']:
            raise ValueError('resource module build identity differs')
        self.record,self.spec=record,spec
        self.lib=C.CDLL(str(Path(record['path']).resolve()),mode=C.RTLD_LOCAL)
        endpoint=spec['endpoint'];p=spec['parent']
        def bind(name,args,result=C.c_int):
            f=getattr(self.lib,'quactlize_kpack_'+name);f.argtypes,f.restype=args,result;return f
        prefix='decode_dense_' if endpoint=='typed' else ''
        getter=bind(prefix+'device_v1',[C.c_char_p,C.c_int,C.POINTER(C.c_int32),C.POINTER(C.c_int32)])
        name=C.create_string_buffer(256);ordinal,cu=C.c_int32(),C.c_int32()
        if getter(name,len(name),C.byref(ordinal),C.byref(cu)) or ordinal.value<0 or cu.value<=0:
            raise ValueError('module device query failed')
        self.device=dict(name=name.value.decode(),ordinal=ordinal.value,compute_units=cu.value)
        # This is an experimental capability probe, not permission to apply
        # the ZW810 policy to ZW610. No name/CU substitution or production edit.
        if not self.device['name'].startswith('PPU-'):raise ValueError('not a PPU target')
        raw=bind(prefix+'identity_v1',[],C.POINTER(Identity))()
        if not raw:raise ValueError('missing resource module identity')
        actual=raw.contents
        if (actual.version!=1 or actual.size!=C.sizeof(Identity) or
                actual.parent.decode()!=p['symbol'] or actual.build_key.decode()!=record['key'] or
                actual.route!=ROUTES.index(p['route']) or actual.delivery_n!=p['dn'] or
                any(getattr(actual,k)!=p[k] for k in ('qtype','tm','tn','tk','wm','wn','stages','ap'))):
            raise ValueError('loaded resource parent/config differs')
        self.mapping=actual.mapping_id
        if endpoint!='native':
            cls=ComputeMetadataIdentity if endpoint=='grouped' else ComputeIdentity
            version=4 if endpoint=='grouped' else 2
            symbol='compute_identity_v4' if endpoint=='grouped' else 'decode_dense_identity_v2'
            typed=bind(symbol,[],C.POINTER(cls))()
            if (not typed or typed.contents.version!=version or typed.contents.size!=C.sizeof(cls) or
                    typed.contents.compute_type!=int(spec['compute']=='bf16') or
                    not typed.contents.parent or C.addressof(typed.contents.parent.contents)!=C.addressof(actual) or
                    (endpoint=='grouped' and typed.contents.metadata_type!=int(spec['compute']=='bf16' and p['qtype']!=8))):
                raise ValueError('loaded resource compute/metadata differs')
        self._bind,self._resources,self._launch=bind,Resources,Launch

    def query(self, spec):
        if spec['module']!=self.spec or spec['mapping']!=self.mapping:
            raise ValueError('request differs from resource module')
        request=request_abi(spec,self.device);endpoint=self.spec['endpoint']
        symbol={'native':'query_v1','typed':'decode_dense_query_dynamic_v2','grouped':'grouped_query_v4'}[endpoint]
        fn=self._bind(symbol,[C.POINTER(type(request)),C.POINTER(self._launch),C.POINTER(self._resources)])
        launch=self._launch(1,C.sizeof(self._launch),spec['algorithm'],spec['split'],spec['algorithm'])
        resources=self._resources()
        rc=fn(C.byref(request),C.byref(launch),C.byref(resources))
        if rc not in (0,1):raise ValueError(f'resource query failed rc={rc}; runtime={resources.runtime_status}')
        if rc==0 and (resources.version!=1 or resources.size!=C.sizeof(self._resources) or resources.occupancy<=0):
            raise ValueError('resource query returned invalid occupancy/ABI')
        return dict(status='READY' if rc==0 else 'UNAVAILABLE',rc=rc,device=self.device,
                    shared_bytes=resources.shared_bytes,workspace_bytes=resources.workspace_bytes,
                    occupancy=resources.occupancy,runtime_status=resources.runtime_status,
                    cutlass_status=resources.cutlass_status,request=spec,
                    build_key=self.record['key'],module_sha256=self.record['sha256'])


def resolve_point(point, receipts):
    """Expand rules using measured resources; keep exact controls independent.

    Missing/failed queries block resolution. Unsupported kernels are recorded,
    not benchmarked, and a rejected control remains a missing comparison.
    """
    w=Workload(**point['workload']);result=dict(point);rejected=[];bindings={};missing=[]
    for field,role in (('candidates','rule'),('champion_controls','control')):
        output={}
        for value in point.get(field,[]):
            r=Recipe.from_dict(value)
            if r.kind!='tc':output[r.key]=value;continue
            domain=grid_search(w,r) if role=='rule' and not point.get('fixed_schedules') else None
            modes=domain['modes'] if domain else [dict(r.parameters).get('grid_mode',0)]
            for algorithm in sorted({int(mode>=2) for mode in modes}):
                spec=probe_spec(w,r,algorithm);key=digest(spec);row=receipts.get(key)
                if not row or row.get('status') not in ('READY','UNAVAILABLE'):
                    missing.append(dict(role=role,recipe=r.key,query=key));continue
                if row.get('request')!=spec:raise ValueError('resource receipt request differs')
                if row['status']=='UNAVAILABLE':
                    rejected.append(dict(role=role,recipe=r.key,query=key,reason='COMPILED_RESOURCE_UNAVAILABLE'));continue
                cu,occupancy=row['device']['compute_units'],row['occupancy']
                variants=(grid_variants(w,r,compute_units=cu,occupancy=occupancy) if domain else
                          [dict(recipe=r,launch=resolved_grid(w,r,cu,occupancy),aliases=[],
                                compute_units=cu,occupancy=occupancy)])
                for v in variants:
                    if v['launch']['algorithm']!=algorithm:continue
                    # B/mode aliases cost no extra timed arm. Keep their
                    # exact mapping in bindings; rule/control pools stay apart.
                    launch_key=digest([scheduled_key(v['recipe']),v['launch']])
                    if launch_key not in output:
                        output[launch_key]=asdict(v['recipe'])
                        bindings[v['recipe'].key]=dict(query=key,launch=v['launch'],aliases=v['aliases'])
        result[field]=list(output.values())
    result['resource_resolution']=dict(status='INCOMPLETE' if missing else 'COMPLETE',missing=missing,
        rejected=rejected,bindings=bindings,scope='RESOURCE_ADMISSION_NOT_CORRECTNESS_OR_PERFORMANCE',
        original_dynamic_query='STILL_REQUIRED',rejected_control_is_not_a_win=True)
    return json.loads(json.dumps(result))

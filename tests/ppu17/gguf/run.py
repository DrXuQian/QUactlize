"""Guarded dense Q4_K group32 numerical gate and bounded SF timing."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

CONTRACT='GGUF_Q4_K_G32_SF_PRIVATE_MMA_V1'
FQ_CONTRACT='GGUF_Q4_K_G32_PACKED_UNITS_PRIVATE_MMA_V1'
class Call(C.Structure):
    _fields_=[(x,C.c_int32) for x in ('dtype','config','m','n','k')]+[
        (x,C.c_void_p) for x in ('a','b','scales','zeros','output','stream')]
class FqCall(C.Structure):
    _fields_=[(x,C.c_int32) for x in ('dtype','config','m','n','k')]+[
        (x,C.c_void_p) for x in ('a','b','units','output','stream')]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def device_geometry():
    # Query the current CUDA context, not a guessed cache capacity or another
    # process's device. Driver attribute IDs are stable CUDA ABI constants.
    driver=C.CDLL('libcuda.so.1')
    current=C.c_int()
    driver.cuCtxGetDevice.argtypes=[C.POINTER(C.c_int)]
    driver.cuDeviceGetAttribute.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_int]
    if driver.cuCtxGetDevice(C.byref(current)):raise ValueError('current CUDA context unavailable')
    values=[]
    for attribute in (38,16):  # L2_CACHE_SIZE, MULTIPROCESSOR_COUNT
        value=C.c_int()
        if driver.cuDeviceGetAttribute(C.byref(value),attribute,current) or value.value<=0:
            raise ValueError('positive queried device geometry required')
        values.append(value.value)
    return values

class Library:
    def __init__(self,bundle,metadata='sf'):
        self.metadata=metadata
        manifest=json.loads((bundle/'manifest.json').read_text())
        row=manifest['builds']['adapter']
        if manifest['contract']!=CONTRACT or sha(bundle/row['file'])!=row['sha256']:
            raise ValueError('GGUF build identity differs')
        self.library=C.CDLL(str(bundle/row['file']),mode=C.RTLD_LOCAL)
        self.fn={}
        for name,result,arguments in (
            ('error',C.c_char_p,[]),('contract',C.c_char_p,[]),
            ('prepare',C.c_int,[C.POINTER(Call),C.POINTER(C.c_void_p)]),
            ('execute',C.c_int,[C.c_void_p]),('destroy',None,[C.c_void_p]),
            ('workspace',C.c_size_t,[C.c_void_p]),
            ('pack',C.c_int,[C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_void_p])):
            f=getattr(self.library,'q17_gguf_'+name)
            f.restype,f.argtypes=result,arguments
            self.fn[name]=f
        if self.fn['contract']().decode()!=CONTRACT:raise ValueError('wrong library contract')
        if metadata=='fq':
            if manifest.get('fq_contract')!=FQ_CONTRACT:raise ValueError('FQ manifest missing')
            contract=self.library.q17_gguf_fq_contract
            contract.restype=C.c_char_p
            if contract().decode()!=FQ_CONTRACT:raise ValueError('wrong FQ input contract')
            prepare=self.library.q17_gguf_prepare_fq
            prepare.restype=C.c_int
            prepare.argtypes=[C.POINTER(FqCall),C.POINTER(C.c_void_p)]
            self.fn['prepare_fq']=prepare
    def check(self,rc):
        if rc:raise ValueError(self.fn['error']().decode())
    def prepare(self,call):
        out=C.c_void_p()
        if self.metadata=='fq':
            fq=FqCall(call.dtype,call.config,call.m,call.n,call.k,
                      call.a,call.b,call.scales,call.output,call.stream)
            self.check(self.fn['prepare_fq'](C.byref(fq),C.byref(out)))
        else:self.check(self.fn['prepare'](C.byref(call),C.byref(out)))
        return out

class BringupBaseline:
    """Immutable canonical-reader control, not a mature-library baseline."""
    def __init__(self,bundle,source):
        sys.path.insert(0,str(source))
        from native import Backend, Call as NativeCall
        self.backend=Backend(bundle,allow_source_validation=True)
        self.Call=NativeCall
        self.fn={'execute':lambda plan:plan.execute(), 'destroy':lambda plan:plan.close(),
                 'workspace':lambda plan:0}
    def check(self,rc):
        if rc:raise ValueError('baseline execution failed')
    def prepare(self,c):
        p=self.backend.problem(qtype=12,operation=0,route=1,compute_type=c.dtype,
            input_type=c.dtype,output_type=c.dtype,m=c.m,n=c.n,k=c.k,experts=1,max_rows=c.m)
        configs=self.backend.configs(p)
        if c.config>=len(configs):raise ValueError('baseline config not available')
        call=self.Call(problem=p,config=configs[c.config],a=c.a,low=c.b,
            scale=c.scales,zero=c.zeros,output=c.output,stream=c.stream,
            a_stride=c.k,output_stride=c.n,device=0)
        return self.backend.prepare(call)

def proof(torch,storage,expected,m,n,dtype):
    all_values=storage.cpu().float()
    if not torch.all(all_values[:64]==-1232) or not torch.all(all_values[64+m*n:]==-1232):
        raise ValueError('output guard overwritten')
    got=all_values[64:64+m*n].reshape(m,n)
    error=got-expected
    rel=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected).clamp_min(1e-9))
    maximum=float(error.abs().max())
    tol=.003 if dtype==0 else .012
    if not torch.isfinite(got).all() or rel>tol or maximum>tol*max(1.,float(expected.abs().max())):
        raise ValueError(f'GGUF typed CPU dot mismatch rel={rel} max_abs={maximum}')
    return {'relative_l2':rel,'max_abs':maximum,'guards':'PASS',
            'sha256':hashlib.sha256(got.numpy().tobytes()).hexdigest()}

def child(a):
    if os.environ.get('Q17_BENCH_GUARDED')!='1':raise ValueError('idle guard required')
    import numpy as np
    import torch
    from fixture import make, stale_unit_groups
    torch.set_num_threads(8)
    lib=Library(a.bundle,a.metadata) if a.arm=='candidate' else BringupBaseline(a.baseline,a.backend_source)
    dtype=torch.float16 if a.dtype==0 else torch.bfloat16
    if torch.cuda.get_device_capability()!=(9,0):raise ValueError('physical SM90 required')
    stream=torch.cuda.Stream()
    torch.cuda.synchronize()
    l2_bytes,sm_count=device_geometry()
    if a.l2_bytes is not None and a.l2_bytes!=l2_bytes:
        raise ValueError('requested L2 differs from the current device')
    print('PPU17_BENCH_READY',flush=True)
    if sys.stdin.readline().strip()!='R':raise ValueError('not admitted by guard')
    shapes=[(a.m,a.n,a.k)] if a.timing or a.single else [(m,n,k) for n,k in ((128,256),(256,512))
                                           for m in (1,2,7,8,9,63,64,65,128)]
    results=[]
    for m,n,k in shapes:
        f=make(m,n,k,dtype,onehot=(not a.timing and n==128))
        packed=np.zeros(n*k//2,dtype=np.uint8)
        if a.arm=='candidate':
            lib.check(lib.fn['pack'](a.dtype,n,k,f['q'].ctypes.data,packed.ctypes.data))
        else:packed=f['canonical_low'].view(np.uint8).reshape(-1)
        with torch.cuda.stream(stream):
            act=f['a'].cuda()
            storage=torch.full((m*n+128,),-1232,dtype=dtype,device='cuda')
            out=storage[64:64+m*n]
            host=([torch.from_numpy(packed),torch.from_numpy(f['units'])] if a.metadata=='fq'
                  else [torch.from_numpy(packed),f['scale'],f['zero']])
            weight_bytes=sum(x.numel()*x.element_size() for x in host)
            copies=math.floor(2.25*l2_bytes/weight_bytes)+1 if a.timing else 1
            if copies>512:raise ValueError('bounded ring limit exceeded')
            weights=[[x.cuda() for x in host] for _ in range(copies)]
            for config in a.configs:
                plans=[]
                try:
                    for w in weights:
                        call=Call(a.dtype,config,m,n,k,act.data_ptr(),w[0].data_ptr(),
                                  w[1].data_ptr(),None if a.metadata=='fq' else w[2].data_ptr(),
                                  out.data_ptr(),stream.cuda_stream)
                        plans.append(lib.prepare(call))
                    def execute(i):lib.check(lib.fn['execute'](plans[i]))
                    checks=[]
                    for i in sorted({0,copies//2,copies-1}):
                        out.fill_(float('nan'));execute(i);stream.synchronize()
                        checks.append(proof(torch,storage,f['expected'],m,n,a.dtype))
                        if not a.timing and n==128 and checks[-1]['max_abs']!=0:
                            raise ValueError('one-hot decoded weight bits differ')
                    # Every second fine group must have independent metadata.
                    # A deliberate upstream-style one-row broadcast must fail.
                    negative=None
                    if not a.timing and m==9 and config==0:
                        original=weights[0][1].clone()
                        if a.metadata=='fq':
                            weights[0][1].copy_(torch.from_numpy(stale_unit_groups(f['units'])))
                        else:weights[0][1][1::2].copy_(original[0::2])
                        execute(0);stream.synchronize()
                        try:proof(torch,storage,f['expected'],m,n,a.dtype)
                        except ValueError as error:
                            if not str(error).startswith('GGUF typed CPU dot mismatch'):raise
                            negative='STALE_GROUP0_REJECTED'
                        else:raise ValueError('stale metadata negative remained green')
                        weights[0][1].copy_(original)
                    graph=torch.cuda.CUDAGraph()
                    launches=copies*max(1,math.ceil(32/copies)) if a.timing else 1
                    with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
                        for i in range(launches):execute(i%copies)
                    for sign in (1,-1,1):
                        act.copy_(f['a']*sign);out.fill_(float('nan'))
                        graph.replay();stream.synchronize()
                        checks.append(proof(torch,storage,f['expected']*sign,m,n,a.dtype))
                    row={'arm':a.arm,'metadata':a.metadata,'shape':[m,n,k],'dtype':a.dtype,'config':config,'status':'PASS',
                         'proof':checks,'raw_gguf_sha256':f['raw_sha256'],
                         'workspace_per_copy':lib.fn['workspace'](plans[0]),
                         'negative':negative}
                    if a.timing:
                        import subprocess
                        def clocks():
                            return subprocess.check_output(['nvidia-smi','--id='+a.device,
                                '--query-gpu=clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True).strip()
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record(stream);graph.replay();end.record(stream);end.synchronize()
                        ms=begin.elapsed_time(end)
                        warmups=math.ceil(250/ms)
                        if warmups>20000:raise ValueError('invalid warmup duration')
                        for _ in range(warmups):graph.replay()
                        stream.synchronize();before=clocks();samples=[]
                        for _ in range(11):
                            begin.record(stream);graph.replay();end.record(stream);end.synchronize()
                            samples.append(begin.elapsed_time(end)*1000/launches)
                        after=clocks()
                        if before!=after:raise ValueError('clock changed; timing rejected')
                        row.update(samples_us=samples,median_us=statistics.median(samples),
                                   weight_bytes=weight_bytes,weight_copies=copies,launches=launches,
                                   clocks=before,cache='ROTATING_GT_2_25_L2')
                    results.append(row)
                    print('GGUF_Q4_CELL '+json.dumps(row),flush=True)
                    graph.reset()
                finally:
                    stream.synchronize()
                    for p in plans:lib.fn['destroy'](p)
    print('PPU17_BENCH '+json.dumps({'status':'PASS','contract':FQ_CONTRACT if a.metadata=='fq' else CONTRACT,
          'scope':'PACKED_UNITS_FQ_NO_SF_PREPASS' if a.metadata=='fq' else 'RESIDENT_SF_NOT_EXPANSION_OR_FQ','timing':a.timing,
          'l2_bytes':l2_bytes,'sm_count':sm_count,'rows':results}),flush=True)

def validate(row):
    if row.get('status')!='PASS' or row.get('contract') not in (CONTRACT,FQ_CONTRACT) or not row.get('rows'):
        raise ValueError('incomplete GGUF gate')
    for r in row['rows']:
        if r['status']!='PASS' or not r['proof']:raise ValueError('missing numerical proof')
        if row['timing']:
            samples=r['samples_us']
            if len(samples)!=11 or any(not math.isfinite(x) or x<=0 for x in samples):
                raise ValueError('invalid timing')
            if statistics.median(samples)!=r['median_us'] or r['launches']%r['weight_copies']:
                raise ValueError('invalid median or ring')
            if r['weight_bytes']*r['weight_copies']<=2.25*row['l2_bytes']:
                raise ValueError('insufficient rotating weight set')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--arm',choices=('candidate','bringup-baseline'),default='candidate')
    p.add_argument('--metadata',choices=('sf','fq'),default='sf')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--backend-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-uuid',required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--dtype',type=int,choices=(0,1),default=0)
    p.add_argument('--configs',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--m',type=int,default=1);p.add_argument('--n',type=int,default=512)
    p.add_argument('--k',type=int,default=2048)
    p.add_argument('--l2-bytes',type=int,help='optional assertion against queried capacity')
    p.add_argument('--timing',action='store_true');p.add_argument('--child',action='store_true')
    p.add_argument('--single',action='store_true',help='one bounded numerical shape for sanitizer')
    a=p.parse_args()
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    allowed=range(manifest.get('config_count',3)) if a.arm=='candidate' else (0,1,2,3,4,5)
    if not a.configs or len(set(a.configs))!=len(a.configs) or any(x not in allowed for x in a.configs):
        raise ValueError('invalid config inventory')
    if a.arm!='candidate' and not a.baseline:raise ValueError('baseline bundle required')
    if a.arm!='candidate' and a.metadata!='sf':raise ValueError('bring-up control is SF only')
    if a.l2_bytes is not None and a.l2_bytes<=0:raise ValueError('positive verified L2 required')
    if a.child:return child(a)
    if a.output.exists():raise ValueError('fresh result directory required')
    a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.backend_source))
    import run_benchmark as guard
    def exact_validate(row):
        validate(row)
        expected=FQ_CONTRACT if a.metadata=='fq' else CONTRACT
        if row['contract']!=expected:raise ValueError('wrong metadata route')
        if any(r.get('metadata','sf')!=a.metadata for r in row['rows']):
            raise ValueError('cell metadata route differs')
    guard.validate_result=exact_validate
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child']
    try:
        result=guard.run_point(command,{**os.environ,'CUDA_VISIBLE_DEVICES':a.device,'Q17_BENCH_GUARDED':'1'},
                               a.output/'run.log',a.device,a.expected_uuid,600)
    except Exception as error:
        result={'status':'INVALID','error':str(error),'result':[]}
    result['runner_sha256']=sha(__file__)
    result['fixture_sha256']=sha(Path(__file__).with_name('fixture.py'))
    result['manifest_sha256']=sha(a.bundle/'manifest.json')
    if a.baseline:result['baseline_manifest_sha256']=sha(a.baseline/'manifest.json')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GGUF_Q4_DONE '+result['status']+' '+str(result.get('error')),flush=True)
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())

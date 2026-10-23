"""One-kernel component measurements only; provider timing uses ordinary Asys calls."""
import ctypes as C
import hashlib
import math
import statistics

import numpy as np

from dev.bf16_compute.fixture import bf16_bits,bf16_float
from dev.dense_helpers.plan import variants
from quactlize.runtime.native import SDK,checked
from tools.run_dense_dequant import Bench
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay


def timed(r,graphs,replays):
    samples={v:[] for v in graphs}
    for round in range(3):
        for v in list(graphs)+list(reversed(graphs)):
            samples[v].extend(t/replays for t in r.samples(graphs[v],5))
    return {str(v):dict(samples_us=values,median_us=statistics.median(values),
                       round_medians_us=[statistics.median(values[i:i+10]) for i in (0,10,20)])
            for v,values in samples.items()}


def dequant(a,job):
    b=Bench(a,shape=(job['q'],job['n'],job['k']))
    try:
        storage=job['storage'];vs=variants(job['q']);proofs={}
        control,proofs[0]=b.check(0,storage);proofs[0]['control_raw_bits_equal']=True
        for v in vs[1:]: _,proofs[v]=b.check(v,storage,control)
        negative=b.negative(0)
        graphs={v:b.graph(v,storage) for v in vs}
        arms=timed(b.r,graphs,b.replays)
        for v in vs:
            checked(graphs[v](),'final dequant replay')
            for i in (0,b.copies-1):
                if not np.array_equal(b.read(i),control): raise ValueError('changed-address dequant bits differ')
            arms[str(v)]['proof']=proofs[v]
        return dict(arms=arms,negative_detected=negative>0,fixture=b.fixture,
                    traffic=b.traffic,copies=b.copies,replays=b.replays)
    finally:b.close()


def io(a,job,library):
    sdk=SDK(a.sdk);graph_bind(sdk);r=Resources(sdk);graphs={}
    fn=library.quactlize_dense_io_v1
    fn.argtypes=[C.c_int]*3+[C.c_void_p]*2+[C.c_int]*2+[C.c_int64,C.c_void_p,C.c_void_p]
    fn.restype=C.c_int
    m,width,storage,input=(job[k] for k in ('m','width','storage','input'))
    count=m*width;guard=256
    values=(np.arange(count,dtype='u4')%1009).astype('<f4').reshape(m,width)/32-16
    packed=bf16_bits(values) if storage else values.astype('<f2').view('<u2')
    if input: source=values;want=packed
    else: source=packed;want=bf16_float(packed) if storage else packed.view('<f2').astype('<f4')
    copies=max(2,math.ceil(2.25*a.l2_bytes/min(source.nbytes,want.nbytes)))
    replays=2*copies;source_stride=source.nbytes+2*guard;dest_stride=want.nbytes+2*guard
    try:
        src=r.alloc(source_stride*copies);dst=r.alloc(dest_stride*copies);error=r.alloc(4)
        r.fill(src,0xa5,source_stride*copies);r.fill(dst,0xa5,dest_stride*copies);r.fill(error,0,4)
        sdk.synchronize(r.stream)
        checked(sdk.lib.hggcMemcpy(src+guard,source.ctypes.data,source.nbytes,1),'IO source upload')
        for i in range(1,copies):
            checked(sdk.lib.hggcMemcpy(src+i*source_stride+guard,src+guard,source.nbytes,3),'IO source ring')
        def launch(v,index=0):
            return fn(storage,input,v,src+index*source_stride+guard,dst+index*dest_stride+guard,
                      m,width,width,error,r.stream)
        def read(index=0):
            sdk.synchronize(r.stream)
            data=np.frombuffer(sdk.download(dst+index*dest_stride,dest_stride),dtype='u1')
            if not np.all(data[:guard]==0xa5) or not np.all(data[-guard:]==0xa5):
                raise ValueError('IO output redzone overwritten')
            return data[guard:-guard].copy()
        desired=want.view('u1').reshape(-1)
        negative=not np.array_equal(read(),desired)
        proofs={}
        for v in range(4):
            checked(launch(v),'IO correctness')
            got=read()
            if not np.array_equal(got,desired):raise ValueError('IO raw bits differ from independent cast')
            proofs[v]=dict(bad=0,guard='PASS',control_raw_bits_equal=True,
                           compared=count,output_sha256=hashlib.sha256(got).hexdigest())
            cursor=[0]
            def submit(v=v,cursor=cursor):
                rc=launch(v,cursor[0]%copies);cursor[0]+=1;return rc
            graphs[v]=Replay(sdk,r.stream,submit,replays)
            for _ in range(2):checked(graphs[v](),'IO excluded warmup')
        arms=timed(r,graphs,replays)
        for v in range(4):
            checked(graphs[v](),'IO final changed-address replay')
            for i in (0,copies-1):
                if not np.array_equal(read(i),desired):raise ValueError('IO changed-address bits differ')
            arms[str(v)]['proof']=proofs[v]
        if not input:
            r.fill(error,1,4);checked(launch(2),'IO error propagation')
            if not np.isnan(read().view('<f4')).all():raise ValueError('IO lost producer error')
        return dict(arms=arms,negative_detected=negative,copies=copies,replays=replays,
            traffic=dict(reads=source.nbytes,writes=want.nbytes,
                         input_ring_bytes=source.nbytes*copies,output_ring_bytes=want.nbytes*copies))
    finally:
        sdk.synchronize(r.stream)
        for graph in graphs.values():graph.close()
        r.close()

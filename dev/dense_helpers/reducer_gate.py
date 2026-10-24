"""Actual existing reducer bodies: independent fixed-S FP32/F16 oracle, no timing."""
import ctypes as C
import hashlib

import numpy as np

from quactlize.runtime.native import SDK, checked
from tools.run_kpack_gemv_gate import Resources


def cases(points):
    result=set()
    for point in points:
        w=point['workload']
        for case in point['cases']:
            p=dict(case['recipe']['parameters'])
            if p.get('dense_reducer')==2:
                m,n,s=w['m'],w['n'],p['split']
                if (case['arm']!='sf' or w['operation']!='dense' or w['compute']!='f16' or
                        not 8<m<=2048 or not 0<n<=65536 or m*n%64 or s not in (2,4,8)):
                    raise ValueError('reducer gate workload differs from native dense prefill')
                result.add((m,n,n,s,0,0))
    if not result:raise ValueError('no compact reducer candidates')
    # M1, weak 128B workspace alignment, non-vector tail, strided output,
    # half-aligned output, S8, and S1 fallback. Offsets are bytes.
    result.update(((1,512,512,2,0,0),(1,512,512,4,16,0),(3,21,21,4,0,0),
                   (3,64,69,8,0,0),(4,128,128,4,0,2),(3,64,64,8,0,0),
                   (3,64,64,1,0,0)))
    return sorted(result)


def partials(m,n,s):
    i=np.arange(m*n,dtype='i8')
    x=np.stack([(((i*17+j*13)%257)-128).astype('<f4')/256+np.float32((j+1)/32)
                for j in range(s)])
    # Cancellation distinguishes increasing-S addition from trees/reordering.
    # Rounding/overflow are tested after summation, not on partials.
    x[:,:8]=0
    x[0,:8]=[2**24,1,1+2**-11,1+3*2**-11,65504,65520,2**-24,-2**-24]
    if s>=2:x[1,0]=-2**24;x[1,1]=2**24
    if s>=4:x[2,0]=1;x[2,1]=-2**24
    return x


def expected(x):
    acc=np.zeros(x.shape[1],dtype='<f4')
    for plane in x:np.add(acc,plane,out=acc)
    with np.errstate(over='ignore'):
        return acc.astype('<f2').view('<u2')


def bind(library):
    lib=C.CDLL(str(library),mode=C.RTLD_LOCAL)
    prepare=lib.prefill_reducer_prepare_strided
    prepare.argtypes=[C.c_int]*4+[C.c_void_p,C.c_uint64,C.c_void_p,C.c_int64,C.POINTER(C.c_void_p)]
    prepare.restype=C.c_int
    run=lib.prefill_reducer_run;run.argtypes=[C.c_void_p,C.c_void_p];run.restype=C.c_int
    fast=lib.prefill_reducer_fast;fast.argtypes=[C.c_void_p];fast.restype=C.c_int
    destroy=lib.prefill_reducer_destroy;destroy.argtypes=[C.c_void_p];destroy.restype=None
    return prepare,run,fast,destroy


def run(sdk_path,library,points):
    sdk=SDK(sdk_path);prepare,launch,fast,destroy=bind(library);records=[]
    for m,n,stride,s,src_offset,dst_offset in cases(points):
        r=Resources(sdk)
        try:
            x=partials(m,n,s);want=expected(x)
            src_prefix=256+src_offset;dst_prefix=256+dst_offset
            source=np.full(src_prefix+x.nbytes+256,0xa5,dtype='u1')
            source[src_prefix:-256]=x.view('u1').reshape(-1)
            src=r.upload(source);span=(m-1)*stride+n
            size=dst_prefix+span*2+256;dst=r.alloc(size)
            indices=(np.arange(m)[:,None]*stride+np.arange(n)).reshape(-1)
            mask=np.zeros(size,dtype=bool)
            mask[dst_prefix+2*indices]=True;mask[dst_prefix+2*indices+1]=True
            control=None
            for compact in (0,1):
                h=C.c_void_p()
                checked(prepare(compact,m,n,s,src+src_prefix,x.nbytes,dst+dst_prefix,stride,C.byref(h)),
                        'reducer numerical prepare')
                if not h.value:raise ValueError('reducer prepare returned null handle')
                try:
                    count=m*n if compact else n
                    want_fast=int((compact or m==1) and stride==n and count%64==0 and s in (2,4,8)
                                  and src_offset%(16 if compact else 128)==0 and dst_offset%16==0)
                    if fast(h)!=want_fast:raise ValueError('reducer path diagnostic differs')
                    for replay in (0,1):
                        r.fill(dst,0xa5,size);checked(launch(h,r.stream),'reducer numerical launch')
                        sdk.synchronize(r.stream)
                        raw=np.frombuffer(sdk.download(dst,size),dtype='u1')
                        if np.any(raw[~mask]!=0xa5):raise ValueError('reducer output guard/padding overwritten')
                        got=raw[dst_prefix:-256].view('<u2')[indices]
                        if not np.array_equal(got,want):raise ValueError('reducer fixed-S FP32/F16 oracle differs')
                        if control is None:control=got.copy()
                        elif not np.array_equal(got,control):raise ValueError('reducer control/replay bits differ')
                        records.append(dict(m=m,n=n,stride=stride,split=s,src_offset=src_offset,
                            dst_offset=dst_offset,compact=compact,fast=want_fast,replay=replay,bad=0,
                            guard='PASS',sha256=hashlib.sha256(got.tobytes()).hexdigest()))
                finally:destroy(h)
                for ps,bytes_,pd,ld in ((src+src_prefix,x.nbytes-1,dst+dst_prefix,stride),
                        (src+src_prefix,x.nbytes,src+src_prefix,n),
                        (src+src_prefix+4,x.nbytes,dst+dst_prefix,stride),
                        (src+src_prefix,x.nbytes,dst+dst_prefix,n-1)):
                    bad=C.c_void_p()
                    status=prepare(compact,m,n,s,ps,bytes_,pd,ld,C.byref(bad))
                    if bad.value:destroy(bad)
                    if status==0 or bad.value:raise ValueError('reducer invalid span/alignment accepted')
            if sdk.download(src,len(source))!=source.tobytes():raise ValueError('reducer partials modified')
        finally:r.close()
    return dict(status='PASS',cases=len(records),records=records,
                scope='EXISTING_REDUCER_BODIES_RAW_BITS_AND_GUARDS_NO_TIMING',performance_admitted=False)

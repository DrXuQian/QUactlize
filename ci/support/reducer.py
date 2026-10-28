"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import numpy as np
def cases(points):
    result=set()
    for point in points:
        w=point['workload']
        for case in point['cases']:
            p=dict(case['recipe']['parameters'])
            if p.get('dense_reducer'):
                m,n,s=w['m'],w['n'],p['split']
                # The reducer flattens contiguous M*N; 2048 was an old
                # experiment limit, not a kernel restriction. Cover the full
                # M-span inventory while preserving dtype/stride/S contracts.
                if (case['arm']!='sf' or w['operation']!='dense' or w['compute']!='f16' or
                        not 8<m<=8192 or not 0<n<=65536 or m*n%64 or s not in (2,4,8) or
                        type(p['dense_reducer']) is not int or p['dense_reducer'] not in (2,4,8)):
                    raise ValueError('reducer gate workload differs from dense F16 prefill: '
                                     f'M={m} N={n} split={s} width={p["dense_reducer"]}; '
                                     'require 8<M<=8192, 0<N<=65536, M*N aligned to 64, '
                                     'split/width in {2,4,8}')
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

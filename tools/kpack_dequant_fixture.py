"""Original-GGUF oracle, independent of the new device expansion reader."""
import numpy as np
from reference import gguf_kpack as ref
from tools.kpack_warmup_fixture import prepare_expert
from tools.run_kpack_gemv_gate import metadata_oracle


def bf16(values):
    bits=np.ascontiguousarray(values,dtype='<f4').view('<u4')
    if not np.isfinite(values).all():raise ValueError('nonfinite BF16 fixture')
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2')


def expert(q,n,k,e):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    s=ref.SPECS[q]
    rng=np.random.default_rng(np.random.SeedSequence([935712,q,n,k,e]))
    raw=rng.integers(0,256,(n*(k//256),s.raw_bytes),dtype='u1')
    for offset in (s.d_offset,s.dmin_offset):
        if offset>=0:
            d=(rng.random(raw.shape[0])*.02+.005).astype('<f2')
            raw[:,offset:offset+2]=d.view('u1').reshape(-1,2)
    planes=prepare_expert(raw,q,n,k)
    full=bf16(dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k))
    sf=metadata_oracle(planes['units'],q,n,k,1)
    return planes,full,tuple(x[0] for x in sf)


def fixture(q,n,k,experts,operation,progress=None):
    first,full,sf=expert(q,n,k,0)
    planes={name:np.empty((experts,*first[name].shape),dtype=first[name].dtype)
            for name in ('low','high','units')}
    gold=np.empty((experts,*full.shape),dtype='<u2') if operation else np.empty((2,experts,*sf[0].shape),dtype='<u2')
    for e in range(experts):
        if e: first,full,sf=expert(q,n,k,e)
        for name in planes:planes[name][e]=first[name]
        if operation:gold[e]=full
        else:
            for i in range(2):gold[i,e]=sf[i].view('<u2')
        if progress and (e+1==experts or (e+1)%32==0):progress(e+1,experts)
    return planes,gold


def compare(got, want):
    if got.shape!=want.shape:raise ValueError('dequant oracle shape differs')
    # Raw signed zero is reported but has no BF16 weight-value difference.
    signed=(got!=want)&((got&0x7fff)==0)&((want&0x7fff)==0)
    bad=(got!=want)&~signed
    indices=np.flatnonzero(bad)
    if indices.size:
        i=int(indices[0]);raise ValueError(f'dequant raw bits bad={len(indices)}/{want.size} first={i} want=0x{want.flat[i]:04x} got=0x{got.flat[i]:04x}')
    return dict(cells=want.size,bad=0,signed_zero_differences=int(signed.sum()))

"""All-format raw GGUF oracle with independently varying metadata fields."""
import hashlib
import numpy as np
from dev.smallm_closure.fixture import Weights as BaseWeights
from dev.bf16_compute.fixture import planes


def raw_weight(q,n,k,seed):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    from reference.gguf_kpack import SPECS
    block,width=(32,34) if q==8 else (256,SPECS[q].raw_bytes)
    raw=np.random.default_rng(seed).integers(0,256,(n,k//block,width),dtype='u1')
    tags=np.arange(n)[:,None]*3+np.arange(k//block)[None,:]+seed
    d=np.ldexp(np.where(tags&1,-1.,1.),-(tags%5+12)).astype('<f2')
    off=0 if q==8 else SPECS[q].d_offset
    raw[...,off:off+2]=d.view('u1').reshape(n,-1,2)
    if q!=8 and SPECS[q].has_min:
        off=SPECS[q].dmin_offset
        raw[...,off:off+2]=(d/np.float16(2)).astype('<f2').view('u1').reshape(n,-1,2)
    if q==14:
        off=SPECS[q].scale_offset
        raw[...,off:off+3]=np.asarray([128,0,127],dtype='u1')
    gold=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k).astype('f4')
    if not np.isfinite(gold).all() or not np.any(gold):raise ValueError('invalid GGUF oracle')
    return raw,gold


class Weights(BaseWeights):
    def __init__(self,q,n,k,experts):
        self.q,self.n,self.k,self.experts=q,n,k,experts
        self.categories=np.random.default_rng(81811+k).integers(0,4,k)
        self.planes={};count=min(experts,17)
        self.sums=np.empty((count,4,n),dtype='f8');self.absolute=np.empty_like(self.sums)
        hashes=[]
        for expert in range(count):
            hasher=hashlib.sha256()
            for first in range(0,n,256):
                size=min(256,n-first)
                raw,gold=raw_weight(q,size,k,710+expert*31+q*1009+first)
                hasher.update(raw.tobytes());packed=planes(raw,q)
                for name in ('low','high','units'):
                    arr=packed[name]
                    if not arr.size:self.planes[name]=np.empty(0,'u1');continue
                    if name not in self.planes:
                        shape=list(arr.shape);shape[1]=n
                        self.planes[name]=np.empty((experts,*shape),dtype=arr.dtype)
                    self.planes[name][expert,:,first:first+size]=arr
                for group in range(4):
                    part=gold[:,self.categories==group]
                    self.sums[expert,group,first:first+size]=part.sum(axis=1,dtype='f8')
                    self.absolute[expert,group,first:first+size]=np.abs(part).sum(axis=1,dtype='f8')
            hashes.append(hasher.hexdigest())
        for arr in self.planes.values():
            if arr.size:
                for expert in range(count,experts):arr[expert]=arr[expert%count]
        self.record=dict(q=q,n=n,k=k,experts=experts,raw_sha256=hashes,unique_patterns=count,
                         oracle='OFFICIAL_GGUF_F32_DOT',metadata='VARYING_ALL_GROUP_FIELDS_SIGNED_SUPERSCALES',
                         activation='FOUR_K_CATEGORIES_ROUNDED_TO_REQUESTED_COMPUTE')


def grouped(data,experts,channels,topk):
    """Fixture-only stable compaction; timed ABI already consumes grouped A.

    This does not claim free model routing or host preparation in production.
    Indexed mode is measured separately with its real GPU IDs lookup.
    """
    ids=data['ids'].ravel();order=np.argsort(ids,kind='stable')
    from_rows=np.arange(len(ids))//topk*channels+np.arange(len(ids))%topk%channels
    counts=np.bincount(ids,minlength=experts)
    return dict(a=np.ascontiguousarray(data['a'][from_rows[order]]),ids=None,
                offsets=np.concatenate(([0],np.cumsum(counts))).astype('i4'),owners=ids[order],
                gold=data['gold'][order],denom=data['denom'][order])

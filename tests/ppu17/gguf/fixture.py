"""Independent raw GGUF Q4_K decoder and typed SF oracle (no production loader)."""
import hashlib
import numpy as np
import torch

def raw_blocks(n, k):
    if n < 128 or n % 128 or k < 256 or k % 256:
        raise ValueError('complete Q4_K superblocks required')
    rng=np.random.default_rng(4073+n+7*k)
    raw=np.zeros((n,k//256,144),dtype=np.uint8)
    d=rng.uniform(.0002,.003,(n,k//256)).astype(np.float16)
    dm=rng.uniform(.0001,.001,(n,k//256)).astype(np.float16)
    raw[:,:,:2]=d.view(np.uint8).reshape(n,k//256,2)
    raw[:,:,2:4]=dm.view(np.uint8).reshape(n,k//256,2)
    scales=rng.integers(1,64,(n,k//256,8),dtype=np.uint8)
    mins=rng.integers(0,64,(n,k//256,8),dtype=np.uint8)
    raw[:,:,4:8]=(scales[:,:,:4]&63)|((scales[:,:,4:]>>4)<<6)
    raw[:,:,8:12]=(mins[:,:,:4]&63)|((mins[:,:,4:]>>4)<<6)
    raw[:,:,12:16]=(scales[:,:,4:]&15)|((mins[:,:,4:]&15)<<4)
    raw[:,:,16:]=rng.integers(0,256,(n,k//256,128),dtype=np.uint8)
    return raw

def decode(raw, dtype):
    n, sb, _=raw.shape
    d=raw[:,:,:2].copy().view(np.float16).reshape(n,sb).astype(np.float32)
    dm=raw[:,:,2:4].copy().view(np.float16).reshape(n,sb).astype(np.float32)
    sc=np.empty((n,sb,8),dtype=np.int32)
    mi=np.empty_like(sc)
    for g in range(8):
        if g<4:
            sc[:,:,g]=raw[:,:,4+g]&63
            mi[:,:,g]=raw[:,:,8+g]&63
        else:
            sc[:,:,g]=(raw[:,:,8+g]&15)|((raw[:,:,g]>>6).astype(np.int32)<<4)
            mi[:,:,g]=(raw[:,:,8+g]>>4)|((raw[:,:,g+4]>>6).astype(np.int32)<<4)
    q=np.empty((n,sb,256),dtype=np.uint8)
    for g in range(8):
        offset=16+(g//2)*32
        q[:,:,g*32:(g+1)*32]=(raw[:,:,offset:offset+32]>>(4*(g%2)))&15
    # The bias correction and two rounded arithmetic operations are the
    # existing GGUF SF contract, not GPTQ scale-only or merged groups.
    scale=torch.from_numpy((d[:,:,None]*sc).astype(np.float32)).to(dtype)
    zero=torch.from_numpy((-dm[:,:,None]*mi).astype(np.float32)).to(dtype)
    correction=scale.float()*8
    if dtype==torch.bfloat16:correction=correction.to(dtype).float()
    zero=(zero.float()+correction).to(dtype)
    codes=torch.from_numpy(q.reshape(n,sb*256).copy()).to(torch.int16)-8
    product=(codes.float()*scale.reshape(n,-1).repeat_interleave(32,1).float()).to(dtype)
    weight=(product.float()+zero.reshape(n,-1).repeat_interleave(32,1).float()).to(dtype)
    return {'q':q.reshape(n,sb*256).T.copy(),
            'scale':scale.reshape(n,-1).T.contiguous(),
            'zero':zero.reshape(n,-1).T.contiguous(),'weight':weight}

def make(m,n,k,dtype, *, onehot=False):
    raw=raw_blocks(n,k)
    f=decode(raw,dtype)
    rng=np.random.default_rng(191+m+n+k)
    a=torch.from_numpy(rng.uniform(-1,1,(m,k)).astype(np.float32)).to(dtype)
    if onehot:
        a.zero_()
        for row in range(m): a[row,(row*37+31)%k]=1
    f['a']=a
    f['expected']=(a.double()@f['weight'].double().T).to(dtype).float()
    f['raw_sha256']=hashlib.sha256(raw.tobytes()).hexdigest()
    # The canonical low plane remains independently available as a control.
    low=np.zeros((k//4,n),dtype=np.uint16)
    for kk in range(k):
        low[(kk//32)*8+kk%8] |= f['q'][kk].astype(np.uint16) << (4*((kk%32)//8))
    f['canonical_low']=low
    return f

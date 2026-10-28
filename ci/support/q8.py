"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import numpy as np
def fixture(n,k,experts):
    rng=np.random.default_rng(83008)
    raw=rng.integers(0,256,(experts,n,k//32,34),dtype=np.uint8)
    d=(rng.integers(1,5,(experts,n,k//32))*0.03125).astype('<f2')
    d[...,::3]*=-1
    raw[...,:2]=d.view('u1').reshape(experts,n,k//32,2)
    codes=raw[...,2:].copy().reshape(experts,n,k)
    kk=np.arange(k)
    low=np.empty((experts,k//2,n,2),dtype=np.uint8)
    low[:,(kk//16)*8+kk%8,:,kk%16//8]=(codes^128).transpose(2,0,1)
    scale=raw[...,:2].transpose(0,2,1,3).copy().reshape(-1)
    # GGUF meaning, independent of placement/reader/converter. W is rounded
    # to FP16 just as the register dequantizer does before FP32 accumulation.
    weight=(codes.view('i1').astype('f4')*np.repeat(d.astype('f4'),32,axis=2)).astype('<f2')
    return raw,low.reshape(-1),scale,weight

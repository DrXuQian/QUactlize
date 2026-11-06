"""Independent GGUF/Flare references, used only by tests and the example."""
import numpy as np
import gguf


def row_blocks(q, rows, columns):
    block = 64 if q == 42 else 32
    assert q in (8,20,42) and columns%block==0
    width=34 if q==8 else 18
    raw=np.empty((rows,columns//block,width),dtype='u1')
    scale=np.resize(np.array([.03125,-.125,1.,.0009765625],dtype='<f2'),(rows,columns//block))
    raw[...,:2]=scale[...,None].view('u1')
    if q==8:
        codes=np.resize(np.arange(-128,128,dtype='i2').astype('i1'),(rows,columns//32,32))
        raw[...,2:]=codes.view('u1')
    elif q == 42:
        raw[...,2:]=np.resize(np.array([0xe4,0x1b,0,255],'u1'),raw[...,2:].shape)
    else:
        # Every codebook entry in BOTH nibbles, with different ordering.
        raw[...,2:]=np.arange(16,dtype='u1')|((15-np.arange(16,dtype='u1'))<<4)
    return raw.reshape(rows,-1)


def dequant_rows(raw,q):
    if q == 42:
        from .q2_0 import dequantize
        return dequantize(raw).reshape(raw.shape[0],-1)
    return gguf.dequantize(raw,gguf.GGMLQuantizationType(q)).astype('f4')


def router_reference(logits,topk,live=None,multiplier=1.):
    logits=np.asarray(logits,dtype='f4');m,e=logits.shape
    live=m if live is None else live
    selected=np.zeros((m,topk),'f4');ids=np.full((m,topk),-1,'i4');weights=selected.copy();status=0
    if not 0<=live<=m:return selected,ids,weights,4
    for r in range(live):
        if not np.isfinite(logits[r]).all():status|=2;continue
        order=np.lexsort((-np.arange(e),-logits[r]))[:topk]
        ids[r]=order;selected[r]=logits[r,order]
        ex=np.exp(selected[r]-selected[r,0]).astype('f4')
        total=np.float32(0)
        for x in ex:total=np.float32(total+x)
        weights[r]=ex/total*np.float32(multiplier)
    return selected,ids,weights,status

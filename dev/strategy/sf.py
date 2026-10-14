"""Independent typed metadata oracle; BF16 never passes through FP16 products."""
import numpy as np


def metadata_bits(units,q,n,k,experts,bf16=False):
    if not bf16:
        from tools.run_kpack_gemv_gate import metadata_oracle
        return tuple(x.view('<u2') for x in metadata_oracle(units,q,n,k,experts))
    from reference import gguf_kpack as ref
    from dev.bf16_compute.fixture import bf16_bits,bf16_float
    spec=ref.SPECS[q];sb=k//256
    packed=units.reshape(experts,sb//spec.superblocks_per_unit,n,spec.superblocks_per_unit,spec.sb_bytes)
    packed=packed.transpose(0,1,3,2,4).reshape(experts,sb,n,spec.sb_bytes)
    d=packed[...,:2].copy().view('<f2')[...,0].astype('f4')
    dm=packed[...,2:4].copy().view('<f2')[...,0].astype('f4') if spec.has_min else np.zeros_like(d)
    scale,zero=np.empty((2,experts,sb*spec.groups,n),dtype='<u2')
    rnd=lambda v:bf16_float(bf16_bits(v))
    for g in range(spec.groups):
        codes=[]
        for which,bits in ((0,spec.scale_bits),(1,spec.min_bits)):
            if not bits:codes.append(np.zeros_like(d));continue
            byte,shift=divmod(ref._unit_bit(spec,g,which),8)
            word=packed[...,byte].astype('u2')
            if shift+bits>8:word|=packed[...,byte+1].astype('u2')<<8
            codes.append(((word>>shift)&((1<<bits)-1)).astype('i2'))
        sc,mn=codes
        if q==11:sc=sc-32
        if q==14:sc=np.where(sc>=128,sc-256,sc)
        s=rnd(d*sc);z=rnd(-dm*mn) if spec.has_min else np.zeros_like(s)
        zmul={10:0,11:-4,12:8,13:8,14:-24}[q]
        if zmul:z=rnd(z+rnd(np.float32(zmul)*s))
        scale[:,g::spec.groups]=bf16_bits(s);zero[:,g::spec.groups]=bf16_bits(z)
    return scale,zero

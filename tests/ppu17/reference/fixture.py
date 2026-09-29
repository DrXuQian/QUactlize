"""Existing GPTQ symmetric INT4 contract; no GGUF or requantization path."""
import importlib.util
from pathlib import Path

import numpy as np

CONTRACT = 'GPTQ_SYM_INT4_G128_F16_V1'
ROOT = Path(__file__).resolve().parents[3]


def importer():
    path = ROOT/'tools/dump_real_weights.py'
    spec = importlib.util.spec_from_file_location('q17_gptq_importer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.unpack_gptq_expert


def decode_checkpoint(qweight, qzeros, scales, *, group_size=128, desc_act=False, g_idx=None):
    """Validate the existing symmetric, direct-zero8, non-act-order importer."""
    if group_size != 128 or desc_act:
        raise ValueError('comparison requires existing group128, desc_act=false contract')
    if qweight.ndim != 2 or qweight.dtype not in (np.dtype('int32'), np.dtype('uint32')):
        raise ValueError('qweight must be packed int32 [K/8,N]')
    k, n = qweight.shape[0]*8, qweight.shape[1]
    if k < 128 or k % 128 or n < 128 or n % 128:
        raise ValueError('GPTQ comparison geometry')
    if scales.shape != (k//128,n) or scales.dtype != np.float16:
        raise ValueError('preserve original FP16 scale tensor and grouping')
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('scales must be finite and positive')
    if qzeros.shape != (k//128,n//8) or qzeros.dtype not in (np.dtype('int32'),np.dtype('uint32')):
        raise ValueError('qzeros geometry/dtype')
    if not np.all(qzeros.view(np.uint32) == np.uint32(0x88888888)):
        raise ValueError('existing GPTQ importer requires directly stored zero8; no guessed +1 convention')
    if g_idx is not None and not np.array_equal(g_idx,np.arange(k)//128):
        raise ValueError('noncanonical g_idx requires an act-order path')
    signed, imported_scales, _ = importer()(qweight,qzeros,scales,128)
    if not np.array_equal(imported_scales.view(np.uint16),scales.view(np.uint16)):
        raise ValueError('import changed original scale bits')
    return (signed.astype(np.int16)+8).astype(np.uint8), imported_scales


def synthetic_checkpoint(n,k):
    """General scale-only data, not exact-dyadic or tied-group subsets."""
    if n < 128 or n % 128 or k < 128 or k % 128:
        raise ValueError('GPTQ comparison geometry')
    rng=np.random.default_rng(1731+n+3*k)
    q=rng.integers(0,16,(k,n),dtype=np.uint8)
    qw=np.zeros((k//8,n),dtype=np.uint32)
    for lane in range(8):
        qw |= q[lane::8].astype(np.uint32) << (4*lane)
    qz=np.full((k//128,n//8),0x88888888,dtype=np.uint32)
    scales=rng.uniform(.001,.025,(k//128,n)).astype(np.float16)
    return qw,qz,scales


def make_fixture(m,n,k,dtype,weights=None):
    import torch
    if dtype != torch.float16 or m<1:
        raise ValueError('existing comparison contract uses FP16 A/scales/C')
    if weights is None:
        checkpoint=synthetic_checkpoint(n,k)
        source='SYNTHETIC_GPTQ_CONTRACT_NOT_MODEL_CHECKPOINT'
    else:
        with np.load(weights,allow_pickle=False) as data:
            if set(data.files) != {'qweight','qzeros','scales','group_size','desc_act'}:
                raise ValueError('checkpoint needs exact qweight/qzeros/scales/group_size/desc_act fields')
            checkpoint=tuple(data[key].copy() for key in ('qweight','qzeros','scales'))
            if int(data['group_size'])!=128 or bool(data['desc_act']):
                raise ValueError('checkpoint quantization contract differs')
        source='ORIGINAL_GPTQ_ARRAYS'
    q,scale=decode_checkpoint(*checkpoint)
    if q.shape!=(k,n):
        raise ValueError('checkpoint shape differs; do not silently crop or reshape weights')
    signed=q.astype(np.int16)-8
    weight=(signed.astype(np.float32)*np.repeat(scale.astype(np.float32),128,axis=0)).astype(np.float16)
    rng=np.random.default_rng(3199+m+k+n)
    a=rng.uniform(-1,1,(m,k)).astype(np.float16)
    expected=(a.astype(np.float64)@weight.astype(np.float64)).astype(np.float16)
    return {'q':q,'a':torch.from_numpy(a),'expected':torch.from_numpy(expected).float(),
            'scale':torch.from_numpy(scale),'source':source,'contract':CONTRACT,
            'qweight':checkpoint[0],'qzeros':checkpoint[1]}


def marlin_metadata(value):
    import torch
    perm=torch.tensor([i+8*j for i in range(8) for j in range(8)])
    return value.reshape(-1,64)[:,perm].reshape(value.shape).contiguous()

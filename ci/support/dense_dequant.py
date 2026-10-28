"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import hashlib
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[2]


def q8_half2():
    rows=re.findall(r'^QKP_Q8_HALF2\((\d+), ([01]), (128|256)\)$',
                    (ROOT/'quactlize/prefill/dense_q8_half2.inc').read_text(),re.M)
    result={int(i):dict(fast=int(f),threads=int(t)) for i,f,t in rows}
    if len(rows)!=len(result) or set(result)!={38}:
        raise ValueError('Q8 half2 inventory changed')
    return result


def q8_paired(include_candidates=False):
    rows=re.findall(r'^QKP_Q8_PAIRED\((\d+), ([01]), (128|256), ([01])\)$',
                    (ROOT/'quactlize/prefill/dense_q8_paired.inc').read_text(),re.M)
    result={int(i):dict(signed=int(s),threads=int(t),byteperm=int(p)) for i,s,t,p in rows}
    if len(rows)!=len(result) or set(result)!={41,43} or result[41]['byteperm'] or not result[43]['byteperm']:
        raise ValueError('Q8 paired inventory changed')
    return result if include_candidates else {k:v for k,v in result.items() if not v['byteperm']}


def fixture(q, n, k, *, include_raw=False):
    import numpy as np
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    from ci.support.fixture import planes, bf16_bits
    from reference.gguf_kpack import SPECS
    block, width = (32, 34) if q == 8 else (256, SPECS[q].raw_bytes)
    rng = np.random.default_rng(71511+q+n+k)
    raw = rng.integers(0, 256, (n, k//block, width), dtype='u1')
    offset = 0 if q == 8 else SPECS[q].d_offset
    scales = rng.uniform(-.009, .009, (n, k//block)).astype('<f2')
    scales.view('<u2').flat[:8] = [0, 0x8000, 1, 0x8001, 0x3555, 0xb555, 0x400, 0x8400]
    raw[..., offset:offset+2] = scales.view('u1').reshape(n, k//block, 2)
    if q != 8 and SPECS[q].has_min:
        offset = SPECS[q].dmin_offset
        minima = rng.uniform(-.009, .009, (n, k//block)).astype('<f2')
        raw[..., offset:offset+2] = minima.view('u1').reshape(n, k//block, 2)
    packed = {name: value for name, value in planes(raw, q).items() if name in ('low', 'high', 'units')}
    gold = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(n, k)
    if not np.isfinite(gold).all() or not np.any(gold):
        raise ValueError('invalid independent GGUF fixture')
    result=packed, [gold.astype('<f2').view('<u2'), bf16_bits(gold)], dict(
        raw_sha256=hashlib.sha256(memoryview(raw)).hexdigest(),
        planes={name: hashlib.sha256(memoryview(value)).hexdigest() for name, value in packed.items()},
        oracle='INDEPENDENT_GGUF_FP32_AFFINE_THEN_ONE_RNE_F16_OR_BF16')
    return (*result,raw) if include_raw else result

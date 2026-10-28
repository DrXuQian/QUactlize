"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import numpy as np
from ci.support.fixture import bf16_bits, bf16_float
class GateFailure(ValueError):
    def __init__(self, report):
        self.report=report
        super().__init__(report['error'])


def compare_bits(got, want, storage, input, source):
    """Count every raw difference, including NaN payload/sign/quiet bits."""
    dtype='<u2' if input else '<u4'
    a=got.view(dtype).reshape(-1);b=want.view(dtype).reshape(-1)
    if a.shape!=b.shape:raise ValueError('dense IO comparison shape differs')
    mask,frac=((0x7f80,0x7f) if storage else (0x7c00,0x3ff)) if input else (0x7f800000,0x7fffff)
    an=((a&mask)==mask)&((a&frac)!=0);bn=((b&mask)==mask)&((b&frac)!=0)
    bad=a!=b;both_nan=an&bn
    src=source.view('<u4' if input else '<u2').reshape(-1)
    first=[dict(index=int(i),input_bits=hex(int(src[i])),want_bits=hex(int(b[i])),
                got_bits=hex(int(a[i])),both_nan=bool(both_nan[i])) for i in np.flatnonzero(bad)[:8]]
    return dict(raw_bad=int(np.count_nonzero(bad)),nan_encoding_bad=int(np.count_nonzero(bad&both_nan)),
                nan_class_bad=int(np.count_nonzero(an!=bn)),
                non_nan_bad=int(np.count_nonzero(bad&~(an|bn))),first=first)


def inputs(storage, input):
    bits=np.arange(65536,dtype='<u2')
    if not input:return bits
    values=bf16_float(bits) if storage else bits.view('<f2').astype('<f4')
    finite=values[np.isfinite(values)]
    positive=np.sort(finite[finite>=0])
    # Midpoints are computed in FP64 so their FP32 representation is not
    # biased by an overflowing FP32 addition near BF16's upper limit.
    mid=((positive[:-1].astype('f8')+positive[1:].astype('f8'))*.5).astype('<f4')
    corners=np.array([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,
                      0x7fc00001,0xffc12345,0x477fe000,0x477ff000,0xc77ff000,
                      0x7f7fffff,0xff7fffff],dtype='<u4').view('<f4')
    return np.concatenate((values,mid,-mid,np.nextafter(mid,np.float32(np.inf)),
                           np.nextafter(mid,np.float32(-np.inf)),corners))


def expected(source, storage, input):
    if input:
        with np.errstate(over='ignore',invalid='ignore'):
            if not storage:return source.astype('<f2').view('<u2')
            finite=np.isfinite(source)
            result=(source.view('<u4')>>16).astype('<u2')
            result[finite]=bf16_bits(source[finite])
            result[np.isnan(source)]=0x7fc0
            return result
    return bf16_float(source) if storage else source.view('<f2').astype('<f4')


def check_oracle(got, want, storage, input):
    """Independent non-NaN bits and NaN class; payload checked against IO1."""
    if input:
        mask=0x7f80 if storage else 0x7c00
        frac=0x7f if storage else 0x3ff
        isnan=lambda x: ((x&mask)==mask)&((x&frac)!=0)
    else:
        mask,frac=0x7f800000,0x7fffff
        isnan=lambda x: ((x&mask)==mask)&((x&frac)!=0)
    a=got.view('<u2' if input else '<u4');b=want.view(a.dtype)
    if not np.array_equal(isnan(a),isnan(b)) or np.any(a[~isnan(b)]!=b[~isnan(b)]):
        raise ValueError('dense IO independent IEEE cast differs')


def check_controls(comparisons, storage, input, error):
    """Keep IO4 bit-exact to IO1; disclose the observed BF16 scalar baseline."""
    if comparisons['4-vs-1']['raw_bad']:
        raise ValueError('dense IO candidate differs from old vector bits')
    observed=False
    for name in ('1-vs-0','4-vs-0'):
        d=comparisons[name]
        if not d['raw_bad']:continue
        # SDK2.1.1's existing BF16 pair widening can preserve the high
        # half's NaN payload while scalar conversion canonicalizes it.
        # This is NOT allowed for F16, narrowing, or producer-error poison.
        if (storage!=1 or input or error or d['raw_bad']!=d['nan_encoding_bad'] or
                d['nan_class_bad'] or d['non_nan_bad']):
            raise ValueError('dense IO scalar-control bits differ')
        observed=True
    return observed

"""Actual dense-IO entry: raw-bit/guard checks, never performance evidence."""
import ctypes as C
import hashlib

import numpy as np

from dev.bf16_compute.fixture import bf16_bits, bf16_float
from quactlize.runtime.native import SDK, checked
from tools.run_kpack_gemv_gate import Resources

SCOPE='ACTUAL_IO_ABI_RAW_BITS_AND_GUARDS_NO_TIMING'


class GateFailure(ValueError):
    def __init__(self, report):
        self.report=report
        super().__init__(report['error'])


def compare_bits(got, want, storage, input, source):
    """Explain raw differences without accepting any previously rejected bit."""
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
    """NaN payload follows scalar device control; finite values compare bits."""
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


def run(sdk_path, library):
    sdk=SDK(sdk_path);lib=C.CDLL(str(library),mode=C.RTLD_LOCAL)
    fn=lib.quactlize_dense_io_v1
    fn.argtypes=[C.c_int]*3+[C.c_void_p]*2+[C.c_int]*2+[C.c_int64,C.c_void_p,C.c_void_p]
    fn.restype=C.c_int
    records=[]
    def fail(message,cell,**details):
        raise GateFailure(dict(status='FAIL',error=message,first_failure=dict(cell,**details),
            cases=len(records),records=records,scope=SCOPE,performance_admitted=False))
    for storage in (0,1):
        for input in (0,1):
            data=inputs(storage,input)
            # Full vectors, partial CTA, element tail, row stride, unaligned
            # vector pointers. Scalar alignment remains valid in every case.
            cases=((1,8,8,0),(1,2056,2056,0),(3,17,23,0),(5,16,16,1),
                   (1,(len(data)+7)//8*8,(len(data)+7)//8*8,0),(1,len(data),len(data),0))
            for rows,width,stride,offset in cases:
                r=Resources(sdk)
                try:
                    count=rows*width;span=(rows-1)*stride+width
                    useful=np.resize(data,count).reshape(rows,width)
                    want=expected(useful,storage,input)
                    src_count=span if input else count;dst_count=count if input else span
                    src_item=4 if input else 2;dst_item=2 if input else 4
                    src_prefix=256+offset*src_item;dst_prefix=256+offset*dst_item
                    source=np.full(src_prefix+src_count*src_item+256,0xa5,dtype='u1')
                    src_view=source[src_prefix:-256].view('<f4' if input else '<u2')
                    for row in range(rows):
                        first=row*(stride if input else width)
                        src_view[first:first+width]=useful[row]
                    src=r.upload(source);dst_bytes=dst_prefix+dst_count*dst_item+256
                    dst=r.alloc(dst_bytes);error=r.upload(np.zeros(1,dtype='<i4'))
                    indices=np.array([row*(width if input else stride)+col
                                      for row in range(rows) for col in range(width)])
                    written=np.zeros(dst_bytes,dtype=bool)
                    for byte in range(dst_item):written[dst_prefix+indices*dst_item+byte]=True
                    for status in (None,0,1):
                        cell=dict(storage=storage,input=input,rows=rows,width=width,stride=stride,
                                  offset=offset,error=status,input_sha256=hashlib.sha256(useful.tobytes()).hexdigest())
                        r.fill(error,0 if not status else 1,4);outputs={};issues=[];oracle={}
                        for vector in (0,1,4):
                            r.fill(dst,0xa5,dst_bytes)
                            checked(fn(storage,input,vector,src+src_prefix,dst+dst_prefix,
                                       rows,width,stride,error if status is not None else None,r.stream),
                                    'dense IO gate launch')
                            sdk.synchronize(r.stream)
                            raw=np.frombuffer(sdk.download(dst,dst_bytes),dtype='u1')
                            if np.any(raw[~written]!=0xa5):
                                fail('dense IO guard/padding overwritten',cell,vector=vector,
                                     first_byte=int(np.flatnonzero((raw!=0xa5)&~written)[0]))
                            got=raw[dst_prefix:-256].view('<u2' if input else '<f4')[indices].reshape(rows,width)
                            if not input and status:
                                poisoned=np.full(got.shape,0x7fc00000,dtype='<u4').view('<f4')
                                oracle[vector]=compare_bits(got,poisoned,storage,input,useful)
                                if oracle[vector]['raw_bad']:issues.append(f'vector={vector}: producer error lost')
                            else:
                                oracle[vector]=compare_bits(got,want,storage,input,useful)
                                try:check_oracle(got,want,storage,input)
                                except ValueError as e:issues.append(f'vector={vector}: {e}')
                            outputs[vector]=got.copy()
                            records.append(dict(storage=storage,input=input,vector=vector,rows=rows,
                                width=width,stride=stride,offset=offset,error=status,guard='PASS',
                                oracle=oracle[vector],
                                sha256=hashlib.sha256(got.tobytes()).hexdigest()))
                        comparisons={f'{a}-vs-{b}':compare_bits(outputs[a],outputs[b],storage,input,useful)
                                     for a,b in ((1,0),(4,0),(4,1))}
                        if comparisons['1-vs-0']['raw_bad'] or comparisons['4-vs-0']['raw_bad']:
                            issues.append('dense IO scalar-control bits differ')
                        if issues:
                            # Complete only this cell's three variants, then stop.
                            # No scalar/old-vector/candidate comparison is waived.
                            fail('; '.join(issues),cell,oracle=oracle,comparisons=comparisons,
                                 source_unchanged=sdk.download(src,len(source))==source.tobytes())
                    if sdk.download(src,len(source))!=source.tobytes():raise ValueError('dense IO source modified')
                    if not fn(storage,input,4,src+src_prefix,src+src_prefix,rows,width,stride,None,r.stream):
                        raise ValueError('dense IO overlapping operands not rejected')
                finally:r.close()
    return dict(status='PASS',cases=len(records),records=records,
                scope=SCOPE,performance_admitted=False)

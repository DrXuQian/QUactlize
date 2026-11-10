"""Device correctness checks for the bounded PPU1.5 bring-up, not performance."""
import ctypes as C
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))

import numpy as np
from ci.support.dense import Arrangement, Sizes
from ci.support.fixture import bf16_bits, round_compute, matmul_reference, compare
from ci.support.graph import Replay
from ci.support.graph_api import graph_bind
from quactlize.execution.native import Call, SimtCallV2
from quactlize.runtime.native import SDK, checked
from ppu.support import Arena, bind, packed, QTYPES

MMA_M=tuple(range(1,9))+(9,16,128)
FIELDS=('ordinal','major','minor','compute_units','warp_size','max_threads_per_block',
        'max_threads_per_cu','shared_per_block','shared_optin_per_block','shared_per_cu',
        'registers_per_block','registers_per_cu','driver_version','runtime_version')

class Pack:
    def __init__(self, lib):
        self.canonical=bind(lib,'quactlize_ppu_kpack_canonical_arrangement_v1',[C.c_int,C.POINTER(Arrangement)])
        self.sizes=bind(lib,'quactlize_ppu_kpack_sizes_for_arrangement_v1',
                        [C.c_int]*4+[C.POINTER(Arrangement),C.POINTER(Sizes)])
        self.pack=bind(lib,'quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2',
                       [C.c_void_p]*4+[C.c_int]*4+[C.POINTER(Arrangement),C.c_void_p])
        self.pack_pair=bind(lib,'quactlize_ppu_prepare_gate_up_dev_for_arrangement_v1',
                            [C.c_void_p]*5+[C.c_int]*4+[C.POINTER(Arrangement),C.c_void_p])

def setup(sdk_path, output):
    sdk=SDK(sdk_path)
    graph_bind(sdk)
    bind(sdk.lib,'hggcStreamCreateWithFlags',[C.POINTER(C.c_void_p),C.c_uint])
    bind(sdk.lib,'hggcStreamDestroy',[C.c_void_p])
    lib=C.CDLL(str(output/'build/libqz15_probe.so'),mode=C.RTLD_LOCAL)
    device=bind(lib,'qz15_device',[C.c_char_p,C.c_char_p,C.POINTER(C.c_int64)])
    name,pci,values=C.create_string_buffer(128),C.create_string_buffer(64),(C.c_int64*len(FIELDS))()
    checked(device(name,pci,values),'PPU1.5 device probe')
    identity=dict(zip(FIELDS,values),name=name.value.decode(),pci=pci.value.decode())
    print('ZW890_DEVICE '+json.dumps(identity),flush=True)
    if 'ZW890' not in identity['name'] or identity['warp_size']!=32:
        raise ValueError('this bring-up requires one visible ZW890 with warp size32; got '+repr(identity))
    return sdk,lib,identity

def mma(sdk,lib,emit):
    run=bind(lib,'qz15_mma',[C.c_void_p]*4+[C.c_int]*4+[C.c_void_p])
    for bf16 in (0,1):
        for m in MMA_M:
            for k in (16,64):
                n=33 # include a non-tile-aligned output tail
                with Arena(sdk) as arena:
                    a=arena.buffer(m*k*2); b=arena.buffer(n*k*2)
                    c=arena.buffer(m*n*4); out=arena.buffer(m*n*4)
                    stream=arena.stream()
                    launch=lambda:run(a.ptr,b.ptr,c.ptr,out.ptr,m,n,k,bf16,stream)
                    graph=None
                    try:
                        worst=0.
                        for replay in range(3):
                            rng=np.random.default_rng(1700+m*17+k+replay)
                            av=rng.uniform(-1,1,(m,k)).astype('f4')
                            bv=rng.uniform(-1,1,(n,k)).astype('f4')
                            cv=rng.uniform(-.1,.1,(m,n)).astype('f4')
                            if replay==0: # one-hot activation isolates every logical weight row
                                av[:]=0
                                av[np.arange(m),np.arange(m)%k]=1
                            if bf16 and replay==2: av*=100000
                            compute='bf16' if bf16 else 'f16'
                            aa=round_compute(av,compute);bb=round_compute(bv,compute)
                            a.write(bf16_bits(av) if bf16 else av.astype('f2'))
                            b.write(bf16_bits(bv) if bf16 else bv.astype('f2'))
                            c.write(cv);out.poison();arena.sync()
                            if replay==1: graph=Replay(sdk,stream,launch,1)
                            checked(graph() if graph else launch(),'native1.5 MMA')
                            arena.sync()
                            got=np.frombuffer(out.bytes(),'f4').reshape(m,n)
                            gold=aa.astype('f8')@bb.astype('f8').T+cv
                            denom=np.abs(aa.astype('f8'))@np.abs(bb.astype('f8')).T+np.abs(cv)
                            result=compare(got,gold,denom,tolerance=5e-6)
                            worst=max(worst,result['relative_l1_error'])
                            a.guard();b.guard();c.guard()
                        emit(dict(suite='mma',compute=compute,m=m,n=n,k=k,replays=2,
                                  max_absdot_error=worst,status='PASS'))
                    finally:
                        arena.sync()
                        if graph:graph.close()

def delivery(sdk,lib,emit):
    run=bind(lib,'qz15_delivery',[C.c_void_p]*2+[C.c_int]*3+[C.c_void_p])
    for width in (32,64):
        for trans in (0,1):
            for padding in (0,32):
                leading=width+padding
                with Arena(sdk) as arena:
                    source=arena.buffer(3*16*leading*2)
                    out=arena.buffer(3*16*width*2)
                    stream=arena.stream();graph=None
                    launch=lambda:run(source.ptr,out.ptr,width,trans,leading,stream)
                    try:
                        for replay in range(3):
                            tags=((np.arange(3*16*leading,dtype='u4')*37+replay*101)&65535).astype('u2').reshape(3,16,leading)
                            source.write(tags);out.poison();arena.sync()
                            if replay==1:graph=Replay(sdk,stream,launch,1)
                            checked(graph() if graph else launch(),'native1.5 AIU/TSM pair')
                            arena.sync()
                            expected=tags[:,:,:width]
                            if trans:expected=expected.transpose(0,2,1)
                            got=np.frombuffer(out.bytes(),'u2').reshape(expected.shape)
                            np.testing.assert_array_equal(got,expected)
                            source.guard()
                        emit(dict(suite='delivery',orientation='B-transposed' if trans else 'A',
                                  width=width,leading=leading,shared_stages=3,replays=2,status='PASS'))
                    finally:
                        arena.sync()
                        if graph:graph.close()

def pack(sdk,lib,emit):
    public=Pack(lib)
    for q in QTYPES:
        for pair in (False,True):
            with Arena(sdk) as arena:
                packed(arena,public,q,experts=3,pair=pair,bit_patterns=True)
            emit(dict(suite='pack',q=q,experts=3,pair=pair,status='PASS'))

def simt(sdk,lib,emit):
    run=bind(lib,'qz15_simt',[C.POINTER(SimtCallV2),C.c_int,C.POINTER(Arrangement)])
    public=Pack(lib)
    for q in (8,14):
        for mode in (0,2):
            # Pack once per scope. Per-call buffers/streams are owned by a child
            # arena so increasing M does not retain previous scratch allocations.
            with Arena(sdk) as weights:
                e=3 if mode else 1;top=2 if mode else 1;n,k=256,512
                arr,_,pointers,gold=packed(weights,public,q,n=n,k=k,experts=e)
                for compute in (0,1):
                    for m in range(1,9):
                        for split in (1,4):
                            with Arena(sdk) as arena:
                                rows=m*top;stride=k+8;out_stride=n+8
                                a=arena.buffer(m*stride*4);out=arena.buffer(rows*out_stride*4)
                                ids=arena.buffer(rows*4) if mode else None
                                workspace=arena.buffer(rows*n*split*4) if split>1 else None
                                stream=arena.stream()
                                call=Call(version=1,size=C.sizeof(Call),qtype=q,n=n,k=k,experts=e,
                                    rows=rows,mode=mode,input_type=1,channels=1,topk=top,
                                    a_row_stride=stride,a_token_stride=stride,ids_stride=top,out_row_stride=out_stride,
                                    a=a.ptr,low=pointers[0],high=pointers[1],units=pointers[2],
                                    offsets=None,ids=ids.ptr if ids else None,output=out.ptr,
                                    workspace=workspace.ptr if workspace else None,
                                    workspace_bytes=workspace.size if workspace else 0,stream=stream)
                                typed=SimtCallV2(call,compute);graph=None;worst=0.
                                launch=lambda:run(C.byref(typed),split,C.byref(arr))
                                try:
                                    for replay in range(3):
                                        source=np.random.default_rng(q*101+m+replay).uniform(-.13,.13,(m,stride)).astype('f4')
                                        selected=((np.arange(rows)+replay)%e).astype('i4')
                                        a.write(source);out.poison()
                                        if ids:ids.write(selected)
                                        arena.sync()
                                        if replay==1:graph=Replay(sdk,stream,launch,1)
                                        checked(graph() if graph else launch(),'existing SIMT body on1.5')
                                        arena.sync()
                                        got=np.frombuffer(out.bytes(),'f4').reshape(rows,out_stride)
                                        if not np.all(got[:,n:].view('u4')==0xffffffff):
                                            raise ValueError('SIMT output stride padding overwritten')
                                        inputs=np.repeat(source[:,:k],top,axis=0)
                                        for expert in range(e):
                                            mask=selected==expert
                                            if not np.any(mask):continue
                                            want,denom=matmul_reference(inputs[mask],gold[expert],
                                                'bf16' if compute else 'f16')
                                            result=compare(got[mask,:n],want,denom)
                                            worst=max(worst,result['relative_l1_error'])
                                        a.guard()
                                        if ids:ids.guard()
                                        if workspace:workspace.guard()
                                    emit(dict(suite='simt',q=q,mode=mode,compute=compute,m=m,n=n,k=k,
                                              split=split,replays=2,max_absdot_error=worst,status='PASS'))
                                finally:
                                    arena.sync()
                                    if graph:graph.close()

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=('device','mma','delivery','pack','simt'),required=True)
    args=p.parse_args()
    sdk,lib,identity=setup(args.sdk,args.output)
    (args.output/'device.json').write_text(json.dumps(identity,indent=2)+'\n')
    if args.suite=='device':return
    with (args.output/(args.suite+'.jsonl')).open('x') as log:
        def emit(record):
            line=json.dumps(record)
            log.write(line+'\n');log.flush()
            print('ZW890_CASE '+line,flush=True)
        globals()[args.suite](sdk,lib,emit)

if __name__=='__main__':main()

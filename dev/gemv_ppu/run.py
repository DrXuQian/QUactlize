#!/usr/bin/env python3
"""Four-arm PPU Q4 comparison, isolated children and resumable validated cells."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import digest
from quactlize.execution.native import Call as VecCall, Config, Sizes, Arrangement, arrangement
from quactlize.runtime.native import SDK, Call, checked
from quactlize.dispatch.native import Dispatch, receipt as choice_receipt
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_pack_gate import device_identity
from tools.verify_kpack_dispatch import verify as verify_native
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command

SHAPES=((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120))
ARMS=("xplane","old","new","fq")


def query_l2_attribute(lib):
    # Native SDK 2.1.1 driver_types.h: hggcDevAttrL2CacheSize=38.
    # This returns one integer, independently of DeviceProperties struct ABI.
    try:fn=lib.hggcDeviceGetAttribute
    except AttributeError:return dict(status="NOT_EXPORTED",bytes=0)
    fn.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_int];fn.restype=C.c_int
    value=C.c_int(-1)
    rc=fn(C.byref(value),38,0)
    if rc:
        # Optional-query rejection is not a kernel failure. Clear only these
        # documented rejections; propagate device and unexpected runtime errors.
        unsupported=(1,801,998)
        checked(rc if rc not in unsupported else 0,"L2 attribute query")
        clear=lib.hggcGetLastError;clear.argtypes=[];clear.restype=C.c_int
        deferred=clear()
        checked(deferred if deferred not in (0,*unsupported) else 0,"L2 attribute deferred error")
        return dict(status=rc,bytes=0)
    if value.value==-1:return dict(status="UNWRITTEN",bytes=0)
    if value.value<0:raise ValueError("negative L2 attribute")
    return dict(status=0,bytes=value.value)


def resolve_l2(reported,override,attribute):
    if min(reported,override,attribute["bytes"])<0:raise ValueError("negative L2 capacity")
    capacity=override or reported or attribute["bytes"]
    if capacity<=0:
        raise ValueError(f"SDK reports no L2 capacity: properties={reported}, attribute={attribute}; "
                         "set L2_BYTES to a confirmed capacity in bytes")
    source="EXPLICIT_OVERRIDE" if override else "DEVICE_PROPERTIES" if reported else "DEVICE_ATTRIBUTE_38"
    return dict(l2_bytes=capacity,reported_l2_bytes=reported,l2_override=bool(override),
                l2_source=source,l2_attribute=attribute)


def configs(arm):
    if arm=="fq":return [(0,0,0)] # The production selector owns this recipe.
    if arm=="xplane":return [(c,w,1) for c in (1,2,4,8) for w in (2,4,8)]
    if arm=="old":return [(c,w,s) for c in (16,32) for w in (2,4,8) for s in (1,2,4,8)]
    if arm=="new":
        return [(c,w,s) for c in (1,2,4,8,16,32) for w in (2,4,8,16) for s in (1,2,4,8)]+[
            (c,w,1) for c in (1,2,4,8,16,32) for w in (5,10)]
    raise ValueError("unknown arm")


def verify_bundle(bundle, *, sources=True):
    m=json.loads((bundle/"manifest.json").read_text())
    if m.get("schema")!="quactlize.q4-simt-ppu-comparison.v1" or set(m["payloads"])!={"old","new","xplane"}:
        raise ValueError("not the PPU Q4 comparison package")
    for arm,r in m["payloads"].items():
        if r["file"]!=f"libq4_ppu_{arm}.so" or digest(bundle/r["file"])!=r["sha256"]:
            raise ValueError("missing/LFS-pointer/changed PPU payload: "+arm)
    for name,want in (m["source_hashes"].items() if sources else []):
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or digest(path)!=want:
            raise ValueError("comparison kernel source differs: "+name)
    return m


def load_library(args,arm):
    m=verify_bundle(args.bundle,sources=False)
    lib=C.CDLL(str(args.bundle/m["payloads"][arm]["file"]),mode=C.RTLD_LOCAL)
    probe=lib.q4_ppu_probe
    probe.argtypes=[C.POINTER(C.c_int)]*3+[C.c_char_p];probe.restype=C.c_int
    l2,sm,warp=C.c_int(),C.c_int(),C.c_int();name=C.create_string_buffer(256)
    checked(probe(C.byref(l2),C.byref(sm),C.byref(warp),name),"real PPU image/marker probe: "+arm)
    if warp.value<=0 or sm.value<=0:raise ValueError("invalid reported device warp/SM geometry")
    attribute=query_l2_attribute(lib) if l2.value==0 else dict(status="NOT_NEEDED",bytes=0)
    identity=dict(name=name.value.decode(),sm=sm.value,warp=warp.value,
                  **resolve_l2(l2.value,args.l2_bytes,attribute))
    print("Q4_PPU_DEVICE "+json.dumps(identity),flush=True)
    return lib,identity


def read_fixture(path):
    with np.load(path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    n,k=int(data["n"]),int(data["k"])
    if (n,k) not in SHAPES or any(int(data[x])!=v for x,v in (("q",12),("experts",1),("channels",1))):
        raise ValueError("unexpected fixture shape or format")
    if data["ids"].tolist()!=[0] or data["high"].size or data["low"].nbytes!=n*k//2 or data["units"].nbytes!=n*k//16:
        raise ValueError("fixture plane extent differs")
    a=data["a"].astype("<f2")
    if not np.array_equal(a.astype("<f4"),data["a"]) or not np.isfinite(a).all():
        raise ValueError("fixture A is not finite and FP16-exact")
    data["a"]=a
    return n,k,data


def check_reduction(partials, output):
    if not np.isfinite(partials).all():raise ValueError("unwritten/nonfinite SIMT partial")
    total=np.zeros(output.shape,dtype="<f4")
    for part in partials:np.add(total,part,out=total)
    if not np.array_equal(total.view("<u4"),output.view("<u4")):
        raise ValueError("SIMT ordered reducer differs from FP32 partials")


class Bench:
    def __init__(self,args):
        self.args=args;self.sdk=SDK(args.sdk);graph_bind(self.sdk)
        self.device=device_identity(self.sdk)
        self.lib,probe=load_library(args,args.arm if args.arm!="fq" else "new")
        self.device.update(probe)
        self.n,self.k,self.data=read_fixture(args.fixture)
        n,k=self.n,self.k
        self.r=Resources(self.sdk);self.arr=arrangement(12)
        self.weight_bytes=n*k//2+n*k//16
        self.copies=1 if args.mode=="warm" else max(2,(9*probe["l2_bytes"]+4*self.weight_bytes-1)//(4*self.weight_bytes))
        if self.copies*self.weight_bytes>2**31:raise ValueError("cold allocation exceeds bounded diagnostic capacity")
        low=self.data["low"].reshape(-1).view("u1");units=self.data["units"].reshape(-1)
        if args.arm=="xplane":
            packed=np.empty(n*k//2,dtype="u1");metadata=np.empty(n*k//16,dtype="u1")
            fn=self.lib.q4_xplane_pack
            fn.argtypes=[C.c_int,C.c_int]+[C.c_void_p]*3;fn.restype=C.c_int
            checked(fn(n,k,self.data["raw"].ctypes.data,packed.ctypes.data,metadata.ctypes.data),"Xplane round trip")
            if not np.array_equal(metadata,units):raise ValueError("Xplane metadata differs")
            low=packed
        self.host_low=np.ascontiguousarray(low)
        self.low=self.r.alloc(low.nbytes*self.copies);self.units=self.r.alloc(units.nbytes*self.copies)
        self.weight_pointers=[]
        for i in range(self.copies):
            lp,up=self.low+i*low.nbytes,self.units+i*units.nbytes
            checked(self.sdk.lib.hggcMemcpy(lp,low.ctypes.data,low.nbytes,1),"low upload")
            checked(self.sdk.lib.hggcMemcpy(up,units.ctypes.data,units.nbytes,1),"unit upload")
            self.weight_pointers.append((lp,up))
        self.a=self.r.upload(self.data["a"])
        self.sdk.synchronize(None) # All default-stream uploads precede the nonblocking consumer.
        self.dtype="<f2" if args.arm=="fq" else "<f4"
        self.output_bytes=n*np.dtype(self.dtype).itemsize
        self.output_base=self.r.alloc(self.output_bytes+32);self.output=self.output_base+16
        self.dispatch=None;self.choice=None;self.calls=[];self.handles=[]
        if args.arm=="fq":
            verify_native(args.native_bundle)
            self.dispatch=Dispatch(args.native_bundle,jit=dict(python=sys.executable,helper=ROOT/"tools/kpack_jit.py",sdk=args.sdk,cache=args.jit_cache))
            self.choice=self.dispatch.query(12,0,1,n,k,1,1,self.arr.mapping_id)
            if self.choice is None:raise ValueError("current FQ selector has no parent: "+str(self.dispatch.last_miss))
        self.workspace_bytes=self.choice.workspace_bytes if self.choice else n*8*4
        self.workspace_base=self.r.alloc(self.workspace_bytes+256);self.workspace=self.workspace_base+128
        if self.choice:
            for lp,up in self.weight_pointers:
                call=Call(version=1,size=C.sizeof(Call),m=1,n=n,k=k,experts=1,group_size=32,
                    device=self.choice.device,compute_units=self.choice.compute_units,mapping_id=self.arr.mapping_id,
                    a=self.a,low=lp,metadata=up,output=self.output,workspace=self.workspace,
                    workspace_bytes=self.workspace_bytes,stream=self.r.stream.value)
                self.handles.append(self.dispatch.prepare(self.choice,call));self.calls.append(call)
        elif args.arm=="xplane":
            self.launch=self.lib.q4_xplane_run
            self.launch.argtypes=[C.c_int]*4+[C.c_void_p]*5;self.launch.restype=C.c_int
        else:
            self.query=self.lib.quactlize_kpack_gemv_pair_query_v1
            self.query.argtypes=[C.POINTER(VecCall),C.POINTER(Config),C.POINTER(Arrangement),C.POINTER(Sizes)]
            self.query.restype=C.c_int
            self.launch=self.lib.quactlize_kpack_gemv_pair_run_v1
            self.launch.argtypes=[C.POINTER(VecCall),C.POINTER(Config),C.POINTER(Arrangement)];self.launch.restype=C.c_int
            self.call=VecCall(version=1,size=C.sizeof(VecCall),qtype=12,n=n,k=k,experts=1,rows=1,
                mode=0,input_type=0,channels=1,topk=1,a_row_stride=k,a_token_stride=k,ids_stride=1,out_row_stride=n,
                a=self.a,output=self.output,workspace=self.workspace,workspace_bytes=self.workspace_bytes,stream=self.r.stream.value)

    def error(self):
        self.sdk.synchronize(self.r.stream)
        data=self.sdk.download(self.output_base,self.output_bytes+32)
        if data[:16]!=b"\xff"*16 or data[-16:]!=b"\xff"*16:raise ValueError("output guard changed")
        got=np.frombuffer(data[16:-16],dtype=self.dtype).astype("f8")
        gold=self.data["golden"].reshape(-1);denom=self.data["denom"].reshape(-1)
        if not np.isfinite(got).all():raise ValueError("unwritten/nonfinite output")
        return float(np.max(np.abs(got-gold)/np.maximum(denom,1e-30)))

    def measure(self,recipe):
        c,w,s=recipe;cfg=Config(c,w,s);n,k=self.n,self.k
        self.r.fill(self.output_base,0xff,self.output_bytes+32)
        self.r.fill(self.workspace_base,0xff,self.workspace_bytes+256)
        def launch(index):
            index%=self.copies
            if self.handles:return self.handles[index]()
            lp,up=self.weight_pointers[index]
            if self.args.arm=="xplane":return self.launch(c,w,n,k,self.a,lp,up,self.output,self.r.stream)
            self.call.low,self.call.units=lp,up
            return self.launch(C.byref(self.call),C.byref(cfg),C.byref(self.arr))
        if self.args.arm in ("old","new"):
            sizes=Sizes()
            checked(self.query(C.byref(self.call),C.byref(cfg),C.byref(self.arr),C.byref(sizes)),"SIMT recipe query")
            if sizes.workspace_bytes>self.workspace_bytes:raise ValueError("workspace query exceeds allocation")
        checked(launch(0),"correctness launch")
        error=self.error()
        if not math.isfinite(error) or error>=.005:raise ValueError(f"independent GGUF dot failed: {error:.8g}")
        # Same immutable input oracle also needs to reject a blank code plane.
        self.r.fill(self.low,0,self.host_low.nbytes)
        checked(launch(0),"zero-code negative")
        if self.error()<=.005:raise ValueError("zero-code negative was not detected")
        checked(self.sdk.lib.hggcMemcpy(self.low,self.host_low.ctypes.data,self.host_low.nbytes,1),"restore codes")
        self.sdk.synchronize(None)
        for _ in range(5):
            for i in range(self.copies):checked(launch(i),"cache setup")
        self.sdk.synchronize(self.r.stream)
        # A graph replay must end on a whole ring, otherwise the next replay
        # can immediately revisit its last weights and contaminate cold timing.
        calls=max(2,(32+self.copies-1)//self.copies)*self.copies;counter=0
        def next_call():
            nonlocal counter
            rc=launch(counter);counter+=1;return rc
        samples=[]
        if self.args.profile:
            with AcuRange(self.sdk):
                checked(launch(0),"ACU selected call");self.sdk.synchronize(self.r.stream)
        else:
            graph=Replay(self.sdk,self.r.stream,next_call,calls)
            try:
                self.r.samples(graph,5) # upload/first launch excluded
                samples=[t/calls for t in self.r.samples(graph,self.args.samples)]
            finally:graph.close()
        error=max(error,self.error())
        if error>=.005:raise ValueError("post-graph independent oracle failed")
        ws=self.sdk.download(self.workspace_base,self.workspace_bytes+256)
        if ws[:128]!=b"\xff"*128 or ws[-128:]!=b"\xff"*128:raise ValueError("workspace guard changed")
        if self.args.arm in ("old","new"):
            used=n*s*4 if s>1 else 0
            if ws[128+used:-128]!=b"\xff"*(self.workspace_bytes-used):
                raise ValueError("SIMT wrote outside its queried workspace")
            if s>1:
                parts=np.frombuffer(ws[128:128+used],dtype="<f4").reshape(s,n)
                out=np.frombuffer(self.sdk.download(self.output,self.output_bytes),dtype="<f4")
                check_reduction(parts,out)
        return dict(status="PASS",arm=self.args.arm,shape=[1,n,k],mode=self.args.mode,recipe=recipe,
            selection=choice_receipt(self.choice) if self.choice else dict(columns=c,warps=w,split=s),
            error=error,zero_code_negative="PASS",device=self.device,copies=self.copies,
            weight_bytes=self.weight_bytes,calls_per_graph=calls,samples_us=samples,
            median_us=statistics.median(samples) if samples else None,
            timing_scope="RESIDENT_COMPLETE_CALL_INCLUDING_REDUCER_NO_HOST_PACK_OR_JIT",
            reducer_check="ORDERED_FP32" if self.args.arm in ("old","new") and s>1 else "NOT_APPLICABLE",
            output_type="F16" if self.choice else "F32")

    def close(self):
        self.sdk.synchronize(self.r.stream)
        if self.dispatch:self.dispatch.close()
        self.r.close()


def child(args):
    bench=None;records=[]
    try:
        bench=Bench(args)
        for recipe in json.loads(args.recipes):
            if tuple(recipe) not in configs(args.arm):raise ValueError("recipe outside arm inventory")
            try:r=bench.measure(recipe)
            except Exception as exc:
                traceback.print_exc();r=dict(status="FAIL",arm=args.arm,recipe=recipe,error=str(exc))
            records.append(r)
            print("Q4_PPU_CELL "+json.dumps(r),flush=True)
    finally:
        if bench:bench.close()
    return 0 if records and all(r["status"]=="PASS" for r in records) else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk",type=Path,required=True)
    p.add_argument("--bundle",type=Path,default=ROOT/"prebuilt/ppu0010/q4-simt-ab-v1")
    p.add_argument("--native-bundle",type=Path,default=ROOT/"prebuilt/ppu0010/kpack-jit-v2")
    p.add_argument("--jit-cache",type=Path,required=True)
    p.add_argument("--fixtures",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--l2-bytes",type=int,default=0)
    p.add_argument("--acu",type=Path)
    p.add_argument("--skip-acu",action="store_true")
    p.add_argument("--child",action="store_true")
    p.add_argument("--fixture",type=Path)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--mode",choices=("warm","rotating"))
    p.add_argument("--recipes")
    p.add_argument("--samples",type=int,default=15)
    p.add_argument("--profile",action="store_true")
    args=p.parse_args()
    args.sdk=args.sdk.resolve(strict=True);args.bundle=args.bundle.resolve(strict=True)
    args.native_bundle=args.native_bundle.resolve(strict=True);args.jit_cache=args.jit_cache.resolve()
    if args.l2_bytes<0 or args.samples<3:p.error("invalid L2 size or sample count")
    if args.child:return child(args)
    from dev.gemv_ppu.campaign import run
    return run(args)


if __name__=="__main__":
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)

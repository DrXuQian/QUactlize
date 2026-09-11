#!/usr/bin/env python3
"""Matched Q4 Xplane A64 vs K-pack N2, with independently selected recipes.

Development-only: both arms use F16 A and F32 output, but the historical
Xplane reader has FP16 intra-group accumulation whereas K-pack uses FP32.
This is an implementation comparison, not a layout-only causal experiment.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.bench import bind, checked, error
from quactlize.execution.native import Call, Config, Sizes, arrangement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def n2_authority(library, expected_reader="cuda-n2"):
    manifest_path = library.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    readers = ("cuda-n2", "cuda-q4-n2", "cuda-q4-n2-wide", "cuda-q4-n2-shared", "cuda-q4-n2-half-control", "cuda-q4-shm-fp32", "cuda-q4-n2-grid", "cuda-q4-n2-aligned", "cuda-q4-n2-metadata", "cuda-q4-n4", "cuda-q4-n4-unsigned", "cuda-q4-n4-tree", "cuda-q4-n2-tree", "cuda-q4-n2-warp", "cuda-q4-n4-coop", "cuda-q4-small-static", "cuda-q4-small-balanced")
    readers += ("cuda-q4-n4-static",)
    if (expected_reader not in readers or manifest.get("reader") != expected_reader or
            manifest.get("pair_column_values_per_thread", 2) != 2 or
            manifest.get("library") != library.name or
            manifest.get("library_sha256") != sha(library)):
        raise ValueError("K-pack library is not the manifest-bound N2 reader")
    columns = [1,2,4,8,16,32] if expected_reader.startswith(("cuda-q4-n4","cuda-q4-small")) or expected_reader in ("cuda-q4-n2-tree","cuda-q4-n2-warp") else [4, 8, 16, 32] if expected_reader in readers[2:] else [16, 32]
    if manifest.get("q4_n_positions", [16, 32]) != columns:
        raise ValueError("manifest-bound N2 column domain differs")
    warps=[2,4,8,16] if expected_reader in ("cuda-q4-n4-tree","cuda-q4-n2-tree","cuda-q4-n2-warp","cuda-q4-n4-coop","cuda-q4-small-static","cuda-q4-small-balanced") else [2,4,8]
    if expected_reader=="cuda-q4-n4-static": warps=[2,4,8,16]
    if manifest.get("q4_warps",[2,4,8])!=warps:
        raise ValueError("manifest-bound N2 warp domain differs")
    extra_s1=[5,10] if expected_reader in ("cuda-q4-small-balanced","cuda-q4-n4-static") else []
    if manifest.get("q4_s1_extra_warps",[])!=extra_s1:
        raise ValueError("manifest-bound S1 warp domain differs")
    return dict(reader=expected_reader, columns=columns,warps=warps,extra_s1=extra_s1, manifest_sha256=sha(manifest_path),
                library_sha256=manifest["library_sha256"])


def xplane_authority(library, arithmetic):
    if arithmetic == "fp16-group":
        return dict(arithmetic=arithmetic, library_sha256=sha(library))
    if arithmetic != "fp32":
        raise ValueError("unknown Xplane arithmetic")
    receipt_path = library.parent.parent / "manifest.json"
    receipt = json.loads(receipt_path.read_text())
    matches = [r for r in receipt["arms"] if r["arm"] == "fp32-control"]
    if len(matches) != 1 or matches[0]["library_sha256"] != sha(library):
        raise ValueError("Xplane FP32 arithmetic control identity differs")
    return dict(arithmetic=arithmetic, library_sha256=sha(library),
                manifest_sha256=sha(receipt_path), generated_header_sha256=matches[0]["generated_header_sha256"])


def compare(path, xplane, kpack, l2_bytes, columns=(16, 32), warps=(2,4,8), extra_s1=()):
    f = np.load(path, allow_pickle=False)
    q, n, k, experts = (int(f[x]) for x in ("q", "n", "k", "experts"))
    if (q, experts, len(f["ids"])) != (12, 1, 1):
        raise ValueError("this comparison requires dense Q4 M1")
    low = np.zeros(n*k//2, dtype="u1")
    units = np.zeros(n*k//256*16, dtype="u1")
    raw = np.ascontiguousarray(f["raw"], dtype="u1")
    checked(xplane.q4_xplane_pack(n, k, raw.ctypes.data, low.ctypes.data, units.ctypes.data))
    if not np.array_equal(units, f["units"].view("u1").reshape(-1)):
        raise ValueError("Xplane and K-pack metadata bytes differ")
    # Same raw GGUF weights, same packed metadata and exactly representable A.
    a_host = np.ascontiguousarray(f["a"], dtype="f2")
    if not np.array_equal(a_host.astype("f4"), f["a"]):
        raise ValueError("fixture activation is not exactly F16 representable")
    a = torch.from_numpy(a_host).cuda()
    one_bytes = low.nbytes + units.nbytes
    cold_copies = math.ceil(2.25*l2_bytes/one_bytes)
    result = dict(shape=[1,n,k], fixture_sha256=sha(path), weight_bytes=one_bytes,
                  l2_bytes=l2_bytes, cold_copies=cold_copies, modes={})
    query, krun = bind(kpack, "pair")
    arr = arrangement(12)

    for mode, copies in (("warm",1), ("rotating",cold_copies)):
        storage = {}
        for name, data in (("xplane",low), ("kpack",f["low"]), ("units",units)):
            host = np.ascontiguousarray(data).view("u1").reshape(1,-1)
            storage[name] = torch.from_numpy(host).cuda().repeat(copies,1)
        out = torch.full((n+8,), float("nan"), device="cuda")
        ws = torch.full((n*8+8,), float("nan"), device="cuda")
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        base = Call(version=1, size=C.sizeof(Call), qtype=12, n=n, k=k,
                    experts=1, rows=1, mode=0, input_type=0, channels=1, topk=1,
                    a_row_stride=k, a_token_stride=k, ids_stride=1, out_row_stride=n,
                    a=a.data_ptr(), output=out.data_ptr()+16, stream=stream.cuda_stream)

        def factory(arm, c, w, s=1):
            if arm == "xplane":
                def launch(copy):
                    return xplane.q4_xplane_run(c,w,n,k,a.data_ptr(),
                        storage[arm][copy].data_ptr(), storage["units"][copy].data_ptr(),
                        out.data_ptr()+16, stream.cuda_stream)
            else:
                config, sizes = Config(c,w,s), Sizes()
                calls = []
                for copy in range(copies):
                    call = Call.from_buffer_copy(base)
                    call.low = storage[arm][copy].data_ptr()
                    call.units = storage["units"][copy].data_ptr()
                    checked(query(C.byref(call),C.byref(config),C.byref(arr),C.byref(sizes)))
                    call.workspace = ws.data_ptr()+16 if sizes.workspace_bytes else None
                    call.workspace_bytes = sizes.workspace_bytes
                    calls.append(call)
                def launch(copy):
                    return krun(C.byref(calls[copy]),C.byref(config),C.byref(arr))
            return launch

        def check_output(split):
            stream.synchronize()
            got = out.cpu().numpy()
            if not np.isnan(got[:4]).all() or not np.isnan(got[-4:]).all():
                raise ValueError("output guard changed")
            err = error(got[4:-4].reshape(1,n),f["golden"],f["denom"])
            parts = ws.cpu().numpy()
            if not np.isnan(parts[:4]).all() or not np.isnan(parts[-4:]).all():
                raise ValueError("workspace guard changed")
            if split > 1:
                total = np.zeros(n,dtype="f4")
                for part in parts[4:4+split*n].reshape(split,n):
                    np.add(total,part,out=total)
                if not np.array_equal(total.view("u4"),got[4:-4].view("u4")):
                    raise ValueError("ordered reducer mismatch")
            return err

        def verify(fn, split):
            with torch.cuda.stream(stream):
                out.fill_(float("nan"))
                ws.fill_(float("nan"))
            checked(fn(0))
            return check_output(split)

        def graph_for(fn):
            calls = max(32, copies*2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for i in range(calls):
                    checked(fn(i%copies))
            return graph,calls

        def measure(graph,calls,samples):
            values=[]
            with torch.cuda.stream(stream):
                for _ in range(5):
                    graph.replay()
                for _ in range(samples):
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    graph.replay()
                    end.record(stream)
                    end.synchronize()
                    value = start.elapsed_time(end)*1000/calls
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("invalid CUDA event duration")
                    values.append(value)
            return values

        cells=[]
        recipes = [("xplane",c,w,1) for c in (1,2,4,8) for w in (2,4,8)]
        recipes += [("kpack",c,w,s) for c in columns for w in warps for s in (1,2,4,8)]
        recipes += [("kpack",c,w,1) for c in columns for w in extra_s1]
        # Interleave arm blocks to reduce a systematic warmup/thermal ordering bias.
        recipes = sorted(recipes,key=lambda x:(x[2],x[3],x[0],x[1]))
        for arm,c,w,s in recipes:
            fn=factory(arm,c,w,s)
            err=verify(fn,s)
            graph,calls=graph_for(fn)
            values=measure(graph,calls,5)
            check_output(s)
            cell=dict(arm=arm,columns=c,warps=w,split=s,kernels=1+(s>1),error=err,
                      median_us=statistics.median(values),samples_us=values)
            cells.append(cell)
            print("XPLANE_KPACK_SCREEN "+json.dumps(dict(shape=[1,n,k],mode=mode,**cell)),flush=True)
            del graph

        groups = {"xplane_best":[x for x in cells if x["arm"]=="xplane"],
                  "kpack_s1_best":[x for x in cells if x["arm"]=="kpack" and x["split"]==1],
                  "kpack_best":[x for x in cells if x["arm"]=="kpack"],
                  "xplane_historical":[x for x in cells if (x["arm"],x["columns"],x["warps"])==("xplane",2,4)]}
        winners={name:min(rows,key=lambda x:x["median_us"]) for name,rows in groups.items()}
        graphs={}
        for name,cell in winners.items():
            fn=factory(cell["arm"],cell["columns"],cell["warps"],cell["split"])
            # A zero-code negative must change this independent oracle.
            plane=storage[cell["arm"]][0]
            with torch.cuda.stream(stream):
                saved=plane.clone()
                plane.zero_()
            try:
                verify(fn,cell["split"])
            except ValueError as exc:
                if "independent official GGUF dot mismatch" not in str(exc):
                    raise
            else:
                raise ValueError("zero-code negative was not rejected")
            with torch.cuda.stream(stream):
                plane.copy_(saved)
            verify(fn,cell["split"])
            graphs[name]=graph_for(fn)
        rounds={name:[] for name in winners}
        for round_id in range(6):
            names=list(winners)
            if round_id%2: names.reverse()
            for name in names:
                rounds[name].append(measure(*graphs[name],11))
                check_output(winners[name]["split"])
        confirmations={}
        for name,cell in winners.items():
            confirmations[name]=dict(recipe=cell,rounds_us=rounds[name],
                median_us=statistics.median(statistics.median(r) for r in rounds[name]))
        best=confirmations["xplane_best"]["median_us"]
        for name,row in confirmations.items():
            row["delta_vs_xplane_pct"]=(row["median_us"]/best-1)*100
            print("XPLANE_KPACK_RESULT "+json.dumps(dict(shape=[1,n,k],mode=mode,name=name,**row)),flush=True)
        result["modes"][mode]=dict(copies=copies,working_set_bytes=copies*one_bytes,
            screen=cells,confirmed=confirmations,zero_code_negatives=len(winners),
            graph_output_checks="PASS")
        del graphs,storage,out,ws
        torch.cuda.synchronize()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures",type=Path,required=True)
    p.add_argument("--xplane-library",type=Path,required=True)
    p.add_argument("--xplane-arithmetic", choices=("fp16-group", "fp32"), default="fp16-group")
    p.add_argument("--kpack-library",type=Path,required=True)
    p.add_argument("--expected-reader", choices=("cuda-n2", "cuda-q4-n2", "cuda-q4-n2-wide", "cuda-q4-n2-shared", "cuda-q4-n2-half-control", "cuda-q4-shm-fp32", "cuda-q4-n2-grid", "cuda-q4-n2-aligned", "cuda-q4-n2-metadata", "cuda-q4-n4", "cuda-q4-n4-unsigned", "cuda-q4-n4-tree", "cuda-q4-n2-tree", "cuda-q4-n2-warp", "cuda-q4-n4-coop", "cuda-q4-small-static", "cuda-q4-small-balanced", "cuda-q4-n4-static"),
                   default="cuda-n2")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError("output already exists")
    fixtures = sorted(args.fixtures.glob("*.npz"))
    if not fixtures: raise ValueError("no fixtures")
    authority = n2_authority(args.kpack_library, args.expected_reader)
    xp_authority = xplane_authority(args.xplane_library, args.xplane_arithmetic)
    if args.xplane_arithmetic == "fp32" and args.expected_reader == "cuda-q4-n2-half-control":
        raise ValueError("FP32 comparison may not use the K-pack half arithmetic control")
    xplane=C.CDLL(str(args.xplane_library.resolve()))
    xplane.q4_xplane_pack.argtypes=[C.c_int,C.c_int]+[C.c_void_p]*3
    xplane.q4_xplane_run.argtypes=[C.c_int]*4+[C.c_void_p]*5
    xplane.q4_xplane_pack.restype=xplane.q4_xplane_run.restype=C.c_int
    xplane.q4_device_l2_bytes.restype=C.c_int
    l2_bytes=xplane.q4_device_l2_bytes()
    if l2_bytes <= 0: raise ValueError("device L2 size query failed")
    kpack=C.CDLL(str(args.kpack_library.resolve()))
    prop=torch.cuda.get_device_properties(0)
    result=dict(scope="MATCHED_Q4_F16_A_F32_OUTPUT_WARM_AND_GT_L2_ROTATION",
        arithmetic=("BOTH_FP32_DOT_AND_REDUCTION_FP16_WEIGHT_AND_A_BOUNDARY"
                    if args.xplane_arithmetic == "fp32" else
                    "BOTH_FP16_GROUP_PARTIAL_FP32_CROSS_GROUP_NOT_IDENTICAL_SUM_ORDER"
                    if args.expected_reader == "cuda-q4-n2-half-control" else
                    "XPLANE_FP16_GROUP_ACCUM_VS_KPACK_FP32_ACCUM_NOT_LAYOUT_ONLY"),
        gpu=str(prop),xplane_library_sha256=sha(args.xplane_library),
        kpack_library_sha256=sha(args.kpack_library),kpack_authority=authority,
        xplane_authority=xp_authority,
        kpack_affine="UNSIGNED_ZMUL0" if args.expected_reader.endswith("-unsigned") else "SIGNED_ZMUL8",
        runner_sha256=sha(Path(__file__)), cases=[])
    started=time.monotonic()
    for path in fixtures:
        result["cases"].append(compare(path,xplane,kpack,l2_bytes,authority["columns"],authority["warps"],authority["extra_s1"]))
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    result["seconds"]=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("XPLANE_KPACK_COMPLETE "+str(args.output),flush=True)


if __name__ == "__main__":
    main()

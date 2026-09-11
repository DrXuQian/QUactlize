#!/usr/bin/env python3
"""Create a compact, hash-linked receipt for the Q4 S1 investigations."""
import argparse
import json
import math
from pathlib import Path
from run_standalone import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","campaign","confirmation","ncu","f16","unaligned","build","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--six-shapes",action="store_true",
                   help="retain the full six-family confirmation and larger warm profiles")
    p.add_argument("--static-numeric",type=Path,
                   help="required with --six-shapes: all new static specializations and negatives")
    a=p.parse_args()
    if a.output.exists():raise ValueError("output already exists")
    if a.six_shapes and a.static_numeric is None:
        p.error("six-shape admission requires the static numerical gate")
    names=("baseline","campaign","confirmation","ncu","f16","unaligned","build")
    if a.static_numeric is not None: names+=("static_numeric",)
    data={name:json.loads(getattr(a,name).read_text()) for name in names}
    library=data["campaign"]["authority"]["candidate"]["sha256"]
    if any(data[name].get("status")!="PASS" for name in ("baseline","campaign","confirmation","f16","unaligned")):
        raise ValueError("incomplete evidence")
    for name in ("baseline","campaign"):
        if data[name]["arithmetic"]!="BOTH_FP32_DOT_AND_REDUCTION_FP16_WEIGHT_AND_A_BOUNDARY":
            raise ValueError("not an FP32 comparison")
        if len(data[name]["cases"])!=12:raise ValueError("missing dense families")
    if data["build"]["library_sha256"]!=library or data["confirmation"]["authority"]["candidate"]["sha256"]!=library:
        raise ValueError("candidate identity differs")
    numeric=[]
    for name in ("f16","unaligned"):
        doc=data[name]
        if doc["library_sha256"]!=library or len(doc["records"])!=116:
            raise ValueError("numeric identity or denominator differs")
        errors=[r["error"] for r in doc["records"]]
        if not all(math.isfinite(x) and 0<=x<.005 for x in errors):
            raise ValueError("numeric error outside bound")
        numeric.append(dict(input_type=doc["input_type"],unaligned=doc["unaligned"],
            scope=doc["scope"],cases=len(errors),max_conditioned_error=max(errors)))
    if len(data["ncu"]["profiles"])!=8 or any(len(r["kernels"])!=1 for r in data["ncu"]["profiles"]):
        raise ValueError("expected eight S1/no-reducer profiles")
    if data["ncu"]["authority"]["kpack"]["sha256"]!=library:
        raise ValueError("profile library differs")
    if a.six_shapes:
        shapes={(1,n,k) for n,k in ((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120))}
        expected={(s,m) for s in shapes for m in ("warm","rotating")}
        confirmation=data["confirmation"]
        actual={(tuple(r["shape"]),r["mode"]) for r in confirmation["cases"]}
        if (len(confirmation["cases"])!=12 or actual!=expected or
                confirmation["performance_verdict"]!="WITHIN_5_PERCENT" or
                any(r["delta_pct"]>5 or r["recipes"]["kpack"][-1]!=1 for r in confirmation["cases"])):
            raise ValueError("six-family S1 confirmation did not meet the target")
        static=data["static_numeric"]
        keys={(tuple(r["shape"]),tuple(r["recipe"])) for r in static["records"]}
        large={(s,(c,w,1)) for s in shapes if s[1]>=4096 for c in (4,8) for w in (4,5,8,10,16)}
        if (static.get("status")!="PASS" or static["library_sha256"]!=library or
                len(static["records"])!=44 or len(keys)!=44 or not large<=keys or
                not all(r["status"]=="PASS" and math.isfinite(r["error"]) and 0<=r["error"]<.005 for r in static["records"])):
            raise ValueError("static numerical coverage or identity differs")
        numeric.append(dict(input_type="F16",unaligned=False,scope=static["scope"],cases=44,
                            max_conditioned_error=max(r["error"] for r in static["records"])))
    cases=[]
    for c in data["campaign"]["cases"]:
        old=[r for r in data["baseline"]["cases"] if (r["shape"],r["mode"])==(c["shape"],c["mode"])]
        if len(old)!=1:raise ValueError("baseline shape ambiguity")
        cases.append(dict(shape=c["shape"],mode=c["mode"],
            fixture_sha256=c["fixture_sha256"],
            baseline_n4_us=old[0]["winners"]["kpack"]["median_us"],
            median_us={k:v["median_us"] for k,v in c["winners"].items()},
            recipes={k:v["recipe"] for k,v in c["winners"].items()},
            delta_pct=c["delta_pct"],s1_delta_pct=c["s1_delta_pct"],
            confirmed=c["confirmed"]))
    result=dict(scope="RTX5070_Q4_M1_F16_A_FP32_DOT_NOT_PPU_NOT_GROUPED_PERFORMANCE",
        acceptance_pct=5.0,source_sha256=sha(Path(__file__)),
        inputs={name:dict(file=getattr(a,name).name,sha256=sha(getattr(a,name))) for name in names},
        library_sha256=library,reader=data["build"]["reader"],
        builder_sha256=data["build"]["builder_sha256"],
        generated_source_sha256=data["build"]["generated_source_sha256"],
        offline_arrangement_changed=False,production_changed=False,
        numeric=numeric,campaign=cases,confirmation=data["confirmation"],ncu=data["ncu"],
        remaining=(["larger-family warm-cache gaps above 5%"] if not a.six_shapes else [])+["multi-token/grouped performance",
                   "other quantized format performance","PPU lowering and device admission",
                   "RTX5090 recheck when the machine is available"])
    result["admitted_shape_scope"]="SIX_DENSE_FAMILIES_BOTH_REGIMES" if a.six_shapes else "TWO_SMALL_FAMILIES_BOTH_REGIMES"
    if a.six_shapes: result["static_numeric"]=data["static_numeric"]
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("Q4_S1_RECEIPT "+str(a.output),flush=True)


if __name__=="__main__":
    main()

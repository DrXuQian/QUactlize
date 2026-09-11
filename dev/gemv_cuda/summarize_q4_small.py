#!/usr/bin/env python3
"""Create a compact, hash-linked receipt for the small-Q4 S1 investigation."""
import argparse
import json
import math
from pathlib import Path
from run_standalone import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","campaign","confirmation","ncu","f16","unaligned","build","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("output already exists")
    names=("baseline","campaign","confirmation","ncu","f16","unaligned","build")
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
        remaining=["larger-family warm-cache gaps above 5%","multi-token/grouped performance",
                   "other quantized format performance","PPU lowering and device admission",
                   "RTX5090 recheck when the machine is available"])
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("Q4_SMALL_RECEIPT "+str(a.output),flush=True)


if __name__=="__main__":
    main()

"""Bounded physical-Hopper correctness and full-chain PDL timing; no policy changes."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quactlize/backends/ppu17"))
from check_device import sha
from check_pdl import run_case
from run_benchmark import require_idle, run_point, snapshot


def correctness_points():
    points=[]
    for q in (8,10,11,12,13,14):
        for compute in (0,1):
            for route in (0,1):
                tail=256 if q==8 else 48
                quantum=512 if q in (11,14) else 256
                for m,n,k,rows in ((1,tail,512,None),(65,256,1024,None),
                                   (74,tail,1024,"0,1,8,65,0"),(1,tail,quantum,"0,0,1,0")):
                    points.append(dict(q=q,compute=compute,route=route,m=m,n=n,k=k,rows=rows))
    return points


def timing_points():
    return [dict(compute=t,route=r,m=m,n=512,k=2048,grouped=g,cache=cache)
            for t in (0,1) for r in (0,1) for m,g in ((1,False),(128,False),(65,True))
            for cache in ("warm","rotating")]


def validate_chain(text, point):
    lines=[s for s in text.splitlines() if s.startswith("PPU17_PDL_CHAIN PASS ")]
    if len(lines)!=1: raise ValueError("missing/duplicate correctness receipt")
    fields=dict(item.split("=",1) for item in lines[0].split()[2:])
    expected={**{k:str(point[k]) for k in ("q","compute","route","m","n","k")},
              "input":str(point["compute"]),"output":"2","config":"5",
              "grouped":str(int(point["rows"] is not None)),"comparisons":"36","prefetch":"1",
              "raw_equal":"1","cpu_oracle":"PASS","stream":"PASS","graph":"PASS","timing":"NOT_MEASURED"}
    if any(fields.get(k)!=v for k,v in expected.items()):
        raise ValueError("correctness receipt differs from requested point")


def validate_timing(row, point):
    expected={"q":12,"compute":point["compute"],"route":point["route"],
              "m":point["m"]+9 if point["grouped"] else point["m"],
              "n":point["n"],"k":point["k"],"experts":4 if point["grouped"] else 1,
              "cache":point["cache"],"raw_equal":True,"producer_config":2,"consumer_config":5,
              "metric_scope":"GEMM_OPTIONAL_SF_GEMM_CHAIN","warmup_graphs_per_arm":3}
    if any(row.get(k)!=v for k,v in expected.items()): raise ValueError("timing point/chain scope differs")
    if set(row.get("arms",{}))!={"ordinary","protocol","prefetch"}: raise ValueError("three arms required")
    for arm in row["arms"].values():
        samples=arm["samples_us"]
        if len(samples)!=11 or any(not math.isfinite(v) or v<=0 for v in samples): raise ValueError("invalid samples")
        if abs(sorted(samples)[5]-arm["median_us"])>1.e-6: raise ValueError("median differs")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="0")
    p.add_argument("--expected-uuid",required=True)
    p.add_argument("--phase",choices=("correctness","timing"),required=True)
    p.add_argument("--correctness",type=Path)
    p.add_argument("--resume-from",type=Path,help="reuse only hash-verified identical successful correctness points")
    p.add_argument("--only",type=int,nargs="+")
    p.add_argument("--reverse",action="store_true")
    p.add_argument("--timeout",type=int,default=180)
    args=p.parse_args()
    build,baseline,out=args.build.resolve(),args.baseline.resolve(),args.output.resolve()
    manifest=json.loads((build/"manifest.json").read_text())
    if manifest["status"]!="PASS" or sha(baseline/"libquactlize_ppu17.so")!=manifest["library_sha256"]:
        raise ValueError("experiment build/baseline identity differs")
    for name,digest in manifest["applications"].items():
        if sha(build/name)!=digest: raise ValueError("application differs: "+name)
    identity=dict(manifest_sha256=sha(build/"manifest.json"),library_sha256=manifest["library_sha256"],
                  applications=manifest["applications"],gpu_uuid=args.expected_uuid)
    if args.phase=="timing":
        if not args.correctness: raise ValueError("full correctness receipt required before timing")
        gate=json.loads(args.correctness.read_text())
        if gate["identity"]!=identity or gate["status"]!="PASS" or len(gate["results"])!=96 or gate["partial"] or gate["points"]!=correctness_points():
            raise ValueError("full same-binary/device correctness has not passed")
    out.mkdir(parents=True,exist_ok=False)
    initial=snapshot(args.device);require_idle(initial,args.expected_uuid)
    points=correctness_points() if args.phase=="correctness" else timing_points()
    indices=list(range(len(points))) if args.only is None else args.only
    if len(set(indices))!=len(indices) or any(i<0 or i>=len(points) for i in indices): raise ValueError("invalid point indices")
    if args.reverse: indices=list(reversed(indices))
    receipt=dict(identity=identity,gpu=initial,phase=args.phase,points=points,indices=indices,
                 partial=set(indices)!=set(range(len(points))),status="INCOMPLETE",results=[],
                 scope="PHYSICAL_HOPPER_NOT_NATIVE_PPU17",production_selection="UNCHANGED",runner_sha256=sha(__file__))
    reused={}
    if args.resume_from:
        previous=json.loads(args.resume_from.read_text())
        if args.phase!="correctness" or previous["phase"]!="correctness" or previous["identity"]!=identity:
            raise ValueError("resume is only for same-binary/device correctness")
        for row in previous["results"]:
            if row["status"]!="PASS":continue
            log=Path(row["log"])
            if not log.is_file() or sha(log)!=row["log_sha256"]:raise ValueError("resume log identity differs")
            validate_chain(log.read_text(),row["point"])
            key=json.dumps(row["point"],sort_keys=True)
            if key in reused:raise ValueError("duplicate resume point")
            reused[key]=row
        receipt["resume_summary_sha256"]=sha(args.resume_from)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":args.device,"Q17_BENCH_GUARDED":"1",
         "LD_LIBRARY_PATH":str(baseline)+(":"+os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
    def save(): (out/"summary.json").write_text(json.dumps(receipt,indent=2)+"\n")
    started=time.monotonic();save()
    for index in indices:
        point=points[index];log=out/f"{index:04d}.log"
        prior=reused.get(json.dumps(point,sort_keys=True))
        if prior:
            receipt["results"].append({**prior,"index":index,"reused":True})
            save()
            print(f"PDL_PREFETCH_PROGRESS phase=correctness completed={len(receipt['results'])}/{len(indices)} status=REUSED_PASS",flush=True)
            continue
        command=[str(build/("chain" if args.phase=="correctness" else "benchmark"))]
        for key in ("compute","m","n","k"): command += ["--"+key,str(point[key])]
        if point["route"]: command.append("--sf")
        if args.phase=="correctness":
            command += ["--prefetch","--config","5","--qtype",str(point["q"]),"--input",str(point["compute"]),"--output","2"]
            if point["rows"]: command += ["--rows",point["rows"]]
        else:
            if point["grouped"]: command.append("--grouped")
            if point["cache"]=="rotating": command.append("--rotating")
        result={}
        try:
            if args.phase=="correctness":
                result=run_case(command,env,log,args.device,args.expected_uuid,args.timeout)
                if result["error"] or result["returncode"]: raise ValueError(result["error"] or "child rc="+str(result["returncode"]))
                validate_chain(log.read_text(),point)
            else:
                result=run_point(command,env,log,args.device,args.expected_uuid,args.timeout)
                if result["status"]!="PASS": raise ValueError(result["error"])
                validate_timing(result["result"][0],point)
            result["status"]="PASS"
        except Exception as error:
            result.update(status="FAIL",error=str(error))
        receipt["results"].append({**result,"index":index,"point":point,"log":str(log),
                                   "log_sha256":sha(log) if log.exists() else None})
        save()
        print(f"PDL_PREFETCH_PROGRESS phase={args.phase} completed={len(receipt['results'])}/{len(indices)} status={result['status']} elapsed_s={time.monotonic()-started:.1f}",flush=True)
        if "GPU" in (result.get("error") or "") or "foreign" in (result.get("error") or ""): break
    receipt["status"]="PASS" if len(receipt["results"])==len(indices) and all(r["status"]=="PASS" for r in receipt["results"]) else "FAIL_OR_INCOMPLETE"
    receipt["elapsed_seconds"]=time.monotonic()-started;save()
    print(f"PDL_PREFETCH_DONE phase={args.phase} status={receipt['status']} results={out}",flush=True)
    return int(receipt["status"]!="PASS")


if __name__=="__main__":
    raise SystemExit(main())

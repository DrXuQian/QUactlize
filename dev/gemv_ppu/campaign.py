"""Orchestration only: keep every case, cache exact batches, never admit dirty output."""
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from dev.gemv_cuda.build import digest
from dev.gemv_ppu.run import ARMS, SHAPES, configs, verify_bundle, load_library, SDK, device_identity
from tools.verify_kpack_dispatch import verify as verify_native
from tools.profile_kpack_gpu_compact import acu_launch_command

ROOT=Path(__file__).resolve().parents[2]


def parse_cells(text,arm,recipes,shape,mode,samples):
    rows=[json.loads(line.split(" ",1)[1]) for line in text.splitlines() if line.startswith("Q4_PPU_CELL ")]
    if len(rows)!=len(recipes):raise ValueError("missing or duplicate cells")
    for r,recipe in zip(rows,recipes):
        if (r.get("status")!="PASS" or r.get("arm")!=arm or r.get("recipe")!=list(recipe) or
                r.get("shape")!=shape or r.get("mode")!=mode or r.get("zero_code_negative")!="PASS"):
            raise ValueError("dirty or mismatched cell: "+str(r))
        if not math.isfinite(r["error"]) or not 0<=r["error"]<.005:raise ValueError("invalid numeric error")
        if arm in ("old","new") and recipe[2]>1 and r.get("reducer_check")!="ORDERED_FP32":
            raise ValueError("missing ordered FP32 reducer check")
        values=r["samples_us"]
        if len(values)!=samples or not all(math.isfinite(x) and x>0 for x in values):
            raise ValueError("invalid sample count or values")
        if samples and abs(statistics.median(values)-r["median_us"])>1e-8:
            raise ValueError("median differs from raw samples")
        if type(r["copies"]) is not int or r["copies"]<1:raise ValueError("invalid cache ring")
        if r["weight_bytes"]!=shape[1]*shape[2]*9//16 or r["device"]["l2_bytes"]<=0:
            raise ValueError("invalid weight/L2 bytes")
        if r["calls_per_graph"]!=max(2,(32+r["copies"]-1)//r["copies"])*r["copies"]:
            raise ValueError("graph denominator differs or cuts a cold ring")
        if mode=="warm" and r["copies"]!=1:raise ValueError("warm uses more than one weight")
        if mode=="rotating" and (r["copies"]<2 or r["copies"]*r["weight_bytes"]<2.25*r["device"]["l2_bytes"]):
            raise ValueError("cold ring fits in L2")
    return rows


def run(args):
    if args.output is None or args.fixtures is None:raise ValueError("--output and --fixtures are required")
    args.output=args.output.resolve();args.fixtures=args.fixtures.resolve(strict=True)
    m=verify_bundle(args.bundle);native=verify_native(args.native_bundle)
    sdk=SDK(args.sdk);device=device_identity(sdk)
    for arm in ("old","new","xplane"):
        library,geometry=load_library(args,arm)
        if arm=="old":device.update(geometry)
        elif any(device[key]!=value for key,value in geometry.items()):
            raise ValueError("comparison arms report different device geometry")
    inputs={"bundle":digest(args.bundle/"manifest.json"),"native":digest(args.native_bundle/"manifest.json"),
            "runner":digest(ROOT/"dev/gemv_ppu/run.py"),"campaign":digest(Path(__file__)),
            "l2_override":args.l2_bytes,"device":device,"confirm_rounds":6,"samples":args.samples,
            "runtime":{name:digest(args.sdk/"lib"/name) for name in m["runtime"]},
            "fixtures":{f"{n}x{k}":digest(args.fixtures/f"q12-n{n}-k{k}-e1-c1.npz") for n,k in SHAPES}}
    runtime_diff=[name for name,want in m["runtime"].items() if inputs["runtime"][name]!=want]
    if runtime_diff:print("Q4_PPU_SDK_DIFFERENCE recorded="+",".join(runtime_diff)+" admission=REAL_MARKER_AND_NUMERIC_REQUIRED",flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    authority=args.output/"authority.json"
    if authority.exists() and json.loads(authority.read_text())!=inputs:
        raise ValueError("resume identity changed; use a new output directory")
    authority.write_text(json.dumps(inputs,indent=2)+"\n")
    started=time.monotonic();jobs=0
    def command(arm,n,k,mode,recipes,samples,profile=False):
        cmd=[sys.executable,"-u",str(ROOT/"dev/gemv_ppu/run.py"),"--child","--sdk",str(args.sdk),
            "--bundle",str(args.bundle),"--native-bundle",str(args.native_bundle),"--jit-cache",str(args.jit_cache),
            "--fixture",str(args.fixtures/f"q12-n{n}-k{k}-e1-c1.npz"),"--arm",arm,"--mode",mode,
            "--recipes",json.dumps(recipes),"--samples",str(samples),"--l2-bytes",str(args.l2_bytes)]
        return cmd+(["--profile"] if profile else [])
    def batch(arm,n,k,mode,recipes,label,samples):
        nonlocal jobs
        log=args.output/f"n{n}-k{k}-{mode}-{arm}-{label}.log"
        saved=log.with_suffix(".json")
        cmd=command(arm,n,k,mode,recipes,samples)
        if saved.is_file() and log.is_file():
            record=json.loads(saved.read_text())
            if record.get("command")==cmd and record.get("log_sha256")==digest(log) and record.get("rc")==0:
                rows=parse_cells(log.read_text(),arm,recipes,[1,n,k],mode,samples)
                if any(r["device"]!=device for r in rows):raise ValueError("cached batch device differs")
                return rows
        print(f"Q4_PPU_PROGRESS phase={label} shape=1x{n}x{k} mode={mode} arm={arm} recipes={len(recipes)} elapsed_s={time.monotonic()-started:.1f}",flush=True)
        if log.exists():log.rename(log.with_name(log.name+f".failed.{time.time_ns()}"))
        with log.open("w") as out:
            proc=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
        record=dict(command=cmd,log_sha256=digest(log),rc=proc.returncode)
        saved.write_text(json.dumps(record,indent=2)+"\n");jobs+=1
        if proc.returncode:raise ValueError(f"child rc={proc.returncode}; log={log}")
        rows=parse_cells(log.read_text(),arm,recipes,[1,n,k],mode,samples)
        if any(r["device"]!=device for r in rows):raise ValueError("child batch device differs")
        return rows
    report=dict(status="RUNNING",scope="Q4_DENSE_M1_PPU_COLD_AND_WARM",authority=inputs,cases=[],failures=[],profiles=[])
    for n,k in SHAPES:
        for mode in ("warm","rotating"):
            winners={}
            for arm in ARMS:
                try:
                    rows=batch(arm,n,k,mode,configs(arm),"screen",5)
                    contenders=sorted(rows,key=lambda r:r["median_us"])[:2]
                    if arm in ("old","new") and not any(r["recipe"][-1]==1 for r in contenders):
                        contenders.append(min((r for r in rows if r["recipe"][-1]==1),key=lambda r:r["median_us"]))
                    winners[arm]=dict(contenders=[r["recipe"] for r in contenders],rounds=[])
                except Exception as exc:
                    failure=dict(shape=[1,n,k],mode=mode,arm=arm,phase="screen",error=str(exc))
                    report["failures"].append(failure);print("Q4_PPU_FAILURE "+json.dumps(failure),flush=True)
            for turn in range(6):
                for arm in (ARMS if turn%2==0 else tuple(reversed(ARMS))):
                    if arm not in winners:continue
                    entry=winners[arm]
                    try:
                        records=batch(arm,n,k,mode,entry["contenders"],f"confirm{turn}",args.samples)
                        entry["rounds"].append(records)
                    except Exception as exc:
                        report["failures"].append(dict(shape=[1,n,k],mode=mode,arm=arm,phase=f"confirm{turn}",error=str(exc)))
                        del winners[arm]
            final={}
            for arm,entry in winners.items():
                options=[dict(recipe=recipe,rounds=[rr[i] for rr in entry["rounds"]]) for i,recipe in enumerate(entry["contenders"])]
                for option in options:option["median_us"]=statistics.median(r["median_us"] for r in option["rounds"])
                final[arm]=min(options,key=lambda r:r["median_us"])
            delta=100*(final["new"]["median_us"]/final["xplane"]["median_us"]-1) if {"new","xplane"}<=final.keys() else None
            row=dict(shape=[1,n,k],mode=mode,winners=final,new_vs_xplane_pct=delta,
                     verdict="INCOMPLETE" if len(final)!=4 else "WITHIN_5_PERCENT" if delta<=5 else "PARITY_OPEN")
            report["cases"].append(row)
            (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
            print("Q4_PPU_RESULT "+json.dumps(dict(shape=row["shape"],mode=mode,
                median_us={a:r["median_us"] for a,r in final.items()},new_vs_xplane_pct=delta,verdict=row["verdict"])),flush=True)
    with (args.output/"summary.tsv").open("w") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["M","N","K","cache",*[a+"_us" for a in ARMS],"new_vs_xplane_pct",*[a+"_effective_weight_GBs" for a in ARMS],"verdict"])
        for r in report["cases"]:
            times=[r["winners"].get(a,{}).get("median_us") for a in ARMS]
            weight_bytes=r["shape"][1]*r["shape"][2]*9//16
            w.writerow([*r["shape"],r["mode"],*[t if t is not None else "NA" for t in times],
                        r["new_vs_xplane_pct"],*[weight_bytes/t/1000 if t else "NA" for t in times],r["verdict"]])
    if not args.skip_acu:
        acu=args.acu or args.sdk/"asight/bin/acu"
        for row in report["cases"]:
            if row["mode"]!="warm":continue
            _,n,k=row["shape"]
            for arm in ("xplane","new"):
                if arm not in row["winners"]:continue
                selected=row["winners"][arm]["recipe"]
                prefix=args.output/f"n{n}-k{k}-{arm}.acu"
                log=prefix.with_suffix(".acu.log")
                saved=prefix.with_suffix(".acu.json")
                record=dict(shape=[1,n,k],arm=arm,recipe=selected,recipe_regime="warm",acu_cache_control="all",timing_authority=False)
                try:
                    cmd=acu_launch_command(acu,prefix,command(arm,n,k,"warm",[selected],args.samples,True))
                    if saved.is_file() and log.is_file():
                        cached=json.loads(saved.read_text())
                        files=cached.get("files",[])
                        if (cached.get("status")=="PASS" and cached.get("command")==list(map(str,cmd)) and
                            cached.get("log_sha256")==digest(log) and files and
                            all(Path(f["file"]).name==f["file"] and (args.output/f["file"]).is_file() and
                                digest(args.output/f["file"])==f["sha256"] for f in files)):
                            parse_cells(log.read_text(),arm,[selected],[1,n,k],"warm",0)
                            report["profiles"].append(cached)
                            continue
                    print(f"Q4_PPU_ACU shape=1x{n}x{k} arm={arm} cache=FORCED_COLD",flush=True)
                    # Preserve diagnostic evidence from an interrupted or failed attempt.
                    stamp=time.time_ns()
                    for previous in args.output.glob(prefix.name+"*"):
                        if previous.is_file():previous.rename(previous.with_name(previous.name+f".previous.{stamp}"))
                    with log.open("w") as out:proc=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
                    if proc.returncode:raise ValueError(f"acu rc={proc.returncode}; log={log}")
                    parse_cells(log.read_text(),arm,[selected],[1,n,k],"warm",0)
                    reports=[p for p in args.output.glob(prefix.name+"*") if p.suffix in (".acurep",".report")]
                    if not reports:raise ValueError("ACU returned no report")
                    record.update(status="PASS",files=[dict(file=p.name,sha256=digest(p)) for p in reports],
                                  command=list(map(str,cmd)),log_sha256=digest(log))
                except Exception as exc:record.update(status="FAIL",error=str(exc))
                saved.write_text(json.dumps(record,indent=2)+"\n")
                report["profiles"].append(record)
    verify_bundle(args.bundle);verify_native(args.native_bundle)
    if digest(args.bundle/"manifest.json")!=inputs["bundle"] or digest(args.native_bundle/"manifest.json")!=inputs["native"]:
        raise ValueError("bundle identity changed during measurement")
    if any(digest(args.sdk/"lib"/name)!=value for name,value in inputs["runtime"].items()):
        raise ValueError("runtime changed during measurement")
    report.update(status="PASS" if not report["failures"] and all(p["status"]=="PASS" for p in report["profiles"]) else "INCOMPLETE",
                  elapsed_seconds=time.monotonic()-started,executed_batches=jobs,
                  acu="SKIPPED" if args.skip_acu else "FORCED_COLD_DIAGNOSTICS_NOT_WARM_TIMING")
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Q4_PPU_COMPLETE status={report['status']} rows={len(report['cases'])}/12 results={args.output}",flush=True)
    return 0 if report["status"]=="PASS" else 1

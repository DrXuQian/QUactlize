#!/usr/bin/env python3
"""PPU-selected Q4 recipes on CUDA plus raw-GGUF FP32, no retuning."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_cuda.compare_q4_native import parse_timing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build",type=Path,required=True);p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--ncu",type=Path)
    a=p.parse_args();a.build=a.build.resolve(strict=True);a.fixture=a.fixture.resolve(strict=True)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.build/"manifest.json").read_text())
    for name,sha in manifest["payloads"].items():
        if digest(a.build/name)!=sha:raise ValueError("modified CUDA binary")
    arms={"xplane":[2,8,1],"kpack":[4,8,1],"raw-reference":[2,8,1]}
    report=dict(status="RUNNING",scope="PPU_FIXED_RECIPES_ON_H800_NOT_RETUNED_NOT_PPU_ADMISSION",
        shape=[1,5120,8192],rounds=6,samples=15,precision="FP16_A_FP32_DOT_REDUCTION_OUTPUT",
        fixture_sha256=digest(a.fixture),build_sha256=digest(a.build/"manifest.json"),
        recipes=arms,cases=[],profiles=[],gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader"],text=True))
    def command(arm,mode,purpose):
        return [str(a.build/"profile"),str(a.fixture),str(a.build/"libreference.so"),str(a.build/"libkpack.so"),
                arm,*map(str,arms[arm]),mode,purpose]
    def save(): (a.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    geometry=None;started=time.monotonic()
    for mode in ("warm","rotating"):
        records={v:[] for v in arms}
        for turn in range(6):
            for arm in (list(arms) if turn%2==0 else list(arms)[::-1]):
                log=a.output/f"{mode}-{turn}-{arm}.log";cmd=command(arm,mode,"timing")
                proc=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
                log.write_text(proc.stdout+proc.stderr)
                if proc.returncode:raise ValueError(f"{arm}: rc={proc.returncode} log={log}")
                row=parse_timing(proc.stdout,arm,arms[arm],mode,report["shape"])
                observed=(int(row["sm"]),int(row["L2_bytes"]))
                if geometry is not None and observed!=geometry:raise ValueError("GPU changed")
                geometry=observed
                one=5120*8192*9//16
                copies=1 if mode=="warm" else max(2,(9*observed[1]+4*one-1)//(4*one))
                calls=max(2,(32+copies-1)//copies)*copies
                if int(row["copies"])!=copies or int(row["calls_per_graph"])!=calls:raise ValueError("cache/graph mismatch")
                records[arm].append(row|dict(log=log.name,log_sha256=digest(log)))
            print(f"Q4_H800_PROGRESS mode={mode} round={turn+1}/6 elapsed_s={time.monotonic()-started:.1f}",flush=True)
        medians={v:statistics.median(float(r["median_us"]) for r in rows) for v,rows in records.items()}
        row=dict(mode=mode,median_us=medians,kpack_vs_xplane_pct=100*(medians["kpack"]/medians["xplane"]-1),
            kpack_vs_raw_reference_pct=100*(medians["kpack"]/medians["raw-reference"]-1),records=records)
        report["cases"].append(row);save()
        print("Q4_H800_RESULT "+json.dumps({k:v for k,v in row.items() if k!="records"}),flush=True)
    report["status"]="PASS";report["device_geometry"]=geometry;save()
    if a.ncu:
        for arm in arms:
            log=a.output/f"{arm}.ncu.log";prefix=a.output/f"{arm}.ncu"
            cmd=[str(a.ncu),"--profile-from-start","off","--replay-mode","kernel","--cache-control","all",
                 "--section","SpeedOfLight","--section","MemoryWorkloadAnalysis","--section","LaunchStats",
                 "--section","Occupancy","--section","WarpStateStats","--export",str(prefix),
                 *command(arm,"warm","profile")]
            with log.open("w") as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode
            text=log.read_text();files=list(a.output.glob(prefix.name+"*.ncu-rep"))
            profile=dict(arm=arm,rc=rc,log=log.name,files=[p.name for p in files],cache="FORCED_COLD",timing_authority=False)
            if "ERR_NVGPUCTRPERM" in text:profile["status"]="COUNTERS_PERMISSION_DENIED"
            elif not rc and files:profile["status"]="PASS"
            else:profile["status"]="FAIL"
            report["profiles"].append(profile);save()
            print("Q4_H800_NCU "+json.dumps(profile),flush=True)
            if profile["status"]=="COUNTERS_PERMISSION_DENIED":break
    report["seconds"]=time.monotonic()-started;save()
    print("Q4_H800_COMPLETE results="+str(a.output),flush=True)


if __name__=="__main__":main()

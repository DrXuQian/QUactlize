"""Bounded physical-Hopper sanitizer gates; never performance evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from check_device import arguments, gpu_identity, gpu_processes, sha


def cases(suite="baseline"):
    result = []
    if suite=="cached-decode":
        for config in (6,7,8,9,10):
            for tool in ("memcheck","racecheck","synccheck"):
                for compute in (0,1):
                    result.append({"tool":tool,"point":dict(q=12,compute=compute,config=config,
                        route=compute,rows=[0,1,2,8,0],m=8,n=257,k=1024,
                        input=2,output=2,phase="gemm")})
        return result
    def add(tool, q, compute, config, storage=None, phase="gemm"):
        result.append({"tool":tool, "point":dict(q=q, compute=compute, config=config,
            route=0, rows=[0,1,8,9,63,64,65,0], m=9, n=256 if q==8 else 48,
            k=1024, input=compute if storage is None else storage, output=2, phase=phase)})
    for q in (8,10,11,12,13,14):
        for compute in (0,1): add("memcheck",q,compute,5)
    for q in (8,12): add("memcheck",q,1,0,storage=2)
    for q in (11,14): add("memcheck",q,1,2,storage=2)
    add("memcheck",12,1,2,phase="scales")
    for q in (11,13):
        for compute in (0,1): add("racecheck",q,compute,5)
    for compute in (0,1): add("racecheck",12,compute,2)
    add("synccheck",13,1,5)
    add("synccheck",8,0,2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--application",type=Path,required=True)
    p.add_argument("--sanitizer",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected-uuid",required=True)
    p.add_argument("--device",default="0")
    p.add_argument("--suite",choices=("baseline","cached-decode"),default="baseline")
    a=p.parse_args()
    bundle,app,tool,out=(v.resolve() for v in (a.bundle,a.application,a.sanitizer,a.output))
    manifest=json.loads((bundle/"manifest.json").read_text())
    library=bundle/"libquactlize_ppu17.so"
    if sha(library)!=manifest["library_sha256"]: raise ValueError("library identity differs")
    identity=gpu_identity(a.device)
    if identity["uuid"]!=a.expected_uuid: raise ValueError("GPU UUID differs")
    if gpu_processes(a.device): raise ValueError("GPU has another job")
    if out.exists() and any(out.iterdir()): raise ValueError("result directory must be fresh")
    out.mkdir(parents=True,exist_ok=True)
    points=cases(a.suite)
    receipt={"status":"INCOMPLETE", "scope":"PHYSICAL_HOPPER_MEMORY_NOT_TIMING",
        "gpu":identity, "library_sha256":sha(library), "application_sha256":sha(app),
        "sanitizer_sha256":sha(tool), "sanitizer_version":subprocess.check_output([str(tool),"--version"],text=True),
        "runner_sha256":sha(__file__), "cases":points, "results":[]}
    env={**os.environ,"CUDA_VISIBLE_DEVICES":a.device,
         "LD_LIBRARY_PATH":str(bundle)+(":"+os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
    for index,case in enumerate(points):
        external=gpu_processes(a.device)
        if external:
            receipt["interruption"]={"reason":"GPU_BECAME_BUSY","processes":external}
            break
        command=[str(tool),"--tool",case["tool"],"--error-exitcode","97","--kill",
                 "--check-bulk-copy","yes","--check-warpgroup-mma","yes"]
        if case["tool"]=="memcheck": command += ["--padding","128"]
        command += arguments(app,case["point"])
        log=out/f"{index:02d}-{case['tool']}.log"
        with log.open("w") as stream:
            stream.write(json.dumps(command)+"\n"); stream.flush()
            try: rc=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired: rc=124
        text=log.read_text()
        clean=("RACECHECK SUMMARY: 0 hazards displayed" in text if case["tool"]=="racecheck"
               else "ERROR SUMMARY: 0 errors" in text)
        status="PASS" if rc==0 and clean and "PPU17_CASE PASS target_launches=1" in text else "FAIL"
        receipt["results"].append({"index":index,"case":case,"status":status,"returncode":rc,
                                   "log":str(log),"log_sha256":sha(log)})
        print(f"PPU17_MEMORY_PROGRESS completed={index+1}/{len(points)} tool={case['tool']} q={case['point']['q']} status={status}",flush=True)
        if status!="PASS": break
    receipt["status"]="PASS" if len(receipt["results"])==len(points) and all(r["status"]=="PASS" for r in receipt["results"]) else "INCOMPLETE_OR_FAILED"
    (out/"summary.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("PPU17_MEMORY_DONE status="+receipt["status"]+" results="+str(out),flush=True)
    return 0 if receipt["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())

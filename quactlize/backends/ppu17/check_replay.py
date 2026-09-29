"""Same-math candidate/control full-byte replay, using fresh GPU processes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

from check_device import arguments, cases, gpu_identity, gpu_processes, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--application",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected-uuid",required=True)
    p.add_argument("--device",default="0")
    a=p.parse_args()
    app,out=a.application.resolve(),a.output.resolve()
    bundles={"baseline":a.baseline.resolve(),"candidate":a.candidate.resolve()}
    hashes={}
    for arm,bundle in bundles.items():
        hashes[arm]=sha(bundle/"libquactlize_ppu17.so")
        if json.loads((bundle/"manifest.json").read_text())["library_sha256"]!=hashes[arm]:
            raise ValueError("library hash differs: "+arm)
    identity=gpu_identity(a.device)
    if identity["uuid"]!=a.expected_uuid: raise ValueError("GPU UUID differs")
    if gpu_processes(a.device): raise ValueError("GPU busy")
    if out.exists() and any(out.iterdir()): raise ValueError("result directory must be fresh")
    out.mkdir(parents=True,exist_ok=True)
    points=[p for p in cases("coverage") if p["config"] in (0,2,5)]
    receipt={"scope":"HOPPER_SAME_MATH_FULL_BYTE_REPLAY_NOT_TIMING","gpu":identity,
        "application_sha256":sha(app),"library_sha256":hashes,"points":points,"results":[]}
    for index,point in enumerate(points):
        record={"index":index,"point":point,"arms":{},"status":"FAIL"}
        payloads=[]
        for arm,bundle in bundles.items():
            if gpu_processes(a.device): raise ValueError("GPU became busy")
            raw=out/f"{index:03d}-{arm}.bin";log=raw.with_suffix(".log")
            command=arguments(app,point)+["--raw-output",str(raw)]
            env={**os.environ,"CUDA_VISIBLE_DEVICES":a.device,
                "LD_LIBRARY_PATH":str(bundle)+(":"+os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
            with log.open("w") as stream:
                stream.write(json.dumps(command)+"\n");stream.flush()
                try: rc=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=60).returncode
                except subprocess.TimeoutExpired: rc=124
            text=log.read_text()
            if rc or "PPU17_CASE PASS target_launches=1" not in text or not raw.is_file():
                record["error"]="arm failed: "+arm;break
            data=raw.read_bytes();payloads.append(data)
            record["arms"][arm]={"log":str(log),"log_sha256":sha(log),"raw":str(raw),"raw_sha256":sha(raw),"bytes":len(data)}
        if len(payloads)==2:
            record["status"]="PASS" if payloads[0]==payloads[1] and payloads[0] else "FAIL"
            if record["status"]=="FAIL": record["error"]="candidate/control bytes differ"
        receipt["results"].append(record)
        receipt["status"]="PASS" if index+1==len(points) and record["status"]=="PASS" else "INCOMPLETE_OR_FAILED"
        (out/"summary.json").write_text(json.dumps(receipt,indent=2)+"\n")
        if index%12==0 or record["status"]!="PASS" or index+1==len(points):
            print(f"PPU17_REPLAY_PROGRESS completed={index+1}/{len(points)} status={record['status']}",flush=True)
        if record["status"]!="PASS": return 1
    return 0


if __name__=="__main__": raise SystemExit(main())

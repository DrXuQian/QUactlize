"""Representative physical-Hopper profiles, separate from ordinary timing."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from check_device import sha
from run_benchmark import snapshot, require_idle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--application",type=Path,required=True)
    p.add_argument("--ncu",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected-uuid",required=True)
    p.add_argument("--device",default="0")
    a=p.parse_args()
    bundle,app,ncu,out=(x.resolve() for x in (a.bundle,a.application,a.ncu,a.output))
    lib=bundle/"libquactlize_ppu17.so"
    manifest=json.loads((bundle/"manifest.json").read_text())
    if sha(lib)!=manifest["library_sha256"]: raise ValueError("library identity differs")
    if out.exists() and any(out.iterdir()): raise ValueError("result directory must be fresh")
    out.mkdir(parents=True,exist_ok=True)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":a.device,
         "LD_LIBRARY_PATH":str(bundle)+(":"+os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
    points=[("simt-m1",0,1,False),("wgmma-m128",2,128,False),("tma-m128",5,128,False),("tma-grouped",5,9,True)]
    receipt={"scope":"PHYSICAL_HOPPER_FORCED_COLD_PROFILE_NOT_RESIDENT_TIMING",
        "application_sha256":sha(app),"library_sha256":sha(lib),"profiler_sha256":sha(ncu),
        "profiler_version":subprocess.check_output([str(ncu),"--version"],text=True),"results":[]}
    metrics=",".join(("gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed","sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active","smsp__warps_eligible.avg.per_cycle_active",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__inst_executed.sum","l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum","launch__registers_per_thread",
        "launch__shared_mem_per_block","launch__grid_size","launch__block_size"))
    for name,config,m,grouped in points:
        telemetry=[]
        quiet=0; deadline=time.monotonic()+15
        while quiet<3:
            s=snapshot(a.device);telemetry.append(s)
            if s["uuid"]!=a.expected_uuid or s["processes"]: require_idle(s,a.expected_uuid)
            quiet=quiet+1 if s["gpu_util"]==0 else 0
            if time.monotonic()>deadline: raise ValueError("GPU did not become idle")
            time.sleep(.25)
        prefix=out/name
        command=[str(ncu),"--clock-control","none","--cache-control","all","--launch-count","1",
            "--metrics",metrics,"--export",str(prefix),str(app),"--qtype","12","--compute","1",
            "--input","1","--output","2","--config",str(config),"--m",str(m),"--n","512","--k","2048"]
        if grouped: command.append("--grouped")
        error=None
        with prefix.with_suffix(".log").open("w") as log:
            log.write(json.dumps(command)+"\n");log.flush()
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+180
            try:
                while child.poll() is None:
                    s=snapshot(a.device);telemetry.append(s)
                    if s["uuid"]!=a.expected_uuid: raise ValueError("GPU UUID changed")
                    for proc in s["processes"]:
                        try: owned=os.getpgid(proc["pid"])==child.pid
                        except ProcessLookupError: owned=False
                        if not owned: raise ValueError("profile contaminated by another GPU process")
                    if time.monotonic()>deadline: raise ValueError("profiler timeout")
                    time.sleep(.1)
            except Exception as failure:
                error=str(failure)
                if child.poll() is None and os.getpgid(child.pid)==child.pid:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if child.poll() is None and os.getpgid(child.pid)==child.pid: os.killpg(child.pid,signal.SIGKILL)
            child.wait()
        text=prefix.with_suffix(".log").read_text()
        report=prefix.with_suffix(".ncu-rep")
        status="PASS" if error is None and child.returncode==0 and report.is_file() and "PPU17_CASE PASS target_launches=1" in text else "FAIL"
        if status=="PASS":
            with prefix.with_suffix(".csv").open("w") as raw:
                subprocess.run([str(ncu),"--import",str(report),"--page","raw","--csv"],stdout=raw,stderr=subprocess.STDOUT,check=True)
        receipt["results"].append(dict(name=name,command=command,status=status,returncode=child.returncode,error=error,
            telemetry=telemetry,report=str(report),report_sha256=sha(report) if report.is_file() else None))
        receipt["status"]="PASS" if len(receipt["results"])==len(points) and status=="PASS" else "INCOMPLETE_OR_FAILED"
        (out/"summary.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(f"PPU17_PROFILE name={name} status={status}",flush=True)
        if status!="PASS": return 1
    return 0


if __name__=="__main__": raise SystemExit(main())

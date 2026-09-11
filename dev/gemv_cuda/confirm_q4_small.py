#!/usr/bin/env python3
"""Confirm fixed S1 small-shape choices independently of the recipe screen."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
from compare_q4_native import parse_timing
from run_standalone import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("runner","fixtures","xplane","candidate","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    manifest=a.candidate.parent/"manifest.json"
    receipt=json.loads(manifest.read_text())
    if receipt["reader"]!="cuda-q4-small-balanced" or receipt["library_sha256"]!=sha(a.candidate):
        raise ValueError("small balanced reader identity differs")
    xp_receipt=a.xplane.parent.parent/"manifest.json"
    matches=[r for r in json.loads(xp_receipt.read_text())["arms"] if r["arm"]=="fp32-control"]
    if len(matches)!=1 or matches[0]["library_sha256"]!=sha(a.xplane):
        raise ValueError("FP32 Xplane identity differs")
    a.output.mkdir(parents=True,exist_ok=False)
    inputs=dict(runner=a.runner,xplane=a.xplane,candidate=a.candidate,
                manifest=manifest,xplane_arithmetic=xp_receipt)
    result=dict(status="RUNNING",scope="FIXED_S1_M1_F16_INPUT_FP32_ACCUMULATION_NOT_PPU",
                threshold_pct=5.0,source_sha256=sha(Path(__file__)),
                authority={k:dict(path=str(v.resolve()),sha256=sha(v)) for k,v in inputs.items()},cases=[])
    for n,k,warps in ((512,2048,8),(1024,5120,10)):
        fixture=a.fixtures/f"q12-n{n}-k{k}-e1-c1.bin"
        for mode in ("warm","rotating"):
            recipes=dict(kpack=[4,warps,1],xplane=[1,(4 if n==512 else 8) if mode=="warm" else (8 if n==512 else 4),1])
            records={arm:[] for arm in recipes}
            for turn in range(6):
                for arm in (("xplane","kpack") if turn%2==0 else ("kpack","xplane")):
                    command=[str(a.runner.resolve()),str(fixture.resolve()),str(a.xplane.resolve()),
                             str(a.candidate.resolve()),arm,*map(str,recipes[arm]),mode,"timing"]
                    proc=subprocess.run(command,capture_output=True,text=True,timeout=180)
                    log=a.output/f"n{n}-k{k}-{mode}-{turn}-{arm}.log"
                    log.write_text(proc.stdout+proc.stderr)
                    if proc.returncode:raise RuntimeError(f"confirmation failed: {log}")
                    r=parse_timing(proc.stdout,arm,recipes[arm],mode,[1,n,k])
                    records[arm].append(dict(median_us=float(r["median_us"]),samples_us=r["samples_us"],
                        error=float(r["error"]),sm=int(r["sm"]),l2_bytes=int(r["L2_bytes"]),log_sha256=sha(log)))
            med={arm:statistics.median(r["median_us"] for r in rr) for arm,rr in records.items()}
            delta=100*(med["kpack"]/med["xplane"]-1)
            row=dict(shape=[1,n,k],mode=mode,recipes=recipes,records=records,median_us=med,
                     delta_pct=delta,within_5pct=delta<=5,fixture_sha256=sha(fixture))
            result["cases"].append(row)
            print("Q4_SMALL_CONFIRM "+json.dumps(dict(shape=row["shape"],mode=mode,median_us=med,
                delta_pct=delta,within_5pct=row["within_5pct"])),flush=True)
    if any(sha(path)!=result["authority"][key]["sha256"] for key,path in inputs.items()):
        raise ValueError("confirmation inputs changed")
    result.update(status="PASS",performance_verdict="WITHIN_5_PERCENT" if all(
        c["within_5pct"] for c in result["cases"]) else "PARITY_OPEN")
    (a.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Q4_SMALL_CONFIRM_COMPLETE "+result["performance_verdict"],flush=True)


if __name__=="__main__":
    main()

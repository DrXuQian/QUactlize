#!/usr/bin/env python3
"""Re-measure fixed Q4 FP32 winners in alternating order without searching."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

if __package__:
    from .compare_q4_native import parse_timing
    from .run_xplane_ncu import sha
else:
    from compare_q4_native import parse_timing
    from run_xplane_ncu import sha


SHAPES=((512,2048),(1024,5120),(4096,2048),(4096,4096),(5120,8192),(8192,5120))


def fixed_plan(receipt):
    if (receipt.get("status")!="PASS" or receipt.get("arithmetic")!=
            "BOTH_FP32_DOT_AND_REDUCTION_FP16_WEIGHT_AND_A_BOUNDARY"):
        raise ValueError("expected a complete FP32 comparison")
    expected={(n,k,mode) for n,k in SHAPES for mode in ("warm","rotating")}
    seen=set()
    rows=[]
    for case in receipt["cases"]:
        m,n,k=case["shape"]
        key=(n,k,case["mode"])
        if m!=1 or key not in expected or key in seen:
            raise ValueError("duplicate or unexpected confirmation shape/regime")
        seen.add(key)
        recipes={arm:r["recipe"] for arm,r in case["winners"].items()}
        if (set(recipes)!={"kpack","xplane"} or
                any(len(r)!=4 or r[0]!=arm for arm,r in recipes.items()) or
                recipes["xplane"][-1]!=1):
            raise ValueError("confirmation recipe arm differs")
        rows.append(dict(shape=case["shape"],mode=case["mode"],
                         recipes={arm:r[1:] for arm,r in recipes.items()},
                         fixture_sha256=case["fixture_sha256"]))
    if seen!=expected:
        raise ValueError("missing confirmation shape/regime")
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("runner","fixtures","xplane","candidate","recipes","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    receipt=json.loads(a.recipes.read_text())
    plan=fixed_plan(receipt)
    manifest=a.candidate.parent/"manifest.json"
    build=json.loads(manifest.read_text())
    inputs=dict(runner=a.runner,xplane=a.xplane,candidate=a.candidate,manifest=manifest,recipes=a.recipes)
    if any(sha(inputs[name])!=receipt["authority"][name]["sha256"]
           for name in ("runner","xplane","candidate","manifest")):
        raise ValueError("measured input identity differs")
    if build["library_sha256"]!=sha(a.candidate):
        raise ValueError("build library differs")
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(status="RUNNING",scope="FIXED_RECIPES_DENSE_Q4_M1_F16_FP32_NOT_PPU",
                source_sha256=sha(Path(__file__)),threshold_pct=5.0,
                authority={key:dict(path=str(path.resolve()),sha256=sha(path)) for key,path in inputs.items()},
                cases=[])
    device=None
    for row in plan:
        _,n,k=row["shape"]
        fixture=a.fixtures/f"q12-n{n}-k{k}-e1-c1.bin"
        if sha(fixture)!=row["fixture_sha256"]:
            raise ValueError("fixture differs from screened comparison")
        records={arm:[] for arm in row["recipes"]}
        for turn in range(6):
            for arm in (("xplane","kpack") if turn%2==0 else ("kpack","xplane")):
                command=[str(a.runner.resolve()),str(fixture.resolve()),str(a.xplane.resolve()),
                         str(a.candidate.resolve()),arm,*map(str,row["recipes"][arm]),row["mode"],"timing"]
                proc=subprocess.run(command,capture_output=True,text=True,timeout=180)
                log=a.output/f"n{n}-k{k}-{row['mode']}-{turn}-{arm}.log"
                log.write_text(proc.stdout+proc.stderr)
                if proc.returncode:
                    raise RuntimeError(f"fixed recipe failed: {log}")
                r=parse_timing(proc.stdout,arm,row["recipes"][arm],row["mode"],row["shape"])
                identity=(int(r["sm"]),int(r["L2_bytes"]))
                if device is not None and device!=identity:
                    raise ValueError("device geometry changed")
                device=identity
                records[arm].append(dict(median_us=float(r["median_us"]),samples_us=r["samples_us"],
                    error=float(r["error"]),log_sha256=sha(log)))
        med={arm:statistics.median(r["median_us"] for r in rr) for arm,rr in records.items()}
        delta=100*(med["kpack"]/med["xplane"]-1)
        result["cases"].append(dict(**row,records=records,median_us=med,delta_pct=delta,within_5pct=delta<=5))
        print("Q4_FIXED_CONFIRM "+json.dumps(dict(shape=row["shape"],mode=row["mode"],median_us=med,
            recipes=row["recipes"],delta_pct=delta)),flush=True)
    if any(sha(path)!=result["authority"][key]["sha256"] for key,path in inputs.items()):
        raise ValueError("confirmation inputs changed")
    result.update(status="PASS",device_geometry=dict(sm=device[0],l2_bytes=device[1]),
                  performance_verdict="WITHIN_5_PERCENT" if all(r["within_5pct"] for r in result["cases"]) else "PARITY_OPEN")
    (a.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Q4_FIXED_CONFIRM_COMPLETE "+result["performance_verdict"],flush=True)


if __name__=="__main__":
    main()

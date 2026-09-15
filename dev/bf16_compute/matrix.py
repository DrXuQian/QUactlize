"""Compile bounded BF16 parent coverage without a device or a tuning sweep."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--jobs",type=int,default=4)
    parser.add_argument("--kind",choices=("grouped","dense"),default="grouped")
    args=parser.parse_args()
    if not 1<=args.jobs<=8: parser.error("jobs must be in 1..8 for this bounded compile gate")
    args.output.mkdir(parents=True,exist_ok=False)
    cells=[(q,route,tm) for q in (8,10,11,12,13,14)
           for route in (("sf",) if q==8 else ("fq","sf"))
           for tm in ((8,64) if args.kind=="grouped" else (8,))]
    started=time.monotonic()
    def build(cell):
        q,route,tm=cell
        path=args.output/f"q{q}-{route}-tm{tm}.log"
        command=[sys.executable,str(ROOT/"dev/bf16_compute/build.py"),"--sdk",str(args.sdk),
                 "--cache",str(args.cache),"--qtype",str(q),"--kind",args.kind,
                 "--quant-route",route,"--tile-m",str(tm)]
        with path.open("w") as output:
            rc=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,cwd=ROOT).returncode
        print(f"BF16_COMPILE q={q} route={route} tm={tm} rc={rc} elapsed_s={time.monotonic()-started:.1f}",flush=True)
        record=dict(qtype=q,route=route,tile_m=tm,rc=rc,log=path.name)
        if not rc: record["module"]=json.loads(path.read_text().splitlines()[-1])
        return record
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results=list(pool.map(build,cells))
    summary=dict(schema="quactlize.bf16-compute.compile.v1",kind=args.kind,device_validated=False,
                 status="PASS" if all(r["rc"]==0 for r in results) else "FAIL",cells=results,
                 seconds=time.monotonic()-started)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"BF16_COMPILE_MATRIX status={summary['status']} cells={len(results)} device_validated=0")
    return int(summary["status"]!="PASS")


if __name__=="__main__":
    raise SystemExit(main())

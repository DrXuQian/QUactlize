#!/usr/bin/env python3
"""Exercise every new large-static S1 specialization, including zero-code negatives."""
import argparse
import json
from pathlib import Path
import subprocess

if __package__:
    from .run_standalone import parse, sha
else:
    from run_standalone import parse, sha


def cases():
    for n,k in ((4096,2048),(4096,4096),(5120,8192),(8192,5120)):
        for c in (4,8):
            for w in (4,5,8,10,16):
                yield n,k,c,w
    for n,k,warps in ((512,2048,(8,10)),(1024,5120,(5,10))):
        for w in warps:
            yield n,k,4,w


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("runner","fixtures","library","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    manifest=a.library.parent/"manifest.json"
    receipt=json.loads(manifest.read_text())
    if receipt["reader"]!="cuda-q4-n4-static" or receipt["library_sha256"]!=sha(a.library):
        raise ValueError("large-static library identity differs")
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(status="RUNNING",scope="DENSE_M1_F16_S1_ALL_40_LARGE_STATIC_PLUS_4_SMALL_REGRESSIONS",
                library_sha256=sha(a.library),manifest_sha256=sha(manifest),runner_sha256=sha(a.runner),
                source_sha256=sha(Path(__file__)),records=[])
    for n,k,c,w in cases():
        fixture=a.fixtures/f"q12-n{n}-k{k}-e1-c1.bin"
        command=[str(a.runner.resolve()),str(fixture.resolve()),str(a.library.resolve()),
                 "pair",str(c),str(w),"1","--f16"]
        proc=subprocess.run(command,capture_output=True,text=True,timeout=180)
        log=a.output/f"n{n}-k{k}-c{c}-w{w}.log"
        log.write_text(proc.stdout+proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"static numeric check failed: {log}")
        r=parse(proc.stdout)
        if (r["q"]!="12" or r["shape"]!=f"1x{n}x{k}" or r["experts"]!="1" or r["channels"]!="1" or
                r["kind"]!="pair" or r["config"]!=f"{c}-{w}-1" or
                proc.stdout.count("GEMV_STANDALONE_INPUT type=F16 a_offset_bytes=0 plane_offset_bytes=0")!=1):
            raise ValueError("static execution identity differs")
        result["records"].append(dict(shape=[1,n,k],recipe=[c,w,1],error=float(r["error"]),
            fixture_sha256=sha(fixture),log_sha256=sha(log),status="PASS"))
        print(f"Q4_STATIC_NUMERIC completed={len(result['records'])}/44 shape={n}x{k} recipe={c},{w},1",flush=True)
    if len(result["records"])!=44 or sha(a.library)!=result["library_sha256"]:
        raise ValueError("numeric denominator or library changed")
    result["status"]="PASS"
    (a.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Q4_STATIC_NUMERIC_COMPLETE status=PASS cases=44",flush=True)


if __name__=="__main__":
    main()

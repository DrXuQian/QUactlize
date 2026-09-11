#!/usr/bin/env python3
"""Independent indexed-GEMV checks on CUDA-only hosts without PyTorch.

This complements dense F16 timing checks. It covers eight selected experts,
F32 endpoints, output/workspace guards, graph replay and the runner's
wrong-expert and zero-code negatives. It is not a multi-token/unaligned gate.
"""
import argparse
import json
from pathlib import Path
import subprocess
from run_standalone import parse, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("runner","fixtures","library","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--f16",action="store_true")
    p.add_argument("--unaligned",action="store_true")
    a=p.parse_args()
    manifest=a.library.parent/"manifest.json"
    authority=json.loads(manifest.read_text())
    if authority["library_sha256"]!=sha(a.library):
        raise ValueError("library identity differs")
    columns=authority["q4_n_positions"]
    warps=authority.get("q4_warps",[2,4,8])
    extra_s1=authority.get("q4_s1_extra_warps",[])
    if columns!=[1,2,4,8,16,32] or warps!=[2,4,8,16] or extra_s1 not in ([],[5,10]):
        raise ValueError("expected the extended Q4 experiment domain")
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(status="RUNNING",scope="INDEXED_TOP8_SINGLE_TOKEN_NOT_ALL_INPUT_ABI",
                input_type="F16" if a.f16 else "F32",unaligned=a.unaligned,
                library_sha256=sha(a.library),manifest_sha256=sha(manifest),
                runner_sha256=sha(a.runner),source_sha256=sha(Path(__file__)),records=[])
    for q in range(10,15):
        fixture=a.fixtures/f"q{q}-n256-k512-e16-c8.bin"
        fixture_hash=sha(fixture)
        recipes=[(c,w,s) for c in columns for w in warps for s in (1,2,4,8)] if q==12 else [(16,4,8),(32,8,1)]
        if q==12:recipes += [(c,w,1) for c in columns for w in extra_s1]
        for c,w,s in recipes:
            command=[str(a.runner.resolve()),str(fixture.resolve()),str(a.library.resolve()),"pair",str(c),str(w),str(s)]
            if a.f16:command.append("--f16")
            if a.unaligned:command.append("--unaligned")
            proc=subprocess.run(command,capture_output=True,text=True,timeout=60)
            log=a.output/f"q{q}-{c}-{w}-{s}.log"
            log.write_text(proc.stdout+proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"device check failed: {log}")
            row=parse(proc.stdout)
            input_line=f"GEMV_STANDALONE_INPUT type={'F16' if a.f16 else 'F32'} a_offset_bytes={(2 if a.f16 else 4) if a.unaligned else 0} plane_offset_bytes={2 if a.unaligned else 0}"
            if proc.stdout.count(input_line)!=1:
                raise ValueError("input control was not executed")
            if (row["q"]!=str(q) or row["shape"]!="8x256x512" or
                row["experts"]!="16" or row["channels"]!="8" or
                row["kind"]!="pair" or row["config"]!=f"{c}-{w}-{s}"):
                raise ValueError("device result identity differs")
            result["records"].append(dict(q=q,recipe=[c,w,s],fixture_sha256=fixture_hash,
                error=float(row["error"]),status="PASS",log_sha256=sha(log)))
        print(f"GEMV_STANDALONE_INPUT_PROGRESS q={q} passed={len(recipes)}",flush=True)
    expected=104+len(columns)*len(extra_s1)
    if len(result["records"])!=expected or sha(a.library)!=result["library_sha256"]:
        raise ValueError("numeric denominator or library changed")
    result["status"]="PASS"
    (a.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"GEMV_STANDALONE_INPUT_COMPLETE status=PASS cells={expected} formats=5",flush=True)


if __name__=="__main__":
    main()

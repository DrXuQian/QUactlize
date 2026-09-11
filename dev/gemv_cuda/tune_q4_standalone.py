#!/usr/bin/env python3
"""Independently retune Q4 readers on a CUDA-only host (no PyTorch needed).

The runner checks each recipe against the independent GGUF dot. Timings
include any reducer, exclude graph upload, and separate warm and >L2 weights.
The named Xplane arithmetic receipt distinguishes the FP32 comparator from
the historical half-partial control. K-pack normally retains FP32 accumulation.
"""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time

if __package__:
    from .compare_q4_native import parse_timing
    from .run_xplane_ncu import sha
else:
    from compare_q4_native import parse_timing
    from run_xplane_ncu import sha


def recipes(columns, warps=(2,4,8), extra_s1=()):
    xp = [("xplane", c, w, 1) for c in (1,2,4,8) for w in (2,4,8)]
    kp = [("kpack", c, w, s) for c in columns for w in warps for s in (1,2,4,8)]
    kp += [("kpack",c,w,1) for c in columns for w in extra_s1]
    return sorted(xp + kp, key=lambda x: (x[2], x[3], x[0], x[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("runner", "fixtures", "xplane", "candidate", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--expected-reader", required=True)
    p.add_argument("--batch-runner", action="store_true",
                   help="runner supports reusing one fixture/allocation for a batch of recipes")
    p.add_argument("--xplane-arithmetic", choices=("fp16-group", "fp32"), default="fp16-group")
    args = p.parse_args()
    manifest_path = args.candidate.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    columns = manifest.get("q4_n_positions")
    warps=manifest.get("q4_warps",[2,4,8])
    expected_warps=[2,4,8,16] if args.expected_reader in ("cuda-q4-n4-tree","cuda-q4-n2-tree","cuda-q4-n2-warp","cuda-q4-n4-coop","cuda-q4-small-static","cuda-q4-small-balanced") else [2,4,8]
    if args.expected_reader=="cuda-q4-n4-static": expected_warps=[2,4,8,16]
    extra_s1=manifest.get("q4_s1_extra_warps",[])
    if (manifest.get("reader") != args.expected_reader or
        warps!=expected_warps or
        extra_s1!=([5,10] if args.expected_reader in ("cuda-q4-small-balanced","cuda-q4-n4-static") else []) or
        columns not in ([16,32], [4,8,16,32], [1,2,4,8,16,32]) or
        manifest.get("pair_column_values_per_thread") != 2 or
        manifest.get("library") != args.candidate.name or
        manifest.get("library_sha256") != sha(args.candidate)):
        raise ValueError("reader / recipe domain / library identity differs")
    xp_receipt = None
    if args.xplane_arithmetic == "fp32":
        if args.expected_reader == "cuda-q4-n2-half-control":
            raise ValueError("K-pack FP16 partial is not an FP32 comparator")
        xp_receipt=args.xplane.parent.parent/"manifest.json"
        arms=json.loads(xp_receipt.read_text())["arms"]
        matches=[r for r in arms if r["arm"]=="fp32-control"]
        if len(matches)!=1 or matches[0]["library_sha256"]!=sha(args.xplane):
            raise ValueError("Xplane FP32 arithmetic control identity differs")
    fixtures = sorted(args.fixtures.glob("*.bin"))
    if not fixtures:
        raise ValueError("no fixtures")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = dict(runner=args.runner, xplane=args.xplane, candidate=args.candidate,
                 manifest=manifest_path)
    if xp_receipt is not None:
        paths["xplane_arithmetic_manifest"]=xp_receipt
    result = dict(status="RUNNING", scope="INDEPENDENT_SAME_GPU_RECIPE_SEARCH_NOT_PPU",
                  arithmetic=("BOTH_FP32_DOT_AND_REDUCTION_FP16_WEIGHT_AND_A_BOUNDARY"
                              if args.xplane_arithmetic == "fp32" else
                              "BOTH_FP16_GROUP_PARTIAL_FP32_CROSS_GROUP_NOT_IDENTICAL_SUM_ORDER"
                              if args.expected_reader == "cuda-q4-n2-half-control" else
                              "XPLANE_FP16_GROUP_PARTIAL_VS_KPACK_FP32_DOT"),
                  authority={k: dict(path=str(v.resolve()), sha256=sha(v)) for k,v in paths.items()},
                  source_sha256=sha(Path(__file__)), cases=[])
    started = time.monotonic()
    device = None
    for fixture in fixtures:
        with fixture.open("rb") as f:
            magic, version, q, n, k, experts, rows, channels, reserved = struct.unpack("<Q8i", f.read(40))
        if (magic != 0x3146584D5647514B or version != 1 or reserved or
            (q,experts,rows,channels) != (12,1,1,1)):
            raise ValueError("not an independent dense Q4 M1 fixture")
        shape = [1,n,k]
        for mode in ("warm", "rotating"):
            prefix = f"n{n}-k{k}-{mode}"
            def run(recipe, label):
                nonlocal device
                arm,c,w,s = recipe
                command = [str(args.runner.resolve()), str(fixture.resolve()), str(args.xplane.resolve()),
                           str(args.candidate.resolve()), arm, str(c), str(w), str(s), mode, "timing"]
                proc = subprocess.run(command, text=True, capture_output=True, timeout=180)
                log = args.output / f"{prefix}-{label}-{arm}-{c}-{w}-{s}.log"
                log.write_text(proc.stdout + proc.stderr)
                if proc.returncode:
                    raise RuntimeError(f"recipe failed: {log}")
                r = parse_timing(proc.stdout, arm, [c,w,s], mode, shape)
                identity = r["sm"],r["L2_bytes"]
                if device is not None and device != identity:
                    raise ValueError("device geometry changed")
                device = identity
                return dict(recipe=list(recipe), median_us=float(r["median_us"]),
                            samples_us=r["samples_us"], error=float(r["error"]), copies=int(r["copies"]))
            def batch(planned,label):
                nonlocal device
                arm=planned[0][0]
                if any(r[0]!=arm for r in planned):
                    raise ValueError("batch may only contain one reader arm")
                encoded=",".join(":".join(map(str,r[1:])) for r in planned)
                command=[str(args.runner.resolve()),str(fixture.resolve()),str(args.xplane.resolve()),
                         str(args.candidate.resolve()),arm,encoded,"0","0",mode,"batch"]
                proc=subprocess.run(command,text=True,capture_output=True,timeout=180)
                log=args.output/f"{prefix}-{label}-{arm}-batch.log"
                log.write_text(proc.stdout+proc.stderr)
                lines=[line for line in proc.stdout.splitlines() if line.startswith("Q4_LAYOUT_PROFILE ")]
                if proc.returncode or len(lines)!=len(planned):
                    raise RuntimeError(f"batched runner failed or record count differs: {log}")
                records=[]
                for recipe,line in zip(planned,lines):
                    r=parse_timing(line,arm,recipe[1:],mode,shape)
                    identity=r["sm"],r["L2_bytes"]
                    if device is not None and device!=identity:
                        raise ValueError("device geometry changed")
                    device=identity
                    records.append(dict(recipe=list(recipe),median_us=float(r["median_us"]),
                        samples_us=r["samples_us"],error=float(r["error"]),copies=int(r["copies"])))
                return records
            screen=[]
            if args.batch_runner:
                for arm in ("xplane","kpack"):
                    screen.extend(batch([r for r in recipes(columns,warps,extra_s1) if r[0]==arm],"screen"))
                    print(f"Q4_RETUNE_PROGRESS shape={shape} mode={mode} screen={len(screen)}/{len(recipes(columns,warps,extra_s1))} batched=1 elapsed_s={time.monotonic()-started:.1f}",flush=True)
            else:
                for i,recipe in enumerate(recipes(columns,warps,extra_s1)):
                    screen.append(run(recipe,"screen"))
                    if (i+1)%12==0:
                        print(f"Q4_RETUNE_PROGRESS shape={shape} mode={mode} screen={i+1}/{len(recipes(columns,warps,extra_s1))} elapsed_s={time.monotonic()-started:.1f}",flush=True)
            contenders=[]
            for arm in ("xplane","kpack"):
                top=sorted((r for r in screen if r["recipe"][0]==arm),key=lambda r:r["median_us"])[:2]
                if arm=="kpack" and not any(r["recipe"][-1]==1 for r in top):
                    top.append(min((r for r in screen if r["recipe"][0]==arm and r["recipe"][-1]==1),
                                   key=lambda r:r["median_us"]))
                contenders.extend(dict(recipe=r["recipe"], rounds=[]) for r in top)
            for turn in range(4):
                ordered=contenders if turn%2==0 else list(reversed(contenders))
                if args.batch_runner:
                    for arm in (("xplane","kpack") if turn%2==0 else ("kpack","xplane")):
                        group=[c for c in ordered if c["recipe"][0]==arm]
                        records=batch([c["recipe"] for c in group],f"confirm{turn}")
                        for cell,record in zip(group,records):
                            cell["rounds"].append(record)
                else:
                    for cell in ordered:
                        cell["rounds"].append(run(cell["recipe"],f"confirm{turn}"))
            for cell in contenders:
                cell["median_us"]=statistics.median(r["median_us"] for r in cell["rounds"])
            winners={arm:min((r for r in contenders if r["recipe"][0]==arm),key=lambda r:r["median_us"])
                     for arm in ("xplane","kpack")}
            delta=100*(winners["kpack"]["median_us"]/winners["xplane"]["median_us"]-1)
            s1=min((r for r in contenders if r["recipe"][0]=="kpack" and r["recipe"][-1]==1),
                   key=lambda r:r["median_us"])
            cell=dict(shape=shape, mode=mode, fixture_sha256=sha(fixture), screen=screen,
                      confirmed=contenders, winners=winners, delta_pct=delta,s1_best=s1,
                      s1_delta_pct=100*(s1["median_us"]/winners["xplane"]["median_us"]-1))
            result["cases"].append(cell)
            (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
            print("Q4_RETUNE_RESULT "+json.dumps(dict(shape=shape,mode=mode,
                winners={k:(v["recipe"],v["median_us"]) for k,v in winners.items()},delta_pct=delta)),flush=True)
    if any(sha(v)!=result["authority"][k]["sha256"] for k,v in paths.items()):
        raise ValueError("inputs changed during measurement")
    result.update(status="PASS",elapsed_seconds=time.monotonic()-started,
                  device_geometry=dict(sm=int(device[0]),l2_bytes=int(device[1])))
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Q4_RETUNE_COMPLETE status=PASS",flush=True)


if __name__ == "__main__":
    main()

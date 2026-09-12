#!/usr/bin/env python3
"""Batch-screen Q4 C/W/P configs, then confirm a shortlist against frozen readers."""
import argparse
import csv
import ctypes as C
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.config_space import SHAPES,inventory,lookup,payload,plan,verify
from dev.gemv_ppu.access_pattern import analyze,observed_patterns,warp_pattern
from dev.gemv_ppu.campaign import parse_cells
from dev.gemv_ppu.run import checked
from dev.gemv_ppu.run_bload import ExperimentBench
from dev.gemv_ppu.run_cold_geometry import probe_device
from dev.gemv_ppu.run_cold_shapes import C8Bench,implementation as old_implementation,recipe as old_recipe
from dev.gemv_ppu.run_h800_port import PortBench
from tools.profile_kpack_gpu_compact import acu_launch_command

BASELINES=ROOT/"docs/measurements/q4_cold_shapes_20260912/selection.json"
PREFIX="Q4_CONFIG_CELL "
FAIL_PREFIX="Q4_CONFIG_FAILURE "
ANCHORS=("xplane","raw-reference","baseline")
ROUNDS=6
SCREEN_SAMPLES=5
CONFIRM_SAMPLES=15


def baselines():
    r=json.loads(BASELINES.read_text());result={}
    if r.get("schema")!="quactlize.q4-cold-config-baselines.v1":raise ValueError("baseline receipt schema differs")
    for row in r["cases"]:
        _,n,k=row["shape"];arm=row["baseline"];times=row["median_us"]
        if ((n,k) in result or (n,k) not in SHAPES or arm not in ("kpack-current","kpack-c8") or
                arm!=min(("kpack-current","kpack-c8"),key=times.get) or
                row["recipe"]!=list(old_recipe(arm,n,k)) or row["implementation"]!=old_implementation(arm,n,k)):
            raise ValueError("historical winning config is not preserved")
        result[n,k]=row
    if set(result)!=set(SHAPES):raise ValueError("baseline shape denominator differs")
    return result


def recipe(variant,n,k,key):
    if variant=="config":return lookup(n,k,key).args
    if key!="control" or variant not in ANCHORS:raise ValueError("unknown comparison arm")
    arm=baselines()[n,k]["baseline"] if variant=="baseline" else variant
    return old_recipe(arm,n,k)


class SweepBench(ExperimentBench):
    def __init__(self,args):
        super().__init__(SimpleNamespace(**(vars(args)|dict(variant="kpack-current",arm="new"))))
        self.library=C.CDLL(str(args.candidate/payload(self.n,self.k)),mode=C.RTLD_LOCAL)
        probe=self.library.q4_ppu_probe
        probe.argtypes=[C.POINTER(C.c_int)]*3+[C.c_char_p];probe.restype=C.c_int
        l2,sm,warp,name=C.c_int(),C.c_int(),C.c_int(),C.create_string_buffer(256)
        checked(probe(C.byref(l2),C.byref(sm),C.byref(warp),name),"config image marker")
        if (sm.value,warp.value,name.value.decode())!=(self.device["sm"],self.device["warp"],self.device["name"]):
            raise ValueError("config DSO device differs")
        self.config_launch=getattr(self.library,f"q4_config_run_{self.n}_{self.k}")
        self.config_launch.argtypes=[C.c_int]*3+[C.c_void_p]*5;self.config_launch.restype=C.c_int
        self.allowed={c.args for c in inventory(self.n,self.k)}

    def invoke(self,cfg,index=0,force_aiu=None):
        if force_aiu is not None or tuple(cfg) not in self.allowed:
            raise ValueError("config launch parameters differ")
        low,units=self.weight_pointers[index%self.copies]
        return self.config_launch(*cfg,self.a,low,units,self.output,self.r.stream)


def child(a):
    verify(a.candidate,a.previous,a.controls,a.bundle,sources=False)
    wanted=json.loads(a.keys)
    if not wanted or len(set(wanted))!=len(wanted):raise ValueError("duplicate/empty child batch")
    bench=None;failed=False
    try:
        if a.variant=="config":bench=SweepBench(a)
        else:
            with np.load(a.fixture,allow_pickle=False) as f:n,k=int(f['n']),int(f['k'])
            arm=baselines()[n,k]["baseline"] if a.variant=="baseline" else a.variant
            args=SimpleNamespace(**(vars(a)|dict(candidate=a.previous if arm=="kpack-c8" else a.controls,
                 variant="kpack" if arm=="kpack-current" else arm)))
            bench=C8Bench(args) if arm=="kpack-c8" else PortBench(args)
        n,k=bench.n,bench.k
        for key in wanted:
            try:
                r=bench.measure(list(recipe(a.variant,n,k,key)))
                impl="affine-fast" if a.variant=="config" else (
                    baselines()[n,k]["implementation"] if a.variant=="baseline" else a.variant)
                r.update(arm=a.variant,variant=a.variant,config_key=key,phase=a.phase,implementation=impl,
                    output_type="F32",inter_cta_split=1,launches_per_call=1,
                    weight_arithmetic="FP32_GROUP_AFFINE" if impl.startswith("affine") else "PER_WEIGHT_FP16",
                    timing_scope="RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT",
                    cache_scope="ACU_FORCED_COLD" if a.profile else "ROTATING_AT_LEAST_2_25_L2")
                if a.variant=="config":
                    r["geometry"]=lookup(n,k,key).geometry(n,k)
                    r["dequant"]="LOP3_HALF2_CODES_FP32_GROUP_AFFINE"
                    r["observed_warp_patterns"]=observed_patterns(lookup(n,k,key),n,k,bench.a,bench.weight_pointers)
                if a.variant=="baseline":r["baseline_arm"]=baselines()[n,k]["baseline"]
                print(PREFIX+json.dumps(r),flush=True)
            except Exception as exc:
                failed=True
                print(FAIL_PREFIX+json.dumps(dict(variant=a.variant,key=key,shape=[1,n,k],phase=a.phase,error=str(exc))),flush=True)
                traceback.print_exc()
                break  # preserve prior cells; retry remaining configs in a fresh context
    finally:
        if bench:bench.close()
    return 1 if failed else 0


def parse_row(raw,variant,n,k,key,phase,samples,*,profile=False):
    r=parse_cells("Q4_PPU_CELL "+json.dumps(raw),variant,[recipe(variant,n,k,key)],[1,n,k],"rotating",samples)[0]
    impl="affine-fast" if variant=="config" else baselines()[n,k]["implementation"] if variant=="baseline" else variant
    if (r.get("variant")!=variant or r.get("config_key")!=key or r.get("phase")!=phase or
            r.get("zero_a_check")!="PASS" or r.get("output_type")!="F32" or r.get("implementation")!=impl or
            r.get("inter_cta_split")!=1 or r.get("launches_per_call")!=1 or
            r.get("timing_scope")!="RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT" or
            r.get("cache_scope")!=("ACU_FORCED_COLD" if profile else "ROTATING_AT_LEAST_2_25_L2") or
            r.get("weight_arithmetic")!=("FP32_GROUP_AFFINE" if impl.startswith("affine") else "PER_WEIGHT_FP16") or
            r.get("storage")!=("XPLANE" if variant=="xplane" else "RAW_GGUF" if variant=="raw-reference" else "CANONICAL_KPACK4")):
        raise ValueError("config result identity/precision/cache differs")
    if (profile and r.get("median_us") is not None) or (not profile and not math.isfinite(r["median_us"])):
        raise ValueError("nonfinite or profiler duration is not a timing sample")
    if variant=="config" and (r.get("geometry")!=lookup(n,k,key).geometry(n,k) or
                              r.get("dequant")!="LOP3_HALF2_CODES_FP32_GROUP_AFFINE"):
        raise ValueError("compiled config/decoder differs")
    if variant=="config":
        patterns=r.get("observed_warp_patterns",[]);seen=set()
        if not patterns:raise ValueError("missing actual pointer alignment/address pattern")
        for row in patterns:
            bases=row.get("base_mod128",{})
            if row!=warp_pattern(lookup(n,k,key),n,k,base_mod128=bases) or any(v%16 for v in bases.values()):
                raise ValueError("observed address/coalescing model differs")
            identity=tuple(bases[x] for x in ("A","B","metadata"))
            if identity in seen:raise ValueError("duplicate alignment case")
            seen.add(identity)
    if variant=="baseline" and r.get("baseline_arm")!=baselines()[n,k]["baseline"]:
        raise ValueError("historical winner was replaced")
    return r


def shortlist(rows,top_k):
    return [key for key in sorted(rows,key=lambda key:(rows[key]["median_us"],key))[:top_k]]


def execute(cmd,stream):
    """Keep full child output and expose progress without owning a GPU context."""
    count=0
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
        for line in proc.stdout:
            stream.write(line);stream.flush()
            if line.startswith(PREFIX):
                count+=1
                if count%8==0:
                    r=json.loads(line[len(PREFIX):])
                    print(f"Q4_CONFIG_BATCH_PROGRESS phase={r['phase']} completed={count} current={r['config_key']}",flush=True)
            elif line.startswith(FAIL_PREFIX):print(line.rstrip(),flush=True)
        return proc.wait()


def summarize(n,k,screen,records,shortlisted):
    medians={key:statistics.median(r["median_us"] for r in rows) for key,rows in records.items() if len(rows)==ROUNDS}
    required={*ANCHORS,*shortlisted}
    full=(len(screen)==len(inventory(n,k)) and len(shortlisted)>0 and set(medians)==required)
    r=dict(shape=[1,n,k],status="PASS" if full else "INCOMPLETE",median_us=medians,
           selected=None,reference_verdict="INCOMPLETE",xplane_within_5pct=None)
    available=[key for key in ("baseline",*shortlisted) if key in medians]
    if available:r["selected"]=min(available,key=medians.get)
    if full:
        us=medians[r["selected"]]
        r.update(reference_delta_pct=100*(us/medians["raw-reference"]-1),
                 baseline_delta_pct=100*(us/medians["baseline"]-1),
                 reference_verdict="NOT_SLOWER_THAN_REFERENCE" if us<=medians["raw-reference"] else "REF_REGRESSION",
                 xplane_within_5pct=us<=1.05*medians["xplane"],
                 selected_config=lookup(n,k,r["selected"]).geometry(n,k) if r["selected"]!="baseline" else None)
    return r


def run(a):
    m=verify(a.candidate,a.previous,a.controls,a.bundle)
    old=json.loads(BASELINES.read_text())["authority"]
    for key,path in (("candidate",a.previous),("controls",a.controls),("baseline",a.bundle)):
        if old[key]!=digest(path/"manifest.json"):raise ValueError("baseline image authority differs")
    baselines();device=probe_device(a)
    inputs=["dev/gemv_ppu/run_config_sweep.py","dev/gemv_ppu/access_pattern.py",
            "dev/gemv_ppu/run_cold_shapes.py","dev/gemv_ppu/run_cold_geometry.py","dev/gemv_ppu/run.py",
            "dev/gemv_ppu/campaign.py","dev/gemv_ppu/run_bload.py","dev/gemv_ppu/run_h800_port.py",
            "tools/run_kpack_gemv_gate.py","tools/run_kpack_grouped_decode_probe.py","tools/profile_kpack_gpu_compact.py"]
    authority=dict(schema="quactlize.q4-config-sweep-run.v1",candidate=digest(a.candidate/"manifest.json"),
        baselines=digest(BASELINES),device=device,top_k=a.top_k,rounds=ROUNDS,screen_samples=SCREEN_SAMPLES,
        confirm_samples=CONFIRM_SAMPLES,reference_regression_limit_pct=0,
        fixtures={f"{n}x{k}":digest(a.fixtures/f"q12-n{n}-k{k}-e1-c1.npz") for n,k in SHAPES},
        runtime={name:digest(a.sdk/"lib"/name) for name in m["runtime"]},sources={name:digest(ROOT/name) for name in inputs})
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/"authority.json"
    if path.exists() and json.loads(path.read_text())!=authority:raise ValueError("resume identity differs; use a new output")
    path.write_text(json.dumps(authority,indent=2)+"\n")
    differences=[name for name,sha in authority["runtime"].items() if sha!=m["runtime"][name]]
    if differences:print("Q4_CONFIG_SDK_DIFFERENCE recorded="+",".join(differences)+" real_marker_and_numeric_required=1",flush=True)
    (a.output/"plan.json").write_text(json.dumps(plan(),indent=2)+"\n")
    (a.output/"isa-stats.json").write_bytes((a.candidate/"isa-stats.json").read_bytes())
    (a.output/"build-manifest.json").write_bytes((a.candidate/"manifest.json").read_bytes())
    patterns=[dict(shape=[1,n,k],patterns=[analyze(c,n,k) for c in inventory(n,k)]) for n,k in SHAPES]
    (a.output/"access-patterns.json").write_text(json.dumps(patterns)+"\n")
    report=dict(status="RUNNING",cases=[],failures=[],profiles=[],production_changed=False)
    started=time.monotonic()
    def save():(a.output/"summary.json").write_text(json.dumps(report)+"\n")
    def batch(variant,n,k,keys,phase,*,profile=False):
        receipt=a.output/f"n{n}-k{k}-{variant}-{phase}.json"
        cache=json.loads(receipt.read_text()) if receipt.exists() else {}
        samples=0 if profile else SCREEN_SAMPLES if phase=="screen" else CONFIRM_SAMPLES
        parsed_logs={}
        for key,entry in cache.items():
            log=a.output/entry["log"]
            if log.parent!=a.output:raise ValueError("cached log path differs")
            if log not in parsed_logs:
                rows=[json.loads(s[len(PREFIX):]) for s in log.read_text().splitlines() if s.startswith(PREFIX)]
                parsed_logs[log]=(digest(log),rows)
            sha,rows=parsed_logs[log]
            if sha!=entry["log_sha256"]:raise ValueError("cached log hash differs")
            matching=[r for r in rows if r.get("config_key")==key]
            if len(matching)!=1:raise ValueError("cached result missing/duplicate in log")
            r=parse_row(matching[0],variant,n,k,key,phase,samples,profile=profile)
            if r!=entry["row"] or r["device"]!=device:raise ValueError("cached result/device differs")
            if profile:
                f=a.output/entry["report"]
                if f.parent!=a.output or digest(f)!=entry["report_sha256"]:raise ValueError("cached ACU differs")
        missing=[key for key in keys if key not in cache]
        while missing:
            prefix=a.output/f"n{n}-k{k}-{variant}-{phase}.{time.time_ns()}"
            log=prefix.with_name(prefix.name+(".acu.log" if profile else ".log"))
            cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--child","--sdk",str(a.sdk),
                 "--candidate",str(a.candidate),"--previous",str(a.previous),"--controls",str(a.controls),
                 "--bundle",str(a.bundle),"--fixture",str(a.fixtures/f"q12-n{n}-k{k}-e1-c1.npz"),
                 "--variant",variant,"--keys",json.dumps(missing),"--phase",phase,"--samples",str(samples if samples else 3),
                 "--l2-bytes",str(a.l2_bytes)]
            if profile:cmd=acu_launch_command(a.acu,prefix,cmd+["--profile"])
            print(f"Q4_CONFIG_PROGRESS shape=1x{n}x{k} arm={variant} phase={phase} configs={len(missing)} elapsed_s={time.monotonic()-started:.1f}",flush=True)
            with log.open("x") as stream:rc=execute(cmd,stream)
            consumed=set();failed_keys=set()
            for line in log.read_text().splitlines():
                if line.startswith(PREFIX):
                    raw=json.loads(line[len(PREFIX):]);key=raw.get("config_key")
                    if key not in missing or key in consumed:raise ValueError("unexpected/duplicate child config")
                    consumed.add(key)
                    try:r=parse_row(raw,variant,n,k,key,phase,samples,profile=profile)
                    except Exception as exc:
                        report["failures"].append(dict(shape=[1,n,k],variant=variant,key=key,phase=phase,error=str(exc),log=log.name))
                        failed_keys.add(key);continue
                    if r["device"]!=device:raise ValueError("physical device changed")
                    entry=dict(row=r,log=log.name,log_sha256=digest(log))
                    if profile:
                        files=list(a.output.glob(prefix.name+"*.acurep"))
                        if len(files)!=1:raise ValueError("missing ACU report")
                        entry.update(report=files[0].name,report_sha256=digest(files[0]))
                    cache[key]=entry
                elif line.startswith(FAIL_PREFIX):
                    error=json.loads(line[len(FAIL_PREFIX):]);key=error["key"]
                    if (key not in missing or key in consumed or error["shape"]!=[1,n,k] or
                            error.get("variant")!=variant or error.get("phase")!=phase):raise ValueError("invalid failure identity")
                    consumed.add(key);failed_keys.add(key)
                    report["failures"].append(error|dict(log=log.name))
            if not consumed or (rc and not failed_keys):raise ValueError(f"child infrastructure failure rc={rc}; log={log}")
            receipt.write_text(json.dumps(cache)+"\n")
            missing=[key for key in missing if key not in consumed]
            if missing:print(f"Q4_CONFIG_RESTART remaining={len(missing)} validated={len(cache)} failed={len(failed_keys)}",flush=True)
        return {key:cache[key]["row"] for key in keys if key in cache}
    for n,k in SHAPES:
        case=dict(shape=[1,n,k],screen={},records={a:[] for a in ANCHORS},shortlist=[]);report["cases"].append(case)
        try:case["screen"]=batch("config",n,k,[c.key for c in inventory(n,k)],"screen")
        except Exception as exc:report["failures"].append(dict(shape=[1,n,k],phase="screen",error=str(exc)))
        case["shortlist"]=shortlist(case["screen"],a.top_k)
        case["records"].update({key:[] for key in case["shortlist"]})
        for turn in range(ROUNDS):
            groups=(*ANCHORS,"config")
            for variant in groups if turn%2==0 else groups[::-1]:
                keys=(case["shortlist"] if turn%2==0 else case["shortlist"][::-1]) if variant=="config" else ["control"]
                if not keys:continue
                try:
                    rows=batch(variant,n,k,keys,f"r{turn}")
                    for key,row in rows.items():case["records"][key if variant=="config" else variant].append(row)
                except Exception as exc:report["failures"].append(dict(shape=[1,n,k],variant=variant,phase=f"r{turn}",error=str(exc)))
                save()
        case["comparison"]=summarize(n,k,case["screen"],case["records"],case["shortlist"])
        print("Q4_CONFIG_RESULT "+json.dumps(case["comparison"]),flush=True)
        if not a.skip_acu:
            config_times={key:case["comparison"]["median_us"][key] for key in case["shortlist"] if key in case["comparison"]["median_us"]}
            profiled=[(v,"control") for v in ANCHORS]
            if config_times:profiled.append(("config",min(config_times,key=config_times.get)))
            for variant,key in profiled:
                try:
                    rows=batch(variant,n,k,[key],"profile",profile=True)
                    if key not in rows:raise ValueError("profiled config failed")
                    report["profiles"].append(dict(shape=[1,n,k],variant=variant,key=key,status="PASS",row=rows[key]))
                except Exception as exc:report["profiles"].append(dict(shape=[1,n,k],variant=variant,key=key,status="FAIL",error=str(exc)))
                save()
    verify(a.candidate,a.previous,a.controls,a.bundle)
    if (digest(BASELINES)!=authority["baselines"] or any(digest(ROOT/name)!=sha for name,sha in authority["sources"].items()) or
            any(digest(a.sdk/"lib"/name)!=sha for name,sha in authority["runtime"].items())):raise ValueError("authority changed during run")
    ok=not report["failures"] and all(c["comparison"]["status"]=="PASS" for c in report["cases"]) and all(p["status"]=="PASS" for p in report["profiles"])
    report.update(status="PASS" if ok else "INCOMPLETE",seconds=time.monotonic()-started,
                  not_slower_than_ref=sum(c["comparison"]["reference_verdict"]=="NOT_SLOWER_THAN_REFERENCE" for c in report["cases"]))
    save()
    with (a.output/"summary.tsv").open("w") as stream:
        writer=csv.writer(stream,delimiter="\t");writer.writerow(["N","K","selected","best_us","ref_us","delta_ref_pct","verdict"])
        for c in report["cases"]:
            r=c["comparison"];times=r["median_us"]
            writer.writerow([*c["shape"][1:],r["selected"],times.get(r["selected"],"NA"),times.get("raw-reference","NA"),r.get("reference_delta_pct","NA"),r["reference_verdict"]])
    (a.output/"config-winners.json").write_text(json.dumps(dict(production_changed=False,authority=authority,
        winners=[c["comparison"] for c in report["cases"]]),indent=2)+"\n")
    print(f"Q4_CONFIG_COMPLETE status={report['status']} not_slower_than_ref={report['not_slower_than_ref']}/6 results={a.output}",flush=True)
    return 0 if ok else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk",type=Path,required=True)
    p.add_argument("--candidate",type=Path,default=ROOT/"prebuilt/ppu0010/q4-config-sweep-v1")
    p.add_argument("--previous",type=Path,default=ROOT/"prebuilt/ppu0010/q4-cold-shapes-v1")
    p.add_argument("--controls",type=Path,default=ROOT/"prebuilt/ppu0010/q4-h800-port-v1")
    p.add_argument("--bundle",type=Path,default=ROOT/"prebuilt/ppu0010/q4-simt-ab-v1")
    p.add_argument("--fixtures",type=Path);p.add_argument("--fixture",type=Path);p.add_argument("--output",type=Path)
    p.add_argument("--variant",choices=(*ANCHORS,"config"));p.add_argument("--keys");p.add_argument("--phase",default="screen")
    p.add_argument("--samples",type=int,default=SCREEN_SAMPLES);p.add_argument("--top-k",type=int,default=3)
    p.add_argument("--l2-bytes",type=int,default=0);p.add_argument("--child",action="store_true")
    p.add_argument("--profile",action="store_true");p.add_argument("--skip-acu",action="store_true");p.add_argument("--acu",type=Path)
    a=p.parse_args();a.mode="rotating"
    if not 1<=a.top_k<=8 or a.samples<3 or a.l2_bytes<0:p.error("invalid sample/top-k/L2 setting")
    for name in ("sdk","candidate","previous","controls","bundle"):setattr(a,name,getattr(a,name).resolve(strict=True))
    if a.child:
        if a.fixture is None or a.variant is None or a.keys is None:p.error("child requires fixture/variant/keys")
        a.fixture=a.fixture.resolve(strict=True);return child(a)
    if a.fixtures is None or a.output is None:p.error("fixtures and output are required")
    a.fixtures=a.fixtures.resolve(strict=True);a.output=a.output.resolve();a.acu=a.acu or a.sdk/"asight/bin/acu"
    return run(a)


if __name__=="__main__":
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)

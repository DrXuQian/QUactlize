"""Six-shape experiment contracts; execution and speed still require a PPU."""
import ast
import json
import math
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_cuda.build import digest
from dev.gemv_ppu.cold_geometry import source as single_shape_source
from dev.gemv_ppu.cold_shapes import ROOT, PAYLOAD, SHAPES, RECIPE, geometry, source, verify
from dev.gemv_ppu.run_cold_shapes import (ARMS, MODE, PREFIX, ROUNDS, C8Bench, implementation,
                                       parse_result, recipe, summarize, validate_entry)


def test_all_six_shapes_use_the_exact_original_kernel_body():
    old, new = single_shape_source(), source()
    prefix = old[:old.index('extern "C" int q4_cold_c8_run(')]
    assert new.startswith(prefix)
    entry = new[len(prefix):]
    assert entry.count(">>>(") == 6 and len(set(SHAPES)) == 6
    for n, k in SHAPES:
        assert f"if(n=={n} && k=={k})" in entry
        assert f"q4_group_affine<8,8,4,{n},{k},true><<<n/32,256,0," in entry
    assert "return QKG_SHAPE" in entry and "hggcGetLastError" in entry
    assert "workspace" not in entry and "AIU" not in entry


@pytest.mark.parametrize("n,k",SHAPES)
def test_c8_k_coverage_and_output_column_ownership(n,k):
    g = geometry(n,k)
    visits = np.zeros((k//32,32),dtype=int)
    for tid in range(256):
        col = (tid%8)*4
        for group in range(tid//8,k//32,32):
            visits[group,col:col+4] += 1
    assert np.all(visits==1)
    assert g["grid"]*g["tile_n"]==n and g["last_pass_workers"]==32
    assert g["k_passes"]==k//1024
    assert recipe("kpack-c8",n,k)==RECIPE
    # The old small and medium winners must not be replaced by a global C4/W8.
    assert recipe("kpack-current",512,2048)==(1,16,1)
    assert recipe("kpack-current",1024,5120)==(2,10,1)


def record(arm="kpack-c8",n=8192,k=5120,samples=3,*,profile=False):
    impl = implementation(arm,n,k)
    weight_bytes = n*k*9//16
    l2 = 67108864
    copies = math.ceil(2.25*l2/weight_bytes)
    values = [10.+i for i in range(samples)]
    row = dict(status="PASS",arm=arm,variant=arm,shape=[1,n,k],mode=MODE,recipe=list(recipe(arm,n,k)),
        output_type="F32",error=1e-5,zero_code_negative="PASS",zero_a_check="PASS",
        inter_cta_split=1,launches_per_call=1,timing_scope="RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT",
        cache_scope="ACU_FORCED_COLD" if profile else "ROTATING_GT_2_25_L2",implementation=impl,
        weight_arithmetic="FP32_GROUP_AFFINE" if impl.startswith("affine") else "PER_WEIGHT_FP16",
        storage="CANONICAL_KPACK4" if arm.startswith("kpack-") else "RAW_GGUF" if arm=="raw-reference" else "XPLANE",
        device=dict(l2_bytes=l2),copies=copies,weight_bytes=weight_bytes,
        calls_per_graph=max(2,math.ceil(32/copies))*copies,samples_us=values,
        median_us=statistics.median(values) if values else None)
    if arm=="kpack-c8":
        row["geometry"]=geometry(n,k)
    return row


@pytest.mark.parametrize("arm",ARMS)
@pytest.mark.parametrize("n,k",SHAPES)
def test_all_shape_arm_receipts_and_profile_scope(arm,n,k):
    row = record(arm,n,k)
    assert parse_result(PREFIX+json.dumps(row),arm,n,k,3)==row
    row = record(arm,n,k,0,profile=True)
    assert parse_result(PREFIX+json.dumps(row),arm,n,k,0,profile=True)==row


@pytest.mark.parametrize("key,value",[("mode","warm"),("copies",1),("calls_per_graph",32),
    ("shape",[1,512,2048]),("error",float("nan")),("zero_code_negative","SKIP"),
    ("zero_a_check","SKIP"),("recipe",[4,8,1]),("geometry",{}),("output_type","F16"),
    ("weight_arithmetic","PER_WEIGHT_FP16"),("launches_per_call",2),
    ("cache_scope","ACU_FORCED_COLD"),("samples_us",[1.]),("median_us",float("nan"))])
def test_wrong_shape_or_dirty_result_is_rejected(key,value):
    with pytest.raises(ValueError):
        parse_result(PREFIX+json.dumps(record()|{key:value}),"kpack-c8",8192,5120,3)


def test_duplicates_and_cross_shape_resume_are_rejected(tmp_path):
    row = record(); text = PREFIX+json.dumps(row)
    with pytest.raises(ValueError):
        parse_result(text+"\n"+text,"kpack-c8",8192,5120,3)
    log = tmp_path/"run.log";log.write_text(text)
    entry = dict(row=row,log=log.name,log_sha256=digest(log))
    assert validate_entry(tmp_path,entry,"kpack-c8",8192,5120,3,row["device"])==row
    with pytest.raises(ValueError):
        validate_entry(tmp_path,entry,"kpack-c8",5120,8192,3,row["device"])
    with pytest.raises(ValueError):
        validate_entry(tmp_path,entry,"kpack-c8",8192,5120,3,{"pci":"other"})
    log.write_text("changed")
    with pytest.raises(ValueError):
        validate_entry(tmp_path,entry,"kpack-c8",8192,5120,3,row["device"])


def test_summary_retains_the_existing_winner_and_separates_incomplete_from_slow():
    rows = {arm:[record(arm) for _ in range(ROUNDS)] for arm in ARMS}
    for r in rows["kpack-c8"]:r["median_us"]=20.
    out = summarize(8192,5120,rows)
    assert out["status"]=="PASS" and out["selected_kpack"]=="kpack-current"
    assert out["parity_verdict"]=="WITHIN_5_PERCENT" and out["c8_parity_verdict"]=="PARITY_OPEN"
    for r in rows["kpack-current"]:r["median_us"]=21.
    assert summarize(8192,5120,rows)["parity_verdict"]=="PARITY_OPEN"
    rows["kpack-c8"].pop()
    assert summarize(8192,5120,rows)["parity_verdict"]=="INCOMPLETE"


def test_c8_launch_passes_real_shape_and_rotating_weight_pointers():
    bench = C8Bench.__new__(C8Bench)
    bench.n=512;bench.k=2048;bench.a=1;bench.output=2;bench.r=SimpleNamespace(stream=3)
    bench.copies=2;bench.weight_pointers=[(10,20),(30,40)]
    calls=[];bench.c8_launch=lambda *args:calls.append(args) or 0
    assert bench.invoke(RECIPE,3)==0
    assert calls==[(512,2048,1,30,40,2,3)]
    with pytest.raises(ValueError):bench.invoke((4,8,1))


@pytest.mark.parametrize("fault",[None,"failed_child","bad_numeric"])
def test_whole_six_shape_coordinator_and_retry_only_one_failed_cell(tmp_path,monkeypatch,fault):
    from dev.gemv_ppu import run_cold_shapes as runner
    for n,k in SHAPES:
        (tmp_path/f"q12-n{n}-k{k}-e1-c1.npz").write_bytes(b"host orchestration test only")
    args = SimpleNamespace(sdk=tmp_path,candidate=ROOT/"prebuilt/ppu0010/q4-cold-shapes-v1",
        controls=ROOT/"prebuilt/ppu0010/q4-h800-port-v1",bundle=ROOT/"prebuilt/ppu0010/q4-simt-ab-v1",
        output=tmp_path/"results",fixtures=tmp_path,samples=3,l2_bytes=67108864,skip_acu=True)
    monkeypatch.setattr(runner,"verify",lambda *a,**kw:dict(runtime={}))
    monkeypatch.setattr(runner,"probe_device",lambda args:dict(l2_bytes=67108864))
    jobs=[]
    def fake_child(command,stdout,stderr):
        arm=command[command.index("--variant")+1]
        fixture=Path(command[command.index("--fixture")+1])
        n,k=next((n,k) for n,k in SHAPES if fixture.name==f"q12-n{n}-k{k}-e1-c1.npz")
        key=(arm,n,k)
        bad=fault and arm=="kpack-c8" and (n,k)==SHAPES[0] and key not in jobs
        jobs.append(key)
        if bad and fault=="failed_child":return SimpleNamespace(returncode=1)
        row=record(arm,n,k)
        if bad:row["error"]=float("nan")
        stdout.write(PREFIX+json.dumps(row)+"\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,"run",fake_child)
    assert runner.run(args)==(0 if fault is None else 1)
    report=json.loads((args.output/"summary.json").read_text())
    assert len(jobs)==144 and len(report["cases"])==6
    assert sum(len(c["records"][a]) for c in report["cases"] for a in ARMS)==(144 if fault is None else 143)
    assert runner.run(args)==0
    assert len(jobs)==(144 if fault is None else 145)


def test_package_exports_and_no_parent_gpu_context_or_box_compilation():
    m=verify(ROOT/"prebuilt/ppu0010/q4-cold-shapes-v1",ROOT/"prebuilt/ppu0010/q4-h800-port-v1",
             ROOT/"prebuilt/ppu0010/q4-simt-ab-v1")
    assert not m["device_validated"] and not m["production_changed"]
    assert m["shapes"]==[list(s) for s in SHAPES]
    symbols=subprocess.check_output(["nm","-D","--defined-only",str(ROOT/"prebuilt/ppu0010/q4-cold-shapes-v1"/PAYLOAD)],text=True)
    assert " q4_cold_shapes_run" in symbols and " q4_ppu_probe" in symbols
    script=ROOT/"tools/run_q4_cold_shapes_ppu_box.sh"
    subprocess.run(["bash","-n",str(script)],check=True)
    body=script.read_text()
    assert "\n(\n" in body and body.rstrip().endswith(")")
    assert "timing_cells=144" in body and "cache=ROTATING_ONLY" in body
    assert "compile=NONE JIT=NONE" in body and "RESUME_RUN" in body
    parsed=ast.parse((ROOT/"dev/gemv_ppu/run_cold_shapes.py").read_text())
    run=next(x for x in parsed.body if isinstance(x,ast.FunctionDef) and x.name=="run")
    calls=[x.func.id for x in ast.walk(run) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)]
    assert "probe_device" in calls and "SDK" not in calls and "load_library" not in calls

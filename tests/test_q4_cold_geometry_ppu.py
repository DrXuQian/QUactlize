"""Host layout/orchestration/ABI checks; numeric and speed admission require PPU."""
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_cuda.build import digest
from dev.gemv_ppu.cold_geometry import ROOT, PAYLOAD, RECIPES, SHAPE, geometry, source, verify
from dev.gemv_ppu.h800_port import candidate_source
from dev.gemv_ppu.run_cold_geometry import ARMS, MODE, PREFIX, ROUNDS, GeometryBench, parse_result, summarize, validate_entry


def test_new_source_preserves_the_existing_entire_kernel_and_only_adds_c8_launch():
    old, new = candidate_source("large"), source()
    prefix = old[:old.index('extern "C" int qkg_pair_launch_12(')]
    assert new.startswith(prefix)
    entry = new[len(prefix):]
    assert entry.count(">>>(") == 1
    assert "q4_group_affine<8,8,4,8192,5120,true><<<n/32,256,0," in entry
    assert "n!=8192 || k!=5120" in entry and "(uintptr_t(out)&3)" in entry
    assert "q4_cold_c8_run" in entry and "hggcGetLastError" in entry
    assert "AIU" not in entry and "workspace" not in entry and "reduce" not in entry


@pytest.mark.parametrize("columns",[4,8])
def test_work_coverage_and_reduce_scatter_match_all_columns(columns):
    g = geometry(columns)
    groups, p, warps, tile_n = 160, 4, 8, g["tile_n"]
    workers = g["k_workers"]
    visited = np.zeros((groups,tile_n),dtype=np.int16)
    values = np.zeros((warps*32,p),dtype=np.int64)
    for tid in range(256):
        worker, col = tid//columns, (tid%columns)*p
        for group in range(worker,groups,workers):
            visited[group,col:col+p] += 1
            values[tid] += (group+1)*np.arange(col+1,col+p+1)
    assert np.all(visited==1)
    # Replay the generic P=4/Columns warp reduce-scatter with integer tags.
    lane = np.arange(32)
    values = values.reshape(warps,32,p)
    count,stride = p,columns
    while count>1:
        next_values=np.empty((warps,32,count//2),dtype=np.int64)
        odd=(lane&stride)!=0
        for i in range(count//2):
            keep=np.where(odd[None,:],values[:,:,2*i+1],values[:,:,2*i])
            send=np.where(odd[None,:],values[:,:,2*i],values[:,:,2*i+1])
            next_values[:,:,i]=keep+send[:,lane^stride]
        values=next_values;count//=2;stride*=2
    reduced=values[:,:,0]
    while stride<32:
        reduced=reduced+reduced[:,lane^stride];stride*=2
    partial=np.zeros((warps,tile_n),dtype=np.int64)
    for l in range(tile_n):
        partial[:,(l%columns)*p+l//columns]=reduced[:,l]
    warp_sums=np.array([partial[t//tile_n::32//tile_n,t%tile_n].sum() for t in range(32)])
    d=tile_n
    while d<32:
        warp_sums=warp_sums+warp_sums[lane^d];d*=2
    expected=np.arange(1,tile_n+1)*(groups*(groups+1)//2)
    assert np.array_equal(warp_sums[:tile_n],expected)
    assert g["grid"]*tile_n==SHAPE[1] and g["threads"]==256
    assert (g["k_passes"],g["last_pass_workers"])==((3,32) if columns==4 else (5,32))


def record(arm="kpack-c8", samples=3, *, profile=False):
    kpack=arm.startswith("kpack-")
    values=[10.+i for i in range(samples)]
    row=dict(status="PASS",arm=arm,variant=arm,shape=list(SHAPE),mode=MODE,recipe=list(RECIPES[arm]),
        output_type="F32",error=1e-5,zero_code_negative="PASS",zero_a_check="PASS",
        inter_cta_split=1,launches_per_call=1,timing_scope="RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT",
        cache_scope="ACU_FORCED_COLD" if profile else "ROTATING_GT_2_25_L2",
        implementation="affine4-early-fast-bare" if kpack else arm,
        weight_arithmetic="FP32_GROUP_AFFINE" if kpack else "PER_WEIGHT_FP16",
        storage="CANONICAL_KPACK4" if kpack else "RAW_GGUF" if arm=="raw-reference" else "XPLANE",
        device=dict(l2_bytes=67108864),copies=7,weight_bytes=23592960,calls_per_graph=35,
        samples_us=values,median_us=11. if samples else None)
    if kpack:row["geometry"]=geometry(RECIPES[arm][0])
    return row


@pytest.mark.parametrize("arm",ARMS)
def test_receipts_require_single_shape_rotating_and_actual_geometry(arm):
    row=record(arm)
    assert parse_result(PREFIX+json.dumps(row),arm,3)==row
    prof=record(arm,0,profile=True)
    assert parse_result(PREFIX+json.dumps(prof),arm,0,profile=True)==prof


@pytest.mark.parametrize("key,value",[("mode","warm"),("copies",1),("calls_per_graph",32),
    ("error",float("nan")),("zero_code_negative","SKIP"),("zero_a_check","SKIP"),
    ("recipe",[4,8,1]),("geometry",geometry(4)),("output_type","F16"),
    ("weight_arithmetic","PER_WEIGHT_FP16"),("launches_per_call",2),
    ("cache_scope","ACU_FORCED_COLD"),("samples_us",[1.])])
def test_wrong_or_dirty_cell_is_not_a_timing(key,value):
    with pytest.raises(ValueError):
        parse_result(PREFIX+json.dumps(record()|{key:value}),"kpack-c8",3)


def test_duplicate_row_and_profiler_timing_cannot_be_admitted():
    text=PREFIX+json.dumps(record())
    with pytest.raises(ValueError):parse_result(text+"\n"+text,"kpack-c8",3)
    with pytest.raises(ValueError):parse_result(PREFIX+json.dumps(record(samples=0,profile=True)),"kpack-c8",0)


def test_complete_but_slow_differs_from_missing_data():
    rows={arm:[record(arm) for _ in range(ROUNDS)] for arm in ARMS}
    assert summarize(rows)["parity_verdict"]=="WITHIN_5_PERCENT"
    for r in rows["kpack-c8"]:r["median_us"]=20.
    assert summarize(rows)["status"]=="PASS" and summarize(rows)["parity_verdict"]=="PARITY_OPEN"
    rows["kpack-c8"].pop()
    assert summarize(rows)["status"]=="INCOMPLETE" and summarize(rows)["parity_verdict"]=="INCOMPLETE"


def test_cached_receipt_is_bound_to_log_and_device(tmp_path):
    row=record();log=tmp_path/"run.log";log.write_text(PREFIX+json.dumps(row))
    entry=dict(row=row,log=log.name,log_sha256=digest(log))
    assert validate_entry(tmp_path,entry,"kpack-c8",3,row["device"])==row
    with pytest.raises(ValueError):validate_entry(tmp_path,entry,"kpack-c8",3,{"pci":"wrong"})
    log.write_text("changed")
    with pytest.raises(ValueError):validate_entry(tmp_path,entry,"kpack-c8",3,row["device"])


def test_c8_invokes_the_new_symbol_with_the_rotated_weight_pointers():
    bench=GeometryBench.__new__(GeometryBench)
    bench.is_c8=True;bench.n=8192;bench.k=5120;bench.a=1;bench.output=2
    bench.copies=2;bench.weight_pointers=[(10,20),(30,40)];bench.r=SimpleNamespace(stream=5)
    calls=[]
    bench.c8_launch=lambda *args:calls.append(args) or 0
    assert bench.invoke([8,8,1],3)==0
    assert calls==[(8192,5120,1,30,40,2,5)]


@pytest.mark.parametrize("fault",[None,"failed_child","bad_numeric"])
def test_full_coordinator_and_resume_only_rerun_failed_cell(tmp_path,monkeypatch,fault):
    from dev.gemv_ppu import run_cold_geometry as runner
    fixture=tmp_path/"fixture.npz";fixture.write_bytes(b"CPU orchestration check only")
    args=SimpleNamespace(sdk=tmp_path,candidate=ROOT/"prebuilt/ppu0010/q4-cold-geometry-v1",
        controls=ROOT/"prebuilt/ppu0010/q4-h800-port-v1",bundle=ROOT/"prebuilt/ppu0010/q4-simt-ab-v1",
        output=tmp_path/"results",fixture=fixture,samples=3,l2_bytes=67108864,skip_acu=True)
    monkeypatch.setattr(runner,"verify",lambda *a,**kw:dict(runtime={}))
    monkeypatch.setattr(runner,"probe_device",lambda args:dict(l2_bytes=67108864))
    def no_sdk(*a,**kw):raise AssertionError("parent acquired GPU context")
    monkeypatch.setattr(runner,"SDK",no_sdk)
    jobs=[]
    def fake_child(command,stdout,stderr):
        arm=command[command.index("--variant")+1]
        assert command[command.index("--mode")+1]==MODE
        bad=fault and arm=="kpack-c8" and arm not in jobs
        jobs.append(arm)
        if bad and fault=="failed_child":return SimpleNamespace(returncode=1)
        row=record(arm)
        if bad:row["error"]=float("nan")
        stdout.write(PREFIX+json.dumps(row)+"\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,"run",fake_child)
    assert runner.run(args)==(0 if fault is None else 1)
    report=json.loads((args.output/"summary.json").read_text())
    assert len(jobs)==24 and len(report["records"]["raw-reference"])==6
    assert len(report["records"]["kpack-c8"])==(6 if fault is None else 5)
    # Exact cached rounds survive; only the one missing/dirty round is rerun.
    assert runner.run(args)==0
    assert len(jobs)==(24 if fault is None else 25)


def test_native_payload_and_box_script():
    m=verify(ROOT/"prebuilt/ppu0010/q4-cold-geometry-v1",ROOT/"prebuilt/ppu0010/q4-h800-port-v1",
             ROOT/"prebuilt/ppu0010/q4-simt-ab-v1")
    assert not m["device_validated"] and not m["production_changed"]
    assert m["geometry"]==geometry(8)
    symbols=subprocess.check_output(["nm","-D","--defined-only",str(ROOT/"prebuilt/ppu0010/q4-cold-geometry-v1"/PAYLOAD)],text=True)
    assert " q4_cold_c8_run" in symbols and " q4_ppu_probe" in symbols
    script=ROOT/"tools/run_q4_cold_geometry_ppu_box.sh"
    subprocess.run(["bash","-n",str(script)],check=True)
    body=script.read_text()
    assert "\n(\n" in body and body.rstrip().endswith(")")
    assert "cache=ROTATING_ONLY" in body and "timing_cells=24" in body
    assert "compile=NONE JIT=NONE" in body and "--q4-dense-wide" not in body
    assert "export(Path(sys.argv[1]), 12, 8192, 5120, 1, [0], 1)" in body
    assert "RESUME_RUN" in body and '"$RUN.results.tgz"' in body
    src=(ROOT/"dev/gemv_ppu/run_cold_geometry.py").read_text()
    run=next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=="run")
    calls=[x.func.id for x in ast.walk(run) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)]
    assert "probe_device" in calls and "SDK" not in calls and "load_library" not in calls

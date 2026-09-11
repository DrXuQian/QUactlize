"""Local contract checks, not a PPU correctness/performance admission."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_ppu.bload_source import REFERENCE_SHA256, RECIPES, reference_fp32, reference_wrapper
from dev.gemv_ppu.run_bload import (ROOT, ROUNDS, VARIANTS, CONTROL_RECEIPT, exact_bits,
    parse_result, read_control_recipes, recipes, summarize, transport_expected, verify_packages)
from dev.gemv_cuda.build_fixed_reference import runner_source


def test_reference_is_the_exact_uploaded_source():
    path=ROOT/"dev/gemv_ppu/reference/gemv_ref.cuh"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==REFERENCE_SHA256


def test_reference_changes_dot_not_only_output_and_preserves_raw_reader():
    original=(ROOT/"dev/gemv_ppu/reference/gemv_ref.cuh").read_text()
    for ppu in (True,False):
        src=reference_fp32(original,ppu=ppu)
        assert "half2& dq" not in src and "half2 dq" not in src
        assert "__hfma2(__hfma2" not in src and "__hadd2(__hadd2(dq" not in src
        assert src.count("fp32_dot(__hfma2(")==4
        assert src.count("float2 dq")==4 and src.count("float2& dq")==4
        assert "fmaf(wf.x, af.x, sum.x)" in src
        assert "float*             out" in src and "block_q4_K* w, float* out" in src
        assert "= __float2half(acc[col])" not in src
        for token in ("m4[ii]  = *reinterpret_cast<uint4 const*>(bq)",
                      "qs4[ii] = *reinterpret_cast<uint4 const*>(bq->qs + t_in_sb * 16)",
                      "half2 const zero2  = __hfma2(__half2half2(d8), sc2,",
                      "asm volatile(\"prmt.b32", "for (unsigned i = tid; i < n_vec; i += THREADS)"):
            assert token in original and token in src
        assert "// 0xea = (A & B) | C" in src
        assert ("hggc_runtime.h" in src)==ppu
        for constant in ("FP16_TOP_MAGIC", "ONE_SIXTEENTH", "NEG_72"):
            assert (f'"n"({constant})' in src)==ppu
            assert (f'"r"({constant})' in src)!=ppu
    with pytest.raises(ValueError):reference_fp32(original.replace("half2& dq0","float& dq0"))
    with pytest.raises(ValueError):reference_fp32(original+"\nhalf2& dq0\n")


@pytest.mark.parametrize("bk,wk,_",RECIPES)
def test_transport_stage_ownership_and_k_nibble_bijection(bk,wk,_):
    hits=np.zeros((512,64),dtype=np.int32)
    for tile in range(4):
        for kb in range(0,2048,bk):
            for warp in range(wk):
                for micro in range(warp,bk//64,wk):
                    for lane in range(32):
                        for reg in range(4):
                            nn=tile*16+lane//4+(reg//2)*8
                            kg=kb//4+micro*16+2*(lane%4)+(reg%2)*8
                            hits[kg:kg+2,nn]+=1
    assert np.all(hits==1)
    logical=[(kg//8)*32+kg%8+s*8 for kg in range(16) for s in range(4)]
    assert sorted(logical)==list(range(64))
    tags,expected=transport_expected()
    delivered=expected.reshape(-1).view("<u2")
    assert np.array_equal(np.sort(delivered),np.sort(tags.reshape(-1)))
    for wrong in (np.roll(expected,1,axis=1),expected[..., [2,3,0,1]]):
        with pytest.raises(ValueError):exact_bits(expected.tobytes(),wrong.tobytes(),"planted")


def test_source_has_real_aiu_transpose_and_explicit_lifetime_edges():
    s=(ROOT/"dev/gemv_ppu/bload.cu").read_text()
    assert "PPU0010_AIU_LOAD<cute::C<H * 16 * 16>, cutlass::half_t, true, true>" in s
    assert "PPU0010_TSM_LD_SWZL<cutlass::half_t, H, 16, true, true, 1>" in s
    assert "if (threadIdx.x == 0) Write::copy(shared, low, desc, kg, col);" in s
    assert "cute::cp_async_wait<0>();\n        __syncthreads();" in s
    assert "if constexpr (Aiu) __syncthreads(); // all reads finish before overwrite" in s
    assert "x = fmaf(wf.x, af.x, x)" in s and "y = fmaf(wf.y, af.y, y)" in s
    assert "mma.sync" not in s and "half2&" not in s


def record(variant="fragment-aiu"):
    return dict(status="PASS",arm=variant,variant=variant,recipe=[256,4,1],shape=[1,5120,8192],mode="warm",
        error=.00003,zero_code_negative="PASS",zero_a_check="PASS",output_type="F32",storage="CANONICAL_KPACK4",
        weight_arithmetic="PER_WEIGHT_FP16",launches_per_call=1,inter_cta_split=1,
        samples_us=[10.,11.,12.],median_us=11.,copies=1,weight_bytes=5120*8192*9//16,calls_per_graph=32,
        device=dict(l2_bytes=64*1024**2),transport_b16_sha256="a"*64,matched_fp32_sha256="b"*64)


def parse(row):return parse_result("Q4_PPU_CELL "+json.dumps(row),"fragment-aiu",[256,4,1],[1,5120,8192],"warm",3)


def test_receipt_requires_transport_and_matched_fp32():assert parse(record())["median_us"]==11.


@pytest.mark.parametrize("key,value",[("matched_fp32_sha256",None),("transport_b16_sha256","z"*64),
    ("zero_code_negative","SKIP"),("zero_a_check","SKIP"),("output_type","F16"),("storage","RAW_GGUF"),
    ("error",float("nan")),("samples_us",[10.,11.]),("recipe",[256,4,2]),
    ("weight_arithmetic","FP32_GROUP_AFFINE"),("launches_per_call",2),("inter_cta_split",2)])
def test_bad_receipt_cannot_be_a_timing(key,value):
    with pytest.raises(ValueError):parse(record()|{key:value})


def test_summary_keeps_same_recipe_comparison_and_missing_cells():
    records={v:{} for v in VARIANTS}
    for v in VARIANTS:
        for i,r in enumerate(recipes(v,5120,8192,"warm")):
            records[v][json.dumps(r)]=[dict(median_us=10.+i+(0 if v=="fragment-aiu" else 1))]*ROUNDS
    s=summarize(5120,8192,"warm",records)
    assert s["status"]=="PASS" and len(s["matched_transport"])==3
    assert all(r["delta_pct"]<0 for r in s["matched_transport"])
    assert s["candidate_deltas"]["fragment-aiu"]["parity_verdict"]=="WITHIN_5_PERCENT"
    records["fragment-aiu"][json.dumps(RECIPES[1])].pop()
    incomplete=summarize(5120,8192,"warm",records)
    assert incomplete["status"]=="INCOMPLETE"
    assert incomplete["candidate_deltas"]["fragment-aiu"]["parity_verdict"]=="INCOMPLETE"


def test_reference_replays_ppu_winner_including_k_warps_not_external_split():
    r=recipes("raw-reference",5120,8192,"warm")[0]
    assert r==(4,4,2)
    row=record("raw-reference")|dict(recipe=list(r),storage="RAW_GGUF")
    assert parse_result("Q4_PPU_CELL "+json.dumps(row),"raw-reference",r,[1,5120,8192],"warm",3)["inter_cta_split"]==1
    row["inter_cta_split"]=2
    with pytest.raises(ValueError):
        parse_result("Q4_PPU_CELL "+json.dumps(row),"raw-reference",r,[1,5120,8192],"warm",3)


def test_current_control_is_not_the_old_n4_kernel_or_false_fp16_affine_label():
    r=recipes("kpack-current",5120,8192,"warm")[0]
    row=record("kpack-current")|dict(recipe=list(r))
    with pytest.raises(ValueError):
        parse_result("Q4_PPU_CELL "+json.dumps(row),"kpack-current",r,[1,5120,8192],"warm",3)
    row["weight_arithmetic"]="FP32_GROUP_AFFINE"
    assert parse_result("Q4_PPU_CELL "+json.dumps(row),"kpack-current",r,[1,5120,8192],"warm",3)
    assert recipes("kpack-current",512,2048,"warm")==[(1,16,1)]


@pytest.mark.parametrize("fault",["duplicate","missing","invalid_recipe","incomplete"])
def test_control_selection_receipt_rejects_bad_denominator_and_recipe(tmp_path,fault):
    report=json.loads(CONTROL_RECEIPT.read_text())
    if fault=="duplicate":report["cases"].append(report["cases"][0])
    elif fault=="missing":report["cases"].pop()
    elif fault=="invalid_recipe":report["cases"][0]["winners"]["raw-reference"]["recipe"]=[1,8,99]
    else:report["status"]="INCOMPLETE"
    path=tmp_path/"receipt.json";path.write_text(json.dumps(report))
    with pytest.raises(ValueError):read_control_recipes(path)


def test_parent_device_probe_finishes_before_children_and_does_not_keep_context():
    import ast
    source=(ROOT/"dev/gemv_ppu/run_bload.py").read_text()
    tree=ast.parse(source)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run")
    calls=[n.func.id for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert "SDK" not in calls and "load_library" not in calls and "probe_device" in calls


@pytest.mark.parametrize("fault",[None,"partial_child","invalid_cell","duplicate_cell"])
def test_coordinator_runs_bounded_cohort_without_a_parent_gpu_context(tmp_path,monkeypatch,fault):
    from dev.gemv_ppu import run_bload as runner
    fixture=tmp_path/"q12-n5120-k8192-e1-c1.npz"
    fixture.write_bytes(b"orchestration-only fixture identity")
    args=SimpleNamespace(sdk=tmp_path,bundle=ROOT/"prebuilt/ppu0010/q4-simt-ab-v1",
        candidate=ROOT/"prebuilt/ppu0010/q4-bload-v1",controls=ROOT/"prebuilt/ppu0010/q4-h800-port-v1",
        fixtures=tmp_path,output=tmp_path/"results",all_shapes=False,samples=3,l2_bytes=67108864,skip_acu=True)
    device=dict(l2_bytes=67108864)
    monkeypatch.setattr(runner,"verify_packages",lambda *a,**kw:dict(runtime={}))
    monkeypatch.setattr(runner,"probe_device",lambda args:device)
    def no_sdk(*a,**kw):raise AssertionError("parent acquired a GPU context")
    monkeypatch.setattr(runner,"SDK",no_sdk)
    jobs=[]
    def fake_child(command,stdout,stderr):
        def arg(name):return command[command.index(name)+1]
        arm,mode=arg("--variant"),arg("--mode")
        wanted=json.loads(arg("--recipes"))
        assert arg("--controls")==str(args.controls)
        planted=fault and not any(j[0]=="fragment-global" for j in jobs) and arm=="fragment-global"
        jobs.append((arm,mode,wanted))
        for i,recipe in enumerate(wanted):
            if planted and fault=="partial_child" and i==len(wanted)-1:break
            copies=1 if mode=="warm" else 7
            row=record(arm)|dict(recipe=recipe,mode=mode,device=device,copies=copies,
                calls_per_graph=max(2,(32+copies-1)//copies)*copies,
                storage="RAW_GGUF" if arm=="raw-reference" else "CANONICAL_KPACK4",
                weight_arithmetic=runner.arithmetic(arm,5120,8192))
            if planted and fault=="invalid_cell" and i==len(wanted)-1:row["error"]=float("nan")
            stdout.write("Q4_PPU_CELL "+json.dumps(row)+"\n")
            if planted and fault=="duplicate_cell" and i==len(wanted)-1:
                stdout.write("Q4_PPU_CELL "+json.dumps(row)+"\n")
        return SimpleNamespace(returncode=int(bool(planted and fault=="partial_child")))
    monkeypatch.setattr(runner.subprocess,"run",fake_child)
    rc=runner.run(args)
    report=json.loads((args.output/"summary.json").read_text())
    assert len(jobs)==40 and sum(len(j[2]) for j in jobs)==72
    assert len(report["cases"])==2 and all(set(c["winners"])==set(VARIANTS) for c in report["cases"])
    assert (rc,report["status"])==((0,"PASS") if fault is None else (1,"INCOMPLETE"))
    if fault:
        # Two good recipes in the same child survive the third recipe's failure.
        receipts=list(args.output.glob("*warm-fragment-global-r0.json"))
        assert len(receipts)==1 and len(json.loads(receipts[0].read_text()))==2
        assert len(report["failures"])==1
        assert report["cases"][0]["candidate_deltas"]["fragment-aiu"]["parity_verdict"]=="INCOMPLETE"
        assert report["cases"][1]["status"]=="PASS"


@pytest.mark.parametrize("variant",["raw-reference","kpack-current","fragment-global","fragment-aiu"])
def test_prebuilt_call_signature_and_k_warp_are_preserved(variant):
    from dev.gemv_ppu.run_bload import ExperimentBench
    bench=ExperimentBench.__new__(ExperimentBench)
    bench.variant=variant;bench.n=5120;bench.k=8192;bench.copies=1
    bench.weight_pointers=[(101,102)];bench.a=103;bench.output=104;bench.r=SimpleNamespace(stream=105)
    calls=[]
    def launch(*args):calls.append(args);return 0
    bench.raw_launch=bench.current_launch=bench.blaunch=launch
    recipe=recipes(variant,5120,8192,"warm")[0]
    assert bench.invoke(recipe)==0
    if variant=="raw-reference":expected=(4,4,2,5120,8192,103,101,104,105)
    elif variant=="kpack-current":expected=(5120,8192,103,101,102,104,105)
    else:expected=(int(variant=="fragment-aiu"),256,4,5120,8192,103,101,102,104,105)
    assert calls==[expected]


def test_cuda_profile_adapts_raw_storage_without_weakening_oracle():
    old=(ROOT/"dev/gemv_cuda/profile_xplane.cu").read_text()
    src=runner_source(old)
    assert "rawrun(cfg.columns,cfg.warps,h.n,h.k,a.ptr,call.low,call.output,stream)" in src
    assert 'low_stride=arm=="raw-reference" ? h.lengths[0] : h.lengths[1]' in src
    assert 'block*144+16' in src and 'zero-code negative escaped' in src
    assert "check(cudaDeviceSynchronize());" in src
    assert "max(2,(32+copies-1)/copies)*copies" in src
    assert "for(int s=-5;s<15;++s)" in src
    with pytest.raises(ValueError):runner_source(old.replace("copy*h.lengths[1];","copy*777;"))


def test_script_does_not_compile_or_modify_existing_astage_run():
    path=ROOT/"tools/run_q4_bload_ppu_box.sh"
    subprocess.run(["bash","-n",str(path)],check=True)
    src=path.read_text()
    assert "\n(\n" in src and src.rstrip().endswith(")")
    assert "JIT=NONE" in src and "rounds=4" in src
    assert '--all-shapes' in src and '--anchor-only' not in src
    assert "build_bload.py --" not in src
    assert '"$RUN.results.tgz"' in src


def test_published_native_addon_is_bound_and_not_device_admitted():
    m=verify_packages(ROOT/"prebuilt/ppu0010/q4-bload-v1",ROOT/"prebuilt/ppu0010/q4-simt-ab-v1",
                      ROOT/"prebuilt/ppu0010/q4-h800-port-v1")
    assert m["device_validated"] is False and m["production_changed"] is False
    assert m["reference_original_sha256"]==REFERENCE_SHA256
    assert m["recipes"]==[list(r) for r in RECIPES]


def test_h800_receipt_has_all_fixed_rounds_and_does_not_claim_counters():
    from dev.gemv_cuda.compare_q4_native import parse_timing
    p=ROOT/"docs/measurements/q4_h800_fixed_20260911.json"
    m=json.loads(p.read_text())
    assert m["status"]=="PASS" and m["shape"]==[1,5120,8192]
    assert m["rounds"]==6 and m["samples"]==15
    assert {r["mode"] for r in m["cases"]}=={"warm","rotating"}
    for case in m["cases"]:
        assert set(case["records"])=={"xplane","kpack","raw-reference"}
        for arm,rows in case["records"].items():
            assert len(rows)==6
            for row in rows:
                assert row["status"]=="PASS" and float(row["error"])<.005
                assert len(row["samples_us"])==15
                assert row["config"]=="-".join(map(str,m["recipes"][arm]))
    assert m["profiles"]==[dict(arm="xplane",rc=1,log="xplane.ncu.log",files=[],
        cache="FORCED_COLD",timing_authority=False,status="COUNTERS_PERMISSION_DENIED")]

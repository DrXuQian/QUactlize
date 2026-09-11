"""A fixed-choice confirmation must retain every admitted shape and regime."""
import copy
import json
import math
from pathlib import Path
import statistics

import pytest

from dev.gemv_cuda.confirm_q4_retuned import SHAPES, fixed_plan
from dev.gemv_cuda.check_q4_static import cases


@pytest.mark.parametrize("plant",(None,"missing","duplicate","half","running","shape","arm","xplane-split"))
def test_confirmation_never_silently_drops_an_unfavourable_point(plant):
    receipt=dict(status="PASS",arithmetic="BOTH_FP32_DOT_AND_REDUCTION_FP16_WEIGHT_AND_A_BOUNDARY",cases=[])
    for n,k in SHAPES:
        for mode in ("warm","rotating"):
            receipt["cases"].append(dict(shape=[1,n,k],mode=mode,fixture_sha256="fixture",winners={
                "xplane":dict(recipe=["xplane",2,4,1]),"kpack":dict(recipe=["kpack",4,8,1])}))
    if plant=="missing":receipt["cases"].pop()
    if plant=="duplicate":receipt["cases"].append(copy.deepcopy(receipt["cases"][0]))
    if plant=="half":receipt["arithmetic"]="FP16"
    if plant=="running":receipt["status"]="RUNNING"
    if plant=="shape":receipt["cases"][0]["shape"][0]=2
    if plant=="arm":receipt["cases"][0]["winners"]["kpack"]["recipe"][0]="xplane"
    if plant=="xplane-split":receipt["cases"][0]["winners"]["xplane"]["recipe"][-1]=2
    if plant:
        with pytest.raises(ValueError):fixed_plan(receipt)
    else:
        plan=fixed_plan(receipt)
        assert len(plan)==12 and all(p["recipes"]["kpack"]==[4,8,1] for p in plan)


def test_static_numeric_domain_covers_all_new_dispatch_instantiations():
    plan=list(cases())
    assert len(plan)==len(set(plan))==44
    large={(n,k,c,w) for n,k in SHAPES if n>=4096 for c in (4,8) for w in (4,5,8,10,16)}
    assert {r for r in plan if r[0]>=4096}==large


def test_six_family_receipt_recomputes_all_twelve_fixed_comparisons():
    path=Path(__file__).resolve().parents[1]/"docs/measurements/q4_fp32_large_s1_5070_20260911.json"
    report=json.loads(path.read_text())
    assert report["admitted_shape_scope"]=="SIX_DENSE_FAMILIES_BOTH_REGIMES"
    assert not report["production_changed"] and not report["offline_arrangement_changed"]
    confirmation=report["confirmation"]
    assert confirmation["status"]=="PASS" and confirmation["performance_verdict"]=="WITHIN_5_PERCENT"
    rows=confirmation["cases"]
    assert len(rows)==12
    assert {(tuple(c["shape"]),c["mode"]) for c in rows}=={
        ((1,n,k),mode) for n,k in SHAPES for mode in ("warm","rotating")}
    for c in rows:
        assert c["recipes"]["kpack"][-1]==1
        med={}
        for arm,rr in c["records"].items():
            assert len(rr)==6
            for r in rr:
                assert len(r["samples_us"])==15
                assert all(math.isfinite(x) and x>0 for x in r["samples_us"])
                assert abs(statistics.median(r["samples_us"])-r["median_us"])<2e-5
                assert math.isfinite(r["error"]) and 0<=r["error"]<.005
            med[arm]=statistics.median(r["median_us"] for r in rr)
        assert med==c["median_us"]
        assert abs(100*(med["kpack"]/med["xplane"]-1)-c["delta_pct"])<1e-9
        assert c["delta_pct"]<=5
    assert sum(r["cases"] for r in report["numeric"])==276
    assert len(report["static_numeric"]["records"])==44
    profiles=report["ncu"]["profiles"]
    assert len(profiles)==8 and all(len(p["kernels"])==1 for p in profiles)
    kp=[p for p in profiles if p["key"].endswith("kpack")]
    assert len(kp)==4 and all("kpack_q4_large_static" in p["kernels"][0]["kernel"] for p in kp)
    assert report["remaining"]

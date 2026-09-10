import copy
import math
import pytest
from tools.export_kpack_gemv_policy import CONFIGS, export


def fixture():
    c = dict(mode=2, rows=8, channels=8, topk=8)
    samples = {k: [[10.0 + i] * 11] * 3 for i, k in enumerate(sorted(CONFIGS))}
    row = dict(
        q=13,
        n=2048,
        k=512,
        experts=256,
        case=c,
        input_type=1,
        correctness="PASS",
        zero_low_negative="DETECTED_FINITE",
        errors=[0.0001] * 8,
        gemv_samples_us=samples,
        winner=sorted(CONFIGS)[0],
        gemv_median_us=10.0,
        gemm_samples_us=[[20.0] * 11] * 3,
        gemm_median_us=20.0,
    )
    return dict(
        status="PASS",
        failures=[],
        plan=[dict(q=13, n=2048, k=512, e=256, cases=[c])],
        results=[dict(records=[row])],
    )


def test_recipe_from_samples():
    text, report = export(fixture())
    assert text.startswith("KPACK_GEMV_POLICY_V1\n13\t2048\t512\t256\t2\t8\t8\t8\t1\t")
    assert report[0]["median_us"] == 10 and "NOT_GLOBAL_OPTIMUM" in report[0]["scope"]


def test_slower_gemv_does_not_replace_fq():
    s = fixture()
    r = s["results"][0]["records"][0]
    r["gemm_samples_us"] = [[5.0] * 11] * 3
    r["gemm_median_us"] = 5.0
    text, report = export(s)
    assert text == "KPACK_GEMV_POLICY_V1\n" and report[0]["selected"] == "RETAIN_FQ"


@pytest.mark.parametrize('winner',[True,False])
def test_q8_conservative_admission_and_incumbent_label(winner):
    from tools.run_q8_simt_gate import SCOPE, SHAPES
    s=fixture(); row=s['results'][0]['records'][0]
    s['plan'][0]['q']=row['q']=8
    row['scope']=SCOPE
    row['gemm_median_us']=20.0 if winner else 5.0
    row['gemm_samples_us']=[[row['gemm_median_us']]*11 for _ in range(3)]
    text,report=export(s)
    assert ('\n8\t' in text)==winner
    assert report[0]['selected']==('GEMV' if winner else 'RETAIN_TC')
    assert report[0]['scope']==SCOPE
    assert len(SHAPES)==6 and len(set(SHAPES))==6


@pytest.mark.parametrize("plant", range(8))
def test_invalid_evidence_rejected(plant):
    s = fixture()
    r = s["results"][0]["records"][0]
    if plant == 0:
        s["status"] = "INCOMPLETE"
    elif plant == 1:
        s["failures"] = ["runtime"]
    elif plant == 2:
        r["errors"][0] = float("nan")
    elif plant == 3:
        r["winner"] = "32-8-4"
    elif plant == 4:
        r["gemv_samples_us"][r["winner"]][0][0] = float("nan")
    elif plant == 5:
        r["zero_low_negative"] = "NOT_COLLECTED"
    elif plant == 6:
        s["plan"][0]["cases"].append(dict(mode=0, rows=1, channels=1, topk=1))
    else:
        r["input_type"] = 0
    with pytest.raises(ValueError):
        export(s)

"""The small-shape result cannot stand in for all-shape or PPU admission."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def test_small_receipt_retains_samples_and_only_admits_its_four_points():
    report=json.loads((ROOT/"docs/measurements/q4_fp32_small_s1_5070_20260911.json").read_text())
    confirmation=report["confirmation"]
    assert not report["production_changed"] and not report["offline_arrangement_changed"]
    assert "NOT_GROUPED_PERFORMANCE" in report["scope"]
    assert confirmation["status"]=="PASS" and confirmation["performance_verdict"]=="WITHIN_5_PERCENT"
    assert len(confirmation["cases"])==4
    for case in confirmation["cases"]:
        assert case["shape"] in ([1,512,2048],[1,1024,5120])
        assert case["recipes"]["kpack"][-1]==1
        med={}
        for arm,rounds in case["records"].items():
            assert len(rounds)==6
            for r in rounds:
                assert len(r["samples_us"])==15 and all(math.isfinite(x) and x>0 for x in r["samples_us"])
                assert abs(statistics.median(r["samples_us"])-r["median_us"])<2e-5
                assert math.isfinite(r["error"]) and 0<=r["error"]<.005
            med[arm]=statistics.median(r["median_us"] for r in rounds)
        assert med==case["median_us"]
        assert abs(100*(med["kpack"]/med["xplane"]-1)-case["delta_pct"])<1e-9
        assert case["delta_pct"]<=5
    assert len(report["campaign"])==12
    assert any(c["delta_pct"]>5 for c in report["campaign"])
    assert len(report["ncu"]["profiles"])==8
    assert all(len(p["kernels"])==1 for p in report["ncu"]["profiles"])
    assert sum(r["cases"] for r in report["numeric"])==232
    assert report["remaining"]

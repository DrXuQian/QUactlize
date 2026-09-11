"""Host-side falsification of profile scope and metric interpretation."""

import copy
import csv
import io
import json
from pathlib import Path

import pytest

from dev.gemv_cuda.read_xplane_ncu import CORE, parse
from dev.gemv_cuda.run_xplane_ncu import jobs

ROOT = Path(__file__).resolve().parents[1]


def sample():
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Kernel Name", *CORE])
    writer.writerow(["", *["ns" if x == "gpu__time_duration.sum" else "%" for x in CORE]])
    writer.writerow(["void kernel<2,4,1>()", *[1000 if x == "gpu__time_duration.sum" else 0 for x in CORE]])
    return stream.getvalue()


def test_zero_dram_is_a_measurement_not_missing_data():
    result = parse(sample(), 1)
    assert result[0]["metrics"]["dram__bytes.sum"]["value"] == 0
    assert result[0]["metrics"]["gpu__time_duration.sum"] == {"value": 1000, "unit": "ns"}


def test_replay_hit_ratio_anomaly_is_preserved_but_not_admitted():
    rows = list(csv.reader(io.StringIO(sample())))
    i = rows[0].index("lts__t_sector_hit_rate.pct")
    rows[2][i] = "101.68"
    stream = io.StringIO()
    csv.writer(stream).writerows(rows)
    result = parse(stream.getvalue(), 1)[0]
    hit = result["metrics"]["lts__t_sector_hit_rate.pct"]
    assert hit["value"] == 101.68 and hit["usable_for_conclusion"] is False
    assert result["warnings"]


@pytest.mark.parametrize("plant", ("nan", "absent", "extra", "reducer-first", "zero-duration"))
def test_invalid_profile_does_not_become_a_utilization_number(plant):
    text = sample()
    if plant == "nan":
        text = text.replace(",1000,", ",nan,", 1)
    elif plant == "absent":
        text = text.replace("dram__bytes.sum,", "wrong_metric,", 1)
    elif plant == "extra":
        text += text.splitlines()[-1] + "\n"
    elif plant == "reducer-first":
        text = text.replace("void kernel<2,4,1>()", "kpack_gemv_reduce()")
    else:
        text = text.replace(",1000,", ",0,", 1)
    with pytest.raises(ValueError):
        parse(text, 1)


def test_profiler_selects_the_measured_recipes_without_cross_device_retuning_claim():
    receipt = json.loads((ROOT/"docs/measurements/q4_xplane_kpack_5090_20260911.json").read_text())
    plan = list(jobs(receipt))
    assert len(plan) == len({x["key"] for x in plan}) == 8
    assert next(x["recipe"] for x in plan if x["key"] == "k4096-warm-kpack") == [16,8,8]
    bad = copy.deepcopy(receipt)
    bad["cases"][0]["shape"][0] = 2
    with pytest.raises(ValueError, match="scope"):
        list(jobs(bad))


def test_profiling_boundary_is_after_the_cache_setup():
    source = (ROOT/"dev/gemv_cuda/profile_xplane.cu").read_text()
    assert source.index("for(int pass=0;pass<5;") < source.index("cudaProfilerStart()")
    assert source.index("cudaProfilerStart()") < source.index("cudaProfilerStop()")
    runner = (ROOT/"dev/gemv_cuda/run_xplane_ncu.py").read_text()
    for fragment in ('"--replay-mode", "application"', '"--cache-control", "none"',
                     '"--profile-from-start", "off"'):
        assert fragment in runner

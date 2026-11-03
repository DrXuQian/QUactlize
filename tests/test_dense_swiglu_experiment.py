import argparse
from pathlib import Path
from dev.dense_swiglu.bench import workload_plan
from tools.run_moe_scaffold import workload_plan as runner_plan

ROOT=Path(__file__).resolve().parents[1]


def test_bounded_one_kernel_inventory():
    cases,info=workload_plan()
    assert len(cases)==len({c.key for c in cases})==14
    assert {c.hidden for c in cases}=={512,1024,17408,25600}
    assert {c.tokens for c in cases}=={1,8,512,2048,5120}
    assert all(c.topk==c.experts==1 and c.dtype==0 and c.phase=='transition' for c in cases)
    assert not info['production_changed']
    args=argparse.Namespace(cohort='dense-swiglu',smoke=False,phase='all')
    assert runner_plan(args)==(cases,info)
    assert len(workload_plan(smoke=True)[0])==2


def test_no_extra_helpers_or_native_dependency():
    native=(ROOT/'dev/dense_swiglu/native.cu').read_text()
    assert '#include "../../quactlize/ffn/dense_swiglu.cu"' in native
    assert 'ffn::dense_swiglu(gate,up,' in native
    assert 'quactlize_ppu_moe_transition_run_v1(c)' in native
    bench=(ROOT/'dev/dense_swiglu/bench.py').read_text()
    assert "('control','dense-vec4')" in bench
    assert "b.geometry(arm)" in bench
    assert 'snapshot' not in bench

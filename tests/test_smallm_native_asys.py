"""Native baseline protocol tests; synthetic intervals are not GPU evidence."""
from copy import deepcopy
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dev.native_components.asys import plan,parse_rows,PROTOCOL,METHOD

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    point=plan()['points'][0];auth=dict(test='host-only')
    components=[];schedule=[];rows=[];clock=0
    for i,(role,symbol) in enumerate((('adapter','quantize_q8_1'),('producer','mul_mat_vec_q'))):
        components.append(dict(stage=i,role=role,entry='KERNEL',name=symbol,symbol=symbol,grid=[32,1,1],
            block=[256,1,1],shared_bytes=0,correctness='PASS',first_use_excluded=True,host_launch_cost_added=False,
            timed_kernel_types=1,timing=METHOD,identity=auth,calls_per_graph=16))
    for r in range(6):
        for i in ([0,1] if r%2==0 else [1,0]):
            for sample in range(5):
                schedule.append(dict(stage=i,round=r,sample=sample,kernels=16))
                for _ in range(16):
                    duration=2000 if i==0 else 8000
                    rows.append(dict(components[i],kind='kernel',start=clock,end=clock+duration))
                    clock+=duration+100
    proofs=[dict(status='PASS',copy=i,spot_cells=64,finite_cells_per_arm=1024,
        oracle_activation='NATIVE_Q8_1',arithmetic_spot_normalized_error=.000001,
        raw_spot_normalized_error=[.006],raw_f32_status='FAIL') for i in (0,0,1,0,1,0,0,1,0,1)]
    data=dict(status='CAPTURED_NOT_TIMED',timing_method=METHOD,protocol=PROTOCOL,
        id=point['id'],workload=point['workload'],authority=auth,correctness='PASS',
        linked_backend_comparison='PASS_TWO_INPUTS',omitted_producer='DETECTED',
        last_weight_copy_negative='DETECTED_COPY0_UNCHANGED',proof=proofs,
        native_activation_contract='Q8_1_QUANTIZED_FROM_F32',components=components,trace_schedule=schedule,
        ring=dict(copies=2,l2_bytes=64,active_read_bytes=256,minimum_read_bytes=128))
    return data,point,auth,rows


def test_native_cohort_is_m1_to_8_for_both_types_and_same_workload_keys():
    value=plan()
    assert len(value['points'])==16
    assert [(p['workload']['qtype'],p['workload']['m']) for p in value['points']]==[(q,m) for q in (13,14) for m in range(1,9)]
    from dev.strategy.smallm_asys import workloads
    assert [p['id'] for p in value['points'][8:]]==[w.key for w in workloads('q6-token-cta')]
    for qs,n,k in (([9],1024,5120),([14,14],1024,5120),([14],127,5120),([14],1024,15)):
        with pytest.raises(ValueError):plan(qs,n,k)


def test_native_asys_helper_and_producer_are_separate_means():
    data,point,auth,rows=fixture()
    result=parse_rows(data,point,auth,rows,dict(source='synthetic'))
    assert [c['mean_us'] for c in result['components']]==[2,8]
    assert result['sum_mean_us']==10
    assert result['status']=='PARSED_NOT_ADMITTED' and not result['no_regression_admitted']
    assert data['status']=='CAPTURED_NOT_TIMED'
    # Don't silently reinterpret native Q8_1 as identical F16 arithmetic.
    assert result['proof'][0]['raw_f32_status']=='FAIL'
    rows[1]['start']=rows[0]['start']
    assert parse_rows(data,point,auth,rows,{})['sum_mean_us']>10


@pytest.mark.parametrize('change',['count','symbol','grid','negative','proof','cold','event','provider','repeat','oracle'])
def test_native_asys_does_not_hide_bad_coverage_or_numerics(change):
    data,point,auth,rows=fixture()
    if change=='count':rows.pop()
    elif change=='symbol':rows[0]['symbol']='quactlize::wrong'
    elif change=='grid':rows[0]['grid']=[1,1,1]
    elif change=='negative':data['omitted_producer']='MISSING'
    elif change=='proof':data['proof'][0]['arithmetic_spot_normalized_error']=.01
    elif change=='cold':data['ring']['active_read_bytes']=1
    elif change=='event':data['components'][0]['timing']='CUDA_EVENTS'
    elif change=='provider':data['components'][1]['entry']='PROVIDER'
    elif change=='repeat':data['trace_schedule'][1]=deepcopy(data['trace_schedule'][0])
    else:data['proof'][0]['oracle_activation']='ORIGINAL_F32'
    with pytest.raises(ValueError):parse_rows(data,point,auth,rows,{})


def test_native_plan_only_does_not_need_sdk_or_llama(tmp_path):
    output=tmp_path/'baseline'
    got=subprocess.run([sys.executable,str(ROOT/'tools/run_smallm_native_asys.py'),
        '--output',str(output),'--sdk','/absent-sdk','--plan-only'],text=True,capture_output=True)
    assert got.returncode==0,got.stdout+got.stderr
    assert len(json.loads((output/'plan.json').read_text())['points'])==16
    assert not (output/'build').exists()


def test_real_cpp_profile_warmup_before_range_no_events_and_exception_cleanup(tmp_path):
    # Compile the actual nc_profile function with a host mock launch boundary.
    # This proves orchestration, NOT device graph admission or performance.
    text=(ROOT/'dev/native_components/bridge.cuh').read_text()
    body=text[text.index('extern "C" const char * nc_profile('):text.index('// The shared llama GPU routing helper')]
    assert 'Event' not in body and 'Elapsed' not in body
    source=tmp_path/'profile.cpp'
    vendor=Path('/root/autodl-tmp/llama-kpack-dense/vendor')
    if not (vendor/'nlohmann/json.hpp').exists():pytest.skip('local llama vendor header unavailable')
    source.write_text(r'''
#include <nlohmann/json.hpp>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
static bool active=false;static int starts=0,stops=0,warms=0,runs=0,fail_after=-1;
static int cudaStreamSynchronize(int) { return 0; }
static int cudaProfilerStart() { if(warms!=4)throw std::runtime_error("warmup was not excluded");active=true;++starts;return 0; }
static int cudaProfilerStop() { active=false;++stops;return 0; }
namespace nc {
using Json=nlohmann::ordered_json;
inline void check(int rc,const char * step) { if(rc)throw std::runtime_error(step); }
inline std::string error;
template<class F> int guard(F f) {try {f();return 0;}catch(const std::exception & e) {error=e.what();return 1;}}
struct Stage {};
struct Record {std::vector<Stage> stages=std::vector<Stage>(2);};
struct Context {int stream(){return 0;}};
struct Session {Context ctx;std::vector<std::unique_ptr<Record>> records;std::string text;};
struct KernelGraph {
 explicit KernelGraph(const std::vector<Stage *> & stages) { if(active||stages.size()<16)throw std::runtime_error("bad build"); }
 void run(int) {if(active){if(runs++==fail_after)throw std::runtime_error("injected");}else ++warms;}
};
}
'''+body+r'''
extern "C" const char * exercise(int fail) {
 static nc::Session session;session.records.clear();
 for(int i=0;i<2;++i)session.records.push_back(std::make_unique<nc::Record>());
 active=false;starts=stops=warms=runs=0;fail_after=fail;int rotate[]={0,1};
 return nc_profile(&session,rotate,2,6,5);
}
extern "C" int cleanup_ok() {return starts==1 && stops==1 && !active;}
''')
    lib=tmp_path/'profile.so'
    subprocess.run(['g++','-std=c++17','-O0','-shared','-fPIC','-I'+str(vendor),str(source),'-o',str(lib)],check=True,capture_output=True)
    api=C.CDLL(str(lib));api.exercise.argtypes=[C.c_int];api.exercise.restype=C.c_char_p
    result=json.loads(api.exercise(-1))
    assert len(result['schedule'])==60 and result['calls_per_graph']==[16,16]
    assert [x['stage'] for x in result['schedule'][:20]]==[0]*5+[1]*5+[1]*5+[0]*5
    assert api.cleanup_ok()==1
    assert api.exercise(3) is None and api.cleanup_ok()==1

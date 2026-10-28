"""Indexed native AB contract tests. Host evidence only, not PPU performance."""
import ctypes as C
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

from dev.native_components import moe_gate, selected_gate
from dev.native_components.asys import validate_capture
from dev.strategy.fixture_identity import expected, fingerprint
from dev.strategy.prefill_fixture import routing
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest
from tools import run_smallm_native_gate as runner
from tools.run_kpack_llama_ab import point_request
from test_moe_model_plan import synthetic_inventory
from test_smallm_native_asys import fixture as native_capture

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    return selected_gate.plan(tmp_path_factory.mktemp('moe-native-plan'),synthetic_inventory())


def test_actual_selector_not_historical_config_relabel(plan):
    assert plan['cohort']==moe_gate.COHORT and plan['schema']==selected_gate.SCHEMA
    assert plan['counts']['points']==216 and plan['counts']['arms']==216
    assert plan['counts']['native_passes']==plan['counts']['candidate_passes']==2
    assert plan['protocol']['candidate_arithmetic']=='F32_IO_BF16_COMPUTE_FP32_ACC_CURRENT_SELECTED_RECIPE'
    assert plan['sha256']==digest({k:v for k,v in plan.items() if k!='sha256'})
    assert not plan['config_sweep'] and not plan['historical_times_reused']
    used={'tc':set(),'vectors':set()}
    for point in plan['points']:
        arm,=point['arms']
        assert arm['id']=='original' and arm['required']
        assert point['selection']['request'][-1]==1
        used['tc' if arm['recipe']['kind']=='tc' else 'vectors'].add(arm['module'])
    assert all(set(plan[k])==v for k,v in used.items())
    groups=runner.assignments(plan,5)
    assert all(groups) and max(map(len,groups))-min(map(len,groups))<=5
    assert sorted(pid for group in groups for pid in group)==sorted(p['id'] for p in plan['points'])


@pytest.mark.parametrize('ch',[1,8])
@pytest.mark.parametrize('m',[1,2,8])
@pytest.mark.parametrize('router',['spread','cluster','repeat2','repeat4','real'])
def test_timed_ids_and_a_equal_the_actual_candidate_fixture(ch,m,router):
    from dev.smallm_closure.fixture import Weights
    from dev.bf16_compute.fixture import round_compute
    w=Workload(12,512,2048,m,'indexed','bf16',experts=256,topk=8,channels=ch,router=router)
    # Exercise the production fixture's actual input/router generator without
    # creating hundreds of MB of deterministic weights on the CPU.
    fixture=object.__new__(Weights)
    fixture.q,fixture.n,fixture.k,fixture.experts=12,512,2048,256
    fixture.categories=np.random.default_rng(81811+w.k).integers(0,4,w.k)
    fixture.sums=np.zeros((17,4,w.n));fixture.absolute=np.ones_like(fixture.sums)
    from dev.strategy.indexed import point
    data=fixture.inputs(point(w),1)
    assert fingerprint(data['a'],data['ids'])==expected(w,1)
    p=dict(id=w.key,workload=asdict(w),cohort='decode')
    request=point_request(p,67108864,routing_repeat=1)
    assert request['owners']==list(map(int,data['owners']))
    assert request['ring']['router_sha256']==routing(w,1)['sha256']
    assert request['owners']!=point_request(p,67108864)['owners']


def test_native_moe_output_extent_and_unchanged_oracle():
    data,point,auth,_=native_capture()
    w=Workload(12,512,2048,4,'indexed','bf16',experts=256,topk=8,channels=8,router='repeat2')
    point=dict(point,id=w.key,workload=asdict(w));data.update(id=w.key,workload=asdict(w))
    for proof in data['proof']:proof['finite_cells_per_arm']=w.m*w.topk*w.n
    validate_capture(data,point,auth)
    data['proof'][0]['finite_cells_per_arm']=w.m*w.n
    with pytest.raises(ValueError,match='coverage'):validate_capture(data,point,auth)


@pytest.mark.parametrize('native',[False,True])
def test_mismatched_timed_router_cannot_pass(native):
    w=Workload(13,2048,512,8,'indexed','bf16',experts=256,topk=8,channels=8,router='real')
    point=dict(workload=asdict(w))
    value=dict(timing_input_repeat=1,timed_fixture=expected(w,1))
    data=value if native else dict(arms={'original':value})
    moe_gate.check_fixture(point,data,native=native)
    value['timed_fixture']=expected(w,0)
    with pytest.raises(ValueError,match='timed A or expert IDs'):moe_gate.check_fixture(point,data,native=native)


def test_native_crc_uses_the_same_raw_float_and_id_bytes(tmp_path):
    source=tmp_path/'checksum.cpp'
    source.write_text('#include "dev/native_components/fixture_fingerprint.hpp"\n'
                      'extern "C" unsigned checksum(const void* p, size_t n) {return nc_fixture_crc32(p,n);}\n')
    binary=tmp_path/'checksum.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(ROOT),source,'-o',binary],check=True)
    fn=C.CDLL(str(binary)).checksum;fn.argtypes=[C.c_void_p,C.c_size_t];fn.restype=C.c_uint32
    for raw in [b'',b'123456789',np.arange(512,dtype='<f4').tobytes(),np.arange(64,dtype='<i4').tobytes()]:
        buffer=C.create_string_buffer(raw)
        assert fn(buffer,len(raw))==zlib.crc32(raw)


@pytest.mark.parametrize('entry',['run_smallm_native_gate.py','run_moe_prefill_pilot.py'])
def test_selected_only_cli_actual_cpu_tp_and_resume(tmp_path,entry):
    llama = os.environ.get('LLAMA_TP_SOURCE')
    if not llama:
        pytest.skip('set LLAMA_TP_SOURCE for the actual caller CPU splitter')
    from test_tp_header import fixtures
    from tools.gguf_internal_shape_inventory import _synthetic_gguf
    from dev.strategy import moe_model_plan as models
    command = [sys.executable, str(ROOT/'tools'/entry),'--output',str(tmp_path/'plan'),
        '--llama-source',llama,'--sdk','/no-sdk','--devices','3,4,5,6,7','--plan-only',
        '--moe-models','qwen35-35b-a3b,qwen35-122b-a10b']
    if entry=='run_smallm_native_gate.py':
        command += ['--cohort',moe_gate.COHORT]
    for name,tp,flag,_ in models.MODELS[:2]:
        meta,tensors=fixtures('qwen35moe')
        meta.update({'qwen35moe.expert_count':256,'qwen35moe.expert_used_count':8})
        k,n=2048,512*tp
        tensors += [dict(name='blk.0.ffn_gate_exps.weight',qtype=12,dims_gguf=[k,n,256]),
                    dict(name='blk.0.ffn_up_exps.weight',qtype=12,dims_gguf=[k,n,256]),
                    dict(name='blk.0.ffn_down_exps.weight',qtype=13,dims_gguf=[n,k,256])]
        path=tmp_path/(name+'.gguf')
        path.write_bytes(_synthetic_gguf(list(meta.items()),[(t['name'],tuple(t['dims_gguf']),t['qtype']) for t in tensors]))
        command += ['--'+flag.replace('_','-'),str(path)]
    for extra in ([],['--resume']):
        result=subprocess.run(command+extra,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    value=json.loads((tmp_path/'plan/plan.json').read_text())
    if entry=='run_smallm_native_gate.py':
        assert value['counts']['points']==value['counts']['arms']==72
        assert all(runner.assignments(value,5))
    else:
        assert value['counts']['points']==6 and value['counts']['arms']==14
        from dev.strategy.smallm_asys import assignments
        assert all(assignments(value,5,balanced=True))
    assert not (tmp_path/'plan/run.json').exists()
    # Nothing on a CPU-only inventory can be declared native performance.
    assert not value['production_changed']
    changed=command.copy();changed[changed.index('--moe-models')+1]='qwen35-35b-a3b'
    result=subprocess.run(changed+['--resume'],capture_output=True,text=True)
    assert result.returncode!=0 and 'resume requested models differ' in result.stderr

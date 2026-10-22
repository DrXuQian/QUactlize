"""Production integration + sweep plumbing, no host timing admission."""
from dataclasses import asdict
import copy
import ctypes as C
from pathlib import Path
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from dev.prefill_point.bench import SF, WORKLOAD
from dev.prefill_point.typed import DenseLibrary
from dev.prefill_asys.plan import runnable
from dev.prefill_asys.trace import parse_rows
from dev.strategy.resources import probe_spec
from quactlize.dispatch.table import Recipe
from quactlize.runtime.tuning import digest
from test_prefill_asys import PrefillAsysTests

ROOT=Path(__file__).resolve().parents[1]


def test_runtime_descriptors(tmp_path):
    sdk=Path(os.environ.get('PPU_SDK','/root/ppu-sdk/2.1.1'))
    exe=tmp_path/'descriptor'
    subprocess.run(['g++','-std=c++17','-O2','-ffunction-sections','-fdata-sections',
        '-Wl,--gc-sections','-I'+str(sdk/'include'),str(ROOT/'tests/prefill_point_descriptor_host.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_c_api_choice_is_single_source_not_python_table():
    library=DenseLibrary.__new__(DenseLibrary)
    def policy(c,storage,dq,io):
        s=c._obj;assert (s.weight.qtype,s.m,s.weight.n,s.weight.k)==(12,2048,1024,5120)
        dq._obj.value=20;io._obj.value=2;return 0
    library.policy=policy
    assert library.choice(WORKLOAD,0)==dict(dequant_variant=20,vector_io=2,storage=0)
    library.policy=lambda *a:1
    import pytest
    with pytest.raises(RuntimeError):library.choice(WORKLOAD,0)
    fake=SimpleNamespace(w=WORKLOAD,r=SimpleNamespace(stream=C.c_void_p(4)))
    library.io=C.CFUNCTYPE(C.c_int)(lambda:0)
    d=library.io_call(fake,1,2,input=True,storage=0,vector=2)
    assert d.kind==5 and list(d.p)[:4]==[1,2,None,4]
    assert list(d.v)[:5]==[0,1,2,2048,5120] and d.stride==5120


def test_pool_has_new_precision_paths_and_immutable_old_controls():
    point=dict(id=WORKLOAD.key,workload=asdict(WORKLOAD),candidates=[asdict(SF)],
        champion_controls=[asdict(SF)],original_recipe=asdict(SF),model_config_control=asdict(SF),
        sf_config=4,full_configs=[0,1,12])
    q=probe_spec(WORKLOAD,SF,1)
    point['queries']={digest(q):dict(status='READY',device=dict(compute_units=72),occupancy=4)}
    with patch('dev.strategy.space.resolved_grid',return_value=dict(algorithm=1,split=1,grid=288)):
        rows=runnable(point)
    assert len({r['id'] for r in rows})==len(rows)==7
    tc=[r for r in rows if r['arm']=='sf']
    assert {r['helper_mode'] for r in tc}=={'production','legacy'}
    assert tc[0]['recipe']==tc[1]['recipe']  # SAME GEMM config, different helper receipt.
    typed=[r for r in rows if r['arm'] in ('full-fp16','full-bf16-typed')]
    assert len(typed)==2 and all(r['helper_mode']=='production' for r in typed)
    assert all('dequant_variant' not in dict(r['recipe']['parameters']) for r in typed)
    assert {dict(r['recipe']['parameters'])['dequant_config'] for r in rows if r['arm']=='full-bf16'}=={0,1,12}


def test_new_gpu_inventory_and_precision_guard():
    import pytest
    rows,case=PrefillAsysTests().rows('full-bf16')
    for arm,bf in [('full-fp16','false'),('full-bf16-typed','true')]:
        changed=copy.deepcopy(rows);c=dict(case,arm=arm)
        for r in changed:
            if 'gather_bf16' in r['name']:r['name']=f'dense_convert_stripe4<{bf}, true, true>(...)'
            elif 'finish_bf16' in r['name']:r['name']=f'dense_convert_stripe4<{bf}, false, true>(...)'
            elif 'full_packed_exchange' in r['name']:r['name']=f'dense_dequant_cooperative<(gguf_scale::KType)2, {bf}, 64, 128, true>(...)'
        out=parse_rows(changed,[c])[0]
        assert out['phases_us']==dict(error_reset=1,input=1,expansion=1,main=1,output=1)
        assert out['gpu_sum_us']==5
        if arm=='full-fp16':
            changed[1]['name']='dense_convert_stripe4<true, true, true>(...)'
            with pytest.raises(ValueError,match='precision'):parse_rows(changed,[c])


def test_same_dtype_winner_does_not_relabel_faster_bf16():
    from tools.run_prefill_model_asys import summarize
    point=dict(id='p',workload=asdict(WORKLOAD),tensors=['one'],cases=[
        dict(id='f',arm='full-fp16',recipe=asdict(Recipe.make('full','full')),helper_mode='production',roles=[]),
        dict(id='b',arm='full-bf16-typed',recipe=asdict(Recipe.make('full','full')),helper_mode='production',roles=[])])
    rows={k:dict(case_id=k,gpu_sum_us=t,phases_us={'main':t-2}) for k,t in [('native-fp16',20),('f',21),('b',15)]}
    result=summarize([point],{'p':rows})['points'][0]
    assert result['best_id']=='b' and result['same_dtype_best_id']=='f'
    assert result['same_dtype_delta_pct']>0

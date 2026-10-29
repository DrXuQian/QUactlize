"""Host-only gate inventory, output binding and trace accounting tests."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.moe_grouped.bench import Result,value,parse
from dev.moe_grouped.build import parent
from quactlize.runtime.compiler import validate_parent
from tools.run_moe_grouped import plan

ROOT=Path(__file__).resolve().parents[1]


def test_model_gate_not_a_new_sweep():
    rows=plan(False)
    assert len(rows)==76 and len({r['id'] for r in rows})==76
    assert {r['m'] for r in rows}==set(range(1,10))|{16,32,64,128,256,512,1024,2048,4096,5120}
    for r in rows:
        assert r['k'] in (2048,3072) and r['profile'] in ('spread','cluster')
        assert r['k']%(256*r['gate_split'])==0
        assert 512%(256*r['down_split'])==0
    assert len(plan(True))==24
    for q in (8,10,11,12,13,14):validate_parent(parent(q))


def test_gate_cli_plan_requires_no_sdk():
    out=subprocess.check_output([sys.executable,str(ROOT/'tools/run_moe_grouped.py'),'--plan'],text=True)
    data=json.loads(out)
    assert data['points']==76 and data['modules']==6 and data['contract_points']==28
    assert not data['production_changed'] and not data['config_search']


@pytest.mark.parametrize('compute',[0,1])
@pytest.mark.parametrize('split',[1,2,4,8])
def test_actual_result_drives_helper_state(compute,split):
    r=Result(1,C.sizeof(Result),0x1000,2 if split>1 else compute,split,compute,0,512,8192 if split>1 else 0)
    v=value(r)
    assert v.state==int(split>1) and v.order==0 and v.partial_count==split
    assert v.dtype==(0 if split>1 else compute+1) and v.round_type==compute+1


def fixture(count=16):
    capture=dict(points=[dict(job={'id':'p'},components={'control':['c'],'selected':['s']})],
        schedule=[dict(component=x,kernel='producer',repeats=16) for x in ('c','s')]+
                 [dict(component='p/chain/'+x,kernel=None,repeats=1) for x in ('control','selected')])
    rows=[];calls=[]
    for i in range(4):
        calls.append(dict(name='hggcGraphLaunch',correlation_id=i,return_value=0))
        for j in range(count if i<2 else 3):
            rows.append(dict(correlation_id=i,kind='kernel',name='producer',grid=[8,1,1],block=[256,1,1],
                             start=i*10000+j*100,end=i*10000+j*100+50))
    return capture,rows,calls


def test_component_and_actual_chain_are_distinct_metrics():
    out=parse(*fixture())
    assert out['component_us']['c']==.05
    p=out['points'][0]
    assert p['isolated_component_sum_us']=={'control':.05,'selected':.05}
    assert p['chain_activity_us']=={'control':.15,'selected':.15}
    assert p['chain_span_us']=={'control':.25,'selected':.25}
    assert not out['missing_intervals'] and not out['imputed_intervals']


def test_actual_activity_counts_not_fabricated():
    out=parse(*fixture(12))
    assert out['component_us']['c']==pytest.approx(.05) and out['missing_intervals']==8
    capture,rows,calls=fixture();rows[0]['name']='wrong'
    with pytest.raises(ValueError,match='unexpected'):parse(capture,rows,calls)
    capture,rows,calls=fixture();rows[0]['grid']=[9,1,1]
    with pytest.raises(ValueError,match='launch changed'):parse(capture,rows,calls)

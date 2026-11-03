from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dev.moe_decode_helpers.bench import workload_plan, BASELINE
from dev.moe_scaffold.models import assignments
from dev.moe_scaffold.protocol import parse, weighted

ROOT=Path(__file__).resolve().parents[1]


def test_real_host_predicates(tmp_path):
    target=tmp_path/'contract'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                    '-fsanitize=undefined','-I'+str(ROOT),str(ROOT/'tests/moe_decode_helpers_host.cpp'),
                    '-o',str(target)],check=True)
    assert 'DECODE_HELPER_HOST PASS' in subprocess.check_output([target],text=True)


def test_bounded_inventory_and_multi_device_assignments():
    cases,coverage=workload_plan()
    assert len(cases)==57 and len({c.key for c in cases})==57
    for n in (2048,3072):
        assert set(range(1,9)) <= {c.tokens for c in cases if c.phase=='finish' and c.output==n and c.dtype==0}
    for n in (512,1024):
        for topk in (1,8):
            assert set(range(1,9)) <= {c.tokens for c in cases if c.phase=='transition' and c.hidden==n and c.topk==topk}
    parts=assignments(cases,4)
    assert Counter(c['phase'] for p in parts for c in p)==Counter(c.phase for c in cases)
    assert sum(map(len,parts))==len(cases)
    assert coverage['baseline']==BASELINE
    assert not coverage['production_changed']
    assert len(workload_plan(True)[0])==12
    with pytest.raises(ValueError):workload_plan(phase='prepare')


@pytest.mark.parametrize('threads',[64,128])
@pytest.mark.parametrize('rows,cols',[(1,31),(7,2051),(17,3073),(65536,1)])
def test_finish_launch_ownership(threads,rows,cols):
    # Mirror the explicit 2-D mapping; visit each logical output once, including
    # rows beyond grid.y's 65535 cap. No assumption that N is a tile multiple.
    gy=min(rows,65535)
    row_ids=[row for block in range(gy) for row in range(block,rows,gy)]
    col_ids=[b*threads+t for b in range((cols+threads-1)//threads) for t in range(threads) if b*threads+t<cols]
    assert sorted(row_ids)==list(range(rows)) and col_ids==list(range(cols))


def test_snapshot_unique_writer_and_slot_order():
    nr=8193*8;gy=min(nr,65535)
    owners=[row for block in range(gy) for row in range(block,nr,gy)]
    assert sorted(owners)==list(range(nr))
    x=np.array([1e20,1,-1e20,3,0,0,0,0],np.float32).reshape(1,8,1)
    assert weighted(x,np.ones((1,8),np.float32))[0,0]==3
    assert weighted(x[:,[0,2,1,3,4,5,6,7]],np.ones((1,8),np.float32))[0,0]==4
    source=(ROOT/'dev/moe_decode_helpers/kernels.cuh').read_text()
    assert '__fmul_rn' in source and '__fadd_rn' in source
    assert 'blockIdx.x==0 && threadIdx.x==0' in source
    assert 'expf(-g)' in source


def test_asys_control_means_missing_intervals_are_not_filled():
    schedule=[];rows=[];apis=[]
    for cid,(arm,ns) in enumerate((('control',5000),('typed128',2500)),1):
        schedule.append(dict(component=arm,kernel=arm,grid=[16,1,1],block=[128,1,1],repeats=32))
        apis.append(dict(name='hggcGraphLaunch',return_value=0,correlation_id=cid))
        rows.extend(dict(kind='kernel',name=arm,grid=[16,1,1],block=[128,1,1],
                         start=i*ns,end=(i+1)*ns,correlation_id=cid) for i in range(31))
    capture=dict(schedule=schedule,points=[dict(key='point',components={'control':['control'],'typed128':['typed128']})],
                 device={'name':'host-only'},baseline=BASELINE)
    result=parse(capture,rows,apis)
    assert result['missing_intervals']==2 and result['imputed_intervals']==0
    assert result['baseline']==BASELINE
    point=result['points'][0]
    assert point['arms_us']=={'control':5,'typed128':2.5}
    assert point['versus_control_pct']['typed128']==-50
    assert not result['model_tpot_admitted']


def test_production_choice_and_gemm_sources_untouched():
    src=(ROOT/'quactlize/moe/selection.hpp').read_text()
    assert 'c.down.round_type!=QZM_BF16' in src
    assert 'decode_helpers' not in src
    runner=(ROOT/'tools/run_moe_scaffold.py').read_text()
    assert "scaffold_source='dev/moe_decode_helpers/native.cu'" in runner
    assert "extra_sources=('dev/moe_decode_helpers',)" in runner
    native=(ROOT/'dev/moe_decode_helpers/native.cu').read_text()
    assert 'if(!decode_helpers::finish_fast(*c)) return quactlize_ppu_moe_finish_run_v1(c)' in native

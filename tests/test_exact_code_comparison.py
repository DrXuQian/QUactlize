import copy
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools.decode_gemv.compare_extension import (
    BASELINE, ORDER, checked_capture, cold_plan, compare, export_baseline, inventory, query_l2,
)
from tools.decode_gemv.parse_asys import attribute, stats


def fixture(split=2):
    case = dict(id='direct/test', reader='direct', kind='dense', m=1, qtype=12, n=1024, k=5120,
                proto=dict(split=split))
    check = dict(cells=16, bad=0, nonfinite=0, tolerance=.005, relative_l1_error=1e-8,
                 golden_sha256='independent-oracle')
    raw = dict(case=case, status='OK', check=check, samples=4, marker=[1048576,1048640],
               config='fixed', split=split, graph_calls=2, event_us_per_call=13.0,
               cold_weights=dict(l2_bytes=1024, unique_weight_bytes=2304, ring=2),
               device=dict(pci='0000:7E:00.0', visible_devices='1', ordinal=0),
               fixture=dict(seed=1, gguf_sha256='weight-bytes', a_sha256='input', ids_sha256=None))
    rows = [dict(kind='memset',start=0,end=100,bytes=1048576,stream=3,device=0)]
    cursor = 1000
    for _ in range(4):
        for name,duration in [('mma_reader::kernel<12>',10000)] + ([('register_reuse_reduce<12>',2000)] if split>1 else []):
            rows.append(dict(kind='kernel',start=cursor,end=cursor+duration,name=name,
                             stream=3,device=0,grid=(16,1,split),block=(128,1,1),regs=80,smem=4096))
            cursor += duration+1000
    rows.append(dict(kind='memset',start=cursor,end=cursor+100,bytes=1048640,stream=3,device=0))
    return case,raw,rows


def test_inventory_is_fixed_no_config_sweep():
    rows=inventory(Path('/build'))
    assert len(rows)==75 and len({r['id'] for r in rows})==75
    assert sum(r['reader']=='direct' for r in rows)==46
    assert sum(r['reader']=='staged' for r in rows)==29
    assert ORDER==('baseline','extension','extension','baseline')


def test_queried_sdk_enum_not_device_name(tmp_path):
    (tmp_path/'include').mkdir()
    (tmp_path/'include/driver_types.h').write_text('enum attr { hggcDevAttrL2CacheSize = 38 };')
    def query(out, attribute_id, device):
        assert (attribute_id,device)==(38,0)
        C.cast(out,C.POINTER(C.c_int))[0]=64<<20
        return 0
    sdk=SimpleNamespace(lib=SimpleNamespace(hggcDeviceGetAttribute=query))
    assert query_l2(sdk,tmp_path)==64<<20
    (tmp_path/'include/driver_types.h').write_text('not the enum')
    with pytest.raises(ValueError,match='cannot resolve'):
        query_l2(sdk,tmp_path)


def test_dense_whole_ring_and_cap():
    row=cold_plan(dict(kind='dense'), 3<<20, 64<<20, 50, 192)
    assert row['ring']==48 and row['graph_calls']==96
    assert row['unique_weight_bytes']==144<<20
    with pytest.raises(ValueError,match='needs'):
        cold_plan(dict(kind='dense'), 1<<20, 64<<20, 48, 100)


def test_m3_expert_rotation_has_no_boundary_reuse():
    row=cold_plan(dict(kind='moe',experts=256,m=3,topk=8), 256<<20,64<<20,48,192)
    assert row['ring']==10 and row['graph_calls']==50
    assert row['unique_weight_bytes']==240<<20
    with pytest.raises(ValueError,match='below'):
        cold_plan(dict(kind='moe',experts=256,m=3,topk=8),64<<20,64<<20,48,192)


def test_all_75_historical_cases_have_valid_cold_working_sets():
    data=Path(__file__).resolve().parents[1]/'docs/decode_gemv/data'
    count=0
    for name in ('correct-mma-reader-v3.jsonl','correct-mma-staged-v1.jsonl'):
        for line in (data/name).read_text().splitlines():
            row=json.loads(line)
            if row['status']!='OK':continue
            cold=cold_plan(row['case'],row['weight_bytes'],64<<20,48,192)
            assert cold['graph_calls']%cold['ring']==0
            assert cold['unique_weight_bytes']>=144<<20
            count+=1
    assert count==75


@pytest.mark.parametrize('split',[1,2])
def test_asys_full_call_includes_required_reducer(split):
    case,raw,rows=fixture(split)
    parsed=attribute([raw],rows)
    record=checked_capture([case],[raw],parsed)[0]
    assert record['main_us']==10
    assert record['reducer_us']==(2 if split>1 else 0)
    assert record['total_us']==(12 if split>1 else 10)
    # Event span is diagnostic only, never substituted for the Asys kernel sum.
    assert record['event_us_per_call']==13


def test_partial_capture_and_numeric_failure_never_pass():
    case,raw,rows=fixture()
    with pytest.raises(ValueError):
        checked_capture([case],[raw],attribute([raw],rows[:3]+rows[4:]))
    # A trace containing only main kernels is divisible by samples but still invalid.
    main_only=[r for r in rows if 'reduce' not in r.get('name','')]
    with pytest.raises(ValueError,match='expected main'):
        checked_capture([case],[raw],attribute([raw],main_only))
    raw['check']['bad']=1
    with pytest.raises(ValueError,match='GGUF oracle'):
        checked_capture([case],[raw],attribute([raw],rows))


def captures():
    case,raw,rows=fixture()
    base=checked_capture([case],[raw],attribute([raw],rows))[0]
    result=[]
    for arm,values in [('baseline',[10,12]),('extension',[11,13])]:
        for value in values:
            rec=copy.deepcopy(base)
            rec.update(total_us=value,main_us=value-2)
            result.append(dict(arm=arm,records=[rec]))
    return case,result


def test_repeat_medians_not_fastest_sample_or_blanket_gate():
    case,frames=captures()
    report=compare(frames,[case['id']],2)
    point=report['results'][0]
    assert point['measurements']['baseline']['median_total_us']==11
    assert point['measurements']['extension']['median_total_us']==12
    assert point['delta_pct']==pytest.approx(100/11)
    assert not point['repeat_ranges_separated']
    assert report['slower_medians']==1 and report['slower_separated_ranges']==0
    assert report['status']=='MEASURED' and not report['performance_admitted']


@pytest.mark.parametrize('seam',['device','config','fixture','geometry','count'])
def test_no_cross_device_config_fixture_or_missing_repeat_pooling(seam):
    case,frames=captures()
    last=frames[-1]['records'][0]
    if seam=='device':last['device']['pci']='0000:01:00.0'
    elif seam=='config':last['config']='different'
    elif seam=='fixture':last['fixture']['gguf_sha256']='different'
    elif seam=='geometry':last['components'][0]['grid']=(32,1,2)
    else:frames.pop()
    with pytest.raises(ValueError):compare(frames,[case['id']],2)


def test_baseline_export_is_immutable_prototype(tmp_path):
    output=tmp_path/'baseline'
    export_baseline(output)
    root=Path(__file__).resolve().parents[1]
    name='quactlize/execution/mma_reader.cuh'
    expected=subprocess.check_output(['git','-C',str(root),'show',BASELINE+':'+name])
    assert (output/name).read_bytes()==expected
    assert (output/name).read_bytes()!=(root/name).read_bytes()
    assert not (output/'quactlize/include/actlize_extensions/cutlass/gemm/collective/ppu_mma_exact_code.hpp').exists()


def test_plan_only_without_sdk_or_gpu_and_resume(tmp_path):
    script=Path(__file__).resolve().parents[1]/'tools/decode_gemv/compare_extension.py'
    output=tmp_path/'plan'
    cmd=[sys.executable,str(script),'--plan-only','--sdk','/missing-sdk','--output',str(output)]
    first=subprocess.run(cmd,text=True,capture_output=True)
    assert first.returncode==0,first.stdout+first.stderr
    repeated=subprocess.run(cmd+['--resume'],text=True,capture_output=True)
    assert repeated.returncode==0,repeated.stdout+repeated.stderr
    assert json.loads((output/'summary.json').read_text())['status']=='PLAN_ONLY'
    changed=subprocess.run(cmd+['--resume','--samples','999'],text=True,capture_output=True)
    assert changed.returncode==1 and 'protocol changed' in changed.stdout
    with tarfile.open(output/'review.tgz') as archive:
        assert 'plan.json' in archive.getnames()
        assert not any(n.endswith(('.so','.o','.asysrep','.sqlite')) for n in archive.getnames())

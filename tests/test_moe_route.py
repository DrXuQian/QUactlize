import ctypes as C
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dev.moe_route.protocol import Call, Config, Sizes, Case, ids_for, oracle, tiled_model, parse, workloads

ROOT = Path(__file__).resolve().parents[1]


def test_host_contract(tmp_path):
    target = tmp_path/'host'
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                    '-I'+str(ROOT), str(ROOT/'tests/moe_route_host.cpp'), '-o', str(target)], check=True)
    assert 'MOE_ROUTE_HOST PASS' in subprocess.check_output([target], text=True)
    assert (C.sizeof(Call), C.sizeof(Config), C.sizeof(Sizes)) == (104, 16, 24)
    assert (Call.ids.offset, Call.workspace.offset, Call.stream.offset) == (48, 80, 96)


@pytest.mark.parametrize('m', [0, 1, 3, 8, 17, 129, 513, 4097, 5121])
@pytest.mark.parametrize('geometry', [(1, 1, 1), (33, 6, 3), (256, 8, 8), (1024, 32, 1)])
def test_stable_maps(m, geometry):
    e, k, channels = geometry
    for inverse in (0, 1):
        case = Case(m, e, k, channels, padding=5, inverse=inverse)
        ids = ids_for(case, 5)
        assert np.all([len(set(row[:k]))==k for row in ids])
        want = oracle(ids, case)
        for tile in (256, 512, 1024):
            actual = tiled_model(ids, case, tile)
            for first, second in zip(want, actual): np.testing.assert_array_equal(first, second)
        assert want[2][-1]==m*k


@pytest.mark.parametrize('profile', ['spread', 'cluster', 'repeat2', 'repeat4', 'random'])
def test_profiles_and_changed_ids(profile):
    case = Case(63, profile=profile)
    first, second = ids_for(case, 0), ids_for(case, 1)
    assert not np.array_equal(first, second)
    for ids in (first, second):
        for a,b in zip(oracle(ids,case),tiled_model(ids,case,256)): np.testing.assert_array_equal(a,b)


def trace_fixture():
    c=Case(1);components={'warp':['candidate'],'native':['native']}
    capture=dict(points=[dict(key=c.key,case=dict(tokens=1),components=components,auto='warp',native_status='SUPPORTED')],
                 schedule=[],device=dict(name='HOST_TEST_ONLY'))
    rows=[];apis=[]
    for i, (key, kernel, ns) in enumerate([('candidate','warp_compact<8>',1000),('native','mm_ids_helper<8>',2000)]):
        capture['schedule'].append(dict(component=key,kernel=kernel,grid=[256,1,1],block=[32,1,1],repeats=32))
        apis.append(dict(name='hggcGraphLaunch',return_value=0,correlation_id=i+1))
        rows.extend(dict(kind='kernel',name=kernel,grid=[256,1,1],block=[32,1,1],start=j*ns,end=(j+1)*ns,
                         correlation_id=i+1) for j in range(32))
    return capture,rows,apis


def test_asys_only_mean_and_observed_denominator():
    capture, rows, apis=trace_fixture()
    rows.pop(2)
    result=parse(capture,rows,apis)
    assert result['missing_intervals']==1 and result['imputed_intervals']==0
    assert result['points'][0]['best_us']==1 and result['points'][0]['native_us']==2
    assert result['points'][0]['delta_pct']==-50
    # Adjacent timestamps and inter-component timestamp overlap are not a gate.
    assert result['observed_counts']['candidate']==31


@pytest.mark.parametrize('fault',['name','grid','empty','api','extra','few'])
def test_bad_trace_rejected(fault):
    c,r,a=trace_fixture()
    if fault=='name':r[0]['name']='unrelated'
    if fault=='grid':r[0]['grid']=[1,1,1]
    if fault=='empty':r=[]
    if fault=='api':a[0]['return_value']=1
    if fault=='extra':r.append(r[0])
    if fault=='few':r=r[20:]
    with pytest.raises(ValueError):parse(c,r,a)


def test_coverage_and_no_dirty_admission():
    cases=workloads()
    assert len({c.key for c in cases})==len(cases)
    assert set(range(1,9))<={c.tokens for c in cases}
    assert {512,1024,2048,4096,5120,32769,65536}<={c.tokens for c in cases}
    source=(ROOT/'dev/moe_route/build.py').read_text()
    assert 'commit the sweep' not in source and 'build_adapter(' not in source

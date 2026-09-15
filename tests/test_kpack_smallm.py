import copy
import ctypes as C
import gzip
import json
from pathlib import Path
import subprocess

import pytest

from tools import fit_kpack_smallm as fit
from quactlize.runtime.compiler import validate_parent
from quactlize.dispatch.native import Dispatch, SmallmChoice
from quactlize.execution.native import Call, arrangement

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def data():
    return json.loads(gzip.decompress(fit.EVIDENCE.read_bytes()))


@pytest.fixture(scope='module')
def policy():
    return json.loads(fit.POLICY.read_text())


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    path=tmp_path_factory.mktemp('smallm')/'probe'
    subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',str(ROOT/'tests/kpack_smallm_host.cpp'),'-o',str(path)],check=True)
    return path


def query(probe,keys):
    return subprocess.check_output([str(probe)],input=''.join(' '.join(map(str,k))+'\n' for k in keys),text=True).splitlines()


def test_regenerate_exact_and_bucket_tables(data,policy):
    p=fit.fit(data)
    p['evidence_sha256']=fit.sha(fit.EVIDENCE.read_bytes())
    assert p==policy
    assert fit.header(p)==fit.POLICY.with_suffix('.hpp').read_text()
    assert p['summary']['simt']==30
    assert len(p['exact'])==len({tuple(r['key']) for r in p['exact']})==305
    assert {r['key'][0] for r in p['exact']}=={8,10,11,13,14}


def test_exact_choices_and_legal_tc_parents(probe,policy):
    for row,got in zip(policy['exact'],query(probe,[r['key'] for r in policy['exact']])):
        fields=got.split();c=row['config'];q,mode,n,k,m,ch=row['key']
        assert fields[:4]==['9',str(n),str(k),str(m)]
        assert fields[4]==c['kind']
        if c['kind']=='simt':
            assert list(map(int,fields[5:]))==[c[f] for f in ('variant','columns','warps','values','split')]
        else:
            assert fields[5:]==[c['symbol'],str(c['split'])]
            p={f:c[f] for f in ('qtype','symbol','tm','tn','tk','wm','wn','stages','ap','dn')}
            p.update(route=('fq-dense','sf-dense','fq-grouped','sf-grouped')[c['route']],persistent=c['parent_persistent'])
            validate_parent(p)
            assert k%(c['tk']*c['split'])==0
            assert k//(c['tk']*c['split'])>=c['stages']-1


def test_bucket_can_choose_simt_for_each_other_kformat(probe):
    # New M/shape keys must not get trapped behind the old TC-only fallback.
    keys=[(q,0,768,2048,3,1) for q in (10,11,13,14,8)]
    result=query(probe,keys)
    assert all(r.startswith('10 ') and ' simt ' in r for r in result),result
    # Dense and MoE channel modes stay distinct, and Q4 keeps its own table.
    assert query(probe,[(12,0,512,2048,1,1),(13,2,2048,512,9,8),(13,2,2048,512,1,2)])==['MISS']*3
    assert ' tc ' in query(probe,[(14,0,5120,8192,8,1)])[0]


@pytest.mark.parametrize('mutation',('missing','duplicate','nan','median'))
def test_evidence_negatives(data,mutation):
    d=copy.deepcopy(data)
    if mutation=='missing':d['simt'].pop()
    if mutation=='duplicate':d['simt'][0]=d['simt'][1]
    if mutation=='nan':d['simt'][0]['samples'][0][0]=float('nan')
    if mutation=='median':d['simt'][0]['median_us']*=2
    with pytest.raises(ValueError):fit.fit(d)


def test_tc_costs_do_not_add_reducer_twice(data,policy):
    assert all(r['scope'] in ('COMPLETE_FP16_TC_NO_EXTERNAL_ADAPTERS','PROFILED_MODEL_PRODUCER_PLUS_REDUCER') for r in data['tc'])
    assert policy['performance_admission']=='PENDING_CONTEMPORANEOUS_MODEL_GATE'
    for r in policy['review']:
        if r['decision']=='CROSS_COHORT_SIMT_PROPOSAL':
            assert r['simt_us']<r['tc_us']


@pytest.fixture(scope='module')
def host_dispatch(tmp_path_factory):
    root=tmp_path_factory.mktemp('smallm-binding')
    (root/'catalog.inc').write_text('static std::vector<Image> const kImages{};\nstatic char const kJitSource[]="";\n')
    subprocess.run(['g++','-std=c++17','-O1','-shared','-fPIC','-pthread',f'-I{root}',
        str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl','-o',str(root/'libquactlize_kpack_dispatch.so')],check=True)
    dispatch=Dispatch(root)
    yield dispatch
    dispatch.close()


def small_call(q=8,n=512,k=2048,m=1):
    return Call(version=1,size=C.sizeof(Call),qtype=q,n=n,k=k,experts=1,rows=m,mode=0,
        input_type=1,channels=1,topk=1,a_row_stride=k,a_token_stride=k,ids_stride=1,out_row_stride=n)


def test_c_abi_selects_without_loading_gpu_or_jit(host_dispatch):
    for q,n,m,policy in ((8,512,1,9),(10,768,3,10),(11,768,3,10),(13,768,3,10),(14,768,3,10)):
        call=small_call(q,n,m=m)
        result=host_dispatch.query_smallm(call,arrangement(q))
        assert result is not None and result.size==C.sizeof(SmallmChoice) and result.version==1
        assert result.kind==1 and result.policy==policy and result.tc.ticket==0
        assert result.sizes.workspace_bytes==(m*n*result.simt.split*4 if result.simt.split>1 else 0)
    assert host_dispatch.query_smallm(small_call(q=12),arrangement(12)) is None
    # A legal TC proposal is not silently replaced by SIMT if its module/JIT is absent.
    assert host_dispatch.query_smallm(small_call(q=14,n=5120,k=8192,m=8),arrangement(14)) is None


@pytest.mark.parametrize('field,value',(('size',0),('input_type',7),('rows',0),('n',513),('a_row_stride',1)))
def test_c_abi_rejects_invalid_call(host_dispatch,field,value):
    call=small_call();setattr(call,field,value)
    with pytest.raises(ValueError,match='small-M query'):
        host_dispatch.query_smallm(call,arrangement(8))


def test_c_abi_does_not_reinterpret_another_arrangement(host_dispatch):
    with pytest.raises(ValueError,match='small-M query'):
        host_dispatch.query_smallm(small_call(),arrangement(12))
    call=small_call();call.a=17
    assert host_dispatch.query_smallm(call,arrangement(8)) is None

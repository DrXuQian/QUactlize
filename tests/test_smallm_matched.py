import copy
import ctypes as C
import gzip
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import fit_smallm_closure as fit
from dev.smallm_closure.plan import q4_config
from quactlize.dispatch.native import Dispatch
from quactlize.execution.native import Call, arrangement

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def policy():return json.loads((ROOT/'policies/kpack_smallm_matched_v1.json').read_text())


@pytest.fixture(scope='module')
def evidence():return json.loads(gzip.decompress((ROOT/'docs/measurements/smallm_matched_20260915.json.gz').read_bytes()))


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    path=tmp_path_factory.mktemp('matched-policy')/'probe'
    subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',str(ROOT/'tests/smallm_matched_host.cpp'),'-o',str(path)],check=True)
    return path


def query(probe,keys):
    return subprocess.check_output([str(probe)],input=''.join(' '.join(map(str,k))+'\n' for k in keys),text=True).splitlines()


def test_regenerate_all_and_keep_unresolved_visible(policy,evidence):
    assert fit.fit(evidence)==policy
    assert fit.header(policy)==(ROOT/'policies/kpack_smallm_matched_v1.hpp').read_text()
    assert evidence['expected_points']==2988 and len(evidence['points'])==2972
    assert evidence['receipt_count']==169896
    assert policy['summary']==dict(exact=1842,buckets=634,excluded=30,router_sensitive=88)
    assert len(policy['missing'])==16
    assert {r['key'][0] for r in policy['exact']}=={8,10,11,12,13,14}
    assert {r['key'][-1] for r in policy['exact']}=={0,1}


def test_every_exact_choice_matches_source_table(probe,policy):
    keys=[r['key'] for r in policy['exact']]
    assert len(keys)==len(set(map(tuple,keys)))
    for r,line in zip(policy['exact'],query(probe,keys)):
        f=line.split();q,mode,n,k,e,top,ch,m,compute=r['key'];c=r['config']
        assert f[:4]==[str(14 if r['status']=='ROUTER_SENSITIVE_PARETO' else 12),str(n),str(k),str(m)]
        assert int(f[4])=={'tc':0,'simt':1,'q4':2}[c['kind']]
        if c['kind']=='tc':assert f[5:]==[c['symbol'],str(c['split'])]
        else:assert list(map(int,f[5:]))==[c.get('reader',0),c['variant'],c['columns'],c['warps'],c['values'],c.get('split',1)]


def test_q4_exact_recipes_are_existing_compiled_auto_choices(policy):
    old=json.loads((ROOT/'policies/kpack_q4_decode_v1.json').read_text())
    count=0
    for r in policy['exact']:
        c=r['config']
        if c['kind']!='q4':continue
        q,mode,n,k,e,top,ch,m,compute=r['key'];count+=1
        assert q==12 and compute==0
        hits=[x for x in old['ranges'] if x['role']=='auto' and x['operator']==('dense' if mode==0 else 'grouped') and x['n']==n and x['k']==k and x['first']<=m<=x['last']]
        assert len(hits)==1
        assert q4_config(hits[0]['recipe'])['recipe']=={f:c[f] for f in ('reader','variant','warps','values','columns')}
    assert count==119


def test_open_shapes_cannot_sneak_in_through_bucket(probe,policy):
    keys=[r['key'][:-1]+[int(r['key'][-1]=='bf16')] for r in policy['excluded']]
    assert query(probe,keys)==['MISS']*len(keys)
    invalid=[[8,0,32,2048,1,1,1,1,0],[14,0,151936,5120,1,1,1,1,1],
             [8,0,512,2048,1,1,1,9,0],[8,2,512,2048,256,8,2,1,0],
             [8,0,1048576,1048576,1,1,1,1,0],[8,0,512,2048,1,1,1,1,99]]
    assert query(probe,invalid)==['MISS']*len(invalid)


def test_bucket_has_new_compute_specific_donors(probe):
    keys=[[q,0,768,2048,1,1,1,3,c] for q in (8,10,11,12,13,14) for c in (0,1)]
    rows=query(probe,keys)
    assert all(r.startswith('13 ') for key,r in zip(keys,rows) if key[0]!=12),rows
    # Specialized Q4 rows are not portable to an uncompiled N/K. A miss is
    # preferable to manufacturing an exact recipe for the new geometry.
    assert all(r=='MISS' or r.startswith('13 ') for r in rows)


@pytest.mark.parametrize('mutation',('missing','median','compute','candidate'))
def test_evidence_edits_are_rejected(evidence,mutation):
    e=copy.deepcopy(evidence)
    if mutation=='missing':e['points'].pop()
    if mutation=='median':next(r for r in e['points'] if r['winner'])['winner']['median_us']*=.1
    if mutation=='compute':e['points'][0]['point_spec']['compute']='bf16'
    if mutation=='candidate':e['candidates'].pop(next(iter(e['candidates'])))
    with pytest.raises(ValueError):fit.fit(e)


@pytest.fixture(scope='module')
def host_dispatch(tmp_path_factory):
    path=tmp_path_factory.mktemp('matched-dispatch')
    (path/'catalog.inc').write_text('static std::vector<Image> const kImages{};\nstatic char const kJitSource[]="";\n')
    subprocess.run(['g++','-std=c++17','-O1','-shared','-fPIC','-pthread',f'-I{path}',
                    str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl','-o',str(path/'libquactlize_kpack_dispatch.so')],check=True)
    d=Dispatch(path);yield d;d.close()


def test_typed_c_abi_can_pick_measured_simt_without_gpu_or_jit(host_dispatch,policy):
    picks=[]
    for compute in (0,1):
        r=next(r for r in policy['exact'] if r['key'][0]==8 and r['key'][1]==0 and r['key'][7:]==[1,compute] and r['config']['kind']=='simt')
        q,mode,n,k,e,top,ch,m,_=r['key']
        c=Call(version=1,size=C.sizeof(Call),qtype=q,mode=mode,n=n,k=k,experts=e,topk=top,
            channels=ch,rows=m,input_type=1,a_row_stride=k,a_token_stride=k,ids_stride=top,out_row_stride=n)
        got=host_dispatch.query_smallm_matched(c,arrangement(q),compute)
        assert got and got.compute_type==compute and got.base.policy==12 and got.base.kind==1
        picks.append(got.base.simt.split)
    assert len(picks)==2


def test_matched_tc_keeps_compute_recipe_and_ticket(tmp_path):
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char kJitSource[]="";\n')
    binary=tmp_path/'tc'
    subprocess.run(['g++','-std=c++17','-O1','-pthread',f'-I{ROOT}',f'-I{tmp_path}',
        str(ROOT/'tests/smallm_matched_dispatch_host.cpp'),'-ldl','-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'exact compute/geometry/split/grid/ticket preserved' in result.stdout


@pytest.mark.parametrize('compute',(0,1))
@pytest.mark.parametrize('kind',(0,1,2))
def test_box_chain_gate_keeps_matched_recipe_before_legacy(compute,kind):
    from tools.run_kpack_moe_gate import automatic_smallm
    chosen=SimpleNamespace(base=SimpleNamespace(kind=kind,tc='exact-ticket'),q4='exact-q4')
    class Fake:
        def query_smallm_matched(self,call,arr,c):
            assert c==compute
            return chosen
        def query_smallm(self,*args):raise AssertionError('matched winner reselected')
    base,q4=automatic_smallm(Fake(),SimpleNamespace(qtype=12),None,compute)
    assert base is chosen.base and q4==('exact-q4' if kind==2 else None)


def test_box_chain_gate_only_falls_back_after_matched_miss():
    from tools.run_kpack_moe_gate import automatic_smallm
    calls=[]
    class Fake:
        def query_smallm_matched(self,*args):return None
        def query_smallm(self,call,arr,c):calls.append(c);return 'legacy'
    for c in (0,1):assert automatic_smallm(Fake(),SimpleNamespace(qtype=8),None,c)==('legacy',None)
    assert calls==[None,1]
    assert automatic_smallm(Fake(),SimpleNamespace(qtype=12),None,1)==(None,None)

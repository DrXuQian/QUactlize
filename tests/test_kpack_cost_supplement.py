import collections
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tools.kpack_cost_supplement import plan, domain, dequant_key, validate_point, configs
from tools.build_kpack_cost_supplement import make_plan
from tools.kpack_cost_evidence import EVIDENCE, reuse
from tools.run_kpack_cost_supplement import timing_valid, stable, perform, validate, summarize, keys
from tools.kpack_prefill_measurement import Weights
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def planned():return plan()


def test_exact_missing_denominator(planned):
    assert len(planned['points'])==458
    assert collections.Counter(p['phase'] for p in planned['points'])=={
        'dense-mid':30,'dense-rest':65,'formats':255,'grouped-mid':36,'sparse':72}
    assert len({p['id'] for p in planned['points']})==458
    assert all(p['tokens']>=128 for p in planned['points'])
    assert not planned['small_m_full_dequant']
    for p in planned['points']:validate_point(p)


def test_native_selection_closure_and_json_resume(tmp_path):
    p=make_plan(tmp_path)
    assert len(p['parents'])==134
    assert all(r['status']=='SELECTED' for r in p['selected_requests'])
    assert json.loads(json.dumps(p))==p
    assert digest({k:v for k,v in p.items() if k!='plan_sha256'})==p['plan_sha256']
    assert make_plan(tmp_path)==p


@pytest.mark.parametrize('tokens',[128,512,1024])
@pytest.mark.parametrize('count',[32,64])
def test_sparse_domain_is_real_noncontiguous(tokens,count):
    rows,idx,rh=domain(tokens,256,f'active{count}')
    assert sum(rows)==tokens*8 and len(idx)==tokens*8
    assert np.count_nonzero(rows)==count and max(rows)<=tokens
    active=np.flatnonzero(rows)
    assert not np.array_equal(active,np.arange(count))
    assert np.array_equal(np.bincount(idx,minlength=256),rows)
    assert len(rh)==64


def test_dequant_deduplicates_m_but_never_active_sets(planned):
    ps=[p for p in planned['points'] if p['weight_id']=='q12-n512-k2048-e256']
    assert len({dequant_key(p,0) for p in ps})==1
    assert len({dequant_key(p,1) for p in ps if p['profile']=='active32'})==1
    assert dequant_key(next(p for p in ps if p['profile']=='active32'),1)!=dequant_key(next(p for p in ps if p['profile']=='active64'),1)


@pytest.mark.parametrize('field,value',[('tokens',8),('rows',[1]*256),('active_ids',[0]),('routes_sha256','wrong'),('full_indexed',False)])
def test_changed_plan_rejected(planned,field,value):
    p=copy.deepcopy(next(p for p in planned['points'] if p['profile']=='active32'))
    p[field]=value
    with pytest.raises(ValueError):validate_point(p)


@pytest.mark.parametrize('q',range(10,15))
def test_retained_bf16_matches_existing_weights(q):
    w=dict(q=q,n=256,k=512,experts=1)
    a=Weights(w);b=Weights(w,keep_bf16=True)
    assert a.gold is None and b.gold.shape==(1,256,512)
    assert a.identity==b.identity
    for name in a.planes:np.testing.assert_array_equal(a.planes[name],b.planes[name])


def evidence_args(planned):
    e=json.loads(EVIDENCE.read_text());r=e['entries'][0]
    p=next(p for p in planned['points'] if p['weight_id']==r['weight_id'])
    identity={'fixture_hashes':r['fixture_hashes'],
        'sf_golden_sha256':r['golden_sha256'],'bf16_golden_sha256':'not-sf'}
    return e,p,r,identity


def test_reuse_exact(planned):
    e,p,r,f=evidence_args(planned)
    x=reuse(e,p,0,f,r['device'],r['runtime'],r['python_packages'])
    assert x['measurement']=='REUSED_AUDITED_MEASUREMENT'
    assert x['rows'][0]['samples_us']==r['best']['samples_us']
    assert x['source']==r['source']


@pytest.mark.parametrize('field',['fixture','device','runtime','packages'])
def test_reuse_rejects_context_mix(planned,field):
    e,p,r,f=evidence_args(planned)
    args=[f,r['device'],r['runtime'],r['python_packages']]
    idx=['fixture','device','runtime','packages'].index(field)
    args[idx]=copy.deepcopy(args[idx]);args[idx]['plant']='changed'
    if field=='fixture':args[idx]['sf_golden_sha256']='wrong'
    with pytest.raises(ValueError):reuse(e,p,0,*args)


def test_sparse_never_scales_all_expert_cost(planned):
    e=json.loads(EVIDENCE.read_text())
    p=next(p for p in planned['points'] if p['profile']=='active32')
    assert reuse(e,p,1,{}, {}, {}, {}) is None
    assert configs(12,1,True)==[5,10,11]


def record(kind='gemm'):
    return dict(status='PASS',kind=kind,samples_us=[2.]*15,median_us=2.,round_medians_us=[2.]*3,
        scope=('SELECTED_OR_HISTORICAL_COMPLETE_GEMM_NO_DEQUANT_NO_EXTERNAL_ADAPTERS' if kind=='gemm'
               else 'ISOLATED_BF16_PROVIDER_NO_DEQUANT_NO_EXTERNAL_ADAPTERS'),
        error=0.,guards='PASS',zero_a='PASS',changed_input_graph='PASS',
        reducer_included=True,sf_prepass_timed=False,production_changed=False)


def args(tmp_path):
    for x in ('components','failures'):(tmp_path/x).mkdir()
    return SimpleNamespace(output=tmp_path,authority='fixed')


def test_resume_success_and_retry_only_failure(tmp_path):
    a=args(tmp_path)
    first=perform(a,'green',record)
    before=(tmp_path/'components/green.json').read_bytes()
    assert perform(a,'green',lambda:pytest.fail('must not run valid component'))==first
    assert perform(a,'red',lambda:(_ for _ in ()).throw(ValueError('plant'))) is None
    assert (tmp_path/'components/green.json').read_bytes()==before
    assert perform(a,'red',record)['status']=='PASS'
    assert len(list((tmp_path/'failures').glob('red.*.json')))==1


def test_tampered_success_never_silently_retimed(tmp_path):
    a=args(tmp_path);r=perform(a,'one',record);r['median_us']=3
    (tmp_path/'components/one.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='checksum'):perform(a,'one',record)


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),True])
def test_timing_negatives(value):
    r=record();r['samples_us'][0]=value
    with pytest.raises(ValueError):timing_valid(r)


def test_variance_is_advisory_not_numeric_failure():
    r=record();r['round_medians_us']=[2,2,2.12]
    assert not stable(r)


def test_component_sum_does_not_double_count_reducer(tmp_path):
    a=args(tmp_path)
    p=dict(id='p',weight_id='q12-n256-k512-e1',q=12,n=256,k=512,experts=1,active_ids=[0],
        routes={'0':{'historical':None},'1':{'historical':None}})
    for o in (0,1):
        r=dict(status='PASS',kind='dequant',scope='ISOLATED_DEQUANT_ACTUAL_EXPERT_DOMAIN',gemm_timed=False,
            rows=[dict(samples_us=[float(o+1)]*15,round_medians_us=[float(o+1)]*3,median_us=float(o+1),
                       proof={'bad':0,'signed_zero_differences':0},negative_bad=1,guard='PASS')])
        perform(a,dequant_key(p,o),lambda r=r:r)
    for route in ('0','1'):perform(a,f'p-r{route}-current',record)
    perform(a,'p-bf16',lambda:record('bf16')|dict(dequant_timed=False))
    result=summarize(tmp_path,[p],a.authority)
    assert result['costs'][0]['cost_us']=={'r0-current':2.,'r1-current':3.,'full-bf16':4.}
    assert result['missing']==['dense-m1-n4096-s4-recheck']
    assert not result['production_changed']


def test_header_host_ownership_and_error_contract(tmp_path):
    source=tmp_path/'test.cpp';out=tmp_path/'test'
    source.write_text('''
#define __device__
#define __forceinline__ inline
struct { int x=0; } threadIdx;
void atomicExch(int* p,int v) { *p=v; }
#include "quactlize/dequant/expert_selection.cuh"
using namespace quactlize::dequant;
int main() {
 int ids[8]={7,2,5},count=3,error=0;
 ExpertSelection s{ids,&count,&error,8};
 if(select_expert<true>(0,s)!=7 || select_expert<true>(1,s)!=2) return 1;
 if(select_expert<true>(3,s)!=-1 || error) return 2;
 count=0;if(select_expert<true>(0,s)!=-1 || error) return 3;
 count=9;if(select_expert<true>(0,s)!=-1 || error!=1) return 4;
 count=1;ids[0]=8;error=0;
 if(select_expert<true>(0,s)!=-1 || error!=2) return 5;
 if(select_expert<false>(6,{})!=6) return 6;
}
''')
    subprocess.run(['g++','-std=c++17',f'-I{ROOT}',str(source),'-o',str(out)],check=True)
    subprocess.run([str(out)],check=True)


def test_shell_failure_does_not_exit_caller():
    script=ROOT/'tools/run_kpack_cost_supplement_ppu_box.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    r=subprocess.run(['bash','-c','bash "$1" invalid-phase; printf "PARENT_ALIVE\\n"','bash',str(script)],capture_output=True,text=True)
    assert r.returncode==0 and 'PARENT_ALIVE' in r.stdout

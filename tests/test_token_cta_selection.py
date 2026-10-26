"""Exact Asys winners, actual dispatch, and stale-package rejection; no GPU timing."""
import copy
import io
import json
import subprocess
import sys
import tarfile

import pytest

from quactlize.dispatch.planning import plan_smallm, validate_inventory
from quactlize.execution import token_cta_policy as binding
from quactlize.execution.simt_codegen import runtime_inventory
from quactlize.runtime.tuning import digest
from tools import generate_decode_selection as gen

ROOT = gen.ROOT


@pytest.fixture(scope='module')
def table():
    return binding.load()


@pytest.fixture(scope='module')
def plan(tmp_path_factory, table):
    return plan_smallm(tmp_path_factory.mktemp('tokens-selected'),[r['key'] for r in table['rows']])


def test_exact_means_and_bodies_are_bound_without_changing_buckets(table):
    old=gen.effective(json.loads(gen.EVIDENCE.read_text()),include_token_cta=False)
    new=gen.effective(json.loads(gen.EVIDENCE.read_text()))
    assert digest(old)==table['baseline_sha256']
    assert len(table['rows'])==40 and sum(r['changed'] for r in table['rows'])==35
    assert len(binding.rows())==32
    changed={tuple(r['key']) for r in table['rows'] if r['changed']}
    for a,b in zip(old['exact'],new['exact']):
        if tuple(a['key']) not in changed:assert a==b
    assert {k:v for k,v in old.items() if k!='exact'}=={k:v for k,v in new.items() if k!='exact'}
    before,after=gen.header(old),gen.header(new)
    for array in ('kBucketChoices','kBucketIndex','kBucketDequant'):
        assert before.split(array+'[]={',1)[1].split('};',1)[0]==after.split(array+'[]={',1)[1].split('};',1)[0]
    for row in table['rows']:
        assert row['selected']['mean_us']==min(row['candidate_means_us'].values())
        assert row['latency_provenance']['quactlize_commit']==table['source_revision']['quactlize_commit']
    assert binding.CATALOG.read_text()==binding.catalog()
    assert len(binding.capability()['keys'])==32


def test_actual_selected_recipes_and_workspace_match_the_measured_calls(table,plan):
    tc=[]
    for source,actual in zip(table['rows'],plan['requests']):
        chosen=source['selected'];f=chosen['config'];impl=chosen['implementation']
        q,mode,n,k,e,top,ch,m,compute=source['key']
        assert actual['request']==source['key'] and actual['status']==0
        assert actual['donor']==[n,k,m]
        if f['kind']=='tc':
            tc.append((q,m));assert actual['kind']==0
            assert actual['parent']['symbol']==f['symbol'] and actual['split']==f['split']
        else:
            assert actual['kind']==(1 if f['kind']=='simt' else 2)
            assert actual['config']=={k:v for k,v in f.items() if k!='kind'}
        if impl['kind']=='token-cta':
            assert actual['implementation']['producer']=='token-cta'
            assert actual['implementation']['cta_tokens']==m
            assert actual['implementation']['cta_window']==impl['window']
            assert actual['workspace_bytes']==m*f['split']*n*4
        else:assert not actual.get('implementation',{}).get('cta_tokens')
    assert tc==[(8,8),(12,8)]


def test_future_sweeps_preserve_the_actual_incumbent_body(plan):
    from dev.strategy.decode_plan import original_recipe
    from dev.strategy.vector import spec,source
    from quactlize.dispatch.table import Workload
    for row in plan['requests']:
        if not row.get('implementation',{}).get('cta_tokens'):continue
        q,mode,n,k,e,top,ch,m,compute=row['request']
        item=spec(Workload(q,n,k,m),original_recipe(row))
        assert item['cta_tokens']==m
        if q==8:assert item['cta_window']==1
        assert 'token_cta_launch<' in source(item)


def test_old_execution_with_same_geometry_cannot_pass_readiness(plan):
    p=dict(plan,requests=[r for r in plan['requests'] if r.get('implementation',{}).get('cta_tokens')])
    execution=dict(simt_configs={str(q):[r.record() for r in runtime_inventory(q)] for q in (8,10,11,12,13)},
                   token_cta=binding.capability())
    validate_inventory(p,execution,jit=True)
    for field in ('catalog_sha256','body_hashes','keys','evidence_sha256'):
        wrong=copy.deepcopy(execution);wrong['token_cta'][field]=None
        with pytest.raises(ValueError,match='token CTA'):validate_inventory(p,wrong,jit=True)
    execution.pop('token_cta')
    with pytest.raises(ValueError,match='token CTA'):validate_inventory(p,execution,jit=True)


def test_scope_rejects_unmeasured_dtype_storage_operator_and_shape(tmp_path):
    source=tmp_path/'scope.cpp'
    source.write_text(r'''
#include "quactlize/execution/simt_token_cta_scope.hpp"
#include <cassert>
#include <initializer_list>
using namespace quactlize::execution::simt;
int main() {
int count=0;
#define QTC(Q,N,K,M,V,C,W,P,S,Window) { \
qkg_call_v1 c{1,sizeof(c)};c.qtype=Q;c.n=N;c.k=K;c.rows=M;c.mode=QKG_DENSE; \
c.input_type=QKG_F32;c.experts=c.topk=c.channels=1; \
qkg_simt_call_v2 d{2,sizeof(d),c,QKG_COMPUTE_F16}; \
qkg_simt_config_v1 f{1,sizeof(f),V,C,W,P,S}; \
assert(token_cta_window(d,f)==Window);++count; \
for(int mode:{QKG_GROUPED,QKG_INDEXED}) {auto bad=d;bad.call.mode=mode;assert(token_cta_window(bad,f)<0);} \
for(int storage:{int(QKG_F16),int(QKG_SIMT_BF16)}) {auto bad=d;bad.call.input_type=storage;assert(token_cta_window(bad,f)<0);} \
for(int rows:{0,1,9}) {auto bad=d;bad.call.rows=rows;assert(token_cta_window(bad,f)<0);} \
auto bad=d;bad.compute_type=QKG_COMPUTE_BF16;assert(token_cta_window(bad,f)<0); \
bad=d;bad.call.n+=256;assert(token_cta_window(bad,f)<0); \
bad=d;bad.call.k+=512;assert(token_cta_window(bad,f)<0); \
bad=d;bad.call.experts=256;assert(token_cta_window(bad,f)<0); \
bad=d;bad.call.topk=8;assert(token_cta_window(bad,f)<0); \
bad=d;bad.call.channels=8;assert(token_cta_window(bad,f)<0); \
bad=d;bad.call.qtype=14;assert(token_cta_window(bad,f)<0); \
f.split=16;assert(token_cta_window(d,f)<0); \
}
#include "quactlize/execution/token_cta.inc"
#undef QTC
assert(count==32);
}
''')
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),source,'-o',tmp_path/'scope'],check=True)
    subprocess.run([tmp_path/'scope'],check=True)


def test_old_cpp_all_keys_and_nearby_bucket_choices_are_preserved(tmp_path,table):
    base=tmp_path/'base';base.mkdir()
    data=subprocess.check_output(['git','archive','c6bbb61851a3150f02baf977edf6d65b3f01f702',
                                  'quactlize','policies','tools/kpack_selection.cpp'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as stream:
        members=stream.getmembers()
        assert all((m.isfile() or m.isdir()) and '..' not in m.name.split('/') and
                   not m.name.startswith('/') for m in members)
        stream.extractall(base,members=members,filter='data')
    old=gen.effective(json.loads(gen.EVIDENCE.read_text()),include_token_cta=False)
    keys={tuple(r['key']) for r in old['exact']}
    for r in table['rows']:
        for axis in (2,3):
            for delta in (-512,-256,256,512):
                k=list(r['key']);k[axis]+=delta;keys.add(tuple(k))
        keys.add(tuple(r['key'][:-1]+[1]))
    ordered=sorted(keys);text=''.join(' '.join(map(str,k))+'\n' for k in ordered)
    results=[]
    for root in (base,ROOT):
        exe=tmp_path/('old' if root==base else 'new')
        subprocess.run(['g++','-std=c++17','-O2','-I'+str(root),root/'tools/kpack_selection.cpp','-o',exe],check=True)
        results.append([json.loads(s) for s in subprocess.check_output([exe],input=text,text=True).splitlines()])
    changes={tuple(r['key']) for r in table['rows'] if r['changed']}
    observed=set()
    for k,a,b in zip(ordered,*results):
        if a!=b:observed.add(k)
        if k not in changes:assert a==b,(k,a,b)
    assert observed==changes


@pytest.mark.parametrize('field',['shape','config','mean','provenance'])
def test_resealed_edited_evidence_is_not_admitted(table,field):
    bad=copy.deepcopy(table);row=bad['rows'][0]
    if field=='shape':row['key'][2]=2048
    elif field=='config':row['selected']['config']['split']=16
    elif field=='mean':row['selected']['mean_us']*=.5
    else:row['latency_provenance']['quactlize_commit']='0'*40
    bad['evidence_sha256']=digest({k:v for k,v in bad.items() if k!='evidence_sha256'})
    with pytest.raises(ValueError):binding.verify(bad)


def test_regeneration_and_catalog_are_current():
    subprocess.run([sys.executable,ROOT/'tools/generate_token_cta.py','--check'],check=True)
    subprocess.run([sys.executable,ROOT/'tools/generate_decode_selection.py','--check'],check=True)

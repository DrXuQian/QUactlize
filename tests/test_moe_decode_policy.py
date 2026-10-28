"""Evidence-bound exact Q5 admission; no host test claims device performance."""
import copy
import ctypes as C
import json
import shutil
import subprocess

import pytest

from quactlize.execution import moe_decode_policy as binding
from quactlize.runtime.tuning import digest
from tools import generate_decode_selection as gen

ROOT=gen.ROOT
CHANGED={(13,2,2048,512,256,8,8,m,1) for m in (1,8)}


def test_only_two_exact_q5_keys_change_and_no_bucket_donor():
    evidence=json.loads(gen.EVIDENCE.read_bytes())
    before=gen.effective(evidence,include_moe=False);after=gen.effective(evidence)
    assert len(before['exact'])==len(after['exact'])==2095
    actual=set()
    for old,new in zip(before['exact'],after['exact']):
        assert old['key']==new['key']
        if old!=new:actual.add(tuple(old['key']))
        donor=lambda r:r.get('bucket_config',r['guard'] if r['config']['kind']=='q4' else r['config'])
        assert donor(old)==donor(new)
    assert actual==CHANGED
    assert {k:v for k,v in before.items() if k!='exact'}=={k:v for k,v in after.items() if k!='exact'}
    assert gen.header(after)==gen.OUTPUT.read_text()
    data=binding.load()
    assert len(data['rows'])==3 and sum(r['changed'] for r in data['rows'])==2
    assert not data['runtime_router_readback'] and not data['model_tpot_admitted']
    for row in data['rows']:
        for profile in row['profiles']:
            assert profile['selected']['mean_us']<=profile['original']['mean_us']
        assert 'moe_pair_rows' not in row['config']
    assert not next(r for r in data['rows'] if r['key'][7]==4)['changed']


@pytest.mark.parametrize('field',['shape','config','mean','provenance','body'])
def test_edited_or_resealed_evidence_is_rejected(field):
    data=binding.load();bad=copy.deepcopy(data);row=bad['rows'][0]
    if field=='shape':row['key'][7]=2
    elif field=='config':row['config']['columns']=4
    elif field=='mean':row['profiles'][0]['selected']['mean_us']*=.5
    elif field=='provenance':bad['source_revision']['quactlize_commit']='0'*40
    else:bad['body_hashes']['quactlize/execution/simt_common.cuh']='0'*64
    bad['evidence_sha256']=digest({k:v for k,v in bad.items() if k!='evidence_sha256'})
    with pytest.raises(ValueError):binding.verify(bad)


def test_changed_incumbent_cannot_silently_rebase_the_measurement():
    old=gen.effective(json.loads(gen.EVIDENCE.read_bytes()),include_moe=False)
    next(r for r in old['exact'] if tuple(r['key']) in CHANGED)['config']['split']=4
    with pytest.raises(ValueError,match='incumbent'):binding.apply(old)


def test_actual_cpp_catalog_transition_and_unseen_m_shape_guards(tmp_path):
    base=tmp_path/'base';base.mkdir()
    old=gen.effective(json.loads(gen.EVIDENCE.read_bytes()),include_moe=False)
    # Product CI also runs from shallow checkouts/source archives. Recreate
    # the pre-admission catalog from its tracked evidence, without fetching
    # history or depending on an old commit object being present.
    shutil.copytree(ROOT/'quactlize',base/'quactlize',ignore=shutil.ignore_patterns('__pycache__'))
    (base/'policies').mkdir()
    for header in (ROOT/'policies').glob('*.hpp'):
        shutil.copyfile(header,base/'policies'/header.name)
    (base/'policies'/gen.OUTPUT.name).write_text(gen.header(old))
    keys={tuple(r['key']) for r in old['exact']}
    for key in CHANGED:
        for axis,values in [(0,[8,10,11,12,14]),(1,[0,1]),(2,[1024,1792,2304,4096]),
                            (3,[256,1024]),(4,[128,512]),(5,[4,16]),(6,[1]),(7,range(1,10)),(8,[0])]:
            for value in values:
                modified=list(key);modified[axis]=value;keys.add(tuple(modified))
    keys=sorted(keys);stdin=''.join(' '.join(map(str,key))+'\n' for key in keys);results=[]
    for i,root in enumerate((base,ROOT)):
        exe=tmp_path/str(i)
        subprocess.run(['g++','-std=c++17','-O2','-I'+str(root),ROOT/'tools/kpack_selection.cpp','-o',exe],check=True)
        results.append([json.loads(s) for s in subprocess.check_output([exe],input=stdin,text=True).splitlines()])
    changed={key for key,a,b in zip(keys,*results) if a!=b}
    assert changed==CHANGED
    for row in results[1]:
        if tuple(row['request']) not in CHANGED:continue
        assert row['status']==0 and row['kind']==1 and row['policy']==18 and row['workspace_bytes']==0
        assert row['implementation']['producer']=='common-packet-3d'
        assert row['implementation']['reduction']=='none'
        assert (row['config']['columns'],row['config']['warps'],row['config']['values'])==(
            (8,4,4) if row['request'][7]==1 else (4,2,8))


def test_public_c_abi_selects_both_new_recipes_and_rejects_other_storage(tmp_path):
    from quactlize.dispatch.native import Dispatch
    from quactlize.execution.native import Call,arrangement
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char const kJitSource[]="";\n')
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-shared','-fPIC','-Wl,-Bsymbolic',
        '-I'+str(tmp_path),ROOT/'quactlize/dispatch/binding.cpp','-ldl',
        '-o',tmp_path/'libquactlize_kpack_dispatch.so'],check=True)
    d=Dispatch(tmp_path)
    try:
        for m in (1,8):
            c=Call(version=1,size=C.sizeof(Call),qtype=13,mode=2,n=2048,k=512,experts=256,
                   topk=8,channels=8,rows=m*8,input_type=1,a_row_stride=512,
                   a_token_stride=4096,ids_stride=8,out_row_stride=2048)
            choice=d.query_smallm_matched(c,arrangement(13),1)
            assert choice and choice.base.kind==1 and choice.base.policy==18
            assert (choice.base.simt.columns,choice.base.simt.warps,choice.base.simt.values)==(
                (8,4,4) if m==1 else (4,2,8))
            c.input_type=2
            assert d.query_smallm_matched(c,arrangement(13),1) is None
    finally:d.close()

"""Exact-table admission and real host dispatch; no new GPU timing claims."""
import copy
import json
import subprocess
import sys

import pytest

from dev.strategy import dense_m_selection as admission
from dev.strategy.m_coverage import points, coverage
from dev.strategy.refresh import request
from quactlize.dispatch.planning import plan_smallm, validate_inventory
from quactlize.dispatch.table import Workload
from quactlize.execution.q8_refresh import capability, rows as q8_rows
from quactlize.execution.simt_codegen import runtime_inventory
from quactlize.runtime.tuning import digest
from tools import generate_decode_selection as gen

ROOT = gen.ROOT


@pytest.fixture(scope='module')
def table():
    return admission.load()


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    return plan_smallm(tmp_path_factory.mktemp('dense-m-exact-query'),
        [request(Workload(**p['workload'])) for p in points()], dense_fallbacks=True)


def test_complete_source_scope_and_unstable_rows_remain_unpromoted(table):
    assert table['summary'] == dict(points=243, admitted=237, changed=160, retained=77, pending=6)
    assert not any(table[k] for k in ('bucket_changed','big_m_changed','native_comparison','model_tpot_admitted'))
    pending=[r for r in table['rows'] if not r['admitted']]
    assert len(pending)==6 and all(r['selected']==r['original'] for r in pending)
    # Keep the stable 26% contender as pending; do not relax the spread gate.
    row=next(r for r in pending if r['key']==[12,0,10240,5120,1,1,1,1,0])
    assert row['selected']['config']['legacy_adapters']
    sf=[r for r in table['rows'] if r['admitted'] and r['selected']['dequant_config']>=0]
    assert len(sf)==18 and all(r['key'][0]==13 and r['selected']['dequant_config']==4 for r in sf)


def test_only_new_exact_keys_and_no_bucket_or_old_row_changes(table):
    old=gen.effective(json.loads(gen.EVIDENCE.read_text()),include_dense=False)
    new=gen.effective(json.loads(gen.EVIDENCE.read_text()))
    assert digest(old)==table['baseline_sha256']
    assert len(old['exact'])==1852 and len(new['exact'])==2089
    assert old['exact']==new['exact'][:len(old['exact'])]
    assert {k:v for k,v in old.items() if k!='exact'}=={k:v for k,v in new.items() if k!='exact'}
    admitted={tuple(r['key']):r for r in table['rows'] if r['admitted']}
    for row in new['exact'][len(old['exact']):]:
        picked=admitted[tuple(row['key'])]['selected']
        assert row['config']==picked['config'] and row['implementation']==picked['implementation']
        assert row['dequant_config']==picked['dequant_config'] and row['policy']==18
    # Old bucket indices still reference the same semantic choices.
    for b in old['buckets']:
        assert old['exact'][b['row']]==new['exact'][b['row']]
    # Appending choices must not renumber the old generated table either.
    before,after=gen.header(old),gen.header(new)
    for name in ('kChoices','kExact','kGuards','kBucketChoices'):
        a=before.split(name+'[]={\n',1)[1].split('\n};',1)[0]
        b=after.split(name+'[]={\n',1)[1].split('\n};',1)[0]
        assert b.startswith(a+'\n') or a==b


def test_real_cpp_selects_every_admitted_recipe_and_preserves_six_pending(table,plan):
    got={tuple(r['request']):r for r in plan['requests']}
    for row in table['rows']:
        actual=got[tuple(row['key'])]
        if not row['admitted']:
            assert actual==row['original_cpp']
            continue
        c=row['selected'];f=c['config'];q,mode,n,k,e,top,ch,m,compute=row['key']
        assert actual['status']==0 and actual['policy']==18 and actual['donor']==[n,k,m]
        assert actual['dequant_config']==c['dequant_config']
        if f['kind']=='tc':
            assert actual['kind']==0
            for field in ('symbol','qtype','route','tm','tn','tk','wm','wn','stages','ap','dn'):
                assert actual['parent'][field]==f[field]
            assert actual['parent']['persistent']==f['parent_persistent']
            for field in ('split','grid_mode','grid_b'):
                assert actual[field]==f[field]
        else:
            assert actual['kind']==1
            assert actual['config']=={k:v for k,v in f.items() if k!='kind'}
            assert actual['implementation'].get('q8_body',0)==c['implementation'].get('body',0)
    audit=coverage(plan['requests'])
    assert (audit['expected'],audit['exact'],audit['missing'])==(360,354,6)


def test_q8_body_and_execution_capability_follow_the_new_table(table,plan):
    extra=[r for r in table['rows'] if r['admitted'] and r['selected']['implementation']['kind']=='q8-address']
    assert len(extra)==36 and len(q8_rows())==63
    advertised={tuple(r['key']):r for r in capability()['keys']}
    for row in extra:
        c=row['selected'];binding=advertised[tuple(row['key'])]
        assert binding['body']==c['implementation']['body']
        assert binding['config']=={k:v for k,v in c['config'].items() if k!='kind'}
    execution=dict(simt_configs={str(q):[c.record() for c in runtime_inventory(q)] for q in (8,11,12,13,14)},
                   q8_refresh=capability())
    # The complete 360 keys include unchanged shape-specific Q4 legacy calls;
    # here validate the freshly admitted generic SIMT/typed-TC scope only.
    admitted={tuple(r['key']) for r in table['rows'] if r['admitted']}
    selected=dict(plan,requests=[r for r in plan['requests'] if tuple(r['request']) in admitted])
    validate_inventory(selected,execution,jit=True)
    execution['q8_refresh']['keys']=execution['q8_refresh']['keys'][:27]
    with pytest.raises(ValueError,match='Q8'):
        validate_inventory(selected,execution,jit=True)


@pytest.mark.parametrize('fault',['key','config','body','sample','scope','unresolved'])
def test_resealing_edited_evidence_does_not_admit_it(table,fault):
    bad=copy.deepcopy(table);row=bad['rows'][0]
    if fault=='key':row['key'][7]=9
    elif fault=='config':row['selected']['config']['split']=16
    elif fault=='body':row['selected']['implementation']=dict(kind='q8-address',body=99)
    elif fault=='sample':row['selected']['median_us']*=.5
    elif fault=='scope':bad['bucket_changed']=True
    else:next(r for r in bad['rows'] if not r['admitted'])['admitted']=True
    bad['evidence_sha256']=digest({k:v for k,v in bad.items() if k!='evidence_sha256'})
    with pytest.raises(ValueError):admission.verify(bad)


def test_no_override_when_the_incumbent_catalog_has_changed():
    old=gen.effective(json.loads(gen.EVIDENCE.read_text()),include_dense=False)
    old['exact'][0]['policy']=123
    with pytest.raises(ValueError,match='incumbent'):admission.apply(old)


def test_generated_table_and_bindings_are_reproducible():
    subprocess.run([sys.executable,str(ROOT/'tools/generate_decode_selection.py'),'--check'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/generate_q8_refresh.py'),'--check'],cwd=ROOT,check=True)

"""Raw-backed policy and real host wiring, never a simulated performance pass."""
import copy
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from dev.strategy.prefill_refresh import recipe_of, request, same_recipe, make_plan
from quactlize.dispatch.planning import plan_prefill, plan_smallm, prewarm_groups
from quactlize.dispatch.prefill_policy import EVIDENCE, HEADER, LEGACY, STEM, check_receipt, receipt_fields
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest
from tools.fit_prefill_refresh import header, validate

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'ceccd4494e6bd9cefaa1ab6a9fde9baab30300a2'


def evidence():
    return json.loads(EVIDENCE.read_text())


def test_all_decisions_replay_unchanged_gate_and_keep_pending():
    data=evidence();before=copy.deepcopy(data)
    validate(data)
    assert data==before and header(data)==HEADER.read_text()
    assert sum(p['changed'] for p in data['points'])==94
    pending=[p for p in data['points'] if not p['timing_confirmed']]
    assert len(pending)==6 and all(not p['changed'] for p in pending)
    assert all(p['cost_basis']=='LEGACY_CONFIRMED_UNCHANGED_RECIPE' for p in pending)
    assert sum(p['recipe']['kind']=='full' for p in pending)==5
    assert not data['model_performance_admitted']


@pytest.mark.parametrize('fault', ['seal','recipe','numerics','components','pending','qualified','cost','grid','resource'])
def test_evidence_mutation_fails_closed(fault):
    data=evidence();p=data['points'][0]
    if fault=='recipe':p['recipe']['parameters'][0][1]+=1
    elif fault=='numerics':p['accuracy']['nonfinite']=1
    elif fault=='components':p['component_rounds'][0].pop()
    elif fault=='pending':next(p for p in data['points'] if not p['timing_confirmed'])['changed']=True
    elif fault=='qualified':p['arms'][p['selected']]['qualified']=False
    elif fault=='cost':next(p for p in data['points'] if not p['timing_confirmed'])['gemm_us']+=1
    elif fault=='grid':next(p for p in data['points'] if p['selected_launch'])['selected_launch']['grid']+=1
    elif fault=='resource':next(p for p in data['points'] if p['selected_resource'])['selected_resource']['occupancy']=0
    if fault!='seal':data['sha256']=digest({k:v for k,v in data.items() if k!='sha256'})
    else:data['sha256']='0'*64
    with pytest.raises(ValueError):validate(data)


def test_live_cpp_matches_every_measured_recipe(tmp_path):
    live=plan_prefill(tmp_path)
    rows={tuple(r['request']):r for r in live['requests']}
    assert len(rows)==430
    for p in evidence()['points']:
        row=rows[request(Workload(**p['workload']))]
        selected=recipe_of(row)
        assert same_recipe(selected,Recipe.from_dict(p['recipe']))
        assert selected==Recipe.from_dict(p['recipe'])  # symbols too, not just geometry
    groups=prewarm_groups(live)
    assert {(c,d) for c,d,_ in groups}=={('f16',False),('bf16',False)}
    # Original finalist scripts must reject a current selector as a historical
    # control; their frozen test fixture is explicit and never used at runtime.
    with pytest.raises(ValueError,match='incumbent differs'):
        make_plan(live,['1','2','3','4','5','6','7'])
    historical=plan_prefill(tmp_path,historical=True)
    old={tuple(r['request']):r for r in historical['requests']}
    assert sum(old[k]!=v for k,v in rows.items())==94
    make_plan(historical,['1','2','3','4','5','6','7'])


def test_smallm_unchanged_except_declared_later_dense_exact_additions(tmp_path):
    frozen=tmp_path/'baseline';frozen.mkdir()
    raw=subprocess.check_output(['git','archive',BASELINE,'quactlize','policies','tools/kpack_selection.cpp'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        assert all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                   for m in tar.getmembers())
        tar.extractall(frozen,filter='data')
    exe=tmp_path/'before'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(frozen),frozen/'tools/kpack_selection.cpp','-o',exe],check=True)
    before=[json.loads(line) for line in subprocess.check_output([exe,'--inventory'],text=True).splitlines()]
    after=plan_smallm(tmp_path)['requests']
    from quactlize.execution.token_cta_policy import load as token_table
    tokens={tuple(r['key']) for r in token_table()['rows'] if r['changed']}
    assert len(before)==1852 and len(after)==2089
    for a,b in zip(before,after):
        if tuple(a['request']) not in tokens:assert a==b
    # Neighbor shapes/precision/channels/invalid inputs exercise actual bucket
    # and rejection decisions, not only the exact measured keys.
    from test_selection_refactor import case_keys
    from dev.strategy.dense_m_selection import load
    admitted={tuple(r['key']) for r in load()['rows'] if r['admitted']}
    keys=sorted({k[:9] for k in case_keys()}-admitted-tokens)
    text=''.join(' '.join(map(str,k))+'\n' for k in keys)
    old=subprocess.check_output([exe],input=text,text=True)
    new=subprocess.check_output([tmp_path/'selection-query'],input=text,text=True)
    assert old==new
    print(f'REFRESH_SMALLM_UNCHANGED existing_exact=1852 neighbors_and_invalid={len(keys)} declared_new_exact=237')


def expected_grid(p, w, occupancy=6, cu=72):
    if p['grid_mode']<2:return 0
    ceil=lambda a,b:(a+b-1)//b
    rows=w['m']*w['topk']
    mt=ceil(rows,p['tm']) if w['operation']=='dense' else min(
        w['experts']*ceil(w['m'],p['tm']),(rows+min(rows,w['experts'])*(p['tm']-1))//p['tm'])
    tiles=mt*ceil(w['n'],p['tn'])*(p['split'] if w['operation']=='indexed' else 1)
    cap=cu*max(1,min(p['grid_b'],occupancy))
    return ceil(tiles,ceil(tiles,cap)) if p['grid_mode']==3 else min(tiles,cap)


def check_materialized(text):
    tickets=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('REFRESH_TICKET ')]
    dense=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('REFRESH_DENSE ')]
    expected={request(Workload(**p['workload'])):p for p in evidence()['points']}
    assert len(tickets)==len({tuple(r['request']) for r in tickets})==257
    assert len(dense)==len({tuple(r['request']) for r in dense})==280
    for r in tickets:
        item=expected[tuple(r['request'])];p=dict(item['recipe']['parameters']);w=item['workload']
        assert r['parent']==p['symbol'] and r['split']==p['split']
        assert r['compute']==int(w['compute']=='bf16') and r['endpoint']==0
        assert r['grid']==expected_grid(p,w) and r['algorithm']==int(p['grid_mode']>=2)
    for r in dense:
        item=expected[tuple(r['request'])];p=dict(item['recipe']['parameters']);kind=item['recipe']['route']
        assert r['route']=={'fq':0,'sf':1,'full':2}[kind] and r['dequant']==p['dequant_config']
        if kind=='full':assert r['compute']==1 and not r['parent']
        else:
            assert r['parent']==p['symbol'] and r['split']==p['split']
            assert r['compute']==0 and r['native_f16'] and r['grid']==expected_grid(p,item['workload'])


def test_materializer_uses_measured_resources_not_a_mock_grid(tmp_path):
    data=evidence()
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char kJitSource[]="";\n')
    exe=tmp_path/'tickets'
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT),'-I'+str(tmp_path),
                    ROOT/'tests/prefill_ticket_host.cpp','-ldl','-o',exe],check=True)
    lines=[]
    for p in data['points']:
        r=p['selected_resource'] or dict(occupancy=0,workspace_bytes=0,shared_bytes=0)
        lines.append((*request(Workload(**p['workload'])),r['occupancy'],r['workspace_bytes'],r['shared_bytes']))
    text=''.join(' '.join(map(str,line))+'\n' for line in lines)
    out=subprocess.check_output([exe],input=text,text=True)
    rows=[json.loads(line) for line in out.splitlines()]
    assert len(rows)==430
    for p,row in zip(data['points'],rows):
        assert row['request']==list(request(Workload(**p['workload'])))
        recipe=p['recipe'];params=dict(recipe['parameters'])
        assert row['route']=={'fq':0,'sf':1,'full':2}[recipe['route']]
        assert row['dequant']==params['dequant_config']
        if recipe['kind']=='tc':
            assert row['parent']==params['symbol']
            assert all(row[k]==p['selected_launch'][k] for k in ('split','grid','algorithm'))
            assert row['workspace']==p['selected_resource']['workspace_bytes']
            assert row['shared']==p['selected_resource']['shared_bytes']
        else:assert not row['parent']
    (tmp_path/'tickets.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('REFRESH_ACTUAL_TICKETS PASS points=430 tc=257 full=173 resources=UPLOADED_NOT_LIVE')


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('fault',[None,'source','header','path','mixed','missing'])
def test_package_binding_keeps_legacy_verification_but_rejects_aliases(legacy,fault):
    stem=LEGACY if legacy else STEM
    receipt=dict(header_sha256='header') if legacy else dict(header_sha256='header',**receipt_fields())
    hashes={stem+'.json':'evidence',stem+'.hpp':'header'}
    if fault=='source':hashes[stem+'.json']='different'
    if fault=='header':receipt['header_sha256']='different'
    if fault=='path':receipt['source_path']='../../escape.json'
    if fault=='mixed':receipt.update(source_path=STEM+'.json',header_path=LEGACY+'.hpp')
    if fault=='missing':receipt.pop('header_sha256')
    if fault:
        with pytest.raises(ValueError):check_receipt(receipt,hashes,'evidence')
    else:check_receipt(receipt,hashes,'evidence')

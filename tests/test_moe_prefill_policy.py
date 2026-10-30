"""Actual host selector/ticket wiring. No PPU performance is simulated here."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from dev.moe_confirm.plan import wired_plan
from quactlize.dispatch import moe_prefill_policy as policy
from quactlize.dispatch.planning import plan_prefill,prewarm_groups
from quactlize.runtime.tuning import digest
from quactlize.dispatch.table import Recipe,Workload
from dev.strategy.space import resolved_grid

ROOT=Path(__file__).resolve().parents[1]


def test_reviewed_scope_and_generated_header():
    data=policy.load()
    assert policy.header(data)==policy.HEADER.read_text()
    assert len(data['rows'])==40 and sum(len(r['profiles']) for r in data['rows'])==200
    assert Counter(r['recipe']['route'] for r in data['rows'])=={'fq':22,'sf':18}
    assert not data['native_no_regression_admitted'] and not data['runtime_router_readback']
    assert not data['model_tpot_admitted']


@pytest.mark.parametrize('fault',['missing','router','dtype','samples','nan','regression','accuracy','components','commit'])
def test_incomplete_or_changed_evidence_rejected(fault):
    d=deepcopy(policy.load());r=d['rows'][0];p=r['profiles'][0]
    if fault=='missing':d['rows'].pop()
    elif fault=='router':p['router']='invented'
    elif fault=='dtype':r['recipe']['route']='full'
    elif fault=='samples':p['selected_samples_us'].pop()
    elif fault=='nan':p['selected_samples_us'][0][0]=float('nan')
    elif fault=='regression':p['original_us']=p['selected_us']*.9
    elif fault=='accuracy':p['max_relative_l1']=.1
    elif fault=='components':p['components'].pop()
    elif fault=='commit':d['source_revision']['quactlize_commit']='changed'
    if fault!='nan':d['sha256']=digest({k:v for k,v in d.items() if k!='sha256'})
    with pytest.raises(ValueError):policy.validate(d)


def test_live_plan_and_required_parents(tmp_path):
    p=plan_prefill(tmp_path)
    assert len(p['requests'])==460
    rows={tuple(r['request']):r for r in p['requests']}
    for row in policy.load()['rows']:
        q,n,k,ch,m=row['key'];actual=rows[q,2,n,k,256,8,ch,m,1]
        config=policy.configuration(row['recipe'])
        assert actual['status']==0 and actual['policy']==22
        assert actual['route']==config['route']-2 and actual['dequant_config']==config['dequant_config']
        assert all(actual['parent'][key]==config[key] for key in actual['parent'])
        assert all(actual[key]==config[key] for key in ('split','grid_mode','grid_b'))
    groups=prewarm_groups(p)
    bf16=next(parents for compute,dense,parents in groups if compute=='bf16' and not dense)
    assert {dict(r['recipe']['parameters'])['symbol'] for r in policy.load()['rows']} <= {r['symbol'] for r in bf16}


def test_bf16_selection_and_scope_boundaries_in_cpp(tmp_path):
    exe=tmp_path/'moe-prefill-host'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
        '-Wno-missing-field-initializers','-I'+str(ROOT),str(ROOT/'tests/moe_prefill_policy_host.cpp'),'-o',exe],check=True)
    subprocess.run([exe],check=True)


def test_actual_ticket_materializer_preserves_all_confirmed_bf16_parents(tmp_path):
    # Synthetic resource replies test the real exported materializer. They
    # do not replace the Box query or establish occupancy/performance.
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char kJitSource[]="";\n')
    exe=tmp_path/'tickets'
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT),'-I'+str(tmp_path),
                    ROOT/'tests/prefill_ticket_host.cpp','-ldl','-o',exe],check=True)
    rows=policy.load()['rows'];lines=[]
    for row in rows:
        q,n,k,ch,m=row['key']
        lines.append((q,2,n,k,256,8,ch,m,1,6,256,4096))
    text=''.join(' '.join(map(str,line))+'\n' for line in lines)
    actual=[json.loads(line) for line in subprocess.check_output([exe],input=text,text=True).splitlines()]
    assert len(actual)==40
    for row,got in zip(rows,actual):
        q,n,k,ch,m=row['key'];recipe=Recipe.from_dict(row['recipe']);p=dict(recipe.parameters)
        w=Workload(q,n,k,m,'indexed','bf16',experts=256,topk=8,channels=ch)
        launch=resolved_grid(w,recipe,72,6)
        assert got['parent']==p['symbol'] and got['dequant']==p['dequant_config']
        assert all(got[key]==launch[key] for key in ('algorithm','split','grid'))
        assert (got['workspace'],got['shared'])==(256,4096)


def test_selected_gate_is_not_another_sweep():
    p=wired_plan()
    assert p['counts']==dict(points=200,arms=200,tc_modules=8,vector_modules=0,maximum_candidates=1)
    assert len({point['id'] for point in p['points']})==200
    for point in p['points']:
        assert len(point['arms'])==1 and point['arms'][0]['id']=='original'
        assert point['workload']['m']>8 and point['selection_source']=='PRODUCTION_PREFILL_QUERY_POLICY22'
    assert not p['historical_times_reused']


def test_frozen_decode_policy_unchanged():
    names=['policies/kpack_moe_decode_exact_v1.json','policies/kpack_smallm_effective_v1.hpp',
           'quactlize/execution/moe_decode_policy.py']
    for name in names:
        assert subprocess.check_output(['git','show',policy.COMMIT+':'+name],cwd=ROOT)==(ROOT/name).read_bytes()

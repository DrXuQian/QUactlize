"""Exact helper promotion and production JIT replay, not synthetic GPU timings."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from quactlize.dispatch.dense_prefill import (EVIDENCE, admitted_costs, bucket_costs,
    choose, execution, validate)
from quactlize.dispatch.planning import prewarm_groups, PREFILL_SCHEMA
from quactlize.runtime.compiler import Compiler, materialize_parent
from quactlize.runtime.tuning import digest, ROUTES
from tools.kpack_jit import parent_tuple

ROOT = Path(__file__).resolve().parents[1]


def test_actual_helper_admission_keeps_losses_and_missing_incumbents():
    data = json.loads(EVIDENCE.read_text())
    update = data['helper_updates'][-1]
    assert (update['reviewed'], update['imported'], len(update['retained_pending'])) == (65,61,4)
    points = {p['key']:p for p in data['points']}
    changed = [p for p in points.values() if 'admission' in p]
    assert len(changed) == 61
    assert sum(bool(bucket_costs(p)) for p in points.values()) == 165
    assert sum(data['recipes'][p['winner']]['io'] == 4 or
               (data['recipes'][p['winner']]['tc'] or {}).get('dense_reducer') == 2 for p in changed) == 53
    for pending in update['retained_pending']:
        assert points[pending['key']]['winner'] == pending['incumbent']
        assert 'admission' not in points[pending['key']]
    for p in changed:
        assert p['winner'] == choose(admitted_costs(p))
        r = data['recipes'][p['winner']]
        if (r['tc'] or {}).get('dense_reducer'):
            assert r['tc']['split'] == 4 and p['shape'][1]*p['shape'][2] <= 393216
    assert all(data['recipes'][b['recipe']]['io'] != 4 and
               not (data['recipes'][b['recipe']]['tc'] or {}).get('dense_reducer') for b in data['buckets'])


def test_fast_single_capture_cannot_promote_a_regression():
    p = dict(costs=[['old',10.], ['new',9.5]], admission=dict(
        rule='BOTH_CAPTURE_MEDIANS_BEAT_REMEASURED_CONTROL', controls=['old'], baseline='old',
        incumbent=None, capture_medians=dict(old=[10.,10.], new=[8.,11.])))
    assert admitted_costs(p) == {'old':10.}
    p['admission']['capture_medians']['new'] = [9.4,9.6]
    assert choose(admitted_costs(p)) == 'new'
    p['admission'].update(incumbent='old',baseline='new',controls=['old','new'])
    assert choose(admitted_costs(p)) == 'new'  # a faster old route can also replace the incumbent
    p['admission']['incumbent'] = 'missing'
    with pytest.raises(ValueError, match='incumbent'): admitted_costs(p)


@pytest.mark.parametrize('fault', ['missing-capture','faster-only-one-order','wrong-control'])
def test_resealed_admission_evidence_is_checked(fault):
    data = json.loads(EVIDENCE.read_text())
    p = next(p for p in data['points'] if p.get('admission') and p['winner'] != p['admission']['baseline'])
    a = p['admission']; w = p['winner']
    if fault == 'missing-capture': a['capture_medians'][w].pop()
    elif fault == 'wrong-control': a['baseline'] = w
    else:
        a['capture_medians'][w] = [2*p['us']-10000,10000]
    data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    with pytest.raises(ValueError): validate(data)


def test_cpp_selected_parent_materializes_same_source_for_resolve_and_prewarm(tmp_path):
    data = json.loads(EVIDENCE.read_text())
    points = [p for p in data['points'] if p.get('admission')]
    src = tmp_path/'selected.cpp'
    src.write_text('''#include "quactlize/dispatch/dense_prefill.hpp"
#include <iostream>
int main() {int m,n,k; while(std::cin>>m>>n>>k) {
    qks_request_v1 r{1,sizeof(r),8,1,m,n,k,1,m,UINT64_C(0x51384b5032540001)};
    auto d=quactlize::dispatch::dense_prefill::select(r,0); if(!d) return 2;
    auto const& c=*d.choice; auto const& t=c.tc;
    std::cout<<c.route<<" "<<c.io<<" "<<c.dequant;
    if(c.route!=2) std::cout<<" "<<t.symbol<<" "<<t.qtype<<" "<<t.route<<" "<<t.tm<<" "<<t.tn
        <<" "<<t.tk<<" "<<t.wm<<" "<<t.wn<<" "<<t.stages<<" "<<t.ap<<" "<<t.dn<<" "<<t.parent_persistent;
    std::cout<<"\\n";
}}''')
    exe = tmp_path/'selected'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),src,'-o',exe],check=True)
    lines = subprocess.check_output([exe],text=True,
        input=''.join(' '.join(map(str,p['shape'][1:]))+'\n' for p in points)).splitlines()
    for p,line in zip(points,lines,strict=True):
        r = data['recipes'][p['winner']]; values = line.split()
        assert list(map(int,values[:3])) == [r['route'],r['io'],r['dequant']]
        if r['tc'] is None: continue
        actual = parent_tuple(values[3],list(map(int,values[4:])))
        tc = r['tc']; fields = ('symbol','qtype','tm','tn','tk','wm','wn','stages','ap','dn','persistent')
        expected = {k:tc[k] for k in fields} | dict(route=ROUTES[r['route']])
        if tc.get('dense_reducer'): expected['dense_reducer'] = tc['dense_reducer']
        assert actual == expected
        assert Compiler.source(None,actual,'key') == Compiler.source(None,expected,'key')
        cparent = {k:v for k,v in expected.items() if k != 'dense_reducer'} | dict(route=r['route'])
        plan = dict(schema=PREFILL_SCHEMA,requests=[dict(status=0,kind=0,request=[8,0,0],parent=cparent)])
        assert prewarm_groups(plan) == [('f16',False,[expected])]
        if tc.get('dense_reducer'): assert '#define QK_DENSE_REDUCER 2\n' in Compiler.source(None,actual,'')


@pytest.mark.parametrize('suffix,axis', [('_rc2',0),('_rc99',None),('_rc2',True)])
def test_unknown_or_conflicting_parent_axis_is_not_silently_accepted(suffix,axis):
    p = dict(qtype=8,route='sf-dense',tm=64,tn=64,tk=64,wm=32,wn=32,stages=3,ap=0,dn=64,
             persistent=-1,symbol='sf_test'+suffix)
    if axis is not None:p['dense_reducer']=axis
    with pytest.raises(ValueError):materialize_parent(p)

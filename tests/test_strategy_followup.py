"""Host-only follow-up scheduling/evidence tests; no synthetic PPU claims."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from dev.strategy.components import statistics_checked
from dev.strategy.followup import finalists,paired_verdict,reusable,families
from quactlize.dispatch.table import Recipe,Workload
from quactlize.runtime.tuning import digest


def point():
    w=Workload(12,512,2048,1)
    r=Recipe.make('simt','fq',variant=3,columns=4,warps=4,values=4,split=1)
    return dict(id=w.key,workload=asdict(w),original_recipe=asdict(r),candidates=[asdict(r)],
                champion_controls=[],sf_configs=[],arms=[])


def sample(value):return statistics_checked([[value]*3 for _ in range(3)])


def screen():
    records={'old':sample(10.),'fast':sample(7.),'bad':statistics_checked([[20.]*3,[25.]*3,[24.]*3])}
    result=dict(status='PASS',component_arms={'original':['old'],'fast':['fast'],'bad':['bad']},
                selection=dict(costs_us={'original':10.,'fast':7.,'bad':24.},missing={},
                               best_rule='fast',best_control='original',noisy=['bad']))
    return result,records


def test_noisy_slow_loser_does_not_force_complete_resweep():
    result,records=screen();got=finalists(point(),result,records)
    assert got['status']=='READY' and got['arms']==['original','fast']
    assert got['discarded_noisy']==['bad']


def test_noisy_possible_winner_is_kept_despite_bad_median():
    result,records=screen();records['bad']=statistics_checked([[6.]*3,[25.]*3,[24.]*3])
    got=finalists(point(),result,records)
    assert got['arms']==['original','fast','bad']
    assert got['discarded_noisy']==[]
    blocked=finalists(point(),result,records,max_arms=2)
    assert blocked['status']=='BLOCKED' and len(blocked['arms'])==3


def test_exact_component_reuse_does_not_invent_independent_ab_timing():
    result,records=screen();result['component_arms']['fast']=['old']
    result['selection']['costs_us']['fast']=10.
    got=finalists(point(),result,records)
    assert got['status']=='INCUMBENT_UNCHANGED' and got['aliases']=={'fast':'original'}


def test_missing_original_or_candidates_stays_blocked():
    result,records=screen();result['selection']['missing']={'fast':['lost-record']}
    assert finalists(point(),result,records)['status']=='BLOCKED'


def test_alternating_pairs_can_reject_a_median_only_win():
    old=[[v]*5 for v in [9.9,9.9,9.9,10.2,10.2,10.2]]
    candidate=[[10.]*5 for _ in range(6)]
    result=paired_verdict({'original':old,'fast':candidate},{'original':['original'],'fast':['rule']})
    assert result['winner']=='original' and result['status']=='UNSTABLE_RETAIN_ORIGINAL'
    good=paired_verdict({'original':old,'fast':[[9.]*5 for _ in range(6)]},
                        {'original':['original'],'fast':['rule']})
    assert good['winner']=='fast' and not good['production_changed'] and not good['performance_admitted']
    with pytest.raises(ValueError,match='six'):
        paired_verdict({'original':old[:3]}, {'original':['original']})


def test_control_only_win_is_a_rule_miss_not_injected_champion():
    data={a:[[v]*5 for _ in range(6)] for a,v in [('original',10.),('rule',9.),('control',7.)]}
    result=paired_verdict(data,{'original':['original'],'rule':['rule'],'control':['control']})
    assert result['winner']=='rule' and result['status']=='RULE_MISS'


def test_reuse_requires_current_images_oracle_device_and_exact_arm_roles():
    p=point();p['arms']=[dict(id='original',recipe=p['original_recipe'],module='module')]
    authority=dict(oracle='oracle',body_source='body',l2={'bytes':64},
                   resource=dict(sdk='sdk',device='device'),vectors={'module':'binary'})
    old=dict(points={p['id']:dict(point=deepcopy(p))},authority=deepcopy(authority))
    assert reusable(old,p,authority)
    p2=deepcopy(p);p2['candidates']=[]
    assert not reusable(old,p2,authority)
    for key in ('oracle','body_source','l2'):
        changed=deepcopy(authority);changed[key]='changed'
        with pytest.raises(ValueError):reusable(old,p,changed)
    changed=deepcopy(authority);changed['vectors']['module']='other'
    with pytest.raises(ValueError,match='image'):reusable(old,p,changed)
    assert old['authority']==authority


def test_family_groups_weights_not_tokens_but_never_merges_precision():
    p=point();p2=deepcopy(p);p2['workload']['m']=8;p2['id']='another'
    p3=deepcopy(p);p3['workload']['compute']='bf16';p3['id']='bf16'
    grouped=families([p2,p3,p])
    assert sorted(map(len,grouped.values()))==[1,2]


@pytest.mark.parametrize('late_failure',[False,True])
def test_confirmation_alternates_single_components_and_checks_late_guards(monkeypatch,late_failure):
    pytest.importorskip('torch')
    import dev.strategy.confirm as c
    p=point();w=Workload(**p['workload'])
    recipes=[Recipe.make('simt','fq',variant=v,columns=4,warps=4,values=4,split=2) for v in (2,3)]
    p['arms']=[dict(id=a,recipe=asdict(r),roles=[role]) for a,r,role in
               zip(('original','fast'),recipes,('original','rule'))]
    events=[];calls=[]
    class Graph:
        count=16
        def __init__(self,name,us):self.name,self.us=name,us
        def close(self):events.append('close-'+self.name)
        def sample(self):calls.append(self.name);return self.us
    class Capture:
        def __init__(self,aid):
            self.groups=[[0],[1]];self.aid=aid
            self.descriptions=[dict(name='simt::register_reuse<'+aid+'>',symbol=aid),
                               dict(name='register_reuse_reduce<2>',symbol='reduce')]
        def isolate(self,i,**kwargs):return Graph(self.aid if i==0 else 'reduce',10. if i==0 and self.aid=='original' else 8. if i==0 else 1.)
    @contextmanager
    def prepared(b,w,arm,*args,**kwargs):
        d=dict(correctness='PENDING',numerical=[dict(cells=w.m*w.topk*w.n,bad=0,nonfinite=0,
               tolerance=.005,relative_l1_error=.001)]*5,zero_input='PASS',omitted_producer='DETECTED',
               pointer_negatives=['low','units'],fixture=dict(l2_bytes=64,copies=4,
                   minimum_read_bytes=64,active_ring_bytes=256))
        yield SimpleNamespace(capture=Capture(arm['id']),data=d)
        if late_failure:raise ValueError('late guard')
        d['correctness']='PASS';events.append('verified-'+arm['id'])
    monkeypatch.setattr(c,'prepared_arm',prepared)
    short=dict(status='READY',arms=['original','fast'],aliases={})
    clock_calls=[]
    def observe_clock():
        clock_calls.append(len(calls))
        return dict(cu_mhz=1700, call_position=len(calls))
    run=lambda:c.confirm_point(None,None,None,None,p,{},short,lambda a:(None,None),{},'source',
                              round_observer=observe_clock)
    if late_failure:
        with pytest.raises(ValueError,match='late guard'):run()
    else:
        result=run()
        assert result['decision']['winner']=='fast'
        assert result['decision']['statistics']['original']['median_us']==11.
        assert result['decision']['statistics']['fast']['median_us']==9.
        assert len(result['clock_observations'])==6
        assert clock_calls==[v for r in range(6) for v in (r*15,(r+1)*15)]
        assert all(v['timed_kernel_types']==1 for v in result['component_records'].values())
        from dev.strategy.followup import validate_confirmation
        validate_confirmation(result)
        changed=deepcopy(result);changed['arms']['fast']['pointer_negatives']=[]
        with pytest.raises(ValueError,match='numerical/negative'):validate_confirmation(changed)
        changed=deepcopy(result);changed['decision']['winner']='original'
        with pytest.raises(ValueError,match='decision'):validate_confirmation(changed)
    assert len(calls)==90  # 30 per producer, one shared reducer timed 30 not 60 times
    assert calls[:15]==['original']*5+['reduce']*5+['fast']*5
    assert calls[15:30]==['fast']*5+['reduce']*5+['original']*5
    assert 'close-original' in events and 'close-fast' in events and 'close-reduce' in events

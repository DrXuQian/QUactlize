"""Fault-injected CPU orchestration tests, not PPU correctness/timing."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.strategy.journal import ArmJournal, execute_point, planned_arms
from dev.strategy.components import recommendation, statistics_checked
from quactlize.dispatch.table import Workload, Recipe
from tools.run_kpack_strategy import summarize_point, save
from quactlize.runtime.tuning import digest


def point():
    w=Workload(12,1024,5120,1)
    recipes=[asdict(Recipe.make('simt','fq',variant=0,columns=4,warps=i,values=4,split=1)) for i in (2,4,8)]
    return dict(id=w.key,workload=asdict(w),candidates=recipes[:2],champion_controls=recipes[1:])


def passed():
    return dict(correctness='PASS',components=[digest('component')],receipt={},error=0.,fixture='HOST_MOCK')


def test_controls_deduplicate_but_never_become_rules_by_accident():
    arms=planned_arms(point())
    assert len(arms)==4
    assert [a['roles'] for a in arms]==[['original'],['rule'],['rule','control'],['control']]
    assert not any('rule' in a['roles'] for a in arms if a['recipe']==point()['champion_controls'][-1])


def test_failure_stops_child_and_fresh_child_advances_without_repeating_pass(tmp_path):
    p=point();journal=ArmJournal(tmp_path,{'scope':'HOST_MOCK'},'a'*32)
    arms=planned_arms(p);calls=[]
    def measure(arm):
        calls.append(arm['id'])
        if arm['id']==arms[1]['id']:raise RuntimeError('poisoned mock context')
        return passed()
    assert not execute_point(p,journal,measure)
    assert calls==[a['id'] for a in arms[:2]]
    frozen=(journal.location(p,arms[0])/'current.json').read_bytes()
    assert execute_point(p,journal,measure)  # same invocation, new context/process
    assert calls==[a['id'] for a in arms]
    assert not journal.pending([p])
    assert journal.read(p,arms[1])['status']=='FAIL'
    assert (journal.location(p,arms[0])/'current.json').read_bytes()==frozen
    # An explicit later resume retries only the failed arm.
    resumed=ArmJournal(tmp_path,{'scope':'HOST_MOCK'},'b'*32)
    assert [a['id'] for _,a in resumed.pending([p])]==[arms[1]['id']]
    retry=[]
    assert execute_point(p,resumed,lambda a:retry.append(a['id']) or passed())
    assert retry==[arms[1]['id']]
    history=[json.loads(f.read_text()) for f in resumed.location(p,arms[1]).glob('*.json') if f.name!='current.json']
    assert {r['status'] for r in history}=={'FAIL','PASS'}


def test_crashed_running_arm_is_reaped_after_child_exit(tmp_path):
    p=point();j=ArmJournal(tmp_path,{'scope':'MOCK'},'c'*32);arms=planned_arms(p)
    j.start(p,arms[0]);j.reap([p],-11,'child.log')
    assert j.read(p,arms[0])['status']=='CRASH'
    assert 'rc=-11' in j.read(p,arms[0])['error']
    assert not j.due(p,arms[0]) and j.due(p,arms[1])
    assert ArmJournal(tmp_path,{'scope':'MOCK'},'d'*32).due(p,arms[0])


@pytest.mark.parametrize('changed',['authority','workload','recipe','digest'])
def test_bad_resume_identity_never_overwrites_existing_result(tmp_path,changed):
    p=point();j=ArmJournal(tmp_path,{'sdk':'A'},'a'*32);arm=planned_arms(p)[0]
    j.start(p,arm);j.finish(p,arm,'PASS',data=passed())
    path=j.location(p,arm)/'current.json'
    if changed=='authority':j=ArmJournal(tmp_path,{'sdk':'B'},'a'*32)
    if changed=='workload':p=dict(p,workload=dict(p['workload'],m=2))
    if changed=='recipe':arm=dict(arm,recipe=point()['candidates'][0])
    if changed=='digest':
        data=json.loads(path.read_text());data['data']['error']=123
        path.write_text(json.dumps(data))
    frozen=path.read_bytes()
    with pytest.raises(ValueError,match='identity/digest'):j.read(p,arm)
    assert path.read_bytes()==frozen


def test_unchecked_arm_cannot_be_published_as_pass(tmp_path):
    p=point();j=ArmJournal(tmp_path,{},'a'*32);a=planned_arms(p)[0]
    j.start(p,a)
    with pytest.raises(ValueError,match='checked component'):j.finish(p,a,'PASS',data={'components':['x']})
    j.finish(p,a,'FAIL',error='not checked')
    with pytest.raises(ValueError,match='inactive'):j.finish(p,a,'PASS',data=passed())
    with pytest.raises(ValueError,match='invalid point/arm'):
        j.location(dict(p,id='../escape'),a)


def test_actual_fresh_processes_preserve_pass_and_continue_after_failure(tmp_path):
    p=point();save(tmp_path/'point.json',p)
    code=r'''
import json,sys
from pathlib import Path
from dev.strategy.journal import ArmJournal,execute_point,planned_arms
p=json.loads((Path(sys.argv[1])/'point.json').read_text())
j=ArmJournal(Path(sys.argv[1])/'arms',{'scope':'HOST_CHILD'},'a'*32)
bad=planned_arms(p)[1]['id']
def measure(a):
    if a['id']==bad:raise RuntimeError('mock device error')
    return {'correctness':'PASS','components':['a'*64],'receipt':{},'error':0}
sys.exit(0 if execute_point(p,j,measure) else 1)
'''
    first=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True)
    second=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True)
    assert (first.returncode,second.returncode)==(1,0)
    j=ArmJournal(tmp_path/'arms',{'scope':'HOST_CHILD'},'a'*32);arms=planned_arms(p)
    rows=[j.read(p,a) for a in arms]
    assert [r['status'] for r in rows]==['PASS','FAIL','PASS','PASS']
    assert rows[0]['pid']==rows[1]['pid'] and rows[1]['pid']!=rows[2]['pid']


def test_control_only_winner_is_not_an_admitted_rule_choice():
    def stat(t):return statistics_checked([[t]*3]*3)
    arms={'original':['o'],'rule':['r'],'control':['c']}
    roles={'original':['original'],'rule':['rule'],'control':['control']}
    records={'o':stat(10.),'r':stat(9.),'c':stat(8.)}
    result=recommendation(arms,records,roles=roles)
    assert result['recommended']=='original'
    assert result['best_rule']=='rule' and result['best_control']=='control'
    assert result['control_only_wins'] and result['rule_regret_pct']==12.5
    roles['control'].append('rule')
    assert recommendation(arms,records,roles=roles)['recommended']=='control'


@pytest.mark.parametrize('component', ['../escape', '', 'x'*64, 12, None])
def test_component_paths_are_only_digest_names(tmp_path, component):
    p=point();j=ArmJournal(tmp_path,{},'a'*32);a=planned_arms(p)[0]
    j.start(p,a)
    with pytest.raises(ValueError,match='checked component'):
        j.finish(p,a,'PASS',data=passed()|dict(components=[component]))


def test_case_summary_requires_all_arms_and_valid_component_files(tmp_path):
    p=point();j=ArmJournal(tmp_path/'arms',{'scope':'UNIT_NO_GPU'},'a'*32)
    identity={'authority':j.authority,'kernel':'mock'};key=digest(identity)
    data=statistics_checked([[1.]*3]*3)|dict(identity=identity)
    data['record_sha256']=digest(data)
    save(tmp_path/'components'/f'{key}.json',data)
    def measure(a):return passed()|dict(components=[key])
    arms=planned_arms(p)
    j.start(p,arms[0]);j.finish(p,arms[0],'PASS',data=measure(arms[0]))
    assert summarize_point(tmp_path,p,j)['status']=='FAIL'  # missing peers
    assert execute_point(p,j,measure)
    assert summarize_point(tmp_path,p,j)['status']=='PASS'
    data['median_us']=2;save(tmp_path/'components'/f'{key}.json',data)
    with pytest.raises(ValueError,match='digest'):summarize_point(tmp_path,p,j)

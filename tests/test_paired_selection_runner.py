"""Host checks for the bounded recipe/component/model handoff."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from dev.tp2_chain.production import ProductionBench, measure
from dev.tp2_chain.plan import POINTS
from quactlize.fusion.native import Recipe, Request, Sizes
from tools.run_paired_selection_box import Stages, check_trace, fetch, model_plan
from tools.paired_decode_selection import EVIDENCE

CATALOG=json.loads(EVIDENCE.read_text())


def test_samples_are_normalized_exactly_once_and_order_alternates():
    seen=[]
    def sample(arm,time):seen.append(arm);return time
    graphs={arm:mock.Mock(sample=lambda a=arm,t=time:sample(a,t),calls=40)
            for arm,time in [('baseline',43.),('candidate',39.)]}
    result=measure(graphs,'fixture')
    assert result['baseline']['median_us']==43. and result['candidate']['median_us']==39.
    assert seen[:30]==['baseline']*15+['candidate']*15
    assert seen[30:60]==['candidate']*15+['baseline']*15
    assert len(seen)==180 and seen.count('baseline')==seen.count('candidate')==90


def test_resume_keeps_completed_stages_and_only_retries_failure(tmp_path):
    stages=Stages(tmp_path,dict(source='a'),False)
    def success(out):
        return [sys.executable,'-c','import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir()',out]
    first=stages.run('component',success)
    retry=Stages(tmp_path,dict(source='a'),True)
    callback=mock.Mock(side_effect=AssertionError('must not rerun success'))
    assert retry.run('component',callback)==first
    with pytest.raises(ValueError,match='identity differs'):
        Stages(tmp_path,dict(source='changed'),True).run('component',callback)
    with pytest.raises(ValueError,match='failed stage'):
        stages.run('partial',lambda out:[sys.executable,'-c',
            'import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir(); sys.exit(2)',out])
    saved=json.loads((tmp_path/'partial.stage.json').read_text())['output']
    assert retry.run('partial',lambda out:[sys.executable,'-c','pass'],resume_partial=True)==Path(saved)
    assert len(list(tmp_path.glob('partial-*.log')))==2


def test_ci_worktree_is_outside_archived_results(tmp_path):
    results=tmp_path/'results';results.mkdir()
    stages=Stages(results,{},False)
    path=stages.run('caller-ci',lambda out:[sys.executable,'-c',
        'import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir()',out],work=True)
    assert path.parent==tmp_path/'ci'
    assert (results/'caller-ci.stage.json').is_file()
    assert len(list(results.glob('caller-ci-*.log')))==1


def test_fetch_rejects_wide_or_wrong_artifact_before_git(tmp_path):
    valid=dict(branch='artifacts/kpack-paired-recipes-v2',commit='a'*40,
        path='prebuilt/ppu0010/kpack-model-runtime-v1',manifest_sha256='b'*64)
    with mock.patch('tools.run_paired_selection_box.subprocess.run') as run:
        for update in (dict(commit='bad'),dict(branch='develop'),dict(path='prebuilt/**')):
            with pytest.raises(ValueError):fetch(valid|update,tmp_path)
        run.assert_not_called()


@pytest.mark.parametrize('name,recipe_name',[('qwen35-122b-q4km','q4_paired16_fixed'),
    ('qwen35-35b-q4km','q4_paired32_fixed')])
def test_warmed_trace_requires_exact_recipe_not_only_grouped_activity(tmp_path,name,recipe_name):
    path=tmp_path/name/'native';path.mkdir(parents=True)
    recipe=next(r for r in CATALOG['recipes'] if r['name']==recipe_name)
    good=dict(kernel_execution='PASS_SHORT_REQUEST',
        capture_scope='SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED',
        paired_missing_recipes=[],paired_observed_recipes=['0x'+recipe['id']])
    manifest=dict(paired_gate_up=dict(selection=CATALOG))
    (path/'proof.json').write_text(json.dumps(good))
    assert check_trace(tmp_path,name,manifest)['recipe']==recipe['id']
    for change in (dict(paired_observed_recipes=[]),dict(paired_missing_recipes=['0x123']),
                   dict(capture_scope='FIRST_REQUEST'),dict(kernel_execution='PARTIAL_SHORT_REQUEST')):
        (path/'proof.json').write_text(json.dumps(good|change))
        with pytest.raises(ValueError,match='not observed'):check_trace(tmp_path,name,manifest)


def test_two_model_plan_keeps_122b_tp2_and_35b_control_on_6_7(tmp_path):
    with mock.patch('tools.run_paired_selection_box.resolve_plan',side_effect=lambda p,n,r:p):
        plan=model_plan(tmp_path,['6','7'])
    assert plan['prompts']==[2048] and plan['generations']==[128]
    assert [(m['name'],m['devices'],m['split']) for m in plan['models']]==[
        ('qwen35-122b-q4km','6,7','tensor'),('qwen35-35b-q4km','6','none')]


@pytest.mark.parametrize('point,tokens,backend',[(POINTS[0],1,0),(POINTS[0],8,0),(POINTS[1],3,1)])
def test_component_entry_uses_explicit_recipe_workspace_and_labels_capability(point,tokens,backend):
    from test_tp2_paired_chain import fake_bench
    b=object.__new__(ProductionBench);b.__dict__.update(fake_bench().__dict__)
    b.point=point;b.selected=Recipe();b.layout=mock.Mock()
    queried=[]
    def select(req,out):
        r=C.cast(req,C.POINTER(Request)).contents
        if r.k==3072 and r.tokens!=1:return 24
        recipe=C.cast(out,C.POINTER(Recipe)).contents
        recipe.config.backend=backend
        recipe.config.split=2 if backend else 1
        recipe.recipe_id=31
        return 0
    def query(call,recipe,layout,sizes):
        C.cast(sizes,C.POINTER(Sizes)).contents.workspace_bytes=4096 if backend else 0
        queried.append(C.cast(recipe,C.POINTER(Recipe)).contents.recipe_id)
        return 0
    b.entries=dict(select_v2=select,query_v2=query)
    # The real layout is a C structure; no GPU allocation or launch here.
    from quactlize.fusion.native import Layout
    b.layout=Layout()
    b.configure(tokens)
    assert queried==[31]
    assert b.automatic==(point.k!=3072 or tokens==1)
    assert bool(b.recipe_workspace)==bool(backend)
    if backend:assert b.recipe_workspace.size==4096 and b.recipe_workspace in b.scratch


def test_box_shell_keeps_outer_docker_shell_and_checks_syntax():
    path=Path(__file__).resolve().parents[1]/'tools/run_paired_selection_box.sh'
    subprocess.run(['bash','-n',path],check=True)
    text=path.read_text()
    assert '\n(\n' in text and text.rstrip().endswith(')')
    assert '${PAIRED_DEVICES:-6,7}' in text and '${JOBS:-192}' in text
    assert 'results.tgz' in text and "--exclude='*.asysrep'" in text

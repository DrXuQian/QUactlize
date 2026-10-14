"""Small transfer files retain losers/errors/samples, not build caches."""
from copy import deepcopy
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.pack_kpack_strategy_review import main, digest, encoded


def write(path, value, seal=False):
    value = deepcopy(value)
    if seal:
        value['record_sha256'] = digest(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def fixture(tmp_path):
    root = tmp_path/'run'/'measurement'
    pid = 'a'*64
    workload = dict(qtype=12, n=512, k=2048, m=1, operation='dense', compute='f16')
    identity = dict(node=dict(name='simt_kernel', symbol='simt_kernel', grid=[64,1,1], block=[128,1,1]),
                    kind='producer', contract=dict(workload=workload), cache='COLD',
                    implementation='body', authority={'sha256':'authority'})
    cid = digest(identity)
    component = dict(identity=identity, samples_us=[[3.]*3 for _ in range(3)], median_us=3.,
                     round_medians_us=[3.]*3, spread_pct=0., status='PASS', scope='ISOLATED_SINGLE_KERNEL',
                     kind='producer', timed_kernel_types=1, first_use_excluded=True,
                     host_launch_cost_added=False, calls_per_graph=16)
    write(root/'components'/(cid+'.json'), component, True)
    proof = dict(correctness='PASS', components=[cid], fixture={'l2_bytes':64},
                 numerical=[dict(bad=0, nonfinite=0, relative_l1_error=.001, cells=512, tolerance=.005,
                                 golden_sha256='DO_NOT_EXPORT_FULL_TENSOR_HASH')]*5,
                 zero_input='PASS', omitted_producer='DETECTED', pointer_negatives=['low','units'],
                 receipt=dict(module_sha256='binary', grid=64))
    recipe = dict(kind='simt', route='fq', parameters=[['warps',4],['split',1]])

    def arm(aid, status, roles, data, **extra):
        return write(root/'arms'/pid/aid/'current.json', dict(
            identity=dict(point=pid, workload=workload, authority={'sha256':'authority'},
                          arm=dict(id=aid, roles=roles, recipe=recipe)), status=status, data=data, **extra), True)

    original = arm('original', 'PASS', ['original'], proof)
    arm('b'*64, 'FAIL', ['rule'], {}, error='independent oracle mismatch: first=7')
    confirmation = dict(status='INCUMBENT_UNCHANGED', correctness='PASS', components=[cid],
                        original_proof=proof, winner='original', scope='EXACT_SAME_TIMED_COMPONENTS_NO_INVENTED_AB')
    arm('c'*64, 'PASS', ['confirmation'], confirmation)
    write(root/'summary.json', dict(status='INCOMPLETE', passed=1, expected=1, points=[dict(
        id=pid,workload=workload,status='PASS')], production_changed=False, performance_admitted=False))
    write(root/'authority.json', dict(resource=dict(device={'compute_units':72},sdk='sdk',source='source'),
        harness={'sha256':'harness','files':{'omit':'ALL_SOURCE_HASHES_ARE_NOT_NEEDED'}},
        l2={'bytes':64},protocol='isolated',tc={'module':'binary'}))
    write(root/'screens'/(pid+'.json'), dict(point=pid,workload=workload,status='PASS',
        selection=dict(costs_us={'original':3.,'b'*64:4.},roles={'original':['original'],'b'*64:['rule']}),
        component_arms={'original':[cid]},receipts={'omit':'BIG_DUPLICATE_EXECUTION_MANIFEST'}))
    write(root/'families'/'large.json', {'omit':'FAMILY_MANIFEST_NOT_NEEDED'*10000})
    write(root/'plan'/'plan.json', {'omit':'FULL_PLAN_NOT_NEEDED'*10000})
    write(root/'arms'/pid/'original'/'old-attempt.json', {'omit':'OLD_ATTEMPT_NOT_NEEDED'})
    (root/'success.log').write_text('SUCCESS_LOG_NOT_NEEDED'*1000)
    (root/'failed.log').write_bytes(b'old unrelated output\n'*1000+b'Traceback: bad byte \xff\nValueError: mismatch\n')
    return root, pid, cid, component, original, arm


def unpack(path):
    raw = lzma.decompress(path.read_bytes())
    lines = raw.splitlines(keepends=True)
    values = [json.loads(line) for line in lines]
    assert values[-1]['type'] == 'end'
    assert values[-1]['prior_lines_sha256'] == hashlib.sha256(b''.join(lines[:-1])).hexdigest()
    return raw, values


def test_review_keeps_all_scores_correctness_errors_and_recomputable_samples(tmp_path):
    root, pid, cid, _, _, _ = fixture(tmp_path)
    before = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    output = tmp_path/'review.xz'
    assert main([str(root.parent),'--output',str(output),'--error-tail-bytes','128']) == 0
    raw, records = unpack(output)
    assert {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
    assert records[-1]['export_status'] == 'COMPLETE'
    arms = [r for r in records if r['type']=='arm']
    assert {r['status'] for r in arms} == {'PASS','FAIL'}
    good = next(r for r in arms if r['id']=='original')
    assert good['accuracy']['checks'] == 5 and good['accuracy']['relative_l1_error_max'] == .001
    assert good['components'] == [cid]
    timings = [r for r in records if r['type']=='component']
    assert len(timings)==1 and timings[0]['samples_us']==[[3.]*3]*3
    assert any(r['type']=='failure' and 'first=7' in r.get('error','') for r in records)
    screen = next(r for r in records if r['type']=='screen')
    assert len(screen['selection']['costs_us']) == 2  # Do not export only the winner.
    assert len([r for r in records if r['type']=='definition' and r['table']=='symbol']) == 1
    assert all(s not in raw for s in [b'FULL_PLAN_NOT_NEEDED',b'FAMILY_MANIFEST_NOT_NEEDED',
        b'OLD_ATTEMPT_NOT_NEEDED',b'SUCCESS_LOG_NOT_NEEDED',b'ALL_SOURCE_HASHES_ARE_NOT_NEEDED',
        b'BIG_DUPLICATE_EXECUTION_MANIFEST',b'DO_NOT_EXPORT_FULL_TENSOR_HASH'])
    logs = [r for r in records if r['type']=='error_log']
    assert len(logs)==1 and logs[0]['truncated'] and 'ValueError' in logs[0]['tail']


@pytest.mark.parametrize('damage',['seal','samples','missing','symlink'])
def test_export_reports_damage_instead_of_manufacturing_valid_timing(tmp_path,damage):
    root, _, cid, comp, _, _ = fixture(tmp_path)
    path = root/'components'/(cid+'.json')
    if damage=='seal':
        comp['record_sha256']='wrong';write(path,comp)
    elif damage=='samples':
        comp['samples_us'][0]=[90.]*3;write(path,comp,True)
    elif damage=='missing':
        path.rename(path.with_suffix('.not-json'))
    else:
        other=tmp_path/'private.json';write(other,{'private':'DO_NOT_READ'})
        path.rename(path.with_suffix('.original'));path.symlink_to(other)
    output=tmp_path/'review.xz'
    assert main([str(root),'--output',str(output)])==0
    raw,records=unpack(output)
    assert records[-1]['export_status']=='COMPLETE_WITH_ISSUES'
    assert any(r['type']=='export_issue' for r in records)
    assert not any(r['type']=='component' for r in records)
    assert b'DO_NOT_READ' not in raw


def test_prior_screen_remains_a_reference_not_a_missing_or_new_measurement(tmp_path):
    root,pid,cid,_,_,_=fixture(tmp_path)
    write(root/'imported.json',{pid:dict(previous_root='/not-present/prior/decode',
        previous_receipt='prior-digest',measurement_origin='PRIOR_SCREEN',component_arms={'original':[cid]})})
    path=root/'components'/(cid+'.json');path.rename(path.with_suffix('.held-on-original-box'))
    output=tmp_path/'review.xz';main([str(root),'--output',str(output)])
    _,records=unpack(output)
    assert records[-1]['export_status']=='COMPLETE'
    assert next(r for r in records if r['type']=='prior_components')['ids']==[cid]
    assert not any(r['type']=='component' for r in records)


def test_confirmation_keeps_all_raw_component_rounds_and_declared_decision(tmp_path):
    root,pid,cid,comp,original,arm=fixture(tmp_path)
    comp['samples_us']=[[3.]*5 for _ in range(6)];comp['round_medians_us']=[3.]*6
    data=dict(status='PASS',correctness='PASS',arms={'original':original['data']},
              arm_components={'original':[cid]},component_records={cid:comp},
              decision={'winner':'original','status':'PASS'},roles={'original':['original']})
    arm('c'*64,'PASS',['confirmation'],data)
    output=tmp_path/'review.xz';main([str(root),'--output',str(output)])
    _,records=unpack(output)
    timers=[r for r in records if r['type']=='component']
    assert {r['phase'] for r in timers}=={'screen','confirmation'}
    assert next(r for r in timers if r['phase']=='confirmation')['samples_us']==[[3.]*5]*6
    assert next(r for r in records if r['type']=='confirmation')['decision']==data['decision']


def test_standalone_copy_needs_no_repository_sdk_and_cannot_overwrite(tmp_path):
    root,*_=fixture(tmp_path)
    source=Path(__file__).resolve().parents[1]/'tools/pack_kpack_strategy_review.py'
    script=tmp_path/'pack.py';script.write_bytes(source.read_bytes())
    output=tmp_path/'review.xz'
    run=subprocess.run([sys.executable,'-I',str(script),str(root),'--output',str(output)],
                       cwd=tmp_path,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    assert 'STRATEGY_REVIEW_DONE' in run.stdout
    before=output.read_bytes()
    with pytest.raises(SystemExit):main([str(root),'--output',str(output)])
    assert output.read_bytes()==before


@pytest.mark.parametrize('completed', [True, False])
def test_campaign_finish_uses_small_review_or_bounded_setup_diagnostics(tmp_path, completed):
    root,*_=fixture(tmp_path)
    if not completed:
        (root/'summary.json').rename(root/'missing-summary.json')
    source=Path(__file__).resolve().parents[1]
    shell=(source/'tools/run_kpack_strategy_campaign_box.sh').read_text()
    finish=shell[shell.index('finish() {'):shell.index('trap finish EXIT')]
    env=dict(os.environ,campaign_python=sys.executable,campaign_source=str(source),
             campaign_run=str(root.parent),stage='test')
    result=subprocess.run(['bash','-c',finish+'\nfinish'],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'runner_rc=0 stage=test' in result.stdout
    paths=[Path(l.removeprefix('results=')) for l in result.stdout.splitlines() if l.startswith('results=')]
    assert len(paths)==1
    if completed:
        assert paths[0].suffix=='.xz'
        assert unpack(paths[0])[1][-1]['export_status']=='COMPLETE'
    else:
        with tarfile.open(paths[0]) as archive:
            assert all(m.size<=65536 for m in archive)
            assert not any('families/' in n or 'plan/' in n for n in archive.getnames())

"""Failed-only continuation controls; no host test constitutes device timing."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dev.native_components import retry
from dev.native_components.plan import plan
from quactlize.runtime.tuning import digest
from tools import retry_native_components as runner


class RetryTest(unittest.TestCase):
    def test_reviewed_plan_preserves_all_valid_rows_and_defers_only_native_dense(self):
        root=Path('/root/autodl-tmp/quactlize-strategy-close.UcfHzRy7/native-review-v0eh6y1h')
        if not root.exists():self.skipTest('optional uploaded review')
        work=plan();audit=retry.inspect_previous(root,work['points']);retry.verify_previous(audit)
        self.assertEqual(audit['counts'],dict(reused=1178,retry=268,deferred=98,helpers=36))
        types=[kind for w in audit['workers'].values()for kind in w['kinds'].values()]
        self.assertEqual(types.count('jit-params'),250);self.assertEqual(types.count('q8-activation'),18)
        self.assertEqual({k for w in audit['workers'].values()for k in w['summary']['results']},
                         {r['id']for r in work['points']})
        bad=copy.deepcopy(audit);bad['files_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'receipts/logs'):retry.verify_previous(bad)
        bad=copy.deepcopy(audit);bad['counts']['deferred']=0
        with self.assertRaisesRegex(ValueError,'inventory'):retry.verify_previous(bad)

    def test_reviewed_probe_requires_actual_output_equality_and_source_identity(self):
        path=Path('/root/review_new.tgz')
        if not path.exists():self.skipTest('optional uploaded diagnostic')
        self.assertEqual(retry.verify_diagnostic(path)['archive_sha256'],retry.DIAGNOSTIC_SHA)
        real_sha=retry.sha
        with patch.object(retry,'sha',side_effect=lambda p:'changed'if Path(p).name=='bridge.cuh'else real_sha(p)):
            with self.assertRaisesRegex(ValueError,'source changed: bridge'):retry.verify_diagnostic(path)
        with patch.object(retry,'sha',return_value='different archive'):
            with self.assertRaisesRegex(ValueError,'17-job'):retry.verify_diagnostic(path)

    def test_physical_card_and_production_changes_still_reject(self):
        identity={k:k for k in ('binding','headers','flags','response_files','compiler_sha256','llama_head')}
        identity['libraries']={'/sdk/libcudart.so':'runtime'}
        receipt=dict(identity=identity);dev=dict(sdk='sdk',device={'pci':'01'})
        old=dict(adapter=receipt,authority=dict(device=dev,rounds=3,samples=3))
        probe=dict(adapter=receipt)
        retry.validate_current(old,receipt,dev,probe)
        changed=copy.deepcopy(dev);changed['device']['pci']='02'
        with self.assertRaisesRegex(ValueError,'physical GPU'):retry.validate_current(old,receipt,changed,probe)
        for key in ('llama_head','headers','binding','libraries'):
            changed=copy.deepcopy(receipt)
            changed['identity'][key]={'/sdk/libcudart.so':'new'}if key=='libraries'else'new'
            with self.assertRaises(ValueError):retry.validate_current(old,changed,dev,probe)

    def test_saved_result_cannot_replace_reviewed_sum_or_failure(self):
        prior=dict(status='MEASURED',sum_us=10,sha256='hash',path='/old/a.json')
        retry.same_record(dict(prior,data={'unchanged':True}),prior)
        for changed in (dict(prior,sum_us=1),dict(prior,status='FAIL'),dict(prior,path='/other/a.json')):
            with self.assertRaisesRegex(ValueError,'reviewed summary'):retry.same_record(changed,prior)

    def test_deferred_is_never_full_coverage_or_no_regression(self):
        previous=dict(reused=['a'],retry=['b'],deferred=['c'],summary={'results':dict.fromkeys('abc')})
        records={k:dict(status='MEASURED'if k!='c'else'DEFERRED_NATIVE_FP16_ACCURACY')for k in 'abc'}
        compared=[dict(id=k,status='HISTORICAL_SCREEN_ONLY'if k!='c'else'DEFERRED_NATIVE_FP16_ACCURACY')for k in 'abc']
        result=retry.comparison_summary(records,compared,previous)
        self.assertEqual(result['status'],'COMPLETE_EXCEPT_DEFERRED_NATIVE_FP16')
        self.assertEqual((result['measured'],result['retried_measured'],result['deferred']),(2,1,1))
        self.assertFalse(result['full_coverage']);self.assertFalse(result['no_regression_admitted'])
        compared[1]['status']='INCOMPLETE'
        self.assertEqual(retry.comparison_summary(records,compared,previous)['status'],'INCOMPLETE')

    def test_attempt_path_must_stay_inside_worker(self):
        with tempfile.TemporaryDirectory()as td:
            worker=Path(td)/'gpu-1';path=worker/'attempts/uuid/run.log';path.parent.mkdir(parents=True);path.write_text('error')
            self.assertEqual(retry.local_attempt(worker,'/old/gpu-1/attempts/uuid/run.log'),path)
            with self.assertRaisesRegex(ValueError,'outside'):retry.local_attempt(worker,'/old/gpu-2/attempts/uuid/run.log')

    def test_freeze_never_relabels_previous_content(self):
        with tempfile.TemporaryDirectory()as td:
            path=Path(td)/'result.json';runner.freeze(path,{'authority':'old','sum_us':30})
            before=path.read_bytes();runner.freeze(path,{'authority':'old','sum_us':30})
            self.assertEqual(before,path.read_bytes())
            with self.assertRaisesRegex(ValueError,'changed'):runner.freeze(path,{'authority':'new','sum_us':29})
            self.assertEqual(before,path.read_bytes())

    def test_worker_schedules_only_failure_then_resume_schedules_nothing(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);old=root/'previous/gpu-1';(old/'results').mkdir(parents=True)
            out=root/'retry/gpu-1';sdk=root/'sdk';(sdk/'lib').mkdir(parents=True)
            (sdk/'lib/libhggc_wrapper.so').write_bytes(b'host-only')
            binary=root/'carrier';binary.write_bytes(b'host-only')
            raw=root/'raw.json';raw.write_text('{}')
            value=dict(status='MEASURED',sum_us=30,path=str(raw),sha256='raw',data={'unchanged':True})
            (old/'results/a.json').write_text(json.dumps(value))
            expected={k:v for k,v in value.items()if k!='data'}
            dev=dict(sdk='sdk',device={'pci':'host-test'},l2={'bytes':67108864})
            previous=dict(reused=['a'],retry=['b'],deferred=['c'],helpers={},
                authority=dict(device=dev,sha256='old-auth'),
                summary={'results':{'a':expected,'b':{'status':'FAIL'},'c':{'status':'FAIL'}}})
            audit=dict(files_sha256='reviewed',workers={'gpu-1':previous})
            history=json.loads((runner.ROOT/'docs/measurements/kpack_native_history_20261019.json').read_text())
            work=dict(sha256=history['source_plan'],points=[dict(id=k,workload={})for k in 'abc'])
            args=SimpleNamespace(worker='gpu-1',previous=old.parent,output=out,sdk=sdk,
                                 llama_source=root,llama_build=root)
            def load(path,authority,row):
                return json.loads(path.read_text())if path.exists()else None
            def measured(a,b,request,rows,authority):
                self.assertEqual([r['id']for r in rows],['b'])
                self.assertEqual(a.reuse_fixtures,old/'fixtures')
                v=dict(status='MEASURED',sum_us=5,authority=authority)
                runner.atomic_json(out/'results/b.json',v)
                return {'b':v}
            compared=[dict(id=k,workload={},status='HISTORICAL_SCREEN_ONLY'if k!='c'else'INCOMPLETE')for k in 'abc']
            with patch.object(runner.native,'environment',return_value={}),\
                 patch.object(runner.native,'device',return_value=dev),\
                 patch.object(runner,'check_cohort',return_value={}),\
                 patch.object(runner,'harness_identity',return_value={'sha256':'current'}),\
                 patch.object(runner,'validate_current'),\
                 patch.object(runner.native,'build_adapter',return_value=(root/'adapter',{'binary_sha256':'adapter','identity':{'libraries':{}}})),\
                 patch.object(runner.native,'carrier',return_value=binary),\
                 patch.object(runner.native,'helper_jobs',return_value=([],dict.fromkeys('abc',[]))),\
                 patch.object(runner.native,'load_measured',side_effect=load),\
                 patch.object(runner.native,'validated',return_value=value['data']),\
                 patch.object(runner.native,'point_request',side_effect=lambda r,l2:r),\
                 patch.object(runner,'family',return_value='one'),\
                 patch.object(runner.native,'run_family',side_effect=measured)as run,\
                 patch.object(runner.native,'comparison_rows',return_value=compared),\
                 patch.object(runner.native,'write_comparison'),\
                 patch.object(runner.fcntl,'flock'),redirect_stdout(io.StringIO()):
                before=(old/'results/a.json').read_bytes()
                self.assertEqual(runner.worker(args,audit,{'archive_sha256':'diag'},work),0)
                self.assertEqual(run.call_count,1)
                self.assertEqual(runner.worker(args,audit,{'archive_sha256':'diag'},work),0)
                self.assertEqual(run.call_count,1)
                self.assertEqual(before,(old/'results/a.json').read_bytes())
                self.assertEqual(json.loads((out/'results/a.json').read_text()),value)
                self.assertEqual(json.loads((out/'results/c.json').read_text())['status'],'DEFERRED_NATIVE_FP16_ACCURACY')

    def test_coordinator_does_not_hide_a_worker_exit_behind_complete_summary(self):
        for failed in (False,True):
            with self.subTest(failed=failed),tempfile.TemporaryDirectory()as td:
                root=Path(td);cards=['1','2','3','4','5','7'];started=[]
                workers={'gpu-'+c:dict(authority={'device':{'physical':c}},retry=['b'],reused=['a'],deferred=['c'])for c in cards}
                args=SimpleNamespace(devices=','.join(cards),previous=root/'previous',diagnostic=root/'diagnostic.tgz',
                    output=root,llama_source=root/'llama',llama_build=root/'build',sdk=root/'sdk',l2_bytes=67108864)
                def probe(command,env,**kwargs):
                    out=Path(command[command.index('--output')+1])
                    runner.atomic_json(out/'device.json',{'physical':env['CUDA_VISIBLE_DEVICES']})
                class Child:
                    def __init__(self,command,env,**kwargs):
                        card=env['CUDA_VISIBLE_DEVICES'];started.append(card)
                        self.returncode=1 if failed and card=='2'else 0
                        out=Path(command[command.index('--output')+1])
                        runner.atomic_json(out/'summary.json',dict(status='COMPLETE_EXCEPT_DEFERRED_NATIVE_FP16',
                            retried_measured=1,measured=2,deferred=1))
                    def poll(self):return self.returncode
                with patch.object(runner.subprocess,'run',side_effect=probe),\
                     patch.object(runner.subprocess,'Popen',Child),\
                     patch.object(runner,'harness_identity',return_value={'sha256':'source'}),\
                     patch.object(runner,'pack',return_value=root/'review.tgz')as pack,redirect_stdout(io.StringIO()):
                    rc=runner.coordinator(args,{'workers':workers,'files_sha256':'reviewed'},{'archive_sha256':'diagnostic'})
                self.assertEqual(started,cards);self.assertEqual(pack.call_count,1)
                self.assertEqual(rc,1 if failed else 0)
                summary=json.loads((root/'summary.json').read_text())
                self.assertEqual(summary['status'],'INCOMPLETE'if failed else'COMPLETE_EXCEPT_DEFERRED_NATIVE_FP16')
                self.assertFalse(summary['full_coverage']);self.assertFalse(summary['no_regression_admitted'])

    def test_fixture_reuse_never_creates_missing_files_in_previous_run(self):
        for complete in (True,False):
            with self.subTest(complete=complete),tempfile.TemporaryDirectory()as td:
                root=Path(td);old=root/'old/one';old.mkdir(parents=True)
                (old/'weights.raw').write_bytes(b'old')
                if complete:(old/'weights.json').write_text('{}')
                before={p.name:p.read_bytes()for p in old.iterdir()}
                args=SimpleNamespace(output=root/'new',reuse_fixtures=old.parent)
                with patch.object(runner.native,'family',return_value='one'),\
                     patch('dev.native_components.fixture.weights',side_effect=ValueError('stop before GPU'))as weights,\
                     redirect_stdout(io.StringIO()):
                    runner.native.run_family(args,None,{},[{'id':'p','workload':{}}],{})
                self.assertEqual(weights.call_args.args[1],old if complete else root/'new/fixtures/one')
                self.assertEqual(before,{p.name:p.read_bytes()for p in old.iterdir()})


if __name__=='__main__':unittest.main()

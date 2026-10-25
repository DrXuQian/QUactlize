import unittest
from unittest.mock import patch
import copy
from dataclasses import asdict
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tempfile

from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.review_prefill_cohort import ingest, merge, reparse_attempt, tables

ROOT=Path(__file__).resolve().parents[1]


class ReviewTests(unittest.TestCase):
    def point(self,n,us):
        w=Workload(12,n,5120,2048)
        rows=[dict(candidate=c,case_id=c,arm='full-fp16',status='PASS',helper_mode='experimental',
                   gpu_sum_us=t,components=[c]) for c,t in us.items()]
        rows.append(dict(candidate='native',arm='native-fp16',status='PASS',gpu_sum_us=200))
        return w.key,dict(workload=asdict(w),authority='a',measurements=rows)

    def test_bucket_uses_relative_regret_not_absolute_latency(self):
        points=dict([self.point(5120,dict(a=10,b=11)),self.point(6144,dict(a=120,b=100))])
        exact,buckets=tables(points,include_buckets=True)
        self.assertEqual({r['candidate'] for r in exact.values()},{'a','b'})
        self.assertEqual(len(buckets),1)
        b=next(iter(buckets.values()))
        self.assertEqual(b['candidate'],'b')
        self.assertAlmostEqual(b['worst_regret_pct'],10)
        self.assertFalse(b['holdout_validated'])

    def test_missing_bucket_measurement_is_not_estimated(self):
        points=dict([self.point(5120,dict(a=10)),self.point(6144,dict(b=100))])
        _,b=tables(points,include_buckets=True)
        self.assertEqual(next(iter(b.values()))['status'],'NO_COMMON_MEASURED_CONFIG')

    def test_stale_component_never_relabels_an_old_full_call(self):
        key,point=self.point(5120,dict(a=10,b=11))
        point['measurements'][0]['status']='STALE_IMPLEMENTATION_REMEASURE_FULL_CALL'
        exact,_=tables({key:point})
        self.assertEqual(exact[key]['candidate'],'b')
        point['measurements'][1]['status']='STALE_IMPLEMENTATION_REMEASURE_FULL_CALL'
        self.assertEqual(tables({key:point})[0][key]['status'],'REMEASURE_REQUIRED')

    def test_unresolved_member_does_not_silently_shrink_bucket(self):
        one=self.point(5120,dict(a=10));two=self.point(6144,dict(a=11))
        two[1]['measurements']=[]
        exact,buckets=tables(dict([one,two]),include_buckets=True)
        self.assertEqual(exact[two[0]]['status'],'REMEASURE_REQUIRED')
        self.assertEqual(next(iter(buckets.values()))['status'],'NO_COMMON_MEASURED_CONFIG')

    def test_default_only_reports_each_shape_without_bucket_rules(self):
        points=dict([self.point(5120,dict(a=10)),self.point(6144,dict(a=11))])
        exact,buckets=tables(points)
        self.assertEqual(set(exact),set(points))
        self.assertEqual(buckets,{})
        self.assertTrue(all('bucket' not in row for row in exact.values()))

    def test_quactlize_only_ranks_against_incumbent_without_native_claim(self):
        key,point=self.point(5120,dict(a=10,b=12))
        point['measurements']=point['measurements'][:-1]
        point['native_comparison']=False
        point['measurements'][1]['roles']=['model-config']
        exact,buckets=tables({key:point})
        self.assertEqual(exact[key]['candidate'],'a')
        self.assertEqual(exact[key]['incumbent_us'],12)
        self.assertAlmostEqual(exact[key]['incumbent_delta_pct'],-100/6)
        self.assertIsNone(exact[key]['native_us']);self.assertIsNone(exact[key]['delta_pct'])
        self.assertFalse(exact[key]['native_comparison']);self.assertEqual(buckets,{})
        point['native_comparison']=True
        self.assertEqual(tables({key:point})[0][key]['status'],'REMEASURE_REQUIRED')



class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self,name='run',n=5120,revision='old',complete=True):
        root=self.root/name;root.mkdir()
        authority=dict(revision=revision,device=dict(sdk={'receipt':'sdk-sha'},l2={'bytes':67108864},
            device=dict(compute_units=72,warp_size=32,max_threads=1024,DriverGetVersion=13000,RuntimeGetVersion=13000)))
        aid=digest(authority);w=Workload(12,n,5120,2048)
        native=dict(id='native-fp16',arm='native-fp16')
        candidate=dict(id='config',arm='full-fp16',helper_mode='experimental',roles=['rule'],
                       recipe=dict(kind='full',route='full',parameters=[['dequant_variant',20],['vector_io',3]]))
        spec=dict(id=w.key,workload=asdict(w),consumers=['model:layer'],cases=[candidate])
        (root/'authority.json').write_text(json.dumps(authority))
        (root/'execution-plan.json').write_text(json.dumps(dict(authority=aid,points=[spec])))
        definitions=[native,candidate] if complete else [native]
        proof=[];parsed=[]
        for case in definitions:
            proof.append(dict(case_id=case['id'],arm=case['arm'],status='PASS',
                correctness=[dict(nonfinite=0,relative_l1_error=.001)],weight_bits=[dict(bad=0,bits=512)],
                omitted_producer='DETECTED'))
            kernels=[dict(name=stage,kind='kernel',phase=stage,grid=[20,1,1],block=[128,1,1],shared_bytes=0,
                          samples_us=[v,v,v],median_us=v) for stage,v in [('input',1),('main',10),('output',1)]]
            parsed.append(dict(case_id=case['id'],arm=case['arm'],helper_mode=case.get('helper_mode'),
                status='PASS',first_ring_excluded=True,cuda_graph=False,timing_method='ASYS_GPU_ACTIVITY',
                samples=[dict(gpu_sum_us=12)]*3,gpu_sum_us=12,measured_rounds_us=[12]*3,
                kernel_inventory=kernels,phases_us=dict(input=1,main=10,output=1)))
        path=root/'batches'/'batch'/'pass.json';path.parent.mkdir(parents=True)
        cases=root/'cases.json';cases.write_text(json.dumps(proof))
        receipt=dict(seal=dict(authority=aid,workload=asdict(w),cases=definitions),
            parsed=dict(status='PASS',cases=parsed),cases_path=str(cases),cases_sha256=sha(cases),
            asysrep=str(root/'trace.asysrep'),sqlite=str(root/'trace.sqlite'))
        receipt['record_sha256']=digest(receipt);path.write_text(json.dumps(receipt))
        return root,w.key,path

    def rewrite(self,path,change):
        data=json.loads(path.read_text());change(data)
        data['record_sha256']=digest({k:v for k,v in data.items() if k!='record_sha256'})
        path.write_text(json.dumps(data))

    def test_component_export_retains_identity_samples_and_consumers(self):
        root,key,_=self.fixture()
        value=ingest(root)
        self.assertFalse(value['incomplete'])
        point=value['points'][key];self.assertEqual(point['consumers'],['model:layer'])
        row=point['measurements'][1]
        self.assertEqual(row['complete_samples_us'],[12]*3)
        self.assertEqual(row['component_samples_us'],[[1]*3,[10]*3,[1]*3])
        self.assertEqual([value['components'][c]['phase'] for c in row['components']],['input','main','output'])
        self.assertEqual(row['numerical'][0]['relative_l1_error'],.001)

    def test_optional_resource_rejection_remains_in_compact_review(self):
        for complete in (True,False):
            root,key,_=self.fixture(name=str(complete),complete=complete)
            path=root/'execution-plan.json';plan=json.loads(path.read_text())
            rejected=[dict(case={'id':'optional-heavy'},receipt={'status':'UNAVAILABLE','rc':1})]
            plan['points'][0].update(resource_unavailable=rejected,
                sf_screen={'legal_parents':12},proposal_compute_units=72,table_coverage='MISS')
            path.write_text(json.dumps(plan))
            result=ingest(root)
            point=result['points'][key] if complete else next(iter(result['partial_points'].values()))
            self.assertEqual(point['resource_unavailable'],rejected)
            self.assertEqual(point['table_coverage'],'MISS')
            self.assertEqual(point['proposal_compute_units'],72)

    def test_explicit_quactlize_only_ingest_does_not_demand_native_receipt(self):
        root,key,path=self.fixture()
        plan=json.loads((root/'execution-plan.json').read_text())
        plan['points'][0]['native_comparison']=False
        (root/'execution-plan.json').write_text(json.dumps(plan))
        data=json.loads(path.read_text());proof=json.loads(Path(data['cases_path']).read_text())
        Path(data['cases_path']).write_text(json.dumps(proof[1:]))
        data['seal']['cases']=data['seal']['cases'][1:]
        data['parsed']['cases']=data['parsed']['cases'][1:]
        data['cases_sha256']=sha(data['cases_path'])
        data['record_sha256']=digest({k:v for k,v in data.items() if k!='record_sha256'})
        path.write_text(json.dumps(data))
        value=ingest(root)
        self.assertFalse(value['incomplete']);self.assertFalse(value['points'][key]['native_comparison'])
        self.assertEqual(len(value['points'][key]['measurements']),1)

    def paired(self, *, drift=False):
        root,key,path=self.fixture()
        plan=json.loads((root/'execution-plan.json').read_text())
        plan['points'][0]['confirmation_captures']=2
        (root/'execution-plan.json').write_text(json.dumps(plan))
        data=json.loads(path.read_text());data['seal']['confirmation']=1
        for row in data['parsed']['cases']:
            row['gpu_sum_us']*=2
            row['measured_rounds_us']=[v*2 for v in row['measured_rounds_us']]
            row['phases_us']={k:v*2 for k,v in row['phases_us'].items()}
            for sample in row['samples']:sample['gpu_sum_us']*=2
            for kernel in row['kernel_inventory']:
                kernel['median_us']*=2;kernel['samples_us']=[v*2 for v in kernel['samples_us']]
            if drift:row['kernel_inventory'][0]['block']=[256,1,1]
        second=root/'batches/reverse/pass.json';second.parent.mkdir()
        data['record_sha256']=digest({k:v for k,v in data.items() if k!='record_sha256'})
        second.write_text(json.dumps(data))
        return root,key,path,second

    def test_paired_confirmation_keeps_both_captures_not_fastest(self):
        root,key,_,_=self.paired()
        result=ingest(root)
        self.assertFalse(result['incomplete'])
        for row in result['points'][key]['measurements']:
            self.assertEqual(row['gpu_sum_us'],18)
            self.assertEqual(row['complete_samples_us'],[12]*3+[24]*3)
            self.assertEqual(row['component_samples_us'][0],[1]*3+[2]*3)
            self.assertEqual([r['confirmation'] for r in row['confirmation_captures']],[0,1])
            self.assertNotIn('raw_receipt',row)
            self.assertTrue(all(r['raw_receipt_sha256'] for r in row['confirmation_captures']))

    def test_missing_direction_and_launch_drift_cannot_be_complete(self):
        root,key,_,second=self.paired()
        second.unlink()
        value=ingest(root)
        self.assertEqual(value['incomplete'],[key])
        partial=next(iter(value['partial_points'].values()))
        self.assertEqual(partial['unconfirmed'],['config','native-fp16'])
        self.assertTrue(all(r['status']=='MISSING_CONFIRMATION' for r in partial['measurements']))

    def test_two_captures_with_different_kernel_launches_remain_partial(self):
        root,key,_,_=self.paired(drift=True)
        value=ingest(root)
        self.assertEqual(value['incomplete'],[key])
        rows=next(iter(value['partial_points'].values()))['measurements']
        self.assertTrue(all(r['status']=='CONFIRMATION_KERNEL_OR_ARITHMETIC_DRIFT' for r in rows))

    def test_duplicate_capture_index_is_not_another_confirmation(self):
        root,_,_,second=self.paired()
        self.rewrite(second,lambda d:d['seal'].update(confirmation=0))
        with self.assertRaisesRegex(ValueError,'duplicate point/config'):ingest(root)

    def test_ordinary_plan_cannot_admit_a_second_capture(self):
        root,_,_,_=self.paired()
        plan=json.loads((root/'execution-plan.json').read_text())
        plan['points'][0].pop('confirmation_captures')
        (root/'execution-plan.json').write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'confirmation capture'):ingest(root)

    def test_uploaded_batch_proof_is_relocated_without_changing_its_hash(self):
        root,key,path=self.fixture()
        attempt=path.parent/'attempt-123';attempt.mkdir()
        (root/'cases.json').rename(attempt/'cases.json')
        self.rewrite(path,lambda d:d.update(cases_path='/unavailable/batches/batch/attempt-123/cases.json'))
        self.assertIn(key,ingest(root)['points'])

    def test_partial_rerun_never_overwrites_completed_point(self):
        old,key,_=self.fixture();new,_,_=self.fixture('new',revision='new',complete=False)
        result=ingest(old);result['history']=[]
        previous=copy.deepcopy(result['points'][key])
        merge(result,ingest(new),set())
        self.assertEqual(result['points'][key],previous)
        self.assertEqual(len(result['partial_points']),1)
        partial=next(iter(result['partial_points'].values()))
        self.assertEqual(partial['missing'],['config'])
        self.assertEqual(len(partial['measurements']),1)

    def test_single_point_update_keeps_other_points_and_history(self):
        old,key,_=self.fixture();other,other_key,_=self.fixture('other',n=6144)
        new,_,_=self.fixture('new',revision='new')
        result=ingest(old);result['history']=[]
        merge(result,ingest(other),set());untouched=copy.deepcopy(result['points'][other_key])
        seen=set();merge(result,ingest(new),seen)
        self.assertEqual(result['points'][other_key],untouched)
        self.assertEqual(len(result['history']),1)
        self.assertNotEqual(result['history'][0]['previous']['authority'],result['points'][key]['authority'])
        with self.assertRaisesRegex(ValueError,'multiple new runs'):merge(result,ingest(new),seen)

    def test_reject_relabelled_candidate_and_bad_component(self):
        root,_,path=self.fixture()
        original=path.read_text()
        self.rewrite(path,lambda d:d['seal']['cases'][1].update(helper_mode='production'))
        with self.assertRaisesRegex(ValueError,'recipe differs'):ingest(root)
        path.write_text(original)
        self.rewrite(path,lambda d:d['parsed']['cases'][1]['kernel_inventory'][0].update(
            samples_us=[-1]*3,median_us=-1))
        with self.assertRaisesRegex(ValueError,'component median'):ingest(root)
        data=json.loads(original);data['parsed']['cases'][1]['gpu_sum_us']=float('nan')
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'JSON compliant'):ingest(root)

    def test_reject_incomplete_component_sums_and_changed_numerical_evidence(self):
        root,_,path=self.fixture()
        original=path.read_text()
        self.rewrite(path,lambda d:d['parsed']['cases'][1]['kernel_inventory'][0].update(
            samples_us=[2]*3,median_us=2))
        with self.assertRaisesRegex(ValueError,'component sum'):ingest(root)
        path.write_text(original)
        (root/'cases.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'numerical evidence changed'):ingest(root)

    def test_cli_roundtrip_incremental_and_invalidation(self):
        root,key,_=self.fixture()
        first=self.root/'first.json.xz';second=self.root/'second.json.xz'
        script=ROOT/'tools/review_prefill_cohort.py'
        subprocess.run([sys.executable,str(script),'--run',str(root),'--output',str(first)],check=True,capture_output=True)
        with lzma.open(first,'rt') as stream:one=json.load(stream)
        self.assertEqual(one['buckets'],{})
        self.assertEqual(one['bucket_status'],'PENDING_USER_FEEDBACK')
        component=one['points'][key]['measurements'][1]['components'][1]
        subprocess.run([sys.executable,str(script),'--previous',str(first),'--invalidate-component',component,
                        '--output',str(second)],check=True,capture_output=True)
        with lzma.open(second,'rt') as stream:two=json.load(stream)
        self.assertEqual(two['exact'][key]['status'],'REMEASURE_REQUIRED')
        self.assertEqual(two['review_sha256'],digest({k:v for k,v in two.items() if k!='review_sha256'}))

    def test_device_domain_change_is_not_cross_hardware_tuning(self):
        root,_,_=self.fixture();result=ingest(root);result['history']=[]
        other=copy.deepcopy(result)
        next(iter(other['authorities'].values()))['device']['device']['compute_units']=64
        with self.assertRaisesRegex(ValueError,'domain changed'):merge(result,other,set())

    def reparse_fixture(self):
        root,key,path=self.fixture(complete=False)
        result=ingest(root);result['history']=[];result['review_sha256']='previous-review'
        partial=next(iter(result['partial_points'].values()))
        case=json.loads((root/'execution-plan.json').read_text())['points'][0]['cases'][0]
        case['id']=Recipe.from_dict(case['recipe']).key
        partial['missing']=[case['id']]
        identity={k:v for k,v in case.items() if k not in ('id','roles','launch','module')}
        result['candidates'][digest(identity)]=identity
        seal=dict(authority=partial['authority'],workload=partial['workload'],cases=[case])
        attempt=root/'batches'/digest([key,[case]])/'attempt-123';attempt.mkdir(parents=True)
        old=json.loads(path.read_text())
        proof=json.loads((root/'cases.json').read_text())
        proof[0].update(case_id=case['id'],arm=case['arm'],helper_mode=case['helper_mode'],
                        recipe=case['recipe'],workload=partial['workload'],roles=case['roles'],zero_a='PASS',
                        receipt=dict(arithmetic=dict(a='F16',b='F16',c='F16',accumulator='FP32')))
        parsed=old['parsed'];parsed['cases'][0].update(case_id=case['id'],arm=case['arm'],helper_mode=case['helper_mode'])
        for name,value in [('request.json',seal),('cases.json',proof),('profile.process.json',
                dict(returncode=0,command=['asys','-o',str(attempt/'trace'),'--batch',str(attempt/'request.json')]))]:
            (attempt/name).write_text(json.dumps(value))
        # Parser behavior has its own real SQLite tests; these tests isolate
        # correction binding/merge rules from the raw-activity parser.
        for name in ('trace.sqlite','profile.log','export.log'):(attempt/name).write_text('fixture')
        parsed['sqlite_sha256']=sha(attempt/'trace.sqlite')
        result['diagnostics']=[dict(path='original-failure.json',data=dict(point=key,cases=[case['id']],error='reset missing'))]
        return result,attempt,parsed,key

    def test_reparse_only_completes_missing_rows_and_preserves_raw_evidence(self):
        result,attempt,parsed,key=self.reparse_fixture()
        before=copy.deepcopy(result);raw={p.name:p.read_bytes() for p in attempt.iterdir()}
        with patch('dev.prefill_asys.trace.parse',return_value=parsed):reparse_attempt(result,attempt)
        self.assertFalse(result['partial_points'])
        self.assertEqual(len(result['points'][key]['measurements']),2)
        partial=next(iter(before['partial_points'].values()))
        self.assertEqual(result['points'][key]['measurements'][0],partial['measurements'][0])
        self.assertEqual(result['diagnostics'],before['diagnostics'])
        self.assertEqual(raw,{p.name:p.read_bytes() for p in attempt.iterdir()})
        self.assertFalse((attempt.parent/'pass.json').exists())
        correction=next(iter(result['corrections'].values()))
        self.assertEqual(correction['previous_partial'],partial)
        self.assertEqual(correction['record_sha256'],digest({k:v for k,v in correction.items() if k!='record_sha256'}))
        self.assertEqual(correction['raw_files']['trace.sqlite'],sha(attempt/'trace.sqlite'))
        self.assertFalse(correction['performance_admitted'])
        self.assertEqual(tables(result['points'])[1],{})
        with self.assertRaisesRegex(ValueError,'unresolved'):reparse_attempt(result,attempt)

    def test_reparse_rejects_bad_identity_process_and_numerics_without_changes(self):
        result,attempt,parsed,key=self.reparse_fixture()
        files={p.name:p.read_text() for p in attempt.glob('*.json')}
        for variant in ('authority','workload','recipe','process','proof-recipe','proof-workload',
                        'helper','roles','oracle','weight','zero','coverage','existing','unknown-candidate','arithmetic'):
            trial=copy.deepcopy(result)
            for name,body in files.items():(attempt/name).write_text(body)
            request=json.loads(files['request.json']);proof=json.loads(files['cases.json'])
            process=json.loads(files['profile.process.json'])
            if variant=='authority':request['authority']='wrong'
            elif variant=='workload':request['workload']['m']+=1
            elif variant=='recipe':request['cases'][0]['recipe']['parameters'][0][1]=99
            elif variant=='process':process['returncode']=1
            elif variant=='proof-recipe':proof[0]['recipe']['parameters'][0][1]=99
            elif variant=='proof-workload':proof[0]['workload']['m']+=1
            elif variant=='helper':proof[0]['helper_mode']='legacy'
            elif variant=='roles':proof[0]['roles']=['changed']
            elif variant=='oracle':proof[0]['correctness'][0]['relative_l1_error']=.1
            elif variant=='weight':proof[0]['weight_bits'][0]['bad']=1
            elif variant=='zero':proof[0]['zero_a']='FAIL'
            elif variant=='coverage':proof=[]
            elif variant=='existing':next(iter(trial['partial_points'].values()))['missing']=[]
            elif variant=='arithmetic':proof[0]['receipt']['arithmetic']['accumulator']='F16'
            else:trial['candidates']={}
            for name,data in [('request.json',request),('cases.json',proof),('profile.process.json',process)]:
                (attempt/name).write_text(json.dumps(data))
            before=copy.deepcopy(trial)
            with self.subTest(variant=variant),patch('dev.prefill_asys.trace.parse',return_value=parsed):
                with self.assertRaises(ValueError):reparse_attempt(trial,attempt)
            self.assertEqual(trial,before)

    def test_reparse_uses_the_same_strict_component_sum_validation(self):
        result,attempt,parsed,_=self.reparse_fixture();before=copy.deepcopy(result)
        parsed['cases'][0]['kernel_inventory'][0].update(samples_us=[2]*3,median_us=2)
        with patch('dev.prefill_asys.trace.parse',return_value=parsed):
            with self.assertRaisesRegex(ValueError,'component sum'):reparse_attempt(result,attempt)
        self.assertEqual(result,before)

    def test_reparse_keeps_other_missing_configs_unresolved(self):
        result,attempt,parsed,key=self.reparse_fixture()
        next(iter(result['partial_points'].values()))['missing'].append('other-missing-config')
        with patch('dev.prefill_asys.trace.parse',return_value=parsed):reparse_attempt(result,attempt)
        self.assertNotIn(key,result['points'])
        partial=next(iter(result['partial_points'].values()))
        self.assertEqual(partial['missing'],['other-missing-config'])
        self.assertEqual(len(partial['measurements']),2)


if __name__=='__main__':unittest.main()

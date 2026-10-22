import unittest
import copy
from dataclasses import asdict
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tempfile

from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.review_prefill_cohort import ingest, merge, tables

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
        exact,buckets=tables(points)
        self.assertEqual({r['candidate'] for r in exact.values()},{'a','b'})
        self.assertEqual(len(buckets),1)
        b=next(iter(buckets.values()))
        self.assertEqual(b['candidate'],'b')
        self.assertAlmostEqual(b['worst_regret_pct'],10)
        self.assertFalse(b['holdout_validated'])

    def test_missing_bucket_measurement_is_not_estimated(self):
        points=dict([self.point(5120,dict(a=10)),self.point(6144,dict(b=100))])
        _,b=tables(points)
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
        exact,buckets=tables(dict([one,two]))
        self.assertEqual(exact[two[0]]['status'],'REMEASURE_REQUIRED')
        self.assertEqual(next(iter(buckets.values()))['status'],'NO_COMMON_MEASURED_CONFIG')


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


if __name__=='__main__':unittest.main()

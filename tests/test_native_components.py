"""Host-only regression tests. None of these are GPU performance evidence."""
import copy
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dev.native_components.accounting import checked_component, compose, comparison, reuse_key
from dev.native_components.plan import plan
from dev.native_components.build import between, wrap_providers, adapter_source
from dev.native_components.history import costs, check_cohort, HISTORICAL_COHORT
from tools import run_native_components as runner
from tools import run_native_components_multi as multi
from quactlize.runtime.tuning import digest
from quactlize.runtime.compiler import sha


def sample(role='producer',value=2):
    provider=role=='provider'
    return dict(role=role,correctness='PASS',samples_us=[[value]*3 for _ in range(3)],
        first_use_excluded=True,host_launch_cost_added=False,identity={'source':'host-test'},cache='TEST',
        timing='ORDINARY_PROVIDER_EVENTS'if provider else 'ISOLATED_KERNEL_GRAPH_EVENTS',
        cuda_graph=not provider,internal_reduction_included=provider,timed_kernel_types=None if provider else 1)


class AccountingTest(unittest.TestCase):
    def test_sum_and_separate_helper_cost(self):
        records={'a':sample('adapter',1),'b':sample('producer',10),'c':sample('reducer',2)}
        calls=[dict(id=k,component=k,covers=[k])for k in records]
        result=compose(calls,records,required=records,inventory_complete=True)
        self.assertEqual(result['sum_us'],13)
        self.assertFalse(result['model_tpot_admitted'])

    def test_missing_is_not_zero(self):
        calls=[dict(id='x',component='missing',covers=['core'])]
        got=compose(calls,{},required=['core','helper'],inventory_complete=True)
        self.assertIsNone(got['sum_us']);self.assertEqual(got['missing_work'],['helper'])

    def test_incomplete_inventory_cannot_pass(self):
        got=compose([dict(id='x',component='a',covers=['core'])],{'a':sample()},required=['core'],inventory_complete=False)
        self.assertIsNone(got['sum_us'])

    def test_two_real_invocations_are_not_deduplicated(self):
        calls=[dict(id=str(i),component='shared',covers=['work'+str(i)])for i in (0,1)]
        self.assertEqual(compose(calls,{'shared':sample()},required=['work0','work1'],inventory_complete=True)['sum_us'],4)

    def test_fused_reducer_finish_must_not_be_charged_twice(self):
        calls=[dict(id='a',component='fused',covers=['reduce','finish']),
               dict(id='b',component='finish',covers=['finish'])]
        with self.assertRaises(ValueError):compose(calls,{},required=[],inventory_complete=True)

    def test_provider_is_not_a_single_kernel(self):
        row=sample('provider');self.assertEqual(checked_component(row)['median_us'],2)
        row['cuda_graph']=True
        with self.assertRaises(ValueError):checked_component(row)

    def test_timing_negatives(self):
        for field,value in [('correctness','FAIL'),('role','unclassified'),('first_use_excluded',False),
                            ('host_launch_cost_added',True),('timed_kernel_types',3),('identity',{}),('cache',None)]:
            row=sample();row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):checked_component(row)
        for value in (0,-1,float('inf'),float('nan')):
            with self.subTest(value=value),self.assertRaises(ValueError):checked_component(sample(value=value))

    def test_cross_run_medians_never_prove_no_regression(self):
        value=dict(status='COMPLETE',sum_us=4)
        self.assertFalse(comparison(value,value)['no_regression_admitted'])

    def test_reuse_requires_full_semantics(self):
        values=dict(implementation='a',device={'pci':'0'},sdk='b',shape=[1,2,3],io='f32',
                    router={'id':'x'},launch={'split':1},cache='cold',timing='graph')
        first=reuse_key(**values)
        for key in values:
            altered=dict(values);altered[key]=['changed']
            self.assertNotEqual(first,reuse_key(**altered))
            altered[key]=None
            with self.assertRaises(ValueError):reuse_key(**altered)


class PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p=plan()

    def test_all_measured_shapes(self):
        self.assertEqual(len(self.p['points']),1544)
        self.assertEqual(Counter(r['cohort']for r in self.p['points']),{'decode':1114,'prefill':430})
        self.assertEqual(self.p['summary']['historical_smallm_summaries'],961)
        self.assertEqual(self.p['summary']['pending_counts']['IMPORT_RAW_SCREEN_NOT_AUTOMATIC_REMEASUREMENT'],153)

    def test_no_redundant_casts_added_to_simt(self):
        rows=[r for r in self.p['points']if r['cohort']=='decode'and r['selected_recipe']['kind']in ('q4','simt')]
        self.assertTrue(rows)
        self.assertTrue(all('INPUT_ADAPTER'not in r['required_components']for r in rows))

    def test_prefill_route_coverage(self):
        self.assertEqual(sum('MEASURE_GPU_ROW_MAP_CREATION'in r['pending']for r in self.p['points']),150)
        self.assertEqual(sum('MEASURE_FOUR_BYTE_ERROR_RESET'in r['pending']for r in self.p['points']),186)
        self.assertTrue(all('AUDIT_ACTIVE_DIRECTORY_AND_ERROR_RESET'not in r['pending']for r in self.p['points']))

    def test_supplement_is_six_jobs_not_336_remeasurements(self):
        jobs,bindings=runner.helper_jobs(self.p['points'])
        self.assertEqual(Counter(r['kind']for r in jobs),dict(routing=5,reset=1))
        self.assertEqual(sum(map(len,bindings.values())),336)
        self.assertEqual(sorted(j['workload']['m']for j in jobs if j['kind']=='routing'),[128,512,1024,2048,4096])

    def test_all_historical_costs_are_present_and_digest_checked(self):
        data=json.loads(Path('docs/measurements/kpack_native_history_20261019.json').read_text())
        self.assertEqual(data['sha256'],digest({k:v for k,v in data.items()if k!='sha256'}))
        self.assertEqual(data['source_plan'],self.p['sha256'])
        self.assertEqual({p['id']for p in data['points']},{p['id']for p in self.p['points']})
        self.assertTrue(all(p['sum_us']>0 and p['components']for p in data['points']))

    def test_no_historical_cross_device_or_sdk_comparison(self):
        cohort=HISTORICAL_COHORT
        device=dict(sdk=cohort['sdk'],l2={'bytes':cohort['l2_bytes']},
                    device={k:v for k,v in cohort.items()if k not in ('sdk','l2_bytes')})
        self.assertFalse(check_cohort(device)['paired'])
        device['device']['compute_units']=64
        with self.assertRaisesRegex(ValueError,'compute_units'):check_cohort(device)
        device['device']['compute_units']=72;device['sdk']='other-sdk'
        with self.assertRaisesRegex(ValueError,'sdk'):check_cohort(device)

    def test_no_caller_branch_or_legacy_bundle_requirement(self):
        self.assertFalse(self.p['llama_source_patch']);self.assertFalse(self.p['llama_branch_pin'])
        self.assertFalse(self.p['old_six_library_bundle_required'])

    def test_missing_archive_is_not_remeasured_implicitly(self):
        pending=next(r for r in self.p['points']if r['quactlize']['status']=='IMPORT_UNCHANGED_ARM_SCREEN')
        with self.assertRaises(ValueError):costs(dict(points=[pending],recovered_unchanged={}))

    def test_cli_plan_requires_no_sdk_or_caller(self):
        with tempfile.TemporaryDirectory()as td,redirect_stdout(io.StringIO()):
            with patch('sys.argv',['test','--plan-only','--output',td]):self.assertEqual(runner.main(),0)
            self.assertEqual(len(json.loads((Path(td)/'plan.json').read_text())['points']),1544)


class SourceAdapterTest(unittest.TestCase):
    def test_provider_calls_are_not_captured(self):
        text='\n'.join(name+'(h, f(x, y), z)'for name in (
            'cublasSgemm','cublasGemmEx','cublasGemmStridedBatchedEx','cublasGemmBatchedEx'))
        result,counts=wrap_providers(text)
        self.assertEqual(len(counts),4);self.assertEqual(result.count('nc::provider('),4)
        source=Path('dev/native_components/record.hpp').read_text()
        self.assertLess(source.index('active->end();'),source.index('auto rc=stage.provider();'))
        self.assertLess(source.index('auto rc=stage.provider();'),source.index('active->begin();'))

    def test_changed_source_seam_is_rejected(self):
        with self.assertRaises(ValueError):between('a a b','a','b')
        with self.assertRaises(ValueError):wrap_providers('cublasGemmEx(a)')

    def test_real_dense_mr_native_source_adapter(self):
        llama=Path('/root/autodl-tmp/llama-kpack-dense')
        if not llama.is_dir():self.skipTest('optional local native checkout')
        generated,receipt=adapter_source(llama)
        original=(llama/'ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
        exact=between(original,'static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,',
                      'static bool ggml_cuda_compute_forward(')
        self.assertIn(exact,generated)
        self.assertFalse(receipt['selector_changed']);self.assertFalse(receipt['arithmetic_changed'])

    def test_provider_retains_raw_call_samples_and_full_cold_ring(self):
        text=Path('dev/native_components/bridge.cuh').read_text()
        self.assertIn('us[i]=ms*1000/divisor',text)
        self.assertIn('copies)*((std::max(11,int(copies*2))+int(copies)-1)/int(copies))',text)
        self.assertIn('pools.push_back(std::move(ctx.pools',text)


class ContinuationTest(unittest.TestCase):
    def test_resume_never_retimes_or_relabels_valid_slow_measurement(self):
        row=dict(id='point',workload={})
        with tempfile.TemporaryDirectory()as td:
            raw=Path(td)/'point.json';saved=Path(td)/'saved.json'
            data=dict(row,components=[sample(value=99)])
            runner.atomic_json(raw,data)
            record=dict(status='MEASURED',path=str(raw),sha256=sha(raw),sum_us=99)
            runner.atomic_json(saved,record);before=saved.read_bytes()
            with patch.object(runner,'validated',return_value=data):
                self.assertEqual(runner.load_measured(saved,{},row),record)
                self.assertEqual(saved.read_bytes(),before)
                runner.atomic_json(saved,dict(record,sum_us=1))
                with self.assertRaisesRegex(ValueError,'sum'):runner.load_measured(saved,{},row)
            runner.atomic_json(saved,dict(status='FAIL'))
            self.assertIsNone(runner.load_measured(saved,{},row))

    def test_fixture_failure_is_reported_for_its_family(self):
        rows=[dict(id='a',workload={}),dict(id='b',workload={})]
        with tempfile.TemporaryDirectory()as td,redirect_stdout(io.StringIO()):
            with patch.object(runner,'family',return_value='family'),\
                 patch('dev.native_components.fixture.weights',side_effect=ValueError('bad fixture')):
                results=runner.run_family(SimpleNamespace(output=Path(td)),Path('/fake'),{},rows,{})
            self.assertEqual(set(results),{'a','b'})
            self.assertTrue(all(r['reason']=='FAMILY_FIXTURE_FAILED'for r in results.values()))

    def test_point_abort_preserves_completed_and_restarts_peers(self):
        rows=[dict(id=k,workload={})for k in ('a','b','c')];calls=[]
        with tempfile.TemporaryDirectory()as td:
            args=SimpleNamespace(output=Path(td))
            class Child:
                def __init__(self,cmd,stdout,**kwargs):
                    cfg=json.loads(Path(cmd[1]).read_text());calls.append([p['id']for p in cfg['points']])
                    self.returncode=0
                    for row in cfg['points']:
                        stdout.write('NATIVE_COMPONENT_START point='+row['id']+'\n')
                        if row['id']=='b':self.returncode=1;break
                        runner.atomic_json(Path(cmd[2])/(row['id']+'.json'),dict(row,components=[sample()],authority={}))
                def poll(self):return self.returncode
            with patch.object(runner,'family',return_value='family'),\
                 patch('dev.native_components.fixture.weights',return_value=(Path('/fake'),{})),\
                 patch('dev.native_components.fixture.activation',return_value={}),\
                 patch.object(runner,'environment',return_value={}),\
                 patch.object(runner,'validated',side_effect=lambda p,a:json.loads(p.read_text())),\
                 patch.object(runner.subprocess,'Popen',Child),redirect_stdout(io.StringIO()):
                results=runner.run_family(args,Path('/fake'),{},rows,{})
            self.assertEqual(calls,[['a','b','c'],['c']])
            self.assertEqual({k:v['status']for k,v in results.items()},dict(a='MEASURED',b='FAIL',c='MEASURED'))

    def test_family_setup_failure_does_not_repeat_all_peers(self):
        rows=[dict(id='a'),dict(id='b')]
        with tempfile.TemporaryDirectory()as td:
            args=SimpleNamespace(output=Path(td))
            calls=[]
            class Child:
                def __init__(self,cmd,**kwargs):calls.append(cmd);self.returncode=1
                def poll(self):return self.returncode
            with patch.object(runner,'family',return_value='family'),\
                 patch('dev.native_components.fixture.weights',return_value=(Path('/fake'),{})),\
                 patch('dev.native_components.fixture.activation',return_value={}),\
                 patch.object(runner,'environment',return_value={}),\
                 patch.object(runner.subprocess,'Popen',Child):
                for row in rows:row['workload']={}
                results=runner.run_family(args,Path('/fake'),{},rows,{})
            self.assertEqual(len(calls),1)
            self.assertEqual(set(results),{'a','b'})
            self.assertTrue(all(r['reason']=='FAMILY_SETUP_FAILED'for r in results.values()))


class FixtureTest(unittest.TestCase):
    def test_raw_bytes_and_inputs_equal_historical_fixture(self):
        import numpy as np
        from dev.native_components.fixture import weights,activation
        from dev.bf16_compute.fixture import raw_weight
        from dev.smallm_closure.fixture import Weights
        with tempfile.TemporaryDirectory()as td:
            for q in (8,10,11,12,13,14):
                w=dict(qtype=q,n=256,k=512,experts=2,m=2,channels=1,compute='bf16')
                path,receipt=weights(w,Path(td)/str(q))
                raw=b''.join(raw_weight(q,256,512,710+e*31+q*1009)[0].tobytes()for e in range(2))
                self.assertEqual(path.read_bytes(),raw)
                self.assertEqual(weights(w,Path(td)/str(q))[1],receipt)
                old=Weights(q,256,512,2)
                values=activation(dict(cohort='decode',workload=w))
                for repeat in (0,1):
                    got=np.asarray(values['coefficients'][repeat],dtype='f4').reshape(2,4)[:,values['categories']]
                    want=old.inputs(dict(tokens=2,channels=1,topk=1,mode=0,compute='bf16'),repeat)['a']
                    self.assertTrue(np.array_equal(got,want))


class CompactArchiveTest(unittest.TestCase):
    def test_only_results_identity_and_failure_logs(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);output=root/'gpu-1'
            for d in ('results','helpers','attempts/x','build','fixtures'):(output/d).mkdir(parents=True,exist_ok=True)
            log=output/'attempts/x/run.log';log.write_text('expected diagnostic')
            runner.atomic_json(output/'helpers/helper.json',dict(status='FAIL',log=str(log)))
            runner.atomic_json(output/'results/point.json',dict(status='MEASURED'))
            (output/'fixtures/weights.raw').write_bytes(b'not in archive')
            (output/'build/adapter.o').write_bytes(b'not in archive')
            (output/'build/adapter-build.log').write_text('build diagnostic')
            with tarfile.open(multi.pack(root,[output]))as archive:
                names=set(archive.getnames())
            self.assertIn('gpu-1/helpers/helper.json',names)
            self.assertIn('gpu-1/attempts/x/run.log',names)
            self.assertIn('gpu-1/build/adapter-build.log',names)
            self.assertFalse(any(name.endswith(('.raw','.o'))for name in names))


if __name__=='__main__':unittest.main()

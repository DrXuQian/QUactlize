import copy
import json
import lzma
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

from dev.prefill_asys import random_gate as gate
from dev.prefill_asys.finalists import case_identity, load, partition
from dev.prefill_asys.trace import parse_rows
from dev.prefill_point.bench import trace_contract
from quactlize.dispatch.dense_prefill import EVIDENCE
from quactlize.runtime.tuning import digest
from tools.review_prefill_cohort import confirmations
from tools.run_dense_random_gate import finish
from tools.run_prefill_model_asys import confirmation_jobs
from tools.run_prefill_point import native_stage_names


def seal(value, field='sha256'):
    value[field]=digest({k:v for k,v in value.items() if k!=field})
    return value


class RandomDenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            cls.plan=gate.make_plan(Path(td))  # Runs the real host C++ selector.
        cls.policy=json.loads(EVIDENCE.read_text())

    def test_seeded_geometry_not_winner_latency_selection(self):
        original=gate.draws(self.policy,20261027)
        changed=copy.deepcopy(self.policy)
        for p in changed['points']:
            p.update(winner='irrelevant',us=1e30)
        self.assertEqual(original,gate.draws(changed,20261027))
        self.assertNotEqual(original,gate.draws(self.policy,20261028))
        self.assertEqual(len(original),20)
        anchors={p['shape'][1] for p in self.policy['points']}
        families={(p['shape'][0],*p['shape'][2:]) for p in self.policy['points']}
        for q in (8,11,12,13,14):
            rows=[w for w in original if w.qtype==q]
            self.assertEqual(len(rows),4)
            for lo,hi in gate.BANDS:
                self.assertEqual(sum(lo<=w.m<=hi for w in rows),1)
        for w in original:
            self.assertNotIn(w.m,anchors)
            self.assertIn((w.qtype,w.n,w.k),families)
        for seed in (-1,2**32,True):
            with self.assertRaises(ValueError):gate.draws(self.policy,seed)

    def test_actual_bucket_cpp_and_python_agree_not_a_sweep(self):
        self.assertEqual(load(self.plan),self.plan)
        for p in self.plan['points']:
            self.assertEqual(p['selection']['policy'],21)
            self.assertEqual(p['selection']['coverage'],'BUCKET_PREDICTION')
            self.assertEqual(len(p['cases']),1)
            order=[cases[0]['arm'] for cases,_ in confirmation_jobs(p)]
            self.assertEqual(order,['native-fp16',p['cases'][0]['arm'],p['cases'][0]['arm'],'native-fp16'])
            if p['workload']['m']>64:self.assertNotEqual(p['cases'][0]['arm'],'fq')
        shards=[partition(self.plan['points'],i,4) for i in range(4)]
        self.assertEqual(sorted(p['id'] for s in shards for p in s),sorted(p['id'] for p in self.plan['points']))

    def test_existing_box_worker_accepts_frozen_random_plan(self):
        from tools.run_prefill_model_asys import main
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=root/'plan.json';plan.write_text(json.dumps(self.plan))
            with patch.object(sys,'argv',['run_prefill_model_asys.py','--finalist-plan',str(plan),
                    '--output',str(root/'worker'),'--configs-per-batch','1','--plan-only']):
                self.assertEqual(main(),0)
            frozen=json.loads((root/'worker/plan.json').read_text())
            self.assertEqual(frozen['schema'],gate.SCHEMA)
            self.assertEqual({p['id'] for p in frozen['points']},{p['id'] for p in self.plan['points']})

    def test_resealed_contract_mutations_still_fail(self):
        changes=[lambda p:p.update(seed=42),lambda p:p.update(qtypes=[8]),
                 lambda p:p.update(metric='MINIMUM'),lambda p:p.update(allowed_slowdown_pct=2),
                 lambda p:p.update(precision='F16_ACC'),lambda p:p.update(excluded_qtypes=[]),
                 lambda p:p['points'].pop(),lambda p:p['inputs'].update(helper_policy_sha256='stale'),
                 lambda p:p['points'][0].update(native_comparison=False),
                 lambda p:p['points'][0]['selection'].update(policy=20),
                 lambda p:p['points'][0]['cases'][0].update(roles=['optional'])]
        for change in changes:
            plan=copy.deepcopy(self.plan);change(plan);seal(plan)
            with self.subTest(change=change),self.assertRaises(ValueError):gate.validate(plan)

    def review(self, slow=False):
        revision=dict(quactlize_commit='a'*40,quactlize_tree='b'*40,actlize_commit='c'*40,
                      snapshot='COMMITTED_ARCHIVE')
        authority=dict(source_revision=revision);aid=digest(authority)
        result=dict(schema='quactlize.prefill-component-review.v1',authorities={aid:authority},
                    candidates={},components={},points={},partial_points={})
        for index,spec in enumerate(self.plan['points']):
            rows=[]
            for native in (True,False):
                case=dict(arm='native-fp16') if native else case_identity(spec['cases'][0])
                cid=digest(case);result['candidates'][cid]=case
                arithmetic=dict(input='F32',output='F32',compute='F16',accumulator='FP32')
                if native:arithmetic['native_activation_quantization']=False
                values=[100.,100.,100.] if native else [90.,90.,90.]
                # A minimum/median would wrongly admit this candidate.
                if slow and index==0 and not native:values=[40.,40.,240.]
                times=[[1.]*3,[v-4 for v in values],[2.]*3,[1.]*3]
                components=[]
                for phase in ('input','main','reducer','output'):
                    definition=dict(name=('native-' if native else 'qz-')+phase,phase=phase,
                                    workload=spec['workload'],authority=aid)
                    key=digest(definition);result['components'][key]=definition;components.append(key)
                captures={}
                for confirm in (0,1):
                    captures[confirm]=dict(status='PASS',candidate=cid,arm=case['arm'],
                        case_id='native-fp16' if native else spec['cases'][0]['id'],
                        gpu_sum_us=statistics.median(values),complete_samples_us=values,
                        components=components,component_samples_us=times,
                        phases_us={p:statistics.median(t) for p,t in zip(('input','main','reducer','output'),times)},
                        rounds_us=values,arithmetic=dict(a='F16',b='F16',c='F16',accumulator='FP32'),
                        numerical='PASS' if native else [dict(nonfinite=0,relative_l1_error=.001)],
                        weight_bits=[dict(bad=0)],omitted_producer='DETECTED',
                        random_gate_evidence=dict(raw_sha256=['same-weight-a','same-weight-b'],arithmetic=arithmetic))
                rows.append(confirmations(captures,2))
            result['points'][spec['id']]=dict(workload=spec['workload'],authority=aid,measurements=rows)
        return seal(result,'review_sha256')

    def test_compare_means_includes_helpers_and_keeps_single_regression(self):
        result=gate.compare(self.plan,self.review(slow=True))
        self.assertEqual(result['status'],'REGRESSION')
        self.assertEqual(result['counts'],dict(PASS=19,REGRESSION=1))
        row=result['points'][0]
        self.assertAlmostEqual(row['quactlize_us'],320/3)
        self.assertAlmostEqual(row['delta_us'],20/3)
        self.assertEqual(row['quactlize_commit'],'a'*40)
        self.assertEqual([c['phase'] for c in row['components']['native']],['input','main','reducer','output'])
        self.assertFalse(result['all_shapes_no_regression_proven'])
        self.assertEqual(gate.compare(self.plan,self.review())['status'],'PASS')

    def test_missing_points_remain_in_denominator(self):
        review=self.review();key=self.plan['points'][0]['id'];review['points'].pop(key)
        review['partial_points']['partial']=dict(point_id=key,missing=['native-fp16'])
        result=gate.compare(self.plan,seal(review,'review_sha256'))
        self.assertEqual(result['status'],'INCOMPLETE');self.assertEqual(len(result['points']),20)
        self.assertEqual(result['points'][0]['partial'][0]['missing'],['native-fp16'])

    def test_fixture_identity_numerics_arithmetic_and_commit_are_required(self):
        for change in ('weight','precision','negative','numeric','commit'):
            review=self.review();point=next(iter(review['points'].values()))
            cap=point['measurements'][1]['confirmation_captures'][1]
            if change=='weight':cap['random_gate_evidence']['raw_sha256']=['different']
            elif change=='precision':cap['random_gate_evidence']['arithmetic']['accumulator']='F16'
            elif change=='negative':cap['omitted_producer']='NOT_TESTED'
            elif change=='numeric':cap['numerical'][0]['nonfinite']=1
            else:
                auth=review['authorities'].pop(point['authority']);auth['source_revision']={}
                aid=digest(auth);review['authorities'][aid]=auth
                # The authority is rejected before inspecting component bindings.
                point['authority']=aid
            with self.subTest(change=change),self.assertRaises(ValueError):
                gate.compare(self.plan,seal(review,'review_sha256'))

    def test_native_fp16_accumulator_is_not_a_valid_fp32_reference(self):
        review=self.review()
        for row in next(iter(review['points'].values()))['measurements']:
            if row['case_id']=='native-fp16':
                for cap in row['confirmation_captures']:
                    cap['random_gate_evidence']['arithmetic']['accumulator']='F16'
        with self.assertRaisesRegex(ValueError,'native arithmetic'):
            gate.compare(self.plan,seal(review,'review_sha256'))

    def test_compact_packet_keeps_evidence_not_binary_reports(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td);(output/'run').mkdir()
            (output/'run/summary-1.json').write_text(json.dumps(dict(status='SCREEN_COMPLETE')))
            with lzma.open(output/'run/review-1.json.xz','wt') as f:json.dump(self.review(slow=True),f)
            self.assertEqual(finish(output,self.plan),2)
            packet=next(output.glob('review-*.json.xz'))
            with lzma.open(packet,'rt') as f:data=json.load(f)
            self.assertEqual(data['summary']['counts']['REGRESSION'],1)
            self.assertIn('evidence',data)
            self.assertEqual(data['raw_reports'],'RETAINED_ON_BOX_NOT_EMBEDDED')
            self.assertTrue((output/'comparison.csv').is_file())


class NativeDirectTraceTests(unittest.TestCase):
    def fixture(self, reducer=False):
        nodes=[('quantize_q8_1','adapter'),('mul_mat_q','producer')]
        if reducer:nodes.append(('fixup','reducer'))
        inventory=[dict(name=name+'(...)',symbol='_Z'+name,entry='KERNEL',role=role,
                        grid=[8,1,1],block=[128,1,1],shared_bytes=2048) for name,role in nodes]
        contract=trace_contract(2)
        case=dict(case_id='native-fp16',arm='native-fp16',trace=contract,
                  stages=native_stage_names(inventory,random_dense_probe=True),
                  native_direct_inventory=inventory)
        rows=[]
        for i in range(contract['captured_calls']):
            for j,node in enumerate(inventory):
                tick=(i*len(inventory)+j)*1100
                rows.append(dict(kind='kernel',start=tick,end=tick+1000,
                    **{k:node[k] for k in ('name','symbol','grid','block','shared_bytes')}))
        return rows,case

    def test_direct_native_without_output_cast_counts_all_components(self):
        for reducer in (False,True):
            rows,case=self.fixture(reducer)
            result=parse_rows(rows,[case])[0]
            self.assertEqual(result['calls'],24)
            self.assertEqual(result['gpu_sum_us'],3 if reducer else 2)
            self.assertEqual(result['phases_us']['input'],1)
            self.assertEqual(result['phases_us'].get('reducer',0),int(reducer))
            self.assertFalse(result['cuda_graph'])
            with self.assertRaises(ValueError):native_stage_names(case['native_direct_inventory'])

    def test_missing_extra_or_launch_drift_is_not_timed(self):
        for field in ('symbol','grid','block','shared_bytes','missing','extra'):
            rows,case=self.fixture()
            if field=='missing':rows.pop()
            elif field=='extra':rows.append(rows[-1])
            else:rows[-1][field]='bad' if field=='symbol' else 0 if field=='shared_bytes' else [1,1,1]
            with self.subTest(field=field),self.assertRaises(ValueError):parse_rows(rows,[case])
        rows,case=self.fixture();case['native_direct_inventory'][0]['role']='unclassified'
        with self.assertRaises(ValueError):parse_rows(rows,[case])


if __name__=='__main__':unittest.main()

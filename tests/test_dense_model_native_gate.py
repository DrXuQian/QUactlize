import copy
import json
import lzma
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dev.prefill_asys import model_gate as gate
from dev.prefill_asys.finalists import load, partition
from quactlize.runtime.tuning import digest
from tools.run_dense_model_native_gate import check_models, coverage_csv, detail_csvs
from tools.run_dense_random_gate import finish
from tools.run_prefill_model_asys import confirmation_jobs


def seal(p):
    p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
    return p


def fixtures():
    def tensor(name,q,n,k):return dict(name=name,qtype=q,dims_gguf=[k,n])
    shared=tensor('blk.0.attn_k.weight',12,1024,5120)
    return {
        'qwen35-35b-a3b':[
            tensor('blk.0.ffn_gate_shexp.weight',8,512,2048),
            tensor('blk.1.ffn_gate_shexp.weight',8,512,2048),
            tensor('blk.0.ffn_down_shexp.weight',8,2048,512),shared,
            tensor('blk.0.attn_gate.weight',8,48,5120),
            dict(name='blk.0.ffn_down_exps.weight',qtype=12,dims_gguf=[512,2048,256]),
            tensor('blk.0.attn_floating.weight',1,512,2048),
            tensor('output.weight',8,151936,2048)],
        'qwen3-32b':[
            shared,tensor('blk.0.ffn_gate.weight',12,25600,5120),
            tensor('blk.0.ffn_down.weight',14,5120,25600)]}


def loader(name):
    tensors=fixtures()[name]
    return [dict(path='/models/'+name+'.gguf',size=1234,header=dict(tensors=tensors))],tensors


class ModelNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            cls.plan=gate.make_plan({k:k for k in gate.MODEL_MS},Path(td),loader=loader)

    def test_actual_cpp_selector_and_complete_model_m_inventory(self):
        p=self.plan
        self.assertEqual(load(p),p)
        self.assertEqual(p['counts']['points'],17)
        self.assertEqual(p['counts']['configs'],17)
        self.assertEqual(p['counts']['captures'],68)
        self.assertEqual(p['counts']['timed_arms'],68)
        for name,ms in gate.MODEL_MS.items():
            points=[v for v in p['points'] if any(c['model']==name for c in v['consumers'])]
            self.assertEqual({v['workload']['m'] for v in points},set(ms))
            self.assertEqual(len(points),3*len(ms))
        shared=[p for p in p['points'] if len({c['model'] for c in p['consumers']})==2]
        self.assertEqual(len(shared),1)
        self.assertEqual(shared[0]['workload']['m'],5120)
        for point in p['points']:
            self.assertEqual(point['selection']['policy'],20)
            self.assertEqual(point['selection']['coverage'],'EXACT')
            arm=point['cases'][0]['arm'];self.assertNotEqual(arm,'fq')
            self.assertEqual([cs[0]['arm'] for cs,_ in confirmation_jobs(point)],
                             ['native-fp16',arm,arm,'native-fp16'])

    def test_moe_native_only_and_output_are_explicit_exclusions(self):
        reasons=[p['reason'] for p in self.plan['excluded']]
        self.assertIn('MOE_NOT_IN_DENSE_INTEGRATION',reasons)
        self.assertIn('NATIVE_FLOAT_OR_UNSUPPORTED_QUANT_TYPE',reasons)
        self.assertIn('NOT_LAYER_DENSE_MATRIX',reasons)
        self.assertEqual(sum(s.startswith('NATIVE_ONLY_KPACK_SHAPE_CONTRACT:') for s in reasons),5)

    def test_shards_keep_whole_family_and_preserve_every_point(self):
        shards=[partition(self.plan['points'],i,4) for i in range(4)]
        self.assertEqual(sorted(p['id'] for s in shards for p in s),sorted(p['id'] for p in self.plan['points']))
        owners={}
        for i,points in enumerate(shards):
            for p in points:
                w=p['workload'];key=(w['qtype'],w['n'],w['k'])
                self.assertEqual(owners.setdefault(key,i),i)

    def test_resealed_omissions_wrong_m_arithmetic_and_recipe_fail(self):
        edits=[lambda p:p['points'].pop(),lambda p:p['models'][0]['ms'].pop(),
            lambda p:p.update(precision='F16_ACC'),lambda p:p.update(allowed_slowdown_pct=1),
            lambda p:p['points'][0].update(native_comparison=False),
            lambda p:p['points'][0]['consumers'].pop(),lambda p:p['excluded'].pop(),
            lambda p:p['points'][0]['selection'].update(policy=21),
            lambda p:p['points'][0]['cases'][0]['helper_options'].update(vector_io=99),
            lambda p:p['points'][0]['cases'].append(copy.deepcopy(p['points'][0]['cases'][0]))]
        for edit in edits:
            p=copy.deepcopy(self.plan);edit(p)
            with self.subTest(edit=edit),self.assertRaises(ValueError):gate.validate(seal(p))

    def test_existing_worker_plan_only_accepts_m5120_and_one_candidate(self):
        from tools.run_prefill_model_asys import main
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);f=root/'plan.json';f.write_text(json.dumps(self.plan))
            with patch.object(sys,'argv',['run_prefill_model_asys.py','--finalist-plan',str(f),
                    '--output',str(root/'worker'),'--configs-per-batch','1','--plan-only']):
                self.assertEqual(main(),0)
            frozen=json.loads((root/'worker/plan.json').read_text())
            self.assertEqual(frozen['schema'],gate.SCHEMA)
            self.assertEqual(frozen['counts']['points'],17)
        source=(Path(__file__).parents[1]/'tools/run_prefill_model_asys.py').read_text()
        self.assertIn("'quactlize.dense-random-native.v1','quactlize.dense-model-native.v1'",source)

    def review(self,slow=False):
        from tests.test_dense_random_gate import RandomDenseTests
        return RandomDenseTests.review(SimpleNamespace(plan=self.plan),slow=slow)

    def test_single_regression_not_hidden_by_model_weighted_sum(self):
        d=gate.compare(self.plan,self.review(slow=True))
        self.assertEqual(d['status'],'REGRESSION')
        self.assertEqual(d['counts'],dict(REGRESSION=1,PASS=16))
        self.assertEqual(len(d['models']),6)
        self.assertFalse(d['model_tpot_admitted'])
        for model in d['models']:
            self.assertIsNotNone(model['delta_ms'])
            self.assertIn('NOT_ACTUAL_GRAPH_COUNTS',model['scope'])
        self.assertEqual(gate.compare(self.plan,self.review())['status'],'PASS')

    def test_missing_native_is_incomplete_not_zero_time(self):
        r=self.review();key=self.plan['points'][0]['id'];r['points'].pop(key)
        r['review_sha256']=digest({k:v for k,v in r.items() if k!='review_sha256'})
        d=gate.compare(self.plan,r)
        self.assertEqual(d['status'],'INCOMPLETE')
        self.assertEqual(d['counts']['INCOMPLETE'],1)
        self.assertTrue(any(m['native_dense_matrix_ms'] is None for m in d['models']))

    def test_compact_export_keeps_scope_and_nonzero_regression_rc(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'run').mkdir()
            with lzma.open(out/'run/review-1.json.xz','wt') as f:json.dump(self.review(slow=True),f)
            (out/'run/summary-1.json').write_text(json.dumps(dict(status='SCREEN_COMPLETE')))
            self.assertEqual(finish(out,self.plan),2)
            with lzma.open(next(out.glob('review-*.json.xz')),'rt') as f:d=json.load(f)
            self.assertEqual(d['schema'],'quactlize.dense-model-native-review.v1')
            self.assertEqual(d['summary']['scope'],gate.SCOPE)
            self.assertEqual(d['summary']['counts']['REGRESSION'],1)
            coverage_csv(out/'coverage.csv',self.plan)
            self.assertIn('5120',(out/'coverage.csv').read_text())
            detail_csvs(out,d['summary'])
            self.assertIn('native-reducer',(out/'components.csv').read_text())
            self.assertIn('NOT_ACTUAL_GRAPH_COUNTS',(out/'model-summary.csv').read_text())

    def test_resume_checks_both_model_headers(self):
        with patch('tools.run_dense_model_native_gate.headers',side_effect=loader):
            check_models(self.plan,{k:k for k in gate.MODEL_MS})
            p=copy.deepcopy(self.plan);p['models'][0]['files'][0]['size']+=1
            with self.assertRaisesRegex(ValueError,'resume GGUF'):check_models(p,{k:k for k in gate.MODEL_MS})

    def test_cli_plan_only_keeps_native_and_needs_no_sdk_or_llama(self):
        from tools.run_dense_model_native_gate import main
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'plan'
            with patch('tools.run_dense_model_native_gate.make_plan',return_value=self.plan),\
                 patch.object(sys,'argv',['run_dense_model_native_gate.py','--output',str(output),
                    '--sdk','/no-sdk','--llama-source','/no-llama','--gpus','1,7','--plan-only']),\
                 patch('tools.run_dense_model_native_gate.revision',side_effect=AssertionError('GPU preflight')):
                self.assertEqual(main(),0)
            plan=json.loads((output/'plan.json').read_text())
            self.assertEqual(plan,self.plan)
            self.assertTrue(all(p['native_comparison'] for p in plan['points']))


if __name__=='__main__':unittest.main()

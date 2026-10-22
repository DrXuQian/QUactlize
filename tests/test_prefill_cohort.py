import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dev.prefill_asys.cohort import bucket, collect, local_shapes, partition, tp_records, validate
from quactlize.dispatch.table import Workload
from tools.run_prefill_cohort_multi import gpu_ordinals, worker_environment


class CohortTests(unittest.TestCase):
    def tensors(self):
        return [dict(name='blk.0.attn_q.weight',qtype=12,dims_gguf=[5120,8192]),
                dict(name='blk.0.ffn_down.weight',qtype=14,dims_gguf=[25600,5120]),
                dict(name='blk.0.attn_gate.weight',qtype=8,dims_gguf=[5120,48]),
                dict(name='blk.0.ffn_down_exps.weight',qtype=13,dims_gguf=[512,2048,256]),
                dict(name='token_embd.weight',qtype=12,dims_gguf=[5120,151936])]

    def test_dedup_preserves_model_consumers_and_separate_moe(self):
        models=[dict(name=n,path='/model') for n in ('one','two')]
        doc=collect(models,[128,2048],loader=lambda p:([],self.tensors()))
        self.assertEqual(len(doc['points']),4)
        self.assertEqual(validate(doc),doc)
        self.assertEqual(len(doc['points'][0]['consumers']),2)
        self.assertTrue(any(v['reason']=='MOE_SEPARATE_COHORT' for v in doc['excluded']))
        self.assertTrue(any(v['reason'].startswith('NATIVE_ONLY') for v in doc['excluded']))
        self.assertTrue(all(v['call_count'] is None for v in doc['points']))
        self.assertTrue(all('bucket' not in v for v in doc['points']))
        self.assertEqual(doc['bucket_status'],'PENDING_USER_FEEDBACK')
        bad=copy.deepcopy(doc);bad['points'][0]['workload']['n']+=256
        with self.assertRaisesRegex(ValueError,'digest'):validate(bad)

    def test_tp4_missing_is_not_guessed(self):
        doc=collect([dict(name='ds',path='/ds',tp=4)],[2048],loader=lambda p:([],self.tensors()))
        self.assertEqual(doc['status'],'INCOMPLETE')
        self.assertFalse(doc['points'])
        with self.assertRaisesRegex(ValueError,'unresolved'):validate(doc)

    def test_tp_observed_n_k_replication_and_missing_rank(self):
        t=self.tensors()[0]
        for n,k,axis in ((2048,5120,'n'),(8192,1280,'k'),(8192,5120,'replicated')):
            rows={t['name']:{(r,12,n,k) for r in range(4)}}
            got,error=local_shapes(t,4,rows)
            self.assertIsNone(error)
            self.assertTrue(all(v[-1]==axis for v in got))
            rows[t['name']].pop()
            self.assertEqual(local_shapes(t,4,rows)[1],'TP_LOCAL_RECORDS_MISSING')
        bad={t['name']:{(r,12,2048,1280) for r in range(4)}}
        self.assertEqual(local_shapes(t,4,bad)[1],'TP_LOCAL_EXTENTS_REQUIRE_REVIEW')

    def test_tp_log_reads_local_ordinals_and_deduplicates_repeats(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'log'
            line='[quactlize-dense] tensor=blk.0.attn_q.weight (view) device=2 q=12 m=2048 n=2048 k=5120 route=0\n'
            p.write_text(line*2)
            rows,receipts=tp_records([p],4)
            self.assertEqual(rows['blk.0.attn_q.weight'],{(2,12,2048,5120)})
            self.assertEqual(len(receipts[0]['sha256']),64)
            with self.assertRaisesRegex(ValueError,'invalid local rank'):tp_records([p],2)

    def test_buckets_preserve_format_precision_and_dimensions(self):
        w=Workload(12,8192,5120,2048)
        for change in (dict(qtype=13),dict(compute='bf16'),dict(m=512),dict(n=1024)):
            self.assertNotEqual(bucket(w),bucket(Workload(**(asdict(w)|change))))
        self.assertEqual(bucket(w)['aspect'],'balanced')

    def test_family_shards_cover_once_and_keep_all_m_together(self):
        points=[]
        for n in (512,1024,2048,4096,8192,16384,32768):
            for m in (128,512,2048):
                w=Workload(12,n,5120,m)
                points.append(dict(id=w.key,workload=asdict(w),candidates=[{}]*8,
                                   champion_controls=[{}],helper_candidates=[{}]*16))
        shards=[partition(points,i,3) for i in range(3)]
        self.assertEqual(sorted(p['id'] for s in shards for p in s),sorted(p['id'] for p in points))
        for n in (512,1024,2048,4096,8192,16384,32768):
            self.assertEqual(sum(any(p['workload']['n']==n for p in s) for s in shards),1)
        self.assertEqual({p['id'] for p in shards[1]},
                         {p['id'] for p in partition(list(reversed(points)),1,3)})
        with self.assertRaises(ValueError):partition(points,0,8)

    def test_coordinator_physical_ordinals_and_frozen_worker_environment(self):
        self.assertEqual(gpu_ordinals('01,2,7'),['1','2','7'])
        for invalid in ('1,01','','1,','1,2,1','x','-1','1, 2'):
            with self.assertRaises(ValueError):gpu_ordinals(invalid)
        with patch.dict('os.environ',dict(RUN='/previous',PLAN_ONLY='1',POINT_ID='old',
                        PREVIOUS_RUN='/old-build',RESUME_RUN='/wrong',PPU_SDK='/sdk')):
            env=worker_environment(Path('/fresh'),'7',2,3,'pinned-sha')
        self.assertEqual((env['GPU'],env['SHARD'],env['SHARDS']),('7','2','3'))
        self.assertEqual(env['SOURCE_REV'],'pinned-sha')
        self.assertEqual(env['TC_CACHE'],'/fresh/tc-cache')
        self.assertEqual(env['PPU_SDK'],'/sdk')
        self.assertTrue(all(k not in env for k in ('RUN','PLAN_ONLY','POINT_ID','PREVIOUS_RUN','RESUME_RUN')))


if __name__=='__main__':unittest.main()

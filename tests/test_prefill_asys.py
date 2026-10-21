import copy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from dev.prefill_point.bench import WORKLOAD, SF, trace_contract
from dev.prefill_asys.plan import model_points
from dev.prefill_asys.trace import activities, parse_rows, parse
from tools.run_prefill_model_asys import profile_command, summarize


class PrefillAsysTests(unittest.TestCase):
    def header(self):
        ts=[dict(name='output.weight',dims_gguf=[5120,151936],qtype=14),
            dict(name='token_embd.weight',dims_gguf=[5120,151936],qtype=12)]
        for i in range(64):
            for name,q,n,k in [('attn_q',12,8192,5120),('attn_k',12,1024,5120),
                    ('attn_v',14 if i%2 else 12,1024,5120),('attn_output',12,5120,8192),
                    ('ffn_gate',12,25600,5120),('ffn_up',12,25600,5120),
                    ('ffn_down',14 if i%2 else 12,5120,25600)]:
                ts.append(dict(name=f'blk.{i}.{name}.weight',dims_gguf=[k,n],qtype=q))
        return dict(metadata={'general.architecture':'qwen3'},tensors=ts)

    def test_fixed_model_inventory_excludes_lookup_and_m1(self):
        h=self.header();points,omitted=model_points(h)
        self.assertEqual(len(points),7)
        self.assertEqual(sum(len(p['tensors']) for p in points.values()),445)
        self.assertEqual(len(omitted),5)
        self.assertTrue(all(p['workload']['m']==2048 for p in points.values()))
        for change in ('unknown','missing','moe'):
            bad=copy.deepcopy(h)
            if change=='unknown':bad['tensors'].append(dict(name='extra.weight',dims_gguf=[5120,1024],qtype=12))
            elif change=='missing':bad['tensors'].pop()
            else:bad['metadata']['general.architecture']='qwen3moe'
            with self.assertRaises(ValueError):model_points(bad)

    def rows(self,arm='sf',split=1):
        c=trace_contract(2)
        stages=['input','expansion','main','output']
        if arm=='native-fp16':
            names=['void dequantize_block_q4_K<__half>(...)',
                   'void convert_cont_ppu_kernel<float, __half>(...)',
                   'provider_compute(...)','provider_internal_reduce(...)',
                   'void convert_cont_ppu_kernel<__half, float>(...)']
            stages=['expansion','input','main','output']
        elif arm=='full-bf16':
            names=['MEMSET','quactlize::prefill::gather_bf16(...)',
                   'full_packed_exchange<12>(...)','provider_compute(...)',
                   'quactlize::prefill::finish_bf16(...)']
            stages=['error_reset',*stages]
        else:
            names=['convert_rows<true>(...)','sf_unit16<12>(...)',
                   'cutlass::device_kernel<GemmUniversal>(...)']
            if split>1:names.append('cutlass::Kernel<ReductionKernel>(...)')
            names.append('convert_rows<false>(...)')
        rows=[];clock=0
        for i in range(c['captured_calls']):
            for name in names:
                row=dict(start=clock,end=clock+1000,kind='kernel',name=name,symbol=name,
                         grid=[1,1,1],block=[256,1,1],shared_bytes=0,device=0,context=1,stream=2,pid=100)
                if name=='MEMSET': row.update(kind='memset',bytes=4,value=0)
                rows.append(row);clock+=1100
        case=dict(case_id='case',arm=arm,trace=c,stages=stages,recipe=dict(parameters=[['split',split]]))
        return rows,case

    def test_each_gpu_component_not_host_intervals(self):
        for arm in ('sf','native-fp16','full-bf16'):
            rows,case=self.rows(arm)
            parsed=parse_rows(rows,[case])[0]
            self.assertEqual(parsed['calls'],24)
            self.assertEqual(parsed['gpu_sum_us'],4 if arm=='sf' else 5)
            self.assertAlmostEqual(parsed['idle_gap_us'],.3 if arm=='sf' else .4)
            self.assertEqual(parsed['timing_method'],'ASYS_GPU_ACTIVITY')
            self.assertFalse(parsed['cuda_graph'])
            self.assertEqual(len(parsed['measured_rounds_us']),3)

    def test_split_reducer_counted_once(self):
        rows,c=self.rows(split=2);v=parse_rows(rows,[c])[0]
        self.assertEqual(v['phases_us']['reducer'],1)
        self.assertEqual(v['gpu_sum_us'],5)
        bad=[r for r in rows if 'ReductionKernel' not in r['name']]
        with self.assertRaisesRegex(ValueError,'no reducer'):parse_rows(bad,[c])

    def test_merge_two_captures_and_reject_missing_extra_or_drift(self):
        rows,a=self.rows();other,b=self.rows('full-bf16');b['case_id']='second'
        for r in other:r['start']+=10**9;r['end']+=10**9
        self.assertEqual(len(parse_rows(rows+other,[a,b])),2)
        for bad in (rows[:-1],rows+[rows[-1]],rows[1:]):
            with self.assertRaises(ValueError):parse_rows(bad,[a])
        drift=copy.deepcopy(rows);drift[-2]['grid']=[2,1,1]
        with self.assertRaisesRegex(ValueError,'inventory changed'):parse_rows(drift,[a])

    def test_no_gpu_table_never_uses_runtime_api(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'trace.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE HGPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER)')
                db.execute('INSERT INTO HGPTI_ACTIVITY_KIND_RUNTIME VALUES(1,90000)')
            with self.assertRaisesRegex(ValueError,'no GPU kernel'):activities(path)

    def test_sqlite_kernel_and_reset_roundtrip(self):
        rows,case=self.rows('full-bf16');case.update(status='PASS',correctness=[dict(nonfinite=0,relative_l1_error=.001)])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'trace.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE StringIds(id INTEGER,value TEXT)')
                db.execute('CREATE TABLE HGPTI_ACTIVITY_KIND_KERNEL(start INT,end INT,demangledName INT,'
                    'mangledName INT,gridX INT,gridY INT,gridZ INT,blockX INT,blockY INT,blockZ INT,'
                    'staticSharedMemory INT,dynamicSharedMemory INT,deviceId INT,contextId INT,streamId INT,globalPid INT)')
                db.execute('CREATE TABLE HGPTI_ACTIVITY_KIND_MEMSET(start INT,end INT,bytes INT,value INT,'
                    'deviceId INT,contextId INT,streamId INT,globalPid INT)')
                for i,r in enumerate(rows):
                    if r['kind']=='kernel':
                        db.execute('INSERT INTO StringIds VALUES(?,?)',(i,r['name']))
                        db.execute('INSERT INTO HGPTI_ACTIVITY_KIND_KERNEL VALUES('+','.join('?'*16)+')',
                            (r['start'],r['end'],i,i,*r['grid'],*r['block'],0,0,0,1,2,100))
                    else:
                        db.execute('INSERT INTO HGPTI_ACTIVITY_KIND_MEMSET VALUES('+','.join('?'*8)+')',
                            (r['start'],r['end'],4,0,0,1,2,100))
            self.assertEqual(parse(path,[case])['cases'][0]['gpu_sum_us'],5)
            bad=copy.deepcopy(case);bad['correctness'][0]['relative_l1_error']=.1
            with self.assertRaisesRegex(ValueError,'numerical proof'):parse(path,[bad])

    def test_asys_command_does_not_kill_after_first_config(self):
        cmd=profile_command(Path('/sdk/asys'),Path('/tmp/trace'),['python','worker.py'])
        for option,value in [('--capture-range','hggcProfilerApi'),('--capture-range-end','merge'),
                             ('--kill','none'),('--python-sampling','false')]:
            self.assertEqual(cmd[cmd.index(option)+1],value)
        self.assertNotIn('--hgtx-capture',cmd)

    def test_summary_best_is_full_gpu_sum_not_main_only(self):
        point=dict(id='p',workload=asdict(WORKLOAD),tensors=['one','two'],cases=[
            dict(id='a',recipe=asdict(SF),roles=['rule']),dict(id='b',recipe=asdict(SF),roles=['model-config'])])
        rows={key:dict(case_id=key,gpu_sum_us=t,phases_us={'main':main})
              for key,t,main in [('native-fp16',20,12),('a',19,7),('b',18,10)]}
        v=summarize([point],{'p':rows})
        self.assertEqual(v['points'][0]['best_id'],'b')
        self.assertFalse(v['no_regression_admitted'])
        self.assertEqual(v['status'],'SCREEN_COMPLETE')
        rows.pop('a');self.assertEqual(summarize([point],{'p':rows})['status'],'INCOMPLETE')


if __name__=='__main__':unittest.main()

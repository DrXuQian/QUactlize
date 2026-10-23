"""Host-only gates for the bounded Q3/Q8 experiment and its compact handoff."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from tools import run_dense_dequant as run


class DenseDequantRunner(unittest.TestCase):
    def test_plan_has_only_eight_families_and_two_candidates(self):
        p = run.plan(run.selected('0,1,2,3,4,5,6,7'), 67108864)
        self.assertEqual(len(p['points']), 8)
        self.assertEqual([r['q'] for r in p['points']], [11]*2+[8]*6)
        self.assertEqual(p['variants'], [0,22,23])
        self.assertEqual(p['order'], [0,22,23,23,22,0])
        self.assertEqual(p['correctness_storage'], ['F16','BF16'])
        self.assertEqual(p['timed_storage'], 'F16')
        self.assertFalse(p['production_changed'])
        for value in ('', '8', '-1', '0,0', '0;1'):
            with self.assertRaises(ValueError):
                run.selected(value)

    def test_plan_only_needs_no_sdk_or_device(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'plan'
            subprocess.run([sys.executable, str(Path(run.__file__)), '--output', str(output),
                            '--sdk', '/no-device-or-sdk', '--plan-only'], check=True, capture_output=True)
            self.assertFalse((output/'build').exists())
            self.assertEqual(json.loads((output/'plan.json').read_text())['variants'], [0,22,23])

    def test_q8_bounded_plan_and_both_precisions(self):
        p=run.plan([2,3,4,5,6,7],67108864,True)
        self.assertEqual(p['variants'],[22,23,24,25])
        self.assertEqual(p['order'],[22,23,24,25,25,24,23,22])
        self.assertEqual(p['timed_storage'],['F16','BF16'])
        self.assertTrue(all(r['q']==8 for r in p['points']))
        with self.assertRaises(ValueError):run.plan([0],67108864,True)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'q8'
            subprocess.run([sys.executable,run.__file__,'--output',str(out),
                            '--q8-exchange','--plan-only'],check=True,capture_output=True)
            self.assertEqual(json.loads((out/'plan.json').read_text()),run.plan([2,3,4,5,6,7],0,True))

    def test_q8_fast_tail_uses_cxx_inventory_and_bounded_profiles(self):
        p=run.plan([2,3,4,5,6,7],67108864,fast_tail=True)
        self.assertEqual(p['variants'],[22,*range(26,35)])
        self.assertEqual(p['timed_storage'],['F16','BF16'])
        self.assertEqual(p['experiment'],'Q8_FAST_TAIL')
        self.assertEqual(p['configs']['34'],dict(fast=1,threads=512,vector_scale=1))
        with self.assertRaises(ValueError):run.plan([2],67108864,True,True)
        with self.assertRaises(ValueError):run.plan([0],67108864,fast_tail=True)
        for winner in range(26,35):
            medians={str(v):float(v) for v in p['variants']};medians[str(winner)]=1.
            profile=run.profile_variants(medians,fast_tail=True)
            self.assertTrue({22,26,31,winner}<=set(profile))
            self.assertLessEqual(len(profile),6)
            self.assertEqual(len(profile),len(set(profile)))
        symbol='_ZN9quactlize7dequant22dense_dequant_q8_tunedILb1ELb1ELj512ELb1EEEvPKtS3_Ptii'
        expected=run.profile_entry(dict(kernels={symbol:{}}),5,34,1)
        self.assertEqual(expected,dict(symbol=symbol,grid=[128,16,1],block=[512,1,1]))
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'fast'
            subprocess.run([sys.executable,run.__file__,'--output',str(output),'--q8-fast-tail','--plan-only'],
                           check=True,capture_output=True)
            self.assertEqual(json.loads((output/'plan.json').read_text()),run.plan([2,3,4,5,6,7],0,fast_tail=True))

    def test_q8_bf16_missing_samples_are_not_reused(self):
        q,n,k=run.POINTS[5]
        variants=(22,23,24,25)
        timing=dict(samples_us={str(v):[10.0]*30 for v in variants},
                    median_us={str(v):10.0 for v in variants},
                    round_medians_us={str(v):[10.0]*3 for v in variants})
        proof=dict(bad=0,compared=n*k,guard='PASS',control_raw_bits_equal=True)
        data=dict(status='PASS',point=5,q=q,n=n,k=k,scope=run.SCOPE,experiment='Q8_SHARED_EXCHANGE',
                  production_changed=False,performance_admitted=False,negative_bad=100,
                  traffic=dict(reads=3*67108864,writes=3*67108864),ring_copies=2,calls_per_graph=4,
                  **copy.deepcopy(timing),bf16_timing=timing,
                  proof={str(s):{str(v):proof.copy() for v in variants} for s in (0,1)})
        self.assertIs(run.validate_point(data,5,67108864,True),data)
        data['bf16_timing']['samples_us']['24'].pop()
        with self.assertRaises(ValueError):run.validate_point(data,5,67108864,True)

    def test_q8_fast_tail_resume_cannot_import_exchange_results(self):
        q,n,k=run.POINTS[5];vs=run.variants_for(fast_tail=True)
        timing=dict(samples_us={str(v):[10.]*30 for v in vs},median_us={str(v):10. for v in vs},
                    round_medians_us={str(v):[10.]*3 for v in vs})
        proof=dict(bad=0,compared=n*k,guard='PASS',control_raw_bits_equal=True)
        data=dict(status='PASS',point=5,q=q,n=n,k=k,scope=run.SCOPE,experiment='Q8_FAST_TAIL',
                  production_changed=False,performance_admitted=False,negative_bad=100,
                  traffic=dict(reads=3*67108864,writes=3*67108864),ring_copies=2,calls_per_graph=4,
                  **copy.deepcopy(timing),bf16_timing=timing,
                  memory_alignment={p:{str(w):[0] for w in (16,32,64,128)} for p in ('low','units','output')},
                  proof={str(s):{str(v):proof.copy() for v in vs} for s in (0,1)})
        self.assertIs(run.validate_point(data,5,67108864,fast_tail=True),data)
        with self.assertRaises(ValueError):run.validate_point(data,5,67108864,exchange=True)
        data['memory_alignment']['units']['128']=[64]
        with self.assertRaises(ValueError):run.validate_point(data,5,67108864,fast_tail=True)
        data['memory_alignment']['units']['128']=[0]
        data['proof']['1']['34']['bad']=1
        with self.assertRaises(ValueError):run.validate_point(data,5,67108864,fast_tail=True)

    def test_q8_profile_symbol_geometry_and_direct_launch_boundary(self):
        from dev.native_components.profile import acu_command,check_report
        import csv,io
        symbol='_ZN9quactlize7dequant25dense_dequant_q8_exchangeILb1ELj128EEEvPKtS3_Ptii'
        expected=run.profile_entry(dict(kernels={symbol:{}}),5,24,1)
        self.assertEqual(expected,dict(symbol=symbol,grid=[128,16,1],block=[128,1,1]))
        with self.assertRaises(ValueError):run.profile_entry(dict(kernels={symbol:{}}),5,24,0)
        cmd=acu_command('acu','report',['python','child'],graph_profiling='none')
        self.assertEqual(cmd[cmd.index('--graph-profiling')+1],'none')
        self.assertEqual(cmd[cmd.index('--launch-count')+1],'1')
        output=io.StringIO();w=csv.writer(output,quoting=csv.QUOTE_ALL)
        w.writerow(['ID','Kernel Name','Kernel Mangled Name','Grid Size','Block Size'])
        w.writerow(['0','demangled',symbol,'(128,16,1)','(128,1,1)'])
        self.assertEqual(check_report(output.getvalue(),expected)['launches'],1)
        with self.assertRaises(ValueError):check_report(output.getvalue().replace('(128,16,1)','(128,8,1)'),expected)

    def test_partial_or_changed_results_are_not_reused(self):
        q,n,k = run.POINTS[2]
        proof = dict(bad=0, compared=n*k, guard='PASS', control_raw_bits_equal=True)
        data = dict(status='PASS', point=2, q=q, n=n, k=k, scope=run.SCOPE,
                    production_changed=False, performance_admitted=False, negative_bad=100,
                    traffic=dict(reads=3*67108864,writes=3*67108864), ring_copies=2, calls_per_graph=4,
                    samples_us={str(v): [10.0]*30 for v in run.VARIANTS},
                    median_us={str(v):10.0 for v in run.VARIANTS},
                    round_medians_us={str(v):[10.0]*3 for v in run.VARIANTS},
                    proof={str(s):{str(v):proof.copy() for v in run.VARIANTS} for s in (0,1)})
        self.assertIs(run.validate_point(data,2,67108864),data)
        for key, value in (('point',3), ('n',n+256), ('negative_bad',0), ('production_changed',True),
                           ('calls_per_graph',1), ('scope','MODEL_E2E')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.validate_point(dict(data,**{key:value}),2,67108864)
        for mutate in (
            lambda d:d['samples_us']['22'].pop(),
            lambda d:d['samples_us']['22'].__setitem__(0,float('nan')),
            lambda d:d['median_us'].__setitem__('23',8),
            lambda d:d['proof']['1']['22'].__setitem__('control_raw_bits_equal',False),
            lambda d:d['traffic'].__setitem__('reads',1),
        ):
            bad=copy.deepcopy(data);mutate(bad)
            with self.assertRaises(ValueError):run.validate_point(bad,2,67108864)

    def test_review_excludes_reports_binaries_and_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('point-0.json','compile.log','profile-0-raw.csv','profile-0.acurep','dense.o','lib.so','weights.bin'):
                (root/name).write_text('fixture')
            run.package(root)
            with tarfile.open(next(root.glob('review-*.tgz'))) as archive:
                self.assertEqual(set(archive.getnames()),{'point-0.json','compile.log','profile-0-raw.csv'})

    def test_new_trace_kernel_keeps_precision_check(self):
        from dev.prefill_asys.trace import helper_role
        for q in (8,11):
            for dtype in ('true','false'):
                row=dict(kind='kernel',name=f'void quactlize::dequant::dense_dequant_plane_pair<{q}, {dtype}, 128u>(...)')
                self.assertEqual(helper_role(row,'full-fp16'),'expansion')
        for dtype in ('true','false'):
            row=dict(kind='kernel',name=f'void quactlize::dequant::dense_dequant_q8_exchange<{dtype}, 128u>(...)')
            self.assertEqual(helper_role(row,'full-fp16'),'expansion')


if __name__ == '__main__':
    unittest.main()

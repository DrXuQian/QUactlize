"""Host-only gates for the bounded Q3/Q8 experiment and its compact handoff."""
import copy
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tools import run_dense_dequant as run


class DenseDequantRunner(unittest.TestCase):
    def test_product_inventory_separates_bf16_and_rejects_old_flags(self):
        p=run.plan([2,3,4,5,6,7],67108864,q8=True)
        self.assertEqual(p['variants'],[0,22,23,38,41])
        self.assertEqual(p['correctness_storage'],['F16'])
        self.assertEqual(p['timed_storage'],'F16')
        self.assertEqual(p['experiment'],'Q8_PRODUCT_F16_AB')
        self.assertEqual(p['configs']['38'],dict(fast=1,threads=256))
        self.assertEqual(p['paired_configs']['41'],dict(signed=0,threads=128))
        with self.assertRaises(ValueError):run.plan([0],67108864,q8=True)
        for control in (0,22,23):
            medians={str(v):10. for v in p['variants']};medians[str(control)]=1.
            self.assertEqual(run.profile_variants(medians,q8=True),(control,38,41))
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'q8'
            subprocess.run([sys.executable,run.__file__,'--output',str(out),'--q8','--plan-only'],
                           check=True,capture_output=True)
            self.assertEqual(json.loads((out/'plan.json').read_text()),run.plan([2,3,4,5,6,7],0,q8=True))
            for flag in ('--q8-exchange','--q8-fast-tail','--q8-half2','--q8-paired'):
                child=subprocess.run([sys.executable,run.__file__,'--output',str(out),'--plan-only',flag],
                                     capture_output=True,text=True)
                self.assertNotEqual(child.returncode,0)
                self.assertIn('unrecognized arguments',child.stderr)

    def test_half2_resume_requires_exhaustive_device_proof(self):
        q,n,k=run.POINTS[4];vs=run.variants_for(q8=True)
        proof=dict(bad=0,compared=n*k,guard='PASS',control_raw_bits_equal=True)
        gate=dict(pairs=256*63488,lanes=2,bad_ordinary=0,bad_fast=0,negative_bad=1,bf16_rejected=[38])
        data=dict(status='PASS',point=4,q=q,n=n,k=k,scope=run.SCOPE,experiment='Q8_PRODUCT_F16_AB',
            production_changed=False,performance_admitted=False,negative_bad=100,
            traffic=dict(reads=3*67108864,writes=3*67108864),ring_copies=2,calls_per_graph=4,
            samples_us={str(v):[10.]*30 for v in vs},median_us={str(v):10. for v in vs},
            round_medians_us={str(v):[10.]*3 for v in vs},proof={'0':{str(v):proof.copy() for v in vs}},
            memory_alignment={p:{str(w):[0] for w in (16,32,64,128)} for p in ('low','units','output')},
            bf16_timing=None,half2_gate=gate,paired_gate=run.paired_gate_receipt())
        self.assertIs(run.validate_point(data,4,67108864,q8=True),data)
        for field,value in (('pairs',256),('bad_fast',1),('negative_bad',0),('bf16_rejected',[])):
            bad=copy.deepcopy(data);bad['half2_gate'][field]=value
            with self.assertRaises(ValueError):run.validate_point(bad,4,67108864,q8=True)
        with self.assertRaises(ValueError):run.validate_point(data,4,67108864)
        for mutate in (
            lambda d:d['memory_alignment']['units'].__setitem__('128',[64]),
            lambda d:d['proof']['0']['38'].__setitem__('bad',1),
            lambda d:d['proof']['0']['41'].__setitem__('control_raw_bits_equal',False),
            lambda d:d.__setitem__('experiment','Q8_PAIRED_F16_AB'),
        ):
            bad=copy.deepcopy(data);mutate(bad)
            with self.assertRaises(ValueError):run.validate_point(bad,4,67108864,q8=True)

    def test_shell_rejects_retired_flags_before_sdk_or_device_access(self):
        script=Path(run.__file__).with_name('run_dense_dequant_box.sh')
        retired=('Q8_EXCHANGE','Q8_FAST_TAIL','Q8_HALF2','Q8_PAIRED')
        env={k:v for k,v in os.environ.items() if k not in (*retired,'Q8','Q8_NATIVE','PROFILE_ACU')}
        env.update(GPU='1',IDLE_CONFIRMED='1',PPU_SDK='/no-sdk')
        for name in retired:
            for value in ('0','1'):
                child=subprocess.run(['bash',str(script)],env=dict(env,**{name:value}),capture_output=True,text=True)
                self.assertNotEqual(child.returncode,0)
                self.assertIn(name+' was retired',child.stderr)

    def test_half2_profile_cannot_mislabel_fast_as_bf16(self):
        symbol='_ZN9quactlize7dequant22dense_dequant_q8_half2ILb1ELj256EEEvPKtS3_Ptii'
        native=dict(kernels={symbol:{}})
        self.assertEqual(run.profile_entry(native,4,38,0),dict(symbol=symbol,grid=[64,16,1],block=[256,1,1]))
        with self.assertRaises(ValueError):run.profile_entry(native,4,38,1)
        with self.assertRaises(ValueError):run.profile_entry(native,4,37,0)

    def test_half2_gate_checks_observed_coverage_and_negative(self):
        import numpy as np
        def bench(coverage=256*63488,negative=1,rejection=24):
            results=[np.array(r,dtype='<u4').tobytes() for r in
                     ([0,0,0xffffffff,coverage],[0,negative,0,coverage])]
            return SimpleNamespace(lib=SimpleNamespace(quactlize_q8_half2_gate=Mock(return_value=0)),
                r=SimpleNamespace(alloc=Mock(return_value=4096),stream=C.c_void_p(16)),
                sdk=SimpleNamespace(lib=SimpleNamespace(hggcMemcpy=Mock(return_value=0)),
                    synchronize=Mock(),download=Mock(side_effect=results)),
                launch=Mock(return_value=rejection))
        good=bench()
        self.assertEqual(run.half2_gate(good)['pairs'],256*63488)
        self.assertEqual(good.sdk.synchronize.call_count,4)
        self.assertEqual(good.launch.call_count,1)
        for options in (dict(coverage=256),dict(negative=0),dict(rejection=0),dict(rejection=1)):
            with self.subTest(options=options),self.assertRaises(ValueError):run.half2_gate(bench(**options))

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

    def test_q8_profile_symbol_geometry_and_direct_launch_boundary(self):
        from dev.native_components.profile import acu_command,check_report
        import csv,io
        symbol='_ZN9quactlize7dequant25dense_dequant_plane_pairILi8ELb1ELj128EEEvPKtS3_Ptii'
        expected=run.profile_entry(dict(kernels={symbol:{}}),5,22,1)
        self.assertEqual(expected,dict(symbol=symbol,grid=[128,16,1],block=[128,1,1]))
        with self.assertRaises(ValueError):run.profile_entry(dict(kernels={symbol:{}}),5,22,0)
        cmd=acu_command('acu','report',['python','child'],graph_profiling='none')
        self.assertEqual(cmd[cmd.index('--graph-profiling')+1],'none')
        self.assertEqual(cmd[cmd.index('--launch-count')+1],'1')
        output=io.StringIO();w=csv.writer(output,quoting=csv.QUOTE_ALL)
        w.writerow(['ID','Kernel Name','Kernel Mangled Name','Grid Size','Block Size'])
        w.writerow(['0','demangled',symbol,'(128,16,1)','(128,1,1)'])
        self.assertEqual(check_report(output.getvalue(),expected)['launches'],1)
        with self.assertRaises(ValueError):check_report(output.getvalue().replace('(128,16,1)','(128,8,1)'),expected)

    def test_direct_control_profile_and_retired_id_rejection(self):
        symbol='_ZN9quactlize7dequant16dense_dequant_q8ILb0EEEvPKtS3_Ptii'
        native=dict(kernels={symbol:{}})
        self.assertEqual(run.profile_entry(native,4,0,0),
                         dict(symbol=symbol,grid=[64,64,1],block=[128,1,1]))
        for v in (*range(24,38),39,40,42):
            with self.assertRaises(ValueError):run.profile_entry(native,4,v,0)

    def test_partial_or_changed_results_are_not_reused(self):
        q,n,k = run.POINTS[2]
        proof = dict(bad=0, compared=n*k, guard='PASS', control_raw_bits_equal=True)
        data = dict(status='PASS', point=2, q=q, n=n, k=k, scope=run.SCOPE,experiment='PLANE_PAIR',
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

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


if __name__ == '__main__':
    unittest.main()

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dev.native_components.profile import (POINT, WORKLOAD, CONFIG, NATIVE, selected,
                                            check_native, check_report, acu_command)
from dev.native_components.plan import plan
from dev.native_components.fixture import activation
from dev.strategy.vector import spec, source
from quactlize.dispatch.table import Workload


class ProfileContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row, cls.recipe = selected(plan()['points'])

    def test_frozen_measured_simt_s1_not_tc(self):
        self.assertEqual(self.row['id'], POINT)
        self.assertEqual(self.row['selected_recipe'], CONFIG)
        self.assertEqual(len(self.row['quactlize']['components']), 1)
        self.assertFalse(self.row['pending'])
        cpp = source(spec(Workload(**WORKLOAD), self.recipe))
        self.assertIn('measured_decode_launch<14>', cpp)
        self.assertIn('simt::launch_v2<14,3,4,8,8>', cpp)
        self.assertIn('d->compute_type!=1', cpp)

    def test_cannot_silently_change_workload_recipe_or_obligations(self):
        for mutation in ('shape', 'split', 'helper'):
            row = copy.deepcopy(self.row)
            if mutation == 'shape':
                row['workload']['m'] = 1
            elif mutation == 'split':
                row['selected_recipe']['split'] = 2
            else:
                row['pending'].append('MISSING_HELPER')
            with self.assertRaises(ValueError):
                selected([row])
        with self.assertRaises(ValueError):
            selected([self.row, self.row])

    def test_native_inventory_must_remain_exact(self):
        self.assertEqual(check_native(NATIVE), NATIVE)
        wrong = copy.deepcopy(NATIVE)
        wrong[1]['block'][1] = 4
        with self.assertRaises(ValueError):
            check_native(wrong)
        with self.assertRaises(ValueError):
            check_native(NATIVE[:1])

    def test_profile_one_warmed_native_launch(self):
        cmd = acu_command('acu', 'report', ['native', 'req', 'out'], symbol=NATIVE[1]['symbol'])
        for option, value in (('--cache-control', 'all'), ('--launch-skip', '3'),
                              ('--launch-count', '1'), ('--kill', 'no'),
                              ('--kernel-name-base', 'mangled'), ('--kernel-name', NATIVE[1]['symbol'])):
            self.assertEqual(cmd[cmd.index(option)+1], value)
        self.assertEqual(cmd[-3:], ['native', 'req', 'out'])
        candidate = acu_command('acu', 'report', ['python', 'profile'])
        self.assertEqual(candidate[candidate.index('--profile-from-start')+1], 'no')
        self.assertNotIn('--launch-skip', candidate)

    @staticmethod
    def csv_text(nodes):
        stream = io.StringIO()
        fields = ['ID', 'Kernel Name', 'Kernel Mangled Name', 'Grid Size', 'Block Size', 'Metric Name', 'Metric Value']
        writer = csv.DictWriter(stream, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for index, node in nodes:
            writer.writerow(dict(zip(fields, [index, node['symbol'], node['symbol'],
                str(tuple(node['grid'])), str(tuple(node['block'])), 'dram__bytes_read.sum', 1234])))
        return 'ACU import banner\n'+stream.getvalue()

    def test_full_counter_rows_but_exactly_one_launch(self):
        node = NATIVE[1]
        text = self.csv_text([(0, node), (0, node)])
        self.assertEqual(check_report(text, node)['launches'], 1)
        for rows in ([], [(0, node), (1, node)], [(0, NATIVE[0])]):
            with self.assertRaises(ValueError):
                check_report(self.csv_text(rows), node)
        wrong = dict(node, grid=[1, 1, 1])
        with self.assertRaises(ValueError):
            check_report(text, wrong)

    def test_candidate_native_input_and_owner_contract(self):
        import numpy as np
        from dev.smallm_closure.fixture import Weights
        from dev.strategy.indexed import point
        from tools.run_kpack_llama_ab import point_request
        # Input generation uses only metadata/oracle sums, not a GPU or full B.
        fixture = Weights.__new__(Weights)
        fixture.k, fixture.experts = WORKLOAD['k'], WORKLOAD['experts']
        a = activation(self.row)
        fixture.categories = np.asarray(a['categories'])
        fixture.sums = np.zeros((17, 4, WORKLOAD['n']))
        fixture.absolute = np.zeros_like(fixture.sums)
        actual = fixture.inputs(point(Workload(**WORKLOAD)), 0)
        values = np.asarray(a['coefficients'][0], dtype='<f4').reshape(64, 4)
        np.testing.assert_array_equal(actual['a'], values[:, fixture.categories])
        native = point_request(self.row, 67108864)
        self.assertEqual(actual['ids'].ravel().tolist(), native['owners'])
        self.assertEqual(native['ring']['minimum_read_bytes'], 6881280)
        self.assertEqual(native['copies'], 22)

    def test_child_accepts_parent_log_but_not_existing_result(self):
        from tools import profile_q6_native as tool
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'run.log').write_text('parent opened log before spawn')
            argv = ['profile_q6_native', '--child', 'timing', '--request', str(out/'request.json'), '--output', str(out)]
            with mock.patch('sys.argv', argv), mock.patch.object(tool, 'candidate', return_value=0) as child:
                self.assertEqual(tool.main(), 0)
                child.assert_called_once()
            (out/'candidate.json').write_text('{}')
            with mock.patch('sys.argv', argv), self.assertRaises(SystemExit):
                tool.main()

    def test_review_excludes_weights_objects_and_source(self):
        from tools.profile_q6_native import archive
        import tarfile
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for path in ('fixture.raw', 'kernel.so', 'vector-cache/manifest.json', 'source/a.json',
                         'summary.json', 'native/profile.acurep', 'native/raw.csv', 'native/acu.log'):
                target = out/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'fixture')
            with tarfile.open(archive(out)) as tar:
                self.assertEqual(set(tar.getnames()), {'summary.json', 'native/profile.acurep', 'native/raw.csv', 'native/acu.log'})

    def test_profile_failure_does_not_discard_other_reports(self):
        from tools import profile_q6_native as tool
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder))
            jobs = [('kpack-producer', {}), ('native-producer', {}), ('native-adapter', {})]
            responses = [RuntimeError('one profiler failure'), dict(status='PASS'), dict(status='PASS')]
            with mock.patch.object(tool, 'profile_one', side_effect=responses) as run:
                records = tool.profiles(args, jobs, None, None, None, None, None)
            self.assertEqual(run.call_count, 3)
            self.assertEqual([r['status'] for r in records], ['FAIL', 'PASS', 'PASS'])
            self.assertTrue((args.output/'kpack-producer/failure.json').is_file())


if __name__ == '__main__':
    unittest.main()

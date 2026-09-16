"""Box test discovery, L2 admission and visible child-process failures."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

from dev.gemv_simt.run_q8_topology import resolve_l2, wait_logged

ROOT = Path(__file__).resolve().parents[1]


class Entry(unittest.TestCase):
    def run_shadowed(self, args):
        # A regular third-party tests package shadows this repository's
        # namespace directory, even with the repository on sys.path.
        program = (
            'import sys, types, unittest; '
            'shadow=types.ModuleType("tests"); shadow.__path__=[]; '
            'sys.modules["tests"]=shadow; '
            'unittest.main(module=None, argv=' + repr(['unittest', *args]) + ')'
        )
        return subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)

    def test_old_package_entry_reproduces_missing_module(self):
        result = self.run_shadowed(['tests.test_q8_topology', '-v'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No module named 'tests.test_q8_topology'", result.stderr)

    def test_path_discovery_still_runs_all_eight_checks(self):
        result = self.run_shadowed(['discover', '-s', str(ROOT / 'tests'),
                                    '-p', 'test_q8_topology.py', '-v'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Ran 8 tests', result.stderr)
        self.assertIn('\nOK\n', result.stderr)
        script = (ROOT / 'tools/run_q8_topology_box.sh').read_text()
        self.assertIn('unittest discover -s "$ROOT/tests" -p \'test_q8_topology*.py\'', script)
        self.assertIn('set -Eeuo pipefail', script)
        self.assertIn('tee -a "$RUN/results/host-tests.log" "$RUN/console.log"', script)


class L2Admission(unittest.TestCase):
    capacity = 64 * 1024**2

    def resolve(self, properties=0, attribute=0, verified=0):
        rt = Mock()
        rt.attribute.return_value = attribute
        identity = dict(l2_bytes=properties, name='test-device')
        result = resolve_l2(rt, identity, verified)
        rt.attribute.assert_called_once_with(38)
        self.assertEqual(identity['l2_bytes'], properties, 'raw probe receipt must remain unchanged')
        self.assertEqual(result['properties_l2_bytes'], properties)
        self.assertEqual(result['attribute_l2_bytes'], attribute)
        return result

    def test_scalar_query_recovers_missing_properties(self):
        result = self.resolve(attribute=self.capacity)
        self.assertEqual(result['l2_bytes'], self.capacity)
        self.assertEqual(result['l2_source'], 'DEVICE_ATTRIBUTE_38')

    def test_positive_properties_when_attribute_is_missing(self):
        result = self.resolve(properties=self.capacity)
        self.assertEqual(result['l2_bytes'], self.capacity)
        self.assertEqual(result['l2_source'], 'DEVICE_PROPERTIES')

    def test_verified_receipt_fills_missing_queries(self):
        result = self.resolve(verified=self.capacity)
        self.assertEqual(result['l2_bytes'], self.capacity)
        self.assertEqual(result['l2_source'], 'OPERATOR_VERIFIED_BYTES')
        self.assertEqual(result['verified_l2_bytes'], self.capacity)

    def test_matching_receipt_does_not_override_query(self):
        result = self.resolve(properties=self.capacity, attribute=self.capacity, verified=self.capacity)
        self.assertEqual(result['l2_source'], 'DEVICE_ATTRIBUTE_38')

    def test_no_query_or_receipt_fails_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, 'set L2_BYTES to independently verified bytes'):
            self.resolve()

    def test_disagreeing_queries_are_not_hidden_by_receipt(self):
        with self.assertRaisesRegex(ValueError, 'L2 queries conflict'):
            self.resolve(properties=self.capacity // 2, attribute=self.capacity, verified=self.capacity)

    def test_disagreeing_or_negative_receipt_is_rejected(self):
        for value in (self.capacity // 2, -1):
            with self.subTest(verified=value), self.assertRaisesRegex(ValueError, 'conflict'):
                self.resolve(attribute=self.capacity, verified=value)

    def test_query_error_is_not_masked(self):
        rt = Mock()
        rt.attribute.side_effect = RuntimeError('device attribute failed')
        with self.assertRaisesRegex(RuntimeError, 'device attribute failed'):
            resolve_l2(rt, dict(l2_bytes=0), self.capacity)


class ChildDiagnostics(unittest.TestCase):
    def test_failed_child_prints_bounded_tail_and_preserves_log_and_rc(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'child.log'
            output = io.StringIO()
            program = 'print("prefix\\n" * 100); print("ValueError: L2 missing"); raise SystemExit(13)'
            with redirect_stdout(output):
                rc = wait_logged([sys.executable, '-c', program], log, 'point=0')
            self.assertEqual(rc, 13)
            self.assertIn(f'Q8_TOPOLOGY_CHILD_FAIL point=0 rc=13 log={log}', output.getvalue())
            self.assertIn('ValueError: L2 missing', output.getvalue())
            self.assertLessEqual(len(output.getvalue().splitlines()), 41)
            self.assertEqual(log.read_text().count('prefix'), 100)

    def test_successful_child_does_not_print_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'child.log'
            output = io.StringIO()
            with redirect_stdout(output):
                rc = wait_logged([sys.executable, '-c', 'print("PASS")'], log, 'point=0')
            self.assertEqual(rc, 0)
            self.assertEqual(output.getvalue(), '')
            self.assertEqual(log.read_text(), 'PASS\n')


if __name__ == '__main__':
    unittest.main()

"""The box gate must not depend on ownership of the top-level tests package."""
from pathlib import Path
import subprocess
import sys
import unittest

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
        self.assertIn('unittest discover -s "$ROOT/tests" -p test_q8_topology.py', script)
        self.assertIn('set -Eeuo pipefail', script)
        self.assertIn('tee -a "$RUN/results/host-tests.log" "$RUN/console.log"', script)


if __name__ == '__main__':
    unittest.main()

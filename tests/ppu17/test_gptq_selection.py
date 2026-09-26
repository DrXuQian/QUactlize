"""Host-only leader/control coverage; no GPU or numerical dependency."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).parent/'reference/reconfirm.py'
spec = importlib.util.spec_from_file_location('gptq_reconfirm', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Selection(unittest.TestCase):
    def test_transitive_control_is_retained(self):
        rows = [
            dict(arm='quactlize',config=17,median_us=1.,same_math_parent=6),
            dict(arm='quactlize',config=18,median_us=2.,same_math_parent=6),
            dict(arm='quactlize',config=6,median_us=3.,same_math_parent=3),
            dict(arm='quactlize',config=3,median_us=4.),
            dict(arm='marlin',config=2,median_us=1.5),
        ]
        got = module.selected_configs(rows)
        self.assertEqual(set(got), {'quactlize:17','quactlize:18','quactlize:6','quactlize:3','marlin:2'})
        self.assertEqual(len(got),len(set(got)))
        with self.assertRaisesRegex(ValueError,'missing'):
            module.selected_configs([row for row in rows if row['config'] != 3])

    def test_duplicate_configuration_rejected(self):
        row = dict(arm='marlin',config=0,median_us=1.)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            module.selected_configs([row,row])


if __name__ == '__main__':
    unittest.main()

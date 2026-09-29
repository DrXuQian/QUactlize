"""Host-only leader/control coverage; no GPU or numerical dependency."""
import importlib.util
from pathlib import Path
import sys
import unittest
from copy import deepcopy

path = Path(__file__).parent/'reference/reconfirm.py'
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location('gptq_reconfirm', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from machete_parity import paired_winner


class Selection(unittest.TestCase):
    def test_parity_uses_median_of_rounds(self):
        rows = [dict(arm=arm,config=config,median_us=us,config_detail=detail,
                     clocks_after={'sm_mhz':1755,'memory_mhz':1593})
                for arm,config,us,detail in (
                    ('machete',0,10.,'128x16x64_cooperative_streamk'),
                    ('quactlize',0,10.1,'128x16x64_prepacked_cooperative_streamk'),
                    ('quactlize',16,10.2,'128x16x64_prepacked_streamk_s8'))]
        rounds = [deepcopy(rows) for _ in range(3)]
        rounds[0][2]['median_us'] = 1.
        result = paired_winner(rounds)
        self.assertEqual(result['best']['quactlize']['config'], 0)
        self.assertTrue(result['within_5pct'])
        self.assertFalse(result['strict_no_slower'])
        with self.assertRaisesRegex(ValueError, 'three independent'):
            paired_winner(rounds[:2])
        rounds[0][1]['clocks_after']['sm_mhz'] = 1590
        with self.assertRaisesRegex(ValueError, 'clocks differ'):
            paired_winner(rounds)
        rounds[0][1]['clocks_after']['sm_mhz'] = 1755
        rounds[0].pop()
        with self.assertRaisesRegex(ValueError, 'inventories differ'):
            paired_winner(rounds)

    def test_independent_machete_rejects_faster_hybrid(self):
        rows = [
            dict(arm='machete',config=0,median_us=10.,config_detail='128x16x64_cooperative_streamk'),
            dict(arm='quactlize',config=0,median_us=11.,config_detail='128x16x64_prepacked_cooperative_streamk'),
            dict(arm='quactlize',config=16,median_us=9.,same_math_parent=0,
                 config_detail='128x16x64_prepacked_streamk_s8'),
            dict(arm='quactlize',config=6,median_us=1.,config_detail='128x16x64_prepacked_marlin_self_reset'),
            dict(arm='quactlize_marlin',config=0,median_us=.5,config_detail='marlin'),
        ]
        self.assertEqual(set(module.selected_configs(rows, 'machete-streamk')),
                         {'machete:0', 'quactlize:0', 'quactlize:16'})
        with self.assertRaisesRegex(ValueError, 'missing same-math'):
            module.selected_configs([r for r in rows if not (r['arm']=='quactlize' and r['config']==0)],
                                    'machete-streamk')
        with self.assertRaisesRegex(ValueError, 'both pure families'):
            module.selected_configs([r for r in rows if r['arm']!='machete'], 'machete-streamk')
        with self.assertRaisesRegex(ValueError, 'unknown comparison family'):
            module.selected_configs(rows, 'machete')

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

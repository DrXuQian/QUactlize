import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dev.strategy.journal import atomic_json
from dev.strategy.prefill_refresh import (load_seed, make_plan, recipe_of, same_recipe, bind,
                                         checked_round, collect)
from dev.strategy.prefill_plan import family
from dev.strategy.refresh_clock import ADVISORY
from dev.strategy.prefill_confirmation import ordered_jobs, ROUNDS
from quactlize.dispatch.planning import plan_prefill
from quactlize.dispatch.table import Recipe
from quactlize.runtime.tuning import digest
from tools.run_prefill_refresh import summarize, env_for, command


def round_fixture():
    """Host data only, never device accuracy or performance evidence."""
    from dev.strategy.components import statistics_checked
    from dev.strategy.refresh_clock import parse
    point = dict(id='host-test', jobs={'j': dict(id='j')}, workload={},
        arms={'a': dict(jobs=['j'], roles=['rule', 'control'], original=True, recipe={})}, original='a')
    authority = dict(device=dict(pci='0000:08:00.0'),
        clock=dict(policy=ADVISORY, requested_mhz=1700))
    epoch = dict(sha256=digest(authority), point=point['id'], round=0)
    identity = dict(authority=epoch, job='j')
    cid = digest(identity)
    component = dict(statistics_checked([[10.]*3]*3), identity=identity, kind='producer',
                     first_use_excluded=True, host_launch_cost_added=False)
    component['record_sha256'] = digest(component)
    text = '0000:08:00.0, 1550, 1800'
    clock = dict(parse(text, '0000:08:00.0', 1700, advisory=True), stdout=text)
    row = dict(point=point['id'], round=0, authority=epoch, status='PASS', order=['j'],
        jobs={'j': dict(correctness='PASS', components=[cid],
                        numerical=[dict(nonfinite=0, relative_l1_error=.001)])},
        components={cid: component}, clock_observations=dict(before=clock, after=clock))
    row['record_sha256'] = digest(row)
    return point, authority, row


class PrefillRefresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='prefill-refresh-test-')
        cls.selector = plan_prefill(Path(cls.tmp.name))
        cls.seed = load_seed()
        cls.plan = make_plan(cls.selector, list('1234567'), cls.seed)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_finalists_and_actual_controls(self):
        p = self.plan
        self.assertEqual(p['counts'], dict(points=430, arms=1138, families=86, modules=115, queries=782, rounds=6))
        self.assertEqual(p['plan_sha256'], digest({k: v for k, v in p.items() if k != 'plan_sha256'}))
        for row, old in zip(p['points'], self.seed['points']):
            self.assertEqual(set(row['arms']), set(old['arms']))
            control = row['arms'][row['original']]
            self.assertIn('control', control['roles'])
            self.assertIn('rule', control['roles'])
            self.assertTrue(same_recipe(recipe_of(row['current_selector']), Recipe.from_dict(control['recipe'])))
            self.assertNotIn('statistics', row)
            for arm in row['arms'].values():
                self.assertNotIn('cost_us', arm)
                self.assertNotIn('components', arm)

    def test_no_old_original_selector(self):
        source = Path('tools/run_prefill_refresh.py').read_text()
        self.assertIn('plan_prefill', source)
        self.assertNotIn('prefill_original', source)
        self.assertNotIn('--previous', source)
        self.assertNotIn('PREVIOUS_RUN', Path('tools/run_prefill_refresh_box.sh').read_text())

    def test_family_ownership_and_balance(self):
        owners = {}
        for p in self.plan['points']:
            owners.setdefault(family(p), set()).add(p['shard'])
        self.assertEqual(len(owners), 86)
        self.assertTrue(all(len(v) == 1 for v in owners.values()))
        loads = self.plan['historical_shard_seconds']
        self.assertLess(max(loads)/min(loads), 1.03)
        six = make_plan(self.selector, ['1', '2', '3', '4', '5', '7'], self.seed)
        self.assertEqual(len(six['historical_shard_seconds']), 6)
        self.assertEqual({p['id'] for p in six['points']}, {p['id'] for p in self.plan['points']})
        self.assertNotEqual(six['plan_sha256'], self.plan['plan_sha256'])

    def test_selector_drift_rejected(self):
        rows = copy.deepcopy(self.selector)
        row = next(r for r in rows['requests'] if r['kind'] == 0)
        row['split'] *= 2
        with self.assertRaisesRegex(ValueError, 'current C\\+\\+'):
            make_plan(rows, list('1234567'), self.seed)

    def test_missing_control_rejected(self):
        seed = copy.deepcopy(self.seed)
        seed['points'][0]['arms'].pop(seed['points'][0]['incumbent'])
        with self.assertRaisesRegex(ValueError, 'missing incumbent'):
            make_plan(self.selector, list('1234567'), seed)

    def test_precision_drift_rejected(self):
        seed = copy.deepcopy(self.seed)
        seed['points'][0]['workload']['compute'] = 'bf16'
        with self.assertRaisesRegex(ValueError, 'semantics'):
            make_plan(self.selector, list('1234567'), seed)

    def test_no_duplicate_devices(self):
        with self.assertRaisesRegex(ValueError, 'unique physical'):
            make_plan(self.selector, ['1', '1'], self.seed)

    def test_fresh_resource_binding_all_points(self):
        queries = {k: dict(request=v, status='READY', rc=0, occupancy=2,
                           device=dict(compute_units=72, ordinal=0))
                   for k, v in self.plan['queries'].items()}
        bound = [bind(p, queries) for p in self.plan['points']]
        self.assertEqual(sum(len(p['arms']) for p in bound), 1138)
        self.assertTrue(all(len(p['jobs']) <= 8 for p in bound))
        for p in bound:
            self.assertEqual(list(reversed(ordered_jobs(p, 0))), ordered_jobs(p, 1))
            for job in p['jobs'].values():
                if job['role'] == 'tc':
                    self.assertIn(job['arm']['module'], self.plan['modules'])
                    self.assertIn(job['arm']['binding']['query'], queries)

    def test_unavailable_finalist_not_silently_omitted(self):
        point = next(p for p in self.plan['points'] if any(a['recipe']['kind'] == 'tc' for a in p['arms'].values()))
        queries = {k: dict(request=v, status='UNAVAILABLE', rc=1, occupancy=0,
                           device=dict(compute_units=72, ordinal=0))
                   for k, v in self.plan['queries'].items()}
        with self.assertRaisesRegex(ValueError, 'cannot drop'):
            bind(point, queries)

    def test_pending_selection_not_missing_measurement(self):
        result = summarize([dict(status='PASS'), dict(status='UNSTABLE'), dict(status='RULE_REGRESSION')])
        self.assertEqual((result['status'], result['measured']), ('COMPLETE', 3))
        self.assertFalse(result['performance_admitted'])
        self.assertFalse(result['production_changed'])
        self.assertEqual(summarize([dict(status='INCOMPLETE')])['status'], 'INCOMPLETE')

    def test_missing_rounds_remain_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            result = collect(self.plan['points'][0], Path(directory), {})
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertEqual(result['completed_rounds'], 0)

    def test_partial_resume_checks_then_preserves_receipt(self):
        point, authority, row = round_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'points'/point['id']/'round-0.json'
            atomic_json(path, row)
            before = path.read_bytes()
            result = collect(point, root, authority)
            self.assertEqual(result['completed_rounds'], 1)
            self.assertEqual(path.read_bytes(), before)
            row['jobs']['j']['numerical'][0]['relative_l1_error'] = .1
            row['record_sha256'] = digest({k: v for k, v in row.items() if k != 'record_sha256'})
            atomic_json(path, row)
            with self.assertRaisesRegex(ValueError, 'numerical'):
                collect(point, root, authority)

    def test_changed_round_seal_or_authority_is_not_resumed(self):
        point, authority, row = round_fixture()
        checked_round(row, point, 0, authority)
        with self.assertRaisesRegex(ValueError, 'another run'):
            checked_round(row, point, 0, authority | dict(other_run=True))
        row['status'] = 'FAIL'
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            checked_round(row, point, 0, authority)

    def test_throttling_is_diagnostic_not_numeric_failure(self):
        point, authority, row = round_fixture()
        checked_round(row, point, 0, authority)
        self.assertTrue(row['clock_observations']['before']['target_differs'])
        self.assertFalse(row['clock_observations']['before']['affects_verdict'])
        row['clock_observations']['before']['cu_mhz'] = 1700
        row['record_sha256'] = digest({k: v for k, v in row.items() if k != 'record_sha256'})
        with self.assertRaisesRegex(ValueError, 'clock receipt'):
            checked_round(row, point, 0, authority)

    def test_child_command_and_card_isolation(self):
        args = SimpleNamespace(sdk=Path('/sdk'), output=Path('/result'), l2_bytes=67108864)
        argv = command(args, '--child', '--family', 'abc', output=Path('/result/shard-0'))
        self.assertEqual(argv[argv.index('--output')+1], '/result/shard-0')
        self.assertIn('--child', argv)
        self.assertEqual(env_for('7')['CUDA_VISIBLE_DEVICES'], '7')

    def test_package_keeps_evidence_not_binaries(self):
        from tools.confirm_kpack_prefill import package
        import tarfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'request.json', dict(clock='1700'))
            atomic_json(root/'clock-lock-1.json', dict(status='REQUESTS_ACCEPTED'))
            atomic_json(root/'shard-0/resources.json', dict(queries=[]))
            (root/'ignored.so').touch()
            package(root)
            with tarfile.open(next(root.glob('confirmation-*.tgz'))) as tar:
                names = tar.getnames()
            self.assertIn('clock-lock-1.json', names)
            self.assertIn('shard-0/resources.json', names)
            self.assertNotIn('ignored.so', names)

    def test_no_clock_failure_or_normalization_policy(self):
        from dev.strategy.refresh_clock import parse
        obs = parse('0000:08:00.0, 1550, 1800', '0000:08:00.0', 1700, advisory=True)
        self.assertFalse(obs['affects_verdict'])
        self.assertFalse(obs['frequency_normalization'])
        self.assertEqual(obs['clock_policy'], ADVISORY)

    def test_no_new_provider_timing_implementation(self):
        worker = Path('dev/strategy/prefill_confirm_worker.py').read_text()
        self.assertIn('measure_ordinary(b, full)', worker)
        self.assertIn('measure_full_expansion', worker)
        self.assertIn('measure_expansion', worker)
        self.assertIn('measure_core', worker)
        runner = Path('tools/run_prefill_refresh.py').read_text()
        self.assertNotIn('GraphClone', runner)
        self.assertNotIn('asys', runner)
        self.assertNotIn('24*3600', runner)
        self.assertIn('skipped.add(failed)', runner)


if __name__ == '__main__':
    unittest.main()

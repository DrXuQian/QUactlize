import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from dev.strategy.journal import atomic_json
from dev.strategy.prefill_plan import family
from dev.strategy.prefill_reconfirm import (load_scope, make_reconfirm_plan, cohort, check_cohort,
                                            build_paths, reuse_images)
from dev.strategy.prefill_refresh import make_plan, bind
from quactlize.dispatch.planning import plan_prefill
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.confirm_kpack_prefill import save_once


class Reconfirm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='prefill-reconfirm-test-')
        cls.selector = plan_prefill(Path(cls.tmp.name), historical=True)
        cls.scope = load_scope()
        cls.old = make_plan(cls.selector, cls.scope['prior_cards'])
        cls.plan = make_reconfirm_plan(cls.selector, list('12345'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_scope_is_whole_suspect_cohort_not_only_losing_points(self):
        plan, scope = self.plan, self.scope
        self.assertEqual(plan['counts'], dict(points=123, arms=316, families=27, modules=71,
                                            retained=307, queries=229, rounds=6))
        self.assertEqual(self.old['plan_sha256'], scope['prior_plan_sha256'])
        old = {p['id']: p for p in self.old['points']}
        suspect = {p['id'] for p in old.values() if
                   self.old['cards'][p['shard']] in scope['suspected_cards']}
        extra = set(scope['other_unresolved'])
        self.assertEqual((len(suspect), len(extra), len(suspect & extra)), (120, 3, 0))
        self.assertEqual({p['id'] for p in plan['points']}, suspect | extra)
        self.assertEqual(set(plan['retained_point_ids']), set(old)-(suspect | extra))
        self.assertFalse(plan['retained_timings_imported'])
        self.assertFalse(plan['previous_timings_reused'])
        for p in plan['points']:
            before = old[p['id']]
            self.assertEqual({k: v for k, v in p.items() if k not in ('shard', 'reconfirm_reason')},
                             {k: v for k, v in before.items() if k != 'shard'})
            self.assertNotIn('statistics', p)

    def test_all_arms_have_fresh_resource_queries(self):
        queries = {k: dict(request=v, status='READY', rc=0, occupancy=2,
                           device=dict(compute_units=72, ordinal=0)) for k, v in self.plan['queries'].items()}
        points = [bind(p, queries) for p in self.plan['points']]
        self.assertEqual(sum(len(p['arms']) for p in points), 316)
        self.assertTrue(all(p['original'] in p['arms'] for p in points))

    def test_same_family_same_card_and_balanced_walls(self):
        owners = {}
        for p in self.plan['points']:
            owners.setdefault(family(p), set()).add(p['shard'])
        self.assertTrue(all(len(v) == 1 for v in owners.values()))
        loads = self.plan['historical_shard_seconds']
        self.assertLess(max(loads)/min(loads), 1.04)
        self.assertEqual(sum(loads), sum(self.scope['scheduling_seconds'].values()))

    def test_family_ring_unchanged_when_dense_family_is_partial(self):
        from dev.strategy.prefill_fixture import routing
        from quactlize.dispatch.table import Workload
        for key in self.plan['family_order']:
            old = [p for p in self.old['points'] if family(p) == key]
            new = [p for p in self.plan['points'] if family(p) == key]
            active = lambda rows: min(len(routing(Workload(**p['workload']), r)['active'])
                                     for p in rows for r in (0, 1, 2))
            self.assertEqual(active(old), active(new))
            if len(old) != len(new):
                self.assertTrue(all(p['workload']['operation'] == 'dense' and
                                    p['workload']['experts'] == 1 for p in new))

    def test_new_cards_allowed_but_resume_cannot_repartition(self):
        other = make_reconfirm_plan(self.selector, ['0', '2', '4'])
        self.assertEqual({p['id'] for p in other['points']}, {p['id'] for p in self.plan['points']})
        self.assertNotEqual(other['plan_sha256'], self.plan['plan_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'finalists.json'
            save_once(path, self.plan)
            before = path.read_bytes()
            save_once(path, json.loads(json.dumps(self.plan)))
            with self.assertRaisesRegex(ValueError, 'resume identity changed'):
                save_once(path, other)
            self.assertEqual(path.read_bytes(), before)

    def test_changed_controls_are_not_reinterpreted_as_new_followup(self):
        selector = copy.deepcopy(self.selector)
        next(p for p in selector['requests'] if p['kind'] == 0)['split'] *= 2
        with self.assertRaisesRegex(ValueError, 'current C\\+\\+'):
            make_reconfirm_plan(selector, list('12345'))
        with self.assertRaisesRegex(ValueError, 'unique physical'):
            make_reconfirm_plan(self.selector, ['1', '1'])

    def test_cohort_allows_physical_card_change_not_sdk_or_provider_change(self):
        d = dict(hardware=dict(sdk='sdk', l2=dict(bytes=64*2**20), device=dict(
            pci='0000:01:00.0', host='old', ordinal=0, compute_units=72, warp_size=32)),
            providers=dict(dense='blas', grouped='deepgemm'), oracle='gguf', graph_abi='abi')
        plan = dict(reconfirm=dict(cohort_sha256=digest(cohort(d))))
        moved = copy.deepcopy(d)
        moved['hardware']['device'].update(pci='0000:08:00.0', host='new', ordinal=1)
        check_cohort(plan, [d, moved])
        for field, value in [('sdk', 'changed'), ('l2', dict(bytes=1))]:
            bad = copy.deepcopy(moved)
            bad['hardware'][field] = value
            with self.assertRaisesRegex(ValueError, 'cohort'):
                check_cohort(plan, [bad])
        moved['providers']['dense'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'cohort'):
            check_cohort(plan, [moved])


class ReadOnlyBuildReuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='prefill-reuse-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        old = dict(points=['host-only'])
        old['plan_sha256'] = digest(old)
        atomic_json(self.root/'finalists.json', old)
        self.c = SimpleNamespace(identity=dict(source='same', sdk='same'),
                                 source=lambda parent, key: 'body', check_inputs=Mock())
        spec = dict(endpoint='native', compute='f16', parent=dict(symbol='fixture'))
        key = digest(dict(identity=self.c.identity, parent=spec['parent'], source='body'))
        image = self.root/'cache/tc'/key/'kernel.so'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'host-test-not-an-executable')
        self.record = dict(path=str(image), key=key, parent=spec['parent'], identity=self.c.identity, sha256=sha(image))
        atomic_json(image.parent/'manifest.json', {k: v for k, v in self.record.items() if k != 'path'})
        self.images = dict(tc={'module': self.record}, helpers=dict(sha256='h'), dequant=dict(sha256='d'))
        atomic_json(self.root/'images.json', self.images)
        self.plan = dict(reconfirm=dict(prior_plan_sha256=old['plan_sha256']), modules={'module': spec})
        self.addCleanup(patch.stopall)
        patch('dev.strategy.prefill_reconfirm.compiler', return_value=self.c).start()
        patch('dev.strategy.prefill_build.verify', return_value=self.images['helpers']).start()
        patch('tools.run_kpack_strategy_decode.dequant_checked', return_value=self.images['dequant']).start()

    def run_reuse(self):
        return reuse_images(Path('/sdk'), self.root, self.plan)

    def test_read_only_cache_and_no_measurement_import(self):
        # No build method on the fake compiler: an accidental recompile fails.
        atomic_json(self.root/'summary.json', dict(status='UNSTABLE'))
        atomic_json(self.root/'shard-0/results/fake.json', dict(median_us=.001))
        before = {p: sha(p) for p in self.root.rglob('*') if p.is_file()}
        images = self.run_reuse()
        self.assertEqual(images['tc'], self.images['tc'])
        self.assertEqual(images['reuse']['new_compiles'], 0)
        self.assertFalse(images['reuse']['previous_timings_reused'])
        self.assertEqual({p: sha(p) for p in self.root.rglob('*') if p.is_file()}, before)
        self.assertNotIn('median_us', json.dumps(images))
        self.assertEqual(build_paths(Path('/new'), self.root), images['reuse']['paths'])
        self.c.check_inputs.assert_called_once()

    def test_changed_image_rejected(self):
        Path(self.record['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'image/path/parent'):
            self.run_reuse()

    def test_changed_cache_receipt_rejected(self):
        path = Path(self.record['path']).parent/'manifest.json'
        atomic_json(path, self.record | dict(identity=dict(source='other')))
        with self.assertRaisesRegex(ValueError, 'cache manifest'):
            self.run_reuse()

    def test_sdk_or_compiled_source_change_rejected(self):
        self.c.identity = dict(source='changed', sdk='same')
        with self.assertRaisesRegex(ValueError, 'source/SDK/host changed'):
            self.run_reuse()

    def test_wrong_run_or_missing_module_rejected(self):
        self.plan['reconfirm']['prior_plan_sha256'] = 'another'
        with self.assertRaisesRegex(ValueError, 'not the reviewed'):
            self.run_reuse()
        self.plan['reconfirm']['prior_plan_sha256'] = json.loads((self.root/'finalists.json').read_text())['plan_sha256']
        self.plan['modules']['missing'] = self.plan['modules']['module']
        with self.assertRaisesRegex(ValueError, 'missing required'):
            self.run_reuse()

    def test_external_image_or_symlink_rejected(self):
        outside = self.root/'wrong.so'
        outside.write_bytes(b'host-test-not-an-executable')
        self.images['tc']['module']['path'] = str(outside)
        atomic_json(self.root/'images.json', self.images)
        with self.assertRaisesRegex(ValueError, 'image/path/parent'):
            self.run_reuse()

    def test_changed_helpers_rejected(self):
        with patch('dev.strategy.prefill_build.verify', return_value=dict(sha256='changed')):
            with self.assertRaisesRegex(ValueError, 'helper/dequant receipt'):
                self.run_reuse()


if __name__ == '__main__':
    unittest.main()

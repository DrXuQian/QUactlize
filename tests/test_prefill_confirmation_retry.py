import copy
from dataclasses import asdict
import json
import lzma
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.review_prefill_cohort import SCHEMA, ingest, merge
from tools.run_prefill_model_asys import confirmation_jobs
from tools.retry_prefill_confirmation import combine, equivalent_authorities, missing_jobs, read_review
from tools import retry_prefill_confirmation as tool


class ConfirmationRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.w = asdict(Workload(8, 1024, 4096, 128, storage='f32', compute='f16'))
        self.point_id = Workload(**self.w).key
        self.old = dict(harness='original-frozen-hash', plan='plan-sha', cache='original-cache',
            source_revision={'commit': 'old-commit'}, build={'timer_sha256': 'timer'},
            payload={'path': 'original-payload', 'dense': 'dense-sha'}, asys_sha256='asys', batch_size=16,
            frequency_normalization=False,
            device=dict(sdk={'runtime': 'sdk-sha'}, l2={'bytes': 67108864},
                device=dict(host='box', ordinal=0, pci='0000:08:00.0', compute_units=72,
                    warp_size=32, max_threads=1024, DriverGetVersion=13000, RuntimeGetVersion=13000)))
        self.new = copy.deepcopy(self.old)
        self.new['device']['device']['pci'] = '0000:04:00.0'
        self.new['confirmation_retry'] = dict(parent_authority=digest(self.old), equivalent_cards_user_authorized=True)
        self.cases = [dict(id=k, arm='sf', helper_mode='cohort-current',
            roles=['model-config'], recipe={'kind': 'tc', 'route': 'sf', 'parameters': [['split', i + 1]]})
            for i, k in enumerate(('pending', 'complete'))]
        self.spec = dict(id=self.point_id, workload=self.w, cases=self.cases,
            native_comparison=False, confirmation_captures=2, capture_batch_limit=1)
        self.orig_root = self.fixture('original', self.old, [('pending', 0, 10), ('complete', 0, 20), ('complete', 1, 24)])
        self.retry_root = self.fixture('retry', self.new, [('pending', 1, 14)])
        self.original = ingest(self.orig_root)
        self.retry = ingest(self.retry_root)
        self.result = dict(schema=SCHEMA, authorities={}, points={}, components={}, candidates={},
            history=[], diagnostics=[], partial_points={})
        merge(self.result, self.original, set())
        self.result['points']['unrelated'] = dict(workload=self.w, authority=digest(self.old),
            native_comparison=False, measurements=copy.deepcopy(next(iter(self.original['partial_points'].values()))['measurements'][1:]))

    def fixture(self, name, authority, captures):
        root = self.root / name
        root.mkdir()
        (root / 'authority.json').write_text(json.dumps(authority))
        (root / 'execution-plan.json').write_text(json.dumps(dict(points=[self.spec], authority=digest(authority))))
        for key, index, us in captures:
            case = next(c for c in self.cases if c['id'] == key)
            path = root / 'batches' / (key + str(index))
            path.mkdir(parents=True)
            proof = [dict(case_id=key, arm='sf', status='PASS',
                correctness=[dict(nonfinite=0, relative_l1_error=.001)], weight_bits=[dict(bad=0)])]
            (path / 'cases.json').write_text(json.dumps(proof))
            seal = dict(authority=digest(authority), workload=self.w, cases=[case])
            if index:
                seal['confirmation'] = index
            row = dict(case_id=key, arm='sf', helper_mode='cohort-current', status='PASS',
                arithmetic={'accumulator': 'f32'}, first_ring_excluded=True, cuda_graph=False,
                timing_method='ASYS_GPU_ACTIVITY', samples=[dict(gpu_sum_us=us)] * 3,
                gpu_sum_us=us, phases_us={'main': us}, measured_rounds_us=[us] * 3,
                kernel_inventory=[dict(name='main', phase='main', grid=[144, 1, 1], block=[128, 1, 1],
                    shared_bytes=8192, samples_us=[us] * 3, median_us=us)])
            receipt = dict(seal=seal, parsed=dict(status='PASS', cases=[row]),
                cases_path=str(path / 'cases.json'), cases_sha256=sha(path / 'cases.json'),
                asysrep=str(path / 'trace.asysrep'), sqlite=str(path / 'trace.sqlite'))
            receipt['record_sha256'] = digest(receipt)
            (path / 'pass.json').write_text(json.dumps(receipt))
        return root

    def partial(self):
        return next(iter(self.original['partial_points'].values()))

    def test_missing_only_preserves_original_direction_and_batch_order(self):
        jobs = missing_jobs(self.spec, self.partial(), confirmation_jobs(self.spec))
        self.assertEqual(jobs, [([self.cases[0]], 1)])

    def test_screen_gaps_and_numerical_failures_are_not_confirmation_recovery(self):
        for modification in ({'missing': ['pending']}, {'measurements': []}):
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                missing_jobs(self.spec, dict(self.partial(), **modification), confirmation_jobs(self.spec))
        value = copy.deepcopy(self.partial())
        value['measurements'][0]['status'] = 'FAIL'
        with self.assertRaisesRegex(ValueError, 'not a missing'):
            missing_jobs(self.spec, value, confirmation_jobs(self.spec))

    def test_refuses_batch_that_would_repeat_a_valid_confirmation(self):
        spec = dict(self.spec, capture_batch_limit=16)
        with self.assertRaisesRegex(ValueError, 'repeat valid'):
            missing_jobs(spec, self.partial(), confirmation_jobs(spec))

    def test_same_model_other_pci_allowed_but_measurement_domain_unchanged(self):
        equivalent_authorities(self.old, self.new)
        for container, key, value in [('card', 'compute_units', 64), ('sdk', 'runtime', 'new-sdk'),
                                      ('root', 'harness', 'other-source'), ('root', 'payload', {'dense': 'changed'})]:
            new = copy.deepcopy(self.new)
            target = new if container == 'root' else new['device']['sdk' if container == 'sdk' else 'device']
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                equivalent_authorities(self.old, new)

    def test_merge_only_fills_missing_preserving_raw_physical_device_and_samples(self):
        original_before, retry_before = copy.deepcopy(self.original), copy.deepcopy(self.retry)
        retained = copy.deepcopy(self.result['points']['unrelated'])
        merged = combine(self.result, self.original, self.retry, self.point_id)
        self.assertEqual(merged['points']['unrelated'], retained)
        point = merged['points'][self.point_id]
        self.assertTrue(point['cross_device_confirmations'])
        joint = merged['authorities'][point['authority']]
        self.assertNotIn('pci', joint['device']['device'])
        self.assertEqual(len(joint['physical_devices']), 2)
        rows = {r['case_id']: r for r in point['measurements']}
        self.assertEqual(rows['pending']['gpu_sum_us'], 12)
        self.assertEqual(rows['complete']['gpu_sum_us'], 22)
        self.assertEqual(rows['pending']['complete_samples_us'], [10] * 3 + [14] * 3)
        self.assertEqual([r['authority'] for r in rows['pending']['confirmation_captures']],
                         [digest(self.old), digest(self.new)])
        for capture in rows['pending']['confirmation_captures']:
            for cid in capture['components']:
                self.assertEqual(merged['components'][cid]['authority'], capture['authority'])
            self.assertIn('raw_receipt_sha256', capture)
        self.assertFalse(merged['pending_points'])
        self.assertFalse(merged['no_regression_admitted'])
        self.assertEqual(self.original, original_before)
        self.assertEqual(self.retry, retry_before)
        self.assertEqual(merged['review_sha256'], digest({k: v for k, v in merged.items() if k != 'review_sha256'}))

    def test_duplicate_confirmation_rejected(self):
        fresh = next(iter(self.retry['partial_points'].values()))
        fresh['measurements'][0]['confirmation_captures'][0]['confirmation'] = 0
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            combine(self.result, self.original, self.retry, self.point_id)

    def test_kernel_or_arithmetic_drift_does_not_mutate_previous_review(self):
        for drift in ('launch', 'arithmetic'):
            retry = copy.deepcopy(self.retry)
            capture = next(iter(retry['partial_points'].values()))['measurements'][0]['confirmation_captures'][0]
            if drift == 'launch':
                retry['components'][capture['components'][0]]['block'] = [256, 1, 1]
            else:
                capture['arithmetic'] = {'accumulator': 'f16'}
            before = copy.deepcopy(self.result)
            with self.subTest(drift=drift), self.assertRaisesRegex(ValueError, 'kernel/arithmetic'):
                combine(self.result, self.original, retry, self.point_id)
            self.assertEqual(self.result, before)

    def test_user_equivalence_assertion_required(self):
        retry = copy.deepcopy(self.retry)
        next(iter(retry['authorities'].values()))['confirmation_retry']['equivalent_cards_user_authorized'] = False
        with self.assertRaisesRegex(ValueError, 'explicitly authorized'):
            combine(self.result, self.original, retry, self.point_id)

    def test_prior_review_must_match_and_cannot_replace_complete_point(self):
        self.result['points'][self.point_id] = self.result['points']['unrelated']
        with self.assertRaisesRegex(ValueError, 'previous review differs'):
            combine(self.result, self.original, self.retry, self.point_id)

    def test_compact_review_digest_checked(self):
        merged = combine(self.result, self.original, self.retry, self.point_id)
        path = self.root / 'review.json.xz'
        with lzma.open(path, 'wt') as stream:
            json.dump(merged, stream)
        self.assertEqual(read_review(path), merged)
        merged['points'][self.point_id]['measurements'][0]['gpu_sum_us'] = .1
        with lzma.open(path, 'wt') as stream:
            json.dump(merged, stream)
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            read_review(path)

    def test_entry_reuses_original_payload_and_runs_only_missing_direction(self):
        sdk = self.root / 'sdk'
        (sdk / 'asight/bin').mkdir(parents=True)
        (sdk / 'asight/bin/asys').write_bytes(b'asys')
        (sdk / 'bin').mkdir()
        (sdk / 'bin/ppu-smi').write_bytes(b'smi')
        plan = dict(finalist_plan_sha256='finalists', points=[self.spec])
        plan['plan_sha256'] = digest(plan)
        old = copy.deepcopy(self.old)
        old.update(plan=plan['plan_sha256'], asys_sha256=sha(sdk / 'asight/bin/asys'))
        timer = self.root / 'timer'
        timer.write_bytes(b'timer')
        old['build']['timer_sha256'] = sha(timer)
        root = self.fixture('flow-original', old,
                            [('pending', 0, 10), ('complete', 0, 20), ('complete', 1, 24)])
        (root.parent / 'source').mkdir()
        (root / 'plan.json').write_text(json.dumps(plan))
        (root / 'modules.json').write_text('{}')
        (root / 'build/dense').mkdir(parents=True)
        (root / 'build/dense/manifest.json').write_text('{}')
        (root / 'build/dense/libquactlize_ppu_prefill.so').write_bytes(b'dense')
        (root / 'build/libpoint-timer.so').write_bytes(b'timer')
        previous = dict(schema=SCHEMA, authorities={}, points={}, components={}, candidates={},
                        partial_points={}, history=[], diagnostics=[])
        merge(previous, ingest(root), set())
        previous['review_sha256'] = digest(previous)
        archive = self.root / 'previous.json.xz'
        with lzma.open(archive, 'wt') as stream:
            json.dump(previous, stream)
        before = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
        calls = []

        def profile(args, point, cases, authority, env, confirmation):
            calls.append(([c['id'] for c in cases], confirmation))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
            self.assertEqual(authority['device'], self.new['device'])
            self.assertEqual(args.reuse_build, Path(old['payload']['path']))
            # Simulated Asys PASS: same receipt format/validation as a device run.
            captured = self.fixture('captured', authority, [('pending', 1, 14)])
            import shutil
            shutil.copytree(captured / 'batches', args.output / 'batches')

        runner = SimpleNamespace(source_seal=lambda: old['harness'], confirmation_jobs=confirmation_jobs,
                                 validate_payload=lambda args: old['payload'], profile_batch=profile)
        output = self.root / 'recovered'
        argv = ['retry', '--previous-run', str(root), '--previous-review', str(archive),
                '--point', self.point_id, '--sdk', str(sdk), '--device', '1', '--output', str(output),
                '--equivalent-cards', '--confirm-idle']
        with patch.object(tool, 'frozen_runner', return_value=runner) as frozen, \
             patch('sys.argv', argv), patch.dict('os.environ'), \
             patch('tools.run_kpack_llama_ab.device', return_value=self.new['device']), \
             patch('dev.strategy.refresh_clock.request_lock', return_value={'returncode': 0}), \
             patch('dev.strategy.refresh_clock.observe', return_value={'cu_mhz': 1300, 'affects_verdict': False}):
            self.assertEqual(tool.main(), 0)
        frozen.assert_called_once_with(root.parent / 'source')
        self.assertEqual(calls, [(['pending'], 1)])
        self.assertEqual(before, {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()})
        result = read_review(next(output.glob('review-*.json.xz')))
        self.assertEqual(len(result['points']), 1)
        self.assertFalse(result['pending_points'])
        self.assertEqual(read_review(archive), previous)


if __name__ == '__main__':
    unittest.main()

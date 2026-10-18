"""Host-only coverage and fault injection; no synthetic GPU performance evidence."""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dev.llama_ab.contract import SCOPE, coverage, coverage_breakdown, family, inventory, judge, phase_points, pilot
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_kpack_llama_ab as runner


def host_samples(ratio=.8):
    return dict(correctness='PASS', scope=SCOPE, rounds=[
        [dict(arm=a, event_us=[10*(ratio if a else 1)]*3) for a in (0, 1, 1, 0)] for _ in range(6)])


class FullCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = inventory()

    def test_default_plan_schedules_every_point(self):
        out = io.StringIO()
        with patch('sys.argv', ['run_kpack_llama_ab.py', '--plan-only']), redirect_stdout(out):
            self.assertEqual(runner.main(), 0)
        plan = json.loads(out.getvalue().split('LLAMA_AB_PLAN ', 1)[1])
        self.assertEqual(plan['phase'], 'all')
        self.assertEqual((plan['points'], plan['scheduled_points'], plan['families']), (1544, 1544, 108))
        self.assertEqual(sum(r['expected'] for r in plan['breakdown']), 1544)
        self.assertEqual(plan['configs_per_shape'], 'PRODUCTION_SELECTOR_ONLY')
        binding = json.loads(Path('tools/kpack_strategy_wiring.json').read_text())
        self.assertEqual(binding['default_phase'], plan['phase'])
        self.assertEqual(binding['expected_native_points'], plan['scheduled_points'])
        self.assertEqual(binding['expected_native_pilot'], plan['pilot_points'])
        self.assertIn('AB_PHASE=${AB_PHASE:-all}', Path('tools/run_kpack_strategy_wiring_box.sh').read_text())

    def test_inventory_matches_both_complete_evidence_sets(self):
        small = json.loads(Path('policies/kpack_component_decode_v1.json').read_text())
        prefill = json.loads(Path('policies/kpack_prefill_confirmation_v1.json').read_text())
        expected = {p['point'] for r in small['rows'] for p in r['profiles']}
        expected.update(p['point'] for p in prefill['points'])
        actual = phase_points(self.rows, 'all')
        self.assertEqual({r['id'] for r in actual}, expected)
        self.assertEqual(Counter(r['cohort'] for r in actual), dict(decode=1114, prefill=430))
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(len(phase_points(self.rows, 'pilot')), 32)
        with self.assertRaises(ValueError):phase_points(self.rows, 'unknown')

    def test_failure_does_not_disappear_in_cohort_breakdown(self):
        records = {r['id']: dict(status='PASS') for r in self.rows}
        slow = self.rows[0]
        records[slow['id']]['status'] = 'REGRESSION'
        del records[self.rows[-1]['id']]
        total = coverage(self.rows, records)
        self.assertEqual(total['status'], 'NOT_ADMITTED')
        self.assertEqual(total['measured'], 1543)
        groups = coverage_breakdown(self.rows, records)
        self.assertEqual(sum(g['expected'] for g in groups), 1544)
        self.assertEqual(sum(g['measured'] for g in groups), 1543)
        self.assertEqual(sum(g['counts'].get('REGRESSION', 0) for g in groups), 1)
        self.assertIn(self.rows[-1]['id'], [k for g in groups for k in g['missing']])

    def test_resume_preserves_valid_regressions_and_inconclusive_samples(self):
        rows = self.rows[:5]
        request = dict(points=rows, shards=[dict(hardware={}, families=sorted({family(r) for r in rows}))])
        authority = dict(request=digest(request), hardware={}, shard=0)
        with tempfile.TemporaryDirectory() as td:
            records = Path(td)
            for row, ratio in zip(rows[:3], (.8, 1.2, 1)):
                data = host_samples(ratio)
                if ratio == 1:data['rounds'][0][1]['event_us'] = [12]*3
                result = judge(data)
                runner.save_record(records/(row['id']+'.json'), dict(result, id=row['id'], data=data, authority=authority))
            runner.save_record(records/(rows[3]['id']+'.json'), dict(id=rows[3]['id'], authority=authority, status='FAIL'))
            before = {p: p.read_bytes() for p in records.iterdir()}
            pending = runner.pending_groups(request, 0, 'all', records)
            self.assertEqual({r['id'] for group in pending.values() for r in group}, {r['id'] for r in rows[3:]})
            self.assertEqual(before, {p: p.read_bytes() for p in records.iterdir()})


class WorkerFaultTest(unittest.TestCase):
    def check_fault(self, setup_failure):
        all_rows = inventory()
        bad = next(r for r in pilot(all_rows) if r['cohort']=='decode' and r['workload']['operation']=='dense')
        peers = [r for r in all_rows if family(r)==family(bad) and r['id']!=bad['id']][:2]
        other = next(r for r in all_rows if family(r)!=family(bad))
        rows = [bad, *peers, other]
        attempted, invocations = [], []
        fail = [bad['id']]
        hardware = dict(device={'pci':'HOST_ONLY_LLAMA_AB_FAULT_TEST'}, l2={'bytes':1})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root/'llama-ab-carrier';binary.write_text('HOST TEST ONLY, NOT EXECUTABLE')
            request = dict(points=rows, harness='host-test', sources={}, carrier={'binary_sha256':sha(binary)},
                           shards=[dict(hardware=hardware, families=sorted({family(r) for r in rows}))])
            (root/'request.json').write_text(json.dumps(request))
            args = SimpleNamespace(output=root, worker=0, phase='all', hours=0, rounds=6, samples=3, sdk=Path('/host-test-sdk'))

            class FakeProcess:
                def __init__(self, command, env, stdout, stderr):
                    cfg = json.loads(Path(command[1]).read_text())
                    points = cfg['points'];invocations.append([r['id'] for r in points])
                    self.returncode = 0
                    if setup_failure and fail and any(r['id'] in fail for r in points):
                        self.returncode = 1
                        stdout.write('HOST_TEST_SETUP_FAILURE\n')
                        return
                    for row in points:
                        attempted.append(row['id'])
                        stdout.write('LLAMA_AB_START id='+row['id']+'\n')
                        if row['id'] in fail:
                            self.returncode = 1
                            break
                        data = dict(host_samples(), id=row['id'], workload=row['workload'],
                                    authority=cfg['authority'], ring={'copies':1})
                        runner.atomic_json(Path(command[2])/(row['id']+'.json'), data)

                def poll(self):return self.returncode

            with ExitStack() as stack:
                for name, value in [('device', lambda a: hardware), ('harness_identity', lambda root: 'host-test'),
                                    ('environment', lambda a: {}), ('point_request', lambda row,l2: row),
                                    ('route_proof', lambda text,row: ['tensor=ab_'+row['id'][:20]+'_0']),
                                    ('validate_loaded', lambda data,request: {})]:
                    stack.enter_context(patch.object(runner, name, value))
                stack.enter_context(patch.object(runner.subprocess, 'Popen', FakeProcess))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(runner.worker(args), 0)
                records = root/'records'
                result = {p.stem:json.loads(p.read_text()) for p in records.glob('*.json')}
                self.assertEqual(set(result), {r['id'] for r in rows})
                failed = {r['id'] for r in [bad, *peers]} if setup_failure else {bad['id']}
                self.assertEqual({k for k,v in result.items() if v['status']=='FAIL'}, failed)
                self.assertEqual(result[other['id']]['status'], 'PASS')
                if not setup_failure:
                    self.assertEqual(Counter(attempted), Counter(r['id'] for r in rows))
                    self.assertEqual(len(invocations), 3)  # crashed family, fresh peers, other family
                valid = {p:p.read_bytes() for p in records.glob('*.json') if p.stem not in failed}
                first_logs = sorted(root.glob('shard-0/*/run.log'))
                fail.clear();attempted.clear();invocations.clear()
                self.assertEqual(runner.worker(args), 0)
                self.assertEqual(set(attempted), failed)
                self.assertEqual(valid, {p:p.read_bytes() for p in valid})
                self.assertTrue(all(p.exists() for p in first_logs))
                self.assertTrue(all(json.loads(p.read_text())['status']=='PASS' for p in records.glob('*.json')))

    def test_pilot_point_abort_does_not_stop_full_set(self):
        self.check_fault(setup_failure=False)

    def test_family_setup_failure_does_not_stop_other_families(self):
        self.check_fault(setup_failure=True)


class CoordinatorTest(unittest.TestCase):
    def test_full_mode_launches_each_worker_once_and_retains_failed_exit(self):
        all_rows = inventory()
        first = all_rows[0]
        rows = [first, next(r for r in all_rows if family(r)!=family(first))]
        for worker_rc in (0, 1):
            with self.subTest(worker_rc=worker_rc), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                for folder in ('llama', 'build', 'sdk'):(root/folder).mkdir()
                output = root/'output'
                launches = []

                def build_carrier(args, path):
                    binary = path/'llama-ab-carrier';binary.write_text('HOST TEST CARRIER')
                    receipt = dict(identity={'libraries':{}}, binary_sha256=sha(binary))
                    runner.atomic_json(path/'carrier.json', receipt)
                    (path/'carrier-build.log').write_text('HOST TEST: NO GPU COMPILATION')
                    return binary, receipt

                def probe(command, env, **kwargs):
                    return json.dumps(dict(device=dict(host='host-test', pci=env['CUDA_VISIBLE_DEVICES']), l2={'bytes':1}))

                class FakeWorker:
                    def __init__(self, command, **kwargs):
                        index = int(command[command.index('--worker')+1])
                        launches.append((index, command[command.index('--phase')+1]))
                        request = json.loads((output/'request.json').read_text())
                        shard = request['shards'][index]
                        authority = dict(request=digest(request), hardware=shard['hardware'], shard=index)
                        for row in request['points']:
                            if family(row) not in shard['families']:continue
                            data = dict(host_samples(), point_seconds=1, setup_family_seconds=1)
                            runner.save_record(output/'records'/(row['id']+'.json'),
                                               dict(judge(data), id=row['id'], data=data, authority=authority))
                        self.returncode = worker_rc if index == 0 else 0

                    def poll(self):return self.returncode

                argv = ['run_kpack_llama_ab.py', '--llama-source', str(root/'llama'),
                        '--llama-build', str(root/'build'), '--sdk', str(root/'sdk'),
                        '--devices', '6,7', '--output', str(output), '--idle-confirmed']
                with ExitStack() as stack:
                    stack.enter_context(patch('sys.argv', argv))
                    stack.enter_context(patch.object(runner, 'inventory', lambda: rows))
                    stack.enter_context(patch.object(runner, 'build_carrier', build_carrier))
                    stack.enter_context(patch.object(runner, 'runtime_payloads', lambda args: []))
                    stack.enter_context(patch.object(runner, 'harness_identity', lambda root: 'HOST_TEST'))
                    stack.enter_context(patch.object(runner.subprocess, 'check_output', probe))
                    stack.enter_context(patch.object(runner.subprocess, 'Popen', FakeWorker))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    self.assertEqual(runner.main(), worker_rc)
                self.assertEqual(launches, [(0, 'all'), (1, 'all')])
                result = json.loads((output/'summary.json').read_text())
                self.assertEqual(result['measured'], len(rows))
                self.assertEqual(result['status'], 'NOT_ADMITTED' if worker_rc else 'PASS')
                self.assertEqual(result['worker_exit_codes'], {'0':worker_rc, '1':0})
                self.assertEqual(sum(g['expected'] for g in result['breakdown']), len(rows))
                with tarfile.open(next(output.glob('review-*.tgz'))) as archive:
                    names = archive.getnames()
                    self.assertIn('summary.json', names)
                    self.assertEqual(sum(n.startswith('records/') for n in names), len(rows))
                    self.assertNotIn('llama-ab-carrier', names)


if __name__=='__main__':unittest.main()

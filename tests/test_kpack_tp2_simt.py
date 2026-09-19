import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools import run_kpack_tp2_simt as probe


def log(arm, fixture, fail=False):
    lines = ['Q4_TP2_INPUT ' + ' '.join(f'{k}={v}' for k, v in
        zip(('raw','low','units','a','ids','golden'), fixture['field_fnv']))]
    errors = 0
    for producer in ('HOST','GPU'):
        lines.append(f'Q4_TP2_PACK producer={producer} low_bad=0 units_bad=0')
        for reader in (('simt','scalar') if arm=='fresh' else ('simt',)):
            for compute in ('F16','BF16'):
                bad = fail and reader=='simt' and compute=='BF16'
                errors += bad
                lines.append(f'Q4_TP2_LOCAL reader={reader} producer={producer} compute={compute} '
                    'variant=0 columns=4 warps=4 values=4 split=1 '
                    + ('relative=0.1 max_abs=0.03 nonfinite=0 bad=32/1024 status=FAIL' if bad else
                       'relative=1e-7 max_abs=8e-8 nonfinite=0 bad=0/1024 status=PASS'))
    lines += ['Q4_TP2_NEGATIVE kind=wrong_expert relative=1.2 status=EXPECTED_RED',
              f'Q4_TP2_COMPLETE arm={arm} cells={8 if arm=="fresh" else 4} failures={errors}']
    return '\n'.join(lines)+'\n'


class Tp2Simt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='q4-tp2-host-')
        cls.fixtures = probe.unpack(Path(cls.temp.name)/'fixtures')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fixture_has_exact_box_goldens_and_rank_denominator(self):
        self.assertEqual(len(self.fixtures),6)
        self.assertEqual(self.fixtures[0]['field_fnv'][0], 'df1668dc7ea3b843')
        self.assertEqual(self.fixtures[3]['field_fnv'][0], 'a29e1d9b60fed398')
        self.assertEqual(self.fixtures[0]['field_fnv'][-1], '3b28bc5ab3c46aac')
        self.assertEqual(self.fixtures[3]['field_fnv'][-1], '0947ec09749102d0')
        for f in self.fixtures:
            self.assertEqual(probe.sha(Path(self.temp.name)/'fixtures'/f['file']),f['sha256'])

    def test_wrong_fixture_archive_rejected_before_extraction(self):
        with patch.object(probe,'FIXTURE_SHA','0'*64), self.assertRaisesRegex(ValueError,'archive differs'):
            probe.unpack(Path(self.temp.name)/'rejected')
        self.assertFalse((Path(self.temp.name)/'rejected').exists())

    def test_positive_and_real_numerical_failure_are_complete(self):
        f = self.fixtures[0]
        for arm in ('fresh','shipped'):
            self.assertEqual(probe.evidence(log(arm,f),arm,f,0)['status'],'PASS')
            self.assertEqual(probe.evidence(log(arm,f,True),arm,f,1)['status'],'NUMERIC_MISMATCH')

    def test_nonfinite_failure_is_preserved_as_valid_json(self):
        f = self.fixtures[0]
        text = log('fresh', f, True).replace('relative=0.1 max_abs=0.03 nonfinite=0',
                                            'relative=nan max_abs=inf nonfinite=32')
        result = probe.evidence(text, 'fresh', f, 1)
        json.dumps(result, allow_nan=False)
        failures = [c for c in result['cells'] if c['status']=='FAIL']
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(c['relative'] is None and c['max_abs'] is None for c in failures))

    def test_missing_duplicate_wrong_hash_and_nan_negatives(self):
        f = self.fixtures[0]
        text = log('fresh',f)
        mutations = [
            text.replace('cells=8','cells=7'),
            text.replace('compute=BF16','compute=F16'),
            text.replace('reader=scalar','reader=simt'),
            text.replace('variant=0','variant=3'),
            text.replace('warps=4','warps=2'),
            text.replace('producer=HOST','producer=GPU'),
            text.replace(f['field_fnv'][0],'0'*16),
            text.replace('relative=1e-7','relative=nan'),
            text.replace('max_abs=8e-8','max_abs=nan'),
            text.replace('relative=1.2','relative=nan'),
            text.replace('relative=1.2','relative=0.01'),
            text.replace('status=EXPECTED_RED','status=FAIL'),
            text + text.splitlines()[1]+'\n',
            text.replace('nonfinite=0','nonfinite=1'),
            text.replace('status=PASS','status=FAIL'),
        ]
        for index, bad in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                probe.evidence(bad,'fresh',f,0)
        with self.assertRaises(ValueError):
            probe.evidence(text,'fresh',f,1)

    def test_adjudication_retains_every_boundary(self):
        rows = [dict(arm=arm,fixture=f['file'],**probe.evidence(log(arm,f),arm,f,0))
                for arm in ('shipped','fresh') for f in self.fixtures]
        self.assertEqual(probe.verdict(rows),'STANDALONE_NOT_REPRODUCED')
        self.assertEqual(probe.verdict(rows[:-1]),'INCOMPLETE')
        self.assertEqual(probe.verdict(rows[:-1]+[rows[0]]),'INCOMPLETE')
        for index,reader,want in [(0,'simt','SHIPPED_SIMT_ONLY_FAILURE'),
                                  (6,'simt','FRESH_SIMT_COMPUTE_FAILED'),
                                  (6,'scalar','SCALAR_CANONICAL_COMPUTE_FAILED')]:
            case = copy.deepcopy(rows)
            cell = next(c for c in case[index]['cells'] if c['reader']==reader)
            cell.update(status='FAIL',bad=32,relative=.1,max_abs=.1)
            case[index]['status']='NUMERIC_MISMATCH'
            self.assertEqual(probe.verdict(case),want)
        for producer, want in [('HOST','HOST_TRANSFER_DIFFERED'),('GPU','GPU_PACK_BYTES_DIFFER')]:
            case = copy.deepcopy(rows)
            case[0]['packs'] = [(p,'1' if p==producer else lo,u) for p,lo,u in case[0]['packs']]
            self.assertEqual(probe.verdict(case),want)

    def test_shipped_caller_is_built_without_a_device_image(self):
        with tempfile.TemporaryDirectory(prefix='q4-build-plan-') as d:
            root = Path(d)
            sdk = root/'sdk'
            (sdk/'bin').mkdir(parents=True)
            (sdk/'bin/hgcc').write_bytes(b'compiler')
            calls = []
            def fake(argv, output, timeout=180):
                argv = list(map(str,argv))
                calls.append(argv)
                Path(argv[argv.index('-o')+1]).write_bytes(b'host command contract only')
                return dict(command=argv,rc=0,seconds=0)
            with patch.object(probe,'command',side_effect=fake):
                probe.build(sdk,root/'build',192)
            shipped = next(c for c in calls if '-DQTP_SHIPPED_ONLY=1' in c)
            self.assertEqual(shipped[0],'g++')
            link = next(c for c in calls if c[-1]==str(root/'build/shipped'))
            self.assertNotIn(str(root/'build/fresh.o'),link)
            self.assertNotIn(str(root/'build/pack.o'),link)
            self.assertTrue(any(c[0]==str(sdk/'bin/hgcc') for c in calls))
            self.assertIn(f'-I{sdk}/include',shipped)
            self.assertIn(f'-I{sdk}/targets/x86_64-linux/include',shipped)

    def test_a_failed_process_does_not_skip_remaining_fixtures(self):
        with tempfile.TemporaryDirectory(prefix='q4-processes-') as d:
            root = Path(d)
            args = SimpleNamespace(sdk=root, output=root/'run', jobs=4, compile_only=False)
            calls = []
            def fake(argv, output, timeout=180):
                arm, filename = str(argv[2]), Path(argv[1]).name
                calls.append((arm, filename))
                if len(calls)==1:
                    output.write_text('runtime failed before output\n')
                    return dict(rc=2)
                fixture = next(f for f in self.fixtures if f['file']==filename)
                output.write_text(log(arm, fixture))
                return dict(rc=0)
            with patch.object(probe, 'bundle_paths', return_value=(root/'pack', root/'execution', {})), \
                    patch.object(probe, 'build', return_value=root/'build'), \
                    patch.object(probe, 'command', side_effect=fake), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(args), 1)
            summary = json.loads((root/'run/summary.json').read_text())
            self.assertEqual(len(calls), 12)
            self.assertEqual(summary['verdict'], 'INCOMPLETE')
            self.assertEqual(sum(c['status']=='PASS' for c in summary['cases']), 11)

    def test_runner_uses_one_device_and_no_llama_or_jit_build(self):
        path = probe.ROOT/'tools/run_kpack_tp2_simt_box.sh'
        subprocess.run(['bash','-n',str(path)],check=True)
        text = path.read_text()
        self.assertIn('CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}',text)
        self.assertIn('unittest discover -s tests',text)
        self.assertNotIn('build_kpack_model_ci',text)
        self.assertNotIn('git lfs pull',text)
        self.assertNotIn('kpack_jit.py',text)


if __name__=='__main__':
    unittest.main()

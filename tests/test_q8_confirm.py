"""Host-only supplement gates; dummy files are not GPU evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('q8_confirm', ROOT / 'tools/confirm_q8_dense.py')
confirm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirm)


def fixture(run):
    from dev.gemv_simt.q8_address_plan import plan
    source = run / 'source'
    output = run / 'results'
    (source / 'tools').mkdir(parents=True)
    (source / 'tools/run_q8_dense.py').write_text('raise RuntimeError("plan-only must not import")\n')
    (source / 'kernel.cuh').write_text('// dummy host fixture\n')
    (output / 'build/candidate').mkdir(parents=True)
    harness = {'tools/run_q8_dense.py': confirm.sha(source / 'tools/run_q8_dense.py')}
    authority = dict(harness=dict(files=harness, sha256=confirm.digest(harness)))
    rows = []
    for compute in ('f16', 'bf16'):
        image = output / f'build/candidate/{compute}.so'
        image.write_bytes(b'dummy-not-a-shared-library')
        rows.append(dict(compute=compute, image=image.name, sha256=confirm.sha(image)))
    build = dict(object_only=False, sources={'kernel.cuh': confirm.sha(source / 'kernel.cuh')},
                 records=rows)
    for name, value in {
            'plan.json': plan(), 'authority.json': authority,
            'summary.json': dict(status='PASS', production_changed=False),
            'build/candidate/manifest.json': build}.items():
        (output / name).write_text(json.dumps(value))
    return source, output


@pytest.mark.parametrize('cohort,reports,geometries,points', [
    ('m1', 180, 10, 8), ('m8', 90, 5, 4), ('all', 270, 15, 8)])
def test_plan_only_is_read_only_and_needs_no_gpu(tmp_path, cohort, reports, geometries, points):
    fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = subprocess.run([sys.executable, str(ROOT / 'tools/confirm_q8_dense.py'),
        '--run', str(tmp_path), '--cohort', cohort, '--plan-only'], check=True, capture_output=True, text=True)
    row = json.loads(result.stdout.split('Q8_CONFIRM_PLAN ', 1)[1])
    assert row['acu_reports'] == reports and row['new_compiles'] == 0 and not row['cold_timing_rerun']
    assert len(row['focuses']) == geometries and len({f['point'] for f in row['focuses']}) == points
    assert row['cohort'] == cohort
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('path', ['source/tools/run_q8_dense.py', 'source/kernel.cuh',
                                 'results/build/candidate/f16.so'])
def test_mutated_frozen_input_rejected(tmp_path, path):
    fixture(tmp_path)
    (tmp_path / path).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        confirm.parent(tmp_path)


def test_incomplete_or_wrong_experiment_rejected(tmp_path):
    _, output = fixture(tmp_path)
    (output / 'summary.json').write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError, match='completed Q8'):
        confirm.parent(tmp_path)
    (output / 'summary.json').write_text('{"status":"PASS"}')
    (output / 'plan.json').write_text('{"schema":"q8-dense-ablation-v1"}')
    with pytest.raises(ValueError, match='completed Q8'):
        confirm.parent(tmp_path)


def test_no_escape_or_linked_image(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match='invalid parent'):
        confirm.contained(tmp_path, '../elsewhere')
    image = tmp_path / 'results/build/candidate/f16.so'
    outside = tmp_path / 'other.so'
    outside.write_bytes(image.read_bytes())
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(ValueError, match='non-local'):
        confirm.parent(tmp_path)


@pytest.mark.parametrize('cohort', ['m1', 'm8', 'all'])
def test_supplement_only_calls_frozen_profiler_and_preserves_parent(tmp_path, monkeypatch, cohort):
    source, output = fixture(tmp_path)
    frozen = json.loads((output / 'plan.json').read_bytes())
    authority = dict(expected_ce_hz=1.7e9,
                     device=dict(l2=dict(bytes=67108864), device=dict(pci='unit-test-q8-only')))
    sdk = tmp_path / 'sdk'
    sdk.mkdir()
    acu = sdk / 'acu'
    acu.write_bytes(b'not-executed')
    before = {name: (output / name).read_bytes() for name in ('plan.json', 'authority.json', 'summary.json')}
    calls = []
    runner = SimpleNamespace(
        __file__=str(source / 'tools/run_q8_dense.py'),
        configure=lambda mode: calls.append(('configure', mode)),
        plan=lambda: frozen,
        binding=lambda args: calls.append(('binding', args.output)),
        complete_point=lambda out, point, auth: calls.append(('numeric-reuse', point)),
        profile_focus=lambda args, focus, attempt, auth:
            calls.append(('profile', focus)) or dict(focus=focus, status='PASS'),
        sealed=lambda path, result: path.write_text(json.dumps(result)),
    )
    real_import = importlib.import_module
    monkeypatch.setattr(confirm.importlib, 'import_module',
                        lambda name: runner if name == 'tools.run_q8_dense' else real_import(name))
    from tools import pack_simt_split_review
    monkeypatch.setattr(pack_simt_split_review, 'package', lambda out: calls.append(('pack', out)))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    a = SimpleNamespace(sdk=sdk, acu=acu, expected_ce_ghz=1.7, cohort=cohort)
    assert confirm.execute(a, source, output, frozen, authority) == 0
    focuses = confirm.selected_focuses(cohort)
    assert [value for kind, value in calls if kind == 'profile'] == list(focuses)
    assert [value for kind, value in calls if kind == 'numeric-reuse'] == sorted({f['point'] for f in focuses})
    assert calls[-1] == ('pack', output)
    assert all((output / name).read_bytes() == content for name, content in before.items())
    saved, = output.glob('attempts/finalists-*/confirmation.json')
    result = json.loads(saved.read_bytes())
    assert result['status'] == 'PASS' and result['new_compiles'] == 0
    assert result['cohort'] == cohort and result['requested_acu_reports'] == len(focuses)*18
    assert not result['cold_timing_rerun'] and not result['production_changed']


def test_m1_covers_every_shape_and_both_precision_controls_first():
    assert {f['point'] for f in confirm.M1_FOCUSES} == set(range(8))
    assert all(f['m'] == 1 for f in confirm.M1_FOCUSES)
    assert confirm.selected_focuses('all')[:10] == confirm.M1_FOCUSES
    assert len({(f['point'], f['m'], f['config']) for f in confirm.selected_focuses('all')}) == 15
    expected = {0: {1}, 1: {1}, 2: {0, 1}, 3: {0, 1}, 4: {1}, 5: {0}, 6: {1}, 7: {1}}
    assert {p: {f['config'] for f in confirm.M1_FOCUSES if f['point'] == p} for p in range(8)} == expected
    with pytest.raises(ValueError, match='cohort'):
        confirm.selected_focuses('unknown')


def test_bf16_control_is_not_substituted_with_f16(tmp_path):
    _, output = fixture(tmp_path)
    frozen = json.loads((output/'plan.json').read_bytes())
    frozen['points'][6]['workload']['compute'] = 'f16'
    (output/'plan.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='workload differs'):
        confirm.parent(tmp_path, 'm1')

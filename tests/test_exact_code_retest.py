import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.decode_gemv.retest_extension import load_cases, sources, specialization, verdict


@pytest.mark.parametrize('delivery,count,excluded', [('direct', 46, 0), ('staged', 29, 2)])
def test_frozen_positive_inventory(delivery, count, excluded):
    cases, omissions = load_cases(delivery, Path('/new/test.so'))
    assert len(cases) == count and len(omissions) == excluded
    assert all(c['proto']['lib'] == '/new/test.so' for c in cases)
    units = sources(cases, delivery)
    generated = '\n'.join(text for name, text in units.items() if name.endswith('.cu'))
    assert generated.count('extern "C" int') == len({specialization(c) for c in cases})
    assert 'qgemv_mma_run' in units['router.cpp']
    assert all(specialization(c)[6] == 1 for c in cases if c['kind'] == 'moe')


def test_verdict_requires_actual_oracle_and_complete_coverage():
    cases = [dict(id='a'), dict(id='b')]
    rows = [dict(case=c, status='OK', check=dict(cells=8, bad=0, nonfinite=0,
            tolerance=0.005, relative_l1_error=1e-7)) for c in cases]
    assert verdict(cases, rows)['status'] == 'PASS'
    assert verdict(cases, rows, 1)['status'] == 'FAIL'
    assert verdict(cases, rows[:1])['missing'] == ['b']
    assert verdict(cases, [dict(case=cases[0], status='OK'), rows[1]])['status'] == 'FAIL'
    rows[0]['check']['relative_l1_error'] = float('nan')
    assert verdict(cases, rows)['status'] == 'FAIL'
    with pytest.raises(ValueError, match='duplicate'):
        verdict(cases, [rows[0], rows[0]])


def test_plan_is_not_reported_as_device_pass(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'plan'
    result = subprocess.run([sys.executable, str(root/'tools/decode_gemv/retest_extension.py'),
                             '--plan-only', '--sdk', '/missing-sdk', '--output', str(output)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'PLAN_ONLY'
    assert summary['suites'] == {} and not summary['performance_admitted']
    with tarfile.open(output/'review.tgz') as archive:
        assert 'summary.json' in archive.getnames()
        assert not any(name.endswith(('.so', '.o', '.acurep', '.asysrep')) for name in archive.getnames())

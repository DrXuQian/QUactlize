"""Host-only subprocess accounting tests; fixture passes are not GPU evidence."""
import os
import subprocess

import pytest

from tools.verify_installed_wheel import device_tests, pytest_gate


@pytest.mark.parametrize('body,accepted', [
    ('assert True', True),
    ('pytest.skip("no hardware")', False),
    ('assert False', False),
])
def test_real_pytest_results_reject_skips_and_failures(tmp_path, body, accepted):
    (tmp_path / 'test_fixture.py').write_text('import pytest\ndef test_fixture():\n    ' + body + '\n')
    if accepted:
        result = pytest_gate(('test_fixture.py',), cwd=tmp_path, env=os.environ)
        assert result == dict(tests=1, failures=0, errors=0, skipped=0)
    else:
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            pytest_gate(('test_fixture.py',), cwd=tmp_path, env=os.environ)


def test_insufficient_case_count_is_not_qualification(tmp_path):
    (tmp_path / 'test_fixture.py').write_text('def test_fixture():\n    pass\n')
    with pytest.raises(RuntimeError, match='at least 10'):
        pytest_gate(('test_fixture.py',), cwd=tmp_path, env=os.environ, minimum_tests=10)


def test_runner_sets_device_opt_in_and_verified_library_without_inherited_selection(tmp_path, monkeypatch):
    source, flare = tmp_path / 'source', tmp_path / 'flare'
    for folder in ('tests/ppu', 'tests/serving', 'ci/support', 'reference'):
        (source / folder).mkdir(parents=True)
    for filename in ('tests/ppu/test_packed_native.py', 'tests/ppu/test_compat_library.py',
                     'tests/serving/test_serving_device.py'):
        (source / filename).write_text(
            'import os\ndef test_host_fixture():\n'
            '    assert os.environ["QUACTLIZE_SERVING_LIBRARY"] == "/verified/operator.so"\n')
    (flare / 'tests/preparation').mkdir(parents=True)
    (flare / 'tests/preparation/test_gguf_packing_cuda.py').write_text(
        'import os, pytest\n@pytest.mark.parametrize("index", range(10))\n'
        'def test_host_fixture(index):\n'
        '    assert os.environ["FLARE_RUN_CUDA_TESTS"] == "1"\n'
        '    assert os.environ["FLARE_TEST_PPU_MATMUL_LIBRARY"] == "/verified/operator.so"\n')
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k does_not_exist')
    counts = device_tests(source, '/verified/operator.so', flare)
    assert counts['operators']['tests'] == 3
    assert counts['flare']['tests'] == 10
    assert counts['operators']['skipped'] == counts['flare']['skipped'] == 0

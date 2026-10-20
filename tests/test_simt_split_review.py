"""Transfer-only tests; dummy data is not GPU/performance evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.pack_simt_split_review import package, result_root


def fixture(root):
    retained = {
        'multi.json': b'{"schema":"simt-split-multigpu-v1"}',
        'summary.json': b'{"status":"INCOMPLETE","failed_shards":[1]}',
        'gpu-1/points/0/m1-c0-s2.json': b'{"samples":[1,2,3],"status":"FAIL","access_path":"access/a.json"}',
        'gpu-1/attempts/1/r0-a0.raw.csv': b'kernel,cycles\nbaseline,100\n',
        'gpu-1/attempts/1/failure.log': b'compiler error \xff\n',
        'build/10/manifest.json': b'{"sha256":"unit-test-only"}',
        'runner-status.txt': b'failed: keep errors visible\n',
    }
    omitted = ['sample.acurep', 'gpu-1/attempts/1/r0-a0', 'build/10/f16.isa.txt',
               'build/10/f16.resources.txt', 'gpu-1/access/a.json',
               'build/10/test.so', 'build/10/test.o', 'weights.bin', 'review-old.tgz']
    for name, data in {**retained, **dict.fromkeys(omitted, b'omit-only-dummy-data')}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return retained, omitted


def test_original_bytes_samples_failures_and_hashes_preserved(tmp_path):
    retained, omitted = fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    archive = package(tmp_path)
    with tarfile.open(archive) as reader:
        assert set(reader.getnames()) == set(retained) | {'transfer-manifest.json'}
        manifest = json.load(reader.extractfile('transfer-manifest.json'))
        assert not manifest['gpu_work'] and not manifest['raw_results_modified']
        assert not manifest['verdicts_recomputed']
        for row in manifest['included']:
            data = reader.extractfile(row['path']).read()
            assert data == retained[row['path']]
            assert hashlib.sha256(data).hexdigest() == row['sha256']
        assert sum(row['files'] for row in manifest['omitted'].values()) == len(omitted)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_run_results_or_archive_locator(tmp_path):
    root = tmp_path / 'results'
    fixture(root)
    for source in (tmp_path, root, root / 'review-old.tgz'):
        assert result_root(source) == root
    (root / 'multi.json').write_text('{"schema":"wrong"}')
    with pytest.raises(ValueError, match='original Split-K results'):
        result_root(root / 'review-old.tgz')
    (root / 'plan.json').write_text('{"schema":"simt-common-split-v1"}')
    assert result_root(root) == root


@pytest.mark.parametrize('schema', ['q8-dense-ablation-v1', 'q8-address-window-v1'])
def test_q8_evidence_without_full_reports(tmp_path, schema):
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'plan.json').write_text(json.dumps({'schema': schema}))
    for arm in range(6):
        (root / f'r0-a{arm}').write_bytes(b'dummy-large-report')
        (root / f'r0-a{arm}.acurep').write_bytes(b'dummy-large-report')
        (root / f'r0-a{arm}.raw.csv').write_bytes(b'kernel,cycles\ncontrol,100\n')
    assert result_root(tmp_path) == root
    archive = package(root)
    with tarfile.open(archive) as reader:
        assert set(reader.getnames()) == {
            'plan.json', 'transfer-manifest.json',
            *(f'r0-a{arm}.raw.csv' for arm in range(6))}


def test_no_link_traversal_or_overwrite(tmp_path):
    root = tmp_path / 'results'
    fixture(root)
    outside = tmp_path / 'external'
    outside.mkdir()
    (outside / 'private.json').write_bytes(b'do-not-include')
    (root / 'linked').symlink_to(outside, target_is_directory=True)
    (root / 'linked.json').symlink_to(outside / 'private.json')
    archive = package(root)
    with tarfile.open(archive) as reader:
        assert 'linked/private.json' not in reader.getnames()
        assert 'linked.json' not in reader.getnames()
        assert all(member.isfile() for member in reader.getmembers())
    before = archive.read_bytes()
    with pytest.raises(FileExistsError):
        package(root, archive)
    assert archive.read_bytes() == before


def test_standalone_cli_needs_no_repo_imports_or_sdk(tmp_path):
    fixture(tmp_path / 'run/results')
    script = tmp_path / 'pack.py'
    script.write_bytes((Path(__file__).resolve().parents[1] /
                        'tools/pack_simt_split_review.py').read_bytes())
    result = subprocess.run([sys.executable, '-I', str(script), str(tmp_path / 'run')],
                            capture_output=True, text=True, check=True)
    archive = Path(next(line[8:] for line in result.stdout.splitlines() if line.startswith('results=')))
    assert archive.is_file() and archive.name.endswith('.tar.xz')
    failed = subprocess.run([sys.executable, '-I', str(script), str(tmp_path)],
                            capture_output=True, text=True)
    assert failed.returncode and 'original Split-K results' in failed.stderr

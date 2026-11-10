import importlib.util
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('zw890_probe',ROOT/'tools/run_zw890_bringup.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

def test_scope_is_not_production_admission():
    p=probe.plan()
    assert p['target']=='ppu_15'
    assert not p['production_changed'] and not p['performance_admitted']
    assert sum(p['expected'].values())==192
    assert 'PRODUCTION_FQ_SF_TC_NOT_PORTED' in p['unsupported']

def test_missing_duplicate_and_failed_cases_cannot_pass():
    rows=[dict(suite='delivery',case=i,status='PASS') for i in range(8)]
    assert probe.verdict('delivery',rows,0)['status']=='PASS'
    assert probe.verdict('delivery',rows[:-1],0)['status']=='FAIL'
    assert probe.verdict('delivery',rows[:-1]+rows[:1],0)['status']=='FAIL'
    assert probe.verdict('delivery',rows,1)['status']=='FAIL'
    rows[0]['status']='FAIL'
    assert probe.verdict('delivery',rows,0)['status']=='FAIL'

def test_review_contains_only_receipts_and_logs(tmp_path):
    output=tmp_path/'run';output.mkdir()
    (output/'summary.json').write_text(json.dumps(probe.plan()))
    (output/'error.log').write_text('diagnostic')
    (output/'kernel.o').write_bytes(b'not for upload')
    (output/'libqz15_probe.so').write_bytes(b'not for upload')
    path=probe.review(output)
    with tarfile.open(path) as archive:
        assert sorted(archive.getnames())==['error.log','summary.json']

def test_product_does_not_reference_the_probe():
    for file in (ROOT/'quactlize').rglob('*'):
        if file.is_file() and file.suffix in ('.py','.h','.cu','.hpp','.cuh','.cpp'):
            text=file.read_text()
            assert 'tools/ppu15' not in text and 'qz15_simt' not in text

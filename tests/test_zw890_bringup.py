import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import pytest

from tools.ppu15.device import validate_identity

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


def test_a_reader_repair_and_exact_legacy_negative_remain_bound():
    native=(ROOT/'tools/ppu15/native.cu').read_text()
    assert 'constexpr bool Swap=Trans || LegacyA;' in native
    assert 'Int<Rows>,Int<Cols>,Swap>' in native
    assert 'native_delivery<false,32,true>' in native
    assert 'PPU0015_TSM_LD_SWZL<E,16,Width,Swap,Trans,1>' in native
    check=(ROOT/'tools/ppu15/check.py').read_text()
    assert "bind(lib,'qz15_delivery_legacy'" in check
    assert 'np.testing.assert_array_equal(old,wrong)' in check
    assert 'mismatches!=768' in check


def test_quadrant_swap_matches_reported_raw_bit_signature():
    # No GPU/NumPy or tolerance: tags identify their exact original word.
    bad=0
    maximum=0
    for stage in range(3):
        for row in range(16):
            for col in range(32):
                rr=row%8+8*((col%16)//8)
                cc=16*(col//16)+col%8+8*(row//8)
                got=37*(stage*512+rr*32+cc)
                want=37*(stage*512+row*32+col)
                bad+=got!=want
                maximum=max(maximum,abs(got-want))
                if stage==row==0 and col==31:
                    assert (want,got)==(1147,10323)
    assert bad==768 and maximum==9176


@pytest.mark.parametrize('name', ['ZW890', 'PPU-ZW890', 'ZW-M890P'])
def test_zw890_runtime_names(name):
    # Real ZW-M890P reports a CUDA compatibility capability of8.9, not1.5.
    validate_identity(dict(name=name, warp_size=32, major=8, minor=9,
                           compute_units=39))


@pytest.mark.parametrize('name', [
    '', 'unknown PPU', 'PPU-ZW810', 'ZW8900', 'ZW890-extra',
    'ZW-M890', 'ZW-M890P0', 'ZW-M890P-extra', 'NVIDIA RTX 4090',
])
def test_other_devices_and_similar_names_are_not_admitted(name):
    with pytest.raises(ValueError, match='requires one visible'):
        validate_identity(dict(name=name, warp_size=32, major=8, minor=9))


@pytest.mark.parametrize('warp', [None, 0, 16, 64])
def test_alias_does_not_relax_warp_check(warp):
    with pytest.raises(ValueError, match='warp size32'):
        validate_identity(dict(name='ZW-M890P', warp_size=warp))


def test_probe_guards_and_name_authority_remain_bound():
    native = (ROOT/'tools/ppu15/native.cu').read_text()
    assert '__HGGC_ARCH__ != 150' in native
    assert 'if(count!=1) return -1;' in native
    assert 'tools/ppu15/device.py' in probe.FIELDS
    assert 'tools/hgcc_identity.py' in probe.FIELDS
    check = (ROOT/'tools/ppu15/check.py').read_text()
    assert check.index("checked(device(name,pci,values)") < check.index('validate_identity(identity)')


@pytest.mark.parametrize('native_status', [0, -1])
def test_actual_setup_cannot_skip_native_single_device_failure(monkeypatch, tmp_path, native_status):
    from tools.ppu15 import check

    def native_probe(name, pci, values):
        name.value = b'ZW-M890P'
        pci.value = b'0000:01:00.0'
        values[check.FIELDS.index('warp_size')] = 32
        values[check.FIELDS.index('major')] = 8
        values[check.FIELDS.index('minor')] = 9
        # qz15_device returns-1 when there is not exactly one visible device.
        return native_status

    library = object()
    monkeypatch.setattr(check, 'SDK', lambda _: SimpleNamespace(lib=library))
    monkeypatch.setattr(check, 'graph_bind', lambda _: None)
    monkeypatch.setattr(check.C, 'CDLL', lambda *a, **kw: library)
    monkeypatch.setattr(check, 'bind', lambda lib, name, args: native_probe
                        if name == 'qz15_device' else None)
    if native_status:
        with pytest.raises(RuntimeError, match='PPU1.5 device probe failed: status=-1'):
            check.setup(tmp_path, tmp_path)
    else:
        _, got_library, identity = check.setup(tmp_path, tmp_path)
        assert got_library is library
        assert identity['name'] == 'ZW-M890P' and identity['warp_size'] == 32


def test_compiler_identity_failure_precedes_compile_or_output_creation(monkeypatch, tmp_path):
    def failed_query(*args):
        raise RuntimeError('missing compiler version banner')
    monkeypatch.setattr(probe, 'compiler_identity', failed_query)
    with pytest.raises(RuntimeError, match='missing compiler version banner'):
        probe.build(tmp_path / 'sdk', tmp_path, 1, {})
    assert not (tmp_path / 'build').exists()

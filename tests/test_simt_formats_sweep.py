import copy
from pathlib import Path
import subprocess

import pytest

from tools import run_simt_formats as sweep


def test_bounded_plan():
    full = sweep.cases(sweep.FORMATS, range(1, 9))
    assert len(full) == len({p['id'] for p in full}) == 200
    assert len(sweep.cases(sweep.FORMATS, [1])) == 25
    assert {p['qtype'] for p in full} == {8, 10, 11, 13, 14}
    for p in full:
        assert p['experts'] == (1 if p['mode']=='dense' else 256)
        assert p['channels'] == (8 if 'moe-down' in p['id'] else 1)


@pytest.mark.parametrize('qtypes,tokens', [([], [1]), ([12], [1]), ([10, 10], [1]),
                                        ([10], []), ([10], [0]), ([10], [9]), ([10], [1, 1])])
def test_plan_rejects_undeclared_axes(qtypes, tokens):
    with pytest.raises(ValueError):
        sweep.cases(qtypes, tokens)


def test_old_inventory_matches_real_driver():
    from dev.gemv_simt.run import Baseline
    for q in sweep.FORMATS:
        actual = {f'old-{a}-c{c}-w{w}-s{s}' for a, c, w, s in Baseline.candidates(None, q)}
        assert sweep.candidate_keys(q)['old'] == actual


def test_physical_devices_not_ordinal_labels():
    devices = [dict(pci='0000:08:00.0', marker='PASS'), dict(pci='0000:7e:00.0', marker='PASS')]
    sweep.validate_devices(devices)
    with pytest.raises(ValueError):
        sweep.validate_devices([devices[0], devices[0]])
    with pytest.raises(ValueError):
        sweep.validate_devices([dict(pci='', marker='PASS')])


def valid_result():
    point = sweep.cases([10], [1])[-2]
    authority = dict(bundle_sha256='a'*64, rounds=4, samples=11, l2_bytes=64*1024**2)
    keys = sweep.candidate_keys(10)
    screen = [dict(arm=arm, key=key, error=0.00001, samples_us=[1., 2., 3.], median_us=2.)
              for arm, values in keys.items() for key in sorted(values)]
    finalists = [row for arm in keys for row in [r for r in screen if r['arm']==arm][:2]]
    confirmation = [row | dict(round=i, samples_us=[2.]*11) for row in finalists for i in range(4)]
    result = {k:v for k,v in point.items() if k!='id'} | dict(
        status='PASS', production_admitted=False, cache='rotating', l2_bytes=authority['l2_bytes'],
        active_experts=8, weight_bytes=8*1024**2, resident_weight_bytes=256*1024**2,
        ring_copies=18, ring_bytes=144*1024**2, allocated_ring_bytes=4608*1024**2,
        calls_per_graph=36, replay_proof=dict(replays=3,errors=[0.,0.,0.],negative='ZERO_A_REJECTED'),
        screen=screen, confirmation=confirmation,
        best={arm:dict(key=next(r['key'] for r in finalists if r['arm']==arm),median_us=2.) for arm in keys})
    return point, authority, dict(status='PASS', qtype=10, phase='perf',
                                  manifest_sha256=authority['bundle_sha256'], result=result)


def test_result_requires_complete_scope(tmp_path):
    point, authority, good = valid_result()
    path = tmp_path/'result.json'
    sweep.write(path, good)
    assert sweep.result_valid(path, point, authority)
    changes = [
        lambda d: d['result'].update(best={}),
        lambda d: d['result']['screen'].pop(),
        lambda d: d['result']['screen'].append(d['result']['screen'][0]),
        lambda d: d['result']['confirmation'].pop(),
        lambda d: d['result']['confirmation'][0].update(samples_us=[2.]),
        lambda d: d['result']['screen'][0].update(error=0.01),
        lambda d: d['result']['screen'][0].update(median_us=0.01),
        lambda d: d['result']['best']['new'].update(median_us=0.01),
        lambda d: d['result']['replay_proof'].update(negative='PASS'),
        lambda d: d['result'].update(active_experts=256),
        lambda d: d['result'].update(ring_copies=1, ring_bytes=256*1024**2),
        lambda d: d['result'].update(calls_per_graph=37),
        lambda d: d.update(manifest_sha256='stale'),
    ]
    for change in changes:
        bad=copy.deepcopy(good);change(bad);sweep.write(path,bad)
        assert not sweep.result_valid(path,point,authority)


def test_cold_union_uses_accessed_not_allocated_experts():
    points = sweep.cases([10], range(1, 9))
    for p in points:
        active = sweep.active_experts(p)
        assert active == 1 if p['mode']=='dense' else 8<=active<=8*p['tokens']
        assert active < p['experts'] if p['mode']=='indexed' else active==p['experts']


def test_box_shell_and_explicit_scope():
    root=Path(__file__).resolve().parents[1]
    path=root/'tools/run_simt_formats_box.sh'
    subprocess.run(['bash','-n',str(path)],check=True)
    text=path.read_text()
    assert 'git lfs pull --include=' in text and 'simt-register-reuse-v1/*.so' in text
    assert 'RESUME_RUN' in text and 'Current Docker shell is preserved.' in text

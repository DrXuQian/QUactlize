"""Host-only orchestration checks. No device use or synthetic performance claims."""
from copy import deepcopy
from pathlib import Path
import tarfile

import pytest

from dev.strategy.journal import atomic_json
from tools.run_kpack_strategy_prefill_multigpu import bind_devices, command, package


def runs():
    return [dict(previous='/old/gpu-'+str(i), cache='/old/cache', partition=dict(shard=i, shards=2),
                 device=dict(host='test-host', pci=str(i), ordinal=0, compute_units=72)) for i in range(2)]


def test_physical_cards_not_visible_ordinals_choose_shards():
    old = runs()
    rows = bind_devices(old, [('7', old[0]['device']), ('6', old[1]['device'])])
    assert [r['selector'] for r in rows] == ['7', '6']
    assert [r['previous'] for r in rows] == ['/old/gpu-0', '/old/gpu-1']


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'host', 'compute_units', 'extra'])
def test_device_substitutions_rejected(damage):
    old = runs(); devices = [('1',deepcopy(old[0]['device'])), ('2',deepcopy(old[1]['device']))]
    if damage == 'duplicate': devices[1] = ('2', devices[0][1])
    elif damage == 'missing': devices.pop()
    elif damage == 'extra': devices.append(('3', dict(devices[0][1],pci='3')))
    else: devices[0][1][damage] = 'changed'
    with pytest.raises(ValueError): bind_devices(old, devices)


def test_commands_always_reuse_original_shards_and_cache(tmp_path):
    row = runs()[1]
    cmd = command(row, tmp_path, Path('/sdk'), 27, 24)
    assert '--upgrade-cublas-events' in cmd and '--resume' not in cmd
    for flag, value in [('--previous-output',row['previous']), ('--cache',row['cache']),
                        ('--shard','1'), ('--shards','2')]:
        assert cmd[cmd.index(flag)+1] == value
    (tmp_path/'measurement').mkdir()
    assert '--resume' in command(row, tmp_path, Path('/sdk'), 27, 24)


def test_setup_failures_package_logs_not_binary_or_weight_cache(tmp_path):
    atomic_json(tmp_path/'request.json', {'test':'host-only'})
    atomic_json(tmp_path/'summary.json', {'status':'INCOMPLETE'})
    for i in range(2):
        folder = tmp_path/('shard-'+str(i)); folder.mkdir()
        (folder/'console.log').write_text('FAIL expected host fixture\n')
        (folder/'kernel.so').write_bytes(b'NEVER_ARCHIVE')
        (folder/'weights.bin').write_bytes(b'NEVER_ARCHIVE')
    archive = package(tmp_path, runs())
    with tarfile.open(archive) as reader:
        assert set(reader.getnames()) == {'request.json', 'summary.json',
                                         'shard-0/console.log', 'shard-1/console.log'}

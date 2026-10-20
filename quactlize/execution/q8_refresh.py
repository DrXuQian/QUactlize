"""Build-time binding of the reviewed Q8 dense bodies; not a runtime tuner."""
import copy
import json
from pathlib import Path

from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT/'policies/kpack_decode_refresh_v1.json'
EVIDENCE = '67fe29755f8ecc07b30374f4f76c1609f9ea9fccf52753c29063dc6f00997841'
FIELDS = ('variant', 'columns', 'warps', 'values', 'split')


def rows():
    table = json.loads(TABLE.read_text())
    if table['evidence_sha256'] != EVIDENCE or digest(
            {k: v for k, v in table.items() if k != 'evidence_sha256'}) != EVIDENCE:
        raise ValueError('Q8 binding requires the reviewed refresh evidence')
    result = [r for r in table['rows'] if r['implementation']['kind'] == 'q8-address']
    if len(result) != 27 or any(not r['changed'] for r in result):
        raise ValueError('Q8 binding denominator differs')
    return result


def apply(baseline):
    """Only exact Q8 keys change. Preserve their previous bucket donors."""
    table = json.loads(TABLE.read_text())
    if digest(baseline) != table['baseline_sha256']:
        raise ValueError('Q8 binding incumbent catalog differs')
    result = copy.deepcopy(baseline)
    exact = {tuple(r['key']): r for r in result['exact']}
    for measured in rows():
        row = exact[tuple(measured['key'])]
        if row['config'] != measured['original_config'] or row.get(
                'dequant_config', -1) != measured['original_dequant_config']:
            raise ValueError('Q8 binding incumbent recipe differs')
        row['bucket_config'] = copy.deepcopy(row['config'])
        row['bucket_dequant_config'] = row.get('dequant_config', -1)
        row.update(config=measured['config'], dequant_config=-1, policy=18,
                   implementation=measured['implementation'])
    return result


def capability():
    return dict(evidence_sha256=EVIDENCE,
                catalog_sha256=sha(ROOT/'quactlize/execution/q8_refresh.inc'),
                body_sha256=sha(ROOT/'quactlize/execution/simt_q8_refresh.cuh'),
                keys=[dict(key=r['key'], body=r['implementation']['body'],
                           config={f: r['config'][f] for f in FIELDS}) for r in rows()])


def check_cpp_transition(originals, current):
    """Re-audit frozen measurements after this exact-only source admission."""
    table = json.loads(TABLE.read_text())
    if digest(originals) != table['cpp_originals_sha256'] or len(originals) != len(current):
        raise ValueError('Q8 original C++ receipts differ')
    bound = {tuple(r['key']): r for r in rows()}
    changed = 0
    for old, new in zip(originals, current):
        r = bound.get(tuple(old['request']))
        expected = old
        if r:
            expected = dict(request=r['key'], status=0, kind=1, policy=18,
                dequant_config=-1, donor=r['key'][2:4]+[r['key'][7]], workspace_bytes=0,
                config={f: r['config'][f] for f in FIELDS},
                implementation=dict(producer='q8-address', measured='none', reduction='none',
                                    changes=0, hoist=True, fixed=False, q8_body=r['implementation']['body']))
            changed += 1
        if new != expected:
            raise ValueError('C++ selector changed beyond the Q8 admission: '+repr(old['request']))
    if changed != len(bound):
        raise ValueError('Q8 C++ query coverage differs')
    return changed

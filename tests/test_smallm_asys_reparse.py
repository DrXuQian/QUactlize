"""Synthetic trace migration tests; no device/performance claims."""
import json
from pathlib import Path
import sqlite3

import pytest

from dev.strategy.asys_reparse import reparse, measurement_provenance
from dev.strategy.asys_components import parse
from dev.strategy.smallm_asys import METHOD
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_decode_refresh import sealed as write_sealed
from test_smallm_asys import capture_fixture, provenance


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


def fixture(root):
    data, rows = capture_fixture()
    for record in data['component_records'].values():
        node = record['identity']['node']
        name = node['name']
        if record['kind'] != 'producer':
            continue
        symbol = 'cutlass::device_kernel<' + name + '>'
        node.update(name=symbol, symbol=symbol)
        for row in rows:
            if row['name'] == name:
                row.update(name=symbol, symbol=symbol)
    keys = {key:digest(value['identity']) for key,value in data['component_records'].items()}
    data['component_records'] = {keys[key]:value for key,value in data['component_records'].items()}
    data['components'] = list(data['component_records'])
    data['arm_components'] = {key:[keys[c] for c in value] for key,value in data['arm_components'].items()}
    for item in data['trace_schedule']:
        item['component'] = keys[item['component']]
    # Deliberate intersection, retaining both actual per-record durations.
    rows[1]['start'] = rows[0]['start']
    point = dict(id='point', workload=data['workload'], arms=[dict(id=a, spec={}, recipe=dict(kind='tc'))
                                                            for a in data['arms']])
    write_sealed(root/'plan.json', dict(schema='quactlize.smallm-asys.v1', points=[point]))
    p = provenance()
    run = dict(plan=read(root/'plan.json')['sha256'], assignments=[['point']], cards=['3'],
        source_revision={k:p[k] for k in ('quactlize_commit', 'quactlize_tree', 'actlize_commit')},
        source=dict(sha256=p['source_sha256']), harness=dict(sha256=p['harness_sha256']),
        images=dict(test='synthetic'), sdk=p['sdk'], devices=[p['device']], asys_sha256=p['asys_sha256'],
        timing_method=METHOD, python_version='synthetic')
    write_sealed(root/'run.json', run)
    run = read(root/'run.json')
    attempt = root/'gpu-3/attempts/first'
    attempt.mkdir(parents=True)
    write_sealed(attempt/'capture.json', dict(point='point', data=data,
        authority=dict(run=run['sha256'], shard=0, device=run['devices'][0])))
    report = attempt/'trace.asysrep'
    report.write_bytes(b'synthetic host fixture, not a real report')
    db_path = attempt/'trace.sqlite'
    with sqlite3.connect(db_path) as db:
        strings = {s:i for i,s in enumerate(sorted({r[k] for r in rows for k in ('name', 'symbol')}), 1)}
        db.execute('CREATE TABLE StringIds (id INTEGER,value TEXT)')
        db.executemany('INSERT INTO StringIds VALUES (?,?)', [(v,k) for k,v in strings.items()])
        fields = ('start end demangledName mangledName gridX gridY gridZ blockX blockY blockZ '
                  'staticSharedMemory dynamicSharedMemory deviceId contextId streamId globalPid').split()
        db.execute('CREATE TABLE HGPTI_ACTIVITY_KIND_KERNEL ('+','.join(f+' INTEGER' for f in fields)+')')
        for row in rows:
            values = [row['start'],row['end'],strings[row['name']],strings[row['symbol']],
                      *row['grid'],*row['block'],0,row['shared_bytes'],0,1,2,99]
            db.execute('INSERT INTO HGPTI_ACTIVITY_KIND_KERNEL VALUES ('+','.join('?'*len(fields))+')', values)
    failure = dict(status='FAIL', point='point', run=run['sha256'], attempt=str(attempt),
                   error='isolated kernel activities overlap or are unordered')
    point_path = root/'gpu-3/points/point.json'
    write_sealed(attempt/'failure.json', failure)
    write_sealed(point_path, failure)
    return point_path, attempt, run, data


def test_offline_reparse_preserves_authority_and_all_old_bytes(tmp_path):
    point, attempt, run, _ = fixture(tmp_path)
    before = {str(p):sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    target, result = reparse(tmp_path, ROOT)
    assert result['status'] == 'COMPLETE' and result['points'][0]['action'] == 'OFFLINE_REPARSED'
    assert not result['gpu_work'] and not result['old_receipts_modified']
    recovered = read(target/'point.json')
    assert recovered['data']['latency_provenance']['quactlize_commit'] == 'a'*40
    assert recovered['data']['decision']['winner'] == 'fast'
    assert recovered['parser']['sha256'] != run['harness']['sha256']
    assert {p:sha(Path(p)) for p in before} == before
    assert read(point)['status'] == 'FAIL'
    again, other = reparse(tmp_path, ROOT)
    assert again != target and other['status'] == 'COMPLETE'


@pytest.mark.parametrize('change', ['authority', 'kernel', 'seal', 'unrelated_failure', 'missing_trace'])
def test_reparse_does_not_admit_unrelated_or_corrupt_results(tmp_path, change):
    point, attempt, run, _ = fixture(tmp_path)
    if change == 'authority':
        data = read(attempt/'capture.json');data['authority']['shard'] = 1
        write_sealed(attempt/'capture.json', data)
    elif change == 'kernel':
        with sqlite3.connect(attempt/'trace.sqlite') as db:
            db.execute('UPDATE HGPTI_ACTIVITY_KIND_KERNEL SET gridX=999')
    elif change == 'seal':
        data = read(point);data['point'] = 'other';point.write_text(json.dumps(data))
    elif change == 'unrelated_failure':
        data = read(point);data['error'] = 'numeric gate failed';write_sealed(point, data)
    else:
        (attempt/'trace.sqlite').rename(attempt/'missing.sqlite')
    _, result = reparse(tmp_path, ROOT)
    assert result['status'] == 'INCOMPLETE' and result['points'][0]['status'] == 'FAIL'


def test_already_passing_receipt_is_reused_without_relabelling(tmp_path):
    point, attempt, run, data = fixture(tmp_path)
    measured = parse(data, attempt/'trace.sqlite', attempt/'trace.asysrep', measurement_provenance(run, 0))
    write_sealed(point, dict(status='PASS', point='point', run=run['sha256'], data=measured,
        evidence={str(p):sha(p) for p in (attempt/'trace.sqlite', attempt/'trace.asysrep', attempt/'capture.json')}))
    before = point.read_bytes()
    target, result = reparse(tmp_path, ROOT)
    assert result['points'][0]['action'] == 'PRESERVED_OLD_PASS'
    assert point.read_bytes() == before and not (target/'point.json').exists()

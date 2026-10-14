"""Per-arm receipts: preserve PASS peers and leave a failed GPU process.

The parent owns a physical-device lock, including across child lifetimes.
An invocation attempts each non-PASS arm at most once. A later --resume gets
a new invocation ID and retries failures only. This is not a timing cache:
component files and their exact authority must also validate before reuse.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time
import traceback
import uuid

from dev.strategy.prescreen import semantic_config
from quactlize.dispatch.table import Recipe
from quactlize.runtime.tuning import digest


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def planned_arms(point):
    result = {'original': dict(id='original', recipe=point.get('original_recipe'),
                               roles=['original'], aliases=[])}
    for source, role in (('candidates', 'rule'), ('champion_controls', 'control')):
        for value in point.get(source, []):
            recipe = Recipe.from_dict(value)
            # Same current source/body/geometry with an old symbol alias does
            # not need a second measurement. Dequant/S/grid remain in the key.
            key = semantic_config(recipe)
            arm = result.setdefault(key, dict(id=key, recipe=asdict(recipe), roles=[], aliases=[]))
            if role not in arm['roles']:
                arm['roles'].append(role)
            if recipe.key not in arm['aliases']:
                arm['aliases'].append(recipe.key)
    return list(result.values())


def valid_pass(data):
    return (isinstance(data, dict) and data.get('correctness') == 'PASS' and
            isinstance(data.get('components'), list) and bool(data['components']) and
            all(isinstance(c, str) and re.fullmatch('[0-9a-f]{64}', c) for c in data['components']))


class ArmJournal:
    def __init__(self, directory, authority, invocation):
        if not re.fullmatch('[0-9a-f]{32}', invocation):
            raise ValueError('invocation requires a UUID hex string')
        self.root = Path(directory)
        self.authority = json.loads(json.dumps(authority, allow_nan=False))
        self.invocation = invocation

    def location(self, point, arm):
        if not re.fullmatch('[0-9a-f]{64}', point['id']) or not re.fullmatch(
                'original|[0-9a-f]{64}', arm['id']):
            raise ValueError('invalid point/arm ID')
        return self.root / point['id'] / arm['id']

    def identity(self, point, arm):
        # Dataclass recipes contain tuples, while receipts contain JSON arrays.
        # Compare the serialized identity, not Python's tuple/list distinction.
        return json.loads(json.dumps(dict(authority=self.authority, point=point['id'],
                                         workload=point['workload'], arm=arm), allow_nan=False))

    def read(self, point, arm):
        path = self.location(point, arm) / 'current.json'
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        sealed = {k: v for k, v in row.items() if k != 'record_sha256'}
        if (row.get('schema') != 'quactlize.component-arm.v1' or
                row.get('identity') != self.identity(point, arm) or
                row.get('record_sha256') != digest(sealed) or
                row.get('status') not in ('RUNNING', 'PASS', 'FAIL', 'CRASH')):
            raise ValueError('arm resume identity/digest differs: ' + str(path))
        if row['status'] == 'PASS':
            data = row.get('data', {})
            if not valid_pass(data):
                raise ValueError('PASS arm lacks correctness/components: ' + str(path))
        return row

    def due(self, point, arm):
        row = self.read(point, arm)
        return row is None or row['status'] != 'PASS' and row['invocation'] != self.invocation

    def publish(self, point, arm, row):
        row['record_sha256'] = digest({k: v for k, v in row.items() if k != 'record_sha256'})
        base = self.location(point, arm)
        atomic_json(base / (row['attempt'] + '.json'), row)
        atomic_json(base / 'current.json', row)
        return row

    def start(self, point, arm):
        if not self.due(point, arm):
            raise ValueError('arm already complete or attempted in this invocation')
        return self.publish(point, arm, dict(schema='quactlize.component-arm.v1',
            identity=self.identity(point, arm), invocation=self.invocation,
            attempt=uuid.uuid4().hex, status='RUNNING', pid=os.getpid(),
            started_ns=time.time_ns()))

    def finish(self, point, arm, status, *, data=None, error=None):
        if status not in ('PASS', 'FAIL', 'CRASH'):
            raise ValueError('invalid terminal arm status')
        row = self.read(point, arm)
        if not row or row['status'] != 'RUNNING' or row['invocation'] != self.invocation:
            raise ValueError('cannot finish an inactive arm attempt')
        if status == 'PASS' and not valid_pass(data):
            raise ValueError('PASS requires checked component measurements')
        row.update(status=status, ended_ns=time.time_ns())
        if data is not None: row['data'] = data
        if error is not None: row['error'] = str(error)
        return self.publish(point, arm, row)

    def pending(self, points):
        return [(p, a) for p in points for a in planned_arms(p) if self.due(p, a)]

    def reap(self, points, process_rc, log):
        """Call ONLY after the child exited, never for a live worker."""
        for point in points:
            for arm in planned_arms(point):
                row = self.read(point, arm)
                if row and row['status'] == 'RUNNING' and row['invocation'] == self.invocation:
                    self.finish(point, arm, 'CRASH', error=f'child rc={process_rc}; log={log}')


def execute_point(point, journal, measure):
    """One fresh child may do many good arms, but stops at its first error."""
    for arm in planned_arms(point):
        if not journal.due(point, arm):
            continue
        journal.start(point, arm)
        try:
            data = measure(arm)
            journal.finish(point, arm, 'PASS', data=data)
        except Exception as exc:
            # Persist before caller cleanup: a poisoned context may also fail
            # to synchronize/free. The parent will launch a fresh process.
            traceback.print_exc()
            journal.finish(point, arm, 'FAIL', error=repr(exc))
            return False
    return True

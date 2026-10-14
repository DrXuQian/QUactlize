#!/usr/bin/env python3
"""Export accuracy, timings and failures from a finished strategy run.

Standalone, Python standard library only. No SDK, repository imports, GPU work,
source checkout update, result deletion or selection changes. Raw evidence stays
on Box; this compact review file is NOT a resumable run or a full admission proof.
"""
import argparse
from collections import Counter
import hashlib
import json
import lzma
import math
from pathlib import Path
import re
import statistics
import time


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def select(value, keys):
    return {k: value[k] for k in keys.split() if k in value}


def reject_constant(value):
    raise ValueError('nonfinite JSON: '+value)


def accuracy(data):
    """Summarize all recorded repeats, not just the best repeat or its hashes."""
    proofs = []

    def visit(value):
        if isinstance(value, dict):
            if 'bad' in value or 'relative_l1_error' in value:
                proofs.append(value)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data.get('numerical', []))
    out = select(data, 'correctness zero_input zero_a pointer_negatives omitted_producer '
                       'changed_input_and_router mutable_count')
    out['checks'] = len(proofs)
    for field in ('bad', 'nonfinite', 'relative_l1_error'):
        values = [p[field] for p in proofs if field in p]
        if values:
            out[field + '_max'] = max(values)
    for field in ('cells', 'bits', 'tolerance', 'active_experts', 'inactive_experts_untouched'):
        out[field] = sorted({p[field] for p in proofs if field in p})
    return out


class Review:
    def __init__(self, root, stream):
        self.root, self.stream = root, stream
        self.counts = Counter()
        self.tables = {}
        self.content = hashlib.sha256()
        self.needed = set()
        self.written_components = set()
        self.previous_components = set()
        self.previous_points = set()
        self.screens = set()
        self.points = set()
        self.input_bytes = 0

    def emit(self, record_type, **value):
        raw = encoded(dict(type=record_type, **value)) + b'\n'
        self.stream.write(raw)
        self.content.update(raw)
        self.counts[record_type] += 1

    def issue(self, path, error):
        self.emit('export_issue', path=str(path), error=str(error))

    def ref(self, kind, value):
        if value is None:
            return None
        key = encoded(value)
        table = self.tables.setdefault(kind, {})
        if key not in table:
            index = len(table)
            table[key] = index
            self.emit('definition', table=kind, index=index, value=value)
        return table[key]

    def read(self, path):
        """Never follow a result-directory link into other data or caches."""
        path = Path(path)
        if not path.resolve(strict=True).is_relative_to(self.root):
            raise ValueError('result path escapes the selected run')
        relative = path.relative_to(self.root)
        if any((self.root.joinpath(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts) + 1)):
            raise ValueError('symlink result is not followed')
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('result changed while packing; wait for the run to finish')
        self.input_bytes += len(raw)
        value = json.loads(raw, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        if 'record_sha256' in value and value['record_sha256'] != digest(
                {k: v for k, v in value.items() if k != 'record_sha256'}):
            raise ValueError('receipt digest differs')
        return value

    def proof(self, data):
        execution = data.get('receipt')
        if isinstance(execution, dict) and isinstance(execution.get('provider'), dict):
            execution = dict(execution, provider=self.provider(execution['provider']))
        out = dict(accuracy=accuracy(data),
                   fixture=self.ref('fixture', data.get('fixture')),
                   execution=self.ref('execution', execution))
        out.update(select(data, 'components scope cost_scope wall_seconds timed_router_sha256 '
                               'timed_active_experts provider_loaded_images output_cache'))
        return out

    def provider(self, value):
        compact = {k:v for k,v in value.items() if k!='files'}
        if 'files' in value:
            compact.update(source_files_sha256=digest(value['files']), source_files_count=len(value['files']))
        return self.ref('provider', compact)

    def point(self, value):
        pid = value.get('id', value.get('point'))
        if pid not in self.points:
            self.points.add(pid)
            self.emit('point', id=pid, **select(value, 'workload status decision previous_screen reason issues'))

    def component(self, cid, data, phase, point=None):
        if not re.fullmatch('[0-9a-f]{64}', cid):
            raise ValueError('invalid component ID')
        identity = data['identity']
        if digest(identity) != cid:
            raise ValueError('component identity differs')
        rounds = data['samples_us']
        if len(rounds) < 3 or any(len(r) < 3 for r in rounds) or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for r in rounds for v in r):
            raise ValueError('invalid component samples')
        medians = [statistics.median(r) for r in rounds]
        expected = dict(median_us=statistics.median(medians), round_medians_us=medians,
                        spread_pct=100*(max(medians)/min(medians)-1))
        if any(data.get(k) != v for k, v in expected.items()):
            raise ValueError('component statistics differ from samples')
        if (data.get('status') != 'PASS' or data.get('scope') != 'ISOLATED_SINGLE_KERNEL' or
                data.get('timed_kernel_types') != 1 or not data.get('first_use_excluded') or
                data.get('host_launch_cost_added') is not False):
            raise ValueError('component timing scope differs')
        node = dict(identity['node'])
        for field in ('name', 'symbol'):
            if field in node:
                node[field] = self.ref('symbol', node[field])
        row = select(data, 'kind median_us spread_pct samples_us calls_per_graph status scope record_sha256')
        row.update(id=cid, phase=phase, point=point, cache=identity.get('cache'),
                   node=self.ref('kernel', node),
                   contract=self.ref('component_contract', identity.get('contract')),
                   authority=identity.get('authority'), implementation=identity.get('implementation'),
                   first_use_excluded=True, timed_kernel_types=1, host_launch_cost_added=False)
        self.emit('component', **row)
        if phase == 'screen':
            self.written_components.add(cid)

    def screen(self, data):
        pid = data.get('point', data.get('id'))
        if pid in self.screens:
            return
        self.point(data)
        self.screens.add(pid)
        row = select(data, 'status selection missing best_sf expansion_configs component_arms finalists '
                           'measurement_origin previous_root previous_receipt best_helpers components '
                           'scope excluded admission')
        self.emit('screen', point=pid, **row)
        for keys in data.get('component_arms', data.get('components', {})).values():
            self.needed.update(keys)

    def arm(self, row):
        identity = row['identity']
        pid, arm = identity['point'], identity['arm']
        out = dict(point=pid, id=arm['id'], status=row['status'], roles=arm['roles'],
                   recipe=self.ref('config', arm.get('recipe')),
                   authority=identity.get('authority'), record_sha256=row['record_sha256'])
        if 'config' in arm:
            out['expansion_config'] = arm['config']
        data = row.get('data', {})
        if arm['roles'] == ['confirmation']:
            out.update(select(data, 'status correctness decision protocol scope winner measurement_origin '
                                   'arm_components roles wall_seconds production_changed performance_admitted'))
            out['journal_status'] = row['status']
            if 'original_proof' in data:
                out['original_proof'] = self.proof(data['original_proof'])
                if pid in self.previous_points:
                    self.previous_components.update(data['original_proof'].get('components', []))
                else:
                    self.needed.update(data['original_proof'].get('components', []))
            out['arms'] = {aid: self.proof(proof) for aid, proof in data.get('arms', {}).items()}
            out['expansion_proofs'] = data.get('expansion_proofs', {})
            out['expansion_fixture'] = self.ref('fixture', data.get('expansion_fixture'))
            for cid, record in data.get('component_records', {}).items():
                try:
                    self.component(cid, record, 'confirmation', pid)
                except (ValueError, KeyError, TypeError) as exc:
                    self.issue(pid+'/'+cid, exc)
            self.emit('confirmation', **out)
        else:
            out.update(self.proof(data))
            self.needed.update(data.get('components', []))
            if row['status'] == 'PASS' and data.get('correctness') != 'PASS':
                self.issue(pid+'/'+arm['id'], 'PASS arm lacks recorded correctness PASS')
            self.emit('arm', **out)
        if row['status'] != 'PASS':
            self.emit('failure', point=pid, arm=arm['id'], status=row['status'],
                      error=row.get('error'), recipe=out['recipe'], receipt_sha256=row['record_sha256'])

    def collect(self, *, tail_bytes, max_error_logs):
        summary = self.read(self.root/'summary.json')
        authority = self.read(self.root/'authority.json')
        resource = authority.get('resource', {})
        self.emit('run', root=str(self.root), summary={k:v for k,v in summary.items() if k!='points'},
                  authority_sha256=digest(authority),
                  authority=dict(resource=resource, harness_sha256=authority.get('harness', {}).get('sha256'),
                      **select(authority, 'plan body_source dequant oracle l2 idle protocol scope')),
                  images={k: authority[k] for k in ('tc', 'vectors') if k in authority},
                  providers={k:self.provider(v) for k,v in authority.get('providers', {}).items()})
        for p in summary.get('points', []) + summary.get('blocked', []):
            pid = p.get('id', p.get('point'))
            self.point(p)
            if p.get('previous_screen'):
                self.previous_points.add(pid)
            if p.get('status') not in ('PASS', 'COMPLETE'):
                self.emit('failure', point=pid, status=p.get('status'),
                          error=p.get('reason', 'point is not complete; see summary/result'))
            if 'selection' in p:
                self.screen(p)
        if (self.root/'imported.json').is_file():
            for pid, data in self.read(self.root/'imported.json').items():
                self.previous_points.add(pid)
                self.emit('prior_screen', point=pid, **select(data, 'previous_root previous_receipt measurement_origin'))
                for keys in data.get('component_arms', {}).values():
                    self.previous_components.update(keys)
        folder = 'screens' if (self.root/'screens').is_dir() else 'cases'
        for path in sorted((self.root/folder).glob('*.json')):
            try:
                self.screen(self.read(path))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.issue(path.relative_to(self.root), exc)
        # Explicit depth/pattern, no recursive scan of compile caches or attempts.
        for index, path in enumerate(sorted((self.root/'arms').glob('*/*/current.json')), 1):
            try:
                row = self.read(path)
                if (row['identity']['point'] != path.parent.parent.name or
                        row['identity']['arm']['id'] != path.parent.name):
                    raise ValueError('receipt point/arm path differs')
                self.arm(row)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.issue(path.relative_to(self.root), exc)
            if index % 1000 == 0:
                print(f'STRATEGY_REVIEW_PROGRESS current_receipts={index} input_MiB={self.input_bytes/2**20:.1f}', flush=True)
        for cid in sorted(self.needed - self.previous_components):
            if not isinstance(cid, str) or not re.fullmatch('[0-9a-f]{64}', cid):
                self.issue('components', 'invalid referenced component ID: '+repr(cid))
                continue
            path = self.root/'components'/(cid+'.json')
            try:
                self.component(cid, self.read(path), 'screen')
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.issue(path.relative_to(self.root), exc)
        if self.previous_components:
            self.emit('prior_components', ids=sorted(self.previous_components),
                      scope='ALREADY_UPLOADED_PRIOR_SCREEN_NOT_REMEASURED_OR_REPACKED')
        error_logs = 0
        paths = sorted(self.root.glob('*.log')) + [self.root.parent/'console.log']
        for path in paths:
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(self.root.parent):
                continue
            with path.open('rb') as f:
                size = path.stat().st_size
                f.seek(max(0, size-tail_bytes))
                tail = f.read(tail_bytes).decode('utf-8', errors='replace')
            if not re.search(r'Traceback|\b(?:Error|Exception):|\bFAIL(?:ED)?\b|status=CRASH|SETUP_FAIL', tail):
                continue
            self.emit('error_log', path=str(path), bytes=size, truncated=size>tail_bytes,
                      tail=tail if error_logs<max_error_logs else None)
            error_logs += 1
        self.emit('end', counts=dict(self.counts), input_bytes=self.input_bytes,
                  prior_lines_sha256=self.content.hexdigest(),
                  scope='REVIEW_ONLY_RAW_EVIDENCE_RETAINED_ON_BOX',
                  export_status='COMPLETE' if not self.counts['export_issue'] else 'COMPLETE_WITH_ISSUES',
                  production_changed=False, performance_admitted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='finished outer run or its measurement/decode directory')
    parser.add_argument('--output', type=Path, help='new .jsonl.xz file; never overwrite')
    parser.add_argument('--error-tail-bytes', type=int, default=12288)
    parser.add_argument('--max-error-logs', type=int, default=32)
    args = parser.parse_args(argv)
    if not 0 < args.error_tail_bytes <= 2**20 or not 0 <= args.max_error_logs <= 1000:
        parser.error('invalid error-log bounds')
    root = args.run.resolve(strict=True)
    for name in ('measurement', 'decode'):
        if (root/name/'summary.json').is_file():
            root = root/name
            break
    if not (root/'summary.json').is_file() or not (root/'authority.json').is_file():
        parser.error('finished run summary.json and authority.json are required')
    output = (args.output or args.run.resolve().with_name(args.run.resolve().name+
              '.'+str(time.time_ns())+'.review.jsonl.xz')).absolute()
    if output.exists() or output.is_symlink():
        parser.error('output already exists; choose a new file')
    started = time.monotonic()
    with output.open('xb') as raw, lzma.LZMAFile(raw, 'w', preset=6) as stream:
        report = Review(root, stream)
        report.emit('format', schema='quactlize.strategy-review.v1',
                    producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    references='definition records are indexed by table and index',
                    retention='ALL_CURRENT_ARMS_ALL_SCREEN_COSTS_CONFIRMATIONS_REFERENCED_SAMPLES_ERRORS',
                    excludes=['BINARIES', 'COMPILE_CACHE', 'FULL_PLANS', 'REPEATED_MANIFESTS',
                              'SUCCESS_LOGS', 'OLD_ATTEMPTS', 'ALREADY_UPLOADED_PRIOR_SCREEN'])
        report.collect(tail_bytes=args.error_tail_bytes, max_error_logs=args.max_error_logs)
    print('STRATEGY_REVIEW_DONE '+json.dumps(dict(output=str(output), bytes=output.stat().st_size,
        seconds=round(time.monotonic()-started, 2), records=dict(report.counts),
        raw_results_unchanged=True, gpu_work=False)), flush=True)
    print('results='+str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

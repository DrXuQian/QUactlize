#!/usr/bin/env python3
"""Profile four screened Q8 winners using an unchanged, completed Box run."""
import argparse
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

# Fixed before this confirmation, selected from the completed cold screen.
# Keep all six controls/candidates, not only the nominal event winner.
FOCUSES = (
    dict(point=0, m=8, config=1),
    dict(point=0, m=8, config=0),  # Original best recipe differs at this shape.
    dict(point=2, m=8, config=0),
    dict(point=3, m=8, config=0),
    dict(point=5, m=8, config=0),
)
SHAPES = {0: (512, 2048), 2: (4096, 2048), 3: (4096, 4096), 5: (8192, 5120)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def contained(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('invalid parent evidence path: ' + name)
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('missing/non-local parent file: ' + str(path))
    return path


def parent(run):
    source = (run / 'source').resolve(strict=True)
    output = (run / 'results').resolve(strict=True)
    frozen = read(output / 'plan.json')
    authority = read(output / 'authority.json')
    summary = read(output / 'summary.json')
    build = read(output / 'build/candidate/manifest.json')
    if (frozen.get('schema') != 'q8-address-window-v1' or
            frozen.get('numeric_pairs') != 280 or frozen.get('timing_pairs') != 160 or
            len(frozen.get('arms', [])) != 6 or summary.get('status') != 'PASS' or
            summary.get('production_changed') or build.get('object_only')):
        raise ValueError('requires the completed Q8 address-window parent, not an ablation/rebuild')
    specs = {p['point']: p for p in frozen['points']}
    for f in FOCUSES:
        p = specs[f['point']]
        w = p['workload']
        if (w['qtype'] != 8 or w['compute'] != 'f16' or w['operation'] != 'dense' or
                (w['n'], w['k']) != SHAPES[f['point']] or f['m'] not in p['timing_m']):
            raise ValueError('frozen finalist workload differs')
    harness = authority['harness']
    if digest(harness['files']) != harness['sha256']:
        raise ValueError('parent harness receipt differs')
    for label, files in (('harness', harness['files']), ('compile input', build['sources'])):
        for name, expected in files.items():
            if sha(contained(source, name)) != expected:
                raise ValueError(f'parent {label} changed: {name}; do not update frozen sources')
    if {r['compute'] for r in build['records']} != {'f16', 'bf16'}:
        raise ValueError('missing parent precision image')
    for r in build['records']:
        image = contained(output, 'build/candidate/' + r['image'])
        if sha(image) != r['sha256']:
            raise ValueError('parent library changed: ' + str(image))
    contained(source, 'tools/run_q8_dense.py')
    return source, output, frozen, authority


def execute(a, source, output, frozen, authority):
    # Import the immutable parent's implementation, not the updated checkout's
    # harness. Child commands inside profile_focus also use this frozen path.
    sys.path.insert(0, str(source))
    runner = importlib.import_module('tools.run_q8_dense')
    if Path(runner.__file__).resolve() != source / 'tools/run_q8_dense.py':
        raise ValueError('wrong Q8 runner imported')
    runner.configure('address')
    if runner.plan() != frozen:
        raise ValueError('parent plan differs from its own source')
    if not 0 < a.expected_ce_ghz < 10 or abs(a.expected_ce_ghz * 1e9 /
                                             authority['expected_ce_hz'] - 1) > .001:
        raise ValueError('requested clock differs from the completed parent')
    sdk = a.sdk.resolve(strict=True)
    acu = (a.acu or sdk / 'asight/bin/acu').resolve(strict=True)
    args = SimpleNamespace(output=output, sdk=sdk, acu=acu, experiment='address',
        l2_bytes=authority['device']['l2']['bytes'], child=0)
    runner.binding(args)  # Physical PCI, source, SDK and loaded image identity.
    original = {p: sha(output / p) for p in ('plan.json', 'authority.json', 'summary.json')}
    lock_name = 'quactlize-components-' + digest(authority['device']['device']['pci'])[:24] + '.lock'
    with (Path('/tmp') / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for point in sorted({f['point'] for f in FOCUSES}):
            runner.complete_point(output, point, authority)
        attempt = output / 'attempts' / ('finalists-' + str(time.time_ns()))
        attempt.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        profiles = []
        try:
            for f in FOCUSES:
                print('Q8_CONFIRM_FOCUS ' + json.dumps(f), flush=True)
                try:
                    profiles.append(runner.profile_focus(args, f, attempt, authority))
                except Exception as exc:
                    profiles.append(dict(focus=f, status='FAIL', error=str(exc)))
            result = dict(
                schema='q8-address-cycle-confirm-v1',
                status='PASS' if all(p['status'] == 'PASS' for p in profiles) else 'INCOMPLETE',
                profiles=profiles, original_evidence_sha256=original,
                authority_sha256=digest(authority), coordinator_sha256=sha(Path(__file__).resolve()),
                elapsed_seconds=time.monotonic() - started, new_compiles=0,
                cold_timing_rerun=False, original_summary_modified=False,
                production_changed=False, performance_admitted=False,
                scope='FOUR_SCREEN_WINNER_ACU_CONFIRMATION_NOT_GLOBAL_SELECTOR_OR_MODEL_TPOT')
            if any(sha(output / p) != value for p, value in original.items()):
                raise ValueError('original summary/plan/authority changed during confirmation')
            runner.sealed(attempt / 'confirmation.json', result)
            print('Q8_CONFIRM_DONE ' + json.dumps({k: v for k, v in result.items() if k != 'profiles'}), flush=True)
            return int(result['status'] != 'PASS')
        finally:
            from tools.pack_simt_split_review import package
            package(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='existing run containing source/ and results/')
    p.add_argument('--sdk', type=Path, default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--acu', type=Path)
    p.add_argument('--expected-ce-ghz', type=float, default=1.7)
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    try:
        source, output, frozen, authority = parent(a.run.resolve(strict=True))
        print('Q8_CONFIRM_PLAN ' + json.dumps(dict(focuses=FOCUSES, acu_reports=90,
            new_compiles=0, cold_timing_rerun=False, production_changed=False)), flush=True)
        if a.plan_only:
            return 0
        if not a.confirm_idle:
            p.error('confirm the same physical GPU is otherwise idle')
        return execute(a, source, output, frozen, authority)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())

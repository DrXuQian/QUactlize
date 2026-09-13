#!/usr/bin/env python3
"""Incremental Q4 decode sweep with independent-cell recovery and frozen images."""
import argparse
import csv
import ctypes as C
from dataclasses import asdict
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_ppu import decode_sweep as spec
from dev.gemv_ppu.decode_access import access
from dev.gemv_ppu.decode_bench import Base, DenseTC, GroupedTC, Unsupported, probe
from dev.gemv_ppu.moe_s1_bench import error
from dev.gemv_ppu.run_moe_compare import validate_samples
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, checked
from quactlize.runtime.tuning import digest
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command

STRUCTURAL = {'PACKED_ROW_A_REQUIRES_M1', 'INSUFFICIENT_K_TILES_PER_PIPELINE_SLICE',
              'DEVICE_QUERY_UNSUPPORTED', 'NO_PERSISTENT_RESIDENCY'}
TC_SCOPES = {'DENSE_F32_CAST_TC_REAL_REDUCER_F32_CAST',
             'NATIVE_FUSED_INDEXED_PREPARE_TC_REDUCE_SCATTER',
             'GPU_RANK_GATHER_METADATA_DIRECTORY_TC_REDUCER_SCATTER'}


def save(path, value):
    """Replace a single checkpoint atomically; an interrupted write is not a pass."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def package_identity():
    result = {}
    for name in ('numpy', 'torch', 'gguf'):
        module = importlib.import_module(name)
        try: version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: version = getattr(module, '__version__', 'UNREPORTED')
        result[name] = dict(version=version, module_sha256=sha(Path(module.__file__)))
    result['gguf.quants'] = dict(module_sha256=sha(Path(importlib.import_module('gguf.quants').__file__)))
    return result


def identity(args):
    # Pure Python helpers also define execution/fixture semantics. A new
    # harness cannot silently adopt old numerical receipts from another run.
    paths = ['dev/gemv_ppu/decode_bench.py', 'dev/gemv_ppu/decode_access.py',
             'dev/gemv_ppu/run_decode_sweep.py', 'dev/gemv_ppu/decode_sweep.py',
             'dev/gemv_ppu/moe_s1.py', 'dev/gemv_ppu/moe_s1_bench.py',
             'dev/gemv_ppu/moe_compare.py', 'dev/gemv_ppu/moe_compare_bench.py',
             'dev/gemv_ppu/run_moe_compare.py', 'dev/gemv_ppu/run_moe_s1.py',
             'dev/gemv_ppu/smallm.py', 'dev/gemv_ppu/small_latency.py',
             'dev/gemv_ppu/reader_reuse.py', 'dev/gemv_ppu/config_space.py',
             'dev/gemv_ppu/access_pattern.py', 'dev/gemv_ppu/run.py',
             'quactlize/execution/native.py', 'quactlize/dispatch/native.py',
             'reference/gguf_kpack.py', 'tools/gguf_internal_shape_inventory.py']
    paths += ['tools/' + f for f in (
        'kpack_execution_fixture.py', 'kpack_warmup_fixture.py', 'run_kpack_pack_gate.py', 'run_kpack_gemv_gate.py',
        'run_kpack_grouped_device_gate.py', 'run_kpack_grouped_decode_probe.py',
        'run_kpack_decode_sweep.py', 'profile_kpack_gpu_compact.py')]
    files = sorted({ROOT / p for p in paths} | set(ROOT.glob('quactlize/runtime/*.py')))
    return dict(schema=spec.SCHEMA, manifest_sha256=sha(args.bundle / 'manifest.json'),
                harness={str(p.relative_to(ROOT)): sha(p) for p in files},
                old_images={name: sha(ROOT / 'prebuilt/ppu0010' / name / 'manifest.json')
                            for name in ('q4-smallm-v1', 'q4-moe-s1-v2')},
                python_packages=package_identity(),
                screen_samples=5, rounds=6, confirmation_samples=15,
                l2_override_bytes=args.l2_bytes, workload_plan_sha256=digest(spec.plan()),
                scope=spec.plan()['scope'], production_changed=False)


def policy_key(manifest, workload):
    rows = [r for r in manifest['selection'] if tuple(r['request']) == spec.request(workload)]
    if len(rows) != 1 or rows[0]['status'] != 'SELECTED':
        raise ValueError('current policy missing/duplicated in compile receipt')
    r = rows[0]
    return f'tc:{r["parent"]}:s{r["split"]}:b{r["grid_b"]}:g{r["grid_mode"]}'


def verify_inventory(manifest):
    shapes = sorted(set(spec.DENSE) | set(spec.GROUPED))
    payloads = {'libq4_dense_io.so', 'libq4_moe_io.so'}
    if set(manifest['simt_recipes']) != {f'{n}x{k}' for n, k in shapes}:
        raise ValueError('SIMT family inventory differs')
    for n, k in shapes:
        recipes = spec.simt_inventory(n, k)
        if manifest['simt_recipes'][f'{n}x{k}'] != [asdict(c) | dict(key=c.key) for c in recipes]:
            raise ValueError('SIMT ID-to-template mapping differs')
        path = f'libq4_decode_n{n}_k{k}.so'; payloads.add(path)
        native = manifest['simt_native'][path]
        if set(native) != {f'{c.key}:a{a}' for c in recipes for a in (0, 1)}:
            raise ValueError('native SIMT context denominator differs')
        for context in native.values():
            if not context['code_lowering'] or not any(x.startswith('v.fma.f32') for x in context['operations']):
                raise ValueError('native fast-dequant/FP32 arithmetic receipt missing')
    for record in manifest['modules']:
        payloads.add(record['path'])
        if manifest['payloads'][record['path']] != record['sha256']:
            raise ValueError('TC module hash differs')
    if set(manifest['payloads']) != payloads:
        raise ValueError('undeclared/missing payload')
    for w in spec.workloads():
        key = policy_key(manifest, w)
        if key not in {c['key'] for c in spec.catalog(manifest, w)}:
            raise ValueError('current policy omitted from sweep: ' + w['id'])


def finalists(screen, current):
    result = []
    for arm in ('simt', 'tc'):
        valid = sorted((r for r in screen if r['arm'] == arm and r['status'] == 'PASS'),
                       key=lambda r: (r['median_us'], r['key']))
        result += [r['key'] for r in valid[:2]]
    if any(r['key'] == current and r['status'] == 'PASS' for r in screen) and current not in result:
        result.append(current)  # Quantify a change from today's heuristic, not a compiled default.
    return result


def immutable_reader_control(workload, config):
    n, k = workload['n'], workload['k']
    if workload['operator'] == 'grouped':
        return 'EXACT_BITS'  # All grouped families reuse the immutable 16-recipe image.
    if (n, k) in spec.smallm.SHAPES:
        prior = {spec.old_dense_recipe(n, k, c.key) for c in spec.smallm.inventory(n, k)}
        return 'EXACT_BITS' if config in prior else 'UNMEASURED_RECIPE_EXTENSION'
    return 'NEW_SHAPE_NO_OLD_IMAGE'


def validate_partial(p, ident, workload, inventory, current):
    if p['identity'] != ident or p['workload'] != workload or p['current_policy'] != current:
        raise ValueError('resume workload/identity/policy differs')
    allowed = {c['key']: c for c in inventory}
    if len(allowed) != len(inventory) or current not in allowed:
        raise ValueError('duplicate inventory or omitted current policy')
    seen = set()
    for r in p['screen']:
        key = r['key']
        if key not in allowed or key in seen or r['arm'] != allowed[key]['arm']:
            raise ValueError('screen identity differs')
        seen.add(key)
        if r['status'] == 'PASS':
            if allowed[key]['reason']:
                raise ValueError('structurally excluded candidate measured')
            validate_samples(r, 5)
            proof = r['correctness']
            if not 0 <= proof['error'] < .005:
                raise ValueError('independent GGUF proof differs')
            for f in ('zero_codes', 'zero_a', 'output_guard', 'mutable_input_replay', 'eager_graph_bits'):
                if proof.get(f) != 'PASS':
                    raise ValueError('missing correctness control: ' + f)
            execution = r['execution']
            if r['arm'] == 'simt':
                if execution['scope'] != 'F32_DIRECT_S1_NO_ADAPTER_NO_INTER_CTA_REDUCER' or execution['split'] != 1:
                    raise ValueError('SIMT call scope differs')
                if proof.get('f16_f32_bits') != 'PASS' or 'address_model' not in r or 'native' not in r:
                    raise ValueError('SIMT arithmetic/address/ISA receipt missing')
                config = spec.recipe(workload['n'], workload['k'], allowed[key]['recipe'])
                if proof.get('immutable_reader') != immutable_reader_control(workload, config):
                    raise ValueError('immutable reader control missing/different')
            elif execution['scope'] not in TC_SCOPES or execution['split'] != allowed[key]['split']:
                raise ValueError('TC adapter/reducer call scope differs')
        elif r['status'] == 'STRUCTURAL':
            if r.get('samples_us') or r.get('reason') not in STRUCTURAL:
                raise ValueError('not a structural exclusion')
        else:
            raise ValueError('unrecognized checkpoint row')
    if set(p['failed']) & seen or not set(p['failed']) <= set(allowed):
        raise ValueError('failed and successful cell identities overlap/differ')
    pairs = set()
    for r in p['confirmation']:
        pair = (r['round'], r['key'])
        if pair in pairs or pair[0] not in range(6) or pair[1] not in seen:
            raise ValueError('confirmation identity differs')
        pairs.add(pair)
        validate_samples(r, 15)


def best_rows(p):
    chosen = finalists(p['screen'], p['current_policy'])
    med = {key: statistics.median(r['median_us'] for r in p['confirmation'] if r['key'] == key)
           for key in chosen}
    best = {}
    for arm in ('simt', 'tc'):
        candidates = [key for key in chosen if key.startswith(arm + ':')]
        if candidates:
            winner = min(candidates, key=lambda key: (med[key], key))
            rounds = [r['median_us'] for r in sorted(p['confirmation'], key=lambda r: r['round']) if r['key'] == winner]
            best[arm] = dict(key=winner, median_us=med[winner], round_medians_us=rounds,
                             round_span_pct=100 * (max(rounds) - min(rounds)) / med[winner])
    return best, med.get(p['current_policy'])


def validate_result(p, ident, workload, inventory, current):
    validate_partial(p, ident, workload, inventory, current)
    if p['status'] != 'PASS' or p['failed'] or len(p['screen']) != len(inventory):
        raise ValueError('case is incomplete')
    chosen = finalists(p['screen'], current)
    expected = {(rr, key) for rr in range(6) for key in chosen}
    if len(p['confirmation']) != len(expected) or {(r['round'], r['key']) for r in p['confirmation']} != expected:
        raise ValueError('confirmation denominator differs')
    best, current_us = best_rows(p)
    if set(best) != {'simt', 'tc'} or p['best'] != best or p['current_policy_us'] != current_us:
        raise ValueError('best/current policy timing differs')
    if p['simt_vs_tc_pct'] != 100 * (best['simt']['median_us'] / best['tc']['median_us'] - 1):
        raise ValueError('relative timing differs')
    active = 1 if workload['operator'] == 'dense' else 8
    n, k = workload['n'], workload['k']
    if (p['copies'] * active * n * k * 9 / 16 < 2.25 * p['device']['l2_bytes'] or
            p['calls_per_graph'] % p['copies'] or p['calls_per_graph'] < 2 * p['copies']):
        raise ValueError('cold ring does not cover verified L2')
    if p['first_launch'] != 'EXCLUDED' or p['production_changed']:
        raise ValueError('measurement/production scope differs')
    if workload['operator'] == 'grouped':
        ids = spec.routed_ids(workload['tokens'], workload['router'])
        if p['actual_ids_sha256'] != digest(ids.tolist()) or p['max_rows'] != int(np.bincount(ids.ravel()).max()):
            raise ValueError('actual router does not match declared workload')
    return p


def child(args, manifest, workload):
    key = workload['id']; ident = identity(args)
    inventory = spec.catalog(manifest, workload); current = policy_key(manifest, workload)
    checkpoint = args.output / (key + '.partial.json'); target = args.output / (key + '.json')
    p = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(
        identity=ident, workload=workload, current_policy=current, screen=[], failed={}, confirmation=[])
    validate_partial(p, ident, workload, inventory, current)
    if args.retry_failed and p['failed']:
        p['failure_history'] = p.get('failure_history', []) + list(p['failed'].values())
        p['failed'] = {}
        p['previous_confirmation'] = p.get('previous_confirmation', []) + p['confirmation']
        p['confirmation'] = []
    base = None; instances = {}; active_key = None
    def persist(): save(checkpoint, p)
    try:
        base = Base(args, workload)
        device_path = args.output / 'device.json'
        if p.get('device', base.device) != base.device or (device_path.exists() and
                json.loads(device_path.read_text()) != base.device):
            raise ValueError('physical device differs from campaign')
        p['device'] = base.device; persist()
        parents = {r['parent']['symbol']: r for r in manifest['modules']}
        configs = {c['key']: c for c in inventory}
        def make(c):
            if c['arm'] == 'simt': return None
            cls = DenseTC if workload['operator'] == 'dense' else GroupedTC
            return cls(base, args.bundle, parents[c['parent']], c)
        def measure(c, instance, count):
            return base.measure(c['recipe'], count) if instance is None else instance.measure(count)
        if not args.profile:
            seen = {r['key'] for r in p['screen']}
            for i, c in enumerate(inventory):
                if c['key'] in seen or c['key'] in p['failed']: continue
                active_key = c['key']; instance = None
                print(f'Q4_DECODE_PROGRESS case={key} phase=screen cell={i+1}/{len(inventory)} key={active_key}', flush=True)
                try:
                    if c['reason']: raise Unsupported(c['reason'])
                    instance = make(c)
                    proof = base.correctness(c['recipe']) if instance is None else instance.correctness()
                    if proof.get('gpu_ids_replay') == 'PASS': proof['mutable_input_replay'] = 'PASS'
                    samples = measure(c, instance, 5)
                    row = dict(key=active_key, arm=c['arm'], status='PASS', correctness=proof,
                               samples_us=samples, median_us=statistics.median(samples))
                    if instance is None:
                        recipe = spec.recipe(base.n, base.k, c['recipe'])
                        row.update(execution=dict(scope='F32_DIRECT_S1_NO_ADAPTER_NO_INTER_CTA_REDUCER', split=1,
                            output_rounding='F32', arithmetic='PER_WEIGHT_FP16' if recipe.reader == 0 else 'FP32_GROUP_AFFINE'),
                            address_model=access(recipe, base), native=manifest['simt_native'][
                                f'libq4_decode_n{base.n}_k{base.k}.so'][c['recipe'] + ':a1'])
                    else: row['execution'] = instance.receipt()
                except Unsupported as exc:
                    row = dict(key=active_key, arm=c['arm'], status='STRUCTURAL', reason=str(exc), samples_us=[])
                finally:
                    if instance: instance.close()
                p['screen'].append(row); persist()
                print('Q4_DECODE_SCREEN ' + json.dumps({f: row[f] for f in ('key', 'status', 'reason', 'median_us') if f in row}), flush=True)
            chosen = finalists(p['screen'], current)
            if any(r['key'] not in chosen for r in p['confirmation']):
                p['previous_confirmation'] = p.get('previous_confirmation', []) + p['confirmation']
                p['confirmation'] = []; persist()
        else:
            result = json.loads(target.read_text())
            validate_result(result, ident, workload, inventory, current)
            chosen = [r['key'] for r in result['best'].values()]
        for name in chosen:
            active_key = name; instances[name] = make(configs[name])
            if args.profile:
                c = configs[name]; instance = instances[name]
                (base if instance is None else instance).poison()
                launch = (lambda: base.invoke(c['recipe'])) if instance is None else instance.launch
                stream = base.case_r.stream if instance is None else instance.r.stream
                checked(launch(), 'profile warmup excluded'); base.sdk.synchronize(stream)
                with AcuRange(base.sdk):
                    checked(launch(), 'profile complete call'); base.sdk.synchronize(stream)
                got = base.read() if instance is None else instance.read()
                if error(got, base.data) >= .005: raise ValueError('profile numeric failure')
                print('Q4_DECODE_PROFILE ' + json.dumps(dict(case=key, key=name, status='PASS')), flush=True)
        if args.profile: return 0
        done = {(r['round'], r['key']) for r in p['confirmation']}
        for rr in range(6):
            for name in chosen[::1 if rr % 2 == 0 else -1]:
                if (rr, name) in done: continue
                active_key = name; samples = measure(configs[name], instances[name], 15)
                p['confirmation'].append(dict(key=name, round=rr, samples_us=samples, median_us=statistics.median(samples)))
                persist()
            print(f'Q4_DECODE_PROGRESS case={key} phase=confirm round={rr+1}/6', flush=True)
        best, current_us = best_rows(p)
        p.update(best=best, current_policy_us=current_us,
                 status='PASS' if not p['failed'] and len(best) == 2 else 'INCOMPLETE',
                 copies=base.copies, calls_per_graph=base.calls_per_graph, weight_sha256=base.weight_sha256,
                 actual_ids_sha256=digest(base.data['ids'][:, :8].tolist()) if base.ids else None,
                 active_experts=int(np.unique(base.data['expert']).size),
                 max_rows=int(np.bincount(base.data['expert']).max()),
                 first_launch='EXCLUDED', cache='ROTATING_ACTIVE_WEIGHTS_AT_LEAST_2_25_L2', production_changed=False)
        if len(best) == 2:
            p['simt_vs_tc_pct'] = 100 * (best['simt']['median_us'] / best['tc']['median_us'] - 1)
        if p['status'] == 'PASS': validate_result(p, ident, workload, inventory, current)
        save(target, p)
        print('Q4_DECODE_RESULT ' + json.dumps({f: p[f] for f in ('workload', 'status', 'best', 'current_policy_us')}), flush=True)
        return 0 if p['status'] == 'PASS' else 2
    except BaseException as exc:
        if active_key and not args.profile:
            p['failed'][active_key] = dict(key=active_key, error=str(exc), phase='screen_or_confirmation')
            p['screen'] = [r for r in p['screen'] if r['key'] != active_key]
            p['confirmation'] = [r for r in p['confirmation'] if r['key'] != active_key]
            persist()
        raise
    finally:
        for instance in instances.values():
            if instance: instance.close()
        if base: base.close()


def selected_workloads(args):
    return [w for w in spec.workloads() if args.operators == 'all' or w['operator'] == args.operators]


def profile_workload(w):
    if w['operator'] == 'dense':
        return w['tokens'] in (1, 8) and (w['n'], w['k']) in ((512, 2048), (4096, 4096), (5120, 25600))
    return (w['n'], w['k']) in spec.GROUPED[:4] and (w['tokens'], w['channels'], w['router']) in (
        (1, 1, 'spread'), (8, 8, 'real'), (8, 8, 'cluster'))


def run_process(command, log):
    with log.open('a') as output:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            output.write(line); output.flush(); print(line, end='', flush=True)
        return process.wait()


def run_case(command, output, key, size, retry_failed):
    previous = None
    for attempt in range(size + 2):
        rc = run_process(command + (['--retry-failed'] if attempt == 0 and retry_failed else []), output / (key + '.log'))
        if rc in (0, 2): return rc
        path = output / (key + '.partial.json')
        p = json.loads(path.read_text()) if path.exists() else {}
        progress = (len(p.get('screen', [])), len(p.get('failed', {})))
        if not progress[1] or progress == previous: return rc
        previous = progress
        print(f'Q4_DECODE_RESTART case={key} fresh_process=1 failed_cells_preserved=1', flush=True)
    return rc


def profile_case(args, command, key, ident):
    result = json.loads((args.output / (key + '.json')).read_text())
    result_sha = sha(args.output / (key + '.json'))
    receipt = args.output / (key + '.acu.json')
    old = json.loads(receipt.read_text()) if receipt.exists() else None
    if (old and old.get('identity') == ident and old.get('result_sha256') == result_sha and all(
            (args.output / old[f]).is_file() and sha(args.output / old[f]) == old[f + '_sha256'] for f in ('report', 'log'))):
        return old
    report = args.output / (key + f'.{time.time_ns()}.acurep'); log = args.output / (key + '.acu.log')
    with log.open('w') as output:
        rc = subprocess.run(acu_launch_command(args.acu, report, command + ['--profile']), stdout=output, stderr=subprocess.STDOUT).returncode
    files = [p for p in args.output.glob(report.name + '*') if p.is_file() and p.stat().st_size]
    entries = [json.loads(line.split(' ', 1)[1]) for line in log.read_text(errors='replace').splitlines() if line.startswith('Q4_DECODE_PROFILE ')]
    if (rc or len(files) != 1 or len(entries) != 2 or {r.get('key') for r in entries} != {v['key'] for v in result['best'].values()}
            or any(r.get('case') != key or r.get('status') != 'PASS' for r in entries)):
        raise ValueError(f'ACU incomplete rc={rc}; log={log}')
    proof = dict(identity=ident, case=key, result_sha256=result_sha, entries=entries,
                 report=files[0].name, report_sha256=sha(files[0]), log=log.name, log_sha256=sha(log))
    save(receipt, proof)
    return proof


def summarize(args, manifest, ident, workloads, failures, profiles, started):
    rows = []; exported = []
    for w in workloads:
        path = args.output / (w['id'] + '.json')
        if not path.exists(): continue
        r = json.loads(path.read_text()); rows.append(r)
        if r.get('status') != 'PASS': continue
        validate_result(r, ident, w, spec.catalog(manifest, w), policy_key(manifest, w))
        winner = min(r['best'].values(), key=lambda x: (x['median_us'], x['key']))
        exported.append(dict(workload=w, best=r['best'], current_policy=r['current_policy'],
                             current_policy_us=r['current_policy_us'], selected=winner,
                             evidence=path.name, evidence_sha256=sha(path),
                             scope=ident['scope'], production_admission='PENDING_REVIEW'))
    with (args.output / 'summary.tsv').open('w') as output:
        writer = csv.writer(output, delimiter='\t')
        writer.writerow(['case', 'status', 'simt_us', 'tc_us', 'current_policy_us', 'simt_vs_tc_pct', 'simt_key', 'tc_key'])
        for r in rows:
            writer.writerow([r['workload']['id'], r['status'], *[r['best'].get(a, {}).get('median_us') for a in ('simt', 'tc')],
                             r['current_policy_us'], r.get('simt_vs_tc_pct'), *[r['best'].get(a, {}).get('key') for a in ('simt', 'tc')]])
    save(args.output / 'selection-input.json', dict(schema=spec.SCHEMA, rows=exported, production_changed=False))
    result = dict(status='PASS' if not failures and len(exported) == len(workloads) else 'INCOMPLETE',
                  expected=len(workloads), passed=len(exported), failures=failures, profiles=profiles,
                  profile_expected=sum(profile_workload(w) for w in workloads) if args.acu else 0,
                  seconds=time.monotonic() - started, production_changed=False,
                  files={p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name not in ('result.json', 'console.log')})
    save(args.output / 'result.json', result)
    print('Q4_DECODE_DONE ' + json.dumps(result | dict(files='IN_RESULT_JSON')), flush=True)
    return 0 if result['status'] == 'PASS' else 1


def main(args):
    manifest = spec.verify(args.bundle)
    verify_inventory(manifest)
    for name, value in manifest['runtime'].items():
        if sha(args.sdk / 'lib' / name) != value: raise ValueError('runtime differs: ' + name)
    if args.probe:
        sdk = SDK(args.sdk)
        lib = C.CDLL(str(args.bundle / 'libq4_decode_n256_k3072.so'), mode=C.RTLD_LOCAL)
        print('Q4_DECODE_DEVICE ' + json.dumps(probe(sdk, lib, args.l2_bytes)), flush=True)
        return 0
    if args.case:
        w = next(w for w in spec.workloads() if w['id'] == args.case)
        return child(args, manifest, w)
    args.output.mkdir(parents=True, exist_ok=True)
    ident = identity(args); workloads = selected_workloads(args); authority = args.output / 'authority.json'
    if authority.exists() and json.loads(authority.read_text()) != ident: raise ValueError('campaign identity differs')
    save(authority, ident)
    plan = dict(identity=ident, workloads=workloads, inventory={w['id']: spec.catalog(manifest, w) for w in workloads},
                current_policy={w['id']: policy_key(manifest, w) for w in workloads})
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan: raise ValueError('campaign subset/inventory differs')
    save(plan_path, plan)
    common = [sys.executable, '-u', __file__, '--sdk', str(args.sdk), '--bundle', str(args.bundle),
              '--output', str(args.output), '--l2-bytes', str(args.l2_bytes)]
    # The parent holds no GPU context while child/ACU work runs.
    observed = subprocess.run(common + ['--probe'], capture_output=True, text=True)
    devices = [json.loads(s.split(' ', 1)[1]) for s in observed.stdout.splitlines() if s.startswith('Q4_DECODE_DEVICE ')]
    if observed.returncode or len(devices) != 1: raise ValueError('device probe failed: ' + observed.stdout + observed.stderr)
    device = devices[0]; device_path = args.output / 'device.json'
    if device_path.exists() and json.loads(device_path.read_text()) != device: raise ValueError('physical device differs')
    save(device_path, device)
    print('Q4_DECODE_START ' + json.dumps(dict(cases=len(workloads), dense=sum(w['operator'] == 'dense' for w in workloads),
          grouped=sum(w['operator'] == 'grouped' for w in workloads), profiles=sum(profile_workload(w) for w in workloads) if args.acu else 0,
          compile='NONE', jit='NONE', screen_samples=5, rounds=6, confirmation_samples=15)), flush=True)
    started = time.monotonic(); failures = []; profiles = []; new_seconds = []; completed = 0
    for w in workloads:
        key = w['id']; command = common + ['--case', key]; path = args.output / (key + '.json')
        old = json.loads(path.read_text()) if path.exists() else None
        if old and old.get('status') == 'PASS':
            validate_result(old, ident, w, plan['inventory'][key], policy_key(manifest, w))
            if old['device'] != device: raise ValueError('old result physical device differs')
            rc = 0; print(f'Q4_DECODE_RESUME case={key}', flush=True)
        else:
            before = time.monotonic()
            rc = run_case(command, args.output, key, len(plan['inventory'][key]), args.retry_failed)
            new_seconds.append(time.monotonic() - before)
        if rc: failures.append(dict(case=key, phase='sweep', rc=rc))
        completed += 1
        remaining = statistics.mean(new_seconds) * (len(workloads) - completed) / 60 if new_seconds else None
        print('Q4_DECODE_ETA ' + json.dumps(dict(completed=completed, total=len(workloads), elapsed_minutes=(time.monotonic()-started)/60,
              remaining_sweep_minutes=remaining, profile_minutes='NOT_ESTIMATED', method='OBSERVED_MEAN_CASE_WALL', advisory_only=True)), flush=True)
        if rc == 0 and args.acu and profile_workload(w):
            try: profiles.append(profile_case(args, command, key, ident))
            except Exception as exc:
                failures.append(dict(case=key, phase='acu', error=str(exc)))
                print(f'Q4_DECODE_PROFILE_FAIL case={key} error={exc} timing_preserved=1', flush=True)
    return summarize(args, manifest, ident, workloads, failures, profiles, started)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=spec.BUNDLE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--operators', choices=('all', 'dense', 'grouped'), default='all')
    p.add_argument('--l2-bytes', type=int, default=0)
    p.add_argument('--acu', type=Path)
    p.add_argument('--case', choices=[w['id'] for w in spec.workloads()])
    p.add_argument('--probe', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--retry-failed', action='store_true')
    args = p.parse_args()
    if args.profile and not args.case: p.error('--profile needs --case')
    try: raise SystemExit(main(args))
    except Exception: traceback.print_exc(); raise SystemExit(1)

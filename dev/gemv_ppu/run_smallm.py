#!/usr/bin/env python3
"""Cold Q4 M2..8 scan: tuned SIMT, raw FP32 reference and selected/scanned TC."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu import smallm as spec
from dev.gemv_ppu.smallm_bench import Bench, tc_key, tc_reason
from dev.gemv_ppu.run_cold_geometry import probe_device
from dev.gemv_ppu.run_config_sweep import execute
from tools.profile_kpack_gpu_compact import acu_launch_command

PREFIX = 'Q4_SMALLM_CELL '
FAIL_PREFIX = 'Q4_SMALLM_FAILURE '
ARMS = ('reference', 'kpack', 'tc')
ROUNDS, SCREEN_SAMPLES, CONFIRM_SAMPLES = 6, 5, 15


def policy_key(manifest, m, n, k):
    rows = [r for r in manifest['tc_selection'] if r['request'] == [12,0,m,n,k,1,m]]
    if len(rows) != 1:
        raise ValueError('missing/duplicate TC selection')
    return tc_key({'symbol':rows[0]['parent']}, rows[0]['split'])


def keys_for(arm, n, k, manifest):
    if arm == 'kpack':
        return [c.key for c in spec.inventory(n,k)]
    if arm == 'reference':
        return ['-'.join(map(str,c)) for c in spec.reference_recipes(n,k)]
    return [tc_key(r['parent'],s) for r in manifest['modules'] for s in (1,2,4,8)]


def item_id(m, key):
    return f'm{m}/{key}'


def parse_row(row, arm, n, k, m, key, phase, samples, manifest, profile=False):
    if (m not in spec.MS or (n,k) not in spec.SHAPES or key not in keys_for(arm,n,k,manifest) or
        row.get('arm') != arm or row.get('shape') != [m,n,k] or row.get('key') != key or row.get('phase') != phase):
        raise ValueError('small-M row identity differs')
    parent, reason = None, None
    if arm == 'tc':
        symbol, s = key.rsplit(':s',1)
        parent = next(r['parent'] for r in manifest['modules'] if r['parent']['symbol'] == symbol)
        reason = tc_reason(parent,k,int(s))
    if reason:
        if row.get('status') != 'STRUCTURAL' or row.get('reason') != reason or row.get('samples_us') != [] or row.get('median_us') is not None or profile:
            raise ValueError('structural exclusion differs from module contract')
        return row
    if row.get('status') != 'PASS' or not isinstance(row.get('error'), (int,float)) or not math.isfinite(row['error']) or not 0 <= row['error'] < .005:
        raise ValueError('dirty/missing numeric cell')
    for name in ('zero_code_negative','zero_a_check','output_guard','row_alias_negative','replay_check'):
        if row.get(name) != 'PASS':
            raise ValueError('missing numeric check: '+name)
    if row.get('row_alias_negative_scope') != 'HOST_ORACLE_PLANT':
        raise ValueError('row-alias oracle scope differs')
    values = row.get('samples_us', [])
    if len(values) != (0 if profile else samples) or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('sample count/finite values differ')
    if (profile and row.get('median_us') is not None) or (not profile and (not math.isfinite(row.get('median_us',float('nan'))) or abs(statistics.median(values)-row['median_us']) > 1e-8)):
        raise ValueError('median differs from samples')
    if (row.get('output_type') != ('F16' if arm=='tc' else 'F32') or row.get('accumulator') != 'F32' or
        row.get('timing_scope') != 'RESIDENT_FULL_CALL_INCLUDING_TC_REDUCER_NO_HOST_SETUP_OR_JIT' or
        row.get('cache_scope') != ('ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2') or
        row.get('storage') != ('RAW_GGUF' if arm=='reference' else 'CANONICAL_KPACK4')):
        raise ValueError('precision/whole-call/cache contract differs')
    copies, size = row.get('copies'), row.get('weight_bytes')
    if (type(copies) is not int or copies < 2 or size != n*k*9//16 or row['device']['l2_bytes'] <= 0 or
        copies*size < 2.25*row['device']['l2_bytes'] or
        row.get('calls_per_graph') != max(2,math.ceil(32/copies))*copies):
        raise ValueError('ring is too small or a partial traversal')
    if not isinstance(row.get('input_sha256'),str) or len(row['input_sha256']) != 64:
        raise ValueError('activation identity missing')
    if arm == 'kpack':
        c = spec.lookup(n,k,key)
        if (row.get('geometry') != c.geometry(n,k,m) or row.get('weight_arithmetic') != c.arithmetic or
            not isinstance(row.get('matched_fp32_sha256'),str) or len(row['matched_fp32_sha256']) != 64 or
            row.get('multirow_scope') != 'ONE_LAUNCH_ROW_GRID'):
            raise ValueError('multirow geometry/arithmetic/blue control differs')
        immutable = row.get('immutable_m1_sha256')
        if ((key==spec.selected(n,k).key and (not isinstance(immutable,str) or len(immutable)!=64)) or
            (key!=spec.selected(n,k).key and immutable is not None)):
            raise ValueError('immutable M1 evidence differs')
        models = row.get('access_models',[])
        if len(models) != 1:
            raise ValueError('missing observed address model')
        model = models[0]
        bases = model.get('base_mod128', model.get('first_warp',{}).get('base_mod128'))
        if (not isinstance(bases,dict) or set(bases) != {'A','B','metadata'} or
            any(type(v) is not int or v < 0 or v >= 128 or v % 16 for v in bases.values()) or
            model != spec.access(c,n,k,m,bases)):
            raise ValueError('lane address model differs')
    elif arm == 'tc':
        selection = row.get('selection',{})
        record = next(r for r in manifest['modules'] if r['parent'] == parent)
        if (selection.get('parent') != parent or selection.get('build_key') != record['key'] or
            selection.get('split') != int(key.rsplit(':s',1)[1]) or selection.get('algorithm') != 'ORDINARY' or
            selection.get('grid') != 0 or selection.get('current_policy_key') != policy_key(manifest,m,n,k) or
            selection.get('is_current_policy') != (key==policy_key(manifest,m,n,k)) or
            row.get('weight_arithmetic') != 'FQ_TC_F16_RECONSTRUCTION' or row.get('multirow_scope') != 'TC_COMPLETE_CALL'):
            raise ValueError('TC parent/policy/Split-K differs')
    elif row.get('weight_arithmetic') != 'PER_WEIGHT_FP16':
        raise ValueError('raw reference arithmetic differs')
    return row


def child(args):
    manifest = spec.verify(args.candidate, sources=False)
    items = json.loads(args.items)
    if not items or len({(m,key) for m,key in items}) != len(items):
        raise ValueError('empty/duplicate child items')
    for m,key in items:
        if m not in spec.MS or key not in keys_for(args.arm,args.n,args.k,manifest):
            raise ValueError('child item outside compiled scope')
    bench = None
    try:
        bench = Bench(args,manifest)
        if (bench.n,bench.k) != (args.n,args.k):
            raise ValueError('fixture shape differs')
        for m,key in items:
            try:
                row = bench.measure(m,key,args.samples,args.profile)
                row['phase'] = args.phase
                parse_row(row,args.arm,args.n,args.k,m,key,args.phase,args.samples,manifest,args.profile)
                print(PREFIX+json.dumps(row),flush=True)
            except Exception as exc:
                print(FAIL_PREFIX+json.dumps(dict(arm=args.arm,shape=[m,args.n,args.k],key=key,phase=args.phase,error=str(exc))),flush=True)
                traceback.print_exc()
                return 1  # remaining cells restart in a fresh device process
        return 0
    finally:
        if bench:
            bench.close()


def shortlist(screen, m, arm, n, k, manifest):
    rows = [r for r in screen.values() if r['shape'][0]==m and r['status']=='PASS']
    keys = [r['key'] for r in sorted(rows,key=lambda r:r['median_us'])[:2]]
    if arm == 'tc':
        selected = policy_key(manifest,m,n,k)
        if any(r['key']==selected for r in rows) and selected not in keys:
            keys.append(selected)
    return keys


def summarize(m,n,k,records,shortlists,manifest,screen_complete):
    medians = {arm:{key:statistics.median(row['median_us'] for row in values)
                   for key,values in by_key.items() if len(values)==ROUNDS}
               for arm,by_key in records.items()}
    complete = screen_complete and all(set(medians[arm])==set(shortlists[arm]) and len(shortlists[arm])>=2 for arm in ARMS)
    current = policy_key(manifest,m,n,k)
    complete = complete and current in medians['tc']
    best = {arm:min(values,key=values.get) for arm,values in medians.items() if values}
    inputs = {row['input_sha256'] for values in records.values() for rows in values.values() for row in rows}
    if len(inputs) > 1:
        raise ValueError('comparison arms used different activations')
    result = dict(shape=[m,n,k],status='PASS' if complete else 'INCOMPLETE',median_us=medians,
                  best=best,tc_current_key=current,reference_limit_pct=5,global_optimum_proven=False)
    if complete:
        simt, ref, tc = [medians[arm][best[arm]] for arm in ('kpack','reference','tc')]
        result.update(kpack_us=simt,reference_us=ref,tc_best_us=tc,tc_current_us=medians['tc'][current],
            kpack_vs_reference_pct=100*(simt/ref-1),kpack_vs_tc_best_pct=100*(simt/tc-1),
            kpack_vs_tc_current_pct=100*(simt/medians['tc'][current]-1),
            reference_verdict='WITHIN_5_PERCENT' if simt<=ref*1.05 else 'PARITY_OPEN',
            faster_implementation='SIMT' if simt<tc else 'TENSOR_CORE')
    return result


def run(args):
    manifest = spec.verify(args.candidate)
    spec.medium_refine.verify(*(ROOT/'prebuilt/ppu0010'/p for p in spec.PACKAGES))
    device = probe_device(args)
    names = set(manifest['source_hashes']) | {'dev/gemv_ppu/run_smallm.py','dev/gemv_ppu/smallm_bench.py',
        'dev/gemv_ppu/run_cold_geometry.py','dev/gemv_ppu/run_bload.py','dev/gemv_ppu/run_config_sweep.py',
        'tools/profile_kpack_gpu_compact.py','tools/run_kpack_gemv_gate.py',
        'tools/run_kpack_grouped_decode_probe.py','tools/run_kpack_grouped_device_gate.py','tools/run_kpack_pack_gate.py'}
    authority = dict(schema=spec.SCHEMA,manifest=digest(args.candidate/'manifest.json'),device=device,
        sources={name:digest(ROOT/name) for name in sorted(names)},
        runtime={name:digest(args.sdk/'lib'/name) for name in manifest['runtime']},
        fixtures={f'{n}x{k}':digest(args.fixtures/f'q12-n{n}-k{k}-e1-c1.npz') for n,k in spec.SHAPES},
        rounds=ROUNDS,screen_samples=SCREEN_SAMPLES,confirm_samples=CONFIRM_SAMPLES,
        requested_m=list(spec.MS),profiles=not args.skip_acu)
    args.output.mkdir(parents=True,exist_ok=True)
    receipt = args.output/'authority.json'
    if receipt.exists() and json.loads(receipt.read_text()) != authority:
        raise ValueError('resume source/device/runtime/fixture identity differs')
    receipt.write_text(json.dumps(authority,indent=2)+'\n')
    (args.output/'build-manifest.json').write_bytes((args.candidate/'manifest.json').read_bytes())
    (args.output/'plan.json').write_text(json.dumps(spec.plan(),indent=2)+'\n')
    diff = [name for name,sha in authority['runtime'].items() if manifest['runtime'][name]!=sha]
    if diff:
        print('Q4_SMALLM_SDK_DIFFERENCE recorded='+','.join(diff)+' admission=REAL_DEVICE_NUMERIC',flush=True)
    result = dict(status='RUNNING',cases=[],profiles=[],failures=[],production_changed=False)
    started = time.monotonic()

    def save():
        (args.output/'summary.json').write_text(json.dumps(result)+'\n')

    def batch(arm,n,k,items,phase,profile=False):
        samples = 0 if profile else SCREEN_SAMPLES if phase=='screen' else CONFIRM_SAMPLES
        suffix = '.'+hashlib.sha256(json.dumps(items).encode()).hexdigest()[:12] if profile else ''
        path = args.output/f'n{n}-k{k}-{arm}-{phase}{suffix}.json'
        cached = json.loads(path.read_text()) if path.exists() else {}
        wanted = {item_id(m,key):(m,key) for m,key in items}
        parsed_logs = {}
        for identity,entry in cached.items():
            # A recovered screen failure may change the shortlist. Keep
            # already-valid confirmations, validate their compiled identity,
            # and return only the currently requested subset below.
            m,key = entry['row']['shape'][0],entry['row']['key']
            if identity != item_id(m,key) or m not in spec.MS or key not in keys_for(arm,n,k,manifest):
                raise ValueError('cached item is outside the compiled inventory')
            log = args.output/entry['log']
            if log.parent != args.output or digest(log) != entry['log_sha256']:
                raise ValueError('cached log differs')
            if log not in parsed_logs:
                parsed_logs[log] = [json.loads(s[len(PREFIX):]) for s in log.read_text().splitlines() if s.startswith(PREFIX)]
            rows = [r for r in parsed_logs[log] if r['shape'][0]==m and r['key']==key]
            if len(rows)!=1:
                raise ValueError('cached cell missing/duplicate')
            row = parse_row(rows[0],arm,n,k,m,key,phase,samples,manifest,profile)
            if row!=entry['row'] or (row['status']=='PASS' and row['device']!=device):
                raise ValueError('cached row/device differs')
            if profile:
                report = args.output/entry['report']
                if report.parent!=args.output or digest(report)!=entry['report_sha256']:
                    raise ValueError('cached profile differs')
        remaining = [x for x in items if item_id(*x) not in cached]
        while remaining:
            prefix = args.output/f'n{n}-k{k}-{arm}-{phase}.{time.time_ns()}'
            log = prefix.with_name(prefix.name+('.acu.log' if profile else '.log'))
            cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--child','--sdk',str(args.sdk),
                '--candidate',str(args.candidate),'--bundle',str(args.bundle),'--fixture',str(args.fixtures/f'q12-n{n}-k{k}-e1-c1.npz'),
                '--n',str(n),'--k',str(k),'--arm',arm,'--items',json.dumps(remaining),'--phase',phase,
                '--samples',str(samples),'--l2-bytes',str(args.l2_bytes)]
            if profile:
                cmd = acu_launch_command(args.acu,prefix,cmd+['--profile'])
            print(f'Q4_SMALLM_PROGRESS shape={n}x{k} arm={arm} phase={phase} pending={len(remaining)} elapsed_s={time.monotonic()-started:.1f}',flush=True)
            with log.open('x') as f:
                rc = execute(cmd,f)
            seen, failed = set(), set()
            requested = {item_id(*x):x for x in remaining}
            for line in log.read_text().splitlines():
                if line.startswith(PREFIX):
                    raw = json.loads(line[len(PREFIX):])
                    identity = item_id(raw['shape'][0],raw['key'])
                    if identity not in requested or identity in seen:
                        raise ValueError('unexpected/duplicate child row')
                    seen.add(identity)
                    m,key = requested[identity]
                    row = parse_row(raw,arm,n,k,m,key,phase,samples,manifest,profile)
                    if row['status']=='PASS' and row['device']!=device:
                        raise ValueError('device changed')
                    entry = dict(row=row,log=log.name,log_sha256=digest(log))
                    if profile:
                        reports = list(args.output.glob(prefix.name+'*.acurep'))
                        if len(reports)!=1:
                            raise ValueError('profile report missing/duplicate')
                        entry.update(report=reports[0].name,report_sha256=digest(reports[0]))
                    cached[identity] = entry
                elif line.startswith(FAIL_PREFIX):
                    error = json.loads(line[len(FAIL_PREFIX):])
                    identity = item_id(error['shape'][0],error['key'])
                    if identity not in requested or identity in seen or error['arm']!=arm or error['phase']!=phase or error['shape'][1:]!=[n,k]:
                        raise ValueError('failure identity differs')
                    seen.add(identity); failed.add(identity)
                    result['failures'].append(error | dict(log=log.name))
                    print(FAIL_PREFIX+json.dumps(result['failures'][-1]),flush=True)
            path.write_text(json.dumps(cached)+'\n')  # Preserve complete independent cells even if child crashed later.
            if not seen or (rc and not failed):
                raise ValueError(f'child infrastructure rc={rc}; log={log}')
            remaining = [x for x in remaining if item_id(*x) not in seen]
        return {identity:cached[identity]['row'] for identity in wanted if identity in cached}

    for n,k in spec.SHAPES:
        screens, selected, records = {}, {}, {}
        for arm in ARMS:
            items = [(m,key) for m in spec.MS for key in keys_for(arm,n,k,manifest)]
            try:
                screens[arm] = batch(arm,n,k,items,'screen')
            except Exception as exc:
                screens[arm] = {}
                result['failures'].append(dict(shape=[n,k],arm=arm,phase='screen',error=str(exc)))
            selected[arm] = {m:shortlist(screens[arm],m,arm,n,k,manifest) for m in spec.MS}
            records[arm] = {m:{key:[] for key in selected[arm][m]} for m in spec.MS}
            save()
        for turn in range(ROUNDS):
            for arm in ARMS if turn%2==0 else ARMS[::-1]:
                items = [(m,key) for m in spec.MS for key in selected[arm][m]]
                if not items:
                    continue
                try:
                    rows = batch(arm,n,k,items if turn%2==0 else items[::-1],f'r{turn}')
                    for row in rows.values():
                        if row['status']=='PASS':
                            records[arm][row['shape'][0]][row['key']].append(row)
                except Exception as exc:
                    result['failures'].append(dict(shape=[n,k],arm=arm,phase=f'r{turn}',error=str(exc)))
                save()
        for m in spec.MS:
            full = all({r['key'] for r in screens[arm].values() if r['shape'][0]==m}==set(keys_for(arm,n,k,manifest)) for arm in ARMS)
            case = dict(shape=[m,n,k],screen={arm:{r['key']:r for r in rows.values() if r['shape'][0]==m} for arm,rows in screens.items()},
                        shortlist={arm:selected[arm][m] for arm in ARMS},records={arm:records[arm][m] for arm in ARMS})
            case['comparison'] = summarize(m,n,k,case['records'],case['shortlist'],manifest,full)
            result['cases'].append(case)
            print('Q4_SMALLM_RESULT '+json.dumps(case['comparison']),flush=True)
            save()
            if args.skip_acu or m not in (2,8) or case['comparison']['status']!='PASS':
                continue
            targets = [(arm,case['comparison']['best'][arm]) for arm in ARMS]
            current = case['comparison']['tc_current_key']
            if ('tc',current) not in targets:
                targets.append(('tc',current))
            for arm,key in targets:
                try:
                    rows = batch(arm,n,k,[(m,key)],'profile',True)
                    result['profiles'].append(dict(shape=[m,n,k],arm=arm,key=key,status='PASS',row=rows[item_id(m,key)]))
                except Exception as exc:
                    result['profiles'].append(dict(shape=[m,n,k],arm=arm,key=key,status='FAIL',error=str(exc)))
                save()
    if any(digest(ROOT/name)!=sha for name,sha in authority['sources'].items()):
        raise ValueError('source changed during campaign')
    if (any(digest(args.sdk/'lib'/name)!=sha for name,sha in authority['runtime'].items()) or
        any(digest(args.fixtures/f'q12-n{n}-k{k}-e1-c1.npz')!=authority['fixtures'][f'{n}x{k}'] for n,k in spec.SHAPES)):
        raise ValueError('SDK/fixture changed during campaign')
    spec.verify(args.candidate)
    ok = (not result['failures'] and len(result['cases'])==42 and all(c['comparison']['status']=='PASS' for c in result['cases'])
          and all(p['status']=='PASS' for p in result['profiles']))
    result.update(status='PASS' if ok else 'INCOMPLETE',seconds=time.monotonic()-started)
    save()
    fields = ['M','N','K','status','kpack_key','kpack_us','reference_key','reference_us','tc_best_key','tc_best_us',
              'tc_current_key','tc_current_us','kpack_vs_reference_pct','kpack_vs_tc_best_pct','kpack_vs_tc_current_pct','faster_implementation']
    with (args.output/'summary.tsv').open('w') as f:
        w = csv.DictWriter(f,fields,delimiter='\t'); w.writeheader()
        for c in result['cases']:
            r = c['comparison']
            row = dict(zip(('M','N','K'),r['shape'])) | {k:r.get(k,'NA') for k in fields[3:]}
            row.update(kpack_key=r['best'].get('kpack','NA'),reference_key=r['best'].get('reference','NA'),tc_best_key=r['best'].get('tc','NA'))
            w.writerow(row)
    print(f'Q4_SMALLM_COMPLETE status={result["status"]} shapes={len(result["cases"])}/42 elapsed_s={result["seconds"]:.1f} production=UNCHANGED results={args.output}',flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--candidate',type=Path,default=ROOT/'prebuilt/ppu0010/q4-smallm-v1')
    p.add_argument('--bundle',type=Path,default=ROOT/'prebuilt/ppu0010/q4-simt-ab-v1')
    p.add_argument('--fixtures',type=Path); p.add_argument('--fixture',type=Path); p.add_argument('--output',type=Path)
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--acu',type=Path); p.add_argument('--skip-acu',action='store_true')
    p.add_argument('--child',action='store_true'); p.add_argument('--profile',action='store_true')
    p.add_argument('--arm',choices=ARMS); p.add_argument('--items'); p.add_argument('--phase',default='screen')
    p.add_argument('--samples',type=int,default=SCREEN_SAMPLES); p.add_argument('--n',type=int); p.add_argument('--k',type=int)
    a = p.parse_args()
    for name in ('sdk','candidate','bundle'):
        setattr(a,name,getattr(a,name).resolve(strict=True))
    if a.l2_bytes<0:
        p.error('negative L2 capacity')
    if a.child:
        if not a.items or a.fixture is None or a.arm is None or (a.n,a.k) not in spec.SHAPES:
            p.error('child requires items/fixture/arm/shape')
        a.fixture = a.fixture.resolve(strict=True)
        return child(a)
    if a.fixtures is None or a.output is None:
        p.error('fixtures/output required')
    a.fixtures = a.fixtures.resolve(strict=True); a.output = a.output.resolve()
    a.acu = a.acu or a.sdk/'asight/bin/acu'
    return run(a)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

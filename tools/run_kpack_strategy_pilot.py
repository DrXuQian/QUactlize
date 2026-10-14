#!/usr/bin/env python3
"""Small TC execution/graph-timer gate, reusing an existing resource run.

No device compiler or module build is launched; the SDK host ABI is checked.
No optimum, cold timing or production admission is claimed. A missing or
mismatched cached module is an error, never an implicit JIT.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from dev.strategy.resource_pilot import load_probe, smoke_cases, module_record, current_module, sealed, validate_pass, SCOPE
from dev.strategy.resources import compiler
from quactlize.runtime.tuning import digest
from tools.probe_kpack_strategy import device_probe, write


def oracle_identity():
    import numpy
    import torch
    import gguf.quants
    import gguf.constants
    from quactlize.runtime.compiler import sha
    return dict(numpy=numpy.__version__, torch=torch.__version__,
                gguf_quants=sha(gguf.quants.__file__), gguf_constants=sha(gguf.constants.__file__))


def checked_case(probe, case, sdk, compilers):
    record = module_record(probe, case)
    key = (case['spec']['endpoint'], case['spec']['compute'])
    if key not in compilers:
        compilers[key] = compiler(case['spec'], sdk, probe['root'] / 'cache')
    current_module(record, case['spec'], compilers[key])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--idle-confirmed', action='store_true')
    parser.add_argument('--case', help=argparse.SUPPRESS)
    parser.add_argument('--receipt', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    probe = load_probe(args.resource_run)
    plan = smoke_cases(probe)
    plan['resource_plan'] = probe['resolved']['plan_sha256']
    plan['plan_sha256'] = digest(plan)
    if args.case:
        if not args.receipt or args.receipt.resolve().parent != (args.output / 'attempts').resolve():
            parser.error('child receipt must be inside the pilot attempts directory')
        original = json.loads((args.output / 'plan.json').read_text())
        if original != json.loads(json.dumps(plan)):
            raise ValueError('child pilot plan changed')
    else:
        args.output.mkdir(parents=True, exist_ok=args.resume)
        destination = args.output / 'plan.json'
        if destination.exists():
            if json.loads(destination.read_text()) != json.loads(json.dumps(plan)):
                raise ValueError('pilot resume plan changed')
        else:
            atomic_json(destination, plan)
        print('STRATEGY_EXEC_PLAN ' + json.dumps(dict(cases=len(plan['cases']), omitted=plan['omitted'],
              compile_modules=0, scope=SCOPE, performance_admitted=False)), flush=True)
        if args.plan_only:
            return 0
    if not args.idle_confirmed:
        parser.error('confirm the selected PPU has no competing workload with --idle-confirmed')
    from dev.strategy.isolated import check_sdk_graph_abi
    from quactlize.runtime.native import sdk_identity
    if sdk_identity(args.sdk) != probe['authority']['sdk']:
        raise ValueError('SDK differs from resource probe; keep old results and run a new probe')
    physical = device_probe(args.sdk)
    if physical != probe['authority']['device']:
        raise ValueError('physical device/host/driver differs from resource probe')
    authority = dict(resource=probe['authority'], plan=plan['plan_sha256'],
        harness=harness_identity(ROOT), oracle=oracle_identity(), graph_abi=check_sdk_graph_abi(args.sdk),
        protocol=dict(rounds=3, samples=3, replays=16, weight_copies=2, cache='WARM_SMOKE'),
        idle='OPERATOR_CONFIRMED_NOT_INDEPENDENTLY_AUDITED', scope=SCOPE)
    authority = json.loads(json.dumps(authority))
    apath = args.output / 'authority.json'
    if apath.exists():
        if json.loads(apath.read_text()) != authority:
            raise ValueError('pilot source/device/oracle/protocol changed; use a new output directory')
    elif args.case:
        raise ValueError('missing parent authority')
    else:
        atomic_json(apath, authority)
    cases = {case['id']: case for case in plan['cases']}
    if args.case:
        case = cases[args.case]
        identity = dict(authority=authority, case=case)
        try:
            record = checked_case(probe, case, args.sdk, {})
            from quactlize.runtime.native import SDK
            from dev.strategy.pilot_native import measure_case
            result = validate_pass(case, measure_case(SDK(args.sdk), case, record,
                                   probe['receipts'][case['binding']['query']]))
        except Exception as exc:
            traceback.print_exc()
            result = dict(status='FAIL', error=repr(exc), timing_valid=False)
        write(args.receipt, identity, result)
        return int(result['status'] != 'PASS')
    # Validate every cached image BEFORE the first test. This does not build
    # anything and prevents discovering stale compiled code halfway through.
    compilers = {}
    for case in cases.values():
        checked_case(probe, case, args.sdk, compilers)
    lock = Path('/tmp') / ('quactlize-components-' + digest(physical['pci'])[:24] + '.lock')
    start = time.monotonic()
    attempts = args.output / 'attempts'
    attempts.mkdir(exist_ok=True)
    outcomes = {}
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for ordinal, (cid, case) in enumerate(cases.items(), 1):
            identity = dict(authority=authority, case=case)
            current = args.output / 'cases' / (cid + '.json')
            if current.exists():
                previous = sealed(current, identity)
                if previous['status'] == 'PASS':
                    validate_pass(case, previous)
                    outcomes[cid] = 'PASS'
                    print(f'STRATEGY_EXEC_CASE case={cid} status=REUSED', flush=True)
                    continue
            attempt = attempts / (cid + '.' + uuid.uuid4().hex + '.json')
            log = attempt.with_suffix('.log')
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--resource-run', str(probe['root']),
                '--sdk', str(args.sdk.resolve()), '--output', str(args.output), '--idle-confirmed',
                '--case', cid, '--receipt', str(attempt)]
            with log.open('x') as out:
                child = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, pass_fds=(held.fileno(),))
                while True:
                    try:
                        rc = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        print(f'STRATEGY_EXEC_WAIT case={ordinal}/{len(cases)} elapsed_s={time.monotonic()-start:.1f} log={log}', flush=True)
            if attempt.exists():
                result = sealed(attempt, identity)
                if result['status'] == 'PASS' and rc:
                    result = dict(status='FAIL', error='child failed after PASS receipt', process_rc=rc, timing_valid=False)
                elif result['status'] == 'PASS':
                    validate_pass(case, result)
            else:
                result = dict(status='FAIL', error='child exited without a sealed result', process_rc=rc, timing_valid=False)
            write(current, identity, result)
            outcomes[cid] = result['status']
            print(f'STRATEGY_EXEC_CASE case={ordinal}/{len(cases)} status={result["status"]} rc={rc} log={log}', flush=True)
    passed = sum(status == 'PASS' for status in outcomes.values())
    summary = dict(status='PASS' if passed == len(cases) else 'FAIL', passed=passed, expected=len(cases),
        cases=outcomes, elapsed_seconds=time.monotonic()-start, new_compiles=0,
        device=physical, scope=SCOPE, performance_admitted=False, production_changed=False,
        full_sweep_started=False, omissions=plan['excluded'])
    atomic_json(args.output / 'summary.json', summary)
    print('STRATEGY_EXEC_DONE ' + json.dumps(summary), flush=True)
    return int(summary['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())

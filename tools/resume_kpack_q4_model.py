#!/usr/bin/env python3
"""Continue validated model phases without rebuilding or replacing artifacts."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha
from tools.run_kpack_batched_bench import save, validate_plan, sequence, parse_row
from tools.run_kpack_model_validation import run
from tools.resolve_kpack_batched_models import resolve_plan
from tools.verify_kpack_dispatch import verify


def check_dispatcher_refresh(old, current, original_hash):
    allowed = {'dispatch_sha256', 'host_command', 'policy_hashes', 'dispatcher_refresh'}
    if {k for k in old.keys() | current.keys() if old.get(k) != current.get(k)} - allowed:
        raise ValueError('dispatcher repair changed device payloads or execution contracts')
    before, after = old['policy_hashes'], current['policy_hashes']
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    receipt = current.get('dispatcher_refresh', {})
    if (changed != ['quactlize/dispatch/policy.hpp'] or
            receipt != dict(base_manifest_sha256=original_hash, changed_policy_inputs=changed,
                            scope='DENSE_N_EXTENSION_PREDICTED', gpu_compilations=0)):
        raise ValueError('dispatcher repair is not the bounded N-extension update')


def repaired_source(previous):
    results = previous / 'results'
    status = json.loads((results / 'status.json').read_text())
    numerical = json.loads((results / 'numerical/status.json').read_text())
    if (status.get('status') != 'FAIL' or status.get('phase') != 'numerical' or
            any(r.get('error') != 'model numerical run has missing selected operations or a legacy fallback'
                for r in numerical if r['status'] != 'PASS') or
            not any(r['status'] == 'FAIL' for r in numerical)):
        raise ValueError('prefill retry requires the saved numerical selection failure')
    for row in numerical:
        if row['status'] == 'PASS': continue
        log = (results / 'numerical' / row['model'] / 'b2048-kpack-reference.log').read_text()
        fallbacks = re.findall(r'\[quactlize\] ([^\n]*native policy miss[^\n]*)', log)
        want = 'output.weight: native policy miss, retain legacy K-pack FQ (no same-family policy choice)'
        if not fallbacks or any(line != want for line in fallbacks):
            raise ValueError('prefill retry contains another unresolved fallback')
    receipt = json.loads((results / 'continuation.json').read_text())
    original = Path(receipt['previous']).resolve(strict=True)
    if (original.parent != previous.parent or original == previous or
            sha(original / 'results/bundle-manifest.json') != receipt['runtime_manifest_sha256']):
        raise ValueError('saved continuation points to a different original run')
    return original


def completed_benchmark_controls(results):
    root = results / 'benchmark/results'
    protocol = json.loads((root / 'protocol.json').read_text())
    plan = validate_plan(protocol['plan'])
    status = json.loads((root / 'status.json').read_text())
    expected = Counter((model['name'], arm) for model in plan['models']
                       for arm in ('reference', 'kpack', 'kpack', 'reference'))
    if Counter((r.get('model'), r.get('arm')) for r in status) != expected or any(r.get('status') != 'PASS' for r in status):
        raise ValueError('trace-only requires every ABBA benchmark arm to have passed')
    controls, paired = set(), False
    for model in plan['models']:
        for label in ('0-reference', '1-kpack', '2-kpack', '3-reference'):
            prefix = root / model['name'] / label
            process = json.loads(prefix.with_suffix('.process.json').read_text())
            rows = json.loads(prefix.with_suffix('.timings.json').read_text())
            wanted = sequence(plan, protocol['repeats'])
            if process.get('rc') != 0 or len(rows) != len(wanted):
                raise ValueError('trace-only benchmark process or sample count differs: ' + str(prefix))
            for row, point in zip(rows, wanted):
                parsed = parse_row(json.dumps({'n_kv_max': row['n_kv_max'], **row}), point, plan)
                if row['phase'] != parsed['phase'] or row['repeat'] != parsed['repeat']:
                    raise ValueError('trace-only warmup/sample identity differs')
            command = json.loads(prefix.with_suffix('.command.json').read_text())
            if command.get('devices') != model['devices']:
                raise ValueError('trace-only device differs from executed benchmark')
            if 'kpack' in label:
                controls.add(command.get('compute'))
                selected = json.loads(prefix.with_suffix('.selection.json').read_text())
                if selected.get('plan_admission') != 'PASS' or selected.get('fully_selected') is not True:
                    raise ValueError('trace-only requires complete selected compute coverage')
                paired |= bool(selected.get('paired_plans'))
    if len(controls) != 1 or not controls <= {'fp16', 'bf16'}:
        raise ValueError('trace-only benchmark compute contract differs')
    return controls.pop(), paired


def inputs(previous, llama, sdk, repair_prefill=False, performance_only=False, extend_models=False, trace_only=False):
    results = previous / 'results'
    stopped = (results / 'runner-status.txt').read_text()
    if performance_only:
        allowed_status = (r'runner_rc=0 stage=complete\b' if extend_models else
            r'runner_rc=[1-9]\d* stage=(model-benchmark|model-trace|paired-model-proof|model-acu)\b')
        if repair_prefill or not re.search(allowed_status, stopped):
            raise ValueError('model extension requires a successful complete run' if extend_models else
                'performance continuation requires completed component gates and a failed model phase')
        if extend_models:
            for name in ('benchmark/results/status.json', 'trace/status.json'):
                rows = json.loads((results / name).read_text())
                if not rows or any(row.get('status') != 'PASS' for row in rows):
                    raise ValueError('model extension requires a successful original run: ' + name)
        protocol = json.loads((results / 'benchmark/results/protocol.json').read_text())
        saved_plan = json.loads((results / 'model-plan.json').read_text())
        actual_plan = protocol.get('plan')
        if actual_plan != saved_plan:
            # --device changes only the physical ordinal of a single-GPU plan.
            # Bind it to all four executed commands instead of trusting a default
            # ordinal in the unresolved model plan.
            def without_devices(plan):
                return plan | dict(models=[{k:v for k,v in model.items() if k!='devices'}
                                           for model in plan['models']])
            if not actual_plan or without_devices(actual_plan) != without_devices(saved_plan):
                raise ValueError('saved performance model plan differs')
            for model in actual_plan['models']:
                if model.get('split') != 'none' or not re.fullmatch(r'\d+', model.get('devices','')):
                    raise ValueError('saved performance device override is not single-GPU')
                for label in ('0-reference','1-kpack','2-kpack','3-reference'):
                    receipt = json.loads((results/'benchmark/results'/model['name']/(label+'.command.json')).read_text())
                    if receipt.get('devices') != model['devices']:
                        raise ValueError('saved performance device differs from executed command')
        if (protocol.get('first_pass_excluded') is not True or protocol.get('order') != 'abba' or
                protocol.get('repeats') != 2 or protocol.get('production_fusion') not in
                ('MOE_CHAIN_AND_GPU_GATE_UP_PAIR', 'MOE_CHAIN')):
            raise ValueError('saved performance protocol differs')
    else:
        if not re.search(r'runner_rc=[1-9]\d* stage=model-numerical\b', stopped):
            raise ValueError('this continuation requires a run stopped at model-numerical')
        status = json.loads((results / 'numerical/status.json').read_text())
        failures = [r for r in status if r['status'] != 'PASS']
        if not failures or any(r.get('error') != 'explicit BF16 request fell back to FP16 compute' for r in failures):
            raise ValueError('original failure is not the repaired compute-scope check')
        if any((results / name).exists() for name in ('benchmark', 'trace', 'acu')):
            raise ValueError('later-stage results already exist; preserve them for a separate continuation')
    pin = json.loads((ROOT / 'tools/kpack_q4_model_artifact.json').read_text())
    bundle = previous.parent / ('quactlize-model-artifact-' + pin['commit'][:10]) / pin['path']
    if sha(bundle / 'manifest.json') != pin['manifest_sha256']:
        raise ValueError('selected runtime package differs from the pin')
    current = verify(bundle, sdk=sdk)
    if repair_prefill:
        check_dispatcher_refresh(json.loads((results / 'bundle-manifest.json').read_text()),
            current, sha(results / 'bundle-manifest.json'))
    elif sha(bundle / 'manifest.json') != sha(results / 'bundle-manifest.json'):
        raise ValueError('original and selected runtime packages differ')
    ci = json.loads((results / 'caller-ci-build.json').read_text())
    required = {'bin/' + name for name in ('llama-server', 'llama-batched-bench', 'llama-perplexity',
        'libggml-cuda.so', 'libncp_fa.so', 'libncp_moe.so')}
    if ci.get('entry') != '.aoneci/scripts/build.sh' or not required <= ci.get('files', {}).keys():
        raise ValueError('original caller build receipt is incomplete')
    build = Path(ci['build']).resolve(strict=True)
    for name, expected in ci['files'].items():
        file = (build / name).resolve(strict=True)
        if not file.is_relative_to(build) or sha(file) != expected:
            raise ValueError('original caller build changed: ' + name)
    checker = (llama / 'tests/quactlize_native.py').read_text()
    if 'compute scope mismatch:' not in checker or 'or int(p["rows"]) <= 8' in checker:
        raise ValueError('pull the grouped-only BF16 checker in the supplied llama checkout')
    compute, paired = completed_benchmark_controls(results) if trace_only else ('bf16', None)
    gates = ('production-q8.json', 'mixed-chain/summary.json')
    if compute == 'bf16':
        gates += ('bf16-metadata/summary.json', 'bf16/summary.json',
                  'bf16-selected-q4/summary.json', 'bf16-matched-prefill/summary.json')
    if performance_only:
        if protocol['binary_sha256'] != ci['files']['bin/llama-batched-bench']:
            raise ValueError('saved benchmark binary differs from caller receipt')
        if paired is True or (paired is None and protocol['production_fusion']=='MOE_CHAIN_AND_GPU_GATE_UP_PAIR'):
            if 'paired_gate_up' not in current:
                raise ValueError('paired runtime is absent')
            gates += ('paired-integration/summary.json',)
        if pin.get('model_reader_gate'):
            readers=json.loads((results/'model-readers/summary.json').read_text())
            expected={'q4-paired-routed','q5-routed-down','q8-paired-shared','q8-shared-down',
                      'q8-ssm-out','q8-qkv','q8-attn-gate'}
            rows=readers.get('records',[])
            if (len(rows)!=7 or {r.get('point') for r in rows}!=expected or any(
                    r.get('status')!='PASS' or r.get('rc')!=0 or r.get('admitted_m1_bitdiff')!=0 or
                    r.get('manifest_sha256')!=pin['manifest_sha256'] or
                    r.get('execution_sha256')!=current['execution_sha256'] for r in rows)):
                raise ValueError('saved model reader integration is incomplete or differs')
            gates += ('model-readers/summary.json',)
        if pin.get('prepare_integration_gate'):
            prepare = json.loads((results / 'prepare-integration/summary.json').read_text())
            if (prepare.get('passed') != 16 or prepare.get('expected') != 16 or
                    prepare.get('router_edge_cases') != 56):
                raise ValueError('saved prepare integration is incomplete')
            gates += ('prepare-integration/summary.json',)
    for name in gates:
        if json.loads((results / name).read_text()).get('status') != 'PASS':
            raise ValueError('earlier component gate did not pass: ' + name)
    return bundle, build, {name: sha(results / name) for name in gates}


def extension_plan(previous, source, device):
    """Change only the models, keeping the prior single-request workload."""
    before = json.loads((previous / 'results/model-plan.json').read_text())
    plan = validate_plan(json.loads(source.read_text()))
    for key in ('prompts', 'generations', 'parallel', 'batch', 'ubatch'):
        if plan.get(key) != before.get(key):
            raise ValueError('model extension changes workload: ' + key)
    old_names = {model['name'] for model in before['models']}
    if not plan['models'] or any(model['name'] in old_names for model in plan['models']):
        raise ValueError('model extension must select new models only')
    for model in plan['models']:
        if model['split'] != 'none' or 'tensor_split' in model:
            raise ValueError('model extension requires single-device K-pack; tensor parallel is not admitted')
        model['devices'] = device
    resolved = resolve_plan(plan)
    old_paths = {Path(model['path']).resolve() for model in before['models'] if model.get('path')}
    if any(Path(model['path']).resolve() in old_paths for model in resolved['models']):
        raise ValueError('model extension aliases an already tested model')
    return resolved


def continuation(args, output):
    previous, llama, sdk = args.previous.resolve(strict=True), args.llama.resolve(strict=True), args.sdk.resolve(strict=True)
    latest = previous
    repair = getattr(args, 'repair_prefill', False)
    performance_only = getattr(args, 'performance_only', False)
    trace_only = getattr(args, 'trace_only', False)
    extend = getattr(args, 'extend_plan', None)
    if trace_only and (not performance_only or repair or extend):
        raise ValueError('trace-only requires unchanged performance-only artifacts and the original model plan')
    if extend and (not performance_only or repair):
        raise ValueError('model extension requires unchanged performance-only artifacts')
    if repair:
        previous = repaired_source(latest)
    bundle, build, gates = inputs(previous, llama, sdk, repair_prefill=repair,
                                 performance_only=performance_only, extend_models=bool(extend), trace_only=trace_only)
    if trace_only:
        protocol = json.loads((previous / 'results/benchmark/results/protocol.json').read_text())
        if any(model['devices'] != args.device or model['split'] != 'none' for model in protocol['plan']['models']):
            raise ValueError('trace-only must use the original single-device benchmark ordinal')
    plan = extension_plan(previous, extend, args.device) if extend else None
    result = output / 'results'
    if any(p.is_symlink() for p in (latest / 'results').rglob('*')):
        raise ValueError('original results contain links; no recursive copy started')
    shutil.copytree(latest / 'results', result / 'prior', ignore=shutil.ignore_patterns('*.asysrep', '*.sqlite*', '*.tgz'))
    save(result / 'continuation.json', dict(previous=str(previous), caller_build=str(build),
        runtime_manifest_sha256=sha(bundle / 'manifest.json'), prior_gates=gates,
        validator_sha256=sha(llama / 'tests/quactlize_native.py'), compile='NONE',
        numerical_source=str(latest), native_reexecution=repair,
        original_results='UNMODIFIED', model_admission='PENDING_RECHECK',
        numerical_scope='NOT_RETESTED' if performance_only else 'RECHECK_COMPLETED_CALLS',
        benchmark_scope='REUSED_UNCHANGED' if trace_only else 'REEXECUTED',
        new_models=[model['name'] for model in plan['models']] if plan else [],
        model_plan_source=str(extend.resolve()) if extend else None))
    env = dict(os.environ)
    for name in ('DG_LIBRARY_ROOT', 'GGML_NCP_FA_LIB', 'GGML_NCP_MOE_LIB',
                 'QUACTLIZE_KPACK_PREFILL_POLICY', 'QUACTLIZE_KPACK_GEMV_POLICY',
                 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION'):
        env.pop(name, None)
    base = previous.parent
    cache = Path(env.get('CACHE_DIR', str(base / 'kpack-model-cache'))).resolve(strict=True)
    jit = Path(env.get('JIT_CACHE', str(base / 'kpack-model-jit-cache'))).resolve(strict=True)
    env.update(PPU_SDK=str(sdk), CUDA_HOME=str(sdk / 'CUDA_SDK'), CUDA_VISIBLE_DEVICES=args.device,
        DG_JIT_HGCC_COMPILER=str(sdk / 'bin/hgcc'),
        LD_LIBRARY_PATH=str(build / 'bin') + ':' + env.get('LD_LIBRARY_PATH', ''),
        DG_JIT_CACHE_DIR=str(previous / 'ci/ncp-jit-cache'),
        QUACTLIZE_KPACK_COMPUTE='bf16', QUACTLIZE_KPACK_ROUTE='auto', QUACTLIZE_KPACK_PAIR_WEIGHTS='1',
        QUACTLIZE_PPU_PACK_LIBRARY=str(bundle / 'pack/libquactlize_ppu_pack.so'),
        QUACTLIZE_KPACK_EXECUTION=str(bundle), QUACTLIZE_KPACK_JIT_HELPER=str(ROOT / 'tools/kpack_jit.py'),
        QUACTLIZE_KPACK_JIT_PYTHON=sys.executable, QUACTLIZE_KPACK_JIT_CACHE=str(jit),
        QUACTLIZE_KPACK_DEEPGEMM_HELPER=str(bundle / 'kpack_deepgemm_prewarm.py'))
    paired = False
    if performance_only:
        protocol=json.loads((previous/'results/benchmark/results/protocol.json').read_text())
        paired=protocol['production_fusion']=='MOE_CHAIN_AND_GPU_GATE_UP_PAIR'
        env['QUACTLIZE_KPACK_GATE_UP']=str(int(paired))
    if trace_only:
        compute, paired = completed_benchmark_controls(previous / 'results')
        env['QUACTLIZE_KPACK_COMPUTE'] = compute
        env['QUACTLIZE_KPACK_GATE_UP'] = str(int(paired))
    asys, acu = sdk / 'asight/bin/asys', sdk / 'asight/bin/acu'
    if not all(p.is_file() and os.access(p, os.X_OK) for p in (asys, acu)):
        raise ValueError('Asys/ACU executable is missing')
    model_plan = result / 'model-plan.json' if extend else previous / 'results/model-plan.json'
    if extend:
        save(model_plan, plan)
    common = ['--llama', llama, '--build', build, '--bundle', bundle, '--plan', model_plan,
        '--cache', cache, '--jit-cache', jit, '--logits', previous / 'logits', '--corpus', args.corpus,
        '--asys', asys, '--inspector', sdk / 'bin/hgobjdump']
    validation = [sys.executable, '-u', ROOT / 'tools/run_kpack_model_validation.py']
    phases = [
        ('numerical', validation + common + ['--phase', 'numerical', '--output', result / 'numerical',
            '--reuse-from', latest / 'results/numerical'] + (['--rerun-native'] if repair else [])),
        ('benchmark', [sys.executable, '-u', ROOT / 'tools/run_kpack_batched_bench.py',
            '--binary', build / 'bin/llama-batched-bench', '--llama-dir', llama, '--bundle', bundle,
            '--jit-cache', jit, '--cache-root', cache, '--output-root', base, '--output', result / 'benchmark',
            '--plan', model_plan, '--device', args.device, '--order', 'abba', '--require-selected', '--repeats', '2']),
        ('trace', validation + common + ['--phase', 'trace', '--output', result / 'trace']),
        ('acu', [sys.executable, '-u', ROOT / 'tools/profile_kpack_model_decode.py',
            '--sdk', sdk, '--bundle', bundle, '--llama', llama, '--trace', result / 'trace',
            '--acu', acu, '--output', result / 'acu']),
    ]
    if performance_only:
        profile = phases[-1]
        phases = phases[2:3] if trace_only else phases[1:3]
        if paired and not trace_only:
            phases.append(('paired-model-proof', [sys.executable, ROOT/'tools/check_kpack_paired_model.py',
                                                  '--results', result] + (['--selected-scope'] if extend else [])))
        if getattr(args,'profile_acu',False):
            phases.append(profile)
    failed = []
    for phase, command in phases:
        save(result / 'status.json', dict(status='RUNNING', phase=phase))
        print(f'KPACK_MODEL_CONTINUE phase={phase} compile=NONE log={result / (phase + ".log")}', flush=True)
        try:
            run(command, result / (phase + '.log'), env)
            save(result / 'status.json', dict(status='PASS', phase=phase, failed_phases=failed))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            if not extend:
                raise
            failed.append(phase)
            save(result / 'status.json', dict(status='FAIL', phase=phase, failed_phases=failed, error=str(error)))
            print(f'KPACK_MODEL_CONTINUE FAIL phase={phase} remaining_phases_continue=1 error={error}', flush=True)
    if failed:
        raise ValueError('model extension failed phases: ' + ','.join(failed))
    print(f'KPACK_MODEL_CONTINUE COMPLETE Asys={result / "trace"} '
          f'ACU={result / "acu" if any(p[0]=="acu" for p in phases) else "NOT_REQUESTED"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'llama', 'sdk', 'corpus'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--repair-prefill', action='store_true', help='use the dispatcher-only N-extension package; rerun native calls')
    p.add_argument('--performance-only', action='store_true', help='reuse unchanged component gates; rerun warmed ABBA and Asys only')
    p.add_argument('--trace-only', action='store_true', help='reuse passed ABBA samples and retry Asys only; no build or benchmark')
    p.add_argument('--profile-acu', action='store_true', help='also run ACU after a performance-only continuation')
    p.add_argument('--extend-plan', type=Path, help='test only new models after a successful run; same workload, no build')
    a = p.parse_args()
    if a.trace_only:
        a.performance_only = True
        if a.extend_plan or a.repair_prefill:
            p.error('--trace-only cannot change models or the runtime')
    if a.performance_only and a.repair_prefill:
        p.error('--performance-only cannot change the runtime')
    if a.extend_plan and not a.performance_only:
        p.error('--extend-plan requires --performance-only')
    if not re.fullmatch(r'\d+', a.device):
        p.error('one physical device ordinal is required')
    previous = a.previous.resolve(strict=True)
    output = Path(tempfile.mkdtemp(prefix='kpack-q4-resume.', dir=previous.parent))
    (output / 'results').mkdir()
    print(f'KPACK_MODEL_CONTINUE run={output} previous={previous}', flush=True)
    rc = 1
    try:
        continuation(a, output)
        rc = 0
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        status_file = output / 'results/status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else dict(phase='precheck')
        save(status_file, dict(status, status='FAIL', error=str(error)))
        print('KPACK_MODEL_CONTINUE FAIL: ' + str(error), flush=True)
    finally:
        with tarfile.open(str(output) + '.results.tgz', 'w:gz') as archive:
            def include(info):
                return None if info.name.endswith(('.asysrep', '.tgz')) or '.sqlite' in Path(info.name).name else info
            archive.add(output / 'results', arcname='results', filter=include)
        print(f'results={output}.results.tgz\nrunner_rc={rc}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())

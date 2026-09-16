#!/usr/bin/env python3
"""GPU-reference logits and separate, warmed Asys proofs for the model plan."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha
from tools.run_kpack_batched_bench import inventory, save
from tools.verify_kpack_dispatch import verify


def run(argv, log, env=None):
    save(log.with_suffix('.command.json'), dict(argv=list(map(str, argv))))
    start = update = time.monotonic()
    with log.open('x') as stream:
        process = subprocess.Popen(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                if time.monotonic() - update >= 30:
                    print(f'KPACK_MODEL_WAIT log={log} elapsed_minutes={(time.monotonic()-start)/60:.1f}', flush=True)
                    update = time.monotonic()
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    save(log.with_suffix('.process.json'), dict(rc=process.returncode, wall_seconds=time.monotonic()-start,
         timing_scope='PROCESS_INCLUDING_LOAD_JIT_NOT_INFERENCE_LATENCY'))
    if process.returncode:
        raise ValueError(f'process rc={process.returncode}; log={log}')
    return log.read_text(errors='replace')


def reuse_completed(argv, log, previous):
    """Copy evidence from an identical completed call; always re-run its checks."""
    sources = [previous, previous.with_suffix('.command.json'), previous.with_suffix('.process.json')]
    if not any(p.exists() for p in sources):
        return None
    if not all(p.is_file() and not p.is_symlink() for p in sources):
        raise ValueError('incomplete saved numerical call: ' + str(previous))
    saved = json.loads(sources[1].read_text())['argv']
    expected = list(map(str, argv))
    if saved.count('-f') != 1 or expected.count('-f') != 1:
        raise ValueError('saved numerical corpus argument differs')
    old_at, new_at = saved.index('-f') + 1, expected.index('-f') + 1
    if sha(saved[old_at]) != sha(expected[new_at]):
        raise ValueError('saved numerical corpus content differs')
    normalized = saved.copy()
    normalized[old_at] = expected[new_at]
    if normalized != expected or json.loads(sources[2].read_text()).get('rc') != 0:
        raise ValueError('saved numerical command or process status differs')
    targets = [log, log.with_suffix('.command.json'), log.with_suffix('.process.json')]
    if any(p.exists() for p in targets):
        raise ValueError('reused evidence output already exists')
    for source, target in zip(sources, targets):
        shutil.copy2(source, target)
    save(log.with_suffix('.reuse.json'), dict(source=str(previous),
        sha256=sha(previous), source_command_sha256=sha(sources[1]),
        source_process_sha256=sha(sources[2]), checks='PENDING_RECHECK', timing='NOT_REMEASURED'))
    print(f'KPACK_MODEL_REUSE log={previous} process_rc=0 checks=PENDING_RECHECK', flush=True)
    return log.read_text(errors='replace')


def numerical_metrics(text, metrics, batch, context, chunks, save_logits):
    label = 'perplexity: calculating perplexity' if save_logits else 'kl_divergence: computing'
    pattern = re.escape(label) + rf' over {chunks} chunks, n_ctx={context}, batch_size={batch}, n_seq=1\b'
    if not re.search(pattern, text) or re.search(r'CUDA error:|PPU error:|failed to decode|failed reading log-probs', text):
        raise ValueError('model numerical coverage/runtime differs; inspect the evaluation log')
    values = {}
    for key, pattern in metrics.items():
        match = re.search(pattern, text)
        if match:
            values[key] = float(match[1])
    need = {'ppl'} if save_logits else {'mean_kld', 'max_kld', 'ppl_ratio', 'same_top_pct'}
    if not need <= values.keys() or not all(math.isfinite(v) for v in values.values()):
        raise ValueError('missing or nonfinite numerical metrics')
    return values


def numerical(args, model, inv, directory, helpers):
    metric_patterns, corpus_fn, logprobs, model_selection = helpers
    corpus = directory / 'corpus.txt'
    corpus.write_text(corpus_fn(args.corpus, 64))
    pattern = '^(' + '|'.join(re.escape(n) for n in inv['eligible']) + ')$'
    records = []
    for batch, context in ((1, 256), (2048, 2048)):
        base = args.logits / model['name'] / f'b{batch}-reference'
        base.parent.mkdir(parents=True, exist_ok=True)
        for phase in ('reference-save', 'reference-self', 'kpack-reference'):
            native = phase == 'kpack-reference'
            env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
            if not native:
                env = {k: v for k, v in env.items() if not k.startswith('QUACTLIZE_KPACK_')}
            argv = [args.build / 'bin/llama-perplexity', '-m', model['path'], '-ngl', 'all',
                '-sm', 'none', '--fit', 'off', '--mmap', '-c', context, '-b', batch, '-ub', batch,
                '--chunks', 2, '-f', corpus, '--log-colors', 'off', '--verbosity', 4, '--no-warmup',
                '-ot', pattern + '=CUDA0' + ('_KPACK' if native else '')]
            if native:
                argv += ['--kpack-cache', args.cache / model['name']]
            if phase == 'reference-save':
                argv += ['--save-all-logits', base]
            else:
                argv += ['--kl-divergence', '--kl-divergence-base', base]
            log = directory / f'b{batch}-{phase}.log'
            print(f'KPACK_MODEL_NUMERICAL model={model["name"]} batch={batch} phase={phase}', flush=True)
            previous = getattr(args, 'reuse_from', None)
            text = reuse_completed(argv, log, previous / model['name'] / log.name) if previous else None
            if text is None:
                if previous and phase == 'reference-save' and base.exists():
                    raise ValueError('refusing to overwrite an existing reference without a completed log: ' + str(base))
                text = run(argv, log, env)
            values = numerical_metrics(text, metric_patterns, batch, context, 2, phase == 'reference-save')
            record = dict(model=model['name'], phase=phase, token_batch=batch, request_batch=1,
                coverage=logprobs(base, context, 2), metrics=values,
                accuracy_admission='PENDING_REVIEW', timing_scope='NOT_A_PERFORMANCE_SAMPLE')
            if native:
                evidence = model_selection(args, text)
                if not evidence['fully_selected']:
                    raise ValueError('model numerical run has missing selected operations or a legacy fallback')
                record['selection'] = evidence
            elif 'CUDA0_KPACK model buffer size' in text or '[quactlize-plan]' in text:
                raise ValueError('GPU reference entered the K-pack route')
            reuse_receipt = log.with_suffix('.reuse.json')
            if reuse_receipt.exists():
                save(reuse_receipt, dict(json.loads(reuse_receipt.read_text()), checks='REEXECUTED'))
            records.append(record)
            save(directory / 'summary.json', records)
    save(directory / 'corpus-identity.json', dict(source=str(args.corpus), source_sha256=sha(args.corpus),
        corpus_sha256=sha(corpus), records=64, scope='LIKELIHOOD_NOT_GSM8K_ANSWER_ACCURACY'))
    return records


def traces(args, model, directory, inv_path):
    records = []
    for arm in ('reference', 'native'):
        output = directory / arm
        argv = [sys.executable, args.llama / 'tests/quactlize_native.py',
            '--binary', args.build / 'bin/llama-server', '--model', model['path'],
            '--cache', args.cache / model['name'], '--bundle', args.bundle,
            '--asys', args.asys, '--inspector', args.inspector, '--jit-cache', args.jit_cache,
            '--jit-helper', ROOT / 'tools/kpack_jit.py', '--jit-python', sys.executable,
            '--output', output, '--proof-only', '--proof-arm', arm,
            '--proof-prompt', 2048, '--proof-generate', 16, '--tensor-inventory', inv_path]
        if arm == 'native':
            argv += ['--proof-tokens', directory / 'reference/proof-request/input-tokens.json']
        print(f'KPACK_MODEL_TRACE model={model["name"]} arm={arm} first_request=EXCLUDED', flush=True)
        run(argv, directory / f'{arm}.log')
        record = json.loads((output / 'proof.json').read_text())
        records.append(record)
        save(directory / 'summary.json', records)
    if records[0]['input_tokens_sha256'] != records[1]['input_tokens_sha256']:
        raise ValueError('trace requests differ')
    if records[1]['missing_ops']:
        raise ValueError('trace lacks a selected compute operation; partial traces preserved')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('llama', 'build', 'bundle', 'plan', 'cache', 'jit-cache', 'output', 'logits', 'corpus', 'asys', 'inspector'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--phase', choices=('numerical', 'trace'), required=True)
    p.add_argument('--reuse-from', type=Path, help='recheck completed numerical calls; execute only absent calls')
    args = p.parse_args()
    if args.reuse_from and args.phase != 'numerical':
        p.error('--reuse-from is numerical-only')
    if args.reuse_from:
        args.reuse_from = args.reuse_from.resolve(strict=True)
        if args.output.resolve().is_relative_to(args.reuse_from):
            p.error('reuse output must be outside the original numerical directory')
    args.manifest = verify(args.bundle)
    args.jit_helper, args.jit_python = ROOT / 'tools/kpack_jit.py', Path(sys.executable)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.llama / 'tests'))
    from quactlize_numerical import METRICS, gsm8k_corpus, logprobs
    from quactlize_native import model_selection
    results = []
    for model in json.loads(args.plan.read_text())['models']:
        directory = args.output / model['name']
        directory.mkdir()
        try:
            inv = inventory(Path(model['path']))
            args.expected_ops = inv['operators']
            if not inv['eligible']:
                raise ValueError('no supported model matrices')
            save(directory / 'inventory.json', inv)
            if args.phase == 'numerical':
                numerical(args, model, inv, directory, (METRICS, gsm8k_corpus, logprobs, model_selection))
            else:
                traces(args, model, directory, directory / 'inventory.json')
            results.append(dict(model=model['name'], phase=args.phase, status='PASS'))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            results.append(dict(model=model['name'], phase=args.phase, status='FAIL', error=str(error)))
            print(f'KPACK_MODEL_VALIDATION FAIL model={model["name"]} error={error} remaining_continue=1', flush=True)
        save(args.output / 'status.json', results)
    return int(any(r['status'] != 'PASS' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())

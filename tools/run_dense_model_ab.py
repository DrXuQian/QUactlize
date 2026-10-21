#!/usr/bin/env python3
"""Two existing llama builds, three single-GPU models: warmed timings + Asys.

No build, model conversion tool, numerical evaluation, online config sweep or
production-policy modification. Raw traces INCLUDE warmup; analysis must exclude
the first complete PP/TG pass, not just the benchmark's 64-token initial warmup.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gguf_internal_shape_inventory import read_gguf_header

MODELS = {
    'qwen38-27b': 'Qwen3.8-27B-Q4_K_M-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf',
    'qwen35-35b-a3b': 'Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf',
    'qwen3-32b': 'Qwen3-32B-Q4_K_M-GGUF/Qwen3-32B-Q4_K_M.gguf',
}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def environment(args, build, arm):
    # Never let old PoC flags, -ot environment defaults, or a library override
    # leak into this comparison. SDK/system paths remain available after bin.
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('LLAMA_ARG_', 'LLAMA_NUMERICAL_', 'QUACTLIZE_', 'GGML_QUACTLIZE_'))}
    for key in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION',
                'GGML_NCP_FA_LIB', 'GGML_NCP_MOE_LIB', 'GGML_NCP_GDN_LIB', 'DG_LIBRARY_ROOT'):
        env.pop(key, None)
    paths = [build / 'bin', args.sdk / 'lib', args.sdk / 'targets/x86_64-linux/lib',
             args.sdk / 'CUDA_SDK/targets/x86_64-linux/lib']
    env.update(PPU_SDK=str(args.sdk), PPU_PATH=str(args.sdk),
               CUDA_HOME=str(args.sdk / 'CUDA_SDK'), CUDA_PATH=str(args.sdk / 'CUDA_SDK'),
               CUDA_TOOLKIT_ROOT=str(args.sdk / 'CUDA_SDK'),
               CUDA_VISIBLE_DEVICES=args.device, LC_ALL='C',
               GGML_QUACTLIZE_DENSE='1' if arm == 'quactlize' else '0',
               LD_LIBRARY_PATH=':'.join(map(str, paths)) + ':' + env.get('LD_LIBRARY_PATH', ''),
               PATH=str(args.sdk / 'bin') + ':' + str(args.sdk / 'CUDA_SDK/bin') + ':' + env.get('PATH', ''))
    if arm == 'quactlize':
        env['QUACTLIZE_PPU_PACK_LIBRARY'] = str(build / 'bin/libquactlize_ppu.so')
        # Reuse the installed package's own helper and the normal JIT cache.
        # Never point this binary at the source checkout's newer helper.
        if args.jit_cache:
            env['QUACTLIZE_PPU_CACHE_DIR'] = str(args.jit_cache)
    return env


def command(args, build, model, phase):
    repeats = args.repeats if phase == 'perf' else args.trace_repeats
    return [str(build / 'bin/llama-batched-bench'), '-m', str(model),
            '-npp', ','.join([str(args.pp)] * (1 + repeats)), '-ntg', str(args.tg), '-npl', '1',
            '-b', str(args.pp), '-ub', str(args.pp), '-c', str(args.pp + args.tg),
            '-ngl', 'all', '-sm', 'none', '-fa', '1', '--fit', 'off', '--mmap',
            '-t', str(args.threads), '-tb', str(args.threads_batch),
            '--output-format', 'jsonl', '--log-colors', 'off', '--verbosity', '4']


def parse_rows(text, pp, tg, repeats):
    rows = []
    for line in text.splitlines():
        start = line.find('{"n_kv_max"')
        if start < 0:
            continue
        row = json.loads(line[start:])
        for key, value in dict(pp=pp, tg=tg, pl=1, n_batch=pp, n_ubatch=pp,
                               flash_attn=1, is_pp_shared=0, n_kv=pp+tg).items():
            if row.get(key) != value:
                raise ValueError(f'benchmark {key}={row.get(key)!r}, wanted {value}')
        if row.get('n_kv_max', 0) < pp + tg:
            raise ValueError('context too short')
        for key, tokens in [('pp', pp), ('tg', tg)]:
            seconds, speed = row.get('t_' + key), row.get('speed_' + key)
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in (seconds, speed)):
                raise ValueError('invalid benchmark timer')
            if not math.isclose(seconds * speed, tokens, rel_tol=2e-4, abs_tol=1e-3):
                raise ValueError('timer/token denominator differs')
        row.update(pass_index=len(rows), excluded=not rows,
                   pp_total_ms=row['t_pp'] * 1000, tpot_ms=row['t_tg'] * 1000 / tg)
        rows.append(row)
    if len(rows) != repeats + 1:
        raise ValueError(f'benchmark rows={len(rows)}/{repeats+1}; see run.log')
    kept = rows[1:]
    return dict(rows=rows, measured_passes=len(kept), excluded_complete_passes=1,
                pp_total_ms=statistics.median(r['pp_total_ms'] for r in kept),
                tpot_ms=statistics.median(r['tpot_ms'] for r in kept))


def check_routes(text, arm, pp):
    plans = [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in text.splitlines()
             if '[quactlize-dense]' in line]
    if arm == 'reference':
        if plans or '[quactlize-plan]' in text:
            raise ValueError('reference unexpectedly uses Quactlize')
    elif not plans or not {'1', str(pp)} <= {p.get('m') for p in plans}:
        raise ValueError('missing Quactlize dense plans for BOTH decode M1 and prefill; see run.log')
    return dict(plans=plans, scope='DENSE_ONLY_MOE_REMAINS_NATIVE',
                kernel_execution='REQUIRES_TRACE_MATCHING', no_regression_admitted=False)


def short_command(argv, env, output):
    try:
        result = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=15, check=False)
        output.write_bytes(result.stdout)
        return result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        output.write_text(str(exc) + '\n')
        return -1


def collect_maps(pid):
    # No binary bytes, process environment, or credentials in this receipt.
    try:
        return {line.split(maxsplit=5)[5] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
                if len(line.split(maxsplit=5)) == 6 and '.so' in line.split(maxsplit=5)[5]}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return set()


def execute(argv, env, directory, label):
    save(directory / 'command.json', dict(argv=argv, environment={k: v for k, v in env.items()
         if k in ('CUDA_VISIBLE_DEVICES', 'PPU_SDK', 'PPU_PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH',
                  'GGML_QUACTLIZE_DENSE', 'QUACTLIZE_PPU_PACK_LIBRARY', 'QUACTLIZE_PPU_CACHE_DIR')}))
    start = update = time.monotonic()
    maps = set()
    print(f'DENSE_MODEL_AB_START {label} log={directory / "run.log"}', flush=True)
    with (directory / 'run.log').open('xb') as log:
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while process.poll() is None:
                maps.update(collect_maps(process.pid))
                if time.monotonic() - update >= 30:
                    print(f'DENSE_MODEL_AB_WAIT {label} minutes={(time.monotonic()-start)/60:.1f} '
                          f'log={directory / "run.log"}', flush=True)
                    update = time.monotonic()
                time.sleep(1)
        finally:
            if process.poll() is None:
                # Only our newly created process group, never host Asys services.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            save(directory / 'process.json', dict(rc=process.returncode,
                 seconds=time.monotonic()-start, loaded_images=sorted(maps)))
    if process.returncode:
        raise ValueError(f'process rc={process.returncode}; see {directory / "run.log"}')


def trace_inventory(path):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        counts = {}
        for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            if 'KERNEL' not in table.upper():
                continue
            quoted = '"' + table.replace('"', '""') + '"'
            cols = {r[1].lower() for r in db.execute('PRAGMA table_info(' + quoted + ')')}
            if {'start', 'end'} <= cols:
                counts[table] = db.execute('SELECT count(*) FROM ' + quoted).fetchone()[0]
    if not sum(counts.values()):
        raise ValueError('Asys contains no GPU kernel activities; NOT a valid timing trace')
    return dict(kernel_tables=counts, includes_initial_warmup=True,
                exclude_first_complete_pp_tg_pass=True, steady_state_analysis='PENDING',
                no_regression_admitted=False)


def snapshot(args, arm, build, output):
    output.mkdir()
    env = environment(args, build, arm)
    files = [build / 'bin/llama-batched-bench']
    files += sorted((build / 'bin').glob('lib*.so*'))
    receipts, seen = {}, set()
    for path in files:
        real = path.resolve(strict=True)
        if not real.is_file() or real in seen:
            continue
        seen.add(real)
        receipts[str(path)] = dict(realpath=str(real), bytes=real.stat().st_size, sha256=digest(real))
    save(output / 'libraries.json', receipts)
    short_command(['ldd', str(build / 'bin/llama-batched-bench')], env, output / 'ldd.log')
    short_command([str(build / 'bin/llama-batched-bench'), '--version'], env, output / 'version.log')
    help_rc = short_command([str(build / 'bin/llama-batched-bench'), '--help'], env, output / 'help.log')
    help_text = (output / 'help.log').read_text(errors='replace')
    if help_rc or any(flag not in help_text for flag in ('--output-format', '-npp', '-ntg', '--fit')):
        raise ValueError(f'{arm}: batched-bench lacks required flags; see {output / "help.log"}')
    for source in [build / 'CMakeCache.txt', build / 'bin/quactlize-runtime/final-selection.json',
                   build / 'bin/quactlize-runtime/prefill-confirmed.json']:
        if source.is_file():
            shutil.copyfile(source, output / source.name)
    if arm == 'quactlize' and not (build / 'bin/libquactlize_ppu.so').is_file():
        raise ValueError('missing unified library in candidate bin directory')
    return receipts


def check_unchanged(receipts):
    for entry in receipts.values():
        path = Path(entry['realpath'])
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('binary/library changed during comparison: ' + str(path))


def run_arm(args, model, arm, phase, build, directory):
    directory.mkdir(parents=True)
    env = environment(args, build, arm)
    argv = command(args, build, model, phase)
    if phase == 'trace':
        argv = [str(args.asys), 'profile', '-t', 'hggc,hgtx', '-f', 'true',
                '-o', str(directory / 'trace'), *argv]
    smi = next((p for p in (args.sdk / 'ppu-smi/bin/ppu-smi', args.sdk / 'ppu-smi/ppu-smi',
                           args.sdk / 'bin/ppu-smi') if p.is_file()), None)
    if smi:
        short_command([str(smi)], env, directory / 'device-before.log')
    execute(argv, env, directory, f'model={model.parent.name} arm={arm} phase={phase}')
    text = (directory / 'run.log').read_text(errors='replace')
    repeats = args.repeats if phase == 'perf' else args.trace_repeats
    timings = parse_rows(text, args.pp, args.tg, repeats)
    timings['profiler_attached'] = phase == 'trace'
    timings['scope'] = 'UNPROFILED_MODEL_LATENCY' if phase == 'perf' else 'PROFILED_DIAGNOSTIC_NOT_MODEL_PERF'
    save(directory / 'timings.json', timings)
    save(directory / 'selection.json', check_routes(text, arm, args.pp))
    if phase == 'perf':
        maps = json.loads((directory / 'process.json').read_text())['loaded_images']
        for stem in ('libggml-cuda.so', 'libllama.so'):
            expected = {p.resolve() for p in (build / 'bin').glob(stem + '*') if p.is_file()}
            actual = {Path(p).resolve() for p in maps if Path(p).name.startswith(stem)}
            if not actual or not actual <= expected:
                raise ValueError(f'{stem}: loaded image not from this build, or missing maps evidence: {actual}')
        if arm == 'quactlize' and (build / 'bin/libquactlize_ppu.so').resolve() not in {Path(p).resolve() for p in maps}:
            raise ValueError('public Quactlize library not found in loaded images')
    else:
        report, database = directory / 'trace.asysrep', directory / 'trace.sqlite'
        if not report.is_file():
            raise ValueError('Asys did not produce trace.asysrep')
        with (directory / 'export.log').open('xb') as log:
            subprocess.run([str(args.asys), 'export', '-o', str(database), str(report)], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        save(directory / 'trace-scope.json', trace_inventory(database))
    if smi:
        short_command([str(smi)], env, directory / 'device-after.log')
    print(f'DENSE_MODEL_AB_RESULT arm={arm} phase={phase} '
          f'pp_ms={timings["pp_total_ms"]:.3f} tpot_ms={timings["tpot_ms"]:.3f}', flush=True)
    return timings


def pack_review(run):
    # No weights, binary, JIT cache, core dump or duplicate Asys blob. SQLite
    # contains the raw GPU/API timeline; .asysrep stays on box for the GUI.
    archive = run.with_name(run.name + '.review.tgz')
    with tarfile.open(archive, 'x:gz', compresslevel=5) as tar:
        for path in sorted(run.rglob('*')):
            if path.is_file() and not path.is_symlink() and (
                    path.suffix in ('.json', '.log', '.txt', '.sqlite', '.csv')):
                tar.add(path, arcname=str(path.relative_to(run)), recursive=False)
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-dir', type=Path, required=True)
    p.add_argument('--reference-build', type=Path)
    p.add_argument('--quactlize-build', type=Path)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    p.add_argument('--asys', type=Path)
    p.add_argument('--model-root', type=Path, default=Path('/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo'))
    p.add_argument('--model', action='append', choices=list(MODELS))
    p.add_argument('--device', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    p.add_argument('--idle-confirmed', action='store_true', required=True,
                   help='the selected physical GPU must remain idle except for this serial comparison')
    p.add_argument('--pp', type=int, default=2048)
    p.add_argument('--tg', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--trace-repeats', type=int, default=2)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--threads-batch', type=int, default=32)
    p.add_argument('--phases', choices=('perf,trace', 'perf', 'trace'), default='perf,trace', metavar='perf,trace')
    p.add_argument('--jit-cache', type=Path, help='optional existing absolute cache; default library cache unchanged')
    p.add_argument('--output-root', type=Path, default=Path('/workspace'))
    a = p.parse_args()
    if not a.device.isdigit() or min(a.pp, a.tg, a.repeats, a.trace_repeats, a.threads, a.threads_batch) < 1:
        p.error('one GPU ordinal and positive workload sizes/repeats are required')
    if a.jit_cache and not a.jit_cache.is_absolute():
        p.error('--jit-cache must be absolute')
    a.llama_dir, a.sdk = a.llama_dir.resolve(strict=True), a.sdk.resolve(strict=True)
    builds = {arm: (value or a.llama_dir / default).resolve(strict=True) for arm, value, default in (
        ('reference', a.reference_build, 'build-ci-ppu-dev'),
        ('quactlize', a.quactlize_build, 'build-ci-quactlize-dense-gemm'))}
    if builds['reference'] == builds['quactlize']:
        p.error('two distinct existing build directories are required')
    a.asys = (a.asys or a.sdk / 'asight/bin/asys').resolve()
    if 'trace' in a.phases and not os.access(a.asys, os.X_OK):
        p.error('missing Asys executable: ' + str(a.asys))
    run = Path(tempfile.mkdtemp(prefix='dense-model-ab.', dir=a.output_root.resolve(strict=True)))
    summary = dict(status='INCOMPLETE', run=str(run), device=a.device, pp=a.pp, tg=a.tg,
                   request_batch=1, token_batch=a.pp, warmup='EXCLUDE_FIRST_COMPLETE_PASS_EACH_PROCESS',
                   input_protocol='BATCHED_BENCH_PSEUDORANDOM_TOKENS_NOT_GENERATED_TEXT_OR_ACCURACY_TEST',
                   builds={k: str(v) for k, v in builds.items()}, results={}, failures=[],
                   no_regression_admitted=False, correctness_admitted=False,
                   scope='DENSE_INTEGRATION_AB_NOT_CONFIG_SWEEP_MOE_REMAINS_NATIVE')
    print(f'DENSE_MODEL_AB_RUN run={run} device={a.device} pp={a.pp} tg={a.tg} '
          'request_batch=1 phases=' + a.phases, flush=True)
    try:
        save(run / 'protocol.json', summary)
        save(run / 'sdk.json', dict(path=str(a.sdk), receipts={str(f.relative_to(a.sdk)): f.read_text(errors='replace')
             for f in (a.sdk / 'VERSION.txt', a.sdk / 'release.yaml') if f.is_file()}))
        models = {name: (a.model_root / MODELS[name]).resolve(strict=True) for name in (a.model or MODELS)}
        for name, model in models.items():
            with model.open('rb') as stream:
                header = read_gguf_header(stream, str(model))
            if header['metadata'].get('split.count', 1) != 1:
                raise ValueError('this exact-file protocol expects an unsplit model: ' + str(model))
            save(run / (name + '.inventory.json'), dict(path=str(model), file_bytes=model.stat().st_size,
                 mtime_ns=model.stat().st_mtime_ns, header=header, weights_hashed=False))
        receipts = {arm: snapshot(a, arm, build, run / ('build-' + arm)) for arm, build in builds.items()}
        for name, model in models.items():
            summary['results'][name] = {}
            # Same physical GPU, serial arms. Reverse trace order, but do NOT
            # mix profiler timings with the unprofiled model comparison.
            for phase in a.phases.split(','):
                for arm in (('reference', 'quactlize') if phase == 'perf' else ('quactlize', 'reference')):
                    directory = run / name / phase / arm
                    try:
                        summary['results'][name][phase + '-' + arm] = run_arm(a, model, arm, phase, builds[arm], directory)
                    except Exception as exc:
                        error = dict(model=name, phase=phase, arm=arm, error=str(exc), log=str(directory / 'run.log'))
                        summary['failures'].append(error)
                        save(directory / 'failure.json', error)
                        print('DENSE_MODEL_AB_FAIL ' + json.dumps(error), flush=True)
                save(run / 'summary.json', summary)
            values = summary['results'][name]
            if 'perf-reference' in values and 'perf-quactlize' in values:
                values['model_delta_pct'] = {key: (values['perf-quactlize'][key] / values['perf-reference'][key] - 1) * 100
                                             for key in ('pp_total_ms', 'tpot_ms')}
                print('DENSE_MODEL_AB_MODEL ' + json.dumps(dict(model=name,
                    reference={k: values['perf-reference'][k] for k in ('pp_total_ms', 'tpot_ms')},
                    quactlize={k: values['perf-quactlize'][k] for k in ('pp_total_ms', 'tpot_ms')},
                    delta_pct=values['model_delta_pct'], no_regression_admitted=False)), flush=True)
        for receipt in receipts.values():
            check_unchanged(receipt)
        summary['status'] = 'COMPLETE' if not summary['failures'] else 'INCOMPLETE'
    except Exception as exc:
        summary['failures'].append(dict(error=str(exc)))
        print('DENSE_MODEL_AB_FAIL ' + str(exc), flush=True)
    finally:
        save(run / 'summary.json', summary)
        archive = pack_review(run)
        print(f'DENSE_MODEL_AB_DONE status={summary["status"]}\nresults={archive}\n'
              f'Asys={run}/*/trace/{{reference,quactlize}}/trace.asysrep', flush=True)
    return int(summary['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())

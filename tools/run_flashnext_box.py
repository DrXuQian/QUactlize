#!/usr/bin/env python3
"""Fresh-worktree PPU acceptance: build, compat, serving, and real-geometry graphs.

Run from any Quactlize checkout, including via `git show REF:tools/... | python -`.
Uses the existing build.sh; never replaces an installed llama/Flare library.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def checkout(repo, source, revision, log, env):
    """Only detached new worktrees; never reset the caller or update a pin."""
    with log.open('x') as output:
        def run(command):
            subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(source), revision])
        pin = git(source, 'rev-parse', 'HEAD:third_party/actlize')
        local = repo / 'third_party/actlize'
        available = (local / '.git').exists() and subprocess.run(
            ['git', '-C', str(local), 'cat-file', '-e', pin + '^{commit}'],
            stdout=output, stderr=subprocess.STDOUT).returncode == 0
        if available:
            run(['git', '-C', str(local), 'worktree', 'add', '--detach',
                 str(source / 'third_party/actlize'), pin])
        else:
            # Command-local SSH rewrite, not HTTPS nested checkout or global config.
            run(['git', '-C', str(source), '-c',
                 'url.git@github.com:DrXuQian/.insteadOf=https://github.com/DrXuQian/',
                 'submodule', 'update', '--init', 'third_party/actlize'])
        if git(source / 'third_party/actlize', 'rev-parse', 'HEAD') != pin:
            raise ValueError('actlize checkout differs from the requested gitlink')
        return pin


def commands(source, package, results, sdk, jobs):
    from ci.suites import SERVING
    python = sys.executable
    yield 'precheck', [python, '-c',
        'import sys,torch,pytest,numpy,gguf; '
        'assert torch.cuda.is_available(), "Use existing PPU GPU Torch, not CPU Torch"; '
        'assert hasattr(torch.library,"custom_op"), "Torch custom_op is required"; '
        'print("python:",sys.executable,"torch:",torch.__version__,"device:",torch.cuda.get_device_name(0))']
    yield 'build', ['bash', str(source / 'build.sh'), '--library', '--sdk', str(sdk),
                    '--output', str(package), '--jobs', str(jobs)]
    yield 'compat', [python, '-m', 'pytest', '-q', '-s', '-x', '--tb=short',
        '-o', 'addopts=', '-o', 'xfail_strict=true', '-p', 'no:cacheprovider',
        '--junitxml=' + str(results / 'compat.xml'), 'tests/ppu/test_compat_library.py']
    # One child process group per phase, so cancellation also reaps the test.
    yield 'serving', [python, '-m', 'pytest', '-q', '-s', '-x', '--tb=short',
        '-o', 'addopts=', '-o', 'xfail_strict=true', '-p', 'no:cacheprovider',
        '--junitxml=' + str(results / 'serving.xml'),
        *[str(source / 'tests' / name) for name in SERVING]]
    for m in (1, 4, 16):
        yield f'graph-m{m}', [python, str(source / 'examples/serving_graph.py'),
            '--library', str(package / 'libncp_quactlize.so'), '--m', str(m),
            '--experts', '512', '--topk', '10', '--k', '2560', '--hidden', '640', '--iterations', '20']


def check_counts(counts):
    if not counts['tests'] or any(counts[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError(f'incomplete device coverage: {counts}')


def package_results(run, build_failed):
    results = run / 'results'
    build = run / 'package.build'
    if build_failed and build.is_dir():
        # Compiler diagnostics are separate from the build console. Include
        # only text logs, not objects, modules, libraries or model data.
        for path in build.rglob('*.log'):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(build.resolve()):
                continue
            target = results / 'compiler' / path.relative_to(build)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    archive = run.with_name(run.name + '.review.tgz')
    with tarfile.open(archive, 'x:gz') as tar:
        tar.add(results, arcname='results')
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default='HEAD', help='committed revision; FETCH_HEAD for a fetched script')
    parser.add_argument('--device', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                        help='one idle physical PPU ordinal/UUID; default visible device or 0')
    parser.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK') or '/usr/local/PPU_SDK'))
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=7200, help='seconds per build/test/graph phase')
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.timeout < 1 or not args.device.strip() or ',' in args.device:
        parser.error('positive jobs/timeout and one device are required')
    repo = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel'))
    revision = git(repo, 'rev-parse', '--verify', args.ref + '^{commit}')
    run = Path(tempfile.mkdtemp(prefix='qz-flashnext-', dir='/workspace' if Path('/workspace').is_dir() else None))
    source, package, results = run / 'source', run / 'package', run / 'results'
    results.mkdir()
    sdk = args.sdk.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, PPU_SDK=str(sdk), PYTHON=sys.executable,
        PYTHONPATH=str(source), PYTHONUNBUFFERED='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        QUACTLIZE_CI_PACKAGE=str(package), QUACTLIZE_SERVING_LIBRARY=str(package / 'libncp_quactlize.so'),
        QUACTLIZE_PPU_CACHE_DIR=str(run / 'jit-cache'), GIT_TERMINAL_PROMPT='0')
    env.setdefault('GIT_SSH_COMMAND', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new')
    for key in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONOPTIMIZE'):
        env.pop(key, None)
    summary = dict(status='INCOMPLETE', commit=revision, device=args.device, sdk=str(sdk), phases=[],
                   scope='FLASHNEXT_LIBRARY_ACCEPTANCE_NOT_MODEL_OR_PERFORMANCE_ADMISSION')
    def save():
        (results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    stage = 'checkout'; rc = 1
    print(f'FLASHNEXT_START commit={revision} device={args.device} run={run}', flush=True)
    try:
        summary['actlize'] = checkout(repo, source, revision, results / 'checkout.log', env)
        # Load helpers from the detached revision, not the caller's branch.
        sys.path.insert(0, str(source))
        from ci.run import run_command, pytest_counts
        for stage, command in commands(source, package, results, sdk, args.jobs):
            row = dict(stage=stage, status='FAIL', command=command)
            summary['phases'].append(row); save()
            log = results / (stage + '.log')
            print(f'FLASHNEXT_STAGE stage={stage} log={log}', flush=True)
            started = time.monotonic()
            row['returncode'] = run_command(command, log, env, args.timeout)
            row['seconds'] = time.monotonic() - started
            if row['returncode']:
                raise RuntimeError(f'{stage} rc={row["returncode"]}; see {log}')
            if stage == 'build':
                shutil.copy2(package / 'manifest.json', results / 'package-manifest.json')
                with (package / 'libncp_quactlize.so').open('rb') as lib:
                    digest = hashlib.sha256()
                    for block in iter(lambda: lib.read(1 << 20), b''): digest.update(block)
                summary['library_sha256'] = digest.hexdigest()
            if stage in ('compat', 'serving'):
                xml = results / (stage + '.xml')
                row['counts'] = pytest_counts(xml); check_counts(row['counts'])
            row['status'] = 'PASS'; save()
            print(f'FLASHNEXT_PASS stage={stage}', flush=True)
        summary['status'] = 'PASS'; rc = 0
    except (Exception, KeyboardInterrupt) as exc:
        detail = str(exc) or type(exc).__name__
        summary.update(status='FAIL', failed_stage=stage, error=detail)
        (results / 'error.log').write_text(traceback.format_exc())
        print(f'FLASHNEXT_FAIL stage={stage}: {detail}', flush=True)
        log = results / (stage + '.log')
        if log.is_file():
            print('\n'.join(log.read_text(errors='replace').splitlines()[-60:]), flush=True)
    finally:
        save()
        archive = package_results(run, stage == 'build' and rc != 0)
        print(f'FLASHNEXT_DONE status={summary["status"]} runner_rc={rc} results={archive}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())

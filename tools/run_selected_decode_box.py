#!/usr/bin/env python3
"""Reuse the last successful TP2 environment with the new selected runtime.

No checkout switch, cleanup, compilation sweep, or selection search is done
here. The existing box runner performs incremental caller CI, numerical gates,
matched model ABBA and Asys. All paths can be overridden explicitly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compatibility(previous, explicit=None):
    expected = digest(previous/'results/compatibility-bundle-manifest.json')
    if explicit:
        path = Path(explicit).resolve(strict=True)
        if digest(path/'manifest.json') != expected:
            raise ValueError('explicit compatibility bundle differs from previous successful run')
        return path
    # Search only the prior experiment root, do not follow directory links or
    # descend into models, build trees, source worktrees or cached results.
    parent = previous.parent
    matches = []
    excluded = {'build', 'ci', 'results', 'cache', '.git', 'modules', 'third_party',
                'kpack-tp2-model-cache', 'kpack-model-jit-cache', 'kpack-tp2-deepgemm-jit'}
    for directory, dirs, files in os.walk(parent, followlinks=False):
        path = Path(directory)
        depth = len(path.relative_to(parent).parts)
        dirs[:] = sorted(d for d in dirs if d not in excluded and not d.startswith('.')
                         and not (path/d).is_symlink()) if depth < 6 else []
        if 'manifest.json' not in files:
            continue
        manifest = path/'manifest.json'
        if not manifest.is_symlink() and manifest.stat().st_size < 4*1024*1024 and digest(manifest) == expected:
            matches.append(path)
    if not matches:
        raise ValueError('old compatibility bundle not found below '+str(parent)+
                         '; set QUACTLIZE_PPU_BUNDLE to the previous six-library overlay')
    return min(matches, key=lambda p: (len(p.parts), str(p)))


def environment(previous, inherited):
    previous = previous.resolve(strict=True)
    receipt = json.loads((previous/'results/caller-ci-build.json').read_text())
    legacy = json.loads((previous/'results/compatibility-bundle-manifest.json').read_text())
    defaults = dict(LLAMA_CI_DIR=receipt['llama_worktree']['directory'],
                    LLAMA_CI_BUILD_DIR=receipt['build'], NCP_CI_DIR=receipt['ncp_directory'],
                    NCP_LIB_DIR='/sim/eec/shared/junfu.qx/ncp_flash_lib', RESULT_ROOT=str(previous.parent),
                    CUDA_VISIBLE_DEVICES='0,1', JOBS='192', TP2_MODE='model')
    if legacy.get('sdk'):
        defaults['PPU_SDK'] = legacy['sdk']
    env = dict(inherited)
    for key, value in defaults.items():
        env.setdefault(key, value)
    env['QUACTLIZE_PPU_BUNDLE'] = str(compatibility(previous, env.get('QUACTLIZE_PPU_BUNDLE')))
    required = {'LLAMA_CI_DIR': '.aoneci/scripts/build.sh',
                'LLAMA_CI_BUILD_DIR': 'CMakeCache.txt', 'NCP_CI_DIR': 'build/CMakeCache.txt',
                'NCP_LIB_DIR': 'CMakeLists.txt'}
    for key, file in required.items():
        if not (Path(env[key])/file).is_file():
            raise ValueError(f'{key}: missing {env[key]}/{file}; override this path explicitly')
    if env['TP2_MODE'] != 'model':
        raise ValueError('selected integration handoff requires TP2_MODE=model')
    return env, defaults.keys() | {'QUACTLIZE_PPU_BUNDLE'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    env, keys = environment(args.previous_run, os.environ)
    pin = json.loads((ROOT/'tools/kpack_q4_model_artifact.json').read_text())
    head = subprocess.check_output(['git', '-C', env['LLAMA_CI_DIR'], 'rev-parse', 'HEAD'], text=True).strip()
    if head != pin['llama_ci_commit']:
        raise ValueError('caller checkout differs; fetch private branch '+pin['llama_branch']+
                         ' and switch to '+pin['llama_ci_commit'])
    print('SELECTED_MODEL_ENV '+json.dumps({k: env[k] for k in sorted(keys)}), flush=True)
    print('SELECTED_MODEL scope=TP2_NUMERICS_MODEL_ABBA_ASYS selection=LOCALLY_AUDITED sweep=NONE', flush=True)
    if args.plan_only:
        return 0
    return subprocess.run(['bash', str(ROOT/'tools/run_kpack_tp2_box.sh')], cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())

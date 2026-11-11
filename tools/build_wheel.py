#!/usr/bin/env python3
"""Build a precompiled PPU wheel using the existing complete public-library builder."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    parser.add_argument('--bundle', type=Path, help='reuse a verified complete bundle from this source revision')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    parser.add_argument('--work', type=Path, help='fresh build directory; default is a temporary directory')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--with-host', action='store_true', help='also compile the legacy extension against installed Torch')
    args = parser.parse_args()
    from tools.build_ppu_library import build
    from tools.wheel_support import wheel_metadata
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    with tempfile.TemporaryDirectory(prefix='quactlize-wheel-') as temporary:
        work = args.work.resolve() if args.work else Path(temporary) / 'build'
        work.mkdir(parents=True, exist_ok=False)
        bundle = args.bundle.resolve(strict=True) if args.bundle else work / 'package'
        if not args.bundle:
            build(args.sdk, bundle, args.jobs)
        wheel_metadata(bundle, ROOT)
        args.output.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, QUACTLIZE_WITH_DEVICE='1',
                           QUACTLIZE_WITH_HOST=str(int(args.with_host)), QUACTLIZE_PPU_BUNDLE=str(bundle))
        # No build isolation, package index or Torch download. Fresh build_lib prevents stale extension leakage.
        log = work / 'wheel.log'
        with log.open('w') as output:
            result = subprocess.run([sys.executable, 'setup.py', 'build', '--build-base', str(work / 'python'),
                                     'bdist_wheel', '--bdist-dir', str(work / 'wheel'),
                                     '--dist-dir', str(args.output.resolve())], cwd=ROOT, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            # Include the cause even when the default temporary work directory is cleaned up.
            raise RuntimeError('wheel build failed:\n' + '\n'.join(log.read_text(errors='replace').splitlines()[-40:]))
        print(f'QUACTLIZE_WHEEL_COMPLETE output={args.output.resolve()} target=ppu10 device_tests=NOT_RUN')


if __name__ == '__main__':
    main()

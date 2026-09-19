#!/usr/bin/env python3
"""Run Asys with private /tmp and helper PIDs, leaving host services untouched."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def private_command(command, temporary, inputs):
    executable = shutil.which('unshare')
    if not executable:
        raise ValueError('private Asys needs unshare; no host service was stopped')
    for path in [temporary, *inputs]:
        if Path(path).resolve().is_relative_to('/tmp'):
            raise ValueError('private Asys inputs/output must be outside /tmp: ' + str(path))
    if temporary.exists():
        raise ValueError('private profiler directory already exists: ' + str(temporary))
    temporary.mkdir(mode=0o700)
    return [executable, '--mount', '--pid', '--fork', '--kill-child', '--mount-proc', '--propagation', 'private',
        sys.executable, str(Path(__file__).resolve()), '--parent-mount', os.readlink('/proc/self/ns/mnt'),
        '--parent-pid', os.readlink('/proc/self/ns/pid'), '--temporary', str(temporary.resolve()),
        '--', *map(str, command)]


def enter(args):
    if (os.getpid() != 1 or os.readlink('/proc/self/ns/mnt') == args.parent_mount or
            os.readlink('/proc/self/ns/pid') == args.parent_pid):
        raise ValueError('refusing to mount over /tmp outside the private profiler namespaces')
    path = args.temporary
    if path.is_symlink() or not path.is_dir() or path.resolve().is_relative_to('/tmp') or any(path.iterdir()):
        raise ValueError('private profiler directory is not a new empty directory')
    if not args.command or args.command[0] != '--' or len(args.command) < 2:
        raise ValueError('missing trace command')
    mount = shutil.which('mount')
    if not mount:
        raise ValueError('mount is unavailable in the private profiler namespace')
    subprocess.run([mount, '--bind', str(path.resolve()), '/tmp'], check=True)
    # Keep the trace runner as PID 1. Its exit ends only this namespace's helpers.
    command = args.command[1:]
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-mount', required=True)
    parser.add_argument('--parent-pid', required=True)
    parser.add_argument('--temporary', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    enter(parser.parse_args())

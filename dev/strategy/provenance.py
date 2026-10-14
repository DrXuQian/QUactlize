"""Freeze a host rule audit, including dirty/untracked source and evidence."""
import hashlib
from pathlib import Path

from quactlize.runtime.tuning import digest


def source_identity(root):
    root = Path(root)
    files = {}
    scopes = {
        'dev/strategy': {'.py'},
        'tools': {'.py', '.sh'},
        'quactlize': {'.py', '.h', '.hpp', '.cuh', '.cpp', '.cu', '.inl', '.inc'},
        'policies': {'.json', '.hpp'},
        'docs/measurements': {'.json', '.gz'},
        'third_party/actlize/include': {'.h','.hpp','.cuh','.inl','.inc'},
        'third_party/actlize/tools/util/include': {'.h','.hpp','.cuh','.inl','.inc'},
        'third_party/actlize/examples/common': {'.h','.hpp','.cuh','.inl','.inc'},
    }
    for directory, suffixes in scopes.items():
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and path.suffix in suffixes:
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(schema='quactlize.host-rule-source.v1', files=files, sha256=digest(files),
                scope='HOST_AUDIT_SOURCE_AND_INPUTS_NOT_DEVICE_BINARY')


def harness_identity(root):
    """Bind transitive fixture/oracle helpers, including locally added files.

    Conservative source-tree closure avoids a handwritten direct-import list
    silently missing a changed nested fixture or independent GGUF oracle.
    Libraries/SDK/device identity are bound separately by the runtime receipt.
    """
    root=Path(root);files={}
    for directory in ('dev','tools','quactlize','reference'):
        for path in sorted((root/directory).rglob('*.py')):
            if path.is_file():
                files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(schema='quactlize.component-harness.v1',files=files,sha256=digest(files))


def verify_unchanged(before, root):
    after = source_identity(root)
    if before != after:
        changed = sorted(k for k in before['files'].keys() | after['files'].keys()
                         if before['files'].get(k) != after['files'].get(k))
        raise ValueError('rule audit inputs changed during execution: ' + ', '.join(changed[:10]))
    return after

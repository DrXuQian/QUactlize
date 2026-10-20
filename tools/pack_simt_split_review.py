#!/usr/bin/env python3
"""CPU-only Split-K evidence export; original results and reports stay untouched."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import time


def result_root(source):
    source = Path(source).expanduser().resolve(strict=True)
    # An old archive is a locator only: never unpack a potentially huge archive.
    if source.is_file():
        if not source.name.endswith(('.tgz', '.tar.gz', '.tar.xz')):
            raise ValueError('supply a run directory, results directory, or its review archive')
        source = source.parent
    for root in (source, source / 'results'):
        for name, schema in (('multi.json', 'simt-split-multigpu-v1'),
                             ('plan.json', 'simt-common-split-v1')):
            marker = root / name
            if marker.is_file() and json.loads(marker.read_bytes()).get('schema') == schema:
                return root
    raise ValueError('original Split-K results directory not found; keep the original run on Box')


def exclusion(relative):
    if 'access' in relative.parts:
        return 'access_footprints'
    if relative.suffix == '.acurep' or relative.name in {
            f'r{r}-a{a}' for r in range(3) for a in (0, 1)}:
        return 'full_acu_reports'
    if relative.name.endswith(('.isa.txt', '.resources.txt', '.sass.txt', '.ptx.txt')):
        return 'codegen_text'
    if relative.name == 'transfer-manifest.json':
        return 'previous_transfer_manifest'
    if relative.suffix not in ('.json', '.csv', '.log', '.txt'):
        return 'binaries_archives_and_other_files'
    return None


def package(output, target=None):
    output = Path(output).resolve(strict=True)
    if not output.is_dir():
        raise ValueError('results must be a directory')
    target = (Path(target).absolute() if target is not None else
              output / f'review-lite-{time.time_ns()}.tar.xz')
    included, omitted = [], {}
    started = time.monotonic()

    def omit(reason, size):
        entry = omitted.setdefault(reason, dict(files=0, bytes=0))
        entry['files'] += 1
        entry['bytes'] += size

    # Exclusive creation: old archives and raw results are never overwritten.
    with target.open('xb') as sink, tarfile.open(fileobj=sink, mode='w:xz', preset=6) as archive:
        for directory, dirs, files in os.walk(output, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
            for name in sorted(files):
                path = Path(directory) / name
                if path == target:
                    continue
                relative = path.relative_to(output)
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    omit('links_or_special_files', 0)
                    continue
                reason = exclusion(relative)
                if reason:
                    # Stat only: do not read/hash omitted multi-GB ACU reports.
                    omit(reason, info.st_size)
                    continue
                data = path.read_bytes()
                member = tarfile.TarInfo(relative.as_posix())
                member.size = len(data)
                member.mtime = int(info.st_mtime)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(data))
                included.append(dict(path=relative.as_posix(), bytes=len(data),
                                     sha256=hashlib.sha256(data).hexdigest()))
        manifest = dict(
            schema='simt-split-review-transfer-v1', source=str(output),
            gpu_work=False, raw_results_modified=False, verdicts_recomputed=False,
            scope='TRANSFER_ONLY_NOT_NEW_PERFORMANCE_ADMISSION',
            snapshot='PER_FILE_COPY_NOT_ATOMIC_RUN_SNAPSHOT',
            retained='Original JSON, raw timing samples, CSV counters and text diagnostics',
            omissions='Full ACU reports, access footprints and codegen text remain on Box; '
                      'references to them in original JSON are intentionally unchanged.',
            included=included, omitted=omitted)
        data = (json.dumps(manifest, sort_keys=True) + '\n').encode()
        member = tarfile.TarInfo('transfer-manifest.json')
        member.size = len(data)
        member.mode = 0o644
        archive.addfile(member, io.BytesIO(data))
    print('SPLIT_REVIEW_PACK ' + json.dumps(dict(
        files=len(included), bytes=target.stat().st_size, omitted=omitted,
        elapsed_seconds=round(time.monotonic() - started, 2), gpu_work=False)), flush=True)
    print('results=' + str(target), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='run/results directory, or old archive with original results still present')
    parser.add_argument('--output', type=Path, help='new .tar.xz filename; never overwrite')
    args = parser.parse_args()
    try:
        if args.output is not None and not args.output.name.endswith('.tar.xz'):
            raise ValueError('--output must end in .tar.xz')
        package(result_root(args.source), args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

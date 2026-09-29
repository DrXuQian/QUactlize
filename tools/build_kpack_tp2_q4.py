#!/usr/bin/env python3
"""Rebuild only the compatibility Q4 library's short-K selector, then resume TP2.

The native runtime and its JIT source checkout are not modified. The output
uses an explicit mixed-source receipt, not a six-library single-source claim.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize import ppu_bundle
from tools.build_kpack_model_ci import clone_checkout, git
from tools.verify_kquant_selected_config import ArrangementV2, ConfigV4

BASE_SOURCE = '2826cf12451e02ca4590f7a44682b57d2098bfb9'
BASE_MANIFEST = '46fc3096e1a14b712ad5d7a50de096d2a973ad5826aa3ffe6a6764d1fc12180d'
FIX_SOURCE = '025c7e4d4b92330287099675a26bf2879b814f01'
FIX_BRANCH = 'fix/q4-smallk-admission'
Q4_LIBRARY = 'libquactlize_ppu_fmt0.so'
DEFS = 'PPU_PACKED_SCALE=1 QUACTLIZE_DENSE_ONLY=12 PPU_PACKED_FORMAT=0'
SCHEMA = 'quactlize.ppu-compatibility-overlay.v1'
QUERY_PREFIX = 'Q4_SMALLK_QUERY '


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_query_output(output):
    records = [line[len(QUERY_PREFIX):] for line in output.splitlines()
               if line.startswith(QUERY_PREFIX)]
    if len(records) != 1:
        raise ValueError('expected one Q4 query receipt')
    result = json.loads(records[0])
    if result.get('status') != 'PASS' or result.get('scope') != 'HOST_QUERY_ONLY_NO_KERNEL_LAUNCH':
        raise ValueError('Q4 query did not pass')
    return result


def probe(path, mode):
    lib = C.CDLL(str(path.resolve(strict=True)), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    identity = lib.quactlize_ppu_build_packed_format_v1
    identity.argtypes = []
    identity.restype = C.c_int32
    if identity() != 0:
        raise ValueError('replacement is not FMT0/Q4_K')
    arr = ArrangementV2()
    canonical = lib.quactlize_ppu_canonical_arrangement_v2
    canonical.argtypes = [C.c_int, C.POINTER(ArrangementV2)]
    if canonical(12, C.byref(arr)) or tuple(getattr(arr, n) for n, _ in arr._fields_) != (
            2, 1, 4, 0, 0, 64, 32, 0, 0x51344B5034540001):
        raise ValueError('canonical Q4 arrangement differs')
    any_m = lib.quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2
    any_m.argtypes = [C.c_int, C.c_int, C.c_int, C.POINTER(ArrangementV2)]
    selected = lib.quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2
    selected.argtypes = [C.POINTER(ConfigV4)] + [C.c_int]*5 + [C.POINTER(ArrangementV2), C.c_char_p]
    grouped = lib.quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2
    grouped.argtypes = [C.c_int]*4 + [C.POINTER(ArrangementV2)]
    s1 = b'kpack4:8x64x256:8x16:s2:S1'
    rows = []
    for n, k, previous_ok in ((512,512,0), (256,1024,0), (512,1536,0),
                              (4096,1536,0), (8192,512,0), (512,2048,1),
                              (4096,1024,1), (8192,4096,1), (16384,512,1)):
        want = 1 if mode == 'candidate' else previous_ok
        observed = any_m(n,k,12,C.byref(arr))
        if observed != want or grouped(n,k,4,12,C.byref(arr)) != 1:
            raise ValueError(f'{mode}: any-M admission differs N={n} K={k}: {observed}, expected {want}')
        defaults = []
        for m in (1,2,3,4,5,6,7,8,9,32):
            out = ConfigV4()
            rc = selected(C.byref(out),m,n,k,32,12,C.byref(arr),None)
            if rc != (want if m <= 8 else 1):
                raise ValueError(f'{mode}: null selection differs M={m} N={n} K={k}')
            if rc:
                if mode == 'candidate' and not previous_ok and m <= 8 and (out.name != s1 or out.split_k_slices != 1):
                    raise ValueError('short-K repair did not select the existing S1 parent')
                defaults.append(dict(m=m, name=out.name.decode(), split=out.split_k_slices))
            if selected(C.byref(out),m,n,k,32,12,C.byref(arr),s1) != 1:
                raise ValueError('explicit existing S1 parent is unavailable')
        rows.append(dict(n=n,k=k,any_m=observed,defaults=defaults))
    out = ConfigV4()
    if selected(C.byref(out),1,512,512,32,12,C.byref(arr),b'kpack4:8x32x256:8x16:s3:S4') != 0:
        raise ValueError('invalid explicit S4 was silently replaced')
    arr.mapping_id ^= 1
    if any_m(512,512,12,C.byref(arr)) != 0:
        raise ValueError('foreign arrangement admitted')
    return dict(mode=mode,status='PASS',scope='HOST_QUERY_ONLY_NO_KERNEL_LAUNCH',rows=rows)


def verify_overlay(root):
    root = root.resolve(strict=True)
    if (root/'manifest.json').is_symlink():
        raise ValueError('overlay manifest must be a regular file')
    record = json.loads((root/'manifest.json').read_text())
    if (record.get('schema') != SCHEMA or record.get('base_manifest_sha256') != BASE_MANIFEST or
            record.get('replacement_source') != FIX_SOURCE or record.get('definitions') != DEFS or
            record.get('device_admission') != 'PENDING'):
        raise ValueError('Q4 compatibility overlay identity differs')
    entries = record['libraries']
    expected = {r.filename: r for r in ppu_bundle.LIBRARY_ROLES}
    if len(entries) != 6 or {e['filename'] for e in entries} != set(expected):
        raise ValueError('overlay must contain exactly the original six roles')
    base = record['base_manifest']
    ppu_bundle._validate_manifest(base)
    # Store the original text so the upstream manifest identity remains exact.
    text = record['base_manifest_text']
    if hashlib.sha256(text.encode()).hexdigest() != BASE_MANIFEST or json.loads(text) != base:
        raise ValueError('overlay base manifest differs')
    originals = {e['filename']: e for e in base['libraries']}
    if {p.name for p in root.iterdir()} != {'manifest.json', *expected}:
        raise ValueError('overlay file inventory differs')
    for item in entries:
        name = item['filename']
        path = root/name
        source = FIX_SOURCE if name == Q4_LIBRARY else BASE_SOURCE
        if (set(item) != set(originals[name]) | {'source_commit'} or item.get('source_commit') != source or
                any(item[k] != v for k,v in originals[name].items() if k not in ('size','sha256'))):
            raise ValueError('overlay role/source differs: '+name)
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != item['size'] or
                sha(path) != item['sha256']):
            raise ValueError('overlay payload differs: '+name)
        if name != Q4_LIBRARY and (item['sha256'],item['size']) != (originals[name]['sha256'],originals[name]['size']):
            raise ValueError('unchanged compatibility library was replaced: '+name)
    return record


def build(args):
    base, sdk = args.base.resolve(strict=True), args.sdk.resolve(strict=True)
    original = ppu_bundle.verify_bundle(base, inspect_binaries=False)
    if sha(base/'manifest.json') != BASE_MANIFEST or original['source']['commit'] != BASE_SOURCE:
        raise ValueError('use the original restored runtime6 bundle, not a previous overlay')
    if args.jobs <= 0 or not (sdk/'bin/hgcc').is_file():
        raise ValueError('valid SDK and positive jobs are required')
    versions = [line.split(':',1)[1].strip() for line in (sdk/'release.yaml').read_text().splitlines()
                if line.startswith('version:')]
    if versions != [ppu_bundle.SDK_RELEASE]:
        raise ValueError(f'SDK release differs: {versions}; expected {ppu_bundle.SDK_RELEASE}')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PPU_SDK=str(sdk), PPU_ARCHS='ppu0010',
        PPU_DEFS=DEFS, TARGET='quactlize_ppu', JOBS=str(args.jobs),
        PPU_BUILD_DIR=str(out/'build'), PPU_BUILD_RESUME='0', GIT_LFS_SKIP_SMUDGE='1',
        PPU_PRESERVE_STALE_BUILD_TREES='1')
    env['LD_LIBRARY_PATH'] = ':'.join([str(sdk/'targets/x86_64-linux/lib'),
        str(sdk/'CUDA_SDK/targets/x86_64-linux/lib'),str(sdk/'lib'),env.get('LD_LIBRARY_PATH','')])
    def query(path, mode):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'probe', str(path), mode],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out/(mode+'-query.log')).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise ValueError(f'{mode} query failed; see {out/(mode+"-query.log")}')
        return parse_query_output(result.stdout)
    before = query(base/Q4_LIBRARY,'baseline')
    print('Q4_SMALLK baseline=EXPECTED_REJECTION explicit_S1=AVAILABLE',flush=True)
    if subprocess.run(['git','-C',str(ROOT),'cat-file','-e',FIX_SOURCE+'^{commit}'],
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        git(ROOT,'fetch','origin',FIX_BRANCH)
    if git(ROOT,'rev-parse',FIX_SOURCE+'^') != BASE_SOURCE:
        raise ValueError('replacement is not based on the published compatibility source')
    changed = set(git(ROOT,'diff','--name-only',BASE_SOURCE,FIX_SOURCE).splitlines())
    if changed != {'quactlize/include/ppu_q4_kpack4_shipping_policy.hpp','tests/test_kquant_any_m_abi.py'}:
        raise ValueError('replacement changes more than the selector and its regression')
    source = out/'source'
    subprocess.run(['git','-C',str(ROOT),'worktree','add','--detach',str(source),FIX_SOURCE],env=env,check=True)
    pin = git(source,'ls-tree','HEAD','--','third_party/actlize').split()[2]
    if subprocess.run(['git','-C',str(ROOT/'third_party/actlize'),'cat-file','-e',pin+'^{commit}'],
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        git(ROOT/'third_party/actlize','fetch','origin',pin)
    clone_checkout(ROOT/'third_party/actlize',source/'third_party/actlize',pin)
    git(source,'submodule','init','--','third_party/actlize')
    git(source,'submodule','absorbgitdirs','third_party/actlize')
    if git(source,'status','--porcelain'):
        raise ValueError('replacement source is not clean')
    log = out/'build.log'
    started = time.monotonic()
    print(f'Q4_SMALLK_BUILD format=Q4_K jobs={args.jobs} source={FIX_SOURCE} log={log}',flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(['bash',str(source/'build.sh')],cwd=source,env=env,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                try:
                    rc = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'Q4_SMALLK_BUILD running=1 elapsed_minutes={(time.monotonic()-started)/60:.1f} log={log}',flush=True)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            raise
    if rc:
        raise ValueError(f'Q4-only build failed rc={rc}; preserved at {out}; see {log}')
    candidates = list((out/'build').rglob('libquactlize_ppu.so'))
    if len(candidates) != 1 or candidates[0].is_symlink():
        raise ValueError('Q4 build did not produce exactly one regular library')
    if (git(source,'rev-parse','HEAD') != FIX_SOURCE or git(source,'status','--porcelain') or
            (out/'build/.quactlize-source-head').read_text().strip() != FIX_SOURCE or
            (out/'build/.quactlize-source-dirty').exists()):
        raise ValueError('replacement source changed during the build')
    candidate_sha = sha(candidates[0])
    after = query(candidates[0],'candidate')
    bundle = out/'bundle'
    bundle.mkdir()
    entries = []
    for old in original['libraries']:
        name = old['filename']
        replacement = name == Q4_LIBRARY
        shutil.copy2(candidates[0] if replacement else base/name,bundle/name)
        entries.append(dict(old,source_commit=FIX_SOURCE if replacement else BASE_SOURCE,
            size=(bundle/name).stat().st_size,sha256=sha(bundle/name)))
    if sha(bundle/Q4_LIBRARY) != candidate_sha:
        raise ValueError('replacement changed after its host query')
    record = dict(schema=SCHEMA,base_manifest_sha256=BASE_MANIFEST,base_manifest=original,
        base_manifest_text=(base/'manifest.json').read_text(),replacement_source=FIX_SOURCE,
        definitions=DEFS,sdk=str(sdk),libraries=entries,queries=[before,after],
        build_seconds=time.monotonic()-started,device_admission='PENDING')
    (bundle/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    verify_overlay(bundle)
    print(f'Q4_SMALLK_COMPLETE rebuilt=fmt0 unchanged=5 bundle={bundle} device_admission=PENDING',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('build')
    for name in ('base','sdk','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=192)
    p = sub.add_parser('probe')
    p.add_argument('library',type=Path)
    p.add_argument('mode',choices=('baseline','candidate'))
    p = sub.add_parser('verify')
    p.add_argument('bundle',type=Path)
    args = parser.parse_args()
    if args.command == 'build': build(args)
    elif args.command == 'probe': print(QUERY_PREFIX+json.dumps(probe(args.library,args.mode)))
    else:
        verify_overlay(args.bundle)
        print('Q4_SMALLK_OVERLAY VERIFIED rebuilt=fmt0 unchanged=5')


if __name__ == '__main__':
    main()

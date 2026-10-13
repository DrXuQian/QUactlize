#!/usr/bin/env python3
"""Off-device compile check of dynamic dense parents; NOT device admission."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import FLAGS, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()
    if args.jobs < 1:
        p.error('--jobs must be positive')
    sdk = args.sdk.resolve(strict=True)
    if not (sdk / 'bin/hgcc').is_file():
        p.error('SDK lacks bin/hgcc')
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PATH=str(sdk / 'bin') + ':' + os.environ.get('PATH', ''))
    include = [ROOT, ROOT/'quactlize/include', ROOT/'quactlize/runtime', ROOT/'quactlize/decode',
               ROOT/'third_party/actlize/include', ROOT/'third_party/actlize/tools/util/include',
               ROOT/'third_party/actlize/examples/common']
    cases = [(q, route, typed, 0) for q in (8, 10, 11, 12, 13, 14)
             for route in ((1,) if q == 8 else (0, 1)) for typed in (0, 1)]
    cases += [(q, 1 if q == 8 else 0, 1, 1) for q in (8, 10, 11, 12, 13, 14)]
    # Changing the base handle must not break the existing grouped module.
    cases += [(12, 2, 0, 0)]

    def compile_case(case):
        q, route, typed, bf16 = case
        name = f'q{q}-r{route}-typed{typed}-bf16{bf16}'
        obj, log = args.output/(name+'.o'), args.output/(name+'.log')
        command = [str(sdk/'bin/hgcc'), *FLAGS, f'-DQK_QTYPE={q}', f'-DQK_ROUTE={route}',
                   f'-DQKD_TEST_TYPED={typed}', f'-DQKD_USE_BF16_COMPUTE={bf16}',
                   f'-DPPU_PACKED_SCALE={int(route in (0, 2))}',
                   f'-DPPU_PACKED_FORMAT={ {8:0,10:2,11:3,12:0,13:1,14:4}[q] }',
                   *[f'-I{x}' for x in include], '-c', str(ROOT/'tests/kpack_dynamic_compile.cu'), '-o', str(obj)]
        with log.open('x') as stream:
            stream.write(json.dumps(command)+'\n'); stream.flush()
            rc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        return dict(case=name, rc=rc, object=str(obj), sha256=sha(obj) if rc==0 else None, log=str(log))

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(compile_case, case) for case in cases]):
            record = future.result(); records.append(record)
            print('KPACK_DYNAMIC_COMPILE '+json.dumps(record), flush=True)
    passed = all(r['rc']==0 for r in records)
    (args.output/'summary.json').write_text(json.dumps(dict(status='PASS' if passed else 'FAIL',
        scope='COMPILE_ONLY_NOT_DEVICE_ADMISSION', source=subprocess.check_output(
            ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), cases=records), indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

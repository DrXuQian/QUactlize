#!/usr/bin/env python3
"""Re-run the Q8 reader numerical gate through the production v2 C ABI."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt.native import Runtime
from dev.gemv_simt.production import Library
from dev.gemv_simt.q8_vector_run import numeric
from quactlize.runtime.compiler import sha
from tools.verify_kpack_dispatch import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','bundle','output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    manifest = verify(a.bundle, sdk=a.sdk)
    if 'q8_vector_policy' not in manifest:
        raise ValueError('production Q8 vector policy is missing')
    rt = Runtime(a.sdk, 'ppu')
    try:
        lib = a.bundle / 'libquactlize_ppu_execution.so'
        result = numeric(a, rt, [Library(lib, arm=i) for i in (0,1)])
        if result['expected'] != 12480 or len(result['records']) != 12480:
            raise ValueError('production Q8 gate denominator differs')
        result.update(execution_sha256=manifest['execution_sha256'],
                      manifest_sha256=sha(a.bundle/'manifest.json'), endpoint='PRODUCTION_SIMT_V2',
                      harness_sha256={str(f.relative_to(ROOT)):sha(f) for f in (
                          Path(__file__), ROOT/'dev/gemv_simt/production.py',
                          ROOT/'dev/gemv_simt/q8_vector_run.py', ROOT/'dev/gemv_simt/run.py')})
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('KPACK_DECODE_UPDATES PASS contexts=12480 endpoint=PRODUCTION_SIMT_V2 timing=NOT_MEASURED', flush=True)
    finally:
        rt.close()


if __name__ == '__main__':
    main()

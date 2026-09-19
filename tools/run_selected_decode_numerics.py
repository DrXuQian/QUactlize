#!/usr/bin/env python3
"""Check published selected entries numerically, without a performance sweep."""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.generate_decode_selection import EVIDENCE, effective, FIELDS
from tools.verify_kpack_dispatch import verify
from quactlize.runtime.compiler import sha


def gate_rows():
    evidence = json.loads(EVIDENCE.read_text())
    effective(evidence)  # Sealed authority, not an editable list of recipes.
    return [w for w in evidence['rows'] if w['automatic'] and w['config']['kind'] == 'simt']


class SelectedProvider:
    def __init__(self, bench, bundle, winner):
        from dev.gemv_model.plan import Candidate
        from dev.gemv_simt.production import Library
        from quactlize.dispatch.native import Dispatch
        from quactlize.execution.native import arrangement
        from quactlize.fusion.native import Library as FusionLibrary, integration_entries, Config
        self.b, self.arm = bench, 'selected'
        self.config = Candidate(**winner['body'])
        p = bench.point
        self.fusion = None
        if p.paired:
            self.fusion = FusionLibrary(bundle/'libquactlize_ppu_gate_up.so')
            self.entries = integration_entries(self.fusion)
            self.layout, self.paired_config = self.fusion.arrangement(p.q), Config()
            rc = self.entries['select'](p.q, p.n, p.k, p.experts, 1, p.compute, C.byref(self.paired_config))
            if rc or (self.paired_config.backend, self.paired_config.split, self.paired_config.warps) != (0, 1, 8):
                raise ValueError('published paired selection differs')
        else:
            d = Dispatch(bundle)
            try:
                choice = d.query_smallm_matched(bench.call(), arrangement(p.q), p.compute)
                if (choice is None or choice.base.kind != 1 or
                        {f: getattr(choice.base.simt, f) for f in FIELDS} != {f: winner['config'][f] for f in FIELDS}):
                    raise ValueError('published ordinary selection differs')
            finally:
                d.close()
            self.shipping = Library(bundle/'libquactlize_ppu_execution.so', p.compute)

    def prepare(self, copy=0):
        from quactlize.fusion.native import MappedCall, FusionCall
        from quactlize.execution.simt_codegen import Config
        b, p = self.b, self.b.point
        call = b.call(copy)
        if self.fusion:
            mapped = MappedCall(FusionCall(call, p.compute, 1, p.compute),
                                b.mapping.ptr if b.mapping and b.mapped_call else None,
                                b.status.ptr if b.status else None)
            return lambda: self.entries['run'](C.byref(mapped), C.byref(self.paired_config), C.byref(self.layout))
        return self.shipping.prepare(call, Config(**{f: getattr(self.config, f) for f in FIELDS}))

    def close(self):
        pass


def run_point(args, winner):
    from dev.gemv_model.plan import Point
    from dev.gemv_model.fixture import Bench
    from dev.gemv_model.engine import correctness
    from dev.gemv_simt.native import Runtime
    rt = Runtime(args.sdk, 'ppu')
    result = dict(point=winner['point']['name'], status='FAIL', timing_valid=False,
                  manifest_sha256=sha(args.bundle/'manifest.json'))
    provider = None
    try:
        # One resident copy suffices for numerics; there is no cache/timing claim.
        point = Point(**winner['point'])
        bench = Bench(rt, point, 0, profile=True)
        provider = SelectedProvider(bench, args.bundle, winner)
        proof, _ = correctness(provider, token_controls=range(1, 9))
        result.update(status='PASS', controls=proof,
                      scope='M1_SELECTED_RECIPE_NUMERICS_T1_TO_8_NOT_M2_TO_8_RANKING')
        return 0
    except Exception as exc:
        result['error'] = str(exc)
        traceback.print_exc()
        return 1
    finally:
        if provider:
            provider.close()
        rt.close()
        (args.output/(winner['point']['name']+'.json')).write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sdk', 'bundle', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--point', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.sdk, args.bundle, args.output = (p.resolve() for p in (args.sdk, args.bundle, args.output))
    verify(args.bundle, sdk=args.sdk)
    rows = gate_rows()
    if args.point:
        return run_point(args, next(w for w in rows if w['point']['name'] == args.point))
    args.output.mkdir(parents=True, exist_ok=False)
    failures = []
    for i, w in enumerate(rows):
        name = w['point']['name']
        command = [sys.executable, __file__, '--sdk', str(args.sdk), '--bundle', str(args.bundle),
                   '--output', str(args.output), '--point', name]
        with (args.output/(name+'.log')).open('w') as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            failures.append(name)
        print(f'SELECTED_DECODE_NUMERICS point={name} rc={rc} completed={i+1}/{len(rows)} remaining_continue=1', flush=True)
    summary = dict(status='FAIL' if failures else 'PASS', points=len(rows), failed=failures,
                   performance_sweep=False, timing_valid=False,
                   manifest_sha256=sha(args.bundle/'manifest.json'))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Profile one already-validated Q6 image: one weight copy, no graphs or timing ring.

--source is the immutable source snapshot of the original experiment. Import
its fixture, ABI and launcher, and verify its receipt before touching the GPU.
This script never recompiles or rewrites the original experiment's results.
"""
import argparse
from contextlib import contextmanager, ExitStack
import ctypes as C
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


@contextmanager
def phase(name):
    started = time.monotonic()
    print(f'Q6_PROFILE_STAGE stage={name} state=BEGIN', flush=True)
    try:
        yield
    except BaseException:
        print(f'Q6_PROFILE_STAGE stage={name} state=FAIL seconds={time.monotonic()-started:.3f}', flush=True)
        raise
    print(f'Q6_PROFILE_STAGE stage={name} state=END seconds={time.monotonic()-started:.3f}', flush=True)


def validate_prior(prior, authority, record, workload, digest):
    if (prior.get('status') != 'PASS' or prior.get('profiled') is not False or
            prior.get('raw_equal') is not True or len(prior.get('timings', [])) != 2 or
            prior.get('authority_sha256') != digest(authority) or
            prior.get('library_sha256') != record['sha256'] or
            prior.get('workload') != asdict(workload)):
        raise ValueError('matching completed numerical/timing point required; do not bypass the gate')


def measure(a, experiment, sdk, workload, library, api):
    proofs = []
    with ExitStack() as cleanup:
        with phase('cpu-fixture'):
            weights = api.Weights(workload.n, workload.k, workload.experts)
        with phase('one-copy-upload'):
            owner = api.Resources(sdk)
            cleanup.callback(owner.close)
            buffers = {name: api.Buffer(owner, value.nbytes)
                       for name, value in weights.planes.items() if value.nbytes}
            for name, buffer in buffers.items():
                buffer.upload(weights.planes[name])
            planes = {name: buffers[name].ptr if name in buffers else None for name in weights.planes}
            ring = SimpleNamespace(w=weights, planes=[planes])
            weight_bytes = sum(value.nbytes for value in weights.planes.values())
            print(f'Q6_PROFILE_STORAGE copies=1 weight_bytes={weight_bytes} cold_ring=0 graphs=0', flush=True)
        with phase('context-and-image'):
            p = api.point(workload)
            p['mode'] = 2 if workload.operation == 'indexed' else 0
            b = api.Context(sdk, ring, p)
            cleanup.callback(b.close)
            core = experiment.Kernel(b, library)
        with phase('warmup-and-correctness'):
            for repeat in (0, 1, 0):
                b.update(repeat)
                b.out.poison()
                api.checked(core.launch(a.arm), 'selected Q6 warmup')
                proofs.append(b.check())
        start, stop = sdk.lib.hggcProfilerStart, sdk.lib.hggcProfilerStop
        for fn in (start, stop):
            fn.argtypes, fn.restype = [], C.c_int
        # Print BEFORE ProfilerStart: it too may block in the profiler/runtime.
        with phase('profiler-start'):
            api.checked(start(), 'ACU profiler start')
        try:
            with phase('selected-kernel'):
                api.checked(core.launch(a.arm), 'one profiled Q6 call')
                sdk.synchronize(b.r.stream)
        finally:
            with phase('profiler-stop'):
                api.checked(stop(), 'ACU profiler stop')
        with phase('post-profile-correctness'):
            proofs.append(b.check())
            for buffer in buffers.values():
                buffer.guard()
    return dict(status='PASS', workload=asdict(workload), numerical=proofs,
                arm=a.arm, weight_copies=1, weight_bytes=weight_bytes,
                cold_ring=False, graph_api_calls=0, timings=[],
                profiled=True, performance_admitted=False, production_changed=False,
                scope='ONE_DIRECT_KERNEL_ACU_NOT_ROTATING_EVENT_TIMING')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'results', 'sdk', 'result'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--point', type=int, default=0)
    parser.add_argument('--arm', type=int, choices=(0, 1), required=True)
    parser.add_argument('--l2-bytes', type=int, default=67108864)
    a = parser.parse_args()
    a.source = a.source.resolve(strict=True)
    a.results = a.results.resolve(strict=True)
    a.sdk = a.sdk.resolve(strict=True)
    a.result = a.result.resolve()
    if a.result.exists() or not a.result.parent.is_dir():
        parser.error('--result must be a new file in an existing directory')
    with phase('frozen-source-imports'):
        sys.path.insert(0, str(a.source))
        experiment = importlib.import_module('tools.run_q6_metadata')
        if experiment.ROOT != a.source:
            raise ValueError('imports did not resolve to the original frozen source')
        from dev.strategy.provenance import harness_identity
        from dev.strategy.journal import atomic_json
        from quactlize.runtime.tuning import digest
        from quactlize.runtime.native import SDK, sdk_identity, checked
        from quactlize.runtime.compiler import sha
        from tools.run_kpack_llama_ab import device
        from dev.gemv_simt.q6_metadata_fixture import Weights
        from dev.smallm_closure.bench import Buffer, Context
        from dev.strategy.indexed import point
        from tools.run_kpack_gemv_gate import Resources
    with phase('source-and-device-verification'):
        authority = json.loads((a.results/'authority.json').read_text())
        manifest = json.loads((a.results/'build/manifest.json').read_text())
        if (harness_identity(a.source) != authority['harness'] or
                experiment.inputs() != manifest['sources'] or manifest.get('object_only') or
                manifest.get('sdk') != sdk_identity(a.sdk) or device(a) != authority['device']):
            raise ValueError('frozen source, SDK or physical device differs from the original run')
        workloads = experiment.workloads()
        if not 0 <= a.point < len(workloads):
            raise ValueError('point outside the original workload inventory')
        workload = workloads[a.point]
        record, = [r for r in manifest['records'] if r['compute'] == workload.compute]
        library = a.results/'build'/record['image']
        if sha(library) != record['sha256']:
            raise ValueError('previous compiled image changed')
        prior_path = a.results/f'point-{a.point}.json'
        prior = json.loads(prior_path.read_text())
        validate_prior(prior, authority, record, workload, digest)
    result = measure(a, experiment, SDK(a.sdk), workload, library,
                     SimpleNamespace(Weights=Weights, Resources=Resources, Buffer=Buffer,
                                     Context=Context, point=point, checked=checked))
    result.update(authority_sha256=digest(authority), library_sha256=record['sha256'],
                  prior_point_sha256=sha(prior_path), expected_node=prior['nodes'][a.arm],
                  profile_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(a.result, result)
    print('Q6_PROFILE_DONE '+json.dumps(dict(status='PASS', point=a.point, arm=a.arm,
                                            cold_ring=False, graphs=False, result=str(a.result))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

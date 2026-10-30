"""Correctness before separate single-kernel timing; no mixed-call intervals."""
from contextlib import contextmanager
import ctypes as C
import time

import numpy as np

from dev.strategy.cold_decode import load_component, publish_components, replay_extracted
from dev.strategy.components import component_key, stage_role, statistics_checked
from dev.strategy.isolated import Captured
from quactlize.dispatch.table import Recipe
from quactlize.runtime.native import checked
from quactlize.runtime.tuning import digest


def expect_oracle_failure(proof, label):
    try:
        proof()
    except ValueError as exc:
        if not str(exc).startswith(('prefill oracle:', 'full BF16 GGUF oracle/')):
            raise
        return
    raise ValueError(label+' was not detected by the independent oracle')


def stages(capture, b, recipe, authority, source, output, proof, *, helper=False):
    split = dict(recipe.parameters).get('split', 1) if recipe else 1
    nodes = capture.descriptions[:len(capture.groups)]
    roles = [stage_role(n['name'], split) for n in nodes]
    if 'unclassified' in roles or (helper and roles != ['expansion']) or (
            not helper and 'producer' not in roles) or (
            not helper and split > 1 and 'reducer' not in roles):
        raise ValueError('prefill component inventory differs: '+repr(list(zip(roles, nodes))))
    replay_extracted(capture, proof)
    pending, result = [], []
    for i, (node, role) in enumerate(zip(nodes, roles)):
        rotating = role in ('producer', 'expansion')
        cache = 'ROTATING_ACTIVE_READ_SET_AT_LEAST_2P25_L2' if rotating else 'REPEATED_RESIDENT_SCRATCH'
        node = dict(node, split=split)
        if helper and b.w.operation == 'indexed':
            # This isolated indexed expansion consumes IDs/count, not token
            # rows or their per-expert distribution. Exact ID-set reuse is
            # legal across M; a different active set remains a different job.
            node['active_expert_ids'] = list(map(int, b.route['active']))
        cid, identity = component_key(b.w, recipe, node, authority=authority, source=source, cache=cache)
        path = output/'components'/(cid+'.json')
        if path.exists():
            row = load_component(path, identity)
        else:
            graph = capture.isolate(i, weight_ring=rotating, repeats=16)
            try:
                row = statistics_checked([[graph.sample() for _ in range(3)] for _ in range(3)])
                row.update(identity=identity, component_id=cid, kind=role, first_use_excluded=True,
                    timed_kernel_types=1, calls_per_graph=graph.count, host_launch_cost_added=False,
                    cache=cache)
                row['record_sha256'] = digest(row)
                pending.append((path, row))
            finally:
                graph.close()
        result.append(dict(id=cid, role=role, median_us=row['median_us'], spread_pct=row['spread_pct']))
    replay_extracted(capture, proof)
    return result, pending


@contextmanager
def prepared(b, core, recipe, *, config=0, full=False):
    capture = None
    start = time.monotonic()
    try:
        numerical = []
        for repeat in (0, 1):
            b.update(repeat)
            core.setup(config)
            for copy in (0, core.copies-1 if full else b.ring.copies-1):
                checked(core.launch(copy), 'prefill independent correctness')
                numerical.append(b.check())
        if b.w.compute == 'bf16':
            b.update(2, amplitude=2**20)
            if float(np.max(np.abs(b.coeff))) <= 65504: raise ValueError('large BF16 fixture missing')
            core.setup(config); checked(core.launch(), 'large BF16 activation')
            numerical.append(b.check())
        b.update(0, zero=True); core.setup(config)
        checked(core.launch(), 'prefill zero A'); b.check(zero=True)
        expect_oracle_failure(b.check, 'zero-output negative')
        b.update(0); core.setup(config)
        copies = core.copies if full else b.ring.copies
        for i in range(copies): checked(core.launch(i), 'untimed provider/JIT/upload warmup')
        b.check()
        # Every weight plane must be consumed from the last ring copy; keep
        # copy0 immutable as an independent address-binding control.
        pointer_negatives = []
        if not full:
            for name, plane in b.ring.objects[-1].items():
                try:
                    b.ring.r.fill(plane.ptr, 0, plane.size)
                    b.sdk.synchronize(b.ring.r.stream)
                    core.setup(config, validate=False)
                    checked(core.launch(copies-1), 'last weight copy negative')
                    expect_oracle_failure(b.check, 'last-copy '+name)
                    checked(core.launch(0), 'first weight copy isolation'); b.check()
                    pointer_negatives.append(name)
                finally:
                    b.rt.sync(); plane.upload(b.ring.w.planes[name])
            core.setup(config)
        else:
            back = core.weights[-1]
            b.r.fill(back.data_ptr(), 0, back.numel()*2); b.rt.sync()
            checked(core.launch(copies-1), 'last expanded weight copy negative')
            expect_oracle_failure(b.check, 'last expanded weight copy')
            checked(core.launch(0), 'first expanded weight isolation'); b.check()
            pointer_negatives.append('BF16_WEIGHTS')
            core.setup(config)
        for i in range(copies): checked(core.launch(i), 'restored whole ring')
        capture = core.capture() if full else Captured(b.rt, [lambda i=i: core.launch(i) for i in range(copies)])
        kinds = [stage_role(n['name'], dict(recipe.parameters).get('split', 1))
                 for n in capture.descriptions[:len(capture.groups)]]
        if 'producer' not in kinds: raise ValueError('captured provider contains no recognized GEMM')
        b.update(1); core.setup(config)
        for i, kind in enumerate(kinds):
            if kind != 'producer':
                g = capture.isolate(i, weight_ring=False, validation_only=True); g.close()
        expect_oracle_failure(b.check, 'omitted producer')
        for i in range(copies): checked(core.launch(i), 'changed-input whole ring')
        b.out.poison(); replay_extracted(capture, b.check)
        # Every compared component uses the SAME resident routing domain.
        # Changed-A/router checks are controls, not a different timed workload.
        b.update(0); core.setup(config)
        for i in range(copies): checked(core.launch(i), 'restore timing input/router')
        b.out.poison(); replay_extracted(capture, b.check)
        data = dict(correctness='PENDING', numerical=numerical, pointer_negatives=pointer_negatives,
            omitted_producer='DETECTED', changed_input_and_router='PASS', zero_a='PASS',
            receipt=core.receipt, fixture=b.ring.receipt, timed_router_sha256=b.route['sha256'],
            timed_active_experts=len(b.route['active']), production_changed=False, performance_admitted=False)
        if hasattr(core, 'timing_fixture'):
            data.update(timing_input_repeat=0, timed_fixture=core.timing_fixture())
        yield capture, data
        b.out.poison(); replay_extracted(capture, b.check)
        checked(core.launch(), 'final complete call'); numerical.append(b.check())
        for buf in [*b.ring.buffers, *getattr(core, 'buffers', [])]: buf.guard()
        if full:
            from tools.kpack_bf16_providers import loaded_images
            core.guards(); data['provider_loaded_images'] = loaded_images()
        data.update(correctness='PASS', wall_seconds=time.monotonic()-start)
    finally:
        if capture: capture.close()


def measure_core(b, core, recipe, authority, source, output, *, config=0, full=False, stage_measure=None):
    stage_measure = stage_measure or stages
    with prepared(b, core, recipe, config=config, full=full) as (capture, data):
        images = None
        if full:
            from tools.kpack_bf16_providers import loaded_images
            images = loaded_images()
            if not images: raise ValueError('no installed BF16 provider image identity')
            source = digest([source, images])
        rows, pending = stage_measure(capture, b, recipe, authority, source, output, b.check)
        data.update(components=[r['id'] for r in rows], stages=rows)
        if full and images != loaded_images():
            raise ValueError('provider images changed during component timing')
    publish_components(pending)
    return data


def measure_full_expansion(b, full, config, authority, source, output, *, stage_measure=None):
    stage_measure = stage_measure or stages
    b.update(0); full.update(); full.poison_weights()
    for i in range(b.ring.copies): checked(full.dq(i, config), 'full expansion independent gate')
    proof = full.check_weights()
    negative = []
    for name, plane in b.ring.objects[-1].items():
        try:
            b.ring.r.fill(plane.ptr, 0, plane.size); b.sdk.synchronize(b.ring.r.stream)
            checked(full.dq(b.ring.copies-1, config), 'dequant last plane negative')
            expect_oracle_failure(full.check_weights, 'full-dequant plane '+name)
            checked(full.dq(0, config), 'dequant copy0 control'); full.check_weights()
            negative.append(name)
        finally:
            b.rt.sync(); plane.upload(b.ring.w.planes[name])
    # Mutable GPU count: replaying the same graph with zero selected experts
    # must leave the complete original E-axis allocation untouched.
    capture = Captured(b.rt, [lambda i=i: full.dq(i, config) for i in range(b.ring.copies)])
    try:
        if b.w.operation == 'indexed':
            full.count.upload(np.zeros(1, dtype='i4')); full.poison_weights()
            g = capture.isolate(0, weight_ring=False, validation_only=True); g.close()
            old = b.route['active']; b.route['active'] = np.empty(0, dtype='i4')
            try: full.check_weights()
            finally: b.route['active'] = old
            full.update()
        full.poison_weights()
        rows, pending = stage_measure(capture, b, None, authority, source, output, full.check_weights, helper=True)
        full.guards()
        for buf in b.ring.buffers: buf.guard()
        result = dict(correctness='PASS', components=[r['id'] for r in rows], stages=rows,
            config=config, numerical=proof, pointer_negatives=negative,
            mutable_count='PASS' if b.w.operation == 'indexed' else 'NOT_INDEXED',
            scope='FULL_DEQUANT_ONLY_ACTIVE_EXPERTS_OR_DENSE', fixture=b.ring.receipt,
            output_cache='REUSED_OUTPUT_SEPARATE_FROM_ROTATING_BF16_PROVIDER_RING')
    finally:
        capture.close()
    publish_components(pending)
    return result

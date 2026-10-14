"""Direct, already-built TC modules; no dispatcher/JIT or production edits.

Two resident weight copies exercise pointer binding. They are deliberately
NOT a cold ring, and these smoke timings cannot enter a performance table.
"""
import ctypes as C
import time

import numpy as np

from dev.bf16_compute.fixture import compare, round_compute
from dev.smallm_closure.bench import Buffer, Context
from dev.smallm_closure.fixture import Weights
from dev.strategy.components import stage_role, statistics_checked
from dev.strategy.indexed import RuntimeView, TensorCore as IndexedCore
from dev.strategy.isolated import Captured, checked
from dev.strategy.resources import NativeProbe, request_abi
from quactlize.runtime.native import Recipe as Launch, Resources as Query
from tools.run_kpack_gemv_gate import Resources


def scale_activation_data(data, amplitude):
    # NumPy 1.x promotes float32 * a large Python int to float64. Keep the
    # F32 upload ABI while preserving F64 oracle/denominator accumulation.
    for name in ('a', 'gold', 'denom'):
        data[name] = data[name] * np.asarray(amplitude, dtype=data[name].dtype)


class WarmRing:
    def __init__(self, sdk, w):
        self.r = Resources(sdk)
        self.buffers, self.planes, self.objects, self.copies = [], [], [], 2
        try:
            self.w = Weights(w.qtype, w.n, w.k, w.experts)
            for _ in range(self.copies):
                plane, objects = {}, {}
                for name, value in self.w.planes.items():
                    if not value.size:
                        plane[name] = None
                        continue
                    b = Buffer(self.r, value.nbytes)
                    b.upload(value)
                    self.buffers.append(b)
                    plane[name] = b.ptr
                    objects[name] = b
                self.planes.append(plane)
                self.objects.append(objects)
            self.receipt = dict(fixture=self.w.record, copies=self.copies,
                cache='WARM_SMOKE_NO_L2_CLAIM', cold_ring=False)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.r.close()


class DenseCore:
    """Native F16 prefill and F32-storage decode are distinct endpoints."""
    def __init__(self, b, record, case, query_receipt):
        self.b, self.r = b, Resources(b.sdk)
        self.handles, self.buffers = [], []
        self.native = case['spec']['endpoint'] == 'native'
        try:
            probe = NativeProbe(case['spec'], record)
            self.library = probe.lib
            request = request_abi(query_receipt['request'], probe.device)
            call = request if self.native else request.dense.call
            launch = case['binding']['launch']
            self.recipe = Launch(1, C.sizeof(Launch), launch['algorithm'], launch['split'], launch['grid'])
            def alloc(size):
                value = Buffer(self.r, size)
                self.buffers.append(value)
                return value
            self.a = alloc(b.rows * b.p['k'] * 2) if self.native else b.a
            self.out = alloc(b.rows * b.p['n'] * 2) if self.native else b.out
            prefix = '' if self.native else 'decode_dense_'
            query = probe._bind(prefix + ('query_v1' if self.native else 'query_dynamic_v2'),
                [C.POINTER(type(request)), C.POINTER(Launch), C.POINTER(Query)])
            actual = Query()
            checked(query(C.byref(request), C.byref(self.recipe), C.byref(actual)), 'resolved dense query')
            for field in ('workspace_bytes', 'shared_bytes', 'occupancy'):
                if getattr(actual, field) != query_receipt[field]:
                    raise ValueError('dense resources changed after probe: ' + field)
            prepare = probe._bind(prefix + ('prepare_v1' if self.native else 'prepare_v2'),
                [C.POINTER(type(request)), C.POINTER(Launch), C.POINTER(C.c_void_p)])
            self.run_fn = probe._bind(prefix + 'run_v1', [C.c_void_p, C.c_void_p])
            self.destroy = probe._bind(prefix + 'destroy_v1', [C.c_void_p], None)
            call.a, call.output, call.stream = self.a.ptr, self.out.ptr, b.r.stream.value
            for plane in b.ring.planes:
                scratch = alloc(actual.workspace_bytes)
                call.workspace, call.workspace_bytes = scratch.ptr, scratch.size
                call.low, call.high, call.metadata = plane['low'], plane['high'], plane['units']
                call.zero = plane.get('zero')
                handle = C.c_void_p()
                checked(prepare(C.byref(request), C.byref(self.recipe), C.byref(handle)), 'explicit dense prepare')
                self.handles.append(handle)
            self.receipt = dict(endpoint=case['spec']['endpoint'], compute=case['spec']['compute'],
                storage='F16_NATIVE' if self.native else 'F32', host_cast_untimed=self.native,
                model_adapter_cost_included=False, launch=launch, module_sha256=record['sha256'],
                build_key=record['key'], source='SEALED_RESOURCE_MODULE_NOT_DISPATCHER')
        except BaseException:
            self.close()
            raise

    def update(self, zero=False):
        if self.native:
            value = np.zeros_like(self.b.data['a']) if zero else self.b.data['a']
            self.a.upload(np.asarray(value, dtype='<f2'))

    def read(self):
        if self.native:
            self.out.guard()
            return np.frombuffer(self.b.sdk.download(self.out.ptr, self.out.size), dtype='<f2').astype('f4').reshape(self.b.rows, self.b.p['n'])
        return self.out.read((self.b.rows, self.b.p['n']))

    def launch(self, index=0):
        return self.run_fn(self.handles[index], self.b.r.stream)

    def close(self):
        self.b.sdk.synchronize(self.b.r.stream)
        for handle in self.handles:
            self.destroy(handle)
        self.handles = []
        self.r.close()


def measure_case(sdk, case, record, query_receipt, *, rounds=3, samples=3, replays=16):
    from quactlize.dispatch.table import Workload, Recipe
    w, recipe = Workload(**case['workload']), Recipe.from_dict(case['recipe'])
    started = time.monotonic()
    ring = WarmRing(sdk, w)
    b = core = capture = None
    try:
        point = dict(q=w.qtype, n=w.n, k=w.k, tokens=w.m, mode=2 if w.operation == 'indexed' else 0,
                     experts=w.experts, topk=w.topk, channels=w.channels, compute=w.compute, router=w.router or 'real')
        b = Context(sdk, ring, point)
        rt = RuntimeView(b)
        core = IndexedCore(b, record, recipe, w) if point['mode'] else DenseCore(b, record, case, query_receipt)
        output = b.out if point['mode'] else core.out
        def update(repeat, zero=False, amplitude=1):
            b.update(repeat)
            if amplitude != 1:
                # Power-of-two scaling retains the BF16 oracle rounding and
                # exposes any accidental F16 intermediate (not just ABI tags).
                scale_activation_data(b.data, amplitude)
                b.a.upload(b.data['a'])
            if zero:
                b.a.upload(np.zeros_like(b.data['a']))
            if not point['mode']:
                core.update(zero)
            output.poison()
        def read():
            rt.sync()
            b.a.guard()
            if b.ids:
                b.ids.guard()
            return b.out.read((b.rows, w.n)) if point['mode'] else core.read()
        def proof():
            gold = b.data['gold']
            if not point['mode'] and core.native:
                gold = round_compute(gold, 'f16')
            return compare(read(), gold, b.data['denom'])
        numerical = []
        for repeat in (0, 1):
            update(repeat)
            for copy in range(ring.copies):
                checked(core.launch(copy), 'complete call independent correctness')
                numerical.append(proof())
        large_activation_max = None
        if w.compute == 'bf16':
            update(2, amplitude=2**20)
            large_activation_max = float(np.max(np.abs(b.data['a'])))
            if large_activation_max <= 65504:
                raise ValueError('BF16 large-activation fixture did not exceed F16 range')
            for copy in range(ring.copies):
                checked(core.launch(copy), 'BF16 large-activation check')
                numerical.append(proof())
        update(0, zero=True)
        checked(core.launch(), 'zero-input control')
        zero = read()
        if not np.isfinite(zero).all() or np.any(zero != 0):
            raise ValueError('zero-input control produced nonzero/nonfinite output')
        try:
            proof()
        except ValueError as exc:
            if not str(exc).startswith('typed oracle mismatch:'):
                raise
        else:
            raise ValueError('oracle accepted deliberately wrong zero output')
        update(0)
        for copy in range(ring.copies):
            checked(core.launch(copy), 'untimed capture setup')
        rt.sync()
        positive = read().copy()
        pointer_negatives = []
        # Identical replicas alone cannot prove that the last prepared
        # handle consumes its own planes. Mutate only that fixture copy.
        for name, buffer in ring.objects[-1].items():
            try:
                ring.r.fill(buffer.ptr, 0, buffer.size)
                sdk.synchronize(ring.r.stream)
                checked(core.launch(ring.copies - 1), 'last-copy plane negative')
                if np.array_equal(read().view('u4'), positive.view('u4')):
                    raise ValueError('last-copy plane was not consumed: ' + name)
                checked(core.launch(0), 'first-copy isolation control')
                proof()
                pointer_negatives.append(name)
            finally:
                rt.sync()
                buffer.upload(ring.w.planes[name])
        for copy in range(ring.copies):
            checked(core.launch(copy), 'restored capture setup')
        rt.sync()
        capture = Captured(rt, [lambda i=i: core.launch(i) for i in range(ring.copies)])
        nodes = capture.descriptions[:len(capture.groups)]
        split = dict(recipe.parameters)['split']
        kinds = [stage_role(node['name'], split) for node in nodes]
        if kinds.count('producer') != 1 or 'unclassified' in kinds:
            raise ValueError('unexpected captured component inventory: ' + repr([
                dict(role=kind, **node) for kind, node in zip(kinds, nodes)]))
        if split > 1 and not any(k in ('reducer', 'reducer_finish') for k in kinds):
            raise ValueError('split call has no captured reducer/finish')
        # Old partials contain A0. Change A/router BEFORE executing extracted
        # stages. Omitting the producer must not pass using those stale bytes.
        update(1)
        for stage, kind in enumerate(kinds):
            if kind != 'producer':
                graph = capture.isolate(stage, weight_ring=False, validation_only=True)
                graph.close()
        try:
            proof()
        except ValueError as exc:
            if not str(exc).startswith('typed oracle mismatch:'):
                raise
        else:
            raise ValueError('omitted-producer negative was not detected')
        output.poison()
        for stage in range(len(nodes)):
            graph = capture.isolate(stage, weight_ring=False, validation_only=True)
            graph.close()
        extracted = proof()
        # Publish no measurements until every extracted stage and final
        # independent output check succeed. One kernel type per timer.
        timed = []
        for stage, node in enumerate(nodes):
            graph = capture.isolate(stage, weight_ring=True, repeats=replays)
            try:
                data = [[graph.sample() for _ in range(samples)] for _ in range(rounds)]
                timed.append(dict(statistics_checked(data), node=node, role=kinds[stage],
                    timed_kernel_types=1, first_use_excluded=True, calls_per_graph=graph.count,
                    cache='WARM_SMOKE_NO_COLD_OR_PERFORMANCE_ADMISSION'))
            finally:
                graph.close()
        checked(core.launch(), 'post-timer complete call')
        numerical.append(proof())
        for buffer in ring.buffers:
            buffer.guard()
        owned = core.buffers if isinstance(core, DenseCore) else [v for v in vars(core).values() if isinstance(v, Buffer)]
        for buffer in owned:
            buffer.guard()
        return dict(status='PASS', numerical=numerical, extracted=extracted,
            negatives=dict(zero_input='DETECTED', omitted_producer='DETECTED', ring_planes=pointer_negatives),
            large_activation_max=large_activation_max,
            components=timed, receipt=core.receipt, fixture=ring.receipt,
            wall_seconds=time.monotonic()-started, production_changed=False,
            performance_admitted=False, scope='WARM_TC_MECHANISM_GATE_NOT_COLD_SELECTION')
    finally:
        if capture:
            capture.close()
        if core:
            core.close()
        if b:
            b.close()
        ring.close()

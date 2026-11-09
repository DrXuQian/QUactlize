#!/usr/bin/env python3
"""Cold-weight decode GEMV benchmark through the PUBLIC dense / MoE ABI.

One process loads exactly one package (libncp_quactlize.so + its private runtime).
For every case it packs random-but-valid GGUF blocks with the public packer, prepares
the production plan (or an explicit dense candidate), captures a graph that cycles
through a ring of distinct weight copies (dense) or distinct expert IDs (MoE), and
replays it. GPU kernel durations come from Asys; this script only emits the timed
regions, bracketed by unique-size memset markers on the benchmark stream.

No timing is computed here. Use parse_asys.py on the exported SQLite.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get('QGEMV_SRC', str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(SRC))
from quactlize.runtime.native import SDK, checked  # noqa: E402
from ci.support.packing import bind as bind_stream, device_identity  # noqa: E402
from ci.support.graph_api import graph_bind  # noqa: E402
from ci.support.dense import Arrangement, Sizes, Params, Info, Candidate, Options  # noqa: E402
from ci.support import moe as moe_abi  # noqa: E402
from reference import gguf_kpack as ref  # noqa: E402

MARK_BASE = 1 << 20          # marker memset sizes: MARK_BASE + 64*(2*index+phase)


def raw_blocks(q, rows, k, rng):
    """Random codes; dyadic finite d/dmin. Values only need to be representative and finite."""
    if q == 8:
        raw = rng.integers(0, 256, (rows, k // 32, 34), dtype=np.uint8)
        d = np.ldexp(np.ones((rows, k // 32), np.float32), -rng.integers(8, 12, (rows, k // 32))).astype('<f2')
        raw[..., :2] = d.view(np.uint8).reshape(rows, k // 32, 2)
        return raw
    spec = ref.SPECS[q]
    raw = rng.integers(0, 256, (rows, k // 256, spec.raw_bytes), dtype=np.uint8)
    d = np.ldexp(np.ones((rows, k // 256), np.float32), -rng.integers(8, 12, (rows, k // 256))).astype('<f2')
    raw[..., spec.d_offset:spec.d_offset + 2] = d.view(np.uint8).reshape(rows, k // 256, 2)
    if spec.has_min:
        raw[..., spec.dmin_offset:spec.dmin_offset + 2] = (d / np.float16(4)).astype('<f2').view(np.uint8).reshape(
            rows, k // 256, 2)
    return raw


class Bench:
    def __init__(self, package, sdk_path, cases=2048, packer_only=False):
        self.sdk = SDK(sdk_path)
        graph_bind(self.sdk)
        self.packer_only = packer_only
        library = 'libquactlize_ppu_pack.so' if packer_only else 'libncp_quactlize.so'
        self.lib = C.CDLL(str(Path(package) / library), mode=C.RTLD_LOCAL)
        bind_stream(self.sdk, self.lib)
        self.device = device_identity(self.sdk) if packer_only else None
        L = self.lib
        sig = {
            'quactlize_ppu_kpack_canonical_arrangement_v1': ([C.c_int, C.POINTER(Arrangement)], C.c_int),
            'quactlize_ppu_kpack_sizes_for_arrangement_v1': ([C.c_int] * 4 + [C.POINTER(Arrangement), C.POINTER(Sizes)], C.c_int),
            'quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2': (
                [C.c_void_p] * 4 + [C.c_int] * 4 + [C.POINTER(Arrangement), C.c_void_p], C.c_int),
            'quactlize_ppu_dense_create_with_options_v1': ([C.c_int, C.POINTER(Options), C.POINTER(C.c_void_p)], C.c_int),
            'quactlize_ppu_dense_prepare_v1': ([C.c_void_p, C.POINTER(Params)] + [C.c_void_p] * 3 +
                                               [C.POINTER(C.c_void_p), C.POINTER(Info)], C.c_int),
            'quactlize_ppu_dense_prepare_candidate_v1': ([C.c_void_p, C.POINTER(Params)] + [C.c_void_p] * 3 +
                                                         [C.POINTER(Candidate), C.POINTER(C.c_void_p), C.POINTER(Info)], C.c_int),
            'quactlize_ppu_dense_run_v1': ([C.c_void_p] * 4 + [C.c_uint64, C.c_void_p], C.c_int),
            'quactlize_ppu_dense_destroy_v1': ([C.c_void_p], None),
            'quactlize_ppu_dense_error_v1': ([], C.c_char_p),
            'quactlize_ppu_moe_matmul_create_v1': ([C.c_int32, C.POINTER(C.c_void_p)], C.c_int),
            'quactlize_ppu_moe_matmul_prepare_v1': ([C.c_void_p, C.POINTER(moe_abi.Params)] + [C.c_void_p] * 3 +
                                                    [C.POINTER(C.c_void_p), C.POINTER(moe_abi.Info)], C.c_int),
            'quactlize_ppu_moe_matmul_run_v1': ([C.c_void_p] * 5 + [C.c_uint64, C.c_void_p], C.c_int),
            'quactlize_ppu_moe_matmul_destroy_v1': ([C.c_void_p], None),
            'quactlize_ppu_moe_matmul_error_v1': ([], C.c_char_p),
        }
        if packer_only:
            sig = {k: v for k, v in sig.items() if k.startswith((
                'quactlize_ppu_kpack_', 'quactlize_ppu_prepare_fully_quantized_'))}
        for name, (args, res) in sig.items():
            fn = getattr(L, name)
            fn.argtypes, fn.restype = args, res
        self.stream = C.c_void_p()
        checked(self.sdk.lib.hggcStreamCreateWithFlags(C.byref(self.stream), 1), 'stream')
        # Two marker sizes per case index; the buffer must cover the largest one.
        self.marks = max(2048, cases)
        self.mark = self.sdk.allocate(MARK_BASE + 64 * 2 * self.marks)
        self.dense_ctx = {}
        self.moe_ctx = None
        self.protos = {}
        self.wcache = None

    def dense_context(self, compute):
        if compute not in self.dense_ctx:
            ctx = C.c_void_p()
            opt = Options(1, C.sizeof(Options), compute, 0)
            self.check(self.lib.quactlize_ppu_dense_create_with_options_v1(0, C.byref(opt), C.byref(ctx)), 'dense')
            self.dense_ctx[compute] = ctx
        return self.dense_ctx[compute]

    def moe_context(self):
        if self.moe_ctx is None:
            ctx = C.c_void_p()
            self.check(self.lib.quactlize_ppu_moe_matmul_create_v1(0, C.byref(ctx)), 'moe')
            self.moe_ctx = ctx
        return self.moe_ctx

    def check(self, rc, what):
        if rc:
            if self.packer_only:
                raise RuntimeError(f'{what} rc={rc} (explicit prototype, no production fallback)')
            err = (self.lib.quactlize_ppu_moe_matmul_error_v1() if what == 'moe' else
                   self.lib.quactlize_ppu_dense_error_v1())
            raise RuntimeError(f'{what} rc={rc}: {(err or b"").decode(errors="replace")}')

    def marker(self, index, phase):
        if index >= self.marks:
            raise RuntimeError(f'marker index {index} exceeds the marker buffer ({self.marks})')
        size = MARK_BASE + 64 * (2 * index + phase)
        checked(self.sdk.lib.hggcMemsetAsync(self.mark, phase + 1, size, self.stream), 'marker')

    def packed(self, q, n, k, experts, seed):
        arr, sizes = Arrangement(), Sizes()
        checked(self.lib.quactlize_ppu_kpack_canonical_arrangement_v1(q, C.byref(arr)), 'arrangement')
        checked(self.lib.quactlize_ppu_kpack_sizes_for_arrangement_v1(n, k, experts, q, C.byref(arr), C.byref(sizes)),
                'sizes')
        rng = np.random.default_rng(seed)
        raw = raw_blocks(q, n * experts, k, rng)
        assert raw.nbytes == sizes.raw_bytes, (raw.nbytes, sizes.raw_bytes)
        src = self.sdk.allocate(raw.nbytes)
        checked(self.sdk.lib.hggcMemcpy(src, raw.ctypes.data, raw.nbytes, 1), 'H2D raw')
        planes = [self.sdk.allocate(getattr(sizes, x + '_bytes')) if getattr(sizes, x + '_bytes') else None
                  for x in ('low', 'high', 'units')]
        checked(self.lib.quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(
            src, *planes, n, k, experts, q, C.byref(arr), self.stream), 'pack')
        self.sdk.synchronize(self.stream)
        self.sdk.free(src)
        return arr, sizes, planes, raw

    def clone(self, sizes, planes):
        out = []
        for name, ptr in zip(('low', 'high', 'units'), planes):
            size = getattr(sizes, name + '_bytes')
            if not ptr:
                out.append(None)
                continue
            dst = self.sdk.allocate(size)
            checked(self.sdk.lib.hggcMemcpyAsync(dst, ptr, size, 3, self.stream), 'D2D')
            out.append(dst)
        self.sdk.synchronize(self.stream)
        return out

    def prepare_only(self, case):
        """Selection + JIT/module materialization only; planes are allocated, never read."""
        q, n, k, m = case['qtype'], case['n'], case['k'], case['m']
        experts = case.get('experts', 1)
        arr, sizes = Arrangement(), Sizes()
        checked(self.lib.quactlize_ppu_kpack_canonical_arrangement_v1(q, C.byref(arr)), 'arrangement')
        checked(self.lib.quactlize_ppu_kpack_sizes_for_arrangement_v1(n, k, experts, q, C.byref(arr), C.byref(sizes)),
                'sizes')
        planes = [self.sdk.allocate(getattr(sizes, x + '_bytes')) if getattr(sizes, x + '_bytes') else None
                  for x in ('low', 'high', 'units')]
        try:
            if case['kind'] == 'dense':
                p = Params(1, C.sizeof(Params), q, n, k, m, k, n, arr, sizes)
                plan, info = C.c_void_p(), Info(1, C.sizeof(Info))
                self.check(self.lib.quactlize_ppu_dense_prepare_v1(self.dense_context(case.get('compute', 0)),
                                                                   C.byref(p), *planes, C.byref(plan), C.byref(info)),
                           'dense')
            else:
                topk, channels = case['topk'], case['channels']
                p = moe_abi.Params(1, C.sizeof(moe_abi.Params), q, n, k, experts, m, topk, channels, 0,
                                   k, k * channels, topk, n, arr, sizes)
                plan, info = C.c_void_p(), moe_abi.Info(1, C.sizeof(moe_abi.Info))
                self.check(self.lib.quactlize_ppu_moe_matmul_prepare_v1(self.moe_context(), C.byref(p), *planes,
                                                                        C.byref(plan), C.byref(info)), 'moe')
            return dict(case=case, route=info.route, policy=info.policy, split=info.split,
                        compute_type=info.compute_type, config=info.config.decode(),
                        workspace=int(info.workspace_bytes))
        finally:
            for ptr in planes:
                if ptr:
                    self.sdk.free(ptr)

    def proto_launch(self, case, args, arr, sizes, planes, rng, seed, allocations):
        """Prototype reader entry qgemv_mma_run(Call*, compute, tile, warps, split); same ring/inputs."""
        from quactlize.execution.native import Call
        q, n, k, m, kind = case['qtype'], case['n'], case['k'], case['m'], case['kind']
        experts = case.get('experts', 1)
        proto = case['proto']
        lib = self.protos.get(proto['lib'])
        if lib is None:
            lib = C.CDLL(proto['lib'], mode=C.RTLD_LOCAL)
            lib.qgemv_mma_run.argtypes = [C.POINTER(Call)] + [C.c_int] * 4
            lib.qgemv_mma_run.restype = C.c_int
            self.protos[proto['lib']] = lib
        topk, channels = (case['topk'], case['channels']) if kind == 'moe' else (1, 1)
        rows = m * topk
        split = proto['split']
        workspace_bytes = rows * n * split * 4 if split > 1 else 0
        a_host = rng.standard_normal((m, channels, k) if kind == 'moe' else (m, k)).astype(np.float32)
        a = self.sdk.allocate(a_host.nbytes)
        checked(self.sdk.lib.hggcMemcpy(a, a_host.ctypes.data, a_host.nbytes, 1), 'A')
        out = self.sdk.allocate(rows * n * 4)
        ws = self.sdk.allocate(max(256, workspace_bytes))
        allocations += [a, out, ws]
        weight_bytes = sizes.low_bytes + sizes.high_bytes + sizes.units_bytes
        ids_host = None

        def call_for(pl, ids_ptr):
            c = Call()
            c.version, c.size, c.qtype, c.n, c.k, c.experts = 1, C.sizeof(Call), q, n, k, experts
            c.rows, c.mode, c.input_type, c.channels, c.topk = rows, 2 if kind == 'moe' else 0, 1, channels, topk
            c.a_row_stride, c.a_token_stride = k, k * channels
            c.ids_stride, c.out_row_stride = topk, n
            c.a, c.low, c.high, c.units = a, pl[0], pl[1], pl[2]
            c.ids, c.output, c.workspace, c.workspace_bytes, c.stream = ids_ptr, out, ws, workspace_bytes, self.stream
            return c
        if kind == 'dense':
            ring = self.ring_for(args, weight_bytes)
            copies = self.wcache['copies'][:ring]
            calls_list = [call_for(cp, None) for cp in copies]
            calls = max(ring, args.graph_calls)
        else:
            per_call = rows
            ring = max(args.graph_calls, math.ceil(experts / per_call))
            ids_host = np.empty((ring, m, topk), np.int32)
            for r in range(ring):
                base = (r * per_call) % experts
                ids_host[r] = ((base + np.arange(per_call)) % experts).reshape(m, topk)
            ids = self.sdk.allocate(ids_host.nbytes)
            checked(self.sdk.lib.hggcMemcpy(ids, ids_host.ctypes.data, ids_host.nbytes, 1), 'ids')
            allocations.append(ids)
            calls_list = [call_for(planes, ids + r * per_call * 4) for r in range(ring)]
            calls = ring
        args_t = (proto['compute'], proto['t'], proto['w'], split)

        def launch(i):
            return lib.qgemv_mma_run(C.byref(calls_list[i % ring]), *args_t)
        extra = dict(route=-1, policy=-1, split=split, compute_type=proto['compute'],
                     config=f"mma:t{proto['t']}:w{proto['w']}:s{split}:c{proto['compute']}", workspace=workspace_bytes,
                     ring=ring)
        self._keep = calls_list
        return launch, calls, ring, a_host, out, ids_host, extra

    def run_case(self, index, case, args):
        if self.packer_only and not case.get('proto'):
            raise ValueError('packer-only mode requires an explicit prototype case')
        if args.prepare_only:
            return self.prepare_only(case)
        q, n, k, m = case['qtype'], case['n'], case['k'], case['m']
        kind = case['kind']
        experts = case.get('experts', 1)
        seed = int(hashlib.sha256(f"{q}:{n}:{k}:{experts}".encode()).hexdigest()[:8], 16)
        w = self.weights(case, args, seed)
        arr, sizes, planes, raw = w['arr'], w['sizes'], w['planes'], w['raw']
        weight_bytes = sizes.low_bytes + sizes.high_bytes + sizes.units_bytes
        allocations = []
        rng = np.random.default_rng(seed + m)
        result = dict(case=case, weight_bytes=int(weight_bytes))
        if self.device is not None:
            result['device'] = self.device
        if case.get('proto'):
            launch, calls, ring, a_host, out, ids_host, extra = self.proto_launch(case, args, arr, sizes, planes, rng,
                                                                               seed, allocations)
            result.update(extra)
        elif kind == 'dense':
            ring = self.ring_for(args, weight_bytes)
            copies = self.wcache['copies'][:ring]
            ctx = self.dense_context(case.get('compute', 0))
            plans, infos = [], []
            for c in copies:
                p = Params(1, C.sizeof(Params), q, n, k, m, k, n, arr, sizes)
                plan, info = C.c_void_p(), Info(1, C.sizeof(Info))
                if case.get('candidate'):
                    cand = Candidate(1, C.sizeof(Candidate))
                    for key, value in case['candidate'].items():
                        setattr(cand, key, value.encode() if key == 'parent' else value)
                    self.check(self.lib.quactlize_ppu_dense_prepare_candidate_v1(
                        ctx, C.byref(p), *c, C.byref(cand), C.byref(plan), C.byref(info)), 'dense')
                else:
                    self.check(self.lib.quactlize_ppu_dense_prepare_v1(ctx, C.byref(p), *c, C.byref(plan),
                                                                       C.byref(info)), 'dense')
                plans.append(plan)
                infos.append(info)
            workspace_bytes = max(i.workspace_bytes for i in infos)
            a_host = rng.standard_normal((m, k)).astype(np.float32)
            a = self.sdk.allocate(a_host.nbytes)
            checked(self.sdk.lib.hggcMemcpy(a, a_host.ctypes.data, a_host.nbytes, 1), 'A')
            out = self.sdk.allocate(m * n * 4)
            ws = self.sdk.allocate(max(256, workspace_bytes))
            allocations += [a, out, ws]

            def launch(i):
                return self.lib.quactlize_ppu_dense_run_v1(plans[i % ring], a, out, ws, workspace_bytes, self.stream)
            calls = max(ring, args.graph_calls)
            info0 = infos[0]
            result.update(route=info0.route, policy=info0.policy, split=info0.split, compute_type=info0.compute_type,
                          config=info0.config.decode(), workspace=int(workspace_bytes), ring=ring)
            if len({i.config for i in infos}) != 1:
                raise RuntimeError('ring plans selected different configs')
        else:
            topk, channels = case['topk'], case['channels']
            ctx = self.moe_context()
            p = moe_abi.Params(1, C.sizeof(moe_abi.Params), q, n, k, experts, m, topk, channels, 0,
                               k, k * channels, topk, n, arr, sizes)
            plan, info = C.c_void_p(), moe_abi.Info(1, C.sizeof(moe_abi.Info))
            self.check(self.lib.quactlize_ppu_moe_matmul_prepare_v1(ctx, C.byref(p), *planes, C.byref(plan),
                                                                    C.byref(info)), 'moe')
            workspace_bytes = info.workspace_bytes
            a_host = rng.standard_normal((m, channels, k)).astype(np.float32)
            a = self.sdk.allocate(a_host.nbytes)
            checked(self.sdk.lib.hggcMemcpy(a, a_host.ctypes.data, a_host.nbytes, 1), 'A')
            out = self.sdk.allocate(m * topk * n * 4)
            ws = self.sdk.allocate(max(256, workspace_bytes))
            allocations += [a, out, ws]
            # Spread routing: distinct experts within the call, consecutive calls walk the expert set.
            per_call = m * topk
            ring = max(args.graph_calls, math.ceil(experts / per_call))
            ids_host = np.empty((ring, m, topk), np.int32)
            for r in range(ring):
                base = (r * per_call) % experts
                ids_host[r] = ((base + np.arange(per_call)) % experts).reshape(m, topk)
                if case.get('routing') == 'shuffle':
                    ids_host[r] = np.random.default_rng(seed + r).permutation(experts)[:per_call].reshape(m, topk)
            ids = self.sdk.allocate(ids_host.nbytes)
            checked(self.sdk.lib.hggcMemcpy(ids, ids_host.ctypes.data, ids_host.nbytes, 1), 'ids')
            allocations.append(ids)

            def launch(i):
                return self.lib.quactlize_ppu_moe_matmul_run_v1(plan, a, ids + (i % ring) * per_call * 4, out, ws,
                                                                workspace_bytes, self.stream)
            calls = ring
            result.update(route=info.route, policy=info.policy, split=info.split, compute_type=info.compute_type,
                          config=info.config.decode(), workspace=int(workspace_bytes), ring=ring,
                          source_tokens=info.source_tokens, prepare_algorithm=info.prepare_algorithm,
                          prepare_helper=info.prepare_helper)
        # Eager warmup (required before capture for some providers).
        for i in range(min(calls, 4)):
            rc = launch(i)
            if rc and case.get('proto'):
                raise RuntimeError(f'proto launch rc={rc}')
            self.check(rc, kind)
        self.sdk.synchronize(self.stream)
        graph, instance = C.c_void_p(), C.c_void_p()
        checked(self.sdk.lib.hggcStreamBeginCapture(self.stream, 0), 'capture')
        for i in range(calls):
            rc = launch(i)
            if rc:
                self.sdk.lib.hggcStreamEndCapture(self.stream, C.byref(graph))
                self.check(rc, kind)
        checked(self.sdk.lib.hggcStreamEndCapture(self.stream, C.byref(graph)), 'end capture')
        checked(self.sdk.lib.hggcGraphInstantiateWithFlags(C.byref(instance), graph, 0), 'instantiate')
        checked(self.sdk.lib.hggcGraphLaunch(instance, self.stream), 'warm graph')   # upload + first replay
        self.sdk.synchronize(self.stream)
        launches = max(1, math.ceil(args.samples / calls))
        if getattr(args, 'correctness_only', False):
            launches = 2
            for _ in range(launches):
                checked(self.sdk.lib.hggcGraphLaunch(instance, self.stream), 'correctness graph replay')
            self.sdk.synchronize(self.stream)
            result['scope'] = 'CORRECTNESS_AND_GRAPH_SMOKE_NOT_TIMING'
        else:
            events = [C.c_void_p(), C.c_void_p()]
            for e in events:
                checked(self.sdk.lib.hggcEventCreate(C.byref(e)), 'event')
            self.marker(index, 0)
            checked(self.sdk.lib.hggcEventRecord(events[0], self.stream), 'event')
            for _ in range(launches):
                checked(self.sdk.lib.hggcGraphLaunch(instance, self.stream), 'graph')
            checked(self.sdk.lib.hggcEventRecord(events[1], self.stream), 'event')
            self.marker(index, 1)
            self.sdk.synchronize(self.stream)
            elapsed = C.c_float()
            checked(self.sdk.lib.hggcEventElapsedTime(C.byref(elapsed), events[0], events[1]), 'elapsed')
            for e in events:
                self.sdk.lib.hggcEventDestroy(e)
            result['event_us_per_call'] = elapsed.value * 1000 / (calls * launches)
        result.update(graph_calls=calls, launches=launches, samples=calls * launches)
        if args.check:
            result['check'] = self.check_output(case, raw, a_host, out, ids_host[(calls * launches - 1) % ring]
                                                if kind == 'moe' else None)
        checked(self.sdk.lib.hggcGraphExecDestroy(instance), 'destroy exec')
        checked(self.sdk.lib.hggcGraphDestroy(graph), 'destroy graph')
        for p in allocations:
            self.sdk.free(p)
        return result

    @staticmethod
    def ring_for(args, weight_bytes):
        return max(2, min(args.max_ring, math.ceil(args.ring_mib * 2 ** 20 / weight_bytes)))

    def weights(self, case, args, seed):
        """Packed planes (+ dense ring copies), reused by consecutive cases of one weight shape."""
        q, n, k, experts = case['qtype'], case['n'], case['k'], case.get('experts', 1)
        key = (q, n, k, experts)
        if self.wcache and self.wcache['key'] == key:
            return self.wcache
        self.drop_weights()
        arr, sizes, planes, raw = self.packed(q, n, k, experts, seed)
        copies = [planes]
        if case['kind'] == 'dense':
            ring = self.ring_for(args, sizes.low_bytes + sizes.high_bytes + sizes.units_bytes)
            copies += [self.clone(sizes, planes) for _ in range(ring - 1)]
        self.wcache = dict(key=key, arr=arr, sizes=sizes, planes=planes, raw=raw, copies=copies)
        return self.wcache

    def drop_weights(self):
        if self.wcache:
            for cp in self.wcache['copies']:
                for p in cp:
                    if p:
                        self.sdk.free(p)
            self.wcache = None

    def check_output(self, case, raw, a_host, out, ids):
        """Independent GGUF decode (gguf-py) of sampled output columns."""
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from ci.support.fixture import matmul_reference, compare
        q, n, k, m = case['qtype'], case['n'], case['k'], case['m']
        compute = 'bf16' if case['kind'] == 'moe' or case.get('compute', 0) == 1 else 'f16'
        cols = np.unique(np.concatenate([np.arange(min(n, 64)), np.random.default_rng(7).integers(0, n, 448),
                                         np.arange(max(0, n - 64), n)]))
        if case['kind'] == 'dense':
            got = np.frombuffer(self.sdk.download(out, m * n * 4), np.float32).reshape(m, n)[:, cols]
            w = dequantize(raw[cols].reshape(-1), GGMLQuantizationType(q)).reshape(len(cols), k)
            gold, denom = matmul_reference(a_host, w, compute)
            return compare(got, gold, denom)
        topk, channels = case['topk'], case['channels']
        got = np.frombuffer(self.sdk.download(out, m * topk * n * 4), np.float32).reshape(m, topk, n)
        golds, denoms, gots = [], [], []
        for t in range(m):
            for s in range(topk):
                e = int(ids[t, s])
                w = dequantize(raw[e * n + cols].reshape(-1), GGMLQuantizationType(q)).reshape(len(cols), k)
                g, d = matmul_reference(a_host[t, s % channels][None], w, compute)
                golds.append(g[0]); denoms.append(d[0]); gots.append(got[t, s, cols])
        return compare(np.array(gots), np.array(golds), np.array(denoms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--cases', required=True, help='JSON list of cases')
    parser.add_argument('--out', required=True, help='JSONL result file')
    parser.add_argument('--ring-mib', type=float, default=768)
    parser.add_argument('--max-ring', type=int, default=192)
    parser.add_argument('--graph-calls', type=int, default=48)
    parser.add_argument('--samples', type=int, default=192)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--packer-only', action='store_true', help='explicit prototypes plus standalone packer only')
    parser.add_argument('--correctness-only', action='store_true', help='two graph replays and GGUF oracle, no timing')
    parser.add_argument('--prepare-only', action='store_true', help='selection/JIT prewarm, no kernels timed')
    parser.add_argument('--only', default='', help='comma separated case ids')
    args = parser.parse_args()
    if args.correctness_only:
        args.check = True
        if args.prepare_only:
            parser.error('--correctness-only cannot be combined with --prepare-only')
    cases = json.loads(Path(args.cases).read_text())
    only = set(filter(None, args.only.split(',')))
    bench = Bench(args.package, os.environ['PPU_SDK'], len(cases), args.packer_only)
    with open(args.out, 'a') as fh:
        for index, case in enumerate(cases):
            if only and case['id'] not in only:
                continue
            t0 = time.time()
            try:
                result = bench.run_case(index, case, args)
                result['status'] = 'OK'
            except Exception as exc:  # keep going; failures are recorded, never timed
                result = dict(case=case, status='FAIL', error=repr(exc))
            result.update(index=index, marker=[MARK_BASE + 64 * 2 * index, MARK_BASE + 64 * (2 * index + 1)],
                          wall_s=round(time.time() - t0, 2), package=str(Path(args.package).resolve()))
            fh.write(json.dumps(result) + '\n')
            fh.flush()
            print(json.dumps({k: result.get(k) for k in ('index', 'status', 'config', 'ring', 'samples', 'wall_s',
                                                          'event_us_per_call', 'error')} |
                             ({'err': result['check']['relative_l1_error']} if result.get('check') else {})),
                  flush=True)


if __name__ == '__main__':
    main()

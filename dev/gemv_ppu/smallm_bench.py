"""Device child for the Q4 small-M scan. Setup and blue controls are untimed."""
import ctypes as C
import hashlib
import math
from pathlib import Path
import statistics

import numpy as np

from dev.gemv_ppu import smallm as spec
from dev.gemv_ppu.run import load_library, read_fixture
from dev.gemv_ppu.run_bload import exact_bits
from quactlize.execution.native import arrangement
from quactlize.runtime.native import SDK, Module, Call, Recipe, Resources as QueryResources, checked
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_pack_gate import device_identity
from tools.profile_kpack_gpu_compact import AcuRange


def fixture(path):
    """Independent GGUF dot for eight distinct FP16-exact activation rows."""
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    n, k, data = read_fixture(path)
    rng = np.random.default_rng(0x482 + n + 31*k)
    a = (rng.standard_normal((8, k))*.2).astype('<f2')
    a[0] = data['a'].reshape(-1)
    raw = data['raw'].reshape(n, k//256, 144)
    golden, denom = np.empty((8, n), 'f8'), np.empty((8, n), 'f8')
    a64 = a.astype('f8')
    for begin in range(0, n, 256):
        end = min(begin+256, n)
        w = dequantize(raw[begin:end].reshape(-1), GGMLQuantizationType.Q4_K).reshape(end-begin, k).astype('f8')
        if not np.isfinite(w).all():
            raise ValueError('nonfinite official GGUF weight')
        golden[:, begin:end] = a64 @ w.T
        denom[:, begin:end] = np.abs(a64) @ np.abs(w).T
    if not np.isfinite(golden).all() or not np.all(denom > 0):
        raise ValueError('invalid independent multi-row oracle')
    for m in spec.MS:
        alias = np.broadcast_to(golden[0], (m, n))
        if conditioned(alias, golden[:m], denom[:m]) <= .005:
            raise ValueError('fixture cannot reject a repeated row-zero output')
    data.update(a=np.ascontiguousarray(a), golden=golden, denom=denom)
    return n, k, data


def conditioned(got, golden, denom):
    if got.shape != golden.shape or golden.shape != denom.shape or not np.isfinite(got).all():
        raise ValueError('nonfinite/mismatched output')
    return float(np.max(np.abs(got.astype('f8')-golden)/np.maximum(denom, 1e-30)))


def tc_reason(parent, k, split):
    if k % (parent['tk'] * split):
        return 'K_SPLIT_ALIGNMENT'
    if k // (parent['tk'] * split) < parent['stages']-1:
        return 'PIPELINE_FILL'
    return None


def tc_key(parent, split):
    return parent['symbol'] + ':s' + str(split)


class Bench:
    def __init__(self, args, manifest):
        self.args, self.manifest = args, manifest
        self.sdk = SDK(args.sdk)
        graph_bind(self.sdk)
        self.r = Resources(self.sdk)
        self.control_library, geometry = load_library(args, 'new')
        self.device = device_identity(self.sdk) | geometry
        self.n, self.k, self.data = fixture(args.fixture)
        n, k = self.n, self.k
        self.weight_bytes = n*k*9//16
        self.copies = max(2, math.ceil(2.25*geometry['l2_bytes']/self.weight_bytes))
        if self.copies*self.weight_bytes > 2**31:
            raise ValueError('rotation exceeds bounded allocation')
        self.host_low = np.ascontiguousarray(self.data['low']).view('u1').reshape(-1)
        self.host_units = np.ascontiguousarray(self.data['units']).view('u1').reshape(-1)
        if args.arm == 'reference':
            self.host_low = np.ascontiguousarray(self.data['raw']).view('u1').reshape(-1)
        low = self.r.alloc(self.host_low.nbytes*self.copies)
        units = self.r.alloc(self.host_units.nbytes*self.copies) if args.arm != 'reference' else 0
        self.weight_pointers = []
        for i in range(self.copies):
            lp = low+i*self.host_low.nbytes
            up = units+i*self.host_units.nbytes if units else 0
            checked(self.sdk.lib.hggcMemcpy(lp, self.host_low.ctypes.data, self.host_low.nbytes, 1), 'codes upload')
            if up:
                checked(self.sdk.lib.hggcMemcpy(up, self.host_units.ctypes.data, self.host_units.nbytes, 1), 'units upload')
            self.weight_pointers.append((lp, up))
        self.a = self.r.upload(self.data['a'])
        self.dtype = np.dtype('<f2' if args.arm == 'tc' else '<f4')
        self.capacity_bytes = 8*n*self.dtype.itemsize
        self.output_base = self.r.alloc(self.capacity_bytes+32)
        self.output = self.output_base+16
        self.modules, self.handles = {}, []
        self.workspace_base = 0
        self.workspace_bytes = 0
        self.module = None
        self.m = 0
        self.ids = {c.key:i for i,c in enumerate(spec.inventory(n,k))}
        if args.arm == 'kpack':
            self.library = C.CDLL(str(args.candidate/spec.payload(n,k)), mode=C.RTLD_LOCAL)
            self.launch = self.library.q4_smallm_run
            self.launch.argtypes = [C.c_int]*3 + [C.c_void_p]*5
            self.launch.restype = C.c_int
        elif args.arm == 'reference':
            self.library = C.CDLL(str(args.candidate/'libq4_smallm_reference.so'), mode=C.RTLD_LOCAL)
            self.launch = self.library.q4_smallm_reference
            self.launch.argtypes = [C.c_int]*6 + [C.c_void_p]*4
            self.launch.restype = C.c_int
        else:
            for record in manifest['modules']:
                module = Module(record | dict(path=str(args.candidate/record['path'])))
                self.modules[record['parent']['symbol']] = module
        self.sdk.synchronize(None)

    def release_tc(self):
        if self.handles or self.workspace_base:
            self.sdk.synchronize(self.r.stream)
        for handle in self.handles:
            self.module.destroy(handle)
        self.handles.clear()
        if self.workspace_base:
            self.sdk.free(self.workspace_base)
        self.workspace_base = self.workspace_bytes = 0

    def setup(self, m, key):
        self.m, self.key = m, key
        self.output_bytes = m*self.n*self.dtype.itemsize
        self.selection = None
        if self.args.arm != 'tc':
            return None
        self.release_tc()
        symbol, split = key.rsplit(':s', 1)
        self.module = self.modules[symbol]
        p = self.module.record['parent']
        split = int(split)
        reason = tc_reason(p, self.k, split)
        if reason:
            return reason
        device = self.module.device_identity()
        if (device['device'], device['ordinal']) != (self.device['name'], self.device['ordinal']):
            raise ValueError('TC device differs from SIMT probe')
        lp, up = self.weight_pointers[0]
        base = dict(version=1, size=C.sizeof(Call), m=m, n=self.n, k=self.k, experts=1,
                    group_size=32, device=device['ordinal'], compute_units=device['compute_units'],
                    mapping_id=arrangement(12).mapping_id, a=self.a, low=lp, metadata=up,
                    output=self.output, stream=self.r.stream.value)
        self.tc_recipe = Recipe(1, C.sizeof(Recipe), 0, split, 0)
        call = Call(**base)
        resource = QueryResources()
        checked(self.module.query(C.byref(call), C.byref(self.tc_recipe), C.byref(resource)), 'TC resource query')
        self.workspace_bytes = resource.workspace_bytes
        self.workspace_base = self.sdk.allocate(self.workspace_bytes+256)
        self.r.fill(self.workspace_base, 0xff, self.workspace_bytes+256)
        self.calls = []
        for lp, up in self.weight_pointers:
            call = Call(**(base | dict(low=lp, metadata=up, workspace=self.workspace_base+128,
                                     workspace_bytes=self.workspace_bytes)))
            handle = C.c_void_p()
            checked(self.module.prepare(C.byref(call), C.byref(self.tc_recipe), C.byref(handle)), 'TC prepare')
            self.calls.append(call)
            self.handles.append(handle)
        self.sdk.synchronize(self.r.stream)
        policy = next(r for r in self.manifest['tc_selection'] if r['request'] == [12,0,m,self.n,self.k,1,m])
        self.selection = dict(parent=p, build_key=self.module.record['key'], algorithm='ORDINARY',
            split=split, grid=0, shared_bytes=resource.shared_bytes, workspace_bytes=self.workspace_bytes,
            policy_kind=policy['policy'], is_current_policy=key==tc_key({'symbol':policy['parent']}, policy['split']),
            current_policy_key=tc_key({'symbol':policy['parent']}, policy['split']),
            device_reported_compute_units=device['compute_units'])
        return None

    def invoke(self, index=0, control=False):
        lp, up = self.weight_pointers[index % self.copies]
        if self.args.arm == 'kpack':
            return self.launch(self.ids[self.key], self.m, int(control), self.a, lp, up, self.output, self.r.stream)
        if self.args.arm == 'reference':
            cfg = tuple(map(int, self.key.split('-')))
            return self.launch(*cfg, self.m, self.n, self.k, self.a, lp, self.output, self.r.stream)
        if control:
            raise ValueError('no CPU loop control in TC timing')
        return self.module.run(self.handles[index % self.copies], self.r.stream)

    def output_data(self):
        self.sdk.synchronize(self.r.stream)
        raw = self.sdk.download(self.output_base, self.capacity_bytes+32)
        if raw[:16] != b'\xff'*16 or raw[16+self.output_bytes:] != b'\xff'*(self.capacity_bytes-self.output_bytes+16):
            raise ValueError('row/output guard changed')
        return raw[16:16+self.output_bytes]

    def error(self):
        raw = self.output_data()
        got = np.frombuffer(raw, self.dtype).reshape(self.m, self.n)
        return conditioned(got, self.data['golden'][:self.m], self.data['denom'][:self.m])

    def immutable_check(self):
        """Actual admitted M1 image, row zero; never part of event timing."""
        if self.key != spec.selected(self.n, self.k).key:
            return None
        c = spec.selected(self.n, self.k)
        record = spec.closure()[self.n, self.k]
        root = spec.ROOT/'prebuilt/ppu0010'/record['package']
        n, k = self.n, self.k
        if c.family == 'meta':
            lib = C.CDLL(str(root/spec.small_latency.payload(n,k,'meta')), mode=C.RTLD_LOCAL)
            fn = getattr(lib, f'q4_latency_run_{n}_{k}')
            args = (0,*c.recipe,0,0)
        elif c.family == 'medium':
            lib = C.CDLL(str(root/spec.medium_refine.PAYLOAD), mode=C.RTLD_LOCAL)
            fn, args = lib.q4_medium_run, (0,*c.recipe)
        else:
            followup = (n,k) in spec.reader_followup.SHAPES
            name = spec.reader_followup.payload(n,k) if followup else spec.reader_reuse.payload(n,k)
            lib = C.CDLL(str(root/name), mode=C.RTLD_LOCAL)
            fn = getattr(lib, f'q4_{"followup" if followup else "reader"}_run_{n}_{k}')
            args = c.recipe
        fn.argtypes, fn.restype = [C.c_int]*len(args)+[C.c_void_p]*5, C.c_int
        self.r.fill(self.output_base, 0xff, self.capacity_bytes+32)
        checked(fn(*args,self.a,*self.weight_pointers[0],self.output,self.r.stream), 'immutable M1 reader')
        self.sdk.synchronize(self.r.stream)
        raw = self.sdk.download(self.output,self.n*4)
        return raw

    def measure(self, m, key, samples, profile=False):
        reason = self.setup(m,key)
        common = dict(arm=self.args.arm, shape=[m,self.n,self.k], key=key)
        if reason:
            return dict(common, status='STRUCTURAL', reason=reason, samples_us=[], median_us=None)
        try:
            blue_hash = immutable_hash = None
            if self.args.arm == 'kpack':
                immutable = self.immutable_check()
                self.r.fill(self.output_base, 0xff, self.capacity_bytes+32)
                checked(self.invoke(control=True), 'untimed unchanged per-row bodies')
                if self.error() >= .005:
                    raise ValueError('unchanged row bodies fail independent GGUF')
                blue = self.output_data()
                if immutable is not None:
                    immutable_hash = exact_bits(immutable, blue[:self.n*4], 'original M1 image')
            self.r.fill(self.output_base, 0xff, self.capacity_bytes+32)
            checked(self.invoke(), 'multirow correctness launch')
            error = self.error()
            if error >= .005:
                raise ValueError(f'independent GGUF dot: {error:.8g}')
            original = self.output_data()
            if self.args.arm == 'kpack':
                blue_hash = exact_bits(blue, original, 'multirow versus original FP32 row bodies')
            fault = self.host_low.copy()
            if self.args.arm == 'reference':
                fault.reshape(-1,144)[:,16:] = 0
            else:
                fault.fill(0)
            lp = self.weight_pointers[0][0]
            checked(self.sdk.lib.hggcMemcpy(lp,fault.ctypes.data,fault.nbytes,1), 'zero codes')
            self.sdk.synchronize(None)
            checked(self.invoke(), 'zero-code launch')
            if self.error() <= .005:
                raise ValueError('zero-code negative escaped')
            checked(self.sdk.lib.hggcMemcpy(lp,self.host_low.ctypes.data,self.host_low.nbytes,1), 'restore codes')
            self.r.fill(self.a,0,self.data['a'].nbytes)
            self.sdk.synchronize(None)
            checked(self.invoke(), 'zero-A launch')
            if not np.all(np.frombuffer(self.output_data(),self.dtype)==0):
                raise ValueError('zero-A check failed')
            checked(self.sdk.lib.hggcMemcpy(self.a,self.data['a'].ctypes.data,self.data['a'].nbytes,1), 'restore A')
            self.sdk.synchronize(None)
            for _ in range(3):
                for i in range(self.copies):
                    checked(self.invoke(i), 'full-ring warmup')
            self.sdk.synchronize(self.r.stream)
            calls = max(2,math.ceil(32/self.copies))*self.copies
            times = []
            if profile:
                with AcuRange(self.sdk):
                    checked(self.invoke(), 'profile complete call')
                    self.sdk.synchronize(self.r.stream)
            else:
                counter = 0
                def launch():
                    nonlocal counter
                    rc = self.invoke(counter)
                    counter += 1
                    return rc
                graph = Replay(self.sdk,self.r.stream,launch,calls)
                try:
                    self.r.samples(graph,3)  # graph upload and first use excluded
                    times = [v/calls for v in self.r.samples(graph,samples)]
                finally:
                    graph.close()
            error = max(error,self.error())
            if self.output_data() != original or error >= .005:
                raise ValueError('post-replay output is not deterministic/correct')
            if self.workspace_base:
                raw = self.sdk.download(self.workspace_base,self.workspace_bytes+256)
                if raw[:128] != b'\xff'*128 or raw[-128:] != b'\xff'*128:
                    raise ValueError('TC workspace guard changed')
            c = spec.lookup(self.n,self.k,key) if self.args.arm == 'kpack' else None
            return dict(common,status='PASS',error=error,zero_code_negative='PASS',zero_a_check='PASS',
                output_guard='PASS',row_alias_negative='PASS',replay_check='PASS',
                row_alias_negative_scope='HOST_ORACLE_PLANT',
                matched_fp32_sha256=blue_hash,immutable_m1_sha256=immutable_hash,
                input_sha256=hashlib.sha256(self.data['a'][:m].tobytes()).hexdigest(),
                output_type='F16' if self.args.arm=='tc' else 'F32',accumulator='F32',
                weight_arithmetic=c.arithmetic if c else 'FQ_TC_F16_RECONSTRUCTION' if self.args.arm=='tc' else 'PER_WEIGHT_FP16',
                storage='RAW_GGUF' if self.args.arm=='reference' else 'CANONICAL_KPACK4',
                selection=self.selection,device=self.device,copies=self.copies,weight_bytes=self.weight_bytes,
                calls_per_graph=calls,samples_us=times,median_us=statistics.median(times) if times else None,
                timing_scope='RESIDENT_FULL_CALL_INCLUDING_TC_REDUCER_NO_HOST_SETUP_OR_JIT',
                cache_scope='ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2',
                multirow_scope='ONE_LAUNCH_ROW_GRID' if c else 'TC_COMPLETE_CALL' if self.args.arm=='tc' else 'ONE_LAUNCH_ROW_GRID',
                geometry=c.geometry(self.n,self.k,m) if c else None,
                access_models=[spec.access(c,self.n,self.k,m,dict(A=self.a%128,B=lp%128,metadata=up%128))
                    for lp,up in self.weight_pointers[:1]] if c else [])
        finally:
            self.release_tc()

    def close(self):
        self.release_tc()
        self.r.close()

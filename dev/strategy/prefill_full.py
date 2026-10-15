"""Actual active-expert dequant and installed BF16 providers, separate timers."""
import ctypes as C
import math

import numpy as np

from dev.bf16_compute.fixture import raw_weight, bf16_bits
from dev.smallm_closure.bench import Buffer
from dev.strategy.isolated import Captured
from quactlize.dequant.native import Call, bind
from quactlize.runtime.native import checked
from tools.kpack_bf16_providers import Cublas, DeepGemm, loaded_images
from tools.run_kpack_gemv_gate import Resources


class Full:
    def __init__(self, b, bundle, l2_bytes, sdk_path, *, workspace_bytes=0):
        # Explicit opt-in until the bounded Box capture gate admits it. Do not
        # change an active campaign's provider selection/workspace implicitly.
        if type(workspace_bytes) is not int or workspace_bytes < 0:
            raise ValueError('workspace bytes must be a nonnegative integer')
        import torch
        self.torch, self.b = torch, b
        self.r = Resources(b.sdk)
        self.provider = None
        self.workspace = None
        self.tensors, self.backs, self.calls = [], [], []
        self.weights = []
        try:
            self.stream = torch.cuda.ExternalStream(b.r.stream.value)
            self.library, self.dequant, _ = bind(bundle)
            self.indexed = b.helpers.lib.quactlize_kpack_dequant_indexed_v1
            self.indexed.argtypes = [C.POINTER(Call), C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
            self.indexed.restype = C.c_int
            w = b.w
            minimum_active = b.ring.receipt['minimum_active']
            reads = minimum_active*w.n*w.k*2
            self.copies = max(2, math.ceil(2.25*l2_bytes/reads))
            self.receipt = dict(kind='full', copies=self.copies, active_ring_bytes=reads*self.copies,
                minimum_active=minimum_active, row_maps='RESIDENT_GPU_INPUTS', dequant_in_provider_timer=False,
                scope='ACTIVE_DEQUANT_AND_BF16_PROVIDER_COMPONENTS_NOT_MODEL_E2E')
            def tensor(shape):
                count = math.prod(shape)
                with torch.cuda.stream(self.stream):
                    back = torch.empty(count+128, dtype=torch.bfloat16, device='cuda')
                b.r.fill(back.data_ptr(), 0xa5, (count+128)*2)
                out = back[64:-64].reshape(shape)
                self.backs.append(back); self.tensors.append(out)
                return out
            self.a = tensor((b.rows, w.k))
            self.out = tensor((b.rows, w.n))
            self.ids = Buffer(self.r, w.experts*4)
            self.count = Buffer(self.r, 4)
            with torch.cuda.stream(self.stream):
                self.rows = torch.empty(w.experts, dtype=torch.int32, device='cuda')
                self.indices = torch.empty(b.rows, dtype=torch.int32, device='cuda')
            self.update()
            for _ in range(self.copies):
                self.weights.append(tensor((w.experts, w.n, w.k)))
            self.provider = Cublas(sdk_path, b.r.stream) if w.experts == 1 else DeepGemm()
            self.receipt['provider'] = self.provider.identity
            self.receipt['workspace'] = dict(mode='PROVIDER_DEFAULT')
            if w.experts == 1 and workspace_bytes:
                self.workspace = ResidentWorkspace(self.r, workspace_bytes)
                self.receipt['workspace'] = self.provider.set_workspace(
                    self.workspace.ptr, self.workspace.size)
        except BaseException:
            self.close(); raise

    def update(self):
        b = self.b
        b.rt.sync()
        values = np.ascontiguousarray(b.route['sorted_owners'])
        checked(b.sdk.lib.hggcMemcpy(self.indices.data_ptr(), values.ctypes.data, values.nbytes, 1), 'provider indices')
        if b.w.experts == 1:
            self.ids.upload(np.zeros(1, dtype='i4')); self.count.upload(np.ones(1, dtype='i4'))
            values = b.route['counts']
            checked(b.sdk.lib.hggcMemcpy(self.rows.data_ptr(), values.ctypes.data, values.nbytes, 1), 'dense row count')
        else:
            checked(self.active(), 'active expert directory setup')
        b.rt.sync()

    def active(self):
        b = self.b
        return b.helpers.active(b.offsets.ptr, self.rows.data_ptr(), self.ids.ptr, self.count.ptr,
                                b.error.ptr, b.rows, b.r.stream)

    def dq(self, copy=0, config=4, target=0):
        b, w = self.b, self.b.w
        plane = b.ring.planes[copy]
        c = Call(1, C.sizeof(Call), w.qtype, w.n, w.k, w.experts, 1, config,
            plane['low'], plane['high'], plane['units'],
            b.ring.w.planes['low'].nbytes, b.ring.w.planes['high'].nbytes, b.ring.w.planes['units'].nbytes,
            self.weights[target].data_ptr(), None, w.experts*w.n*w.k*2, b.r.stream.value)
        if w.operation == 'indexed':
            return self.indexed(C.byref(c), C.byref(b.arr), self.ids.ptr, self.count.ptr, b.error.ptr)
        return self.dequant(C.byref(c), C.byref(b.arr))

    def poison_weights(self, target=0):
        w = self.b.w
        self.b.r.fill(self.weights[target].data_ptr(), 0x7f, w.experts*w.n*w.k*2)
        self.b.rt.sync()

    def check_weights(self, target=0):
        """Every selected bit, including inactive-expert write guards.

        Recreate original GGUF chunks, not a CPU decoder of device K-pack.
        No entire E*N*K float32 golden is materialized.
        """
        b, w = self.b, self.b.w
        b.rt.sync()
        if int(np.frombuffer(b.sdk.download(b.error.ptr, 4), dtype='i4')[0]):
            raise ValueError('full dequant device selection status is not zero')
        active = set(map(int, b.route['active']))
        bad, count = 0, 0
        for pattern in range(min(w.experts, 17)):
            for first in range(0, w.n, 256):
                n = min(256, w.n-first)
                _, gold = raw_weight(w.qtype, n, w.k, 710+pattern*31+w.qtype*1009+first)
                bits = bf16_bits(gold)
                for expert in range(pattern, w.experts, 17):
                    offset = (expert*w.n+first)*w.k*2
                    got = np.frombuffer(b.sdk.download(self.weights[target].data_ptr()+offset,
                        n*w.k*2), dtype='<u2').reshape(n, w.k)
                    if expert in active:
                        # Both signed zeros are numerically zero; nonzero bits
                        # must exactly match the independent BF16 expansion.
                        bad += int(np.count_nonzero((got != bits) & (((got | bits) & 0x7fff) != 0)))
                    else:
                        bad += int(np.count_nonzero(got != 0x7f7f))
                    count += got.size
        if bad: raise ValueError(f'full BF16 GGUF oracle/inactive-write guard differs: {bad}/{count}')
        self.guards()
        return dict(bits=count, bad=bad, active_experts=len(active), inactive_experts_untouched=True)

    def setup(self, config):
        self.update()
        for i in range(self.copies):
            self.poison_weights(i)
            checked(self.dq(i % self.b.ring.copies, config, i), 'untimed full BF16 creation')
        self.b.rt.sync()

    def launch(self, copy=0):
        b = self.b
        if b.w.operation == 'indexed': checked(self.active(), 'provider active selection')
        checked(b.input(self.a.data_ptr(), True), 'provider BF16 input')
        with self.torch.cuda.stream(self.stream):
            self.provider(self.a, self.weights[copy][0] if b.w.experts == 1 else self.weights[copy],
                          self.out, self.indices, self.rows)
        return b.finish(self.out.data_ptr(), True)

    def capture(self):
        # Python JIT/provider allocations belong to the Torch graph pool.
        # Cloning without retaining that pool would leave dangling pointers.
        owner = self.torch.cuda.CUDAGraph()
        graph = C.c_void_p()
        self.b.rt.sync()
        try:
            with self.torch.cuda.graph(owner, stream=self.stream):
                for i in range(self.copies): checked(self.launch(i), 'provider capture')
                checked(self.b.helpers.clone(self.b.r.stream, C.byref(graph)), 'clone provider graph during capture')
        except BaseException:
            if graph.value: self.b.rt.GraphDestroy(graph)
            raise
        return Captured(self.b.rt, [None]*self.copies, owned_graph=graph, allocation_owner=owner)

    def guards(self):
        self.b.rt.sync()
        for back in self.backs:
            size = back.numel()*2
            if (self.b.sdk.download(back.data_ptr(), 128) != b'\xa5'*128 or
                    self.b.sdk.download(back.data_ptr()+size-128, 128) != b'\xa5'*128):
                raise ValueError('full provider tensor guard changed')
        self.ids.guard(); self.count.guard()
        if self.workspace: self.workspace.guard()

    def close(self):
        self.b.rt.sync()
        if self.provider: self.provider.close(); self.provider = None
        self.weights.clear(); self.tensors.clear(); self.backs.clear()
        self.a = self.out = self.rows = self.indices = None
        self.r.close()
        self.torch.cuda.empty_cache()  # This child only, outside capture/timing.


class ResidentWorkspace:
    """Guarded 256-byte-aligned allocation, owned by the provider Resources."""
    def __init__(self, resources, size):
        if type(size) is not int or size <= 0:
            raise ValueError('positive resident workspace bytes required')
        self.r, self.size = resources, size
        base = resources.alloc(size+768)
        self.ptr = (base+511) & ~255
        resources.fill(self.ptr-256, 0xa5, size+512)
        resources.sdk.synchronize(resources.stream)

    def guard(self):
        for ptr in (self.ptr-256, self.ptr+self.size):
            if self.r.sdk.download(ptr, 256) != b'\xa5'*256:
                raise ValueError('cuBLAS resident workspace redzone differs')

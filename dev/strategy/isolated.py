"""Extract individual production kernels from a captured, untimed call.

No kernel rewrite and no producer+reducer interval. Each timed graph contains
only repetitions of ONE captured kernel stage, preserving the actual kernel
arguments. The original captured graph owns argument storage until close().
Unsupported/nonlinear graphs fail closed instead of losing hidden work.
"""
import ctypes as C
from functools import lru_cache
import math
import subprocess


@lru_cache(maxsize=4096)
def readable_symbol(symbol):
    # Cold rings can have thousands of copies of the same kernel. Demangling
    # each node by spawning a process would dominate the experiment wall time.
    return subprocess.check_output(['c++filt', symbol], text=True).strip()


class Dim3(C.Structure):
    _fields_ = [(k, C.c_uint) for k in ('x', 'y', 'z')]


class KernelParams(C.Structure):
    _fields_ = [('func', C.c_void_p), ('grid', Dim3), ('block', Dim3), ('shared', C.c_uint),
                ('parameters', C.c_void_p), ('extra', C.c_void_p)]


class EdgeData(C.Structure):
    _fields_ = [('from_port',C.c_ubyte),('to_port',C.c_ubyte),('type',C.c_ubyte),('reserved',C.c_ubyte*5)]


def default_edges(edges):
    if any(bytes(edge)!=b'\0'*C.sizeof(EdgeData) for edge in edges):
        raise ValueError('non-default/PDL graph dependency cannot be replaced by a serial component sum')


def check_sdk_graph_abi(sdk):
    """Host compile only, before any capture can write to ctypes storage."""
    from pathlib import Path
    from quactlize.runtime.tuning import digest
    sdk=Path(sdk).resolve()
    fields={'func':'func','grid':'gridDim','block':'blockDim','shared':'sharedMemBytes',
            'parameters':'kernelParams','extra':'extra'}
    source='''#include <hggc_runtime_api.h>
#include <cstddef>
#include <type_traits>
using Edges = hggcError_t (*)(hggcGraph_t,hggcGraphNode_t*,hggcGraphNode_t*,hggcGraphEdgeData*,size_t*);
static_assert(std::is_same<decltype(&hggcGraphGetEdges),Edges>::value,"GraphGetEdges ABI differs");
'''
    source+=f'static_assert(sizeof(hggcKernelNodeParams)=={C.sizeof(KernelParams)},"kernel ABI differs");\n'
    source+=f'static_assert(sizeof(hggcGraphEdgeData)=={C.sizeof(EdgeData)},"edge ABI differs");\n'
    for py,native in fields.items():
        source+=f'static_assert(offsetof(hggcKernelNodeParams,{native})=={getattr(KernelParams,py).offset},"kernel offset differs");\n'
    for field in ('from_port','to_port','type','reserved'):
        source+=f'static_assert(offsetof(hggcGraphEdgeData,{field})=={getattr(EdgeData,field).offset},"edge offset differs");\n'
    result=subprocess.run(['g++','-std=c++17','-x','c++','-fsyntax-only','-I'+str(sdk/'include'),'-'],
                          input=source,text=True,capture_output=True)
    if result.returncode:raise ValueError('SDK graph ABI validation failed before device work: '+result.stderr)
    return dict(scope='HOST_HEADER_ABI_CHECK_NOT_RUNTIME_GRAPH_SUPPORT',check_sha256=digest(source),edges_arguments=5)


def checked(rc, step):
    if rc:
        raise RuntimeError(f'{step}: runtime rc={rc}; no timing admitted')


def linear_order(nodes, edges):
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError('empty/duplicate captured nodes')
    incoming = {n: set() for n in nodes}
    outgoing = {n: set() for n in nodes}
    for a, b in edges:
        if a not in incoming or b not in incoming or a == b:
            raise ValueError('invalid graph edge')
        incoming[b].add(a)
        outgoing[a].add(b)
    result = []
    ready = [n for n, deps in incoming.items() if not deps]
    while len(result) < len(nodes):
        if len(ready) != 1:
            raise ValueError('nonlinear/cyclic graph cannot be decomposed as a serial cost sum')
        n = ready.pop()
        result.append(n)
        for other in outgoing[n]:
            incoming[other].remove(n)
            if not incoming[other]: ready.append(other)
    return result


def stage_groups(descriptions, copies):
    if copies < 1 or not descriptions or len(descriptions) % copies:
        raise ValueError('captured kernel count does not match weight ring')
    width = len(descriptions) // copies
    for copy in range(1, copies):
        if descriptions[copy * width:(copy + 1) * width] != descriptions[:width]:
            raise ValueError('kernel inventory/geometry changed across ring copies')
    return [[i + j * width for j in range(copies)] for i in range(width)]


class Captured:
    def __init__(self, rt, calls, *, owned_graph=None, allocation_owner=None):
        self.rt, self.graph, self.params, self.descriptions = rt, C.c_void_p(), [], []
        self.allocation_owner = allocation_owner
        signatures = {
            'GraphGetNodes': [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
            'GraphGetEdges': [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.POINTER(EdgeData), C.POINTER(C.c_size_t)],
            'GraphNodeGetType': [C.c_void_p, C.POINTER(C.c_int)],
            'GraphKernelNodeGetParams': [C.c_void_p, C.POINTER(KernelParams)],
            'FuncGetName': [C.POINTER(C.c_char_p), C.c_void_p],
            'GraphCreate': [C.POINTER(C.c_void_p), C.c_uint],
            'GraphAddKernelNode': [C.POINTER(C.c_void_p), C.c_void_p, C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(KernelParams)],
        }
        for name, args in signatures.items():
            try:
                fn = getattr(rt.lib, 'hggc' + name)
            except AttributeError as e:
                raise RuntimeError('SDK lacks isolated-profile graph entry: ' + name) from e
            fn.argtypes, fn.restype = args, C.c_int
            setattr(self, name, fn)
        if owned_graph is not None:
            # Caller transfers a cloned graph and keeps its provider allocator
            # alive. The clone owns parameters; allocation_owner owns storage.
            if not owned_graph.value or allocation_owner is None:
                raise ValueError('external graph needs a live allocation owner')
            self.graph = owned_graph
        else:
            rt.sync()
            checked(rt.StreamBeginCapture(rt.stream, 0), 'untimed capture begin')
            try:
                for call in calls:
                    checked(call(), 'untimed production capture')
            except BaseException:
                # End invalidated capture to leave the stream releasable.
                rt.StreamEndCapture(rt.stream, C.byref(self.graph))
                self.close()
                raise
            checked(rt.StreamEndCapture(rt.stream, C.byref(self.graph)), 'untimed capture end')
        try:
            count = C.c_size_t()
            checked(self.GraphGetNodes(self.graph, None, C.byref(count)), 'node count')
            nodes = (C.c_void_p * count.value)()
            checked(self.GraphGetNodes(self.graph, nodes, C.byref(count)), 'graph nodes')
            ec = C.c_size_t()
            checked(self.GraphGetEdges(self.graph, None, None, None, C.byref(ec)), 'edge count')
            starts, ends = (C.c_void_p * ec.value)(), (C.c_void_p * ec.value)()
            data=(EdgeData*ec.value)()
            checked(self.GraphGetEdges(self.graph, starts, ends, data, C.byref(ec)), 'graph edges')
            default_edges(data)
            for node in linear_order(list(nodes), list(zip(starts, ends))):
                kind, params, name = C.c_int(), KernelParams(), C.c_char_p()
                checked(self.GraphNodeGetType(node, C.byref(kind)), 'node type')
                if kind.value != 0:
                    raise ValueError(f'non-kernel graph node type={kind.value}; do not silently omit work')
                checked(self.GraphKernelNodeGetParams(node, C.byref(params)), 'kernel parameters')
                checked(self.FuncGetName(C.byref(name), params.func), 'kernel symbol')
                if not name.value:
                    raise ValueError('empty captured kernel symbol')
                symbol = name.value.decode()
                readable = readable_symbol(symbol)
                self.params.append(params)
                self.descriptions.append(dict(name=readable, symbol=symbol,
                    grid=[params.grid.x, params.grid.y, params.grid.z],
                    block=[params.block.x, params.block.y, params.block.z], shared_bytes=params.shared))
            self.groups = stage_groups(self.descriptions, len(calls))
        except BaseException:
            self.close()
            raise

    def isolate(self, stage, *, weight_ring, repeats=32, validation_only=False):
        indices = self.groups[stage]
        # Producer/expansion walks the original cold weight ring. Small
        # scratch-only helpers intentionally keep producer output resident.
        if not weight_ring:
            indices = indices[:1]
        if validation_only:
            return Isolated(self,indices[:1],timing=False)
        passes = max(2, (repeats + len(indices) - 1) // len(indices))
        return Isolated(self, indices * passes)

    def close(self):
        if self.graph.value:
            checked(self.rt.GraphDestroy(self.graph), 'destroy untimed capture')
            self.graph = C.c_void_p()
        self.allocation_owner = None


class Isolated:
    def __init__(self, captured, indices, timing=True):
        self.owner, self.rt, self.count = captured, captured.rt, len(indices)
        self.graph, self.instance = C.c_void_p(), C.c_void_p()
        self.events = []
        if not indices or len({captured.descriptions[i]['symbol'] for i in indices}) != 1:
            raise ValueError('timed graph must contain only one kernel component')
        try:
            checked(captured.GraphCreate(C.byref(self.graph), 0), 'isolated graph create')
            previous = None
            for i in indices:
                node = C.c_void_p()
                deps = (C.c_void_p * 1)(previous) if previous else None
                checked(captured.GraphAddKernelNode(C.byref(node), self.graph, deps,
                    int(previous is not None), C.byref(captured.params[i])), 'isolated add kernel')
                previous = node.value
            checked(self.rt.GraphInstantiateWithFlags(C.byref(self.instance), self.graph, 0), 'isolated instantiate')
            for _ in range(2 if timing else 0):
                event = C.c_void_p()
                checked(self.rt.EventCreate(C.byref(event)), 'component event create')
                self.events.append(event)
            for _ in range(2 if timing else 1):
                self.run()
            self.rt.sync()  # graph upload and first use are excluded
        except BaseException:
            self.close()
            raise

    def run(self):
        checked(self.rt.GraphLaunch(self.instance, self.rt.stream), 'isolated replay')

    def sample(self):
        checked(self.rt.EventRecord(self.events[0], self.rt.stream), 'component start event')
        self.run()
        checked(self.rt.EventRecord(self.events[1], self.rt.stream), 'component end event')
        checked(self.rt.EventSynchronize(self.events[1]), 'component event wait')
        value = C.c_float()
        checked(self.rt.EventElapsedTime(C.byref(value), *self.events), 'component interval')
        us = value.value * 1000 / self.count
        if not math.isfinite(us) or us <= 0:
            raise ValueError('invalid component duration')
        return us

    def close(self):
        self.rt.sync()
        if self.instance.value:
            checked(self.rt.GraphExecDestroy(self.instance), 'isolated instance destroy')
            self.instance = C.c_void_p()
        if self.graph.value:
            checked(self.rt.GraphDestroy(self.graph), 'isolated graph destroy')
            self.graph = C.c_void_p()
        for event in self.events:
            checked(self.rt.EventDestroy(event), 'component event destroy')
        self.events = []

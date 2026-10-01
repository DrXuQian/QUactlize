"""Offline admission bindings for the per-call BF16 composition ABI."""
import ctypes as C
import os
from pathlib import Path
import sys

from quactlize.dequant.native import Call as Weight
from quactlize.execution.native import Arrangement
from quactlize.dispatch.native import Request


class Call(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('weight', Weight)] + [
        (name, C.c_int32) for name in ('m', 'device', 'a_rows')] + [
        ('a', C.c_void_p), ('output', C.c_void_p), ('a_stride', C.c_int64), ('output_stride', C.c_int64)] + [
        (name, C.c_void_p) for name in ('src_rows', 'dst_rows', 'offsets', 'workspace')] + [
        ('workspace_bytes', C.c_uint64)]


class Options(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [
        (name, C.c_char_p) for name in ('sdk', 'python', 'deepgemm_helper')]


class Choice(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [
        (name, C.c_int32) for name in ('route', 'dequant_config', 'measured_tokens', 'predicted')] + [
        (name, C.c_double) for name in ('gemm_us', 'dequant_us')]


def selector(bundle):
    library = C.CDLL(str(Path(bundle)/'libquactlize_kpack_dispatch.so'), mode=C.RTLD_LOCAL)
    fn = library.quactlize_kpack_dispatch_prefill_v1
    fn.argtypes, fn.restype = [C.POINTER(Request), C.c_uint32, C.POINTER(Choice)], C.c_int

    def choose(request, mask=7):
        out = Choice()
        rc = fn(C.byref(request), mask, C.byref(out))
        if rc == 1:
            return None
        if rc:
            raise ValueError(f'prefill policy query rc={rc}')
        return out
    return choose


class Prefill:
    def __init__(self, bundle):
        self.bundle = Path(bundle).resolve(strict=True)
        self.library = C.CDLL(str(self.bundle/'libquactlize_ppu_prefill.so'), mode=C.RTLD_LOCAL)
        self.handles = []
        self.fn = {}
        types = {
            'query': ([C.POINTER(Call), C.POINTER(Arrangement), C.POINTER(C.c_uint64)], C.c_int),
            'prepare': ([C.POINTER(Call), C.POINTER(Arrangement), C.POINTER(Options), C.POINTER(C.c_void_p)], C.c_int),
            'run': ([C.c_void_p, C.c_void_p], C.c_int),
            'destroy': ([C.c_void_p], None),
            'error': ([], C.c_char_p),
            'device_status': ([C.c_void_p, C.POINTER(C.c_void_p)], C.c_int),
            'provider_image': ([C.c_void_p], C.c_char_p),
        }
        for name, (args, result) in types.items():
            fn = getattr(self.library, 'quactlize_kpack_prefill_' + name + '_v1')
            fn.argtypes, fn.restype = args, result
            self.fn[name] = fn

    def check(self, status, operation):
        if status:
            raise ValueError(f'prefill {operation} rc={status}: ' + self.fn['error']().decode())

    def query(self, call, arrangement):
        size = C.c_uint64()
        self.check(self.fn['query'](C.byref(call), C.byref(arrangement), C.byref(size)), 'query')
        return size.value

    def prepare(self, call, arrangement, sdk):
        options = Options(1, C.sizeof(Options), *map(os.fsencode, (
            Path(sdk).resolve(strict=True), sys.executable, self.bundle/'kpack_deepgemm_prewarm.py')))
        handle = C.c_void_p()
        self.check(self.fn['prepare'](C.byref(call), C.byref(arrangement), C.byref(options), C.byref(handle)), 'prepare')
        self.handles.append(handle)
        return handle

    def status_pointer(self, handle):
        pointer = C.c_void_p()
        self.check(self.fn['device_status'](handle, C.byref(pointer)), 'device status')
        return pointer.value

    def image(self, handle):
        return Path(os.fsdecode(self.fn['provider_image'](handle)))

    def run(self, handle, stream):
        self.check(self.fn['run'](handle, stream), 'run')
        return 0

    def close(self):
        for handle in reversed(self.handles):
            self.fn['destroy'](handle)
        self.handles.clear()

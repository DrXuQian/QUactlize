"""Unified public FFN ABI; correctness-test bindings, no selection logic."""
import ctypes as C
from ci.support.dense import Arrangement, Sizes


class Weight(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('qtype', 'n', 'k', 'experts')] + [
        ('arrangement', Arrangement), ('sizes', Sizes)] + [(n, C.c_void_p) for n in ('low', 'high', 'units')]


class Params(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [(n, C.c_int32) for n in
        ('kind', 'tokens', 'topk', 'channels', 'merged', 'reserved')] + [(n, C.c_int64) for n in
        ('input_row_stride', 'input_token_stride', 'ids_stride', 'weights_stride', 'output_stride')] + [
        (n, Weight) for n in ('gate', 'up', 'down')]


class ProjectionInfo(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('route', 'policy', 'split', 'compute_type')] + [('config', C.c_char * 192)]


class Info(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('workspace_bytes', C.c_uint64)] + [
        (n, C.c_int32) for n in ('prepare_stage', 'prepare_helper', 'finish_helper', 'reserved')] + [
        ('projections', ProjectionInfo * 3)]


class Public:
    def __init__(self, library):
        def bind(name, args, result=C.c_int):
            fn = getattr(library, 'quactlize_ppu_ffn_' + name)
            fn.argtypes, fn.restype = args, result
            return fn
        self.supported = bind('supported_v1', [C.POINTER(Params)])
        self.create = bind('create_v1', [C.c_int, C.POINTER(C.c_void_p)])
        self.prepare = bind('prepare_v1', [C.c_void_p, C.POINTER(Params), C.POINTER(C.c_void_p), C.POINTER(Info)])
        self.run = bind('run_v1', [C.c_void_p] * 6 + [C.c_uint64, C.c_void_p])
        self.destroy = bind('destroy_v1', [C.c_void_p], None)
        self.error = bind('error_v1', [], C.c_char_p)

    def check(self, rc):
        if rc:
            raise AssertionError(f'public FFN rc={rc}: {self.error().decode(errors="replace")}')

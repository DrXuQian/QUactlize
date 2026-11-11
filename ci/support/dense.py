"""Product correctness-test support; no performance campaign."""
import ctypes as C
from quactlize.packing.native import Arrangement, Sizes


class Params(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [(n, C.c_int32) for n in
        ('qtype', 'n', 'k', 'm')] + [(n, C.c_int64) for n in ('input_stride', 'output_stride')] + [
        ('arrangement', Arrangement), ('sizes', Sizes)]


class Info(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('workspace_bytes', C.c_uint64)] + [
        (n, C.c_int32) for n in ('route', 'policy', 'split', 'compute_type')] + [('config', C.c_char * 192)]


class Candidate(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32)] + [(n, C.c_int32) for n in (
        'kind', 'route', 'native_f16', 'legacy_adapters', 'dequant_config', 'tm', 'tn', 'tk', 'wm', 'wn',
        'stages', 'ap', 'delivery_n', 'split', 'grid_mode', 'grid_b', 'variant', 'columns', 'warps', 'values')] + [
        ('parent', C.c_char * 192)]


class Options(C.Structure):
    _fields_ = [('version', C.c_uint32), ('size', C.c_uint32), ('compute_type', C.c_int32), ('reserved', C.c_int32)]

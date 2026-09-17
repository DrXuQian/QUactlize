"""Explicit paired-N4 C ABI for tests and consumers; no implicit selection."""

import ctypes as C
from quactlize.execution.native import Call, SimtCallV2, Arrangement, Sizes


class Layout(C.Structure):
    _fields_ = [
        ("version", C.c_uint32),
        ("size", C.c_uint32),
        ("layout_id", C.c_uint64),
        ("packing", Arrangement),
    ]


class FusionCall(C.Structure):
    _fields_ = [
        ("version", C.c_uint32),
        ("size", C.c_uint32),
        ("input", SimtCallV2),
        ("output_type", C.c_int32),
        ("round_projection", C.c_int32),
    ]

    def __init__(self, call, compute=0, output_type=1, round_projection=0):
        super().__init__(
            1,
            C.sizeof(type(self)),
            SimtCallV2(call, compute),
            output_type,
            round_projection,
        )


class Config(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32)] + [
        (x, C.c_int32) for x in ("backend", "split", "tile_m", "warps")
    ]

    def __init__(self, backend=0, split=1, tile_m=0, warps=4):
        super().__init__(1, C.sizeof(type(self)), backend, split, tile_m, warps)


class MappedCall(C.Structure):
    _fields_ = [('version',C.c_uint32),('size',C.c_uint32),('call',FusionCall),
                ('input_rows',C.c_void_p),('status',C.c_void_p)]

    def __init__(self, call, input_rows=None, status=None):
        super().__init__(2,C.sizeof(type(self)),call,input_rows,status)


class Repack(C.Structure):
    _fields_ = [('version',C.c_uint32),('size',C.c_uint32)] + [
        (name,C.c_int32) for name in ('qtype','n','k','experts','merged')] + [
        (name,C.c_void_p) for name in ('gate_low','gate_units','up_low','up_units','low','units')]


class MoeBinding(C.Structure):
    _fields_ = [('version',C.c_uint32),('size',C.c_uint32)] + [
        (name,C.c_void_p) for name in ('low','high','units','workspace')] + [
        ('workspace_bytes',C.c_uint64),('layout',Layout),('config',Config)]


def integration_entries(library):
    """Additive entries; old numerical/performance packages remain loadable."""
    functions = {}
    signatures = {
        'repack': [C.POINTER(Repack),C.POINTER(Layout),C.c_void_p],
        'select': [C.c_int]*6+[C.POINTER(Config)],
        'run': [C.POINTER(MappedCall),C.POINTER(Config),C.POINTER(Layout)],
    }
    for name, signature in signatures.items():
        fn=getattr(library.lib,'quactlize_gate_up_'+name+('_v2' if name=='run' else '_v1'))
        fn.argtypes=signature;fn.restype=C.c_int;functions[name]=fn
    return functions


class Library:
    def __init__(self, path):
        self.lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        self.layout = self.lib.quactlize_gate_up_layout_v1
        self.layout.argtypes = [C.c_int, C.POINTER(Layout)]
        self.query = self.lib.quactlize_gate_up_query_v1
        self.query.argtypes = [
            C.POINTER(FusionCall),
            C.POINTER(Config),
            C.POINTER(Layout),
            C.POINTER(Sizes),
        ]
        self.run = self.lib.quactlize_gate_up_run_v1
        self.run.argtypes = self.query.argtypes[:-1]
        self.pack = self.lib.quactlize_gate_up_pack_v1
        self.pack.argtypes = [
            *[C.c_void_p] * 5,
            *[C.c_int] * 4,
            C.POINTER(Layout),
            C.c_void_p,
        ]
        for fn in (self.layout, self.query, self.run, self.pack):
            fn.restype = C.c_int

    def arrangement(self, q):
        out = Layout()
        rc = self.layout(q, C.byref(out))
        if rc:
            raise ValueError(f"paired arrangement rc={rc}")
        return out

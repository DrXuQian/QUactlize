"""Tool-only ctypes declarations. Inference uses the C API directly."""

import ctypes as C
from dataclasses import asdict
from reference import gguf_kpack as ref
from tools.run_kpack_pack_gate import Arrangement


class Call(C.Structure):
    _fields_ = (
        [(x, C.c_uint32) for x in ("version", "size")]
        + [
            (x, C.c_int32)
            for x in (
                "qtype",
                "n",
                "k",
                "experts",
                "rows",
                "mode",
                "input_type",
                "channels",
                "topk",
            )
        ]
        + [
            (x, C.c_int64)
            for x in ("a_row_stride", "a_token_stride", "ids_stride", "out_row_stride")
        ]
        + [
            (x, C.c_void_p)
            for x in (
                "a",
                "low",
                "high",
                "units",
                "offsets",
                "ids",
                "output",
                "workspace",
            )
        ]
        + [("workspace_bytes", C.c_uint64), ("stream", C.c_void_p)]
    )


class Config(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32)] + [
        (x, C.c_int32) for x in ("columns", "warps", "split")
    ]

    def __init__(self, columns=32, warps=4, split=1):
        super().__init__(1, C.sizeof(type(self)), columns, warps, split)


class Sizes(C.Structure):
    _fields_ = [
        (x, C.c_uint64)
        for x in (
            "low_bytes",
            "high_bytes",
            "units_bytes",
            "sf_plane_bytes",
            "workspace_bytes",
        )
    ]


def arrangement(q):
    return Arrangement(**asdict(ref.canonical_arrangement(q)))


def bind(lib):
    query = lib.quactlize_kpack_gemv_query_v1
    query.argtypes, query.restype = [
        C.POINTER(Call),
        C.POINTER(Config),
        C.POINTER(Arrangement),
        C.POINTER(Sizes),
    ], C.c_int
    run = lib.quactlize_kpack_gemv_run_v1
    run.argtypes, run.restype = [
        C.POINTER(Call),
        C.POINTER(Config),
        C.POINTER(Arrangement),
    ], C.c_int
    sf = lib.quactlize_kpack_sf_prepare_v1
    sf.argtypes = [C.c_int] * 4 + [
        C.c_void_p,
        C.c_uint64,
        C.c_void_p,
        C.c_void_p,
        C.c_uint64,
        C.POINTER(Arrangement),
        C.c_void_p,
    ]
    sf.restype = C.c_int
    return query, run, sf

"""Raw-pointer C boundary; caller owns device storage and stream lifetimes."""
from __future__ import annotations

import ctypes as C
import hashlib
import json
from pathlib import Path


class Arrangement(C.Structure):
    _fields_ = [(name, C.c_int32) for name in (
        "version", "layout", "bits", "high_bits", "artifact_tile_k",
        "transport_tile_k", "group_size", "reserved")] + [("mapping_id", C.c_uint64)]


class Versioned(C.Structure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.version = 1
        self.struct_bytes = C.sizeof(self)


class Problem(Versioned):
    _fields_ = [("version", C.c_uint32), ("struct_bytes", C.c_uint32)] + [
        (name, C.c_int32) for name in ("qtype", "operation", "route", "compute_type",
                                     "input_type", "output_type", "m", "n", "k", "experts", "max_rows")
    ] + [("arrangement", Arrangement)]


class Config(Versioned):
    _fields_ = [("version", C.c_uint32), ("struct_bytes", C.c_uint32)] + [
        (name, C.c_int32) for name in ("engine", "tile_m", "tile_n", "tile_k", "threads", "lanes_n")]


class Sizes(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("low_bytes", "high_bytes", "units_bytes", "scale_bytes", "zero_bytes")]


class Call(Versioned):
    _fields_ = [("version", C.c_uint32), ("struct_bytes", C.c_uint32),
                ("problem", Problem), ("config", Config)] + [
        (name, C.c_void_p) for name in ("a", "low", "high", "units", "scale", "zero", "offsets", "output", "stream")
    ] + [("a_stride", C.c_int64), ("output_stride", C.c_int64), ("device", C.c_int32), ("reserved", C.c_int32)]


class Backend:
    def __init__(self, bundle, *, allow_source_validation=False):
        bundle = Path(bundle).resolve()
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        if self.manifest.get("backend") != "ppu17-cutlass430" or self.manifest.get("schema") != 1:
            raise ValueError("not a PPU1.7 backend bundle")
        if self.manifest.get("target") != "native" and not allow_source_validation:
            raise ValueError("CUDA source-validation build is not a native PPU1.7 library")
        library = (bundle / self.manifest["library"]).resolve()
        if library.parent != bundle:
            raise ValueError("library must reside in its bundle")
        if hashlib.sha256(library.read_bytes()).hexdigest() != self.manifest["library_sha256"]:
            raise ValueError("library hash differs from manifest")
        self.library = C.CDLL(str(library), mode=C.RTLD_LOCAL)
        self.functions = {}
        for name, result, arguments in (
            ("last_error", C.c_char_p, []), ("build_target", C.c_char_p, []),
            ("canonical_arrangement", C.c_int, [C.c_int, C.POINTER(Arrangement)]),
            ("sizes_query", C.c_int, [C.POINTER(Problem), C.POINTER(Sizes)]),
            ("config_count", C.c_int, [C.POINTER(Problem)]),
            ("config_get", C.c_int, [C.POINTER(Problem), C.c_int, C.POINTER(Config)]),
            ("can_implement", C.c_int, [C.POINTER(Problem), C.POINTER(Config)]),
            ("launch", C.c_int, [C.POINTER(Call)]), ("expand_scales", C.c_int, [C.POINTER(Call)]),
            ("prepare", C.c_int, [C.POINTER(Call), C.POINTER(C.c_void_p)]),
            ("execute", C.c_int, [C.c_void_p]), ("destroy", None, [C.c_void_p]),
        ):
            function = getattr(self.library, "q17_" + name + "_v1")
            function.restype, function.argtypes = result, arguments
            self.functions[name] = function
        target = self.functions["build_target"]().decode()
        if (target == "ppu17-native") != (self.manifest.get("target") == "native"):
            raise ValueError("library target differs from manifest")

    def checked(self, rc):
        if rc:
            raise ValueError("PPU1.7: " + self.functions["last_error"]().decode())

    def problem(self, **kwargs):
        p = Problem(**kwargs)
        self.checked(self.functions["canonical_arrangement"](p.qtype, C.byref(p.arrangement)))
        self.sizes(p)
        return p

    def sizes(self, problem):
        result = Sizes()
        self.checked(self.functions["sizes_query"](C.byref(problem), C.byref(result)))
        return result

    def configs(self, problem):
        count = self.functions["config_count"](C.byref(problem))
        if count < 0:
            self.checked(-count)
        result = []
        for index in range(count):
            config = Config()
            rc = self.functions["config_get"](C.byref(problem), index, C.byref(config))
            if rc == 2:  # A structural grid limit may exclude an inventory row.
                continue
            self.checked(rc)
            result.append(config)
        return result

    def launch(self, call):
        self.checked(self.functions["launch"](C.byref(call)))

    def expand_scales(self, call):
        self.checked(self.functions["expand_scales"](C.byref(call)))

    def prepare(self, call):
        handle = C.c_void_p()
        self.checked(self.functions["prepare"](C.byref(call), C.byref(handle)))
        return Prepared(self, handle)


class Prepared:
    def __init__(self, backend, handle):
        self.backend, self.handle = backend, handle

    def execute(self):
        if not self.handle:
            raise ValueError("prepared plan has been closed")
        self.backend.checked(self.backend.functions["execute"](self.handle))

    def close(self):
        if self.handle:
            self.backend.functions["destroy"](self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

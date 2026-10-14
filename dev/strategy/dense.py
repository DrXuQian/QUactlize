"""Tool-only binding to the actual dense provider, not a second selector."""
import ctypes as C
from pathlib import Path


class Arrangement(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('version', 'layout', 'bits', 'high_bits',
        'artifact_tile_k', 'transport_tile_k', 'group_size', 'reserved')] + [('mapping_id', C.c_uint64)]


class Sizes(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('raw_bytes', 'low_bytes', 'high_bytes', 'units_bytes')]


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


def candidate(w, recipe):
    p = dict(recipe.parameters)
    c = Candidate(1, C.sizeof(Candidate))
    c.kind = {'tc': 0, 'simt': 1, 'q4': 2, 'full': 3}[recipe.kind]
    c.route = {'fq': 0, 'sf': 1, 'full': 3}[recipe.route]
    c.dequant_config = p.get('dequant_config', -1)
    if recipe.kind == 'tc':
        for field in ('tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'split', 'grid_mode', 'grid_b'):
            setattr(c, field, p.get(field, 0))
        c.delivery_n = p.get('dn', 0)
        c.parent = p['symbol'].encode()
        c.native_f16 = int(w.compute == 'f16' and w.m > 8)
    if recipe.kind == 'simt':
        for field in ('variant', 'columns', 'warps', 'values', 'split'):
            setattr(c, field, p[field])
    return c


class Provider:
    def __init__(self, package, bench, w, recipe=None):
        from dev.gate_up_perf.bench import Buffer
        self.b, self.w, self.recipe = bench, w, recipe
        self.library = C.CDLL(str(Path(package) / 'libquactlize_ppu.so'), mode=C.RTLD_LOCAL)
        self.context = C.c_void_p()
        self.handles, self.scratch, self.info = [], [], None
        def bind(name, args, result=C.c_int):
            f = getattr(self.library, 'quactlize_ppu_' + name)
            f.argtypes, f.restype = args, result
            return f
        self.error = bind('dense_error_v1', [], C.c_char_p)
        self.destroy = bind('dense_destroy_v1', [C.c_void_p], None)
        create = bind('dense_create_with_options_v1', [C.c_int, C.POINTER(Options), C.POINTER(C.c_void_p)])
        p = Params(1, C.sizeof(Params), w.qtype, w.n, w.k, w.m, w.k, w.n)
        canonical = bind('kpack_canonical_arrangement_v1', [C.c_int, C.POINTER(Arrangement)])
        sizes = bind('kpack_sizes_for_arrangement_v1', [C.c_int] * 4 + [C.POINTER(Arrangement), C.POINTER(Sizes)])
        self.check(canonical(w.qtype, C.byref(p.arrangement)))
        self.check(sizes(w.n, w.k, 1, w.qtype, C.byref(p.arrangement), C.byref(p.sizes)))
        if {n: getattr(p.sizes, n + '_bytes') for n in ('low', 'high', 'units')} != bench.ring.widths:
            raise ValueError('fixture plane sizes differ from actual public registry')
        options = Options(1, C.sizeof(Options), int(w.compute == 'bf16'), 0)
        self.check(create(0, C.byref(options), C.byref(self.context)))
        prepare_args = [C.c_void_p, C.POINTER(Params)] + [C.c_void_p] * 3
        if recipe is not None:
            cfg = candidate(w, recipe)
            prepare_args += [C.POINTER(Candidate)]
        prepare_args += [C.POINTER(C.c_void_p), C.POINTER(Info)]
        prepare = bind('dense_prepare_' + ('candidate_v1' if recipe else 'v1'), prepare_args)
        self.run = bind('dense_run_v1', [C.c_void_p] * 4 + [C.c_uint64, C.c_void_p])
        try:
            for i in range(bench.copies):
                handle, info = C.c_void_p(), Info(1, C.sizeof(Info))
                plane = bench.ring.at(i)
                args = [self.context, C.byref(p), plane['low'], plane['high'], plane['units']]
                if recipe is not None:
                    args += [C.byref(cfg)]
                self.check(prepare(*args, C.byref(handle), C.byref(info)))
                if self.info is not None and bytes(info) != bytes(self.info):
                    raise ValueError('production plan changed across weight copies')
                self.info = info
                if recipe and recipe.kind == 'q4':
                    r = dict(recipe.parameters)
                    want = 'q4:r{reader}:v{variant}:w{warps}:x{values}:c{columns}'.format(**r)
                    if info.config.decode() != want:
                        raise ValueError('historical Q4 recipe is not the current compiled selector; needs explicit carrier')
                self.handles.append(handle)
                # Separate per-copy derived weights: full-dequant BF16 GEMM
                # must not repeatedly read one hot expanded matrix.
                self.scratch.append(Buffer(bench.rt, info.workspace_bytes))
            self.receipt = dict(kind='original' if recipe is None else recipe.kind,
                                config=self.info.config.decode(), route=self.info.route,
                                split=self.info.split, compute=self.info.compute_type,
                                policy=self.info.policy, workspace_bytes=self.info.workspace_bytes)
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            raise ValueError('dense provider: ' + self.error().decode(errors='replace'))

    def launch(self, copy=0):
        return self.run(self.handles[copy], self.b.a.ptr, self.b.output.ptr,
                        self.scratch[copy].ptr, self.scratch[copy].size, self.b.rt.stream)

    def close(self):
        self.b.rt.sync()
        if self.context.value:
            self.destroy(self.context)
            self.context = C.c_void_p()

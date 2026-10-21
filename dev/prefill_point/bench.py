"""One fixed prefill point; unchanged production stage bodies, no tuning."""
import ctypes as C
from dataclasses import asdict
import statistics

from dev.strategy.journal import atomic_json
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.native import checked

WORKLOAD = Workload(12, 1024, 5120, 2048, storage='f32', compute='f16')
SF = Recipe.make('tc', 'sf', qtype=12, tm=64, tn=64, tk=64, wm=64, wn=32,
                 stages=3, ap=0, dn=64, persistent=-1, split=1, grid_mode=2,
                 grid_b=8, dequant_config=4,
                 symbol='sf_q12_a0_tm64_tn64_tk64_wm64_wn32_s3_bc0_ap0_dn64')


class Call(C.Structure):
    _fields_ = [('function', C.c_void_p), ('p', C.c_void_p*11),
                ('kind', C.c_int32), ('v', C.c_int32*6), ('stride', C.c_int64)]


def ptr(value):
    return C.cast(value, C.c_void_p).value


def call(fn, kind=0, pointers=(), integers=(), stride=0):
    result = Call()
    result.function, result.kind, result.stride = ptr(fn) if fn else None, kind, stride
    for i, v in enumerate(pointers):
        result.p[i] = v.value if isinstance(v, C.c_void_p) else v
    for i, v in enumerate(integers): result.v[i] = v
    return result


class Timer:
    def __init__(self, path):
        self.lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        if self.lib.point_call_size() != C.sizeof(Call): raise ValueError('point descriptor ABI drift')
        self.lib.point_error.restype = C.c_char_p
        self.lib.point_run.argtypes = [C.POINTER(Call), C.c_int, C.c_int]
        self.lib.point_samples.argtypes = [C.POINTER(Call), C.c_int, C.c_int, C.c_void_p,
            C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double)]
        self.lib.point_profile.argtypes = [C.POINTER(Call), C.c_int, C.c_int, C.c_void_p]

    def check(self, rc):
        if rc: raise RuntimeError('point timer: '+self.lib.point_error().decode(errors='replace'))

    def run(self, calls, stages, copy=0):
        self.check(self.lib.point_run(calls, stages, copy))

    def profile(self, calls, stages, copies, stream):
        self.check(self.lib.point_profile(calls, stages, copies, stream))

    def measure(self, calls, stages, copies, stream, rounds=3, samples=17):
        result = []
        for r in range(rounds):
            for rotate in ((0, 1) if r % 2 == 0 else (1, 0)):
                count = copies if rotate else 1
                for mode in range(3):
                    for stage in range(stages if mode == 0 else 1):
                        width = stages if mode == 1 else 1
                        data = (C.c_double*(max(samples, count)*width))()
                        self.check(self.lib.point_samples(calls, stages, count, stream,
                                                         mode, stage, samples, data))
                        result.append(dict(round=r, rotate=rotate, mode=mode, stage=stage,
                                           width=width, samples_us=list(data)))
        return result


def summarize(rows, names):
    """No mixed-method subtraction, fastest-sample selection or clock scaling."""
    rounds = sorted({r['round'] for r in rows})
    if len(rounds) < 3 or rounds != list(range(len(rounds))) or len(set(names)) != len(names):
        raise ValueError('point timer rounds/stage names differ')
    expected = {(r, rotate, mode, stage) for r in rounds for rotate in (0, 1)
                for mode in range(3) for stage in range(len(names) if mode == 0 else 1)}
    actual = [(r['round'], r['rotate'], r['mode'], r['stage']) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('point timer coverage incomplete or duplicated')
    for row in rows:
        width = len(names) if row['mode'] == 1 else 1
        if row['width'] != width or len(row['samples_us']) < 3*width or len(row['samples_us']) % width:
            raise ValueError('point timer sample width/count differs')
    result = {}
    for rotate, cache in [(0, 'repeated'), (1, 'rotating_weights')]:
        group = [r for r in rows if r['rotate'] == rotate]
        value = {}
        for mode, title in [(0, 'isolated_us'), (1, 'sequence_stage_us')]:
            phases = {}
            for i, name in enumerate(names):
                samples = []
                for r in group:
                    if r['mode'] != mode: continue
                    if mode == 0 and r['stage'] == i: samples += r['samples_us']
                    elif mode == 1: samples += r['samples_us'][i::len(names)]
                if not samples or any(not 0 < x < 1e9 for x in samples):
                    raise ValueError('missing/nonfinite point timer')
                phases[name] = statistics.median(samples)
            value[title] = phases
            value[title+'_sum'] = sum(phases.values())
        samples = [v for r in group if r['mode'] == 2 for v in r['samples_us']]
        if not samples or any(not 0 < x < 1e9 for x in samples): raise ValueError('missing total')
        value['complete_us'] = statistics.median(samples)
        value['complete_range_us'] = [min(samples), max(samples)]
        result[cache] = value
    return result


def qz(args, arm):
    from dev.strategy.cold_decode import Ring, Expansion
    from dev.strategy.prefill_fixture import Context, Helpers
    from dev.strategy.prefill_full import Full
    from dev.strategy.prefill_native import TensorCore
    from dev.strategy.resources import module_spec, probe_spec, NativeProbe, compiler
    from dev.strategy.space import resolved_grid
    from quactlize.runtime.native import SDK
    from quactlize.dequant.native import Call as DequantCall
    from dev.strategy.cublas_timing import NativeCalls
    from tools.kpack_bf16_providers import loaded_images

    sdk = SDK(args.sdk)
    timer = Timer(getattr(args, 'timer_library', args.output/'build/libpoint-timer.so'))
    reusable = getattr(args, 'reuse_build', None)
    payload = getattr(args, 'device_payload', None) or reusable or args.output/'build'
    trace_only = getattr(args, 'trace_only', False)
    ring = b = expansion = core = None
    keep = []
    try:
        w = getattr(args, 'workload', WORKLOAD)
        tc = arm != 'full-bf16'
        recipe = getattr(args, 'recipe', SF if tc else
                         Recipe.make('full', 'full', dequant_config=12, provider='bf16-default'))
        need_sf = tc and recipe.route == 'sf' and w.qtype != 8
        dequant_config = dict(recipe.parameters).get('dequant_config', 4 if tc else 12)
        if dequant_config < 0: dequant_config = 4 if w.qtype == 12 else 0
        ring = getattr(args, 'shared_ring', None) or Ring(sdk, w, args.l2_bytes)
        b = Context(sdk, ring, w, Helpers(payload/'helpers/libstrategy_prefill.so'))
        copies = ring.copies
        if tc:
            spec = module_spec(w, recipe)
            if getattr(args, 'module_record', None):
                record = args.module_record
            elif reusable:
                from dev.prefill_point.reuse import tc_record
                record = tc_record(args.sdk, payload, spec)
            else:
                record = compiler(spec, args.sdk, payload/'tc').build(spec['parent'])
            query = NativeProbe(spec, record).query(probe_spec(w, recipe,
                int(dict(recipe.parameters).get('grid_mode', 0) >= 2)))
            if query['status'] != 'READY': raise ValueError('fixed TC recipe unavailable')
            binding = dict(launch=resolved_grid(w, recipe, query['device']['compute_units'], query['occupancy']))
            if need_sf: expansion = Expansion(b, payload/'dequant')
            core = TensorCore(b, dict(recipe=asdict(recipe), spec=spec, binding=binding), record, query, expansion)
            a, out = core.a.ptr, core.out.ptr
            if expansion:
                for c in expansion.calls: c.config = dequant_config
            names = ['input']+(['expansion'] if need_sf else [])+['main', 'output']
        else:
            core = Full(b, payload/'dequant', args.l2_bytes, args.sdk, workspace_bytes=4 << 20)
            submit = NativeCalls(b, core)
            a, out = core.a.data_ptr(), core.out.data_ptr()
            names = ['error_reset', 'input', 'expansion', 'main', 'output']
        seq = []
        for i in range(copies):
            if not tc: seq.append(call(None, 3, [b.error.ptr, b.r.stream]))
            seq.append(call(b.helpers.io, 1, [b.a.ptr, a, None, b.error.ptr, b.r.stream],
                            [int(not tc), 1, w.m, w.k, w.m], w.k))
            if tc:
                if expansion:
                    seq.append(call(expansion.fn, pointers=[ptr(C.byref(expansion.calls[i])), ptr(C.byref(b.arr))]))
                seq.append(call(core.run_fn, pointers=[core.handles[i], b.r.stream]))
            else:
                plane = ring.planes[i]
                target = core.weights[i % core.copies].data_ptr()
                dq = DequantCall(1, C.sizeof(DequantCall), w.qtype, w.n, w.k, 1, 1, dequant_config,
                    plane['low'], plane['high'], plane['units'],
                    ring.w.planes['low'].nbytes, ring.w.planes['high'].nbytes, ring.w.planes['units'].nbytes,
                    target, None, w.n*w.k*2, b.r.stream.value)
                keep.append(dq)
                seq.append(call(core.dequant, pointers=[ptr(C.byref(dq)), ptr(C.byref(b.arr))]))
                seq.append(call(submit.run, 2, [submit.entry, core.provider.handle, a, target, out],
                                [w.m, w.n, w.k]))
            seq.append(call(b.helpers.io, 1, [out, b.out.ptr, None, b.error.ptr, b.r.stream],
                            [int(not tc), 0, w.m, w.n, w.m], w.n))
        calls = (Call*len(seq))(*seq)
        proof = []
        for repeat in (0, 1):
            b.update(repeat)
            for copy in (0, copies-1):
                timer.run(calls, len(names), copy); proof.append(b.check())
        if tc:
            if expansion: expansion.check(0); expansion.check(copies-1)
        else:
            for i in (0, copies-1):
                timer.run(calls, len(names), i); core.check_weights(i % core.copies)
        # An omitted producer must not pass: zero intermediate output with
        # changed A, then execute only finish; restore via the complete call.
        b.update(1)
        b.r.fill(out, 0, w.m*w.n*2); b.rt.sync()
        checked(b.finish(out, not tc), 'point omitted main')
        try: b.check()
        except ValueError as exc:
            if not str(exc).startswith('prefill oracle:'): raise
        else: raise ValueError('omitted producer not detected')
        b.update(0, zero=True); timer.run(calls, len(names)); b.check(zero=True)
        b.update(0)
        for i in range(copies): timer.run(calls, len(names), i)
        proof.append(b.check())
        if trace_only:
            timer.profile(calls, len(names), copies, b.r.stream)
            rows = []
        else:
            rows = timer.measure(calls, len(names), copies, b.r.stream, args.rounds, args.samples)
        proof.append(b.check())
        for buffer in ring.buffers+b.buffers: buffer.guard()
        if tc:
            for buffer in core.buffers+(expansion.buffers if expansion else []): buffer.guard()
        else: core.guards()
        value = dict(status='PASS', arm=arm, workload=asdict(w), correctness=proof,
            omitted_producer='DETECTED', zero_a='PASS', receipt=core.receipt,
            fixture=ring.receipt, stages=names, point_timings=rows,
            timing=('ASYS_GPU_ACTIVITIES_COMPLETE_CALLS_NO_INNER_EVENTS' if trace_only else
                    'CPP_ORDINARY_EVENTS_NO_GRAPH_NO_ASYS_NO_PYTHON_BETWEEN_EVENTS'),
            cache_note='ROTATING_QUANTIZED_WEIGHTS_RESIDENT_F32_A; SF_METADATA_NOT_SEPARATELY_COLD',
            sequence='EXPANSION_ON_EVERY_CALL_NOT_REUSED_ACROSS_CALLS',
            provider_workspace_bytes=0 if tc else 4 << 20,
            allocation='SHARED_CONVERTED_A_AND_C; EXPANDED_B_RING_AT_LEAST_2P25_L2_FOR_FULL',
            loaded_images=loaded_images(), summary=None if trace_only else summarize(rows, names),
            production_changed=False, model_admission=False)
        if trace_only:
            value['trace'] = trace_contract(copies)
        if getattr(args, 'recipe', None): value['recipe'] = asdict(recipe)
        atomic_json(args.output/'results'/(getattr(args, 'case_id', arm)+'.json'), value)
        return value
    finally:
        if core: core.close()
        if expansion: expansion.close()
        if b: b.close()
        if ring and not getattr(args, 'shared_ring', None): ring.close()


def trace_contract(copies):
    if not isinstance(copies, int) or copies < 2:
        raise ValueError('rotating trace needs at least two weight copies')
    ring = copies*((8+copies-1)//copies)
    return dict(copies=copies, warmup_calls=2*ring, captured_calls=4*ring,
                discarded_calls=ring, measured_calls=3*ring,
                no_inner_events=True, cuda_graph=False,
                scope='ASYS_KERNEL_DURATIONS_WITHIN_COMPLETE_ROTATING_CALLS')

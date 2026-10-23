#!/usr/bin/env python3
"""Bounded Q3/Q8 typed full-dequant A/B; no GEMM, selector or model changes."""
import argparse
from collections import Counter
import ctypes as C
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import FLAGS, LIBRARIES, HEADER_SUFFIXES, sha
from quactlize.runtime.tuning import digest

POINTS = ((11, 17408, 5120), (11, 5120, 17408), (8, 512, 2048),
          (8, 2048, 512), (8, 4096, 2048), (8, 8192, 2048),
          (8, 2048, 4096), (8, 1024, 5120))
VARIANTS = (0, 22, 23)
ORDER = (0, 22, 23, 23, 22, 0)
SCOPE = 'ISOLATED_TYPED_DEQUANT_AB_NOT_GEMM_OR_MODEL_LATENCY'


def selected(text):
    values = [int(x) for x in text.split(',')]
    if not values or len(set(values)) != len(values) or any(x not in range(len(POINTS)) for x in values):
        raise ValueError('points must be distinct indices in 0..7')
    return values


def plan(points, l2):
    return dict(schema='quactlize.dense-dequant-ab.v1', scope=SCOPE,
                points=[dict(index=i, q=POINTS[i][0], n=POINTS[i][1], k=POINTS[i][2]) for i in points],
                variants=list(VARIANTS), order=list(ORDER), rounds=3, samples_per_visit=5,
                correctness_storage=['F16', 'BF16'], timed_storage='F16',
                l2_bytes=l2, cache='INPUT_AND_OUTPUT_RINGS_EACH_EXCEED_2.25_L2',
                warmup='TWO_COMPLETE_GRAPHS_PER_VARIANT_EXCLUDED',
                timing='EVENT_AVERAGE_OVER_SINGLE_KERNEL_GRAPH_NODES_INCLUDES_NODE_GAPS',
                requested_clock_mhz=1700, actual_clock='ADVISORY_NOT_FAILURE',
                production_changed=False, performance_admitted=False)


def inspect(sdk, image, output):
    from tools.build_kpack_dequant import resource_usage
    text = subprocess.check_output([str(sdk/'bin/hgobjdump'), '--dump-isa', str(image)], text=True)
    resources = resource_usage(subprocess.check_output(
        [str(sdk/'bin/hgobjdump'), '--dump-resource-usage=all', str(image)], text=True))
    entries = list(re.finditer(r'Disassembly of section \.text\.kernel\.([^\n]+):', text))
    result = {}
    for i, match in enumerate(entries):
        body = text[match.end():entries[i+1].start() if i+1 < len(entries) else None]
        ops = Counter(re.findall(r'\t([a-z][\w.]+)\s', body))
        if 'dense_dequant_plane_pair' in match[1]:
            for prefix in ('vmem.ld.b32x4', 'tsm.ld.b32x4', 'vmem.st.b32x4'):
                if not any(op.startswith(prefix) for op in ops):
                    raise ValueError('candidate vector delivery scalarized: '+prefix)
            if resources[match[1]]['stack_bytes']:
                raise ValueError('candidate has a local stack allocation')
        result[match[1]] = dict(resources=resources[match[1]], operations=dict(ops),
                                static_instructions=sum(ops.values()))
    candidates = [s for s in result if 'dense_dequant_plane_pair' in s]
    if len(candidates) != 8 or set(result) != set(resources):
        raise ValueError('native specialization inventory differs')
    for q in (8, 11):
        for bf in (0, 1):
            for tk in (128, 256):
                if sum(f'ILi{q}ELb{bf}ELj{tk}E' in s for s in candidates) != 1:
                    raise ValueError('candidate format/storage/tile missing')
    atomic_json(output/'native.json', dict(scope='STATIC_NOT_DYNAMIC_COUNTS', kernels=result))


def build(a):
    output = a.output/'build'
    output.mkdir()
    sources = {str(p.relative_to(ROOT)): sha(p) for directory in
               ('quactlize/dequant', 'quactlize/prefill', 'quactlize/execution',
                'quactlize/include', 'third_party/actlize/include')
               for p in (ROOT/directory).rglob('*') if p.is_file() and
               (p.suffix in HEADER_SUFFIXES or p.name == 'dense.cu')}
    sources.update({str(p.relative_to(ROOT)): sha(p) for directory in ('quactlize', 'dev', 'tools', 'reference')
                    for p in (ROOT/directory).rglob('*.py') if p.is_file()})
    obj = output/'dense.o'
    command = [str(a.sdk/'bin/hgcc'), *FLAGS, '-lineinfo',
               '-I'+str(ROOT/'quactlize/include'), '-I'+str(ROOT/'third_party/actlize/include'),
               '-c', str(ROOT/'quactlize/dequant/dense.cu'), '-o', str(obj)]
    receipt = dict(sources=sources, compiler_sha256=sha(a.sdk/'bin/hgcc'), commands=[command],
                   source_head=os.environ.get('DENSE_DEQUANT_SOURCE_SHA', 'SEE_SOURCE_HASHES'),
                   actlize_head=os.environ.get('DENSE_DEQUANT_ACTLIZE_SHA', 'SEE_SOURCE_HASHES'),
                   runner_sha256=sha(__file__), compile_only=a.compile_only)
    atomic_json(output/'receipt.json', receipt)
    start = time.monotonic()
    with (output/'compile.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    inspect(a.sdk, obj, output)
    if not a.compile_only:
        library = output/'libquactlize_dense_dequant_probe.so'
        command = ['g++', '-shared', '-Wl,-Bsymbolic', str(obj), '-L'+str(a.sdk/'lib'),
                   *['-l'+name for name in LIBRARIES], '-o', str(library)]
        with (output/'link.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt.update(library=str(library), library_sha256=sha(library),
                       runtime={name: sha(a.sdk/'lib'/('lib'+name+'.so')) for name in LIBRARIES})
        receipt['commands'].append(command)
    if any(sha(ROOT/p) != value for p, value in sources.items()):
        raise ValueError('source changed during compilation')
    receipt.update(object_sha256=sha(obj), seconds=time.monotonic()-start)
    atomic_json(output/'receipt.json', receipt)
    return receipt


def verify_receipt(a):
    receipt = json.loads((a.output/'build/receipt.json').read_text())
    if (receipt['compile_only'] or sha(receipt['library']) != receipt['library_sha256'] or
            sha(a.sdk/'bin/hgcc') != receipt['compiler_sha256'] or
            any(sha(ROOT/p) != value for p, value in receipt['sources'].items()) or
            any(sha(a.sdk/'lib'/('lib'+name+'.so')) != value for name, value in receipt['runtime'].items())):
        raise ValueError('resume source, binary or SDK changed; retain old results and start a new run')
    return receipt


def validate_point(data, index, l2):
    if (data.get('status') != 'PASS' or data.get('point') != index or
            tuple(data.get(name) for name in ('q', 'n', 'k')) != POINTS[index] or
            data.get('scope') != SCOPE or data.get('production_changed') is not False or
            data.get('performance_admitted') is not False or data.get('negative_bad', 0) <= 0):
        raise ValueError('saved point identity/proof differs')
    for kind in ('reads', 'writes'):
        if data['traffic'][kind]*data['ring_copies'] < 2.25*l2:
            raise ValueError('saved point has an undersized weight/output ring')
    if data['calls_per_graph'] != 2*data['ring_copies']:
        raise ValueError('saved graph traversal differs')
    for variant in VARIANTS:
        key = str(variant)
        values = data['samples_us'][key]
        if (len(values) != 30 or any(not math.isfinite(v) or v <= 0 for v in values) or
                data['median_us'][key] != statistics.median(values) or
                data['round_medians_us'][key] != [statistics.median(values[i:i+10]) for i in (0, 10, 20)]):
            raise ValueError('saved timing samples are incomplete')
        for storage in ('0', '1'):
            proof = data['proof'][storage][key]
            if (proof['bad'] or proof['guard'] != 'PASS' or proof['compared'] != data['n']*data['k'] or
                    variant and not proof['control_raw_bits_equal']):
                raise ValueError('saved numerical proof incomplete')
    return data


def fixture(q, n, k):
    import numpy as np
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    from dev.bf16_compute.fixture import planes, bf16_bits
    from reference.gguf_kpack import SPECS
    block, width = (32, 34) if q == 8 else (256, SPECS[q].raw_bytes)
    rng = np.random.default_rng(71511+q+n+k)
    raw = rng.integers(0, 256, (n, k//block, width), dtype='u1')
    offset = 0 if q == 8 else SPECS[q].d_offset
    scales = rng.uniform(-.009, .009, (n, k//block)).astype('<f2')
    scales.view('<u2').flat[:8] = [0, 0x8000, 1, 0x8001, 0x3555, 0xb555, 0x400, 0x8400]
    raw[..., offset:offset+2] = scales.view('u1').reshape(n, k//block, 2)
    packed = {name: value for name, value in planes(raw, q).items() if name in ('low', 'high', 'units')}
    gold = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(n, k)
    if not np.isfinite(gold).all() or not np.any(gold):
        raise ValueError('invalid independent GGUF fixture')
    return packed, [gold.astype('<f2').view('<u2'), bf16_bits(gold)], dict(
        raw_sha256=hashlib.sha256(memoryview(raw)).hexdigest(),
        planes={name: hashlib.sha256(memoryview(value)).hexdigest() for name, value in packed.items()},
        oracle='INDEPENDENT_GGUF_FP32_AFFINE_THEN_ONE_RNE_F16_OR_BF16')


class Bench:
    def __init__(self, a):
        from quactlize.runtime.native import SDK, checked
        from quactlize.dequant.native import Call
        from quactlize.execution.native import Arrangement, arrangement
        from tools.run_kpack_gemv_gate import Resources
        from tools.run_kpack_grouped_device_gate import graph_bind
        self.q, self.n, self.k = POINTS[a.child]
        receipt = json.loads((a.output/'build/receipt.json').read_text())
        if sha(receipt['library']) != receipt['library_sha256'] or sha(__file__) != receipt['runner_sha256']:
            raise ValueError('probe binary/runner changed')
        self.sdk = SDK(a.sdk)
        from tools.probe_kpack_strategy import device_probe
        authority = json.loads((a.output/'authority.json').read_text())
        if device_probe(a.sdk) != authority['device']['device']:
            raise ValueError('child physical device/runtime differs')
        if any(sha(a.sdk/'lib'/('lib'+name+'.so')) != value for name, value in receipt['runtime'].items()):
            raise ValueError('child SDK runtime differs')
        graph_bind(self.sdk)
        self.r = Resources(self.sdk)
        self.graphs, self.calls = {}, []
        self.lib = C.CDLL(receipt['library'], mode=C.RTLD_LOCAL)
        self.fn = self.lib.quactlize_kpack_dequant_dense_v1
        self.fn.argtypes, self.fn.restype = [C.POINTER(Call), C.POINTER(Arrangement), C.c_int, C.c_int], C.c_int
        self.arr = arrangement(self.q)
        self.planes, self.gold, self.fixture = fixture(self.q, self.n, self.k)
        reads = sum(v.nbytes for v in self.planes.values())
        self.copies = 1 if a.profile_variant is not None else max(2, math.ceil(2.25*a.l2_bytes/reads))
        self.replays = 2*self.copies
        self.bytes, self.guard = self.n*self.k*2, 128
        self.stride = self.bytes+2*self.guard
        self.traffic = dict(reads=reads, writes=self.bytes, useful_bytes=reads+self.bytes,
                            input_ring_bytes=reads*self.copies, output_ring_bytes=self.bytes*self.copies)
        self.buffers = {}
        copy = self.sdk.lib.hggcMemcpyAsync
        copy.argtypes, copy.restype = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p], C.c_int
        for name, data in self.planes.items():
            if not data.nbytes:
                continue
            base = self.r.alloc(data.nbytes*self.copies)
            checked(self.sdk.lib.hggcMemcpy(base, data.ctypes.data, data.nbytes, 1), 'upload '+name)
            self.sdk.synchronize(None)
            for i in range(1, self.copies):
                checked(copy(base+i*data.nbytes, base, data.nbytes, 3, self.r.stream), 'initialize input ring')
            self.buffers[name] = base
        base = self.r.alloc(self.stride*self.copies)
        self.r.fill(base, 0xa5, self.stride*self.copies)
        for i in range(self.copies):
            self.calls.append(Call(version=1, size=C.sizeof(Call), qtype=self.q, n=self.n, k=self.k,
                experts=1, operation=1, config=0, output=base+i*self.stride+self.guard,
                output_bytes=self.bytes, stream=self.r.stream.value,
                low_bytes=self.planes['low'].nbytes, high_bytes=self.planes['high'].nbytes,
                unit_bytes=self.planes['units'].nbytes,
                **{name: p+i*self.planes[name].nbytes for name, p in self.buffers.items()}))
        self.sdk.synchronize(self.r.stream)

    def launch(self, variant, storage=0, index=0):
        return self.fn(C.byref(self.calls[index]), C.byref(self.arr), storage, variant)

    def read(self, index=0):
        import numpy as np
        self.sdk.synchronize(self.r.stream)
        raw = np.frombuffer(self.sdk.download(self.calls[index].output-self.guard, self.stride), dtype='u1')
        if not np.all(raw[:self.guard] == 0xa5) or not np.all(raw[-self.guard:] == 0xa5):
            raise ValueError('output guard corrupted')
        return raw[self.guard:-self.guard].view('<u2').reshape(self.n, self.k)

    def check(self, variant, storage, control=None):
        import numpy as np
        from quactlize.runtime.native import checked
        self.r.fill(self.calls[0].output, 0x7f, self.bytes)
        checked(self.launch(variant, storage), 'dequant correctness')
        got, want = self.read(), self.gold[storage]
        bad = (got != want) & (((got | want) & 0x7fff) != 0)
        if np.any(bad) or control is not None and not np.array_equal(got, control):
            raise ValueError(f'raw-bit oracle/control mismatch variant={variant} storage={storage} bad={int(bad.sum())}')
        return got.copy(), dict(bad=0, compared=got.size, guard='PASS',
            signed_zero_differences=int(np.count_nonzero(got != want)),
            output_sha256=hashlib.sha256(memoryview(got)).hexdigest(), control_raw_bits_equal=control is not None)

    def negative(self):
        import numpy as np
        from quactlize.runtime.native import checked
        self.r.fill(self.buffers['low'], 0, self.planes['low'].nbytes)
        checked(self.launch(22), 'blank low-plane negative')
        bad = int(np.count_nonzero(self.read() != self.gold[0]))
        if bad < self.n*self.k//100:
            raise ValueError('blank low plane escaped oracle')
        checked(self.sdk.lib.hggcMemcpy(self.buffers['low'], self.planes['low'].ctypes.data,
                                      self.planes['low'].nbytes, 1), 'restore fixture')
        self.sdk.synchronize(None)
        return bad

    def graph(self, variant):
        from tools.run_kpack_grouped_decode_probe import Replay
        from quactlize.runtime.native import checked
        if variant not in self.graphs:
            index = 0
            def launch():
                nonlocal index
                result = self.launch(variant, index=index % self.copies)
                index += 1
                return result
            self.graphs[variant] = Replay(self.sdk, self.r.stream, launch, self.replays)
            for _ in range(2):
                checked(self.graphs[variant](), 'excluded graph warmup')
            self.sdk.synchronize(self.r.stream)
        return self.graphs[variant]

    def close(self):
        self.sdk.synchronize(self.r.stream)
        for graph in self.graphs.values():
            graph.close()
        self.r.close()


def child(a):
    import numpy as np
    from quactlize.runtime.native import checked
    b = Bench(a)
    try:
        if a.profile_variant is not None:
            from tools.profile_kpack_gpu_compact import AcuRange
            control, _ = b.check(0, 0)
            _, proof = b.check(a.profile_variant, 0, control)
            checked(b.launch(a.profile_variant), 'direct profile warmup')
            b.sdk.synchronize(b.r.stream)
            with AcuRange(b.sdk):
                checked(b.launch(a.profile_variant), 'single isolated dequant profile')
                b.sdk.synchronize(b.r.stream)
            result = dict(status='PASS', point=a.child, profile_variant=a.profile_variant, proof=proof,
                          ring_copies=1, profiled_launches=1, timed_loops=0, production_changed=False)
        else:
            proofs = {}
            for storage in (0, 1):
                control, proof = b.check(0, storage)
                proofs[str(storage)] = {'0': proof}
                for v in VARIANTS[1:]:
                    _, proofs[str(storage)][str(v)] = b.check(v, storage, control)
            negative = b.negative()
            for v in VARIANTS:
                b.check(v, 0)
                b.graph(v)
            samples = {v: [] for v in VARIANTS}
            round_medians = {v: [] for v in VARIANTS}
            for round_index in range(3):
                current = {v: [] for v in VARIANTS}
                for v in ORDER:
                    current[v].extend(t/b.replays for t in b.r.samples(b.graph(v), 5))
                for v in VARIANTS:
                    samples[v].extend(current[v])
                    round_medians[v].append(statistics.median(current[v]))
                print(f'DENSE_DEQUANT_PROGRESS point={a.child} round={round_index+1}/3', flush=True)
            for v in VARIANTS:
                checked(b.graph(v)(), 'final changed-address replay')
                for index in (0, b.copies-1):
                    got, want = b.read(index), b.gold[0]
                    if np.any((got != want) & (((got | want) & 0x7fff) != 0)):
                        raise ValueError('rotating output oracle mismatch')
            medians = {v: statistics.median(samples[v]) for v in VARIANTS}
            result = dict(status='PASS', point=a.child, q=b.q, n=b.n, k=b.k, fixture=b.fixture,
                          proof=proofs, negative_bad=negative, samples_us=samples,
                          round_medians_us=round_medians, median_us=medians,
                          candidate_delta_pct={v: 100*(medians[v]/medians[0]-1) for v in VARIANTS[1:]},
                          spread_pct={v: 100*(max(round_medians[v])-min(round_medians[v]))/medians[v] for v in VARIANTS},
                          useful_gbps={v: b.traffic['useful_bytes']/medians[v]/1000 for v in VARIANTS},
                          traffic=b.traffic, ring_copies=b.copies, calls_per_graph=b.replays,
                          scope=SCOPE, production_changed=False, performance_admitted=False)
    finally:
        b.close()
    name = f'point-{a.child}.json' if a.profile_variant is None else f'profile-{a.child}-{a.profile_variant}-proof-{time.time_ns()}.json'
    atomic_json(a.output/name, result)
    print('DENSE_DEQUANT_RESULT '+json.dumps({k: result[k] for k in
          ('status', 'point', 'median_us', 'candidate_delta_pct') if k in result}), flush=True)


def package(output):
    destination = output/f'review-{time.time_ns()}.tgz'
    paths = [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json', '.log', '.csv')]
    with tarfile.open(destination, 'w:gz') as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(output)), recursive=False)
    print('results='+str(destination), flush=True)
    print('ACU reports retained locally: '+str(output/'profile-*.acurep'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--points', default='0,1,2,3,4,5,6,7')
    p.add_argument('--l2-bytes', type=int, default=0, help='independently verified fallback if SDK reports zero')
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='same source/device/plan; missing timing cases, fresh optional profiles')
    p.add_argument('--acu', type=Path)
    p.add_argument('--profile-points', default='0,1,5,6')
    p.add_argument('--smi', type=Path)
    p.add_argument('--child', type=int, choices=range(len(POINTS)), help=argparse.SUPPRESS)
    p.add_argument('--profile-variant', type=int, choices=VARIANTS, help=argparse.SUPPRESS)
    a = p.parse_args()
    points, profiles = selected(a.points), selected(a.profile_points)
    if a.profile_variant is not None and a.child is None:
        p.error('--profile-variant requires --child')
    a.output = a.output.resolve()
    if a.child is not None:
        child(a)
        return 0
    if not a.plan_only and not a.compile_only and not a.confirm_idle:
        p.error('--confirm-idle is required for device work')
    if a.resume:
        if a.plan_only or a.compile_only:
            p.error('--resume requires a device run')
        if json.loads((a.output/'plan.json').read_text()) != plan(points, a.l2_bytes):
            raise ValueError('resume plan changed')
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        atomic_json(a.output/'plan.json', plan(points, a.l2_bytes))
    if a.plan_only:
        print(json.dumps(plan(points, a.l2_bytes), indent=2))
        return 0
    summary = dict(status='INCOMPLETE', points={}, profiles={}, scope=SCOPE,
                   production_changed=False, performance_admitted=False)
    if a.resume and (a.output/'summary.json').is_file():
        atomic_json(a.output/f'previous-summary-{time.time_ns()}.json', json.loads((a.output/'summary.json').read_text()))
    try:
        a.sdk = a.sdk.resolve(strict=True)
        verify_receipt(a) if a.resume else build(a)
        if a.compile_only:
            summary.update(status='COMPILE_ONLY', kernel_launches=0)
            return 0
        from tools.run_kpack_llama_ab import device
        from dev.strategy.refresh_clock import request_lock, observe
        identity = device(a)
        a.l2_bytes = identity['l2']['bytes']
        pci = identity['device']['pci']
        smi = a.smi or next((path for path in (Path(shutil.which('ppu-smi') or '/nonexistent'),
                           a.sdk/'bin/ppu-smi') if path.is_file()), None)
        if smi is None:
            raise ValueError('ppu-smi required to request 1700MHz')
        with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            authority = dict(device=identity, plan=plan(points, a.l2_bytes),
                             idle='OPERATOR_CONFIRMED', frequency_normalization=False)
            if a.resume and json.loads((a.output/'authority.json').read_text()) != authority:
                raise ValueError('resume physical device/SDK/L2 differs')
            request = request_lock(smi, pci, 1700)
            atomic_json(a.output/f'clock-lock-{time.time_ns()}.json', request)
            if request['returncode']:
                raise ValueError('1700MHz request failed')
            if not a.resume:
                atomic_json(a.output/'authority.json', authority)
            def run(index, variant=None):
                name = f'point-{index}' if variant is None else f'profile-{index}-{variant}'
                attempt = name+'-'+str(time.time_ns())
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--sdk', str(a.sdk),
                           '--output', str(a.output), '--l2-bytes', str(a.l2_bytes), '--child', str(index)]
                if variant is not None:
                    from tools.profile_kpack_gpu_compact import acu_launch_command
                    command += ['--profile-variant', str(variant)]
                    command = acu_launch_command(a.acu, a.output/attempt, command)
                atomic_json(a.output/(attempt+'-command.json'), dict(command=command))
                atomic_json(a.output/(attempt+'-before.json'), observe(smi, pci, 1700, advisory=True))
                print(f'DENSE_DEQUANT_START case={name} shape={POINTS[index]}', flush=True)
                with (a.output/(attempt+'.log')).open('x') as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    while True:
                        try:
                            rc = proc.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            print(f'DENSE_DEQUANT_WAIT case={name} log={log.name}', flush=True)
                atomic_json(a.output/(attempt+'-after.json'), observe(smi, pci, 1700, advisory=True))
                if variant is not None and rc == 0:
                    report = a.output/(attempt+'.acurep')
                    if not report.is_file():
                        raise ValueError('ACU succeeded without its report: '+str(report))
                    with (a.output/(attempt+'-raw.csv')).open('x') as raw:
                        subprocess.run([str(a.acu), '--import', str(report), '--page', 'raw', '--csv'],
                                       stdout=raw, stderr=subprocess.STDOUT, check=True)
                print(f'DENSE_DEQUANT_CASE case={name} rc={rc} log={attempt}.log', flush=True)
                return rc
            for index in points:
                saved = a.output/f'point-{index}.json'
                if a.resume and saved.is_file():
                    validate_point(json.loads(saved.read_text()), index, a.l2_bytes)
                    print(f'DENSE_DEQUANT_REUSE point={index} raw_results_unchanged=1', flush=True)
                    rc = 0
                else:
                    rc = run(index)
                summary['points'][str(index)] = dict(returncode=rc)
                if rc == 0:
                    data = validate_point(json.loads(saved.read_text()), index, a.l2_bytes)
                    summary['points'][str(index)].update(median_us=data['median_us'], delta_pct=data['candidate_delta_pct'])
                atomic_json(a.output/'summary.json', summary)
            if a.acu:
                for index in profiles:
                    if index not in points or summary['points'][str(index)]['returncode']:
                        continue
                    medians = summary['points'][str(index)]['median_us']
                    candidate = min(VARIANTS[1:], key=lambda v: medians[str(v)])
                    for v in (0, candidate):
                        # Profiles use fresh report names on a retry; never
                        # overwrite a prior .acurep with a failed capture.
                        summary['profiles'][f'{index}-{v}'] = run(index, v)
                        atomic_json(a.output/'summary.json', summary)
            if (all(r['returncode'] == 0 for r in summary['points'].values()) and
                    all(rc == 0 for rc in summary['profiles'].values())):
                summary['status'] = 'PASS'
        return 0 if summary['status'] == 'PASS' else 1
    except BaseException as error:
        atomic_json(a.output/'error.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        atomic_json(a.output/'summary.json', summary)
        package(a.output)


if __name__ == '__main__':
    raise SystemExit(main())

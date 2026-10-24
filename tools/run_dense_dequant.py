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


def selected(text, shapes=POINTS):
    values = [int(x) for x in text.split(',')]
    if not values or len(set(values)) != len(values) or any(x not in range(len(shapes)) for x in values):
        raise ValueError(f'points must be distinct indices in 0..{len(shapes)-1}')
    return values


def q8_shapes(text):
    """Explicit bounded TP-local helper cohort; never changes the old indices."""
    if not re.fullmatch(r'[0-9]+x[0-9]+(?:,[0-9]+x[0-9]+)*', text):
        raise ValueError('Q8 shapes must be N x K pairs, e.g. 8192x3072,4096x2048')
    shapes = tuple((8, *map(int, v.split('x'))) for v in text.split(','))
    if (not 1<=len(shapes)<=8 or len(set(shapes))!=len(shapes) or
            any(n<=0 or k<=0 or n%64 or k%128 or n*k>64*1024*1024 for _,n,k in shapes)):
        raise ValueError('one to eight distinct Q8 N64/K128-aligned shapes up to 64M elements required')
    return shapes


def q8_tuning():
    """Use the actual C++ launch inventory, not a second handwritten table."""
    rows=re.findall(r'^QKP_Q8_TUNE\((\d+), ([01]), (128|256|512), ([01])\)$',
                    (ROOT/'quactlize/prefill/dense_q8_tuning.inc').read_text(),re.M)
    result={int(i):dict(fast=int(f),threads=int(t),vector_scale=int(v)) for i,f,t,v in rows}
    if len(rows)!=len(result) or set(result)!=set(range(26,35)):
        raise ValueError('Q8 tuning inventory changed')
    return result


def q8_half2():
    rows=re.findall(r'^QKP_Q8_HALF2\((\d+), ([01]), (128|256)\)$',
                    (ROOT/'quactlize/prefill/dense_q8_half2.inc').read_text(),re.M)
    result={int(i):dict(fast=int(f),threads=int(t)) for i,f,t in rows}
    if len(rows)!=len(result) or set(result)!=set(range(35,39)):
        raise ValueError('Q8 half2 inventory changed')
    return result


def q8_paired():
    rows=re.findall(r'^QKP_Q8_PAIRED\((\d+), ([01]), (128|256)\)$',
                    (ROOT/'quactlize/prefill/dense_q8_paired.inc').read_text(),re.M)
    result={int(i):dict(signed=int(s),threads=int(t)) for i,s,t in rows}
    if len(rows)!=len(result) or set(result)!=set(range(39,43)):
        raise ValueError('Q8 paired inventory changed')
    return result


def variants_for(exchange=False,fast_tail=False,half2=False,native=False,paired=False):
    if sum((exchange,fast_tail,half2))>1:raise ValueError('choose one Q8 experiment')
    if paired:
        if not half2:raise ValueError('paired comparison requires the F16 half2 cohort')
        return (*((-1,) if native else ()),22,27,36,38,*q8_paired())
    if native:
        if not half2:raise ValueError('native comparison requires the F16 half2 cohort')
        return (-1,22,27,36,38)
    if half2:return (22,27,28,30,*q8_half2()) # all prior F16 per-shape winners
    return (22,*q8_tuning()) if fast_tail else (22,23,24,25) if exchange else VARIANTS


def mode(exchange=False,fast_tail=False,half2=False,native=False,paired=False):
    if paired:return 'Q8_PAIRED_F16_AB'
    if native:return 'Q8_NATIVE_F16_AB'
    if half2:return 'Q8_F16_HALF2'
    return 'Q8_FAST_TAIL' if fast_tail else 'Q8_SHARED_EXCHANGE' if exchange else 'PLANE_PAIR'


def plan(points, l2, exchange=False,fast_tail=False,half2=False,native=False,paired=False, *, shapes=POINTS):
    variants=variants_for(exchange,fast_tail,half2,native,paired)
    if (exchange or fast_tail or half2) and any(shapes[i][0] != 8 for i in points):
        raise ValueError('experiment supports Q8 only')
    result=dict(schema='quactlize.dense-dequant-ab.v1', scope=SCOPE,
                points=[dict(index=i, q=shapes[i][0], n=shapes[i][1], k=shapes[i][2]) for i in points],
                variants=list(variants), order=list(variants)+list(reversed(variants)), rounds=3, samples_per_visit=5,
                correctness_storage=['F16', 'BF16'], timed_storage=['F16','BF16'] if exchange or fast_tail else 'F16',
                experiment=mode(exchange,fast_tail,half2,native,paired),
                profile_scope='ONE_CHECKED_DIRECT_LAUNCH_AFTER_WARMUP_NO_FLUSH_NOT_ROTATING_EVENT_TIME',
                l2_bytes=l2, cache='INPUT_AND_OUTPUT_RINGS_EACH_EXCEED_2.25_L2',
                warmup='TWO_COMPLETE_GRAPHS_PER_VARIANT_EXCLUDED',
                timing='EVENT_AVERAGE_OVER_SINGLE_KERNEL_GRAPH_NODES_INCLUDES_NODE_GAPS',
                requested_clock_mhz=1700, actual_clock='ADVISORY_NOT_FAILURE',
                production_changed=False, performance_admitted=False)
    if fast_tail:
        result['configs']={str(k):v for k,v in q8_tuning().items()}
        result['tail_scope']='SHORTER_CTA_PHASES_MORE_WAVES_NOT_A_CLAIM_OF_ZERO_PARTIAL_WAVES'
    if half2:
        result.update(configs={str(k):v for k,v in q8_half2().items()},correctness_storage=['F16'],
            arithmetic_gate='ALL_256_CODES_X_63488_FINITE_F16_SCALES_TWO_LANES_RAW_BITS',
            bf16='CANDIDATES_MUST_REJECT_NOT_ROUND_THROUGH_FP16')
    if native:
        result.update(native_entry='EXISTING_LLAMA_Q8_F16_CONVERTER',
            cache_pairing='EQUAL_RAW_AND_PACKED_RING_BYTES_SAME_OUTPUT_ADDRESSES',
            native_recompiled=False)
    if paired:
        result.update(paired_configs={str(k):v for k,v in q8_paired().items()},
            paired_gate='ALL_256_CODES_X_63488_FINITE_F16_SCALES_TWO_PHASES_DISTINCT_N_SCALES_RAW_BITS',
            admission='HELPER_ONLY_RECHECK_CALL_CONTEXT_BEFORE_GLOBAL_ROUTE_BUCKETING')
    return result


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
        if any(name in match[1] for name in ('dense_dequant_plane_pair','dense_dequant_q8_exchange','dense_dequant_q8_tuned','dense_dequant_q8_half2','dense_dequant_q8_paired')):
            for prefix in ('vmem.ld.b32x4', 'tsm.ld.b32x4', 'vmem.st.b32x4'):
                if not any(op.startswith(prefix) for op in ops):
                    raise ValueError('candidate vector delivery scalarized: '+prefix)
            if resources[match[1]]['stack_bytes']:
                raise ValueError('candidate has a local stack allocation')
        result[match[1]] = dict(resources=resources[match[1]], operations=dict(ops),
            static_instructions=sum(ops.values()), instruction_sha256=hashlib.sha256(bytes.fromhex(''.join(
                re.findall(r'^\s*[0-9a-f]+: ((?:[0-9a-f]{2} ){7}[0-9a-f]{2})\s',body,re.M)))).hexdigest())
    candidates = [s for s in result if 'dense_dequant_plane_pair' in s]
    exchange = [s for s in result if 'dense_dequant_q8_exchange' in s]
    tuned=[s for s in result if 'dense_dequant_q8_tuned' in s]
    half=[s for s in result if 'dense_dequant_q8_half2' in s]
    paired=[s for s in result if 'dense_dequant_q8_paired' in s]
    if (len(candidates) != 12 or len(exchange) != 4 or len(tuned)!=2*len(q8_tuning()) or
            len(half)!=4 or len(paired)!=4 or set(result) != set(resources)):
        raise ValueError('native specialization inventory differs')
    for config in q8_paired().values():
        names=[s for s in paired if f'ILb{config["signed"]}ELj{config["threads"]}E' in s]
        if len(names)!=1:raise ValueError('Q8 paired specialization missing/duplicate')
        ops=result[names[0]]['operations']
        for op,expected in (('v.mul.f16x2',4096//config['threads']),
                            ('v.byte.prmt.b32',4096//config['threads']),
                            ('vmem.ld.b32x4',512//config['threads']),
                            ('vmem.st.b32x4',1024//config['threads'])):
            if ops.get(op)!=expected:raise ValueError('paired native delivery/arithmetic differs: '+op)
        if any(op.startswith('vmem.st.') and op!='vmem.st.b32x4' for op in ops):
            raise ValueError('Q8 paired vector stores split')
        if not config['signed'] and any('cnvt.f16.' in op or 'cnvt.f32.' in op or 'mul.f32' in op for op in ops):
            raise ValueError('fast Q8 paired still has code or output float conversions')
    for config in q8_half2().values():
        names=[s for s in half if f'ILb{config["fast"]}ELj{config["threads"]}E' in s]
        if len(names)!=1:raise ValueError('Q8 half2 specialization missing/duplicate')
        ops=result[names[0]]['operations']
        if ops.get('v.mul.f16x2')!=4096//config['threads']:
            raise ValueError('Q8 half2 multiply did not lower to native packed arithmetic')
        if config['fast'] and any('cnvt.f16.' in op or 'cnvt.f32.' in op or 'mul.f32' in op for op in ops):
            raise ValueError('fast Q8 half2 still has code or output float conversions')
        if config['fast'] and any(op.startswith('vmem.st.') and op!='vmem.st.b32x4' for op in ops):
            raise ValueError('fast Q8 half2 vector output stores split')
    for config in q8_tuning().values():
        for bf in (0,1):
            fragment=f'ILb{bf}ELb{config["fast"]}ELj{config["threads"]}ELb{config["vector_scale"]}E'
            names=[s for s in tuned if fragment in s]
            if len(names)!=1:raise ValueError('Q8 tuning body missing/duplicated')
            ops=result[names[0]]['operations']
            if config['fast'] and any('cnvt.f32.i32' in op for op in ops):
                raise ValueError('fast Q8 still uses integer-to-float conversion')
            if config['vector_scale'] and ('vmem.ld.b32x2' not in ops or 'vmem.ld.b16' in ops):
                raise ValueError('Q8 metadata vector load scalarized')
    for bf in (0, 1):
        for tk in (128, 256):
            if sum(f'ILb{bf}ELj{tk}E' in s for s in exchange) != 1:
                raise ValueError('Q8 exchange precision/tile missing')
    for q in (8, 10, 11):
        for bf in (0, 1):
            for tk in (128, 256):
                if sum(f'ILi{q}ELb{bf}ELj{tk}E' in s for s in candidates) != 1:
                    raise ValueError('candidate format/storage/tile missing')
    atomic_json(output/'native.json', dict(scope='STATIC_NOT_DYNAMIC_COUNTS', kernels=result))


def build(a, extra_sources=()):
    output = a.output/'build'
    output.mkdir()
    sources = {str(p.relative_to(ROOT)): sha(p) for directory in
               ('quactlize/dequant', 'quactlize/prefill', 'quactlize/execution',
                'quactlize/include', 'third_party/actlize/include')
               for p in (ROOT/directory).rglob('*') if p.is_file() and
               (p.suffix in HEADER_SUFFIXES or p.name == 'dense.cu' or str(p.relative_to(ROOT)) in extra_sources)}
    sources.update({str(p.relative_to(ROOT)): sha(p) for directory in ('quactlize', 'dev', 'tools', 'reference')
                    for p in (ROOT/directory).rglob('*.py') if p.is_file()})
    for relative in extra_sources:
        path = (ROOT/relative).resolve(strict=True)
        path.relative_to(ROOT)
        sources[relative] = sha(path)
    obj = output/'dense.o'
    command = [str(a.sdk/'bin/hgcc'), *FLAGS, '-lineinfo',
               '-I'+str(ROOT/'quactlize/include'), '-I'+str(ROOT/'third_party/actlize/include'),
               '-c', str(ROOT/'quactlize/dequant/dense.cu'), '-o', str(obj)]
    receipt = dict(sources=sources, compiler_sha256=sha(a.sdk/'bin/hgcc'), commands=[command],
                   source_head=os.environ.get('DENSE_DEQUANT_SOURCE_SHA', 'SEE_SOURCE_HASHES'),
                   actlize_head=os.environ.get('DENSE_DEQUANT_ACTLIZE_SHA', 'SEE_SOURCE_HASHES'),
                   runner_sha256=sha(__file__), compile_only=a.compile_only)
    if getattr(a,'native_library',None):
        from dev.dense_helpers.native_q8 import describe
        receipt['native']=describe(a.native_library)
    atomic_json(output/'receipt.json', receipt)
    start = time.monotonic()
    with (output/'compile.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    inspect(a.sdk, obj, output)
    objects = [obj]
    for relative in extra_sources:
        source = ROOT/relative
        if relative not in sources: raise ValueError('unsealed probe source: '+relative)
        extra = output/(source.stem+'.o')
        command = ([str(a.sdk/'bin/hgcc'), *FLAGS] if source.suffix == '.cu' else
                   ['g++','-std=c++17','-O2','-fPIC'])
        command += ['-I'+str(ROOT), '-I'+str(ROOT/'quactlize/include'),
                    '-I'+str(ROOT/'third_party/actlize/include'), '-I'+str(a.sdk/'include'),
                    '-I'+str(a.sdk/'targets/x86_64-linux/include'), '-c', str(source), '-o', str(extra)]
        with (output/(source.stem+'.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        objects.append(extra); receipt['commands'].append(command)
    if not a.compile_only:
        library = output/'libquactlize_dense_dequant_probe.so'
        command = ['g++', '-shared', '-Wl,-Bsymbolic', *map(str,objects), '-L'+str(a.sdk/'lib'),
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
    if receipt.get('native'):
        from dev.dense_helpers.native_q8 import verify
        verify(receipt['native'])
        if a.native_library is None or a.native_library.resolve()!=Path(receipt['native']['library']):
            raise ValueError('resume native library path changed')
    elif getattr(a,'native_library',None):raise ValueError('resume has no native reference receipt')
    return receipt


def validate_point(data, index, l2, exchange=False,fast_tail=False,half2=False,native=False,paired=False, *, shapes=POINTS):
    if (data.get('status') != 'PASS' or data.get('point') != index or
            tuple(data.get(name) for name in ('q', 'n', 'k')) != shapes[index] or
            data.get('scope') != SCOPE or data.get('production_changed') is not False or
            data.get('performance_admitted') is not False or data.get('negative_bad', 0) <= 0):
        raise ValueError('saved point identity/proof differs')
    for kind in ('reads', 'writes'):
        if data['traffic'][kind]*data['ring_copies'] < 2.25*l2:
            raise ValueError('saved point has an undersized weight/output ring')
    if data['calls_per_graph'] != 2*data['ring_copies']:
        raise ValueError('saved graph traversal differs')
    variants=variants_for(exchange,fast_tail,half2,native,paired)
    if (exchange or fast_tail or half2) and (data.get('experiment')!=mode(exchange,fast_tail,half2,native,paired) or data['q']!=8):
        raise ValueError('wrong Q8 experiment')
    if fast_tail or half2:
        alignment=data.get('memory_alignment',{})
        for name in (('low','units','output','raw') if native else ('low','units','output')):
            if alignment.get(name)!={str(w):[0] for w in (16,32,64,128)}:
                raise ValueError('Q8 footprint experiment requires recorded128B-aligned rings: '+name)
    for variant in variants:
        key = str(variant)
        values = data['samples_us'][key]
        if (len(values) != 30 or any(not math.isfinite(v) or v <= 0 for v in values) or
                data['median_us'][key] != statistics.median(values) or
                data['round_medians_us'][key] != [statistics.median(values[i:i+10]) for i in (0, 10, 20)]):
            raise ValueError('saved timing samples are incomplete')
        for storage in (('0',) if half2 else ('0','1')):
            proof = data['proof'][storage][key]
            if (proof['bad'] or proof['guard'] != 'PASS' or proof['compared'] != data['n']*data['k'] or
                    variant and not proof['control_raw_bits_equal']):
                raise ValueError('saved numerical proof incomplete')
    if half2:
        if data.get('half2_gate')!=dict(pairs=256*63488,lanes=2,bad_ordinary=0,bad_fast=0,
                                      negative_bad=1,bf16_rejected=list(q8_half2())):
            raise ValueError('exhaustive half2 arithmetic/rejection gate incomplete')
        if data.get('bf16_timing') is not None:raise ValueError('half2 must not time BF16')
    if paired and data.get('paired_gate')!=paired_gate_receipt():
        raise ValueError('paired arithmetic/rejection gate incomplete')
    if native:
        if data.get('native_raw_sha256')!=data['fixture']['raw_sha256'] or data.get('native_bytes')!=data['traffic']['reads']:
            raise ValueError('native/canonical fixture or input ring bytes differ')
        if data.get('changed_address_copies_checked')!=data['ring_copies']:
            raise ValueError('native comparison did not check every changed-address output')
    if exchange or fast_tail:
        bf=data.get('bf16_timing',{})
        for v in variants:
            key=str(v);values=bf.get('samples_us',{}).get(key,[])
            if (len(values)!=30 or any(not math.isfinite(x) or x<=0 for x in values) or
                    bf['median_us'][key]!=statistics.median(values) or
                    bf['round_medians_us'][key]!=[statistics.median(values[i:i+10]) for i in (0,10,20)]):
                raise ValueError('BF16 timing/proof incomplete')
    return data


def fixture(q, n, k, *, include_raw=False):
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
    if q != 8 and SPECS[q].has_min:
        offset = SPECS[q].dmin_offset
        minima = rng.uniform(-.009, .009, (n, k//block)).astype('<f2')
        raw[..., offset:offset+2] = minima.view('u1').reshape(n, k//block, 2)
    packed = {name: value for name, value in planes(raw, q).items() if name in ('low', 'high', 'units')}
    gold = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(n, k)
    if not np.isfinite(gold).all() or not np.any(gold):
        raise ValueError('invalid independent GGUF fixture')
    result=packed, [gold.astype('<f2').view('<u2'), bf16_bits(gold)], dict(
        raw_sha256=hashlib.sha256(memoryview(raw)).hexdigest(),
        planes={name: hashlib.sha256(memoryview(value)).hexdigest() for name, value in packed.items()},
        oracle='INDEPENDENT_GGUF_FP32_AFFINE_THEN_ONE_RNE_F16_OR_BF16')
    return (*result,raw) if include_raw else result


class Bench:
    def __init__(self, a, *, shape=None):
        from quactlize.runtime.native import SDK, checked
        from quactlize.dequant.native import Call
        from quactlize.execution.native import Arrangement, arrangement
        from tools.run_kpack_gemv_gate import Resources
        from tools.run_kpack_grouped_device_gate import graph_bind
        self.q, self.n, self.k = shape if shape is not None else (getattr(a, 'shapes', None) or POINTS)[a.child]
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
        self.planes, self.gold, self.fixture,raw = fixture(self.q,self.n,self.k,include_raw=True)
        self.native=None
        if getattr(a,'native_library',None):
            from dev.dense_helpers.native_q8 import NativeQ8
            if self.q!=8 or a.native_library.resolve()!=Path(receipt['native']['library']):
                raise ValueError('native Q8 library/workload binding differs')
            self.native=NativeQ8(receipt['native'])
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
        if self.native:
            if raw.nbytes!=reads:raise ValueError('raw/packed weight sizes differ')
            self.raw_bytes=raw.nbytes
            self.raw_sha256=hashlib.sha256(memoryview(raw)).hexdigest()
            self.raw_base=self.r.alloc(raw.nbytes*self.copies)
            checked(self.sdk.lib.hggcMemcpy(self.raw_base,raw.ctypes.data,raw.nbytes,1),'upload native rawGGUF')
            self.sdk.synchronize(None)
            for i in range(1,self.copies):
                checked(copy(self.raw_base+i*raw.nbytes,self.raw_base,raw.nbytes,3,self.r.stream),'initialize native ring')
        base = self.r.alloc(self.stride*self.copies)
        self.r.fill(base, 0xa5, self.stride*self.copies)
        for i in range(self.copies):
            self.calls.append(Call(version=1, size=C.sizeof(Call), qtype=self.q, n=self.n, k=self.k,
                experts=1, operation=1, config=0, output=base+i*self.stride+self.guard,
                output_bytes=self.bytes, stream=self.r.stream.value,
                low_bytes=self.planes['low'].nbytes, high_bytes=self.planes['high'].nbytes,
                unit_bytes=self.planes['units'].nbytes,
                **{name: p+i*self.planes[name].nbytes for name, p in self.buffers.items()}))
        self.memory_alignment={name:{str(width):sorted({int(getattr(c,name))%width for c in self.calls})
                                    for width in (16,32,64,128)} for name in (*self.buffers,'output')}
        if self.native:
            self.memory_alignment['raw']={str(w):sorted({(self.raw_base+i*self.raw_bytes)%w for i in range(self.copies)})
                                          for w in (16,32,64,128)}
        if (getattr(a,'q8_fast_tail',False) or getattr(a,'q8_half2',False)) and any(value['128']!=[0] for value in self.memory_alignment.values()):
            raise ValueError('Q8 footprint experiment requires128B-aligned input/output ring entries')
        self.sdk.synchronize(self.r.stream)

    def launch(self, variant, storage=0, index=0):
        if variant==-1:
            if not self.native or storage:raise ValueError('native Q8 dequant requires bound F16 converter')
            self.native.launch(self.raw_base+index*self.raw_bytes,self.calls[index].output,self.n*self.k,self.r.stream)
            return self.sdk.lib.hggcGetLastError()
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

    def negative(self, variant=22):
        import numpy as np
        from quactlize.runtime.native import checked
        self.r.fill(self.buffers['low'], 0, self.planes['low'].nbytes)
        checked(self.launch(variant), 'blank low-plane negative')
        bad = int(np.count_nonzero(self.read() != self.gold[0]))
        if bad < self.n*self.k//100:
            raise ValueError('blank low plane escaped oracle')
        checked(self.sdk.lib.hggcMemcpy(self.buffers['low'], self.planes['low'].ctypes.data,
                                      self.planes['low'].nbytes, 1), 'restore fixture')
        self.sdk.synchronize(None)
        return bad

    def graph(self, variant, storage=0):
        from tools.run_kpack_grouped_decode_probe import Replay
        from quactlize.runtime.native import checked
        key = (variant, storage)
        if key not in self.graphs:
            index = 0
            def launch():
                nonlocal index
                result = self.launch(variant, storage, index=index % self.copies)
                index += 1
                return result
            self.graphs[key] = Replay(self.sdk, self.r.stream, launch, self.replays)
            for _ in range(2):
                checked(self.graphs[key](), 'excluded graph warmup')
            self.sdk.synchronize(self.r.stream)
        return self.graphs[key]

    def close(self):
        self.sdk.synchronize(self.r.stream)
        for graph in self.graphs.values():
            graph.close()
        self.r.close()


def measure_storage(b, variants, storage, point):
    import numpy as np
    from quactlize.runtime.native import checked
    for v in variants:
        b.check(v,storage);b.graph(v,storage)
    samples={v:[] for v in variants};round_medians={v:[] for v in variants}
    for round_index in range(3):
        current={v:[] for v in variants}
        for v in (*variants,*reversed(variants)):
            current[v].extend(t/b.replays for t in b.r.samples(b.graph(v,storage),5))
        for v in variants:
            samples[v].extend(current[v]);round_medians[v].append(statistics.median(current[v]))
        print(f'DENSE_DEQUANT_PROGRESS point={point} storage={storage} round={round_index+1}/3',flush=True)
    for v in variants:
        if b.native:
            # Outside every timed interval: another arm's correct outputs must
            # not hide a missing native launch or a stale captured address.
            for call in b.calls:b.r.fill(call.output,0x7f,b.bytes)
        checked(b.graph(v,storage)(),'final changed-address replay')
        for index in (range(b.copies) if b.native else (0,b.copies-1)):
            got,want=b.read(index),b.gold[storage]
            if np.any((got!=want)&(((got|want)&0x7fff)!=0)):
                raise ValueError('rotating output oracle mismatch')
    medians={v:statistics.median(samples[v]) for v in variants}
    return dict(samples_us=samples,round_medians_us=round_medians,median_us=medians,
        changed_address_copies_checked=b.copies if b.native else min(2,b.copies),
        candidate_delta_pct={v:100*(medians[v]/medians[variants[0]]-1) for v in variants[1:]},
        spread_pct={v:100*(max(round_medians[v])-min(round_medians[v]))/medians[v] for v in variants},
        useful_gbps={v:b.traffic['useful_bytes']/medians[v]/1000 for v in variants})


def profile_entry(native, point, variant, storage, *, shapes=POINTS):
    """Bind ACU to an actually compiled specialization and exact geometry."""
    q,n,k=shapes[point]
    if q!=8 or variant not in (22,23,24,25,*q8_tuning(),*q8_half2(),*q8_paired()) or storage not in (0,1):
        raise ValueError('not a Q8 exchange profile')
    if variant in q8_paired():
        if storage:raise ValueError('paired variant is F16 only')
        c=q8_paired()[variant]
        fragment=f'dense_dequant_q8_pairedILb{c["signed"]}ELj{c["threads"]}E'
        symbols=[s for s in native['kernels'] if fragment in s]
        if len(symbols)!=1:raise ValueError('missing/duplicate compiled Q8 paired specialization')
        return dict(symbol=symbols[0],grid=[n//64,k//128,1],block=[c['threads'],1,1])
    if variant in q8_half2():
        if storage:raise ValueError('half2 variant is F16 only')
        c=q8_half2()[variant]
        fragment=f'dense_dequant_q8_half2ILb{c["fast"]}ELj{c["threads"]}E'
        symbols=[s for s in native['kernels'] if fragment in s]
        if len(symbols)!=1:raise ValueError('missing/duplicate compiled Q8 half2 specialization')
        return dict(symbol=symbols[0],grid=[n//64,k//128,1],block=[c['threads'],1,1])
    if variant in q8_tuning():
        c=q8_tuning()[variant]
        fragment=f'dense_dequant_q8_tunedILb{storage}ELb{c["fast"]}ELj{c["threads"]}ELb{c["vector_scale"]}E'
        symbols=[s for s in native['kernels'] if fragment in s]
        if len(symbols)!=1:raise ValueError('missing/duplicate compiled Q8 tuning specialization')
        return dict(symbol=symbols[0],grid=[n//64,k//128,1],block=[c['threads'],1,1])
    tile=128 if variant in (22,24) else 256
    fragment=(f'dense_dequant_plane_pairILi8ELb{storage}ELj{tile}E' if variant in (22,23)
              else f'dense_dequant_q8_exchangeILb{storage}ELj{tile}E')
    symbols=[s for s in native['kernels'] if fragment in s]
    if len(symbols)!=1:raise ValueError('missing/duplicate compiled profile specialization')
    return dict(symbol=symbols[0],grid=[n//64,k//tile,1],block=[128,1,1])


def profile_variants(medians,exchange=False,fast_tail=False,half2=False,native=False,paired=False):
    """Small diagnostic basis plus finalist; never profile all ten configs."""
    if paired:
        return (*((-1,) if native else ()),min((22,27,36,38),key=lambda v:medians[str(v)]),
                min((39,40),key=lambda v:medians[str(v)]),min((41,42),key=lambda v:medians[str(v)]))
    if native:
        return (-1,min((22,27),key=lambda v:medians[str(v)]),min((36,38),key=lambda v:medians[str(v)]))
    if half2:
        candidate=min(q8_half2(),key=lambda v:medians[str(v)])
        ordinary=35 if q8_half2()[candidate]['threads']==128 else 37
        control=min((22,27,28,30),key=lambda v:medians[str(v)])
        return tuple(dict.fromkeys((control,ordinary,ordinary+1)))
    if fast_tail:
        candidate=min(q8_tuning(),key=lambda v:medians[str(v)])
        # Isolate conversion, metadata footprint and the best ordinary CTA
        # grain even when one of them is not part of the combined finalist.
        c=q8_tuning()[candidate]
        ordinary={cfg['threads']:v for v,cfg in q8_tuning().items() if not cfg['fast'] and not cfg['vector_scale']}
        grain=ordinary.get(c['threads'],min((27,29),key=lambda v:medians[str(v)]))
        basis=[22,26,31,grain]
        for v,cfg in q8_tuning().items():
            if cfg['threads']==c['threads'] and (not cfg['fast'] and not cfg['vector_scale'] or
                    c['vector_scale'] and cfg['fast']==c['fast'] and not cfg['vector_scale']):
                basis.append(v)
        return tuple(dict.fromkeys([*basis,candidate]))
    candidate=min((24,25) if exchange else VARIANTS[1:],key=lambda v:medians[str(v)])
    controls=(22,23) if exchange and candidate==25 else (22,) if exchange else (0,)
    return (*controls,candidate)


def half2_gate(b):
    """Run outside timing/profiling; strict bits include subnormals and signed zero."""
    import numpy as np
    from quactlize.runtime.native import checked
    fn=b.lib.quactlize_q8_half2_gate
    fn.argtypes=[C.c_void_p,C.c_int,C.c_void_p];fn.restype=C.c_int
    ptr=b.r.alloc(16)
    for fault in (0,1):
        initial=np.array([0,0,0xffffffff,0],dtype='<u4')
        checked(b.sdk.lib.hggcMemcpy(ptr,initial.ctypes.data,16,1),'gate counters upload')
        b.sdk.synchronize(None)
        checked(fn(ptr,fault,b.r.stream),'exhaustive half2 gate launch')
        b.sdk.synchronize(b.r.stream)
        bad=np.frombuffer(b.sdk.download(ptr,16),dtype='<u4').tolist()
        expected=[0,fault,0 if fault else 0xffffffff,256*63488]
        if bad!=expected:raise ValueError(f'half2 arithmetic gate fault={fault} got={bad} want={expected}')
    rejected=[]
    for v in q8_half2():
        if b.launch(v,1)!=24: # QKG_SHAPE, no launch permitted
            raise ValueError(f'half2 variant {v} did not reject BF16')
        rejected.append(v)
    return dict(pairs=bad[3],lanes=2,bad_ordinary=0,bad_fast=0,negative_bad=1,bf16_rejected=rejected)


def paired_gate_receipt():
    return dict(pairs=2*256*63488,lanes=2,phases=2,distinct_scales=True,
                bad_signed=0,bad_fast=0,negative_bad=1,bf16_rejected=list(q8_paired()))


def paired_gate(b):
    import numpy as np
    from quactlize.runtime.native import checked
    fn=b.lib.quactlize_q8_paired_gate
    fn.argtypes=[C.c_void_p,C.c_int,C.c_void_p];fn.restype=C.c_int
    ptr=b.r.alloc(16)
    for fault in (0,1):
        initial=np.array([0,0,0xffffffff,0],dtype='<u4')
        checked(b.sdk.lib.hggcMemcpy(ptr,initial.ctypes.data,16,1),'paired gate counters upload')
        b.sdk.synchronize(None)
        checked(fn(ptr,fault,b.r.stream),'exhaustive paired gate launch')
        b.sdk.synchronize(b.r.stream)
        bad=np.frombuffer(b.sdk.download(ptr,16),dtype='<u4').tolist()
        expected=[0,fault,0 if fault else 0xffffffff,2*256*63488]
        if bad!=expected:raise ValueError(f'paired arithmetic gate fault={fault} got={bad} want={expected}')
    for v in q8_paired():
        if b.launch(v,1)!=24:raise ValueError(f'paired variant {v} did not reject BF16')
    return paired_gate_receipt()


def child(a):
    import numpy as np
    from quactlize.runtime.native import checked
    b = Bench(a)
    try:
        native=a.native_library is not None
        variants=variants_for(a.q8_exchange,a.q8_fast_tail,a.q8_half2,native,a.q8_paired)
        if a.profile_variant is not None:
            from tools.profile_kpack_gpu_compact import AcuRange
            storage=a.profile_storage
            print(f'DENSE_DEQUANT_PROFILE point={a.child} storage={storage} copies=1 graphs=0 cold_loops=0',flush=True)
            control, _ = b.check(0, storage)
            _, proof = b.check(a.profile_variant, storage, control)
            checked(b.launch(a.profile_variant,storage), 'direct profile warmup')
            b.sdk.synchronize(b.r.stream)
            with AcuRange(b.sdk,description='direct_kernel_calls=1 graphs=0 cold_loops=0'):
                checked(b.launch(a.profile_variant,storage), 'single isolated dequant profile')
                b.sdk.synchronize(b.r.stream)
            got=b.read()
            if not np.array_equal(got,control):raise ValueError('post-profile output differs from control')
            result = dict(status='PASS', point=a.child, profile_variant=a.profile_variant, proof=proof,
                          storage=storage,ring_copies=1, profiled_launches=1, timed_loops=0,
                          memory_alignment=b.memory_alignment,production_changed=False)
        else:
            proofs = {}
            gate=half2_gate(b) if a.q8_half2 else None
            paired=paired_gate(b) if a.q8_paired else None
            for storage in ((0,) if a.q8_half2 else (0,1)):
                control, proof = b.check(0, storage)
                proofs[str(storage)] = {'0': proof}
                for v in variants:
                    if v==0:continue
                    _, proofs[str(storage)][str(v)] = b.check(v, storage, control)
            negative = b.negative()
            f16=measure_storage(b,variants,0,a.child)
            bf16=measure_storage(b,variants,1,a.child) if a.q8_exchange or a.q8_fast_tail else None
            result = dict(status='PASS', point=a.child, q=b.q, n=b.n, k=b.k, fixture=b.fixture,
                          proof=proofs, negative_bad=negative, **f16,
                          bf16_timing=bf16,half2_gate=gate,experiment=mode(a.q8_exchange,a.q8_fast_tail,a.q8_half2,native,a.q8_paired),
                          traffic=b.traffic,memory_alignment=b.memory_alignment,ring_copies=b.copies,calls_per_graph=b.replays,
                          scope=SCOPE, production_changed=False, performance_admitted=False)
            if native:result.update(native_bytes=b.raw_bytes,native_raw_sha256=b.raw_sha256)
            if a.q8_paired:result['paired_gate']=paired
    finally:
        b.close()
    name = f'point-{a.child}.json' if a.profile_variant is None else f'profile-{a.child}-{a.profile_variant}-s{a.profile_storage}-proof-{time.time_ns()}.json'
    atomic_json(a.profile_result or a.output/name, result)
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
    p.add_argument('--points', help='default: all eight original families, or six Q8 families in a Q8 experiment')
    p.add_argument('--q8-shapes', help='with half2/paired/native: at most eight explicit NxK pairs; point indices are local to this list')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--q8-exchange', action='store_true', help='Q8 only: controls22/23 vs candidates24/25, F16 and BF16')
    modes.add_argument('--q8-fast-tail', action='store_true', help='Q8 only: DQ22 vs fast FP32 conversion, CTA grain and vector-scale factorial')
    modes.add_argument('--q8-half2', action='store_true', help='Q8 F16 only: retained winners vs ordinary/fast half2, exhaustive arithmetic gate')
    p.add_argument('--q8-paired',action='store_true',help='with --q8-half2: paired N scales and signed8/fast consumers')
    p.add_argument('--native-library',type=Path,help='with --q8-half2: compare existing llama libggml-cuda.so converter directly')
    p.add_argument('--l2-bytes', type=int, default=0, help='independently verified fallback if SDK reports zero')
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='same source/device/plan; missing timing cases, fresh optional profiles')
    p.add_argument('--acu', type=Path)
    p.add_argument('--profile-points', help='default: 4,6 for half2; 5,7 for other Q8 modes; otherwise 0,1,5,6')
    p.add_argument('--smi', type=Path)
    p.add_argument('--child', type=int, choices=range(len(POINTS)), help=argparse.SUPPRESS)
    p.add_argument('--profile-variant', type=int, choices=(-1,0,22,23,24,25,*q8_tuning(),*q8_half2(),*q8_paired()), help=argparse.SUPPRESS)
    p.add_argument('--profile-storage', type=int, choices=(0,1), default=0, help=argparse.SUPPRESS)
    p.add_argument('--profile-result', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    try:
        a.shapes = q8_shapes(a.q8_shapes) if a.q8_shapes else POINTS
    except ValueError as e:
        p.error(str(e))
    if a.q8_shapes and not (a.q8_half2 and a.q8_paired and a.native_library):
        p.error('--q8-shapes requires --q8-half2 --q8-paired --native-library')
    if a.q8_paired and not a.q8_half2:p.error('--q8-paired requires --q8-half2')
    native=a.native_library is not None
    if native:
        if not a.q8_half2:p.error('--native-library requires --q8-half2')
        a.native_library=a.native_library.resolve(strict=True)
    q8=a.q8_exchange or a.q8_fast_tail or a.q8_half2
    default_points = ','.join(map(str,range(len(a.shapes)))) if a.q8_shapes else ('2,3,4,5,6,7' if q8 else '0,1,2,3,4,5,6,7')
    default_profiles = ','.join(map(str,range(min(2,len(a.shapes))))) if a.q8_shapes else ('4,5,7' if a.q8_paired else '2,3,4,5,6,7' if native else '4,6' if a.q8_half2 else '5,7' if q8 else '0,1,5,6')
    points=selected(a.points or default_points,a.shapes)
    profiles=selected(a.profile_points or default_profiles,a.shapes)
    experiment=plan(points,a.l2_bytes,a.q8_exchange,a.q8_fast_tail,a.q8_half2,native,a.q8_paired,shapes=a.shapes)
    if a.profile_variant is not None and a.child is None:
        p.error('--profile-variant requires --child')
    if a.profile_variant is not None and a.profile_variant not in experiment['variants']:
        p.error('--profile-variant is outside this experiment')
    if a.q8_half2 and a.profile_storage:p.error('half2 profiles require F16 storage')
    if a.child is not None and a.child not in range(len(a.shapes)):
        p.error('--child is outside the shape cohort')
    if q8 and a.child is not None and a.shapes[a.child][0]!=8:
        p.error('Q8 exchange child must be a Q8 point')
    if a.profile_result is not None and (a.profile_variant is None or a.profile_result.exists() or
            a.profile_result.resolve().parent!=a.output.resolve()):
        p.error('--profile-result must be a new file in --output for a profile child')
    a.output = a.output.resolve()
    if a.child is not None:
        child(a)
        return 0
    if not a.plan_only and not a.compile_only and not a.confirm_idle:
        p.error('--confirm-idle is required for device work')
    if a.resume:
        if a.plan_only or a.compile_only:
            p.error('--resume requires a device run')
        if json.loads((a.output/'plan.json').read_text()) != experiment:
            raise ValueError('resume plan changed')
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        atomic_json(a.output/'plan.json', experiment)
    if a.plan_only:
        print(json.dumps(experiment, indent=2))
        return 0
    summary = dict(status='INCOMPLETE', points={}, profiles={}, scope=SCOPE,
                   production_changed=False, performance_admitted=False)
    if a.resume and (a.output/'summary.json').is_file():
        atomic_json(a.output/f'previous-summary-{time.time_ns()}.json', json.loads((a.output/'summary.json').read_text()))
    try:
        a.sdk = a.sdk.resolve(strict=True)
        verify_receipt(a) if a.resume else build(a,('dev/dense_helpers/q8_half_gate.cu',) if a.q8_half2 else ())
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
            authority = dict(device=identity, plan=plan(points, a.l2_bytes,a.q8_exchange,a.q8_fast_tail,a.q8_half2,native,a.q8_paired,shapes=a.shapes),
                             idle='OPERATOR_CONFIRMED', frequency_normalization=False)
            if a.resume and json.loads((a.output/'authority.json').read_text()) != authority:
                raise ValueError('resume physical device/SDK/L2 differs')
            request = request_lock(smi, pci, 1700)
            atomic_json(a.output/f'clock-lock-{time.time_ns()}.json', request)
            if request['returncode']:
                raise ValueError('1700MHz request failed')
            if not a.resume:
                atomic_json(a.output/'authority.json', authority)
            def run(index, variant=None, storage=0):
                name = f'point-{index}' if variant is None else f'profile-{index}-{variant}-s{storage}'
                attempt = name+'-'+str(time.time_ns())
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--sdk', str(a.sdk),
                           '--output', str(a.output), '--l2-bytes', str(a.l2_bytes), '--child', str(index)]
                if a.q8_exchange:command+=['--q8-exchange']
                if a.q8_fast_tail:command+=['--q8-fast-tail']
                if a.q8_half2:command+=['--q8-half2']
                if a.q8_paired:command+=['--q8-paired']
                if a.q8_shapes:command+=['--q8-shapes',a.q8_shapes]
                if native:command+=['--native-library',str(a.native_library)]
                if variant is not None:
                    from dev.native_components.profile import acu_command
                    command += ['--profile-variant', str(variant),'--profile-storage',str(storage),
                                '--profile-result',str(a.output/(attempt+'-proof.json'))]
                    command = acu_command(a.acu, a.output/attempt, command,graph_profiling='none')
                atomic_json(a.output/(attempt+'-command.json'), dict(command=command))
                atomic_json(a.output/(attempt+'-before.json'), observe(smi, pci, 1700, advisory=True))
                print(f'DENSE_DEQUANT_START case={name} shape={a.shapes[index]}', flush=True)
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
                    proof_path=a.output/(attempt+'-proof.json')
                    proof=json.loads(proof_path.read_text())
                    if (proof['status']!='PASS' or proof['point']!=index or proof['profile_variant']!=variant or
                            proof['storage']!=storage or proof['profiled_launches']!=1 or proof['ring_copies']!=1 or
                            proof['timed_loops']!=0 or proof['proof']['bad'] or not proof['proof']['control_raw_bits_equal']):
                        raise ValueError('profile child proof incomplete')
                    if q8:
                        from dev.native_components.profile import check_report
                        codegen=json.loads((a.output/'build/native.json').read_text())
                        raw_path=a.output/(attempt+'-raw.csv')
                        if variant==-1:
                            from dev.dense_helpers.native_q8 import check_profile
                            _,n,k=a.shapes[index]
                            check=check_profile(raw_path.read_text(errors='replace'),n,k)
                        else:check=check_report(raw_path.read_text(errors='replace'),profile_entry(codegen,index,variant,storage,shapes=a.shapes))
                        atomic_json(a.output/(attempt+'-receipt.json'),dict(**check,point=index,variant=variant,storage=storage,
                            report=str(report),report_sha256=sha(report),raw_sha256=sha(raw_path),proof_sha256=sha(proof_path),
                            prior_point_sha256=sha(a.output/f'point-{index}.json'),authority_sha256=digest(authority),
                            build_receipt_sha256=sha(a.output/'build/receipt.json')))
                print(f'DENSE_DEQUANT_CASE case={name} rc={rc} log={attempt}.log', flush=True)
                return rc
            for index in points:
                saved = a.output/f'point-{index}.json'
                if a.resume and saved.is_file():
                    validate_point(json.loads(saved.read_text()), index, a.l2_bytes,a.q8_exchange,a.q8_fast_tail,a.q8_half2,native,a.q8_paired,shapes=a.shapes)
                    print(f'DENSE_DEQUANT_REUSE point={index} raw_results_unchanged=1', flush=True)
                    rc = 0
                else:
                    rc = run(index)
                summary['points'][str(index)] = dict(returncode=rc)
                if rc == 0:
                    data = validate_point(json.loads(saved.read_text()), index, a.l2_bytes,a.q8_exchange,a.q8_fast_tail,a.q8_half2,native,a.q8_paired,shapes=a.shapes)
                    summary['points'][str(index)].update(median_us=data['median_us'], delta_pct=data['candidate_delta_pct'])
                    if q8:summary['points'][str(index)]['bf16_timing']=data['bf16_timing']
                atomic_json(a.output/'summary.json', summary)
            if a.acu:
                for index in profiles:
                    if index not in points or summary['points'][str(index)]['returncode']:
                        continue
                    for storage in ((0,1) if q8 and not a.q8_half2 else (0,)):
                        timing=summary['points'][str(index)]
                        if storage:timing=timing['bf16_timing']
                        medians=timing['median_us']
                        for v in profile_variants(medians,a.q8_exchange,a.q8_fast_tail,a.q8_half2,native,a.q8_paired):
                            # Fresh names never replace a prior raw report.
                            summary['profiles'][f'{index}-{v}-s{storage}']=run(index,v,storage)
                            atomic_json(a.output/'summary.json',summary)
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

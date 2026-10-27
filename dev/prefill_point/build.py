"""Build diagnostic copies without changing historical carriers or their hashes."""
from pathlib import Path
import re

from dev.native_components.build import adapter_source, build_adapter, run_commands
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NATIVE = ROOT/'dev/native_components'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('point carrier source seam changed: '+old)
    return source.replace(old, new, 1)


def fp32_accumulator_source(source):
    """Apply the point experiment's F16-storage/F32-accumulator contract.

    A native checkout need not implement GGML_CUDA_CUBLAS_ACCUMULATOR, even
    when its separately built reference DSO uses FP32 accumulation. Change
    only the generated test TU's F16 traits; never the caller checkout, its
    selector, conversions, output storage or the all-native component runner.
    The carrier still requires this adapter to match the linked DSO on two
    inputs before any timings are admitted.
    """
    pattern = r'(template\s*<\s*>\s*struct\s+batched_mul_mat_traits\s*<\s*GGML_TYPE_F16\s*>\s*\{)(.*?)(\n\};)'
    matches = list(re.finditer(pattern, source, re.S))
    if len(matches) != 1:
        raise ValueError('native F16 traits seam changed; cannot align accumulator')
    match = matches[0]
    original = body = match[2]
    for expected in (r'using\s+cuda_type\s*=\s*half\s*;',
                     r'data_type\s*=\s*CUDA_R_16F\s*;',
                     r'ggml_type_val\s*=\s*GGML_TYPE_F16\s*;',
                     r'ggml_get_to_fp16_cuda\s*\(', r'ggml_get_to_fp16_nc_cuda\s*\('):
        if len(re.findall(expected, body)) != 1:
            raise ValueError('native F16 storage/converter contract changed')

    def replace(pattern, replacement):
        nonlocal body
        body, count = re.subn(pattern, replacement, body)
        if count != 1:
            raise ValueError('native accumulator/scalar traits changed: '+pattern)

    replace(r'(static\s+inline\s+const\s+cublasComputeType_t\s+compute_type\s*=\s*)'
            r'CUBLAS_COMPUTE_(?:16F|32F)(\s*;)', r'\g<1>CUBLAS_COMPUTE_32F\2')
    for scalar in ('alpha', 'beta'):
        # cuBLAS interprets these pointers according to compute_type, NOT
        # data_type. Merely replacing 16F with 32F would reinterpret half bits.
        replace(r'(static\s+inline\s+const\s+)(?:half|float)(\s+'+scalar+r'\s*=)', r'\1float\2')
        replace(r'(static\s+const\s+)(?:half|float)(\s+val\s*=\s*'+scalar+r'\s*;)', r'\1float\2')
    result = source[:match.start(2)]+body+source[match.end(2):]
    # Static inline traits/scalars can have GNU-unique/weak ELF names. Do not
    # let the linked DSO's half alpha satisfy the adapter's float alpha symbol.
    result = re.sub(r'\bbatched_mul_mat_traits\b', 'nc_point_batched_mul_mat_traits', result)
    return result, dict(scope='TEST_ADAPTER_F16_STORAGE_FP32_ACCUMULATOR',
        input='F16', weight='F16', output='F16', accumulator='F32', scalars='STATIC_F32',
        arithmetic_changed=body!=original, selector_changed=False, llama_tree_modified=False,
        private_traits='nc_point_batched_mul_mat_traits',
        linked_backend_required='PASS_TWO_INPUTS',
        source_traits_sha256=digest(original), adapted_traits_sha256=digest(body))


def carrier_source():
    source = (NATIVE/'carrier.cpp').read_text()
    source = once(source, '    void * session = nullptr;',
        '    int (*point_samples)(void *,int,int,int,int,double *);\n'
        '    int (*point_profile)(void *);\n    void * session = nullptr;')
    source = once(source, '        NC_BIND(zero_weight);',
        '        NC_BIND(zero_weight);\n        NC_BIND(point_samples);\n        NC_BIND(point_profile);')
    source = once(source, '            const bool q8_1=native_q8_1_contract(inventory,weights.q);',
        '#include "carrier_contract.cuh"\n'
        '            const bool q8_1=native_q8_1_contract(inventory,weights.q);')
    begin = '            for(size_t i=0;i<inventory.size();++i) {\n                auto stage=inventory[i];'
    end = '            for(int copy=0;copy<point.copies;++copy) {\n                poison(point,copy);nc.check(nc.replay'
    if source.count(begin) != 1:
        raise ValueError('point carrier timer seam changed')
    start = source.index(begin)
    if source[start:].count(end) != 1:
        raise ValueError('point carrier timer end changed')
    stop = source.index(end, start)
    if start >= stop: raise ValueError('point carrier timer boundaries overlap')
    source = source[:start]+'#include "carrier_timing.cuh"\n'+source[stop:]
    return once(source, '            result["fixture"]=request["raw_fixture_receipt"];',
        '            result["fixture"]=request["raw_fixture_receipt"];\n'
        '            result["native_precision_contract"]=providers ? "F16_STORAGE_F32_ACCUMULATOR" : "LINKED_NATIVE_BODY";\n'
        '            result["point_timings"]=point_timings;result["inventory"]=inventory;\n'
        '            result["scope"]="ONE_POINT_ORDINARY_EVENTS_NOT_MODEL";')


def build_native(args, target, env):
    # --reuse-build applies to device payloads, not the small host adapter.
    # A new run builds against the supplied caller/flags and records fresh
    # hashes. Never mutate/relabel an old adapter receipt when this harness
    # changes; completed runs still validate their own receipts on resume.
    base_dir = target
    _, base = build_adapter(args.llama_source, args.llama_build, args.sdk, base_dir)
    target.mkdir(parents=True, exist_ok=True)
    original = (base_dir/'native-dispatch.cu').read_text()
    if original != adapter_source(args.llama_source)[0]:
        raise ValueError('generated native dispatcher differs')
    # The base is an immutable copy of the checkout. Align the point adapter
    # explicitly instead of relying on an env var unsupported by some callers.
    adapted, precision = fp32_accumulator_source(original)
    source = target/'point-dispatch.cu'
    source.write_text(adapted+'\n#include "'+str(HERE/'native.cuh')+'"\n')
    adapter, obj = target/'libpoint-native.so', target/'point-dispatch.o'
    rename = {str(base_dir/'native-dispatch.cu'):str(source), str(base_dir/'native-dispatch.o'):str(obj),
              str(base_dir/'libnative-components.so'):str(adapter)}
    for word in base['commands'][0]:
        if word.startswith('-I') and word.endswith('/dev/native_components'):
            rename[word] = '-I'+str(NATIVE)
    commands = [[rename.get(word, word) for word in row] for row in base['commands']]
    run_commands(commands, Path(base['identity']['compiler_workdir']), target/'point-adapter.log', env=env)
    src, binary = target/'point-carrier.cpp', target/'point-carrier'
    src.write_text(carrier_source())
    command = ['g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
        *['-I'+str(p) for p in (HERE, NATIVE, args.llama_source/'ggml/include',
                               args.llama_source/'ggml/src', args.llama_source/'vendor')],
        str(src), '-L'+str(args.llama_build/'bin'), '-Wl,-rpath,'+str(args.llama_build/'bin'),
        *['-Wl,-rpath-link,'+str(p) for p in (args.llama_build/'bin', args.sdk/'lib',
            args.sdk/'targets/x86_64-linux/lib', args.sdk/'CUDA_SDK/targets/x86_64-linux/lib')],
        '-lggml-cuda', '-lggml-base', '-ldl', '-pthread', '-o', str(binary)]
    run_commands([command], target, target/'point-carrier.log', env=env)
    receipt = dict(base=base, precision=precision, adapter_sha256=sha(adapter), carrier_sha256=sha(binary),
        sources={p.name:sha(p) for p in (*HERE.glob('*.cuh'), HERE/'events.hpp', source, src)},
        commands=commands+[command], production_changed=False, legacy_carrier_changed=False)
    atomic_json(target/'point-build.json', receipt)
    return adapter, binary, receipt

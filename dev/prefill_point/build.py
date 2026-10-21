"""Build diagnostic copies without changing historical carriers or their hashes."""
from pathlib import Path

from dev.native_components.build import build_adapter, run_commands
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NATIVE = ROOT/'dev/native_components'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('point carrier source seam changed: '+old)
    return source.replace(old, new, 1)


def carrier_source():
    source = (NATIVE/'carrier.cpp').read_text()
    source = once(source, '    void * session = nullptr;',
        '    int (*point_samples)(void *,int,int,int,int,double *);\n    void * session = nullptr;')
    source = once(source, '        NC_BIND(zero_weight);',
        '        NC_BIND(zero_weight);\n        NC_BIND(point_samples);')
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
        '            result["point_timings"]=point_timings;result["inventory"]=inventory;\n'
        '            result["scope"]="ONE_POINT_ORDINARY_EVENTS_NOT_MODEL";')


def build_native(args, target, env):
    _, base = build_adapter(args.llama_source, args.llama_build, args.sdk, target)
    # Keep the admitted adapter unchanged. Recompile only its tiny host
    # dispatcher with the diagnostic timer appended, using the SAME flags.
    source = target/'point-dispatch.cu'
    source.write_text((target/'native-dispatch.cu').read_text()+'\n#include "'+str(HERE/'native.cuh')+'"\n')
    adapter, obj = target/'libpoint-native.so', target/'point-dispatch.o'
    rename = {str(target/'native-dispatch.cu'):str(source), str(target/'native-dispatch.o'):str(obj),
              str(target/'libnative-components.so'):str(adapter)}
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
    receipt = dict(base=base, adapter_sha256=sha(adapter), carrier_sha256=sha(binary),
        sources={p.name:sha(p) for p in (*HERE.glob('*.cuh'), HERE/'events.hpp', source, src)},
        commands=commands+[command], production_changed=False, legacy_carrier_changed=False)
    atomic_json(target/'point-build.json', receipt)
    return adapter, binary, receipt

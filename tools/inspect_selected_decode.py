#!/usr/bin/env python3
"""Compare promoted native instructions with the measured component images."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.generate_decode_selection import EVIDENCE, FIELDS, effective
from quactlize.runtime.compiler import sha
from tools.build_kpack_dequant import resource_usage


def instruction_bytes(text, symbol):
    marker = 'Disassembly of section .text.kernel.' + symbol + ':'
    if text.count(marker) != 1:
        raise ValueError('missing/duplicate selected ISA section: ' + symbol)
    text = text.split(marker, 1)[1].split('Disassembly of section ', 1)[0]
    lines = re.findall(r'^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2}[ \t]+)+)', text, re.MULTILINE)
    if not lines:
        raise ValueError('native instruction bytes missing')
    return bytes.fromhex(' '.join(lines))


def target_pattern(w):
    if w['point']['paired']:
        return 'quactlize::fusion::simt_gate_up_q8_tile16('
    q, mode, n, k, e, top, ch, _, compute = w['key']
    b = w['body']
    args = [q, mode, n, k, e, top, ch, compute, *[b[f] for f in FIELDS], b['changes'],
            str(b['hoist']).lower(), str(b['fixed']).lower()]
    return 'quactlize::execution::simt::measured_decode_kernel<' + ','.join(map(str, args)) + '>('


def reviewed_codegen(point, current, previous, resources, before):
    # Same Q5 source body and complete-call recipe, but HGCC lowered five
    # address mad+move pairs to adds in the combined execution translation unit.
    # This is a bounded static review, NOT a new device timing admission.
    if point != 'tp2-q5-down' or resources != dict(registers=114, scalar_registers=128,
            shared_allocation_field=2, stack_bytes=0) or before != dict(registers=114,
            scalar_registers=144, shared_allocation_field=2, stack_bytes=0):
        return False
    ops = lambda s: Counter(re.findall(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}[ \t]+)+([a-z][\w.]+)\b', s, re.MULTILINE))
    a, b = ops(current), ops(previous)
    return a-b == Counter({'v.add.i32': 5}) and b-a == Counter({'s.mov.b32': 5, 'v.madl.i32': 5})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('sdk', 'execution', 'paired', 'reference', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(EVIDENCE.read_text());effective(evidence)
    old = json.loads((args.reference/'native-inspection.json').read_text())
    frozen = json.loads((args.reference/'manifest.json').read_text())
    if sha(args.reference/'native-inspection.json') != frozen['payloads']['native-inspection.json']:
        raise ValueError('reference inspector receipt differs')
    paths = {'execution': args.execution/'libquactlize_ppu_execution.so',
             'paired': args.paired/'libquactlize_ppu_gate_up.so'}
    inspector = args.sdk/'bin/hgobjdump'
    names = {}
    for key, path in paths.items():
        text = subprocess.check_output([inspector, '--dump-resource-usage=all', path], text=True)
        mangled = sorted(set(re.findall(r'^Func \d+ (\S+) RESOURCE INFO:', text, re.MULTILINE)))
        demangled = subprocess.check_output(['c++filt'], input='\n'.join(mangled), text=True).splitlines()
        names[key] = dict(zip(mangled, demangled))
    args.output.mkdir(parents=True, exist_ok=False)

    def check(w):
        point = w['point']['name']
        key = 'paired' if w['point']['paired'] else 'execution'
        pattern = target_pattern(w)
        symbols = [s for s, n in names[key].items() if pattern in re.sub(r'\s+', '', n)]
        if len(symbols) != 1:
            raise ValueError('missing/duplicate promoted implementation: ' + point)
        symbol = symbols[0]
        reference = old[point][w['arm']]
        old_path = args.reference/(point+'.so')
        if sha(old_path) != frozen['payloads'][old_path.name]:
            raise ValueError('reference image differs: ' + point)
        raw = subprocess.check_output([inspector, '--dump-isa', '--dump-function='+symbol, paths[key]], text=True)
        previous = subprocess.check_output([inspector, '--dump-isa', '--dump-function='+reference['symbol'], old_path], text=True)
        instructions = instruction_bytes(raw, symbol)
        before = instruction_bytes(previous, reference['symbol'])
        resources = resource_usage(subprocess.check_output(
            [inspector, '--dump-resource-usage='+symbol, paths[key]], text=True))[symbol]
        result = dict(point=point, symbol=symbol, library_sha256=sha(paths[key]),
                      reference_library_sha256=sha(old_path), instruction_bytes=len(instructions),
                      instruction_sha256=hashlib.sha256(instructions).hexdigest(),
                      reference_instruction_sha256=hashlib.sha256(before).hexdigest(),
                      instructions_identical=instructions == before, resources=resources,
                      reference_resources=reference['resources'], resources_identical=resources == reference['resources'])
        result['static_review'] = ('EXACT' if result['instructions_identical'] and result['resources_identical'] else
            'Q5_ADDRESS_LOWERING_DEVICE_PERFORMANCE_PENDING' if reviewed_codegen(
                point, raw, previous, resources, reference['resources']) else 'REJECT')
        (args.output/(point+'.isa.txt')).write_text(raw)
        print('SELECTED_DECODE_ISA '+json.dumps({k: result[k] for k in
              ('point', 'instructions_identical', 'resources_identical', 'static_review')}), flush=True)
        return result

    rows = [w for w in evidence['rows'] if w['automatic'] and w['implementation'] == 'constant-split']
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, rows))
    result = dict(scope='STATIC_NATIVE_IDENTITY_NOT_MODEL_PERFORMANCE', records=results,
                  exact_instructions=sum(r['instructions_identical'] for r in results), points=len(results))
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    if any(r['static_review'] == 'REJECT' for r in results):
        raise ValueError('promoted native image differs; review before publication')


if __name__ == '__main__':
    main()

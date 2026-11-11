"""Actual runtime query/prepare/dynamic-run with independent GGUF golden."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
import numpy as np
from ci.support.fixture import compare, matmul_reference, planes, raw_weight
from ci.support.graph import Replay
from ppu.support import Arena, bind, packed
from quactlize.runtime.native import Call, Module, Recipe, Resources, checked
from tools.ppu15.check import Pack, setup
from tools.ppu15.tc_plan import MS, NK, SPLITS


def run_case(sdk, module, device, weights, pointers, metadata, zero, gold, arr, m, split):
    n, k = NK
    parent = module.record['parent']
    with Arena(sdk) as arena:
        stream = arena.stream()
        # A shape-only handle deliberately cannot retain these addresses.
        shape = Call(version=1, size=C.sizeof(Call), m=m, n=n, k=k, experts=1,
                     group_size=arr.group_size, device=device['ordinal'],
                     compute_units=device['compute_units'], mapping_id=arr.mapping_id, stream=stream)
        recipe = Recipe(1, C.sizeof(Recipe), 0, split, 0)
        resources = Resources()
        checked(module.query(C.byref(shape), C.byref(recipe), C.byref(resources)), 'TC resource query')
        a = arena.buffer(m*k*2); out = arena.buffer(m*n*2)
        scratch = arena.buffer(resources.workspace_bytes) if resources.workspace_bytes else None
        call = Call.from_buffer_copy(shape)
        call.a, call.output = a.ptr, out.ptr
        call.low, call.high, call.metadata, call.zero = pointers[0], pointers[1], metadata, zero
        call.workspace = scratch.ptr if scratch else None
        call.workspace_bytes = scratch.size if scratch else 0
        prepare = bind(module.lib, 'quactlize_kpack_prepare_dynamic_v1',
                       [C.POINTER(Call), C.POINTER(Recipe), C.POINTER(C.c_void_p)])
        run = bind(module.lib, 'quactlize_kpack_run_dynamic_v1', [C.c_void_p, C.POINTER(Call)])
        handle = C.c_void_p(); graph = None; worst = 0.
        checked(prepare(C.byref(shape), C.byref(recipe), C.byref(handle)), 'TC prepare dynamic')
        launch = lambda: run(handle, C.byref(call))
        try:
            for replay in range(3):
                context = dict(parent=parent['symbol'], m=m, n=n, k=k, split=split, replay=replay)
                print('ZW890_TC_CASE_START '+json.dumps(context), flush=True)
                source = np.random.default_rng(512+m*31+replay).uniform(-.25, .25, (m, k)).astype('f2')
                if replay == 0:
                    source.fill(0)
                    # Visit both Split-K slices and distinct K-pack groups.
                    source[np.arange(m), (np.arange(m)*137+7) % k] = 1
                a.write(source); out.poison(); arena.sync()
                if replay == 1: graph = Replay(sdk, stream, launch, 1)
                checked(graph() if graph else launch(), 'native PPU15 TC run')
                arena.sync()
                got = np.frombuffer(out.bytes(), 'f2').reshape(m, n)
                want, denom = matmul_reference(source, gold, 'f16', output_compute='f16')
                if replay == 0:
                    np.testing.assert_array_equal(got, want, err_msg=json.dumps(context))
                result = compare(got, want, denom)
                worst = max(worst, result['relative_l1_error'])
                a.guard()
                if scratch: scratch.guard()
                for buffer in weights.buffers: buffer.guard()
            return dict(parent=parent['symbol'], q=parent['qtype'], route=parent['route'],
                        m=m, n=n, k=k, split=split, replays=2, exact_one_hot=True,
                        max_absdot_error=worst, workspace_bytes=resources.workspace_bytes,
                        shared_bytes=resources.shared_bytes, occupancy=resources.occupancy, status='PASS')
        finally:
            arena.sync()
            if graph: graph.close()
            module.destroy(handle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parent', required=True)
    a = p.parse_args()
    sdk, lib, device = setup(a.sdk, a.output)
    (a.output/'device.json').write_text(json.dumps(device, indent=2)+'\n')
    records = json.loads((a.output/'modules.json').read_text())
    record = next(r for r in records if r['parent']['symbol'] == a.parent)
    module = Module(record)
    assert module.device_identity() == dict(device=device['name'], compute_units=device['compute_units'],
                                            ordinal=device['ordinal'])
    q = record['parent']['qtype']; n, k = NK
    with Arena(sdk) as weights, (a.output/(a.parent+'.jsonl')).open('x') as log:
        arr, _, pointers, gold = packed(weights, Pack(lib), q, n=n, k=k)
        metadata, zero = pointers[2], None
        if record['parent']['route'] == 'sf-dense' and q != 8:
            # SF's host-expanded fixture isolates the consumer; expansion
            # timing and the production expansion kernel are not claimed here.
            raw, _ = raw_weight(q, n, k, 913+q)
            expanded = planes(raw, q)
            metadata = weights.upload(expanded['scale']).ptr
            zero = weights.upload(expanded['zero']).ptr
        weights.sync()
        for m in MS:
            for split in SPLITS:
                result = run_case(sdk, module, device, weights, pointers, metadata, zero, gold[0], arr, m, split)
                line = json.dumps(result)
                log.write(line+'\n'); log.flush()
                print('ZW890_TC_CASE '+line, flush=True)


if __name__ == '__main__':
    main()

"""Idle-guarded physical H800 comparison, not a production dispatch tuner."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CONTRACT = 'GPTQ_SYM_INT4_G128_F16_V1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def device_clocks(device):
    values=subprocess.check_output(['nvidia-smi','--id='+device,
        '--query-gpu=clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True,timeout=10)
    return dict(zip(('sm_mhz','memory_mhz'),map(int,values.strip().split(','))))


class RefCall(C.Structure):
    _fields_ = [(x, C.c_int32) for x in ('dtype', 'config', 'm', 'n', 'k')] + [
        (x, C.c_void_p) for x in ('a', 'b', 'scales', 'zeros', 'output', 'stream')]


class Reference:
    def __init__(self, directory, arm):
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest.get('schema') != 2 or manifest.get('contract') != CONTRACT:
            raise ValueError('not an admitted GPTQ comparison build; old GGUF-subset bundles are rejected')
        row = manifest['libraries'][arm]
        library = directory / row['file']
        if sha(library) != row['sha256']:
            raise ValueError('reference library hash mismatch')
        self.library = C.CDLL(str(library), mode=C.RTLD_LOCAL)
        self.fn = {}
        for name, result, arguments in (
            ('error', C.c_char_p, []), ('contract', C.c_char_p, []), ('config_count', C.c_int, []),
            ('config_name', C.c_char_p, [C.c_int]),
            ('prepare', C.c_int, [C.POINTER(RefCall), C.POINTER(C.c_void_p)]),
            ('execute', C.c_int, [C.c_void_p]), ('destroy', None, [C.c_void_p]),
            ('workspace', C.c_size_t, [C.c_void_p]),
            ('share_workspace', C.c_int, [C.c_void_p, C.c_void_p]),
            ('device', C.c_int, [C.POINTER(C.c_int), C.POINTER(C.c_int)]),
            ('pack', C.c_int, [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
        ):
            f = getattr(self.library, 'q17_ref_' + name)
            f.restype, f.argtypes = result, arguments
            self.fn[name] = f
        if self.fn['contract']().decode() != CONTRACT:
            raise ValueError('library quantization contract differs')

    def check(self, rc):
        if rc:
            raise ValueError(self.fn['error']().decode())

    def prepare(self, call):
        ptr = C.c_void_p()
        self.check(self.fn['prepare'](C.byref(call), C.byref(ptr)))
        return ptr


def check_output(torch, storage, expected, m, n):
    actual = storage.cpu().float()
    if not torch.all(actual[:64] == -1232) or not torch.all(actual[64+m*n:] == -1232):
        raise ValueError('output guard overwritten')
    actual = actual[64:64+m*n].reshape(m, n)
    if not torch.isfinite(actual).all():
        raise ValueError('nonfinite or unwritten output')
    maximum = float((actual-expected).abs().max())
    relative = float(torch.linalg.vector_norm(actual-expected) / torch.linalg.vector_norm(expected).clamp_min(1e-9))
    # Allow output rounding at summation ties, not a relaxed dequant oracle.
    tolerance = .003 if storage.dtype == torch.float16 else .012
    if relative > tolerance or maximum > tolerance * max(1., float(expected.abs().max())):
        raise ValueError(f'CPU typed FP64 dot mismatch relative={relative} max_abs={maximum}')
    return {'relative_l2': relative, 'max_abs': maximum, 'guards': 'PASS'}


def child(args):
    if os.environ.get('Q17_BENCH_GUARDED') != '1':
        raise ValueError('use the idle-guarded parent runner')
    import numpy as np
    import torch
    from fixture import make_fixture, marlin_metadata
    torch.set_num_threads(8)
    dtype = torch.float16 if args.dtype == 0 else torch.bfloat16
    fixture = make_fixture(args.m, args.n, args.k, dtype, args.weights)
    refs = {name: Reference(args.references, name)
            for name in sorted(set(args.arms + ['marlin']))}
    stream = torch.cuda.Stream()
    l2, sms = C.c_int(), C.c_int()
    refs['marlin'].check(refs['marlin'].fn['device'](C.byref(l2), C.byref(sms)))
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError('physical SM90 required')
    # CPU construction precedes admission, and no timed work precedes this gate.
    torch.cuda.synchronize()
    print('PPU17_BENCH_READY', flush=True)
    if sys.stdin.readline().strip() != 'R':
        raise ValueError('idle guard did not admit this PID')
    rows = []
    arms = list(reversed(args.arms)) if args.reverse else args.arms
    with torch.cuda.stream(stream):
        a = fixture['a'].cuda()
        storage = torch.full((64+args.m*args.n+64,), -1232, dtype=dtype, device='cuda')
        output = storage[64:64+args.m*args.n]
        for arm in arms:
            ref = refs[arm]
            packed = np.zeros(args.n*args.k//2, dtype=np.uint8)
            ref.check(ref.fn['pack'](args.dtype, args.n, args.k,
                fixture['q'].ctypes.data, packed.ctypes.data))
            scales = fixture['scale']
            if arm == 'marlin':
                scales = marlin_metadata(scales)
            host = [torch.from_numpy(packed), scales]
            configs = [(i, None) for i in range(ref.fn['config_count']())]
            if args.configs:
                wanted = {int(x.split(':')[1]) for x in args.configs if x.split(':')[0] == arm}
                configs = [(i, c) for i, c in configs if i in wanted]
            if args.reverse:
                configs.reverse()
            weight_bytes = sum(x.numel()*x.element_size() for x in host)
            copies = math.floor(2.25*l2.value/weight_bytes)+1 if args.cache == 'rotating' else 1
            if copies > 512:
                raise ValueError('fixture ring exceeds bounded limit')
            weights = [[x.cuda() for x in host] for _ in range(copies)]
            for index, config in configs:
                plans = []
                try:
                    for weight in weights:
                        call = RefCall(args.dtype, index, args.m, args.n, args.k,
                            a.data_ptr(), weight[0].data_ptr(), weight[1].data_ptr(),
                            None, output.data_ptr(), stream.cuda_stream)
                        plans.append(ref.prepare(call))
                        if len(plans)>1:
                            ref.check(ref.fn['share_workspace'](plans[-1], plans[0]))
                    def execute(i):
                        ref.check(ref.fn['execute'](plans[i]))
                    proof = []
                    for i in sorted({0, copies//2, copies-1}):
                        output.fill_(float('nan'))
                        execute(i)
                        stream.synchronize()
                        proof.append(check_output(torch, storage, fixture['expected'], args.m, args.n))
                    launches = copies * max(1, math.ceil(32/copies))
                    graph = torch.cuda.CUDAGraph()
                    stream.synchronize()
                    with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                        for i in range(launches):
                            execute(i % copies)
                    for _ in range(3):
                        graph.replay()
                    stream.synchronize()
                    proof.append(check_output(torch, storage, fixture['expected'], args.m, args.n))
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    # Three short graphs can finish before DVFS reaches its
                    # steady state. Warm THIS rotating ring for >=250ms on
                    # the device, without changing any driver clock setting.
                    begin.record(stream)
                    graph.replay()
                    end.record(stream)
                    end.synchronize()
                    graph_ms=begin.elapsed_time(end)
                    if not math.isfinite(graph_ms) or graph_ms<=0:
                        raise ValueError('invalid warmup graph duration')
                    extra_warmups=math.ceil(250/graph_ms)
                    if extra_warmups>20000:
                        raise ValueError('steady warmup exceeds bounded budget')
                    for _ in range(extra_warmups):
                        graph.replay()
                    stream.synchronize()
                    clocks_before=device_clocks(args.device)
                    samples = []
                    for _ in range(11):
                        begin.record(stream)
                        graph.replay()
                        end.record(stream)
                        end.synchronize()
                        samples.append(begin.elapsed_time(end)*1000/launches)
                    clocks_after=device_clocks(args.device)
                    if clocks_before!=clocks_after:
                        raise ValueError(f'clocks changed inside measurement: {clocks_before} -> {clocks_after}')
                    proof.append(check_output(torch, storage, fixture['expected'], args.m, args.n))
                    row = {'arm': arm, 'config': index, 'status': 'PASS', 'correctness': proof,
                           'weight_bytes': weight_bytes, 'weight_copies': copies,
                           'launches_per_graph': launches, 'samples_us': samples,
                           'median_us': statistics.median(samples), 'warmup_graphs': 4+extra_warmups,
                           'warmup_min_device_ms':250,'clocks_before':clocks_before,'clocks_after':clocks_after}
                    row['config_detail'] = ref.fn['config_name'](index).decode()
                    row['workspace_per_copy'] = ref.fn['workspace'](plans[0])
                    rows.append(row)
                    print('PPU17_REFERENCE_CONFIG ' + json.dumps(row), flush=True)
                    graph.reset()
                finally:
                    stream.synchronize()
                    # Shared scratch owner is the first plan; destroy it last.
                    for plan in reversed(plans):
                        ref.fn['destroy'](plan)
            del weights
    print('PPU17_BENCH ' + json.dumps({'status': 'PASS', 'rows': rows,
          'shape': [args.m, args.n, args.k], 'dtype': args.dtype,
          'l2_bytes': l2.value, 'sm_count': sms.value, 'cache': args.cache,
          'scope': 'GPTQ_SYMMETRIC_FULL_PREPARED_CALL', 'contract': CONTRACT,
          'input_source': fixture['source'],
          'logical_codes_sha256': hashlib.sha256(fixture['q'].tobytes()).hexdigest(),
          'scale_bits_sha256': hashlib.sha256(fixture['scale'].numpy().tobytes()).hexdigest(),
          'metadata': 'ORIGINAL_FP16_GROUP128_NO_SCALE_EXPANSION'}), flush=True)


def validate(row):
    if row.get('status') != 'PASS' or not row.get('rows'):
        raise ValueError('empty or failed comparison')
    if row.get('contract') != CONTRACT or row.get('scope') != 'GPTQ_SYMMETRIC_FULL_PREPARED_CALL':
        raise ValueError('wrong comparison contract; do not reuse the rejected GGUF-subset results')
    for item in row['rows']:
        samples = item['samples_us']
        if len(samples) != 11 or any(not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError('invalid device timing samples')
        if statistics.median(samples) != item['median_us']:
            raise ValueError('sample median differs')
        if item['launches_per_graph'] % item['weight_copies']:
            raise ValueError('incomplete rotating ring')
        if row['cache'] == 'rotating' and item['weight_copies']*item['weight_bytes'] <= 2.25*row['l2_bytes']:
            raise ValueError('weight ring too small')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--backend-source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, help='optional original GPTQ arrays, never requantized')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-uuid', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--dtype', type=int, choices=(0,), default=0)
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--k', type=int, default=2048)
    parser.add_argument('--cache', choices=('warm', 'rotating'), default='rotating')
    parser.add_argument('--arms', nargs='+', choices=('machete', 'marlin', 'quactlize'), default=['machete', 'marlin', 'quactlize'])
    parser.add_argument('--configs', nargs='+', help='bounded reconfirmation e.g. machete:0 marlin:1 quactlize:2')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    sys.path.insert(0, str(args.backend_source))
    import run_benchmark as guard
    guard.validate_result = validate
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('result directory must be fresh')
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.device, 'Q17_BENCH_GUARDED': '1'}
    result = guard.run_point(command, env, args.output/'run.log', args.device, args.expected_uuid, 900)
    result['source_sha256'] = {name: sha(HERE/name) for name in ('compare.py', 'fixture.py')}
    result['reference_manifest_sha256'] = sha(args.references/'manifest.json')
    result['importer_sha256'] = sha(HERE.parents[2]/'tools/dump_real_weights.py')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PPU17_REFERENCE_DONE ' + result['status'] + ' ' + str(result.get('error')), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Numerical and rotating-weight SIMT gate. Run one qtype per fresh process."""
import argparse
import ctypes as C
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt import fixture, spec
from dev.gemv_simt.access import pattern
from dev.gemv_simt.native import Call, Library, Runtime, Graph, LegacyConfig, Sizes, Arrangement, arrangement, checked
from dev.gemv_simt.build import sha


class Bench:
    def __init__(self, rt, lib, w, tokens, mode, channels=1, storage=1, copies=1):
        self.rt, self.lib, self.w = rt, lib, w
        self.start = len(rt.allocations)
        self.tokens, self.mode, self.channels, self.storage = tokens, mode, channels, storage
        self.data = fixture.inputs(w, tokens, mode, channels)
        data = self.data
        self.a = rt.upload(data['a'].astype('<f4' if storage else '<f2'))
        self.ids, self.offsets = rt.upload(data['ids']), rt.upload(data['offsets'])
        self.output_bytes = (data['rows']*(w.n+8)+8)*4
        self.output = rt.allocate(self.output_bytes)
        self.workspace_bytes = data['rows']*w.n*8*4
        self.workspace = rt.allocate(self.workspace_bytes+32)
        self.call = Call(version=1, size=C.sizeof(Call), qtype=w.q, n=w.n, k=w.k,
            experts=1 if mode==0 else w.experts, rows=data['rows'], mode=mode, input_type=storage,
            channels=channels, topk=data['topk'], a_row_stride=w.k+8,
            a_token_stride=channels*(w.k+8), ids_stride=11, out_row_stride=w.n+8,
            a=self.a, offsets=self.offsets, ids=self.ids, output=self.output+16,
            workspace=self.workspace+16, workspace_bytes=self.workspace_bytes, stream=rt.stream.value)
        self.calls = []
        for _ in range(copies):
            c = Call.from_buffer_copy(self.call)
            for name in ('low', 'high', 'units'):
                plane = w.planes[name]
                if mode==0 and plane.size:
                    plane = plane[:1]
                setattr(c, name, rt.upload(plane))
            self.calls.append(c)
        self.call = self.calls[0]
        rt.sync()

    def update(self, repeat):
        self.data = fixture.inputs(self.w, self.tokens, self.mode, self.channels, repeat)
        self.rt.copy(self.a, self.data['a'].astype('<f4' if self.storage else '<f2'))
        if self.ids:
            self.rt.copy(self.ids, self.data['ids'])
        self.rt.sync()

    def poison(self):
        self.rt.fill(self.output, self.output_bytes)
        self.rt.fill(self.workspace, self.workspace_bytes+32)

    def output_check(self):
        host = self.rt.download(self.output, self.output_bytes).view('<u4')
        rows = self.data['rows']
        body = host[4:-4].reshape(rows, self.w.n+8)
        if not (np.all(host[:4]==0xa5a5a5a5) and np.all(host[-4:]==0xa5a5a5a5)
                and np.all(body[:, self.w.n:]==0xa5a5a5a5)):
            raise ValueError('output guard/stride changed')
        work = self.rt.download(self.workspace, self.workspace_bytes+32).view('<u4')
        if not (np.all(work[:4]==0xa5a5a5a5) and np.all(work[-4:]==0xa5a5a5a5)):
            raise ValueError('workspace guard changed')
        got = body[:, :self.w.n].view('<f4').copy()
        if not np.isfinite(got).all():
            raise ValueError('nonfinite output')
        error = float(np.max(np.abs(got.astype('f8')-self.data['gold']) / self.data['denom']))
        if error >= 0.005:
            raise ValueError(f'official GGUF dot differs: {error:.9g}')
        return got, error

    def correctness(self, config):
        call = self.lib.prepare(self.call, config)
        self.poison()
        checked(call(), 'correctness full call')
        self.rt.sync()
        return self.output_check()

    def replay_and_negative(self, config):
        graph = Graph(self.rt, [self.lib.prepare(self.call, config)])
        errors = []
        try:
            for repeat in (1, 2, 3):
                self.update(repeat)
                self.poison()
                checked(self.rt.GraphLaunch(graph.instance, self.rt.stream), 'changing-input replay')
                self.rt.sync()
                errors.append(self.output_check()[1])
            # Change the same resident A pointer after capture. The independent
            # nonzero oracle must reject that graph's now-zero projection.
            self.rt.fill(self.a, self.data['a'].size*(4 if self.storage else 2), 0)
            self.poison()
            checked(self.rt.GraphLaunch(graph.instance, self.rt.stream), 'zero-A negative')
            self.rt.sync()
            try:
                self.output_check()
            except ValueError as e:
                if not str(e).startswith('official GGUF dot differs:'):
                    raise
            else:
                raise ValueError('zero-A negative was not detected')
            self.update(0)
            return dict(replays=3, errors=errors, negative='ZERO_A_REJECTED')
        finally:
            graph.close()

    def close(self):
        self.rt.release_after(self.start)


def numeric(a, rt, lib):
    # K512 covers both members of paired Q3/Q6 units, Q5's independent high
    # plane transpose, and empty Split-K partitions in high-worker recipes.
    w = fixture.weights(a.qtype, 256, 512, 16)
    records, replays = [], []
    candidates = lib.candidates(a.qtype)
    cases = [(m, mode, ch) for m in range(1, 9) for mode, ch in ((0,1),(2,1),(2,8))]
    cases += [(7,1,1), (8,1,1)]
    total = len(cases)*2*len(candidates)
    started = time.monotonic()
    for tokens, mode, channels in cases:
        f16 = {}
        for storage in (0,1):
            bench = Bench(rt, lib, w, tokens, mode, channels, storage)
            try:
                for config in candidates:
                    try:
                        got, error = bench.correctness(config)
                    except Exception as e:
                        raise ValueError(f'q={a.qtype} tokens={tokens} mode={mode} channels={channels} '
                                         f'input={storage} config={config.key}: {e}') from e
                    exact = True
                    if not storage:
                        f16[config.key] = got.view('<u4').copy()
                    else:
                        exact = bool(np.array_equal(got.view('<u4'), f16[config.key]))
                    if not exact:
                        raise ValueError('F16-exact A differs between F16/F32 endpoints')
                    row = dict(qtype=a.qtype, tokens=tokens, mode=mode, channels=channels,
                        input=storage, config=config.key, error=error, f16_f32_exact=exact, status='PASS')
                    records.append(row)
                    a.journal.write(json.dumps(row)+'\n')
                if storage and mode==2:
                    for split in spec.SPLITS:
                        config = next(c for c in candidates if c.split==split)
                        replays.append(dict(tokens=tokens, channels=channels, config=config.key,
                                            **bench.replay_and_negative(config)))
                print(f'SIMT_NUMERIC_PROGRESS q={a.qtype} completed={len(records)}/{total} '
                      f'elapsed_s={time.monotonic()-started:.1f}', flush=True)
                a.journal.flush()
            finally:
                bench.close()
    return dict(status='PASS', cases=len(records), expected=total, records=records, replays=replays,
                scope='ALL_COMPILED_RECIPES_M1_TO_8_DENSE_INDEXED_AND_COMPACT_RAW_GGUF_ORACLE')


class Baseline:
    def __init__(self, directory, lib):
        record = lib.manifest['baseline']
        path = directory / record['library']
        if sha(path)!=record['sha256']:
            raise ValueError('baseline image differs')
        self.lib = C.CDLL(str(path.resolve()), mode=C.RTLD_LOCAL)
        self.fn = {}
        for arm in ('scalar','pair'):
            stem = 'quactlize_kpack_gemv_'+('pair_' if arm=='pair' else '')
            q, run = getattr(self.lib, stem+'query_v1'), getattr(self.lib, stem+'run_v1')
            q.argtypes, q.restype = [C.POINTER(Call), C.POINTER(LegacyConfig), C.POINTER(Arrangement), C.POINTER(Sizes)], C.c_int
            run.argtypes, run.restype = q.argtypes[:-1], C.c_int
            self.fn[arm] = q, run

    def candidates(self, q):
        from itertools import product
        result = [('scalar',c,w,s) for c,w,s in product((16,32),(4,8),(1,4))]
        result += [('pair',c,w,s) for c,w,s in product((16,32),(2,4,8),spec.SPLITS)]
        if q==8:  # Scalar public dispatch is already Pair=true for Q8.
            result = [x for x in result if x[0]=='pair']
        return result

    def prepare(self, call, config):
        arm,c,w,s = config
        f, a, sizes = LegacyConfig(c,w,s), arrangement(call.qtype), Sizes()
        query, run = self.fn[arm]
        checked(query(C.byref(call),C.byref(f),C.byref(a),C.byref(sizes)), 'baseline query')
        if sizes.workspace_bytes>call.workspace_bytes:
            raise ValueError('baseline workspace capacity differs')
        return lambda: run(C.byref(call),C.byref(f),C.byref(a))


class Q4Control:
    def __init__(self, directory, n, k):
        manifest=json.loads((directory/'manifest.json').read_text())
        hits=[r for r in manifest['controls'] if (r['n'],r['k'])==(n,k)]
        if len(hits)!=1: raise ValueError('requested Q4 incumbent shape was not compiled')
        self.record=hits[0]
        path=directory/self.record['library']
        if sha(path)!=self.record['sha256']: raise ValueError('Q4 incumbent binary differs')
        self.lib=C.CDLL(str(path.resolve()),mode=C.RTLD_LOCAL)
        self.run=self.lib.simt_q4_control
        self.run.argtypes,self.run.restype=[C.POINTER(Call),C.c_int],C.c_int

    def prepare(self, call, index):
        return lambda:self.run(C.byref(call),index)


def performance(a, rt, lib, identity):
    l2 = identity['l2_bytes'] or a.l2_bytes
    if not l2 or l2<1:
        raise ValueError('positive verified L2 size required; no guessed GPU property')
    if identity['l2_bytes'] and a.l2_bytes and identity['l2_bytes']!=a.l2_bytes:
        raise ValueError('L2 override differs from runtime probe')
    mode = 0 if a.mode=='dense' else 2
    w = fixture.weights(a.qtype,a.n,a.k,1 if mode==0 else a.experts)
    resident = sum(w.planes[n].nbytes for n in ('low','high','units'))
    active = 1 if mode==0 else len(np.unique(fixture.inputs(w,a.tokens,mode,a.channels)['ids'][:,:8]))
    # Unselected experts do not evict the weights actually read. Size the
    # ring by the union of active experts, not the resident allocation.
    distinct = resident//w.experts*active
    copies = math.ceil(2.25*l2/distinct) if a.cache=='rotating' else 1
    traversals = max(1, math.ceil(32/copies))
    bench = Bench(rt,lib,w,a.tokens,mode,a.channels,1,copies)
    baseline = Baseline(a.bundle,lib)
    pool = [('new',c,c.key) for c in lib.candidates(a.qtype)]
    pool += [('old',c,f'old-{c[0]}-c{c[1]}-w{c[2]}-s{c[3]}') for c in baseline.candidates(a.qtype)]
    providers={'new':lib,'old':baseline}
    control=None
    if a.qtype==12 and a.q4_controls:
        control=Q4Control(a.q4_controls,a.n,a.k);providers['q4-incumbent']=control
        pool += [('q4-incumbent',i,'q4-incumbent-'+ '-'.join(map(str,c)))
                 for i,c in enumerate(control.record['configs'])]
    # Every compiled geometry is screened; staged confirmations always
    # retain each implementation's top two, not only the global winner.
    screened, confirmed = [], []
    def graph_for(arm,c):
        provider = providers[arm]
        first = provider.prepare(bench.call,c)
        bench.poison(); checked(first(),'performance correctness'); rt.sync()
        _, error = bench.output_check()
        calls = [provider.prepare(call,c) for call in bench.calls]*traversals
        return Graph(rt,calls), error
    try:
        replay_proof = bench.replay_and_negative(lib.candidates(a.qtype)[0])
        if a.phase=='profile':
            hits=[(arm,c,key) for arm,c,key in pool if key==a.config]
            if len(hits)!=1: raise ValueError('profile requires one exact declared config key')
            arm,c,key=hits[0]
            graph,error=graph_for(arm,c)
            try:
                for _ in range(5): graph.sample()
            finally:
                graph.close()
            prefix='hggc' if lib.manifest['platform']=='ppu' else 'cuda'
            start,stop=(getattr(rt.lib,prefix+name) for name in ('ProfilerStart','ProfilerStop'))
            start.argtypes=stop.argtypes=[];start.restype=stop.restype=C.c_int
            checked(start(),'profiler start')
            checked(providers[arm].prepare(bench.calls[0],c)(),'profile exact complete call')
            rt.sync();checked(stop(),'profiler stop')
            bench.output_check()
            return dict(status='PASS',qtype=a.qtype,arm=arm,key=key,n=a.n,k=a.k,tokens=a.tokens,
                error=error,cache=a.cache,l2_bytes=l2,ring_copies=copies,ring_bytes=copies*distinct,
                scope='PROFILER_RANGE_NOT_EVENT_TIMING',production_admitted=False)
        for index,(arm,c,key) in enumerate(pool):
            graph,error = graph_for(arm,c)
            try:
                samples = [graph.sample() for _ in range(3)]
            finally:
                graph.close()
            screened.append(dict(arm=arm,key=key,error=error,samples_us=samples,median_us=statistics.median(samples)))
            a.journal.write(json.dumps(dict(phase='screen',**screened[-1]))+'\n')
            a.journal.flush()
            if (index+1)%16==0 or index+1==len(pool):
                print(f'SIMT_SCREEN q={a.qtype} completed={index+1}/{len(pool)}',flush=True)
        selected = {r['key'] for arm in providers
                    for r in sorted((r for r in screened if r['arm']==arm),key=lambda r:r['median_us'])[:2]}
        finalists = [(arm,c,key) for arm,c,key in pool if key in selected]
        graphs = [(arm,key,*graph_for(arm,c)) for arm,c,key in finalists]
        try:
            for repeat in range(a.rounds):
                order = graphs if repeat%2==0 else list(reversed(graphs))
                for arm,key,graph,error in order:
                    samples = [graph.sample() for _ in range(a.samples)]
                    confirmed.append(dict(arm=arm,key=key,round=repeat,error=error,
                        samples_us=samples,median_us=statistics.median(samples)))
                    a.journal.write(json.dumps(dict(phase='confirm',**confirmed[-1]))+'\n')
                    a.journal.flush()
        finally:
            for _,_,graph,_ in graphs: graph.close()
        best = {}
        for arm in providers:
            choices = []
            for key in selected:
                samples = [t for r in confirmed if r['arm']==arm and r['key']==key for t in r['samples_us']]
                if samples: choices.append((statistics.median(samples),key))
            value,key = min(choices)
            best[arm] = dict(key=key,median_us=value)
        config = next(c for c in lib.candidates(a.qtype) if c.key==best['new']['key'])
        addresses = pattern(a.qtype,config,a.n,a.k,bases=dict(A=bench.a%128,
            low=bench.call.low%128,high=(bench.call.high or 0)%128,units=bench.call.units%128))
        return dict(status='PASS',qtype=a.qtype,n=a.n,k=a.k,tokens=a.tokens,mode=a.mode,
            channels=a.channels,cache=a.cache,l2_bytes=l2,ring_copies=copies,ring_bytes=copies*distinct,
            experts=w.experts,active_experts=active,resident_weight_bytes=resident,
            allocated_ring_bytes=copies*resident,
            calls_per_graph=copies*traversals,weight_bytes=distinct,best=best,
            delta_pct=(best['new']['median_us']/best['old']['median_us']-1)*100,
            screen=screened,confirmation=confirmed,access=addresses,production_admitted=False,
            replay_proof=replay_proof,
            scope='SIMT_FULL_F32_ENDPOINT_NOT_TC_OR_MODEL',
            q4_optimized_control=control.record if control else None,
            arithmetic=dict(new='F32_GROUP_AFFINE',old='PER_WEIGHT_F16_RECONSTRUCTION',accumulator='F32'))
    finally:
        bench.close()


def main(a):
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): raise ValueError('output exists; use an explicit new attempt path')
    lib = Library(a.bundle)
    rt = Runtime(a.sdk,lib.manifest['platform'])
    a.journal = a.output.with_suffix('.cells.jsonl').open('x')
    result = dict(status='INCOMPLETE',manifest_sha256=sha(a.bundle/'manifest.json'),qtype=a.qtype,phase=a.phase)
    result['harness_hashes'] = {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    try:
        identity = lib.probe();result['device']=identity
        print('SIMT_DEVICE '+json.dumps(identity),flush=True)
        result['result'] = numeric(a,rt,lib) if a.phase=='numeric' else performance(a,rt,lib,identity)
        result['status']='PASS'
        concise = {k:v for k,v in result['result'].items() if k not in ('records','replays','screen','confirmation','access')}
        print('SIMT_RESULT '+json.dumps(concise),flush=True)
    except Exception as e:
        result.update(status='FAIL',error=str(e));traceback.print_exc()
    finally:
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        a.journal.close()
        rt.close()
    return int(result['status']!='PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--qtype',type=int,choices=spec.QTYPES,required=True)
    p.add_argument('--phase',choices=('numeric','perf','profile'),required=True)
    p.add_argument('--config')
    p.add_argument('--n',type=int,default=512);p.add_argument('--k',type=int,default=2048)
    p.add_argument('--tokens',type=int,choices=range(1,9),default=1)
    p.add_argument('--mode',choices=('dense','indexed'),default='dense')
    p.add_argument('--channels',type=int,choices=(1,8),default=1)
    p.add_argument('--experts',type=int,default=8,help='resident expert count for indexed cases')
    p.add_argument('--cache',choices=('rotating','warm'),default='rotating')
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--q4-controls',type=Path)
    p.add_argument('--rounds',type=int,default=4);p.add_argument('--samples',type=int,default=11)
    a=p.parse_args()
    if a.rounds<2 or a.samples<3:p.error('at least two alternating rounds and three samples')
    if a.mode=='dense' and a.channels!=1:p.error('dense channels must be 1')
    if a.experts<8 or a.experts>256:p.error('indexed experts must be in [8,256]')
    raise SystemExit(main(a))

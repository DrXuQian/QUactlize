"""Bounded real-producer binding gate; explicit recipes, never a new sweep."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json

from dev.moe_scaffold.build import build as build_helpers
from dev.strategy.journal import atomic_json
from quactlize.decode.grouped_compiler import GroupedComputeCompiler
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[2]


def parent(q):
    route='sf-grouped' if q==8 else 'fq-grouped'
    return dict(symbol=f'grouped_dynamic_gate_q{q}',qtype=q,route=route,
                tm=16,tn=64,tk=256,wm=16,wn=16,stages=2,ap=0,dn=16,
                persistent=-1 if q==8 else 0)


class Compiler(GroupedComputeCompiler):
    def __init__(self,sdk,cache):
        super().__init__(sdk,cache,compute_type='bf16')
        self.identity['grouped_dynamic_gate']=sha(__file__)
        path=Path(__file__);self.input_stats[path]=(path.stat().st_mtime_ns,path.stat().st_size)
    def source(self,parent,key):
        return '#define QK_GROUPED_DYNAMIC_TESTING 1\n'+super().source(parent,key)


def build(sdk,output,jobs):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    helper=build_helpers(sdk,output/'helpers')
    compiler=Compiler(sdk,output/'cache')
    records={}
    with ThreadPoolExecutor(max_workers=min(jobs,6)) as pool:
        futures={pool.submit(compiler.build,parent(q)):q for q in (8,10,11,12,13,14)}
        for future in as_completed(futures):
            q=futures[future];record=future.result()
            records[str(q)]=[dict(parent=parent(q),compute='bf16',endpoint='grouped'),record]
            print('MOE_GROUPED_MODULE '+json.dumps(dict(q=q,key=record['key'],status='READY')),flush=True)
    sources={str(p.relative_to(ROOT)):sha(p) for d in ('dev/moe_grouped','quactlize/moe','quactlize/routing')
             for p in (ROOT/d).glob('*') if p.is_file()}
    for f in ('tools/run_moe_grouped.py','dev/moe_scaffold/bench.py','dev/moe_scaffold/protocol.py',
              'dev/bf16_compute/fixture.py'):
        sources[f]=sha(ROOT/f)
    receipt=dict(helper=helper,modules=records,sources=sources,compiler=compiler.identity,
        source_contract=digest(compiler.identity),scope='GROUPED_DYNAMIC_BINDING_GATE_EXPLICIT_CONFIGS')
    atomic_json(output/'receipt.json',receipt)
    return receipt


def verify(receipt):
    from dev.moe_scaffold.build import verify as helper_verify
    helper_verify(receipt['helper'])
    for p,h in receipt['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('chain source changed: '+p)
    for spec,record in receipt['modules'].values():
        if sha(record['path'])!=record['sha256']:raise ValueError('chain module changed: '+record['path'])

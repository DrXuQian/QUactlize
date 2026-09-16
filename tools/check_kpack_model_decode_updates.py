#!/usr/bin/env python3
"""Independent-oracle/replay gate for the three bounded production updates."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.native import Runtime
from dev.gemv_simt.production import Library
from dev.gemv_simt.q8_vector_run import Bench
from dev.gemv_simt.fixture import weights
from quactlize.execution.simt_codegen import Config
from quactlize.runtime.compiler import sha
from tools.kpack_execution_fixture import IndexedWeights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','library','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--platform',choices=('ppu','cuda'),default='ppu')
    a=p.parse_args();rt=Runtime(a.sdk,a.platform);records=[]
    points=[(8,2048,4096,0,1,0,Config(5,8,4,4,8)),
            (12,1024,2048,2,1,1,Config(3,4,4,4)),
            (13,2048,512,2,8,1,Config(3,4,2,8))]
    try:
        for q,n,k,mode,ch,compute,cfg in points:
            print(f'MODEL_DECODE_GATE START q={q} shape={n}x{k} compute={compute}',flush=True)
            w=weights(q,n,k,1) if q==8 else IndexedWeights(q,n,k,256,
                progress=lambda i,total:print(f'MODEL_DECODE_GATE fixture={i}/{total}',flush=True))
            lib=Library(a.library,compute)
            # M2 is an unchanged-route control; M1 exercises the new entry.
            for tokens in (1,2):
                b=Bench(rt,lib,w,tokens,mode,ch,compute)
                try:
                    for repeat in (0,1,2,4):
                        b.update(repeat);_,error=b.correctness(cfg)
                        records.append(dict(q=q,tokens=tokens,repeat=repeat,error=error,config=cfg.key))
                    proof=b.replay_and_negative(cfg)
                    b.invalid_id_negative(cfg)
                    print(f'MODEL_DECODE_GATE PASS q={q} tokens={tokens} proof={proof["negative"]}',flush=True)
                finally:b.close()
        result=dict(status='PASS',records=records,contexts=6,replays_per_context=3,
            execution_sha256=sha(a.library),platform=a.platform,
            timing='NOT_MEASURED',scope='PRODUCTION_SIMT_V2_INDEPENDENT_GGUF_CHANGED_REPLAYS_GUARDS')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('MODEL_DECODE_GATE COMPLETE contexts=6 numeric=24 replays=18',flush=True)
    finally:rt.close()


if __name__=='__main__':main()

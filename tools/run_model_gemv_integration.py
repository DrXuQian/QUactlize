#!/usr/bin/env python3
"""Compare selected production readers with the immutable admitted experiment."""
import argparse
import ctypes as C
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_model.plan import POINTS, candidates, inventory
from dev.gemv_model.fixture import Bench
from dev.gemv_model.engine import Provider, correctness, timing_graph
from dev.gemv_model.run import summarize, logged
from dev.gemv_simt.native import Runtime, checked
from dev.gemv_simt.production import Library as SimtLibrary
from dev.gemv_simt.q8_vector_run import l2_identity
from quactlize.dispatch.native import Dispatch
from quactlize.execution.native import arrangement
from quactlize.execution.simt_codegen import Config as SimtConfig
from quactlize.fusion.native import Library as FusionLibrary, Config, integration_entries
from quactlize.runtime.compiler import sha
from tools.run_kpack_pack_gate import device_identity
from tools.verify_kpack_dispatch import verify

EVIDENCE=ROOT/'docs/measurements/model_gemv_20260917.json'
SIMT_FIELDS=('variant','columns','warps','values','split')


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def reference_manifest(bundle):
    pin=json.loads((ROOT/'tools/kpack_model_gemv_artifact.json').read_text())
    if sha(bundle/'manifest.json')!=pin['manifest_sha256']:
        raise ValueError('admitted experiment manifest differs')
    data=json.loads((bundle/'manifest.json').read_text())
    if data['inventory']!=json.loads(json.dumps(inventory())):
        raise ValueError('admitted experiment inventory differs')
    for name,h in data['payloads'].items():
        p=bundle/name
        if p.parent!=bundle or p.is_symlink() or sha(p)!=h:
            raise ValueError('admitted experiment payload differs: '+name)
    # Its source is deliberately frozen at the published commit, not relabelled
    # as the current production checkout. The pinned manifest binds all bytes.
    return data


def selected_simt(selected,config):
    if selected is None or selected.base.kind!=1 or any(
            getattr(selected.base.simt,k)!=getattr(config,k) for k in SIMT_FIELDS):
        raise ValueError('production matched selection differs')
    return dict(policy=selected.base.policy,
                config={k:getattr(selected.base.simt,k) for k in SIMT_FIELDS})


class Integrated(Provider):
    def __init__(self,bench,bundle,config):
        self.b,self.arm,self.handles,self.config=bench,'incumbent',[],config
        p=bench.point
        if p.paired:
            self.fusion=FusionLibrary(bundle/'libquactlize_ppu_gate_up.so')
            self.layout=self.fusion.arrangement(p.q)
            entries=integration_entries(self.fusion);self.fn=entries['run']
            cfg=Config()
            checked(entries['select'](p.q,p.n,p.k,p.experts,1,p.compute,C.byref(cfg)), 'production paired selection')
            if (cfg.backend,cfg.split,cfg.tile_m,cfg.warps)!=(0,1,0,8):
                raise ValueError('paired production recipe differs')
            self.selection=dict(backend='simt',split=1,warps=8)
        else:
            self.shipping=SimtLibrary(bundle/'libquactlize_ppu_execution.so',p.compute)
            dispatch=Dispatch(bundle)
            try:
                selected=dispatch.query_smallm_matched(bench.call(),arrangement(p.q),p.compute)
                self.selection=selected_simt(selected,config)
            finally:
                dispatch.close()

    def prepare(self,copy=0):
        if self.b.point.paired:
            return super().prepare(copy)
        config=SimtConfig(**{k:getattr(self.config,k) for k in SIMT_FIELDS})
        return self.shipping.prepare(self.b.call(copy),config)


def child(args):
    manifest=verify(args.bundle,sdk=args.sdk)
    frozen=reference_manifest(args.reference)
    p=next(x for x in POINTS if x.name==args.point)
    evidence=next(r for r in json.loads(EVIDENCE.read_text())['records'] if r['point']==p.name)
    if evidence['decision']!='EXACT_M1_INTEGRATION_CANDIDATE':
        raise ValueError('point has no admitted integration candidate')
    arm=evidence['arm'];config=candidates(p)[int(arm)]
    if asdict(config)!=evidence['config']:
        raise ValueError('admitted candidate config changed')
    record=next(r for r in frozen['records'] if r['point']['name']==p.name)
    rt=Runtime(args.sdk,'ppu');providers={};graphs={}
    result=dict(status='FAIL',point=p.name,manifest_sha256=sha(args.bundle/'manifest.json'),
                reference_manifest_sha256=sha(args.reference/'manifest.json'),evidence_sha256=sha(EVIDENCE),
                execution_sha256=manifest['execution_sha256'],numerics={},scope='ISOLATED_COMPLETE_CALL_NOT_MODEL_TPOT')
    try:
        result['device']=device_identity(rt)
        l2=l2_identity(dict(l2_bytes=rt.attribute(38)),args.l2_bytes)
        if l2['l2_bytes']<=0:raise ValueError('verified positive L2 capacity required')
        b=Bench(rt,p,l2['l2_bytes']);result['fixture']=b.record
        providers['original']=Provider(b,args.reference,record,'incumbent')
        providers['admitted']=Provider(b,args.reference,record,arm)
        providers['integrated']=Integrated(b,args.bundle,config)
        result['selection']=providers['integrated'].selection
        snapshots={}
        for name,provider in providers.items():
            proof,bits=correctness(provider)
            result['numerics'][name]=proof;snapshots[name]=bits['m1']
            print(f'MODEL_GEMV_INTEGRATION_GATE point={p.name} arm={name} PASS',flush=True)
        bitdiff=int(np.count_nonzero(snapshots['admitted']!=snapshots['integrated']))
        result['admitted_m1_bitdiff']=bitdiff
        if bitdiff:raise ValueError('integrated M1 differs from admitted candidate bits')
        for name,provider in providers.items():
            b.poison();graphs[name]=timing_graph(provider);b.check()
        samples={key:[] for key in graphs}
        for round in range(6):
            order=list(graphs)
            if round%2:order.reverse()
            for name in order:samples[name].append([graphs[name].sample() for _ in range(15)])
            print(f'MODEL_GEMV_INTEGRATION_CONFIRM point={p.name} round={round+1}/6',flush=True)
        target=60 if b.weight_bytes>=16*1024*1024 else 40
        result['timing']={key:summarize(value,b.weight_bytes,target) for key,value in samples.items()}
        times={key:r['median_us'] for key,r in result['timing'].items()}
        result['versus_admitted_pct']=100*(times['integrated']/times['admitted']-1)
        result['versus_original_pct']=100*(times['integrated']/times['original']-1)
        result['status']='PASS' if result['versus_admitted_pct']<=5 and result['versus_original_pct']<=0 else 'PERFORMANCE_REVIEW'
        print('MODEL_GEMV_INTEGRATION_RESULT '+json.dumps({k:result[k] for k in ('point','status','versus_admitted_pct','versus_original_pct')}),flush=True)
        return int(result['status']!='PASS')
    except Exception as error:
        result['error']=str(error);traceback.print_exc();return 1
    finally:
        for graph in graphs.values():graph.close()
        for provider in providers.values():provider.close()
        rt.close();save(args.output,result)


def main(args):
    verify(args.bundle,sdk=args.sdk);reference_manifest(args.reference)
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for p in POINTS[:-1]:
        path=args.output/(p.name+'.json');log=path.with_suffix('.log')
        command=[sys.executable,'-u',str(Path(__file__)), '--sdk',str(args.sdk),'--bundle',str(args.bundle),
                 '--reference',str(args.reference),'--l2-bytes',str(args.l2_bytes),'--point',p.name,'--output',str(path)]
        save(path.with_suffix('.command.json'),command)
        rc=logged(command,log,p.name,echo=True)
        row=dict(point=p.name,rc=rc,status='FAIL',log=log.name)
        if path.is_file():row.update(json.loads(path.read_text()))
        records.append(row)
        print(f'MODEL_GEMV_INTEGRATION_PROGRESS completed={len(records)}/7 status={row["status"]} remaining_continue=1',flush=True)
    passed=all(r['rc']==0 and r['status']=='PASS' for r in records)
    save(args.output/'summary.json',dict(status='PASS' if passed else 'FAIL',records=records,
        scope='SEVEN_EXACT_M1_REPLACEMENTS_M2_M8_NUMERICAL_CONTROLS',q6='RETAIN_TC'))
    return int(not passed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','bundle','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--l2-bytes',type=int,default=0)
    parser.add_argument('--point',choices=[p.name for p in POINTS[:-1]])
    a=parser.parse_args()
    raise SystemExit(child(a) if a.point else main(a))

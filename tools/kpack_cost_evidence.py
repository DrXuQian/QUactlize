"""Reuse audited all-expert dequant costs only for identical resident bytes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compose_kpack_prefill_costs import Results, best_dequant
from tools.kpack_cost_supplement import weight_id, configs
from tools.run_kpack_dequant_gate import save
from quactlize.runtime.compiler import sha

EVIDENCE = ROOT/'docs/measurements/kpack_cost_reuse_20260914.json'


def export(sf_path, full_path, output):
    entries=[]
    for operation, path in ((0,sf_path),(1,full_path)):
        study=Results(path)
        for w in study.authority['workloads']:
            if w['smoke'] or w['operation']!=operation or w['experts'] not in (1,256):
                continue
            r,best=best_dequant(study,w)
            entries.append(dict(weight_id=weight_id(w),operation=operation,
                workload=w,fixture_hashes=r['fixture_hashes'],golden_sha256=r['golden_sha256'],
                best=best,device=study.authority['device'],runtime=study.authority['runtime'],
                python_packages=study.authority['python_packages'],
                source=dict(result=study.reference(w['id']+'.json'),
                    authority_sha256=study.files['authority.json'],
                    summary_sha256=sha(study.folder/'result.json'),
                    manifest_sha256=study.authority['manifest_sha256'])))
    save(output,dict(schema='quactlize.cost-reuse.v1',entries=entries,
        scope='AUDITED_ALL_EXPERT_DEQUANT_NO_PROPORTIONAL_SPARSE_SCALING'))


def reuse(evidence,p,operation,fixture,device,runtime,packages):
    if operation and p['full_indexed']:
        return None
    matches=[r for r in evidence['entries'] if r['weight_id']==p['weight_id'] and r['operation']==operation]
    if not matches:return None
    if len(matches)!=1:raise ValueError('duplicate prior dequant measurement')
    r=matches[0]
    gold=fixture['bf16_golden_sha256' if operation else 'sf_golden_sha256']
    # A mismatch is evidence rejection, not permission to silently mix campaigns.
    if (r['fixture_hashes']!=fixture['fixture_hashes'] or r['golden_sha256']!=gold or
            r['device']!=device or r['runtime']!=runtime or r['python_packages']!=packages):
        raise ValueError('prior dequant fixture/device/runtime/packages differ')
    best=r['best']
    if best['config'] not in configs(p['q'],operation):
        raise ValueError('prior dequant config not in the retained implementation')
    return dict(status='PASS',kind='dequant',scope='ISOLATED_DEQUANT_ACTUAL_EXPERT_DOMAIN',
        weight_id=p['weight_id'],operation=operation,indexed=False,
        active_ids=list(range(p['experts'])),expanded_experts=p['experts'],
        device=device,fixture=fixture,rows=[best],measurement='REUSED_AUDITED_MEASUREMENT',
        source=r['source'],gemm_timed=False,production_changed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sf',type=Path,required=True)
    parser.add_argument('--full',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.sf,args.full,args.output)

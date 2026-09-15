#!/usr/bin/env python3
"""Compact receipts from real local NVIDIA results; never PPU admission."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def metrics(ncu,report):
    raw=subprocess.check_output([str(ncu),'--import',str(report),'--page','raw','--csv'],text=True)
    data=list(csv.DictReader(io.StringIO(raw[raw.index('"ID"'):])) )
    if len(data)!=2:raise ValueError('profile must have one exact kernel')
    units,row=data
    exact={'gpu__time_duration.sum','launch__registers_per_thread','launch__shared_mem_per_block',
        'dram__bytes_read.sum','dram__bytes_write.sum',
        'sm__warps_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
        'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
        'smsp__sass_inst_executed_op_global_ld.sum','smsp__sass_inst_executed_op_local_ld.sum',
        'smsp__sass_inst_executed_op_local_st.sum','smsp__inst_executed.sum'}
    selected={k:dict(value=v,unit=units[k]) for k,v in row.items() if k in exact or
        ('warp_issue_stalled_' in k and k.endswith('_per_warp_active.pct')) or
        (k.startswith('dram__') and 'pct_of_peak' in k) or
        (k.startswith('lts__t_sectors') and k.endswith('.sum'))}
    return dict(sha256=sha(report),kernel=row['Kernel Name'],block=row['Block Size'],grid=row['Grid Size'],
        metrics=selected,scope='NCU_FORCED_COLD_REPLAY_NOT_ROTATING_EVENT_TIME')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--ncu',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root
    q8=json.loads((root/'q8-perf-v1/summary.json').read_text())
    # Read individual result files: summaries vary in whether they embed rows.
    results=[]
    for path in sorted((root/'q8-perf-v1').glob('*.json')):
        r=json.loads(path.read_text())
        if r.get('status')!='PASS' or 'best' not in r:continue
        for arm,record in r['best'].items():
            values=[x for samples in record['rounds'] for x in samples]
            if len(values)!=90 or statistics.median(values)!=record['median_us']:raise ValueError('Q8 median/denominator differs')
        results.append({k:r[k] for k in ('shape','channels','compute','delta_pct','ring_copies','l2_bytes','active_experts','useful_bytes','scope')}|
            dict(result_sha256=sha(path),best={arm:{k:r[k] for k in ('config','median_us','rounds')} for arm,r in r['best'].items()}))
    if len(results)!=50:raise ValueError('Q8 performance context denominator differs')
    numeric=json.loads((root/'q8-numeric-v2.json').read_text())
    if numeric['status']!='PASS' or len(numeric['records'])!=numeric['expected'] or numeric['expected']!=12480:raise ValueError('Q8 numerical denominator differs')
    moe=json.loads((root/'perf-v11/summary.json').read_text())
    gate=json.loads((root/'perf-v11/candidate-correctness.json').read_text())
    if gate['status']!='PASS' or gate['contexts']!=3840:raise ValueError('MoE numerical gate differs')
    result=dict(device='RTX5070',admission='NVIDIA_GUIDANCE_ONLY_PPU_AND_MODEL_PENDING',
        q8_numeric={k:v for k,v in numeric.items() if k!='records'},q8=results,
        moe=moe,moe_numeric=gate,source_hashes={str(Path(__file__).name):sha(Path(__file__))},
        reports={name:metrics(a.ncu,root/(name+'.ncu-rep')) for name in (
            'q8-n2048-k512-baseline','q8-n2048-k512-vector','moe-m8-all-simt-baseline','moe-m8-all-simt-direct')})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('LOCAL_OPTIMIZATIONS_RECEIPT q8_numeric=12480 q8_perf=50 moe_numeric=3840 moe_comparison=46/48 PPU=PENDING',flush=True)


if __name__=='__main__':main()

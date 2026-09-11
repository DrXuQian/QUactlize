#!/usr/bin/env python3
"""Replay one PPU-selected Q4 recipe pair on CUDA; no retuning or policy change."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

if __package__:
    from .compare_q4_native import parse_timing
    from .build import digest
else:
    from compare_q4_native import parse_timing
    from build import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runner','fixture','xplane','candidate','ppu-summary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--n',type=int,default=5120)
    parser.add_argument('--k',type=int,default=8192)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    ppu=json.loads(args.ppu_summary.read_text())
    paths={name:getattr(args,name) for name in ('runner','fixture','xplane','candidate','ppu_summary')}
    paths['candidate_manifest']=args.candidate.parent/'manifest.json'
    paths['xplane_manifest']=args.xplane.parent.parent/'manifest.json'
    candidate=json.loads(paths['candidate_manifest'].read_text())
    xplane=json.loads(paths['xplane_manifest'].read_text())
    controls=[r for r in xplane['arms'] if r['arm']=='fp32-control']
    if (candidate['reader']!='cuda-q4-n4-static' or candidate['library_sha256']!=digest(args.candidate)
            or len(controls)!=1 or controls[0]['library_sha256']!=digest(args.xplane)):
        raise ValueError('expected the static Q4 K-pack and FP32 Xplane comparison builds')
    report=dict(status='RUNNING',shape=[1,args.n,args.k],
        scope='PPU_FIXED_RECIPES_ON_CUDA_NOT_CUDA_OPTIMAL_NOT_PPU_ADMISSION',
        arithmetic='FP16_A_FP32_DOT_AND_OUTPUT',rounds=6,samples_per_round=15,
        authority={name:dict(path=str(path.resolve()),sha256=digest(path)) for name,path in paths.items()},
        cases=[])
    geometry=None
    for mode in ('warm','rotating'):
        selected=[r for r in ppu['cases'] if r['shape']==report['shape'] and r['mode']==mode]
        if len(selected)!=1:raise ValueError('missing/duplicate PPU comparison row')
        source=selected[0]
        recipes={arm:source['winners'][ppu_arm]['recipe'] for arm,ppu_arm in (('xplane','xplane'),('kpack','new'))}
        records={arm:[] for arm in recipes}
        for turn in range(6):
            for arm in (('xplane','kpack') if turn%2==0 else ('kpack','xplane')):
                command=[str(args.runner.resolve()),str(args.fixture.resolve()),str(args.xplane.resolve()),
                         str(args.candidate.resolve()),arm,*map(str,recipes[arm]),mode,'timing']
                log=args.output/f'{mode}-{turn}-{arm}.log'
                proc=subprocess.run(command,text=True,capture_output=True,timeout=180)
                log.write_text(proc.stdout+proc.stderr)
                if proc.returncode:raise ValueError(f'timing rc={proc.returncode}: {log}')
                row=parse_timing(proc.stdout,arm,recipes[arm],mode,report['shape'])
                device=(int(row['sm']),int(row['L2_bytes']))
                if geometry is not None and device!=geometry:raise ValueError('CUDA device geometry changed')
                geometry=device
                copies=int(row['copies']);calls=int(row['calls_per_graph'])
                expected=1 if mode=='warm' else max(2,(9*device[1]+4*(args.n*args.k*9//16)-1)//(4*(args.n*args.k*9//16)))
                if copies!=expected or calls!=max(2,(32+copies-1)//copies)*copies:
                    raise ValueError('cache ring/graph denominator differs from the PPU protocol')
                records[arm].append(dict(**row,command=command,log_sha256=digest(log)))
            print(f'Q4_PPU_REPRO_PROGRESS mode={mode} round={turn+1}/6',flush=True)
        medians={arm:statistics.median(float(r['median_us']) for r in rows) for arm,rows in records.items()}
        row=dict(mode=mode,recipes=recipes,records=records,median_us=medians,
            delta_pct=100*(medians['kpack']/medians['xplane']-1),ppu_delta_pct=source['new_vs_xplane_pct'])
        report['cases'].append(row)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Q4_PPU_REPRO_RESULT '+json.dumps({key:value for key,value in row.items() if key!='records'}),flush=True)
    if any(digest(path)!=report['authority'][name]['sha256'] for name,path in paths.items()):
        raise ValueError('input changed during comparison')
    report.update(status='PASS',device_geometry=dict(sm=geometry[0],l2_bytes=geometry[1]))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Q4_PPU_REPRO_COMPLETE results='+str(args.output),flush=True)


if __name__=='__main__':main()

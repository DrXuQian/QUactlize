"""Reconfirm the two measured GPTQ leaders per arm, without changing semantics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,choices=(1,3),default=3)
    a,common=p.parse_known_args()
    prior=json.loads((a.previous/'summary.json').read_text())
    if not prior['complete'] or prior.get('contract')!='GPTQ_SYM_INT4_G128_F16_V1':
        raise ValueError('complete GPTQ cohort required; rejected GGUF-subset results cannot seed this run')
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('reconfirmation output must be fresh')
    a.output.mkdir(parents=True,exist_ok=True)
    points=[]
    for case in prior['results']:
        dtype,m,n,k=case['point']
        summary=json.loads((a.previous/f'd{dtype}-m{m}-n{n}-k{k}'/'summary.json').read_text())
        if summary['status']!='PASS': raise ValueError('previous point not admitted')
        rows=summary['result'][0]['rows']
        configs=[]
        for arm in ('machete','marlin','quactlize'):
            leaders=sorted([x for x in rows if x['arm']==arm],key=lambda x:x['median_us'])[:2]
            configs.extend(f"{arm}:{x['config']}" for x in leaders)
        points.append((case['point'],configs))
    receipt={'scope':'PAIRED_HOPPER_GPTQ_RECONFIRMATION','contract':prior['contract'],
             'rounds':a.rounds,'points':points,'results':[]}
    for r in range(a.rounds):
        for point,configs in (points if r%2==0 else reversed(points)):
            dtype,m,n,k=point
            out=a.output/f'r{r}-d{dtype}-m{m}-n{n}-k{k}'
            command=[sys.executable,str(Path(__file__).with_name('compare.py')),*common,
                '--output',str(out),'--dtype',str(dtype),'--m',str(m),'--n',str(n),'--k',str(k),
                '--configs',*configs]
            if r%2: command+=['--reverse']
            print(f'PPU17_REFERENCE_CONFIRM round={r+1}/{a.rounds} point={point}',flush=True)
            rc=subprocess.run(command).returncode
            row={'round':r,'point':point,'command':command,'returncode':rc,'summary':str(out/'summary.json')}
            if not rc:
                measured=json.loads((out/'summary.json').read_text())['result'][0]['rows']
                row['best']={arm:min([x for x in measured if x['arm']==arm],key=lambda x:x['median_us'])
                             for arm in ('machete','marlin','quactlize')}
                print('PPU17_REFERENCE_CONFIRM_POINT '+json.dumps({arm:{'config':v['config'],'us':v['median_us']}
                    for arm,v in row['best'].items()}),flush=True)
            receipt['results'].append(row)
            receipt['complete']=not rc and len(receipt['results'])==a.rounds*len(points)
            (a.output/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
            if rc: raise SystemExit(rc)


if __name__=='__main__':
    main()

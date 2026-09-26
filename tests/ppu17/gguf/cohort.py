"""Bounded same-GGUF screen; original bring-up control is not a library claim."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dtypes',type=int,nargs='+',choices=(0,1),default=[0,1])
    args,common=p.parse_known_args()
    if args.output.exists():raise ValueError('fresh cohort output required')
    args.output.mkdir(parents=True)
    rows=[]
    points=[(d,m,n,k) for d in args.dtypes for n,k in ((512,2048),(4096,4096)) for m in (1,8,128)]
    for index,(dtype,m,n,k) in enumerate(points):
        arms=['bringup-baseline','candidate']
        if index%2:arms.reverse()
        point={'dtype':dtype,'shape':[m,n,k],'arms':{}}
        for arm in arms:
            dest=args.output/f'd{dtype}-m{m}-n{n}-k{k}-{arm}'
            configs=['0','1','2'] if arm=='candidate' else ['0','1','2','3','4','5']
            command=[sys.executable,str(Path(__file__).with_name('run.py')),*common,
                '--output',str(dest),'--dtype',str(dtype),'--m',str(m),'--n',str(n),
                '--k',str(k),'--arm',arm,'--configs',*configs,'--timing']
            print(f'GGUF_Q4_PROGRESS point={index+1}/{len(points)} arm={arm} shape={m}x{n}x{k}',flush=True)
            rc=subprocess.run(command).returncode
            point['arms'][arm]={'returncode':rc,'command':command,'summary':str(dest/'summary.json')}
            if rc:
                rows.append(point)
                (args.output/'summary.json').write_text(json.dumps({'status':'INCOMPLETE','points':rows},indent=2)+'\n')
                return rc
            result=json.loads((dest/'summary.json').read_text())['result'][0]
            best=min(result['rows'],key=lambda r:r['median_us'])
            point['arms'][arm].update(config=best['config'],median_us=best['median_us'],
                                     clocks=best['clocks'],raw_gguf_sha256=best['raw_gguf_sha256'])
        base=point['arms']['bringup-baseline'];new=point['arms']['candidate']
        if base['raw_gguf_sha256']!=new['raw_gguf_sha256']:raise ValueError('input differs')
        point['clock_matched']=base['clocks']==new['clocks']
        point['delta_vs_bringup_pct']=100*(new['median_us']/base['median_us']-1)
        point['scope']='SCREEN_NOT_CONFIRMATION_OR_FORMAT_ABLATION_OR_FQ'
        rows.append(point)
        (args.output/'summary.json').write_text(json.dumps({'status':'PASS' if len(rows)==len(points) else 'RUNNING','points':rows},indent=2)+'\n')
        print('GGUF_Q4_POINT '+json.dumps(point),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())

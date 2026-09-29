"""One-axis stage4/stage8 screen, with exact same-math controls retained."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-from',type=Path)
    args,common=p.parse_known_args()
    if args.output.exists():raise ValueError('fresh stage output required')
    args.output.mkdir(parents=True)
    results=[]
    retained={}
    if args.resume_from:
        bundle=Path(common[common.index('--bundle')+1])
        manifest_sha=sha(bundle/'manifest.json')
        runner_sha=sha(Path(__file__).with_name('run.py'))
        for old in json.loads(args.resume_from.read_text())['points']:
            if old['returncode']!=0:continue
            path=Path(old['summary']);receipt=json.loads(path.read_text())
            if receipt['status']!='PASS' or receipt['manifest_sha256']!=manifest_sha or receipt['runner_sha256']!=runner_sha:
                raise ValueError('retained stage authority differs')
            payload=receipt['result'][0]['rows']
            if len(payload)!=6 or any(x['shape']!=old['shape'] or x['dtype']!=old['dtype'] for x in payload):
                raise ValueError('retained stage point differs')
            old.update(retained_from=str(args.resume_from),summary_sha256=sha(path))
            key=(old['dtype'],*old['shape'])
            if key in retained:raise ValueError('duplicate retained stage point')
            retained[key]=old
    points=[(d,m,n,k) for d in (0,1) for n,k in ((512,2048),(4096,4096)) for m in (1,8,128)]
    for index,(d,m,n,k) in enumerate(points):
        if (d,m,n,k) in retained:
            results.append(retained[(d,m,n,k)])
            continue
        out=args.output/f'd{d}-m{m}-n{n}-k{k}'
        configs=['0','3','1','4','2','5']
        if index%2:configs.reverse()
        cmd=[sys.executable,str(Path(__file__).with_name('run.py')),*common,
             '--dtype',str(d),'--m',str(m),'--n',str(n),'--k',str(k),
             '--timing','--configs',*configs,'--output',str(out)]
        print(f'GGUF_Q4_STAGE_PROGRESS point={index+1}/{len(points)} dtype={d} shape={m}x{n}x{k}',flush=True)
        rc=subprocess.run(cmd).returncode
        row={'dtype':d,'shape':[m,n,k],'returncode':rc,'summary':str(out/'summary.json'),'command':cmd,
             'summary_sha256':sha(out/'summary.json')}
        if not rc:
            payload=json.loads((out/'summary.json').read_text())['result'][0]['rows']
            pairs=[]
            for config in range(3):
                base=next(x for x in payload if x['config']==config)
                new=next(x for x in payload if x['config']==config+3)
                # Both replays include +/- input checks in the same order.
                same=[x['sha256'] for x in base['proof']]==[x['sha256'] for x in new['proof']]
                if not same:raise ValueError('stage-only change differs in raw outputs; inspect scheduling/numerics before timing admission')
                pairs.append({'config':config,'s4_us':base['median_us'],'s8_us':new['median_us'],
                              'raw_equal':same,'clock_matched':base['clocks']==new['clocks'],
                              'delta_pct':100*(new['median_us']/base['median_us']-1)})
            row['pairs']=pairs
        results.append(row)
        (args.output/'summary.json').write_text(json.dumps({'status':'PASS' if len(results)==len(points) and not rc else 'INCOMPLETE',
              'scope':'ONE_AXIS_STAGE_SCREEN_NOT_SHIPPING_SELECTION','points':results},indent=2)+'\n')
        if rc:return rc
        print('GGUF_Q4_STAGE_POINT '+json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'status':'PASS',
          'scope':'ONE_AXIS_STAGE_SCREEN_NOT_SHIPPING_SELECTION','points':results},indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())

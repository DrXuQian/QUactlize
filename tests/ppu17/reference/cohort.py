"""Six GPTQ comparison points; stop and preserve output on any failed gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args,common=p.parse_known_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('cohort output must be fresh')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    points=[(0,m,n,k) for n,k in ((512,2048),(4096,4096)) for m in (1,8,128)]
    for index,(dtype,m,n,k) in enumerate(points):
        out=args.output/f'd{dtype}-m{m}-n{n}-k{k}'
        command=[sys.executable,str(Path(__file__).with_name('compare.py')),*common,
                 '--output',str(out),'--dtype',str(dtype),'--m',str(m),'--n',str(n),'--k',str(k)]
        if index%2: command+=['--reverse']
        print(f'PPU17_REFERENCE_COHORT point={index+1}/{len(points)} dtype={dtype} shape={m}x{n}x{k}',flush=True)
        rc=subprocess.run(command).returncode
        row={'point':[dtype,m,n,k],'command':command,'returncode':rc,'summary':str(out/'summary.json')}
        if not rc:
            summary=json.loads((out/'summary.json').read_text())
            result=summary['result'][0]
            best={arm:min((x for x in result['rows'] if x['arm']==arm),key=lambda x:x['median_us'])
                  for arm in result['config_counts']}
            row['best']={arm:{'config':x['config'],'us':x['median_us']} for arm,x in best.items()}
            selected=min((x for name,x in best.items() if name.startswith('quactlize')),key=lambda x:x['median_us'])
            row['candidate_provider']=selected['arm']
            row['delta_vs_faster_reference_pct']=100*(selected['median_us']/min(
                best['machete']['median_us'],best['marlin']['median_us'])-1)
            print('PPU17_REFERENCE_POINT '+json.dumps(row),flush=True)
        rows.append(row)
        (args.output/'summary.json').write_text(json.dumps({'complete':len(rows)==len(points) and rc==0,
            'contract':'GPTQ_SYM_INT4_G128_F16_V1',
            'scope':'SCREEN_ONLY_CONFIRMATION_REQUIRED','results':rows},indent=2)+'\n')
        if rc:
            raise SystemExit(rc)


if __name__=='__main__':
    main()

"""Reconfirm the two measured GPTQ leaders per arm, without changing semantics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def selected_configs(rows):
    """Retain measured leaders and the complete same-math control chain."""
    result = []
    for arm in sorted({row['arm'] for row in rows}):
        candidates = [row for row in rows if row['arm'] == arm]
        by_id = {row['config']: row for row in candidates}
        if len(by_id) != len(candidates):
            raise ValueError('duplicate configuration')
        pending = sorted(candidates, key=lambda row: row['median_us'])[:2]
        seen = set()
        while pending:
            row = pending.pop(0)
            if row['config'] in seen:
                continue
            seen.add(row['config'])
            result.append(f"{arm}:{row['config']}")
            if 'same_math_parent' in row:
                parent = row['same_math_parent']
                if parent not in by_id:
                    raise ValueError('missing same-math control')
                pending.append(by_id[parent])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,choices=(1,3),default=3)
    p.add_argument('--points',type=int,nargs='+',choices=range(6),
                   help='explicit cohort indices; omitted points are not confirmed')
    a,common=p.parse_known_args()
    prior=json.loads((a.previous/'summary.json').read_text())
    if prior.get('contract')!='GPTQ_SYM_INT4_G128_F16_V1' or (not prior['complete'] and a.points is None):
        raise ValueError('complete GPTQ cohort or explicit admitted point indices required')
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('reconfirmation output must be fresh')
    a.output.mkdir(parents=True,exist_ok=True)
    points=[]
    if a.points is not None and len(set(a.points)) != len(a.points):
        raise ValueError('duplicate requested point')
    for index,case in enumerate(prior['results']):
        if a.points is not None and index not in a.points:
            continue
        dtype,m,n,k=case['point']
        summary=json.loads((a.previous/f'd{dtype}-m{m}-n{n}-k{k}'/'summary.json').read_text())
        if summary['status']!='PASS': raise ValueError('previous point not admitted')
        rows=summary['result'][0]['rows']
        points.append((case['point'],selected_configs(rows)))
    if a.points is not None and len(points) != len(a.points):
        raise ValueError('requested point missing from previous cohort')
    receipt={'scope':'PAIRED_HOPPER_GPTQ_RECONFIRMATION','contract':prior['contract'],
             'rounds':a.rounds,'points':points,'results':[],
             'requested_point_indices':a.points or list(range(6))}
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
                             for arm in sorted({x['arm'] for x in measured})}
                print('PPU17_REFERENCE_CONFIRM_POINT '+json.dumps({arm:{'config':v['config'],'us':v['median_us']}
                    for arm,v in row['best'].items()}),flush=True)
            receipt['results'].append(row)
            receipt['complete']=(len(receipt['results'])==a.rounds*len(points)
                                 and all(x['returncode']==0 for x in receipt['results']))
            (a.output/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
            # Each point has a fresh process. Preserve failures and continue
            # independent points; a clock rejection must not erase coverage.
    if not receipt['complete']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
